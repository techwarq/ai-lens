import { Trace, TraceStep, TraceWhyResult, AILensConfig } from '../types'
import { callAnalysisModel } from './why'

/**
 * Analyze why a full pipeline trace failed.
 *
 * Instead of diagnosing individual calls, this looks at the whole
 * step chain and identifies which step caused the cascade.
 *
 * Key insight: a bad final output doesn't mean the last step failed.
 * It usually means an intermediate step produced bad output that
 * propagated forward. This analyzer scores each step for suspicion
 * and identifies the true root cause.
 */
export async function analyzeTraceWhy(
  trace: Trace,
  config: AILensConfig
): Promise<TraceWhyResult> {
  if (trace.steps.length === 0) {
    return {
      trace,
      rootStep: null,
      stepDiagnoses: [],
      diagnosis: 'No steps recorded in this trace.',
      suggestedFix: '',
      severity: 'low',
    }
  }

  // Step 1: Score each step for suspicion
  const stepDiagnoses = await scoreSteps(trace, config)

  // Step 2: Find the root cause step
  const rootStep = findRootCause(trace.steps, stepDiagnoses)

  // Step 3: Generate overall diagnosis
  const { diagnosis, suggestedFix, severity } = await synthesizeTraceDiagnosis(
    trace,
    rootStep,
    stepDiagnoses,
    config
  )

  return {
    trace,
    rootStep,
    stepDiagnoses,
    diagnosis,
    suggestedFix,
    severity,
  }
}

async function scoreSteps(
  trace: Trace,
  config: AILensConfig
): Promise<TraceWhyResult['stepDiagnoses']> {
  const stepsDesc = trace.steps.map((s, i) => {
    const checksStr = s.checks
      ? `checks: ${s.checks.map(c => `${c.passed ? '✓' : '✗'} ${c.rule}`).join(', ')}`
      : 'no checks'
    const errorStr = s.error ? `ERROR: ${s.error}` : ''
    return [
      `Step ${i + 1}: "${s.name}" [${s.type}]`,
      `  Input:  ${s.input.slice(0, 150)}`,
      `  Output: ${s.output.slice(0, 150)}`,
      errorStr ? `  ${errorStr}` : '',
      `  ${checksStr}`,
      `  Latency: ${s.latencyMs}ms`,
    ].filter(Boolean).join('\n')
  }).join('\n\n')

  const finalOutput = trace.steps[trace.steps.length - 1]?.output ?? ''
  const feedback = trace.feedback === 'bad'
    ? 'The whole pipeline was marked as bad by the user.'
    : trace.error
      ? `The pipeline errored: ${trace.error}`
      : 'The pipeline output appears to be low quality.'

  const prompt = [
    `You are diagnosing a multi-step AI pipeline to find which step caused the bad output.`,
    ``,
    `## Pipeline: "${trace.name}"`,
    `## Failure context: ${feedback}`,
    `## Final output: "${finalOutput.slice(0, 200)}"`,
    ``,
    `## Steps`,
    stepsDesc,
    ``,
    `For each step, assess whether it is:`,
    `- "ok": output looks fine, no issues`,
    `- "suspicious": output might have caused downstream problems`,
    `- "root-cause": this step is most likely what caused the failure`,
    ``,
    `Only ONE step should be "root-cause".`,
    `Consider: errors, failed checks, outputs that look wrong for their type,`,
    `outputs that would cause the next step to fail.`,
    ``,
    `Reply with JSON only (no markdown):`,
    `{`,
    `  "stepScores": [`,
    `    {`,
    `      "name": "step-name",`,
    `      "status": "ok" | "suspicious" | "root-cause",`,
    `      "suspicionScore": 0.0-1.0,`,
    `      "issue": "brief issue description or null"`,
    `    }`,
    `  ]`,
    `}`,
  ].join('\n')

  try {
    const text = await callAnalysisModel(prompt, config)
    const parsed = JSON.parse(text.replace(/```json|```/g, '').trim()) as {
      stepScores: Array<{
        name: string
        status: 'ok' | 'suspicious' | 'root-cause'
        suspicionScore: number
        issue?: string
      }>
    }

    return trace.steps.map(step => {
      const score = parsed.stepScores.find(s => s.name === step.name)
      return {
        step,
        status: score?.status ?? 'ok',
        suspicionScore: score?.suspicionScore ?? 0,
        issue: score?.issue,
      }
    })
  } catch {
    // Fallback: mark errored steps as suspicious
    return trace.steps.map(step => ({
      step,
      status: step.error ? 'root-cause' as const : 'ok' as const,
      suspicionScore: step.error ? 1.0 : 0.0,
      issue: step.error,
    }))
  }
}

function findRootCause(
  steps: TraceStep[],
  diagnoses: TraceWhyResult['stepDiagnoses']
): TraceStep | null {
  // First look for explicitly marked root cause
  const rootDiag = diagnoses.find(d => d.status === 'root-cause')
  if (rootDiag) return rootDiag.step

  // Fall back to highest suspicion score
  const highest = diagnoses.reduce((a, b) =>
    a.suspicionScore > b.suspicionScore ? a : b
  )
  return highest.suspicionScore > 0.3 ? highest.step : null
}

async function synthesizeTraceDiagnosis(
  trace: Trace,
  rootStep: TraceStep | null,
  stepDiagnoses: TraceWhyResult['stepDiagnoses'],
  config: AILensConfig
): Promise<{ diagnosis: string; suggestedFix: string; severity: 'low' | 'medium' | 'high' }> {
  const rootStepDesc = rootStep
    ? [
        `Root cause step: "${rootStep.name}" [${rootStep.type}]`,
        `Input: "${rootStep.input.slice(0, 200)}"`,
        `Output: "${rootStep.output.slice(0, 200)}"`,
        rootStep.error ? `Error: ${rootStep.error}` : '',
      ].filter(Boolean).join('\n')
    : 'No clear root cause step identified.'

  const suspiciousSteps = stepDiagnoses
    .filter(d => d.status !== 'ok')
    .map(d => `- "${d.step.name}": ${d.issue ?? 'suspicious output'}`)
    .join('\n')

  const prompt = [
    `You are explaining why a multi-step AI pipeline failed.`,
    ``,
    `Pipeline: "${trace.name}"`,
    `Total steps: ${trace.steps.length}`,
    `Total latency: ${trace.totalLatencyMs}ms`,
    ``,
    rootStepDesc,
    ``,
    suspiciousSteps ? `Other suspicious steps:\n${suspiciousSteps}` : '',
    ``,
    `Write a clear diagnosis and fix for a developer.`,
    `Be specific: which step broke, what its output was, how it cascaded.`,
    ``,
    `Reply with JSON only (no markdown):`,
    `{`,
    `  "diagnosis": "1-2 paragraphs explaining the cascade failure",`,
    `  "suggestedFix": "specific code/prompt change to fix the root step",`,
    `  "severity": "low" | "medium" | "high"`,
    `}`,
  ].filter(Boolean).join('\n')

  try {
    const text = await callAnalysisModel(prompt, config)
    return JSON.parse(text.replace(/```json|```/g, '').trim()) as {
      diagnosis: string
      suggestedFix: string
      severity: 'low' | 'medium' | 'high'
    }
  } catch {
    return {
      diagnosis: 'Could not synthesize trace diagnosis.',
      suggestedFix: '',
      severity: 'low',
    }
  }
}
