import { LensCall, WhyResult, AILensConfig } from '../types'
import { callAnalysisModel } from './why'

/**
 * Causal chain analysis for aiwhy.
 *
 * Instead of asking "why is this bad?" directly, we first build a causal
 * graph: which span of the prompt triggered which span of the output.
 * Then we diagnose based on the causal structure.
 *
 * Inspired by FVDebug (NVIDIA 2025): structure failure traces into DAGs,
 * use for-and-against prompting to identify suspicious nodes, then generate
 * causal explanations.
 *
 * Pipeline:
 *   1. Span extraction — split prompt into semantic chunks
 *   2. Causal attribution — which output tokens trace back to which prompt spans
 *   3. Suspicion scoring — for-and-against prompting on each span
 *   4. Root cause synthesis — explain the causal chain in plain english
 */

export interface CausalSpan {
  id: string
  text: string
  type: 'instruction' | 'context' | 'constraint' | 'example' | 'format' | 'other'
  suspicionScore: number   // 0-1, how likely this span caused the problem
  forEvidence: string[]    // reasons this span IS the cause
  againstEvidence: string[] // reasons this span is NOT the cause
}

export interface CausalChain {
  spans: CausalSpan[]
  outputSpans: Array<{ text: string; causedBy: string[] }>
  rootCause: CausalSpan | null
  diagnosis: string
  promptIssues: string[]
  suggestedFix: string
  severity: 'low' | 'medium' | 'high'
  confidence: number // 0-1, how confident we are in the diagnosis
}

export async function analyzeCausalChain(
  call: LensCall,
  config: AILensConfig
): Promise<WhyResult> {
  const fullPrompt = call.system
    ? `SYSTEM: ${call.system}\n\nUSER: ${call.prompt}`
    : call.prompt

  const failedChecks = call.checks?.filter(c => !c.passed) ?? []
  const failureContext = failedChecks.length > 0
    ? `Failed checks:\n${failedChecks.map(c => `- ${c.rule}: ${c.reason}`).join('\n')}`
    : call.feedback === 'bad'
      ? 'User marked this output as bad.'
      : 'Output appears problematic.'

  // Step 1: Extract semantic spans from the prompt
  const spans = await extractPromptSpans(fullPrompt, config)

  // Step 2: For each span, run for-and-against attribution
  const scoredSpans = await scoreSpansForAgainst(
    spans,
    call.output,
    failureContext,
    config
  )

  // Step 3: Synthesize root cause from the causal structure
  const chain = await synthesizeRootCause(
    scoredSpans,
    fullPrompt,
    call.output,
    failureContext,
    config
  )

  return {
    call,
    diagnosis: chain.diagnosis,
    promptIssues: chain.promptIssues,
    suggestedFix: chain.suggestedFix,
    severity: chain.severity,
    causalChain: chain,
  }
}

async function extractPromptSpans(
  prompt: string,
  config: AILensConfig
): Promise<Array<{ id: string; text: string; type: CausalSpan['type'] }>> {
  // For short prompts, treat as a single span
  if (prompt.length < 200) {
    return [{ id: 's0', text: prompt, type: 'instruction' }]
  }

  const extractPrompt = [
    `Split this prompt into semantic spans — distinct chunks that each serve a different purpose.`,
    ``,
    `Prompt:`,
    `"""`,
    prompt,
    `"""`,
    ``,
    `Identify each span's type:`,
    `- instruction: tells the model what to do`,
    `- context: provides background information`,
    `- constraint: sets limits or rules`,
    `- example: shows an example`,
    `- format: specifies output format`,
    `- other: anything else`,
    ``,
    `Reply with JSON only (no markdown):`,
    `{`,
    `  "spans": [`,
    `    { "id": "s0", "text": "exact text from the prompt", "type": "instruction" }`,
    `  ]`,
    `}`,
  ].join('\n')

  try {
    const text = await callAnalysisModel(extractPrompt, config)
    const parsed = JSON.parse(text.replace(/```json|```/g, '').trim()) as {
      spans: Array<{ id: string; text: string; type: CausalSpan['type'] }>
    }
    return parsed.spans ?? [{ id: 's0', text: prompt, type: 'instruction' }]
  } catch {
    return [{ id: 's0', text: prompt, type: 'instruction' }]
  }
}

async function scoreSpansForAgainst(
  spans: Array<{ id: string; text: string; type: CausalSpan['type'] }>,
  output: string,
  failureContext: string,
  config: AILensConfig
): Promise<CausalSpan[]> {
  if (spans.length === 0) return []

  // For 1-2 spans do them sequentially, otherwise batch
  const spanDescriptions = spans
    .map((s, i) => `Span ${s.id} [${s.type}]: "${s.text.slice(0, 150)}${s.text.length > 150 ? '...' : ''}"`)
    .join('\n')

  const attributionPrompt = [
    `You are doing causal attribution for a bad LLM output.`,
    ``,
    `## Failure context`,
    failureContext,
    ``,
    `## Model output`,
    `"""`,
    output.slice(0, 500),
    `"""`,
    ``,
    `## Prompt spans`,
    spanDescriptions,
    ``,
    `For each span, give:`,
    `- suspicionScore: 0.0-1.0 (how likely this span caused the problem)`,
    `- forEvidence: 1-2 reasons this span IS the root cause`,
    `- againstEvidence: 1-2 reasons this span is NOT the root cause`,
    ``,
    `Reply with JSON only (no markdown):`,
    `{`,
    `  "attributions": [`,
    `    {`,
    `      "id": "s0",`,
    `      "suspicionScore": 0.8,`,
    `      "forEvidence": ["reason 1"],`,
    `      "againstEvidence": ["reason 1"]`,
    `    }`,
    `  ]`,
    `}`,
  ].join('\n')

  try {
    const text = await callAnalysisModel(attributionPrompt, config)
    const parsed = JSON.parse(text.replace(/```json|```/g, '').trim()) as {
      attributions: Array<{
        id: string
        suspicionScore: number
        forEvidence: string[]
        againstEvidence: string[]
      }>
    }

    return spans.map(span => {
      const attr = parsed.attributions.find(a => a.id === span.id)
      return {
        ...span,
        suspicionScore: attr?.suspicionScore ?? 0,
        forEvidence: attr?.forEvidence ?? [],
        againstEvidence: attr?.againstEvidence ?? [],
      }
    })
  } catch {
    return spans.map(span => ({
      ...span,
      suspicionScore: 0,
      forEvidence: [],
      againstEvidence: [],
    }))
  }
}

async function synthesizeRootCause(
  spans: CausalSpan[],
  fullPrompt: string,
  output: string,
  failureContext: string,
  config: AILensConfig
): Promise<CausalChain> {
  // Find the most suspicious span
  const rootCause = spans.length > 0
    ? spans.reduce((a, b) => a.suspicionScore > b.suspicionScore ? a : b)
    : null

  const rootCauseText = rootCause
    ? `Most suspicious span (score ${rootCause.suspicionScore.toFixed(2)}): "${rootCause.text.slice(0, 200)}"`
    : 'No clear root cause span identified.'

  const synthesisPrompt = [
    `You are synthesizing a root cause analysis for a bad LLM output.`,
    ``,
    `## Failure`,
    failureContext,
    ``,
    `## Causal analysis`,
    rootCauseText,
    ``,
    `## Output`,
    `"""`,
    output.slice(0, 400),
    `"""`,
    ``,
    `Based on the causal analysis, provide:`,
    `1. A clear diagnosis (1 paragraph, what went wrong and why)`,
    `2. Specific prompt issues (not vague — point to the exact span)`,
    `3. A concrete suggested fix (rewrite of the problematic span)`,
    `4. Severity: low / medium / high`,
    `5. Confidence in this diagnosis: 0.0-1.0`,
    ``,
    `Reply with JSON only (no markdown):`,
    `{`,
    `  "diagnosis": "...",`,
    `  "promptIssues": ["issue 1", "issue 2"],`,
    `  "suggestedFix": "...",`,
    `  "severity": "medium",`,
    `  "confidence": 0.85`,
    `}`,
  ].join('\n')

  try {
    const text = await callAnalysisModel(synthesisPrompt, config)
    const parsed = JSON.parse(text.replace(/```json|```/g, '').trim()) as {
      diagnosis: string
      promptIssues: string[]
      suggestedFix: string
      severity: 'low' | 'medium' | 'high'
      confidence: number
    }

    return {
      spans,
      outputSpans: [],
      rootCause,
      diagnosis: parsed.diagnosis,
      promptIssues: parsed.promptIssues,
      suggestedFix: parsed.suggestedFix,
      severity: parsed.severity,
      confidence: parsed.confidence ?? 0.7,
    }
  } catch {
    return {
      spans,
      outputSpans: [],
      rootCause,
      diagnosis: 'Could not synthesize root cause.',
      promptIssues: [],
      suggestedFix: '',
      severity: 'low',
      confidence: 0,
    }
  }
}
