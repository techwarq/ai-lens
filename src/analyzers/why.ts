import { LensCall, WhyResult, AILensConfig } from '../types'
import { analyzeCausalChain } from './causal'
import { hasAnalysisCredentials, MISSING_KEY_MESSAGE } from './llm'

export { callAnalysisModel } from './llm'

/** A call is "failing" if the user marked it bad, it errored, or a check did not pass. */
export function isFailingCall(call: LensCall): boolean {
  return call.feedback === 'bad' ||
    call.output.startsWith('[ERROR]') ||
    (call.checks?.some(ch => !ch.passed) ?? false)
}

/**
 * Diagnose failing calls. Only calls with evidence of failure are analyzed —
 * if nothing failed, returns [] rather than inventing problems in good outputs.
 */
export async function analyzeWhy(
  calls: LensCall[],
  config: AILensConfig
): Promise<WhyResult[]> {
  const targets = calls.filter(isFailingCall).slice(0, 10)
  if (targets.length === 0) return []

  if (!hasAnalysisCredentials(config)) throw new Error(MISSING_KEY_MESSAGE)

  // Run causal chain analysis in parallel (max 3 at once to avoid rate limits)
  const results: WhyResult[] = []
  for (let i = 0; i < targets.length; i += 3) {
    const batch = targets.slice(i, i + 3)
    const batchResults = await Promise.all(
      batch.map(call => analyzeCausalChain(call, config))
    )
    results.push(...batchResults)
  }
  return results
}
