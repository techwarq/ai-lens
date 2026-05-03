import { LensCall, WhyResult, AILensConfig } from '../types'
import { analyzeCausalChain } from './causal'

export async function analyzeWhy(
  calls: LensCall[],
  config: AILensConfig
): Promise<WhyResult[]> {
  const apiKey = config.analysisApiKey
  if (!apiKey) {
    throw new Error(
      'No API key found. Set AILENS_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY'
    )
  }

  const badCalls = calls.filter(c =>
    c.feedback === 'bad' ||
    c.output.startsWith('[ERROR]') ||
    (c.checks && c.checks.some(ch => !ch.passed))
  )

  const targets = badCalls.length > 0 ? badCalls.slice(0, 10) : calls.slice(0, 5)

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

export async function callAnalysisModel(
  prompt: string,
  config: AILensConfig
): Promise<string> {
  const apiKey = config.analysisApiKey!

  if (config.analysisProvider === 'openai') {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: config.analysisModel ?? 'gpt-4o',
        max_tokens: 800,
        messages: [{ role: 'user', content: prompt }],
      }),
    })
    const data = await res.json() as { choices: Array<{ message: { content: string } }> }
    return data.choices[0]?.message?.content ?? ''
  }

  // Default: Anthropic
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: config.analysisModel ?? 'claude-sonnet-4-6',
      max_tokens: 800,
      messages: [{ role: 'user', content: prompt }],
    }),
  })
  const data = await res.json() as { content: Array<{ text: string }> }
  return data.content[0]?.text ?? ''
}
