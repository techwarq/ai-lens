import { AILensConfig } from '../types'

export class AnalysisModelError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AnalysisModelError'
  }
}

export const MISSING_KEY_MESSAGE =
  '\nNo API key found for analysis.\n\n' +
  'Set one of these in your environment or .env file:\n' +
  '  ANTHROPIC_API_KEY=sk-ant-...   (default provider)\n' +
  '  OPENAI_API_KEY=sk-...          (set AILENS_PROVIDER=openai)\n' +
  '  AILENS_API_KEY=...             (any provider)\n\n' +
  'Or configure in code: lens({ analysisApiKey: "..." })'

export interface AnalysisCallOptions {
  /** Max output tokens. Default: 1024 */
  maxTokens?: number
}

/** True when the configured provider can be called (local OpenAI-compatible servers need no key). */
export function hasAnalysisCredentials(config: AILensConfig): boolean {
  if (config.analysisApiKey) return true
  return config.analysisProvider === 'openai-compatible' && !!config.analysisBaseURL
}

/**
 * Call the configured analysis model. Throws AnalysisModelError on missing credentials,
 * HTTP errors, API error bodies, empty responses and truncated (max_tokens) responses —
 * callers must never mistake a failed call for a real answer.
 */
export async function callAnalysisModel(
  prompt: string,
  config: AILensConfig,
  options: AnalysisCallOptions = {}
): Promise<string> {
  const provider = config.analysisProvider ?? 'anthropic'
  const maxTokens = options.maxTokens ?? 1024

  if (!hasAnalysisCredentials(config)) throw new AnalysisModelError(MISSING_KEY_MESSAGE)

  if (provider === 'anthropic') {
    const data = await postJson(
      'https://api.anthropic.com/v1/messages',
      {
        'x-api-key': config.analysisApiKey!,
        'anthropic-version': '2023-06-01',
      },
      {
        model: config.analysisModel ?? 'claude-sonnet-4-6',
        max_tokens: maxTokens,
        temperature: 0,
        messages: [{ role: 'user', content: prompt }],
      }
    ) as { content?: Array<{ type?: string; text?: string }>; stop_reason?: string }

    if (data.stop_reason === 'max_tokens') {
      throw new AnalysisModelError(`Analysis response was truncated at max_tokens=${maxTokens}`)
    }
    const text = (data.content ?? [])
      .filter(b => typeof b.text === 'string')
      .map(b => b.text)
      .join('')
    if (!text) throw new AnalysisModelError('Analysis model returned an empty response')
    return text
  }

  // OpenAI and any OpenAI-compatible provider (Groq, Ollama, Mistral, Together, Fireworks, etc.)
  const baseURL = config.analysisBaseURL
    ?? (provider === 'openai' ? 'https://api.openai.com/v1' : undefined)

  if (!baseURL) {
    throw new AnalysisModelError(
      `analysisBaseURL is required when using provider "${provider}". ` +
      `e.g. 'https://api.groq.com/openai/v1' or 'http://localhost:11434/v1'`
    )
  }

  const headers: Record<string, string> = {}
  if (config.analysisApiKey) headers['Authorization'] = `Bearer ${config.analysisApiKey}`

  const data = await postJson(`${baseURL.replace(/\/$/, '')}/chat/completions`, headers, {
    model: config.analysisModel ?? 'gpt-4o',
    max_tokens: maxTokens,
    temperature: 0,
    messages: [{ role: 'user', content: prompt }],
  }) as { choices?: Array<{ message?: { content?: string }; finish_reason?: string }> }

  const choice = data.choices?.[0]
  if (choice?.finish_reason === 'length') {
    throw new AnalysisModelError(`Analysis response was truncated at max_tokens=${maxTokens}`)
  }
  const text = choice?.message?.content ?? ''
  if (!text) throw new AnalysisModelError('Analysis model returned an empty response')
  return text
}

async function postJson(
  url: string,
  headers: Record<string, string>,
  body: unknown
): Promise<unknown> {
  let res: Response
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
    })
  } catch (e) {
    throw new AnalysisModelError(`Could not reach analysis model at ${url}: ${(e as Error).message}`)
  }

  const raw = await res.text()
  if (!res.ok) {
    throw new AnalysisModelError(`Analysis model returned HTTP ${res.status}: ${raw.slice(0, 300)}`)
  }
  try {
    return JSON.parse(raw)
  } catch {
    throw new AnalysisModelError(`Analysis model returned non-JSON body: ${raw.slice(0, 300)}`)
  }
}

/**
 * Parse a JSON object out of a model response. Tolerates markdown fences and
 * leading/trailing prose; throws if no valid JSON object is present.
 */
export function parseJsonResponse<T>(text: string): T {
  const unfenced = text.replace(/```(?:json)?/gi, '').trim()
  try {
    return JSON.parse(unfenced) as T
  } catch {
    const start = unfenced.indexOf('{')
    const end = unfenced.lastIndexOf('}')
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(unfenced.slice(start, end + 1)) as T
      } catch {}
    }
    throw new AnalysisModelError(`Could not parse JSON from analysis model response: ${text.slice(0, 200)}`)
  }
}

/**
 * Wrap untrusted text (model outputs, user inputs) in tags for a judge prompt.
 * Neutralizes any closing tag inside the content so it cannot break out.
 */
export function fence(tag: string, content: string): string {
  const safe = content.split(`</${tag}>`).join(`<\\/${tag}>`)
  return `<${tag}>\n${safe}\n</${tag}>`
}
