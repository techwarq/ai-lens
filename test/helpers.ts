import * as fs from 'fs'
import * as os from 'os'
import * as path from 'path'

export interface RecordedRequest {
  url: string
  headers: Record<string, string>
  body: any
}

type Handler = (req: RecordedRequest) => { status?: number; body: unknown } | Promise<{ status?: number; body: unknown }>

const realFetch = globalThis.fetch

/** Replace global fetch with a handler; returns the list of recorded requests. */
export function mockFetch(handler: Handler): RecordedRequest[] {
  const requests: RecordedRequest[] = []
  globalThis.fetch = (async (input: any, init?: any) => {
    const req: RecordedRequest = {
      url: String(input),
      headers: { ...(init?.headers ?? {}) },
      body: init?.body ? JSON.parse(init.body) : undefined,
    }
    requests.push(req)
    const { status = 200, body } = await handler(req)
    const text = typeof body === 'string' ? body : JSON.stringify(body)
    return new Response(text, { status, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
  return requests
}

export function restoreFetch(): void {
  globalThis.fetch = realFetch
}

/** Anthropic Messages API response carrying `text`. */
export function anthropicReply(text: string, stopReason = 'end_turn') {
  return { body: { content: [{ type: 'text', text }], stop_reason: stopReason } }
}

/** The text of the (single) user message in a recorded Anthropic request. */
export function promptOf(req: RecordedRequest): string {
  return req.body.messages[0].content
}

/** Is this request the rubric (step generation) call rather than a scoring call? */
export function isStepGeneration(req: RecordedRequest): boolean {
  return promptOf(req).includes('designing an evaluation rubric')
}

/** Fresh temp log dir per test. */
export function tempLogDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'ailens-test-'))
}

const ENV_KEYS = [
  'AILENS_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY',
  'AILENS_PROVIDER', 'AILENS_MODEL', 'AILENS_BASE_URL', 'AILENS_LOG_DIR',
]

/** Remove ailens-related env vars so the developer's shell can't leak into tests. */
export function clearEnv(): void {
  for (const k of ENV_KEYS) delete process.env[k]
}
