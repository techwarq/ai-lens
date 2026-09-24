import * as crypto from 'crypto'
import { Tracer } from './tracer'
import { AILensConfig, LensCall, RunOptions, CheckResult } from '../types'
import { Storage } from '../storage'
import { AILensCheckError } from '../errors'
import { resolveConfig } from '../config'
import { pollUntilDone } from './poll'

export { AILensCheckError }

type MediaType = 'image' | 'video' | 'audio'

interface ExecuteExtras {
  system?: string
  /** Extra metadata computed from the output (e.g. media URL) */
  metaFromOutput?: (output: string) => Record<string, unknown>
}

/** Checks that should cause a throw, honouring the `checkErrors` setting. */
export function blockingFailures(checks: CheckResult[], config: AILensConfig): CheckResult[] {
  return checks.filter(c => !c.passed && !(c.error && config.checkErrors === 'ignore'))
}

export class AILens {
  private config: AILensConfig
  private storage: Storage
  private sessionId: string

  constructor(config: AILensConfig = {}) {
    this.sessionId = crypto.randomUUID()
    // Priority: code config > env vars > .ailens/config.json > defaults
    this.config = resolveConfig(config)
    this.storage = new Storage(this.config, this.sessionId)
  }

  /**
   * Wrap any LLM call. Pass your prompt and a function that calls your model.
   *
   * @example
   * const output = await lens.run(prompt, () => myLLM.call(prompt), { tag: 'summarizer' })
   */
  async run(
    prompt: string,
    fn: () => Promise<string>,
    options: RunOptions = {}
  ): Promise<string> {
    const { output } = await this.execute(prompt, fn, options)
    return output
  }

  /**
   * Same as run() but returns { output, id } so you can attach feedback by ID.
   *
   * @example
   * const { output, id } = await l.runWithId(prompt, () => myLLM.call(prompt))
   * if (userThumbsUp) l.feedback(id, 'good')
   */
  async runWithId(
    prompt: string,
    fn: () => Promise<string>,
    options: RunOptions = {}
  ): Promise<{ output: string; id: string }> {
    return this.execute(prompt, fn, options)
  }

  /**
   * Run with system prompt explicitly tracked
   */
  async runWithSystem(
    system: string,
    prompt: string,
    fn: () => Promise<string>,
    options: RunOptions = {}
  ): Promise<string> {
    const { output } = await this.execute(prompt, fn, options, { system })
    return output
  }

  /**
   * Mark a previous output as good or bad — builds up your test suite.
   * Works for calls from earlier sessions too. Returns false if the ID was not found.
   */
  feedback(callId: string, value: 'good' | 'bad'): boolean {
    const found = this.storage.updateCall(callId, { feedback: value })
    if (!found) console.warn(`[ailens] feedback(): no logged call with id ${callId}`)
    return found
  }

  /**
   * Get the current session ID — useful for passing to CLI commands
   */
  getSessionId(): string {
    return this.sessionId
  }

  /**
   * Get all calls from current session
   */
  getSession(): LensCall[] {
    return this.storage.readSession()
  }

  /**
   * Trace a full multi-step pipeline — agents, image/video workflows, RAG chains.
   * Every step is logged individually and linked by traceId.
   *
   * @example
   * const result = await l.trace('image-pipeline', async (t) => {
   *   const refined = await t.run('refine-prompt', input, () => llm.refine(input))
   *   const image   = await t.image('gen-image', refined, () => dalle.generate(refined))
   *   const video   = await t.runAsync('gen-video', refined,
   *     () => runway.submit(image),
   *     (id) => runway.poll(id)
   *   )
   *   return video
   * })
   */
  async trace<T>(
    name: string,
    fn: (tracer: Tracer) => Promise<T>
  ): Promise<T> {
    const tracer = new Tracer(name, this.sessionId, this.storage, this.config)

    if (this.config.verbose) {
      console.log(`[ailens] starting trace: ${name} (${tracer.getTraceId().slice(0, 8)})`)
    }

    try {
      const result = await fn(tracer)
      tracer.finish(result)
      if (this.config.verbose) {
        const t = tracer.getTrace()
        console.log(`[ailens] trace complete: ${name} | ${t.steps.length} steps | ${t.totalLatencyMs}ms`)
      }
      return result
    } catch (e) {
      tracer.fail(e)
      if (this.config.verbose) {
        console.error(`[ailens] trace failed: ${name} — ${(e as Error).message}`)
      }
      throw e
    }
  }

  /**
   * Run a media generation step (image, video) — output is a URL or base64.
   * Simpler than trace() when you just have a single media gen call.
   */
  async runMedia(
    prompt: string,
    fn: () => Promise<string>,
    options: RunOptions & { mediaType?: MediaType } = {}
  ): Promise<string> {
    const mediaType = options.mediaType ?? 'image'
    const { output } = await this.execute(prompt, fn, options, {
      metaFromOutput: out => ({ mediaType, mediaUrl: out }),
    })
    return output
  }

  /**
   * Run an async media generation job — submits, polls, returns result.
   * Perfect for Runway, Sora, Kling, Stable Video, etc.
   */
  async runAsync(
    prompt: string,
    submitFn: () => Promise<string>,
    pollFn: (jobId: string) => Promise<string | null>,
    options: RunOptions & {
      mediaType?: MediaType
      pollInterval?: number
      timeout?: number
    } = {}
  ): Promise<string> {
    const {
      pollInterval = 3000,
      timeout = 300_000,
      mediaType = 'video',
    } = options

    let jobId = ''
    const job = async (): Promise<string> => {
      jobId = await submitFn()
      const result = await pollUntilDone(jobId, pollFn, pollInterval, timeout, elapsed => {
        if (this.config.verbose) {
          console.log(`[ailens] polling ${mediaType} job ${jobId}... (${Math.round(elapsed / 1000)}s)`)
        }
      })
      if (result === null) throw new Error(`Timed out after ${timeout / 1000}s waiting for ${mediaType} job ${jobId}`)
      return result
    }

    const { output } = await this.execute(prompt, job, options, {
      metaFromOutput: out => ({ jobId, mediaType, mediaUrl: out }),
    })
    return output
  }

  /** Shared implementation for every run* method: call, check, log, enforce. */
  private async execute(
    prompt: string,
    fn: () => Promise<string>,
    options: RunOptions,
    extras: ExecuteExtras = {}
  ): Promise<{ output: string; id: string }> {
    const id = crypto.randomUUID()
    const start = Date.now()

    let output = ''
    let error: Error | undefined

    try {
      output = await fn()
    } catch (e) {
      error = e instanceof Error ? e : new Error(String(e))
      output = `[ERROR] ${error.message}`
    }

    const latencyMs = Date.now() - start

    let checks: CheckResult[] = []
    if (options.check && options.check.length > 0 && !error) {
      checks = await this.runChecks(output, options.check, { input: prompt, system: extras.system })
    }

    const extraMeta = !error && extras.metaFromOutput ? extras.metaFromOutput(output) : undefined
    const meta = extraMeta ? { ...options.meta, ...extraMeta } : options.meta

    const call: LensCall = {
      id,
      timestamp: Date.now(),
      prompt,
      ...(extras.system !== undefined ? { system: extras.system } : {}),
      input: options.meta?.input,
      output,
      model: options.model ?? 'unknown',
      provider: options.provider ?? 'unknown',
      latencyMs,
      tag: options.tag,
      meta,
      feedback: options.feedback,
      checks: checks.length > 0 ? checks : undefined,
      sessionId: this.sessionId,
    }

    this.storage.append(call)

    const failed = blockingFailures(checks, this.config)

    if (this.config.verbose) {
      console.log(`[ailens] ${id} | ${latencyMs}ms | ${prompt.slice(0, 60)}...`)
      for (const c of checks.filter(c => !c.passed)) {
        console.warn(`[ailens] ⚠ ${c.error ? 'check could not run' : 'check failed'}: ${c.rule}${c.error ? ` (${c.error})` : ''}`)
      }
    }

    if (error) throw error

    if (failed.length > 0) {
      throw new AILensCheckError(
        `Output failed semantic checks:\n${failed.map(c => `  - ${c.rule}${c.error ? ` (could not evaluate: ${c.error})` : ''}`).join('\n')}`,
        call,
        checks
      )
    }

    return { output, id }
  }

  private async runChecks(
    output: string,
    rules: string[],
    context?: { input?: string; system?: string }
  ): Promise<CheckResult[]> {
    const { gevalBatch } = await import('../analyzers/geval')
    return gevalBatch(output, rules, this.config, context)
  }
}

// Convenience singleton factory
let _default: AILens | null = null

export function createLens(config?: AILensConfig): AILens {
  return new AILens(config)
}

export function getDefaultLens(): AILens {
  if (!_default) _default = new AILens()
  return _default
}

/** Shorthand: configure and export a ready-to-use lens */
export function lens(config?: AILensConfig): AILens {
  return new AILens(config)
}
