import * as crypto from 'crypto'
import { Tracer } from './tracer'
import { AILensConfig, LensCall, RunOptions, CheckResult } from '../types'
import { Storage } from '../storage'
import { AILensCheckError } from '../errors'

export { AILensCheckError }

export class AILens {
  private config: AILensConfig
  private storage: Storage
  private sessionId: string

  constructor(config: AILensConfig = {}) {
    // Merge config with any saved config
    this.sessionId = crypto.randomUUID()

    // Load defaults from env
    const resolved: AILensConfig = {
      logDir: process.env.AILENS_LOG_DIR ?? '.ailens',
      analysisProvider: (process.env.AILENS_PROVIDER as AILensConfig['analysisProvider']) ?? 'anthropic',
      analysisModel: process.env.AILENS_MODEL,
      analysisApiKey: process.env.AILENS_API_KEY ?? process.env.ANTHROPIC_API_KEY ?? process.env.OPENAI_API_KEY,
      maxLogs: 1000,
      verbose: false,
      ...config,
    }

    this.config = resolved
    this.storage = new Storage(resolved, this.sessionId)
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
    const id = crypto.randomUUID()
    const start = Date.now()

    let output = ''
    let error: Error | undefined

    try {
      output = await fn()
    } catch (e) {
      error = e as Error
      output = `[ERROR] ${error.message}`
    }

    const latencyMs = Date.now() - start

    let checks: CheckResult[] = []
    if (options.check && options.check.length > 0 && !error) {
      checks = await this.runChecks(output, options.check)
    }

    const call: LensCall = {
      id,
      timestamp: Date.now(),
      prompt,
      input: options.meta?.input,
      output,
      model: this.config.analysisModel ?? 'unknown',
      provider: this.config.analysisProvider ?? 'unknown',
      latencyMs,
      tag: options.tag,
      meta: options.meta,
      feedback: options.feedback,
      checks: checks.length > 0 ? checks : undefined,
      sessionId: this.sessionId,
    }

    this.storage.append(call)

    if (this.config.verbose) {
      console.log(`[ailens] ${id} | ${latencyMs}ms | ${prompt.slice(0, 60)}...`)
    }

    if (options.check && checks.some(c => !c.passed)) {
      const failed = checks.filter(c => !c.passed).map(c => c.rule)
      throw new AILensCheckError(
        `Output failed semantic checks:\n${failed.map(r => `  - ${r}`).join('\n')}`,
        call,
        checks
      )
    }

    if (error) throw error

    return { output, id }
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
    const id = crypto.randomUUID()
    const start = Date.now()

    let output = ''
    let error: Error | undefined

    try {
      output = await fn()
    } catch (e) {
      error = e as Error
      output = `[ERROR] ${error.message}`
    }

    const latencyMs = Date.now() - start

    // Run semantic checks if provided
    let checks: CheckResult[] = []
    if (options.check && options.check.length > 0 && !error) {
      checks = await this.runChecks(output, options.check)
    }

    const call: LensCall = {
      id,
      timestamp: Date.now(),
      prompt,
      input: options.meta?.input,
      output,
      model: this.config.analysisModel ?? 'unknown',
      provider: this.config.analysisProvider ?? 'unknown',
      latencyMs,
      tag: options.tag,
      meta: options.meta,
      feedback: options.feedback,
      checks: checks.length > 0 ? checks : undefined,
      sessionId: this.sessionId,
    }

    this.storage.append(call)

    if (this.config.verbose) {
      console.log(`[ailens] ${id} | ${latencyMs}ms | ${prompt.slice(0, 60)}...`)
      if (checks.some(c => !c.passed)) {
        console.warn(`[ailens] ⚠ Check failed for call ${id}`)
        checks.filter(c => !c.passed).forEach(c => {
          console.warn(`  ✗ ${c.rule}`)
        })
      }
    }

    // Throw if a semantic check failed (only if user explicitly wants enforcement)
    if (options.check && checks.some(c => !c.passed)) {
      const failed = checks.filter(c => !c.passed).map(c => c.rule)
      throw new AILensCheckError(
        `Output failed semantic checks:\n${failed.map(r => `  - ${r}`).join('\n')}`,
        call,
        checks
      )
    }

    if (error) throw error

    return output
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
    const id = crypto.randomUUID()
    const start = Date.now()
    const output = await fn()
    const latencyMs = Date.now() - start

    const call: LensCall = {
      id,
      timestamp: Date.now(),
      prompt,
      system,
      output,
      model: this.config.analysisModel ?? 'unknown',
      provider: this.config.analysisProvider ?? 'unknown',
      latencyMs,
      tag: options.tag,
      meta: options.meta,
      feedback: options.feedback,
      sessionId: this.sessionId,
    }

    this.storage.append(call)
    return output
  }

  /**
   * Mark a previous output as good or bad — builds up your test suite
   */
  feedback(callId: string, value: 'good' | 'bad'): void {
    this.storage.updateCall(callId, { feedback: value })
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
   *   const refined = await t.run('refine-prompt', () => llm.refine(input))
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

    let result: T
    try {
      result = await fn(tracer)
      if (this.config.verbose) {
        const t = tracer.getTrace()
        console.log(`[ailens] trace complete: ${name} | ${t.steps.length} steps | ${t.totalLatencyMs}ms`)
      }
    } catch (e) {
      if (this.config.verbose) {
        console.error(`[ailens] trace failed: ${name} — ${(e as Error).message}`)
      }
      throw e
    }

    return result
  }

  /**
   * Run a media generation step (image, video) — output is a URL or base64.
   * Simpler than trace() when you just have a single media gen call.
   */
  async runMedia(
    prompt: string,
    fn: () => Promise<string>,
    options: RunOptions & { mediaType?: 'image' | 'video' | 'audio' } = {}
  ): Promise<string> {
    const id = crypto.randomUUID()
    const start = Date.now()
    const output = await fn()
    const latencyMs = Date.now() - start

    const call: LensCall = {
      id,
      timestamp: Date.now(),
      prompt,
      output,
      model: this.config.analysisModel ?? 'unknown',
      provider: this.config.analysisProvider ?? 'unknown',
      latencyMs,
      tag: options.tag,
      meta: { ...options.meta, mediaType: options.mediaType ?? 'image', mediaUrl: output },
      feedback: options.feedback,
      sessionId: this.sessionId,
    }

    this.storage.append(call)

    if (this.config.verbose) {
      console.log(`[ailens] media ${options.mediaType ?? 'image'} | ${latencyMs}ms | ${output.slice(0, 60)}`)
    }

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
      mediaType?: 'image' | 'video' | 'audio'
      pollInterval?: number
      timeout?: number
    } = {}
  ): Promise<string> {
    const {
      pollInterval = 3000,
      timeout = 300_000,
      mediaType = 'video',
    } = options

    const id = crypto.randomUUID()
    const start = Date.now()
    let jobId = ''
    let result: string | null = null

    jobId = await submitFn()

    const deadline = start + timeout
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, pollInterval))
      result = await pollFn(jobId)
      if (result !== null) break
      if (this.config.verbose) {
        console.log(`[ailens] polling ${mediaType} job ${jobId}... (${Math.round((Date.now() - start) / 1000)}s)`)
      }
    }

    const output = result ?? ''
    const latencyMs = Date.now() - start

    const call: LensCall = {
      id,
      timestamp: Date.now(),
      prompt,
      output,
      model: this.config.analysisModel ?? 'unknown',
      provider: this.config.analysisProvider ?? 'unknown',
      latencyMs,
      tag: options.tag,
      meta: { ...options.meta, jobId, mediaType, mediaUrl: output },
      feedback: options.feedback,
      sessionId: this.sessionId,
    }

    this.storage.append(call)

    if (!result) throw new Error(`Timed out after ${timeout / 1000}s waiting for ${mediaType} job ${jobId}`)
    return output
  }

  private async runChecks(
    output: string,
    rules: string[],
    context?: { input?: string; system?: string }
  ): Promise<CheckResult[]> {
    if (!this.config.analysisApiKey) {
      return rules.map(rule => ({ rule, passed: true, reason: 'no api key — skipped' }))
    }
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
