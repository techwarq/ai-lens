import * as crypto from 'crypto'
import {
  Trace,
  TraceStep,
  StepType,
  StepRunOptions,
  AsyncRunOptions,
  AILensConfig,
  CheckResult,
} from '../types'
import { Storage } from '../storage'
import { AILensCheckError } from '../errors'
import { pollUntilDone } from './poll'

/**
 * Tracer — tracks a full multi-step pipeline as a single unit.
 *
 * Handles:
 *   - LLM → LLM → image gen → video gen chains
 *   - Agent tool call loops
 *   - RAG pipelines
 *   - Any async media generation (polling)
 *
 * Every step is logged individually AND linked by traceId, so
 * aiwhy can diagnose the root cause across the whole chain.
 */
export class Tracer {
  private trace: Trace
  private storage: Storage
  private config: AILensConfig

  constructor(
    name: string,
    sessionId: string,
    storage: Storage,
    config: AILensConfig
  ) {
    this.config = config
    this.storage = storage
    this.trace = {
      id: crypto.randomUUID(),
      name,
      steps: [],
      totalLatencyMs: 0,
      success: true,
      timestamp: Date.now(),
      sessionId,
    }
  }

  getTraceId(): string {
    return this.trace.id
  }

  /**
   * Run a text LLM step — prompt → string output.
   * Pass the prompt so it is recorded and available to `ailens traces why`:
   *   t.run('summarize', prompt, () => llm(prompt))
   * The prompt-less form t.run('summarize', () => llm(prompt)) still works.
   */
  async run(name: string, fn: () => Promise<string>, options?: StepRunOptions): Promise<string>
  async run(name: string, prompt: string, fn: () => Promise<string>, options?: StepRunOptions): Promise<string>
  async run(
    name: string,
    promptOrFn: string | (() => Promise<string>),
    fnOrOptions?: (() => Promise<string>) | StepRunOptions,
    maybeOptions?: StepRunOptions
  ): Promise<string> {
    if (typeof promptOrFn === 'string') {
      if (typeof fnOrOptions !== 'function') {
        throw new TypeError(`t.run("${name}", prompt, fn): fn must be a function`)
      }
      return this.runStep(name, 'llm', fnOrOptions, maybeOptions ?? {}, promptOrFn)
    }
    return this.runStep(name, 'llm', promptOrFn, (fnOrOptions as StepRunOptions | undefined) ?? {})
  }

  /**
   * Run an image generation step — prompt → image URL
   */
  async image(
    name: string,
    prompt: string,
    fn: () => Promise<string>,
    options: StepRunOptions = {}
  ): Promise<string> {
    return this.runStep(name, 'image-gen', fn, {
      ...options,
      mediaType: 'image',
      meta: { ...options.meta, prompt },
    }, prompt)
  }

  /**
   * Run a video generation step — prompt → video URL
   * Use this for synchronous video gen (returns immediately)
   */
  async video(
    name: string,
    prompt: string,
    fn: () => Promise<string>,
    options: StepRunOptions = {}
  ): Promise<string> {
    return this.runStep(name, 'video-gen', fn, {
      ...options,
      mediaType: 'video',
      meta: { ...options.meta, prompt },
    }, prompt)
  }

  /**
   * Run an async media generation step — submits job, polls for result.
   * Perfect for Runway, Sora, Kling, etc.
   *
   * @param submitFn  — submits the job, returns a job ID
   * @param pollFn    — polls for result given job ID, returns null if not ready
   */
  async runAsync(
    name: string,
    prompt: string,
    submitFn: () => Promise<string>,
    pollFn: (jobId: string) => Promise<string | null>,
    options: AsyncRunOptions = {}
  ): Promise<string> {
    const {
      pollInterval = 3000,
      timeout = 300_000,
      mediaType = 'video',
    } = options

    const start = Date.now()
    let jobId = ''
    let result: string | null = null
    let error: string | undefined

    try {
      // Submit the job
      jobId = await submitFn()

      if (this.config.verbose) {
        console.log(`[ailens] ${name} submitted, job: ${jobId}`)
      }

      result = await pollUntilDone(jobId, pollFn, pollInterval, timeout, elapsed => {
        if (this.config.verbose) {
          console.log(`[ailens] ${name} polling... (${Math.round(elapsed / 1000)}s)`)
        }
      })

      if (result === null) {
        error = `Timed out after ${timeout / 1000}s waiting for ${name}`
      }
    } catch (e) {
      error = (e as Error).message
    }

    const latencyMs = Date.now() - start
    const output = result ?? ''

    const step = this.buildStep(name, mediaType === 'video' ? 'video-gen' : 'image-gen', {
      input: prompt,
      output,
      latencyMs,
      error,
      mediaUrl: output,
      mediaType,
      meta: { ...options.meta, jobId, prompt },
      dependsOn: options.dependsOn,
      sessionId: this.trace.sessionId,
    })

    this.recordStep(step)

    if (error) throw new Error(error)
    return output
  }

  /**
   * Run a tool call step — logs the tool name, args, and result
   */
  async tool(
    name: string,
    args: Record<string, unknown>,
    fn: () => Promise<unknown>,
    options: StepRunOptions = {}
  ): Promise<unknown> {
    const input = JSON.stringify(args, null, 2)
    let rawResult: unknown

    const result = await this.runStep(
      name,
      'tool-call',
      async () => {
        rawResult = await fn()
        return typeof rawResult === 'string'
          ? rawResult
          : JSON.stringify(rawResult)
      },
      { ...options, meta: { ...options.meta, toolArgs: args } },
      input
    )

    return rawResult ?? result
  }

  /**
   * Run a retrieval/RAG step
   */
  async retrieve(
    name: string,
    query: string,
    fn: () => Promise<string>,
    options: StepRunOptions = {}
  ): Promise<string> {
    return this.runStep(name, 'retrieval', fn, options, query)
  }

  /**
   * Mark the whole trace as good or bad
   */
  feedback(value: 'good' | 'bad'): void {
    this.trace.feedback = value
    this.storage.saveTrace(this.trace)
  }

  /**
   * Mark a specific step as good or bad. If several steps share the name
   * (e.g. an agent loop), the most recent one is marked.
   */
  stepFeedback(stepName: string, value: 'good' | 'bad'): void {
    const step = [...this.trace.steps].reverse().find(s => s.name === stepName)
    if (step) {
      step.feedback = value
      this.storage.saveTrace(this.trace)
    }
  }

  /**
   * Get the completed trace object
   */
  getTrace(): Trace {
    return { ...this.trace, steps: [...this.trace.steps] }
  }

  /** @internal — called by AILens.trace() when the pipeline returns */
  finish(result: unknown): void {
    if (result !== undefined) {
      this.trace.finalOutput = typeof result === 'string' ? result : safeStringify(result)
    }
    this.storage.saveTrace(this.trace)
  }

  /** @internal — called by AILens.trace() when the pipeline throws */
  fail(e: unknown): void {
    this.trace.success = false
    this.trace.error = e instanceof Error ? e.message : String(e)
    this.storage.saveTrace(this.trace)
  }

  // ── Private helpers ────────────────────────────────────────────────────

  private async runStep(
    name: string,
    type: StepType,
    fn: () => Promise<string>,
    options: StepRunOptions = {},
    explicitInput?: string
  ): Promise<string> {
    const start = Date.now()
    let output = ''
    let error: string | undefined

    try {
      output = await fn()
    } catch (e) {
      error = (e as Error).message
      output = `[ERROR] ${error}`
    }

    const latencyMs = Date.now() - start

    // Run G-Eval checks if specified (gevalBatch never throws; errors are on each result)
    let checks: CheckResult[] | undefined
    if (options.check && options.check.length > 0 && !error) {
      const { gevalBatch } = await import('../analyzers/geval')
      checks = await gevalBatch(output, options.check, this.config, { input: explicitInput })
    }

    const step = this.buildStep(name, type, {
      input: explicitInput ?? '',
      output,
      latencyMs,
      error,
      mediaUrl: type === 'image-gen' || type === 'video-gen' ? output : undefined,
      mediaType: options.mediaType,
      checks,
      meta: options.meta,
      dependsOn: options.dependsOn ?? this.inferDependsOn(name),
      sessionId: this.trace.sessionId,
    })

    this.recordStep(step)

    if (this.config.verbose) {
      const icon = error ? '✗' : '✓'
      const checksStr = checks
        ? ` [${checks.filter(c => c.passed).length}/${checks.length} checks passed]`
        : ''
      console.log(`[ailens trace:${this.trace.name}] ${icon} ${name} (${latencyMs}ms)${checksStr}`)
    }

    if (error) throw new Error(error)

    // Throw on failed checks if user requested enforcement
    const failed = checks
      ? checks.filter(c => !c.passed && !(c.error && this.config.checkErrors === 'ignore')).map(c => c.rule)
      : []
    if (checks && failed.length > 0) {
      const fakeCall = {
        id: step.id, timestamp: step.timestamp, prompt: step.input,
        output, model: 'unknown', provider: 'unknown',
        latencyMs, checks, sessionId: this.trace.sessionId,
      }
      throw new AILensCheckError(
        `Step "${name}" failed checks:\n${failed.map(r => `  - ${r}`).join('\n')}`,
        fakeCall as any,
        checks
      )
    }

    return output
  }

  private buildStep(
    name: string,
    type: StepType,
    data: Partial<TraceStep> & { input: string; output: string; latencyMs: number }
  ): TraceStep {
    return {
      id: crypto.randomUUID(),
      traceId: this.trace.id,
      name,
      type,
      timestamp: Date.now(),
      sessionId: this.trace.sessionId,
      ...data,
    }
  }

  private recordStep(step: TraceStep): void {
    this.trace.steps.push(step)
    this.trace.totalLatencyMs += step.latencyMs
    if (step.error) this.trace.success = false
    this.storage.saveTrace(this.trace)
  }

  /** Auto-infer dependsOn — each step depends on the previous one by default */
  private inferDependsOn(currentName: string): string[] {
    const steps = this.trace.steps
    if (steps.length === 0) return []
    const last = steps[steps.length - 1]
    return last ? [last.name] : []
  }
}

function safeStringify(value: unknown): string {
  try { return JSON.stringify(value) ?? String(value) } catch { return String(value) }
}
