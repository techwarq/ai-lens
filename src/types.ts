export interface AILensConfig {
  /** Directory to store logs. Default: .ailens/ */
  logDir?: string
  /** Model to use for analysis (why, diff). Default: reads from AILENS_MODEL env */
  analysisModel?: string
  /** API key for analysis model. Default: reads from env */
  analysisApiKey?: string
  /** Which provider to use for analysis: 'anthropic' | 'openai' */
  analysisProvider?: 'anthropic' | 'openai'
  /** Max log entries to keep per session. Default: 1000 */
  maxLogs?: number
  /** Whether to log to console as well. Default: false */
  verbose?: boolean
}

export interface LensCall {
  /** Unique ID for this call */
  id: string
  /** Unix timestamp */
  timestamp: number
  /** The prompt sent to the model */
  prompt: string
  /** Optional system prompt */
  system?: string
  /** Raw input passed by user (before prompt templating) */
  input?: unknown
  /** The model's output */
  output: string
  /** Model used */
  model: string
  /** Provider: openai, anthropic, etc */
  provider: string
  /** Latency in ms */
  latencyMs: number
  /** Token counts if available */
  tokens?: {
    input: number
    output: number
  }
  /** Optional tag for grouping calls */
  tag?: string
  /** Optional metadata from user */
  meta?: Record<string, unknown>
  /** User feedback: thumbs up/down */
  feedback?: 'good' | 'bad'
  /** Semantic checks that ran on this call */
  checks?: CheckResult[]
  /** Session ID — groups calls together in one run */
  sessionId: string
}

export interface CheckResult {
  rule: string
  passed: boolean
  score?: number
  reason?: string
}

export interface DiffResult {
  promptBefore: string
  promptAfter: string
  calls: {
    before: LensCall[]
    after: LensCall[]
  }
  analysis: {
    toneDelta?: string
    lengthDelta?: string
    behaviorChanges: string[]
    regressions: string[]
    improvements: string[]
    summary: string
    /** Embedding-based semantic distance (0=identical, 1=completely different) */
    driftScore?: number
    /** Cosine similarity between output clusters */
    cosineSimilarity?: number
    /** Per-category drift breakdown */
    slices?: Array<{
      category: string
      beforeCount: number
      afterCount: number
      driftScore: number
    }>
  }
}

export interface WhyResult {
  call: LensCall
  diagnosis: string
  promptIssues: string[]
  suggestedFix: string
  severity: 'low' | 'medium' | 'high'
  /** Causal chain analysis — which prompt spans caused which output spans */
  causalChain?: {
    spans: Array<{
      id: string
      text: string
      type: string
      suspicionScore: number
      forEvidence: string[]
      againstEvidence: string[]
    }>
    rootCause: { id: string; text: string; suspicionScore: number } | null
    confidence: number
  }
}

export interface RunOptions {
  /** Tag this call for later filtering */
  tag?: string
  /** Attach metadata */
  meta?: Record<string, unknown>
  /** Semantic rules to enforce (plain english) */
  check?: string[]
  /** Mark output as good/bad immediately */
  feedback?: 'good' | 'bad'
}

// ── Trace types (for agents, pipelines, multi-step workflows) ──────────────

export type StepType =
  | 'llm'           // text prompt → text output
  | 'image-gen'     // prompt → image URL
  | 'video-gen'     // prompt + optional image → video URL
  | 'tool-call'     // function/tool invocation
  | 'retrieval'     // vector search / RAG fetch
  | 'custom'        // anything else

export interface TraceStep {
  /** Step ID */
  id: string
  /** Parent trace ID */
  traceId: string
  /** Step name — e.g. 'prompt-refiner', 'image-gen', 'summarizer' */
  name: string
  /** What kind of step this is */
  type: StepType
  /** The input to this step (prompt, query, tool args, etc) */
  input: string
  /** The output — text, URL, JSON, etc */
  output: string
  /** For media outputs: the actual URL or base64 */
  mediaUrl?: string
  /** Media type if applicable */
  mediaType?: 'image' | 'video' | 'audio'
  /** Which step's output fed into this step */
  dependsOn?: string[]
  /** Latency in ms */
  latencyMs: number
  /** Did this step error? */
  error?: string
  /** Semantic checks run on this step's output */
  checks?: CheckResult[]
  /** User feedback on this specific step */
  feedback?: 'good' | 'bad'
  /** Optional tag */
  tag?: string
  /** Arbitrary metadata */
  meta?: Record<string, unknown>
  /** Unix timestamp */
  timestamp: number
  /** Session ID */
  sessionId: string
}

export interface Trace {
  /** Trace ID — groups all steps of one pipeline run */
  id: string
  /** Human-readable name for this pipeline */
  name: string
  /** All steps in order */
  steps: TraceStep[]
  /** Final output of the whole pipeline */
  finalOutput?: string
  /** Total latency across all steps */
  totalLatencyMs: number
  /** Did the whole trace succeed? */
  success: boolean
  /** Error if the trace failed */
  error?: string
  /** User feedback on the whole trace */
  feedback?: 'good' | 'bad'
  /** Unix timestamp */
  timestamp: number
  /** Session ID */
  sessionId: string
}

export interface TraceWhyResult {
  trace: Trace
  /** Which step is most likely the root cause */
  rootStep: TraceStep | null
  /** Per-step diagnosis */
  stepDiagnoses: Array<{
    step: TraceStep
    status: 'ok' | 'suspicious' | 'root-cause'
    suspicionScore: number
    issue?: string
  }>
  /** Overall diagnosis across the pipeline */
  diagnosis: string
  suggestedFix: string
  severity: 'low' | 'medium' | 'high'
}

export interface StepRunOptions extends RunOptions {
  /** Which previous step names this step depends on */
  dependsOn?: string[]
  /** For media outputs */
  mediaType?: 'image' | 'video' | 'audio'
}

export interface AsyncRunOptions extends StepRunOptions {
  /** How often to poll in ms. Default: 3000 */
  pollInterval?: number
  /** Max time to wait in ms. Default: 300000 (5 min) */
  timeout?: number
}
