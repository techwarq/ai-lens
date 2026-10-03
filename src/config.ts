import * as fs from 'fs'
import * as path from 'path'
import { AILensConfig } from './types'

type Provider = NonNullable<AILensConfig['analysisProvider']>

const PROVIDERS: Provider[] = ['anthropic', 'openai', 'openai-compatible']

/** Read .ailens/config.json (or <logDir>/config.json). Returns {} if missing or unreadable. */
export function readConfigFile(logDir: string): Partial<AILensConfig> {
  const file = path.join(logDir, 'config.json')
  if (!fs.existsSync(file)) return {}
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8')) as Partial<AILensConfig>
  } catch {
    return {}
  }
}

/** Pick the API key env var that belongs to the chosen provider. */
function apiKeyFromEnv(provider: Provider, env: NodeJS.ProcessEnv): string | undefined {
  if (env.AILENS_API_KEY) return env.AILENS_API_KEY
  if (provider === 'anthropic') return env.ANTHROPIC_API_KEY
  // openai-compatible gateways/proxies commonly reuse OPENAI_API_KEY
  return env.OPENAI_API_KEY
}

/**
 * Resolve config with priority: code config > env vars > config.json > defaults.
 * Used by both the SDK and the CLI so they always agree.
 */
export function resolveConfig(
  code: AILensConfig = {},
  env: NodeJS.ProcessEnv = process.env
): AILensConfig {
  const logDir = code.logDir ?? env.AILENS_LOG_DIR ?? '.ailens'
  const file = readConfigFile(logDir)

  const envProvider = PROVIDERS.includes(env.AILENS_PROVIDER as Provider)
    ? (env.AILENS_PROVIDER as Provider)
    : undefined
  const provider: Provider = code.analysisProvider ?? envProvider ?? file.analysisProvider ?? 'anthropic'

  // Model/base URL/key in config.json belong to the provider written there — don't
  // send e.g. a Claude model name to OpenAI after switching provider via env or code.
  const fromFile: Partial<AILensConfig> = { ...file }
  if (file.analysisProvider && file.analysisProvider !== provider) {
    delete fromFile.analysisModel
    delete fromFile.analysisBaseURL
    delete fromFile.analysisApiKey
  }

  return {
    maxLogs: 1000,
    verbose: false,
    checkErrors: 'fail',
    ...stripUndefined(fromFile),
    ...stripUndefined({
      analysisModel: env.AILENS_MODEL,
      analysisBaseURL: env.AILENS_BASE_URL,
      analysisApiKey: apiKeyFromEnv(provider, env),
    }),
    ...stripUndefined(code),
    logDir,
    analysisProvider: provider,
  }
}

function stripUndefined<T extends object>(obj: T): Partial<T> {
  const out: Partial<T> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined) (out as Record<string, unknown>)[k] = v
  }
  return out
}
