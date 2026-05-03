#!/usr/bin/env node
import * as path from 'path'
import * as fs from 'fs'
import { Storage } from '../storage'
import { analyzeWhy } from '../analyzers/why'
import { analyzeDiff } from '../analyzers/diff'
import { AILensConfig, LensCall } from '../types'

const VERSION = '0.1.0'

const HELP = `
ailens v${VERSION} — AI developer toolkit

USAGE
  npx ailens <command> [options]

COMMANDS
  why                  Diagnose recent bad AI outputs
  why --session <id>   Diagnose a specific session
  why --tag <tag>      Only look at calls with this tag

  diff <before> <after>  Compare two sessions and show behavior changes
  diff --last            Compare last two sessions automatically

  sessions             List all recorded sessions
  sessions --show <id> Show all calls in a session

  traces               List all recorded pipeline traces
  traces --show <id>   Show all steps in a trace
  traces why <id>      Diagnose why a trace failed

  init                 Create .ailens/config.json with defaults

EXAMPLES
  npx ailens why
  npx ailens why --tag summarizer
  npx ailens diff --last
  npx ailens diff abc123 def456
  npx ailens sessions

ENV VARS
  AILENS_API_KEY      API key for analysis (falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY)
  AILENS_PROVIDER     'anthropic' (default), 'openai', or 'openai-compatible'
  AILENS_BASE_URL     Base URL for OpenAI-compatible providers
                      e.g. https://api.groq.com/openai/v1
                           http://localhost:11434/v1  (Ollama)
  AILENS_MODEL        Model to use for analysis
  AILENS_LOG_DIR      Log directory (default: .ailens)
`

function loadDotEnv(): void {
  const envFile = path.join(process.cwd(), '.env')
  if (!fs.existsSync(envFile)) return
  for (const line of fs.readFileSync(envFile, 'utf-8').split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const eq = trimmed.indexOf('=')
    if (eq < 0) continue
    const key = trimmed.slice(0, eq).trim()
    const val = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, '')
    if (!(key in process.env)) process.env[key] = val
  }
}

async function main() {
  loadDotEnv()
  const args = process.argv.slice(2)
  const command = args[0]

  if (!command || command === '--help' || command === '-h') {
    console.log(HELP)
    process.exit(0)
  }

  if (command === '--version' || command === '-v') {
    console.log(VERSION)
    process.exit(0)
  }

  const config = loadConfig()

  switch (command) {
    case 'why':
      await runWhy(args.slice(1), config)
      break
    case 'diff':
      await runDiff(args.slice(1), config)
      break
    case 'sessions':
      await runSessions(args.slice(1), config)
      break
    case 'traces':
      await runTraces(args.slice(1), config)
      break
    case 'init':
      await runInit(config)
      break
    default:
      console.error(`Unknown command: ${command}\nRun 'npx ailens --help' for usage.`)
      process.exit(1)
  }
}

function loadConfig(): AILensConfig {
  const logDir = process.env.AILENS_LOG_DIR ?? '.ailens'
  const configFile = path.join(logDir, 'config.json')
  let saved: Partial<AILensConfig> = {}

  if (fs.existsSync(configFile)) {
    try {
      saved = JSON.parse(fs.readFileSync(configFile, 'utf-8'))
    } catch {}
  }

  return {
    logDir,
    analysisProvider: (process.env.AILENS_PROVIDER as AILensConfig['analysisProvider'])
      ?? saved.analysisProvider
      ?? 'anthropic',
    analysisModel: process.env.AILENS_MODEL ?? saved.analysisModel,
    analysisApiKey: process.env.AILENS_API_KEY
      ?? process.env.ANTHROPIC_API_KEY
      ?? process.env.OPENAI_API_KEY
      ?? saved.analysisApiKey,
    analysisBaseURL: process.env.AILENS_BASE_URL ?? saved.analysisBaseURL,
    ...saved,
  }
}

async function runWhy(args: string[], config: AILensConfig) {
  const storage = new Storage(config, 'cli')

  // Parse flags
  let sessionId: string | undefined
  let tag: string | undefined

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--session' && args[i + 1]) sessionId = args[++i]
    if (args[i] === '--tag' && args[i + 1]) tag = args[++i]
  }

  let calls: LensCall[]

  if (sessionId) {
    calls = storage.readSession(sessionId)
  } else {
    calls = storage.readAllRecent(50)
  }

  if (tag) {
    calls = calls.filter(c => c.tag === tag)
  }

  if (calls.length === 0) {
    console.log('No calls found. Make sure your app is using ailens and has run at least once.')
    return
  }

  console.log(`\n🔍 ailens why — analyzing ${calls.length} calls...\n`)

  const results = await analyzeWhy(calls, config)

  if (results.length === 0) {
    console.log('✓ No issues found in recent calls.')
    return
  }

  for (const result of results) {
    const icon = result.severity === 'high' ? '🔴' : result.severity === 'medium' ? '🟡' : '🟢'
    console.log(`${icon} Call ${result.call.id.slice(0, 8)} [${result.severity}]`)
    console.log(`   ${new Date(result.call.timestamp).toLocaleString()}`)
    console.log(`   Tag: ${result.call.tag ?? 'none'}\n`)

    console.log(`   📋 Prompt (first 120 chars):`)
    console.log(`   "${result.call.prompt.slice(0, 120)}..."\n`)

    console.log(`   💬 Output (first 120 chars):`)
    console.log(`   "${result.call.output.slice(0, 120)}..."\n`)

    console.log(`   🩺 Diagnosis:`)
    console.log(`   ${result.diagnosis}\n`)

    if (result.promptIssues.length > 0) {
      console.log(`   ⚠  Prompt issues:`)
      result.promptIssues.forEach(i => console.log(`   • ${i}`))
      console.log()
    }

    if (result.suggestedFix) {
      console.log(`   ✏  Suggested fix:`)
      console.log(`   ${result.suggestedFix}\n`)
    }

    // Show causal chain if available
    if (result.causalChain && result.causalChain.rootCause) {
      const rc = result.causalChain.rootCause
      const conf = (result.causalChain.confidence * 100).toFixed(0)
      console.log(`   🔗 Root cause span (${conf}% confidence):`)
      console.log(`   suspicion score: ${(rc.suspicionScore * 100).toFixed(0)}%`)
      console.log(`   "${rc.text.slice(0, 120)}"\n`)
    }

    console.log('─'.repeat(60) + '\n')
  }
}

async function runDiff(args: string[], config: AILensConfig) {
  const storage = new Storage(config, 'cli')

  let beforeId: string | undefined
  let afterId: string | undefined
  let useLast = false

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--last') useLast = true
    else if (!beforeId) beforeId = args[i]
    else if (!afterId) afterId = args[i]
  }

  if (useLast) {
    const sessions = storage.listSessions()
    if (sessions.length < 2) {
      console.error('Need at least 2 sessions to diff. Run your app a couple of times first.')
      process.exit(1)
    }
    afterId = sessions[0]
    beforeId = sessions[1]
  }

  if (!beforeId || !afterId) {
    console.error('Usage: ailens diff <before-session> <after-session>\n       ailens diff --last')
    process.exit(1)
  }

  const before = storage.readSession(beforeId)
  const after = storage.readSession(afterId)

  if (before.length === 0) {
    console.error(`Session not found: ${beforeId}`)
    process.exit(1)
  }
  if (after.length === 0) {
    console.error(`Session not found: ${afterId}`)
    process.exit(1)
  }

  console.log(`\n📊 ailens diff`)
  console.log(`   Before: ${beforeId.slice(0, 8)} (${before.length} calls)`)
  console.log(`   After:  ${afterId.slice(0, 8)} (${after.length} calls)\n`)
  console.log('Analyzing behavior changes...\n')

  const result = await analyzeDiff(before, after, config)

  console.log('─'.repeat(60))
  console.log('SUMMARY')
  console.log('─'.repeat(60))
  console.log(result.analysis.summary)
  console.log()

  const rows = [
    ['Tone', result.analysis.toneDelta ?? 'no change'],
    ['Length', result.analysis.lengthDelta ?? 'no change'],
  ]
  for (const [label, value] of rows) {
    console.log(`  ${label.padEnd(12)} ${value}`)
  }
  console.log()

  // Show semantic drift score if available
  if (result.analysis.driftScore !== undefined) {
    const level = result.analysis.driftScore > 0.3 ? '🔴 HIGH' : result.analysis.driftScore > 0.1 ? '🟡 MODERATE' : '🟢 LOW'
    console.log(`  Semantic drift   ${level} (${(result.analysis.driftScore * 100).toFixed(1)}% behavioral change)`)
    if (result.analysis.cosineSimilarity !== undefined) {
      console.log(`  Cosine sim       ${result.analysis.cosineSimilarity.toFixed(3)} (1.0 = identical outputs)`)
    }
    console.log()
  }

  // Show slice-level drift
  if (result.analysis.slices && result.analysis.slices.length > 0) {
    console.log('~ Behavioral slices:')
    result.analysis.slices.forEach(s => {
      const arrow = s.afterCount > s.beforeCount ? '↑' : s.afterCount < s.beforeCount ? '↓' : '~'
      console.log(`   ${arrow} "${s.category}": ${s.beforeCount} → ${s.afterCount} (${(s.driftScore * 100).toFixed(0)}% drift)`)
    })
    console.log()
  }

  if (result.analysis.improvements.length > 0) {
    console.log('✅ Improvements:')
    result.analysis.improvements.forEach(i => console.log(`   + ${i}`))
    console.log()
  }

  if (result.analysis.regressions.length > 0) {
    console.log('❌ Regressions:')
    result.analysis.regressions.forEach(r => console.log(`   - ${r}`))
    console.log()
  }

  if (result.analysis.behaviorChanges.length > 0) {
    console.log('~ Behavior changes:')
    result.analysis.behaviorChanges.forEach(c => console.log(`   ~ ${c}`))
    console.log()
  }

  console.log('─'.repeat(60))
  console.log('PROMPT DIFF')
  console.log('─'.repeat(60))
  printInlineDiff(result.promptBefore, result.promptAfter)
}

async function runSessions(args: string[], config: AILensConfig) {
  const storage = new Storage(config, 'cli')

  const showId = args.find((_, i) => args[i - 1] === '--show')

  if (showId) {
    const calls = storage.readSession(showId)
    if (calls.length === 0) {
      console.log(`No calls found in session ${showId}`)
      return
    }
    console.log(`\nSession ${showId} — ${calls.length} calls\n`)
    for (const call of calls) {
      const fb = call.feedback === 'good' ? '👍' : call.feedback === 'bad' ? '👎' : '  '
      console.log(`${fb} ${call.id.slice(0, 8)} | ${call.latencyMs}ms | ${call.tag ?? 'untagged'}`)
      console.log(`   ${call.prompt.slice(0, 80)}...`)
      console.log()
    }
    return
  }

  const sessions = storage.listSessions()
  if (sessions.length === 0) {
    console.log('No sessions recorded yet.\nMake sure your app uses ailens and has run.')
    return
  }

  console.log(`\nRecorded sessions (${sessions.length} total)\n`)
  console.log('ID              CALLS  DATE')
  console.log('─'.repeat(50))

  for (const sid of sessions.slice(0, 20)) {
    const calls = storage.readSession(sid)
    const date = calls[0]
      ? new Date(calls[0].timestamp).toLocaleString()
      : 'unknown'
    const badCount = calls.filter(c => c.feedback === 'bad').length
    const badStr = badCount > 0 ? ` (${badCount} bad)` : ''
    console.log(`${sid}  ${String(calls.length).padEnd(5)}  ${date}${badStr}`)
  }

  console.log()
  console.log(`Run 'npx ailens sessions --show <id>' to inspect a session`)
  console.log(`Run 'npx ailens why' to diagnose issues`)
  console.log(`Run 'npx ailens diff --last' to compare last two sessions`)
}

async function runTraces(args: string[], config: AILensConfig) {
  const storage = new Storage(config, 'cli')
  const showId = args.find((_, i) => args[i - 1] === '--show')
  const whyId = args.find((_, i) => args[i - 1] === 'why')

  if (showId) {
    const trace = storage.readTrace(showId)
    if (!trace) { console.log(`Trace not found: ${showId}`); return }
    console.log(`\nTrace: ${trace.name} | ${trace.steps.length} steps | ${trace.totalLatencyMs}ms | ${trace.success ? '✓ success' : '✗ failed'}`)
    console.log(`ID: ${trace.id}\n`)
    for (const step of trace.steps) {
      const icon = step.error ? '✗' : '✓'
      const checks = step.checks
        ? ` [${step.checks.filter((c: {passed: boolean}) => c.passed).length}/${step.checks.length} checks]`
        : ''
      console.log(`  ${icon} ${step.name.padEnd(20)} [${step.type.padEnd(10)}] ${step.latencyMs}ms${checks}`)
      if (step.error) console.log(`    ERROR: ${step.error}`)
      console.log(`    In:  ${step.input.slice(0, 80)}`)
      console.log(`    Out: ${step.output.slice(0, 80)}`)
      if (step.mediaUrl) console.log(`    URL: ${step.mediaUrl}`)
      console.log()
    }
    return
  }

  if (whyId) {
    const trace = storage.readTrace(whyId)
    if (!trace) { console.log(`Trace not found: ${whyId}`); return }
    console.log(`\n🔍 ailens traces why — analyzing "${trace.name}"...\n`)
    const { analyzeTraceWhy } = await import('../analyzers/trace-why')
    const result = await analyzeTraceWhy(trace, config)
    const icon = result.severity === 'high' ? '🔴' : result.severity === 'medium' ? '🟡' : '🟢'
    console.log(`${icon} [${result.severity}] ${trace.name}\n`)
    console.log('Step breakdown:')
    for (const d of result.stepDiagnoses) {
      const s = d.status === 'root-cause' ? '🔴 ROOT CAUSE' : d.status === 'suspicious' ? '🟡 suspicious' : '✓ ok'
      console.log(`  ${s.padEnd(18)} ${d.step.name} [${d.step.type}]`)
      if (d.issue) console.log(`              ${d.issue}`)
    }
    console.log()
    if (result.rootStep) {
      console.log(`🔗 Root cause: "${result.rootStep.name}" [${result.rootStep.type}]`)
      console.log(`   Input:  "${result.rootStep.input.slice(0, 100)}"`)
      console.log(`   Output: "${result.rootStep.output.slice(0, 100)}"\n`)
    }
    console.log('🩺 Diagnosis:')
    console.log(`   ${result.diagnosis}\n`)
    if (result.suggestedFix) {
      console.log('✏  Suggested fix:')
      console.log(`   ${result.suggestedFix}`)
    }
    return
  }

  // List all traces
  const traces = storage.readRecentTraces(20)
  if (traces.length === 0) {
    console.log('No traces recorded yet.\nUse l.trace("name", async (t) => { ... }) in your app.')
    return
  }
  console.log(`\nRecorded traces (${traces.length} total)\n`)
  console.log('ID                                    NAME                      STEPS  TIME    STATUS')
  console.log('─'.repeat(90))
  for (const t of traces) {
    const status = t.success ? '✓' : '✗'
    const date = new Date(t.timestamp).toLocaleString()
    console.log(`${t.id}  ${t.name.padEnd(24)}  ${String(t.steps.length).padEnd(5)}  ${String(t.totalLatencyMs)+'ms'.padEnd(8)} ${status}  ${date}`)
  }
  console.log()
  console.log(`Run 'npx ailens traces --show <id>' to inspect a trace`)
  console.log(`Run 'npx ailens traces why <id>' to diagnose a failed trace`)
}

async function runInit(config: AILensConfig) {
  const logDir = config.logDir ?? '.ailens'
  const storage = new Storage(config, 'cli')

  const defaultConfig: AILensConfig = {
    logDir,
    analysisProvider: 'anthropic',
    analysisModel: 'claude-sonnet-4-6',
    maxLogs: 1000,
    verbose: false,
  }

  storage.saveConfig(defaultConfig)

  console.log(`\n✓ Created ${logDir}/config.json`)
  console.log(`✓ Created ${logDir}/.gitignore (logs are gitignored by default)`)
  console.log(`✓ Created ${logDir}/README.md`)
  console.log()
  console.log('Next steps:')
  console.log(`  1. Set your API key: export ANTHROPIC_API_KEY=sk-...`)
  console.log(`     (or add it to .env — ailens loads it automatically)`)
  console.log()
  console.log(`  2. Wrap your LLM calls:`)
  console.log()
  console.log(`     import { lens } from '@techwarq/ailens'`)
  console.log(`     const l = lens()`)
  console.log(`     const output = await l.run(prompt, () => myModel.call(prompt))`)
  console.log()
  console.log(`  3. Run your app, then:`)
  console.log(`     npx @techwarq/ailens why`)
  console.log(`     npx @techwarq/ailens diff --last`)
}

function printInlineDiff(before: string, after: string) {
  const bLines = before.split('\n')
  const aLines = after.split('\n')
  const maxLen = Math.max(bLines.length, aLines.length)

  for (let i = 0; i < maxLen; i++) {
    const b = bLines[i]
    const a = aLines[i]

    if (b === a) {
      console.log(`  ${b ?? ''}`)
    } else {
      if (b !== undefined) console.log(`- ${b}`)
      if (a !== undefined) console.log(`+ ${a}`)
    }
  }
}

main().catch(e => {
  console.error('ailens error:', e.message)
  process.exit(1)
})
