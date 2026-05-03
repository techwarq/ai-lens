import * as fs from 'fs'
import * as path from 'path'
import { LensCall, AILensConfig } from '../types'

export class Storage {
  private logDir: string
  private sessionFile: string
  private sessionId: string

  constructor(config: AILensConfig, sessionId: string) {
    this.logDir = path.resolve(config.logDir ?? '.ailens')
    this.sessionId = sessionId
    this.sessionFile = path.join(this.logDir, 'sessions', `${sessionId}.jsonl`)
    this.ensureDir()
  }

  private ensureDir(): void {
    const sessionsDir = path.join(this.logDir, 'sessions')
    if (!fs.existsSync(sessionsDir)) {
      fs.mkdirSync(sessionsDir, { recursive: true })
    }

    // Write .gitignore to keep logs out of git by default
    const gitignore = path.join(this.logDir, '.gitignore')
    if (!fs.existsSync(gitignore)) {
      fs.writeFileSync(gitignore, '# ailens logs — tracked per-machine\nsessions/\n')
    }

    // Write a README so devs know what this folder is
    const readme = path.join(this.logDir, 'README.md')
    if (!fs.existsSync(readme)) {
      fs.writeFileSync(readme, [
        '# .ailens/',
        '',
        'This folder is created by [ailens](https://github.com/ailens/ailens).',
        '',
        'It stores local logs of your AI calls for debugging and diffing.',
        'Logs are gitignored by default — they live on your machine only.',
        '',
        '## Structure',
        '- `sessions/` — one JSONL file per session',
        '- `config.json` — your ailens config (commit this)',
        '',
        'Run `npx ailens why` to debug recent calls.',
        'Run `npx ailens diff` to compare prompt versions.',
      ].join('\n'))
    }
  }

  append(call: LensCall): void {
    const line = JSON.stringify(call) + '\n'
    fs.appendFileSync(this.sessionFile, line, 'utf-8')
  }

  readSession(sessionId?: string): LensCall[] {
    const sid = sessionId ?? this.sessionId
    const file = this.resolveSessionFile(sid)
    if (!file) return []
    return fs.readFileSync(file, 'utf-8')
      .split('\n')
      .filter(Boolean)
      .map(l => JSON.parse(l) as LensCall)
  }

  private resolveSessionFile(sid: string): string | null {
    const exact = path.join(this.logDir, 'sessions', `${sid}.jsonl`)
    if (fs.existsSync(exact)) return exact
    const dir = path.join(this.logDir, 'sessions')
    if (!fs.existsSync(dir)) return null
    const match = fs.readdirSync(dir).find(f => f.startsWith(sid) && f.endsWith('.jsonl'))
    return match ? path.join(dir, match) : null
  }

  listSessions(): string[] {
    const sessionsDir = path.join(this.logDir, 'sessions')
    if (!fs.existsSync(sessionsDir)) return []
    return fs.readdirSync(sessionsDir)
      .filter(f => f.endsWith('.jsonl'))
      .map(f => f.replace('.jsonl', ''))
      .sort()
      .reverse()
  }

  readAllRecent(limit = 100): LensCall[] {
    const sessions = this.listSessions().slice(0, 10)
    const calls: LensCall[] = []
    for (const sid of sessions) {
      calls.push(...this.readSession(sid))
    }
    return calls
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, limit)
  }

  updateCall(id: string, updates: Partial<LensCall>): void {
    const calls = this.readSession()
    const updated = calls.map(c => c.id === id ? { ...c, ...updates } : c)
    fs.writeFileSync(
      this.sessionFile,
      updated.map(c => JSON.stringify(c)).join('\n') + '\n',
      'utf-8'
    )
  }

  saveConfig(config: AILensConfig): void {
    const configFile = path.join(this.logDir, 'config.json')
    fs.writeFileSync(configFile, JSON.stringify(config, null, 2), 'utf-8')
  }

  loadConfig(): Partial<AILensConfig> {
    const configFile = path.join(this.logDir, 'config.json')
    if (!fs.existsSync(configFile)) return {}
    return JSON.parse(fs.readFileSync(configFile, 'utf-8'))
  }

  getLogDir(): string {
    return this.logDir
  }

  getCurrentSessionId(): string {
    return this.sessionId
  }

  saveTrace(trace: import('../types').Trace): void {
    const tracesDir = path.join(this.logDir, 'traces')
    if (!fs.existsSync(tracesDir)) fs.mkdirSync(tracesDir, { recursive: true })
    fs.writeFileSync(
      path.join(tracesDir, `${trace.id}.json`),
      JSON.stringify(trace, null, 2), 'utf-8'
    )
  }

  readTrace(traceId: string): import('../types').Trace | null {
    const file = this.resolveTraceFile(traceId)
    if (!file) return null
    try { return JSON.parse(fs.readFileSync(file, 'utf-8')) } catch { return null }
  }

  private resolveTraceFile(traceId: string): string | null {
    const exact = path.join(this.logDir, 'traces', `${traceId}.json`)
    if (fs.existsSync(exact)) return exact
    const dir = path.join(this.logDir, 'traces')
    if (!fs.existsSync(dir)) return null
    const match = fs.readdirSync(dir).find(f => f.startsWith(traceId) && f.endsWith('.json'))
    return match ? path.join(dir, match) : null
  }

  readRecentTraces(limit = 20): import('../types').Trace[] {
    const tracesDir = path.join(this.logDir, 'traces')
    if (!fs.existsSync(tracesDir)) return []
    return fs.readdirSync(tracesDir)
      .filter(f => f.endsWith('.json'))
      .map(f => { try { return JSON.parse(fs.readFileSync(path.join(tracesDir, f), 'utf-8')) as import('../types').Trace } catch { return null } })
      .filter((t): t is import('../types').Trace => t !== null)
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, limit)
  }
}
