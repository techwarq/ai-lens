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
        'This folder is created by [ailens](https://github.com/techwarq/ai-lens).',
        '',
        'It stores local logs of your AI calls for debugging and diffing.',
        'Logs are gitignored by default — they live on your machine only.',
        '',
        '## Structure',
        '- `sessions/` — one JSONL file per session',
        '- `config.json` — your ailens config (commit this)',
        '',
        'Run `npx @techwarq/ailens why` to debug recent calls.',
        'Run `npx @techwarq/ailens diff --last` to compare the last two sessions.',
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
      .flatMap(l => {
        // Skip a partially-written/corrupt line instead of failing the whole session
        try { return [JSON.parse(l) as LensCall] } catch { return [] }
      })
  }

  private resolveSessionFile(sid: string): string | null {
    const exact = path.join(this.logDir, 'sessions', `${sid}.jsonl`)
    if (fs.existsSync(exact)) return exact
    const dir = path.join(this.logDir, 'sessions')
    if (!fs.existsSync(dir)) return null
    const match = fs.readdirSync(dir).find(f => f.startsWith(sid) && f.endsWith('.jsonl'))
    return match ? path.join(dir, match) : null
  }

  /** Session IDs, most recently started first. */
  listSessions(): string[] {
    const sessionsDir = path.join(this.logDir, 'sessions')
    if (!fs.existsSync(sessionsDir)) return []
    // Session IDs are random UUIDs, so sort by when the session started
    // (timestamp of its first call), falling back to file mtime.
    return fs.readdirSync(sessionsDir)
      .filter(f => f.endsWith('.jsonl'))
      .map(f => {
        const file = path.join(sessionsDir, f)
        return { id: f.slice(0, -'.jsonl'.length), started: sessionStartTime(file) }
      })
      .sort((a, b) => b.started - a.started || (a.id < b.id ? 1 : -1))
      .map(s => s.id)
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

  /**
   * Update a logged call by ID. Looks in the current session first, then all others.
   * Returns false if no call with that ID exists.
   */
  updateCall(id: string, updates: Partial<LensCall>): boolean {
    const candidates = [this.sessionId, ...this.listSessions().filter(s => s !== this.sessionId)]
    for (const sid of candidates) {
      const file = path.join(this.logDir, 'sessions', `${sid}.jsonl`)
      if (!fs.existsSync(file)) continue
      const calls = this.readSession(sid)
      if (!calls.some(c => c.id === id)) continue
      const updated = calls.map(c => c.id === id ? { ...c, ...updates } : c)
      writeFileAtomic(file, updated.map(c => JSON.stringify(c)).join('\n') + '\n')
      return true
    }
    return false
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
    writeFileAtomic(path.join(tracesDir, `${trace.id}.json`), JSON.stringify(trace, null, 2))
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

/** Timestamp of the first call in a session file (reads only the first line). */
function sessionStartTime(file: string): number {
  let fd: number | undefined
  try {
    fd = fs.openSync(file, 'r')
    const chunks: Buffer[] = []
    const buf = Buffer.alloc(8192)
    let bytes: number
    while ((bytes = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
      const nl = buf.subarray(0, bytes).indexOf(10)
      if (nl >= 0) { chunks.push(Buffer.from(buf.subarray(0, nl))); break }
      chunks.push(Buffer.from(buf.subarray(0, bytes)))
    }
    const first = JSON.parse(Buffer.concat(chunks).toString('utf-8')) as { timestamp?: unknown }
    if (typeof first.timestamp === 'number') return first.timestamp
  } catch {
    // fall back to mtime
  } finally {
    if (fd !== undefined) fs.closeSync(fd)
  }
  try { return fs.statSync(file).mtimeMs } catch { return 0 }
}

/** Write via temp file + rename so readers never see a half-written file. */
function writeFileAtomic(file: string, content: string): void {
  const tmp = `${file}.${process.pid}.${Date.now()}.tmp`
  fs.writeFileSync(tmp, content, 'utf-8')
  fs.renameSync(tmp, file)
}
