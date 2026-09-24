import { test } from 'node:test'
import * as assert from 'node:assert/strict'
import * as fs from 'fs'
import * as path from 'path'
import { Storage } from '../src/storage'
import { LensCall } from '../src/types'
import { tempLogDir } from './helpers'

function call(sessionId: string, id: string, timestamp: number): LensCall {
  return { id, timestamp, prompt: 'p', output: 'o', model: 'm', provider: 'x', latencyMs: 1, sessionId }
}

test('listSessions is ordered by session start time, not by random UUID', () => {
  const logDir = tempLogDir()
  // IDs deliberately sort the opposite way to their start times
  const ids = ['aaaa', 'bbbb', 'cccc', 'dddd']
  ids.forEach((sid, i) => new Storage({ logDir }, sid).append(call(sid, `c-${sid}`, 1000 * (ids.length - i))))
  const storage = new Storage({ logDir }, 'cli')
  assert.deepEqual(storage.listSessions(), ['aaaa', 'bbbb', 'cccc', 'dddd'])
})

test('updateCall finds calls in earlier sessions and reports misses', () => {
  const logDir = tempLogDir()
  const old = new Storage({ logDir }, 'old-session')
  old.append(call('old-session', 'call-1', 1))
  old.append(call('old-session', 'call-2', 2))

  const current = new Storage({ logDir }, 'new-session')
  assert.equal(current.updateCall('call-2', { feedback: 'bad' }), true)
  assert.equal(current.updateCall('nope', { feedback: 'bad' }), false)

  const calls = current.readSession('old-session')
  assert.equal(calls.find(c => c.id === 'call-2')!.feedback, 'bad')
  assert.equal(calls.find(c => c.id === 'call-1')!.feedback, undefined)
  assert.ok(!fs.readdirSync(path.join(logDir, 'sessions')).some(f => f.endsWith('.tmp')))
})

test('a corrupt line does not make the whole session unreadable', () => {
  const logDir = tempLogDir()
  const s = new Storage({ logDir }, 'sess')
  s.append(call('sess', 'ok-1', 1))
  fs.appendFileSync(path.join(logDir, 'sessions', 'sess.jsonl'), '{"id":"half\n')
  s.append(call('sess', 'ok-2', 2))
  assert.deepEqual(s.readSession().map(c => c.id), ['ok-1', 'ok-2'])
})

test('readAllRecent returns newest calls across sessions', () => {
  const logDir = tempLogDir()
  new Storage({ logDir }, 'zzz').append(call('zzz', 'old', 1))
  new Storage({ logDir }, 'aaa').append(call('aaa', 'new', 2))
  assert.deepEqual(new Storage({ logDir }, 'cli').readAllRecent(1).map(c => c.id), ['new'])
})
