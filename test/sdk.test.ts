import { test, beforeEach, afterEach } from 'node:test'
import * as assert from 'node:assert/strict'
import { lens, AILensCheckError } from '../src'
import { clearStepsCache } from '../src/analyzers/geval'
import { Storage } from '../src/storage'
import {
  mockFetch, restoreFetch, anthropicReply, isStepGeneration, promptOf, tempLogDir, clearEnv,
} from './helpers'

beforeEach(() => { clearEnv(); clearStepsCache() })
afterEach(() => restoreFetch())

const judgeSays = (passed: boolean) => mockFetch(req => isStepGeneration(req)
  ? anthropicReply('{"steps":["check it"]}')
  : anthropicReply(JSON.stringify({ reasoning: 'r', score: passed ? 1 : 0, passed })))

test('run() records the app model/provider, not the analysis model', async () => {
  const l = lens({ logDir: tempLogDir(), analysisModel: 'judge-model' })
  await l.run('p', async () => 'out', { model: 'gpt-4o', provider: 'openai' })
  await l.run('p', async () => 'out')
  const [a, b] = l.getSession()
  assert.equal(a.model, 'gpt-4o')
  assert.equal(a.provider, 'openai')
  assert.equal(b.model, 'unknown')
})

test('semantic check with no API key throws instead of silently passing', async () => {
  const l = lens({ logDir: tempLogDir() })
  await assert.rejects(
    l.run('p', async () => 'out', { check: ['is polite'] }),
    (e: unknown) => {
      assert.ok(e instanceof AILensCheckError)
      assert.match(e.message, /could not evaluate/)
      assert.ok(e.checks[0].error)
      return true
    }
  )
})

test("checkErrors: 'ignore' records the error but does not throw", async () => {
  const l = lens({ logDir: tempLogDir(), checkErrors: 'ignore' })
  const out = await l.run('p', async () => 'out', { check: ['is polite', 'not empty'] })
  assert.equal(out, 'out')
  const [call] = l.getSession()
  assert.equal(call.checks![0].passed, false)
  assert.ok(call.checks![0].error)
  assert.equal(call.checks![1].passed, true)
})

test("checkErrors: 'ignore' still throws on a genuine failed check", async () => {
  const l = lens({ logDir: tempLogDir(), checkErrors: 'ignore' })
  await assert.rejects(l.run('p', async () => 'a b c', { check: ['under 2 words'] }), AILensCheckError)
})

test('checks receive the prompt and system prompt as context', async () => {
  const requests = judgeSays(true)
  const l = lens({ logDir: tempLogDir(), analysisApiKey: 'k' })
  await l.runWithSystem('You are terse', 'What is 2+2?', async () => '4', { check: ['answers the question'] })
  const scoring = promptOf(requests.find(r => !isStepGeneration(r))!)
  assert.match(scoring, /<input>\nWhat is 2\+2\?\n<\/input>/)
  assert.match(scoring, /<system_prompt>\nYou are terse\n<\/system_prompt>/)
  assert.equal(l.getSession()[0].system, 'You are terse')
})

test('runWithSystem enforces checks and logs errors like run()', async () => {
  const l = lens({ logDir: tempLogDir() })
  await assert.rejects(l.runWithSystem('s', 'p', async () => 'a b c', { check: ['under 2 words'] }), AILensCheckError)
  await assert.rejects(l.runWithSystem('s', 'p', async () => { throw new Error('boom') }), /boom/)
  const calls = l.getSession()
  assert.equal(calls.length, 2)
  assert.equal(calls[1].output, '[ERROR] boom')
})

test('runMedia and runAsync log failures instead of losing them', async () => {
  const l = lens({ logDir: tempLogDir() })
  await assert.rejects(l.runMedia('p', async () => { throw new Error('gen failed') }), /gen failed/)
  await assert.rejects(
    l.runAsync('p', async () => 'job-1', async () => null, { pollInterval: 1, timeout: 5 }),
    /Timed out/
  )
  const url = await l.runAsync('p', async () => 'job-2', async () => 'https://x/video.mp4', { pollInterval: 1 })
  assert.equal(url, 'https://x/video.mp4')

  const calls = l.getSession()
  assert.equal(calls.length, 3)
  assert.equal(calls[0].output, '[ERROR] gen failed')
  assert.match(calls[1].output, /Timed out/)
  assert.deepEqual(calls[2].meta, { jobId: 'job-2', mediaType: 'video', mediaUrl: 'https://x/video.mp4' })
})

test('feedback() works on a call logged by an earlier lens instance', async () => {
  const logDir = tempLogDir()
  const first = lens({ logDir })
  const { id } = await first.runWithId('p', async () => 'o')

  const second = lens({ logDir })
  assert.equal(second.feedback(id, 'bad'), true)
  const call = new Storage({ logDir }, 'cli').readSession(first.getSessionId())[0]
  assert.equal(call.feedback, 'bad')
})

test('passing judge lets run() return the output', async () => {
  judgeSays(true)
  const l = lens({ logDir: tempLogDir(), analysisApiKey: 'k' })
  assert.equal(await l.run('p', async () => 'hello', { check: ['is polite'] }), 'hello')
})

test('failing judge throws AILensCheckError with the judge reason', async () => {
  judgeSays(false)
  const l = lens({ logDir: tempLogDir(), analysisApiKey: 'k' })
  await assert.rejects(l.run('p', async () => 'go away', { check: ['is polite'] }), (e: unknown) => {
    assert.ok(e instanceof AILensCheckError)
    assert.equal(e.checks[0].passed, false)
    assert.equal(e.checks[0].error, undefined)
    return true
  })
})
