import { test, beforeEach } from 'node:test'
import * as assert from 'node:assert/strict'
import { lens } from '../src'
import { Storage } from '../src/storage'
import { tempLogDir, clearEnv } from './helpers'

beforeEach(() => clearEnv())

function savedTrace(logDir: string, id: string) {
  return new Storage({ logDir }, 'cli').readTrace(id)!
}

test('t.run(name, prompt, fn) records the prompt as the step input', async () => {
  const logDir = tempLogDir()
  const l = lens({ logDir })
  let traceId = ''
  await l.trace('pipe', async t => {
    traceId = t.getTraceId()
    await t.run('with-prompt', 'Summarize: hello', async () => 'hi')
    await t.run('legacy-form', async () => 'still works')
  })
  const trace = savedTrace(logDir, traceId)
  assert.equal(trace.steps[0].input, 'Summarize: hello')
  assert.equal(trace.steps[1].output, 'still works')
  assert.deepEqual(trace.steps[1].dependsOn, ['with-prompt'])
})

test('finalOutput is saved when the pipeline returns', async () => {
  const logDir = tempLogDir()
  const l = lens({ logDir })
  let traceId = ''
  const result = await l.trace('pipe', async t => {
    traceId = t.getTraceId()
    await t.run('a', 'p', async () => 'x')
    return { answer: 42 }
  })
  assert.deepEqual(result, { answer: 42 })
  const trace = savedTrace(logDir, traceId)
  assert.equal(trace.success, true)
  assert.equal(trace.finalOutput, '{"answer":42}')
})

test('an error thrown outside any step marks the trace failed', async () => {
  const logDir = tempLogDir()
  const l = lens({ logDir })
  let traceId = ''
  await assert.rejects(l.trace('pipe', async t => {
    traceId = t.getTraceId()
    await t.run('a', 'p', async () => 'x')
    throw new Error('post-processing failed')
  }), /post-processing failed/)
  const trace = savedTrace(logDir, traceId)
  assert.equal(trace.success, false)
  assert.equal(trace.error, 'post-processing failed')
})

test('repeated step names (agent loops) are all kept; stepFeedback marks the latest', async () => {
  const logDir = tempLogDir()
  const l = lens({ logDir })
  let traceId = ''
  await l.trace('agent', async t => {
    traceId = t.getTraceId()
    for (let i = 0; i < 3; i++) await t.run('think', `step ${i}`, async () => `thought ${i}`)
    t.stepFeedback('think', 'bad')
  })
  const trace = savedTrace(logDir, traceId)
  assert.equal(trace.steps.length, 3)
  assert.deepEqual(trace.steps.map(s => s.feedback), [undefined, undefined, 'bad'])
})

test('step checks use local rules and throw on failure', async () => {
  const l = lens({ logDir: tempLogDir() })
  await assert.rejects(
    l.trace('pipe', t => t.run('a', 'p', async () => 'one two three', { check: ['under 2 words'] })),
    /failed checks/
  )
})
