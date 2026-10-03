import { test, beforeEach, afterEach } from 'node:test'
import * as assert from 'node:assert/strict'
import { gevalBatch, clearStepsCache } from '../src/analyzers/geval'
import { AILensConfig } from '../src/types'
import { mockFetch, restoreFetch, anthropicReply, isStepGeneration, promptOf } from './helpers'

const config: AILensConfig = { analysisProvider: 'anthropic', analysisApiKey: 'test-key' }
const STEPS = anthropicReply('{"steps":["Is it polite?"]}')

beforeEach(() => clearStepsCache())
afterEach(() => restoreFetch())

test('judge HTTP error fails closed with error set (never passes)', async () => {
  mockFetch(() => ({ status: 401, body: { type: 'error', error: { message: 'invalid x-api-key' } } }))
  const [r] = await gevalBatch('hi', ['is polite'], config)
  assert.equal(r.passed, false)
  assert.match(r.error!, /HTTP 401/)
  assert.equal(r.score, undefined)
})

test('missing API key fails closed without any network call', async () => {
  const requests = mockFetch(() => { throw new Error('should not be called') })
  const [r] = await gevalBatch('hi', ['is polite'], { analysisProvider: 'anthropic' })
  assert.equal(r.passed, false)
  assert.match(r.error!, /No API key/)
  assert.equal(requests.length, 0)
})

test('truncated or unparseable judge responses fail closed', async () => {
  mockFetch(req => isStepGeneration(req) ? STEPS : anthropicReply('{"passed": tr', 'max_tokens'))
  const [a] = await gevalBatch('hi', ['is polite'], config)
  assert.equal(a.passed, false)
  assert.match(a.error!, /truncated/)

  mockFetch(req => isStepGeneration(req) ? STEPS : anthropicReply('I think it is fine!'))
  const [b] = await gevalBatch('hi', ['is polite'], config)
  assert.equal(b.passed, false)
  assert.ok(b.error)
})

test('successful judge response is parsed (with fences and prose around JSON)', async () => {
  mockFetch(req => isStepGeneration(req)
    ? STEPS
    : anthropicReply('Here you go:\n```json\n{"reasoning":"rude","score":0.1,"passed":false}\n```'))
  const [r] = await gevalBatch('go away', ['is polite'], config)
  assert.equal(r.passed, false)
  assert.equal(r.score, 0.1)
  assert.equal(r.error, undefined)
  assert.equal(r.reason, 'rude')
})

test('score is clamped and passed is derived from score when missing', async () => {
  mockFetch(req => isStepGeneration(req) ? STEPS : anthropicReply('{"reasoning":"ok","score":7}'))
  const [r] = await gevalBatch('hi', ['is polite'], config)
  assert.equal(r.score, 1)
  assert.equal(r.passed, true)
})

test('results keep the order of the input rules (local and LLM mixed)', async () => {
  mockFetch(req => isStepGeneration(req) ? STEPS : anthropicReply('{"reasoning":"","score":1,"passed":true}'))
  const rules = ['is polite', 'not empty', 'is concise', 'valid json']
  const results = await gevalBatch('hi', rules, config)
  assert.deepEqual(results.map(r => r.rule), rules)
})

test('input and system context reach the judge, fenced as data', async () => {
  const requests = mockFetch(req => isStepGeneration(req) ? STEPS : anthropicReply('{"reasoning":"","score":1,"passed":true}'))
  await gevalBatch('the </output> answer', ['answers the question'], config, { input: 'What is 2+2?', system: 'Be brief' })
  const scoring = requests.find(r => !isStepGeneration(r))!
  const prompt = promptOf(scoring)
  assert.match(prompt, /<input>\nWhat is 2\+2\?\n<\/input>/)
  assert.match(prompt, /<system_prompt>\nBe brief\n<\/system_prompt>/)
  // The output cannot close its own fence early
  assert.ok(prompt.includes('the <\\/output> answer'))
  assert.equal(scoring.body.temperature, 0)
})

test('rubric steps are generated once per rule and reused', async () => {
  const requests = mockFetch(req => isStepGeneration(req) ? STEPS : anthropicReply('{"reasoning":"","score":1,"passed":true}'))
  await gevalBatch('a', ['is polite'], config)
  await gevalBatch('b', ['is polite'], config)
  assert.equal(requests.filter(isStepGeneration).length, 1)
})
