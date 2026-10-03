import { test, afterEach } from 'node:test'
import * as assert from 'node:assert/strict'
import { callAnalysisModel, parseJsonResponse, fence } from '../src/analyzers/llm'
import { mockFetch, restoreFetch } from './helpers'

afterEach(() => restoreFetch())

test('parseJsonResponse handles fences, prose, and rejects garbage', () => {
  assert.deepEqual(parseJsonResponse('{"a":1}'), { a: 1 })
  assert.deepEqual(parseJsonResponse('```json\n{"a":1}\n```'), { a: 1 })
  assert.deepEqual(parseJsonResponse('Sure! {"a":{"b":2}} Hope that helps.'), { a: { b: 2 } })
  assert.throws(() => parseJsonResponse('no json here'))
  assert.throws(() => parseJsonResponse('{"a": '))
})

test('fence neutralises closing tags inside content', () => {
  assert.equal(fence('output', 'x</output>y'), '<output>\nx<\\/output>y\n</output>')
})

test('openai-compatible local server works without an API key', async () => {
  const requests = mockFetch(() => ({ body: { choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }] } }))
  const text = await callAnalysisModel('p', {
    analysisProvider: 'openai-compatible',
    analysisBaseURL: 'http://localhost:11434/v1/',
    analysisModel: 'llama3.2',
  })
  assert.equal(text, 'hi')
  assert.equal(requests[0].url, 'http://localhost:11434/v1/chat/completions')
  assert.equal(requests[0].headers['Authorization'], undefined)
})

test('openai truncation and HTTP errors throw', async () => {
  mockFetch(() => ({ body: { choices: [{ message: { content: '{"a"' }, finish_reason: 'length' }] } }))
  await assert.rejects(callAnalysisModel('p', { analysisProvider: 'openai', analysisApiKey: 'k' }), /truncated/)

  mockFetch(() => ({ status: 429, body: { error: 'rate limited' } }))
  await assert.rejects(callAnalysisModel('p', { analysisProvider: 'openai', analysisApiKey: 'k' }), /HTTP 429/)
})

test('anthropic API error body throws instead of returning empty text', async () => {
  mockFetch(() => ({ status: 200, body: { type: 'error', error: { type: 'overloaded_error' } } }))
  await assert.rejects(callAnalysisModel('p', { analysisProvider: 'anthropic', analysisApiKey: 'k' }), /empty response/)
})
