import { test } from 'node:test'
import * as assert from 'node:assert/strict'
import { tryLocalCheck } from '../src/analyzers/geval'

test('negated contains is not read as contains (README example)', () => {
  assert.equal(tryLocalCheck('Sure, here you go', 'does not contain "I cannot"'), true)
  assert.equal(tryLocalCheck('Sorry, I cannot help', 'does not contain "I cannot"'), false)
  assert.equal(tryLocalCheck('Sorry, I cannot help', 'must not contain "I cannot"'), false)
})

test('contains / starts with / ends with are case-insensitive', () => {
  assert.equal(tryLocalCheck('Hello World', 'contains "world"'), true)
  assert.equal(tryLocalCheck('Hello World', 'must contain "missing"'), false)
  assert.equal(tryLocalCheck('  Hello World', 'starts with "hello"'), true)
  assert.equal(tryLocalCheck('Hello World', 'ends with "WORLD"'), true)
  assert.equal(tryLocalCheck('Hello World', 'does not start with "hello"'), false)
})

test('word and char limits', () => {
  const five = 'one two three four five'
  assert.equal(tryLocalCheck(five, 'under 6 words'), true)
  assert.equal(tryLocalCheck(five, 'under 5 words'), false)
  assert.equal(tryLocalCheck(five, 'at most 5 words'), true)
  assert.equal(tryLocalCheck(five, 'over 4 words'), true)
  assert.equal(tryLocalCheck(five, 'at least 6 words'), false)
  assert.equal(tryLocalCheck('abc', 'under 4 chars'), true)
  assert.equal(tryLocalCheck('abc', 'over 3 characters'), false)
  assert.equal(tryLocalCheck(five, 'Must be under 50 words.'), true)
  assert.equal(tryLocalCheck(five, 'the output should be fewer than 3 words'), false)
})

test('compound rules go to the LLM judge instead of being half-matched', () => {
  assert.equal(tryLocalCheck('x', 'keep it under 50 words unless the user asks for detail'), null)
  assert.equal(tryLocalCheck('x', 'cites sources with no url shorteners'), null)
  assert.equal(tryLocalCheck('x', 'contains "a" or "b"'), null)
  assert.equal(tryLocalCheck('x', 'is polite and professional in tone'), null)
})

test('not empty, valid json, no urls', () => {
  assert.equal(tryLocalCheck('  ', 'not empty'), false)
  assert.equal(tryLocalCheck('x', 'should not be empty'), true)
  assert.equal(tryLocalCheck('{"a":1}', 'valid json'), true)
  assert.equal(tryLocalCheck(' {"a":1}\n', 'must be valid JSON'), true)
  assert.equal(tryLocalCheck('{a:1}', 'is valid json'), false)
  assert.equal(tryLocalCheck('see https://x.com', 'no urls'), false)
  assert.equal(tryLocalCheck('plain text', 'does not contain urls'), true)
  assert.equal(tryLocalCheck('see HTTPS://X.COM', 'without urls'), false)
})
