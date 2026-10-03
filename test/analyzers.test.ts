import { test, afterEach } from 'node:test'
import * as assert from 'node:assert/strict'
import {
  tfidfEmbeddings, measureSemanticDistance, cosine, analyzeDiffWithEmbeddings,
} from '../src/analyzers/semantic-diff'
import { analyzeWhy } from '../src/analyzers/why'
import { LensCall } from '../src/types'
import { mockFetch, restoreFetch, anthropicReply, promptOf } from './helpers'

afterEach(() => restoreFetch())

function calls(outputs: string[], extra: Partial<LensCall> = {}): LensCall[] {
  return outputs.map((output, i) => ({
    id: `c${i}`, timestamp: i, prompt: 'p', output, model: 'm', provider: 'x',
    latencyMs: 1, sessionId: 's', ...extra,
  }))
}

// ── semantic diff ──────────────────────────────────────────────────────────

test('TF-IDF vectors share one vocabulary so they are comparable', () => {
  const vecs = tfidfEmbeddings(['the cat sat', 'a completely different sentence here'])
  assert.equal(vecs[0].values.length, vecs[1].values.length)
})

test('TF-IDF keeps weight for terms that appear in every document', () => {
  const [a] = tfidfEmbeddings(['hello world', 'hello world'])
  assert.ok(a.values.some(v => v > 0))
})

test('identical output sets have ~zero drift; disjoint sets have high drift (no API key needed)', async () => {
  const same = await measureSemanticDistance(
    calls(['refund processed today', 'order shipped quickly']),
    calls(['refund processed today', 'order shipped quickly']),
    {}
  )
  assert.ok(same.driftScore < 1e-9, `drift was ${same.driftScore}`)
  assert.equal(same.method, 'tfidf')

  const different = await measureSemanticDistance(
    calls(['refund processed today', 'order shipped quickly']),
    calls(['unable to assist with that', 'please contact support team']),
    {}
  )
  assert.ok(different.driftScore > 0.9, `drift was ${different.driftScore}`)
})

test('cosine refuses vectors of different dimensions instead of returning 0', () => {
  assert.throws(() => cosine([1, 0], [1, 0, 0]), /dimension mismatch/)
  assert.equal(cosine([1, 0], [1, 0]), 1)
})

test('diff reports measured drift + slices even though the LLM never saw them as output fields', async () => {
  mockFetch(req => {
    const prompt = promptOf(req)
    if (prompt.includes('Categorize these AI outputs')) {
      return anthropicReply(JSON.stringify({
        categories: ['answer', 'refusal'],
        assignments: [
          { index: 0, category: 'answer' }, { index: 1, category: 'answer' },
          { index: 2, category: 'refusal' }, { index: 3, category: 'refusal' },
        ],
      }))
    }
    return anthropicReply(JSON.stringify({ summary: 'More refusals.', behaviorChanges: [], regressions: ['refuses'], improvements: [] }))
  })
  const result = await analyzeDiffWithEmbeddings(
    calls(['refund processed today', 'order shipped quickly']),
    calls(['unable to assist with that', 'please contact support team']),
    { analysisApiKey: 'k' }
  )
  assert.equal(typeof result.analysis.driftScore, 'number')
  assert.equal(typeof result.analysis.cosineSimilarity, 'number')
  assert.equal(result.analysis.summary, 'More refusals.')
  const answer = result.analysis.slices!.find(s => s.category === 'answer')!
  assert.deepEqual([answer.beforeCount, answer.afterCount, answer.driftScore], [2, 0, 1])
})

test('diff survives an all-empty "before" set (no divide-by-zero) and reports LLM failures', async () => {
  mockFetch(() => ({ status: 500, body: 'oops' }))
  const result = await analyzeDiffWithEmbeddings(calls(['', '']), calls(['some text']), { analysisApiKey: 'k' })
  assert.match(result.analysis.summary, /Could not analyze diff: .*HTTP 500/)
  assert.equal(typeof result.analysis.driftScore, 'number')
})

// ── why ────────────────────────────────────────────────────────────────────

test('why does not invent problems when nothing failed (and needs no API key for that)', async () => {
  const requests = mockFetch(() => { throw new Error('should not be called') })
  const results = await analyzeWhy(calls(['fine', 'also fine'], { feedback: 'good' }), {})
  assert.deepEqual(results, [])
  assert.equal(requests.length, 0)
})

test('why requires an API key only when there is something to diagnose', async () => {
  await assert.rejects(analyzeWhy(calls(['bad'], { feedback: 'bad' }), {}), /No API key/)
})

test('why only analyzes failing calls; errored judge checks are not reported as rule failures', async () => {
  const prompts: string[] = []
  mockFetch(req => {
    prompts.push(promptOf(req))
    if (promptOf(req).includes('causal attribution')) {
      return anthropicReply('{"attributions":[{"id":"s0","suspicionScore":0.9,"forEvidence":[],"againstEvidence":[]}]}')
    }
    return anthropicReply('{"diagnosis":"d","promptIssues":[],"suggestedFix":"f","severity":"high","confidence":1.5}')
  })
  const input = [
    ...calls(['good output'], { id: 'good' }),
    ...calls(['bad output'], {
      id: 'bad',
      checks: [
        { rule: 'is polite', passed: false, reason: 'rude' },
        { rule: 'is concise', passed: false, error: 'HTTP 500' },
      ],
    }),
  ]
  const results = await analyzeWhy(input, { analysisApiKey: 'k' })
  assert.deepEqual(results.map(r => r.call.id), ['bad'])
  assert.equal(results[0].causalChain!.confidence, 1) // clamped
  assert.equal(results[0].causalChain!.rootCause!.id, 's0')
  const attribution = prompts.find(p => p.includes('causal attribution'))!
  assert.match(attribution, /is polite: rude/)
  assert.doesNotMatch(attribution, /is concise/)
})
