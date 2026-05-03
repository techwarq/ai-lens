import { LensCall, DiffResult, AILensConfig } from '../types'
import { callAnalysisModel } from './why'

/**
 * Embedding-based semantic diff for aidiff.
 *
 * Research finding (PBSS framework, 2025): surface-level text comparison
 * is unreliable for detecting behavioral drift. The right approach is to:
 *   1. Embed output sets into vector space
 *   2. Measure cosine similarity / centroid distance
 *   3. THEN use LLM to explain what that distance means
 *
 * This separates measurement (objective) from interpretation (subjective).
 *
 * Also: slice-level metrics — compare not just overall but by topic/task
 * category to see where drift concentrates.
 */

interface EmbeddingVector {
  values: number[]
  text: string
}

interface SemanticDistance {
  cosineSimilarity: number    // 0-1, higher = more similar
  driftScore: number          // 0-1, higher = more behavioral change
  centroidDistance: number    // euclidean distance between cluster centroids
  slices: SliceDrift[]        // per-category drift
}

interface SliceDrift {
  category: string
  beforeCount: number
  afterCount: number
  driftScore: number
  representative: string      // example output showing the drift
}

export async function analyzeDiffWithEmbeddings(
  before: LensCall[],
  after: LensCall[],
  config: AILensConfig
): Promise<DiffResult> {
  if (before.length === 0 || after.length === 0) {
    throw new Error('Need at least one call in both before and after sets to diff')
  }

  const promptBefore = before[0].system
    ? `SYSTEM: ${before[0].system}\n\nUSER: ${before[0].prompt}`
    : before[0].prompt

  const promptAfter = after[0].system
    ? `SYSTEM: ${after[0].system}\n\nUSER: ${after[0].prompt}`
    : after[0].prompt

  // Step 1: Try to get embeddings for semantic distance measurement
  let semanticDistance: SemanticDistance | null = null

  try {
    semanticDistance = await measureSemanticDistance(before, after, config)
  } catch {
    // Fall through — LLM analysis alone is still useful
  }

  // Step 2: Classify outputs into slices for per-category drift
  const slices = semanticDistance?.slices ?? await classifyOutputSlices(before, after, config)

  // Step 3: LLM explains the measured drift
  const analysis = await explainDrift(
    before,
    after,
    promptBefore,
    promptAfter,
    semanticDistance,
    slices,
    config
  )

  return {
    promptBefore,
    promptAfter,
    calls: { before, after },
    analysis,
  }
}

async function measureSemanticDistance(
  before: LensCall[],
  after: LensCall[],
  config: AILensConfig
): Promise<SemanticDistance> {
  const beforeOutputs = before.slice(0, 10).map(c => c.output)
  const afterOutputs = after.slice(0, 10).map(c => c.output)

  // Get embeddings from the provider
  const [beforeEmbeddings, afterEmbeddings] = await Promise.all([
    embedTexts(beforeOutputs, config),
    embedTexts(afterOutputs, config),
  ])

  if (beforeEmbeddings.length === 0 || afterEmbeddings.length === 0) {
    throw new Error('Could not get embeddings')
  }

  // Compute centroids
  const beforeCentroid = computeCentroid(beforeEmbeddings.map(e => e.values))
  const afterCentroid = computeCentroid(afterEmbeddings.map(e => e.values))

  const cosineSimilarity = cosine(beforeCentroid, afterCentroid)
  const centroidDistance = euclidean(beforeCentroid, afterCentroid)
  const driftScore = 1 - cosineSimilarity

  return {
    cosineSimilarity,
    driftScore,
    centroidDistance,
    slices: [],
  }
}

async function embedTexts(
  texts: string[],
  config: AILensConfig
): Promise<EmbeddingVector[]> {
  if (!config.analysisApiKey) return []

  // Truncate texts to avoid token limits
  const truncated = texts.map(t => t.slice(0, 512))

  if (config.analysisProvider === 'openai') {
    try {
      const res = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.analysisApiKey}`,
        },
        body: JSON.stringify({
          model: 'text-embedding-3-small',
          input: truncated,
        }),
      })
      const data = await res.json() as {
        data: Array<{ embedding: number[]; index: number }>
      }
      return data.data.map((d, i) => ({ values: d.embedding, text: truncated[i] }))
    } catch {
      return []
    }
  }

  // Anthropic doesn't have a public embeddings API yet —
  // fall back to a lightweight TF-IDF-style approach
  return tfidfEmbeddings(truncated)
}

/**
 * Lightweight TF-IDF embeddings as fallback when no embedding API is available.
 * Not as good as neural embeddings but still captures vocabulary-level drift.
 */
function tfidfEmbeddings(texts: string[]): EmbeddingVector[] {
  const allWords = new Set<string>()
  const tokenized = texts.map(t =>
    t.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 2)
  )

  tokenized.forEach(words => words.forEach(w => allWords.add(w)))
  const vocab = Array.from(allWords)

  // Document frequency for IDF
  const df = new Map<string, number>()
  vocab.forEach(word => {
    const count = tokenized.filter(words => words.includes(word)).length
    df.set(word, count)
  })

  return tokenized.map((words, docIndex) => {
    const tf = new Map<string, number>()
    words.forEach(w => tf.set(w, (tf.get(w) ?? 0) + 1))

    const values = vocab.map(word => {
      const tfVal = (tf.get(word) ?? 0) / Math.max(words.length, 1)
      const idfVal = Math.log(texts.length / Math.max(df.get(word) ?? 1, 1))
      return tfVal * idfVal
    })

    // L2 normalize
    const norm = Math.sqrt(values.reduce((s, v) => s + v * v, 0))
    return {
      values: norm > 0 ? values.map(v => v / norm) : values,
      text: texts[docIndex],
    }
  })
}

async function classifyOutputSlices(
  before: LensCall[],
  after: LensCall[],
  config: AILensConfig
): Promise<SliceDrift[]> {
  const allOutputs = [
    ...before.slice(0, 5).map(c => ({ output: c.output, set: 'before' as const })),
    ...after.slice(0, 5).map(c => ({ output: c.output, set: 'after' as const })),
  ]

  if (allOutputs.length < 2) return []

  const classifyPrompt = [
    `Categorize these AI outputs into behavioral slices (2-4 categories).`,
    `Categories should reflect different types of behavior, not topics.`,
    `Examples of good categories: "direct answer", "hedged response", "refusal",`,
    `"verbose explanation", "concise reply", "error/confusion"`,
    ``,
    `Outputs:`,
    allOutputs.map((o, i) =>
      `Output ${i} [${o.set}]: "${o.output.slice(0, 150)}..."`
    ).join('\n'),
    ``,
    `Reply with JSON only (no markdown):`,
    `{`,
    `  "categories": ["category1", "category2"],`,
    `  "assignments": [{ "index": 0, "category": "direct answer" }]`,
    `}`,
  ].join('\n')

  try {
    const text = await callAnalysisModel(classifyPrompt, config)
    const parsed = JSON.parse(text.replace(/```json|```/g, '').trim()) as {
      categories: string[]
      assignments: Array<{ index: number; category: string }>
    }

    const sliceMap = new Map<string, { before: number; after: number; examples: string[] }>()

    parsed.categories.forEach(cat => {
      sliceMap.set(cat, { before: 0, after: 0, examples: [] })
    })

    parsed.assignments.forEach(({ index, category }) => {
      const item = allOutputs[index]
      const slice = sliceMap.get(category)
      if (slice && item) {
        if (item.set === 'before') slice.before++
        else slice.after++
        if (slice.examples.length < 1) slice.examples.push(item.output.slice(0, 100))
      }
    })

    return Array.from(sliceMap.entries()).map(([category, data]) => {
      const beforeRate = before.length > 0 ? data.before / before.length : 0
      const afterRate = after.length > 0 ? data.after / after.length : 0
      return {
        category,
        beforeCount: data.before,
        afterCount: data.after,
        driftScore: Math.abs(afterRate - beforeRate),
        representative: data.examples[0] ?? '',
      }
    }).filter(s => s.driftScore > 0.1) // Only show slices with meaningful drift
  } catch {
    return []
  }
}

async function explainDrift(
  before: LensCall[],
  after: LensCall[],
  promptBefore: string,
  promptAfter: string,
  distance: SemanticDistance | null,
  slices: SliceDrift[],
  config: AILensConfig
): Promise<DiffResult['analysis']> {
  const avgLenBefore = Math.round(
    before.reduce((s, c) => s + c.output.length, 0) / before.length
  )
  const avgLenAfter = Math.round(
    after.reduce((s, c) => s + c.output.length, 0) / after.length
  )
  const lenDeltaPct = Math.round(((avgLenAfter - avgLenBefore) / avgLenBefore) * 100)

  const badRateBefore = (before.filter(c => c.feedback === 'bad').length / before.length * 100).toFixed(0)
  const badRateAfter = (after.filter(c => c.feedback === 'bad').length / after.length * 100).toFixed(0)

  const semanticBlock = distance
    ? [
        `## Semantic distance measurement`,
        `Cosine similarity: ${distance.cosineSimilarity.toFixed(3)} (1.0 = identical, 0.0 = completely different)`,
        `Drift score: ${distance.driftScore.toFixed(3)} (${distance.driftScore > 0.3 ? 'HIGH' : distance.driftScore > 0.1 ? 'MODERATE' : 'LOW'} behavioral change)`,
      ].join('\n')
    : ''

  const sliceBlock = slices.length > 0
    ? [
        `## Slice-level drift`,
        slices.map(s =>
          `- "${s.category}": ${s.beforeCount} before → ${s.afterCount} after (drift: ${(s.driftScore * 100).toFixed(0)}%)`
        ).join('\n'),
      ].join('\n')
    : ''

  const explainPrompt = [
    `You are explaining behavioral drift between two versions of an AI prompt.`,
    `Your job is to explain what the MEASUREMENTS mean, not re-measure them.`,
    ``,
    `## Stats`,
    `Calls: ${before.length} before → ${after.length} after`,
    `Avg output length: ${avgLenBefore} → ${avgLenAfter} chars (${lenDeltaPct > 0 ? '+' : ''}${lenDeltaPct}%)`,
    `Bad feedback rate: ${badRateBefore}% → ${badRateAfter}%`,
    ``,
    semanticBlock,
    sliceBlock,
    ``,
    `## Prompt BEFORE (first 300 chars)`,
    `"""${promptBefore.slice(0, 300)}"""`,
    ``,
    `## Prompt AFTER (first 300 chars)`,
    `"""${promptAfter.slice(0, 300)}"""`,
    ``,
    `## Sample before output`,
    `"""${before[0]?.output.slice(0, 250) ?? 'none'}"""`,
    ``,
    `## Sample after output`,
    `"""${after[0]?.output.slice(0, 250) ?? 'none'}"""`,
    ``,
    `Explain what changed in behavior. Be specific. Cite the measurements.`,
    `Identify regressions (things that got worse) and improvements (things that got better).`,
    ``,
    `Reply with JSON only (no markdown):`,
    `{`,
    `  "toneDelta": "e.g. 'more formal', 'no change'",`,
    `  "lengthDelta": "e.g. '+40% longer'",`,
    `  "behaviorChanges": ["specific change 1"],`,
    `  "regressions": ["regression 1"],`,
    `  "improvements": ["improvement 1"],`,
    `  "summary": "2-3 sentence plain-english summary with specific numbers"`,
    `}`,
  ].filter(Boolean).join('\n')

  try {
    const text = await callAnalysisModel(explainPrompt, config)
    return JSON.parse(text.replace(/```json|```/g, '').trim()) as DiffResult['analysis']
  } catch {
    return {
      behaviorChanges: [],
      regressions: [],
      improvements: [],
      summary: 'Could not analyze diff.',
    }
  }
}

// --- Math helpers ---

function computeCentroid(vectors: number[][]): number[] {
  if (vectors.length === 0) return []
  const dim = vectors[0].length
  const centroid = new Array(dim).fill(0) as number[]
  vectors.forEach(v => v.forEach((val, i) => { centroid[i] += val }))
  return centroid.map(v => v / vectors.length)
}

function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0
  const dot = a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0)
  const normA = Math.sqrt(a.reduce((s, v) => s + v * v, 0))
  const normB = Math.sqrt(b.reduce((s, v) => s + v * v, 0))
  if (normA === 0 || normB === 0) return 0
  return Math.max(-1, Math.min(1, dot / (normA * normB)))
}

function euclidean(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0
  return Math.sqrt(a.reduce((s, v, i) => s + Math.pow(v - (b[i] ?? 0), 2), 0))
}
