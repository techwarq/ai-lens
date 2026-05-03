# @techwarq/ailens

> Bring real software engineering to your AI app — debug, diff, type-check, and trace your LLM calls.

```bash
npm install @techwarq/ailens
```

Works with **any LLM** — OpenAI, Anthropic, Gemini, Groq, Ollama, Mistral, or any custom model.

---

## The problem

You ship an AI feature. Something breaks. You have no idea why.

- Which prompt caused it?
- Did your last prompt change make things better or worse?
- Are outputs actually following your rules?

Normal dev tools don't understand LLM outputs. `ailens` does.

---

## Quick start — 2 lines

```ts
import { lens } from '@techwarq/ailens'

const l = lens()

// Wrap any LLM call — works with OpenAI, Anthropic, Gemini, anything
const output = await l.run(prompt, () => myModel.call(prompt))
```

That's it. Every call is silently logged locally to `.ailens/sessions/`.

---

## SDK

### `l.run(prompt, fn, options?)`

Wrap any LLM call. Logs it and returns the output.

```ts
const output = await l.run(
  'Summarize this article in one sentence.',
  () => openai.chat.completions.create({ ... }).then(r => r.choices[0].message.content!),
  { tag: 'summarizer' }
)
```

### `l.runWithId(prompt, fn, options?)`

Same as `run()` but returns `{ output, id }` so you can attach feedback.

```ts
const { output, id } = await l.runWithId(
  'Summarize this article.',
  () => myModel.call(prompt)
)

// Later — when user gives feedback:
if (userThumbsUp)   l.feedback(id, 'good')
if (userThumbsDown) l.feedback(id, 'bad')
```

`ailens why` will prioritize bad-marked calls when diagnosing problems.

### `l.run()` with semantic checks

Pass plain-english rules to enforce on every output. Fast local rules run instantly with no API call. Semantic rules use an LLM judge.

```ts
const output = await l.run(prompt, () => myModel.call(prompt), {
  check: [
    'under 50 words',                        // ⚡ local — instant
    'not empty',                             // ⚡ local — instant
    'valid json',                            // ⚡ local — instant
    'does not contain "I cannot"',           // ⚡ local — instant
    'is polite and professional in tone',    // 🤖 LLM judge
    'does not mention competitor products',  // 🤖 LLM judge
  ]
})
// throws AILensCheckError if any check fails
```

**Local checks** (no API needed):
- `under N words` / `over N words`
- `under N chars` / `over N chars`
- `not empty`
- `valid json`
- `contains "text"`
- `does not contain "text"`
- `starts with "text"`
- `ends with "text"`
- `no urls`

**Semantic checks** (LLM judge, uses your configured analysis model):
- Any plain-english rule that doesn't match the above patterns

Catching a failed check:

```ts
import { AILensCheckError } from '@techwarq/ailens'

try {
  const output = await l.run(prompt, fn, { check: ['under 50 words'] })
} catch (e) {
  if (e instanceof AILensCheckError) {
    console.log(e.checks) // array of { rule, passed, reason }
  }
}
```

### `l.trace()` — multi-step pipelines

Trace full agent pipelines, RAG chains, or image/video workflows. Every step is logged individually and linked by trace ID.

```ts
const result = await l.trace('my-pipeline', async (t) => {
  // LLM step
  const refined = await t.run('refine-prompt', () => llm.refine(input))

  // Image generation
  const imageUrl = await t.image('gen-image', refined, () => dalle.generate(refined))

  // Async video generation (submit + poll)
  const videoUrl = await t.runAsync(
    'gen-video',
    refined,
    () => runway.submit(imageUrl),        // returns job ID
    (jobId) => runway.poll(jobId)         // returns URL or null if not ready
  )

  // Tool call
  const data = await t.tool('fetch-data', { url }, () => fetch(url).then(r => r.json()))

  // RAG retrieval
  const context = await t.retrieve('search', query, () => vectorDB.search(query))

  return videoUrl
})
```

Each step type:

| Method | Use for |
|---|---|
| `t.run(name, fn)` | LLM text generation |
| `t.image(name, prompt, fn)` | Image generation (DALL-E, Stability, etc.) |
| `t.video(name, prompt, fn)` | Synchronous video gen |
| `t.runAsync(name, prompt, submitFn, pollFn)` | Async video gen (Runway, Sora, Kling) |
| `t.tool(name, args, fn)` | Tool / function calls |
| `t.retrieve(name, query, fn)` | RAG / vector search |

---

## CLI commands

Set up first:
```bash
npx @techwarq/ailens init
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY, GROQ_API_KEY, etc.
```

---

### `npx @techwarq/ailens why`

Diagnoses recent bad outputs using causal chain analysis — pinpoints exactly which part of your prompt caused the problem.

```
🔍 ailens why — analyzing 12 calls...

🔴 Call 804af8e8 [high]
   1/15/2025, 2:32:01 PM
   Tag: summarizer

   📋 Prompt (first 120 chars):
   "Summarize the following article in one sentence..."

   💬 Output (first 120 chars):
   "This article discusses many important topics including climate change, economic policy..."

   🩺 Diagnosis:
   The prompt doesn't constrain the output format. The model interpreted
   "one sentence" loosely and produced a vague opener instead of a real summary.
   Without a concrete length or style constraint, the model defaults to the
   path of least resistance.

   ⚠  Prompt issues:
   • "One sentence" is ambiguous — no word limit or style constraint
   • No example of what a good summary looks like
   • Missing format instruction (start with subject, not "This article...")

   ✏  Suggested fix:
   Add: "Return exactly one sentence under 20 words starting with the
   main subject. Example: 'Scientists discovered X by doing Y, leading to Z.'"

   🔗 Root cause span (82% confidence):
   suspicion score: 85%
   "Summarize the following article in one sentence."

────────────────────────────────────────────────────────────
```

Severity levels:
- `🔴 [high]` — clear prompt issue with high confidence fix
- `🟡 [medium]` — suspicious but ambiguous
- `🟢 [low]` — minor issue, may be acceptable

Flags:
```bash
npx @techwarq/ailens why                     # analyze last 50 calls
npx @techwarq/ailens why --session <id>      # analyze a specific session
npx @techwarq/ailens why --tag summarizer    # only calls with this tag
```

---

### `npx @techwarq/ailens diff --last`

Compares the last two sessions and explains what **behavior** changed — not just what text changed.

```
📊 ailens diff
   Before: 3f2a1b9c (12 calls)
   After:  7d4e2f1a (15 calls)

Analyzing behavior changes...

────────────────────────────────────────────────────────────
SUMMARY
────────────────────────────────────────────────────────────
Tone shifted more formal after the system prompt rewrite.
Responses are 40% longer on average. One regression: edge
cases with short inputs now return empty strings instead of
a fallback message.

  Tone             more formal
  Length           +40% longer
  Semantic drift   🟡 MODERATE (18.3% behavioral change)
  Cosine sim       0.817 (1.0 = identical outputs)

~ Behavioral slices:
   ↑ "direct answer": 4 → 9 (60% drift)
   ↓ "hedged response": 6 → 2 (40% drift)

✅ Improvements:
   + Fewer hallucinations in product descriptions
   + More consistent JSON structure

❌ Regressions:
   - Short inputs (<10 words) now return empty strings
   - Occasional broken markdown in headers

~ Behavior changes:
   ~ Responses now begin with a direct answer instead of context-setting
   ~ Refusal rate dropped from 8% to 2%

────────────────────────────────────────────────────────────
PROMPT DIFF
────────────────────────────────────────────────────────────
- You are a helpful assistant. Be concise.
+ You are a professional assistant. Be formal and thorough.
```

Flags:
```bash
npx @techwarq/ailens diff --last               # compare last two sessions
npx @techwarq/ailens diff <before> <after>     # compare specific sessions by ID
```

---

### `npx @techwarq/ailens sessions`

Lists all recorded sessions.

```
Recorded sessions (5 total)

ID              CALLS  DATE
──────────────────────────────────────────────────
7d4e2f1a-cc12   15     1/15/2025, 2:45 PM
3f2a1b9c-aa09   12     1/15/2025, 11:20 AM (2 bad)
1a2b3c4d-ff01   8      1/14/2025, 4:10 PM

Run 'npx @techwarq/ailens sessions --show <id>' to inspect a session
Run 'npx @techwarq/ailens why' to diagnose issues
Run 'npx @techwarq/ailens diff --last' to compare last two sessions
```

Inspect a session:
```bash
npx @techwarq/ailens sessions --show 7d4e2f1a
```

```
Session 7d4e2f1a — 15 calls

👍 3f2a1b9c | 142ms | summarizer
   Summarize the following article in one sentence...

👎 804af8e8 | 98ms  | summarizer
   Summarize the following article in one sentence...

   defe565e | 210ms | classifier
   Classify this support ticket into one of: billing...
```

---

### `npx @techwarq/ailens traces`

Lists all recorded pipeline traces.

```
Recorded traces (3 total)

ID         NAME                      STEPS  TIME     STATUS
─────────────────────────────────────────────────────────────────
b77b3cc6   image-pipeline            4      8432ms   ✓  1/15/2025, 2:26 PM
7d1d4f23   rag-chatbot               3      1204ms   ✓  1/15/2025, 1:10 PM
0ae20da7   video-gen-pipeline        5      62000ms  ✗  1/14/2025, 4:01 PM
```

Inspect a trace:
```bash
npx @techwarq/ailens traces --show b77b3cc6
```

```
Trace: image-pipeline | 4 steps | 8432ms | ✓ success
ID: b77b3cc6-...

  ✓ refine-prompt       [llm       ]  312ms
    In:  "a cat sitting on a rooftop at sunset"
    Out: "a photorealistic tabby cat perched on a..."

  ✓ gen-image           [image-gen ]  4200ms
    In:  "a photorealistic tabby cat perched on a..."
    Out: https://cdn.openai.com/images/...
    URL: https://cdn.openai.com/images/...

  ✓ describe-image      [llm       ]  890ms
    In:  "Describe this image: https://..."
    Out: "A realistic tabby cat sits on a..."

  ✓ format-result       [llm       ]  180ms
    In:  "Format this as JSON: ..."
    Out: {"description": "...", "tags": [...]}
```

Diagnose a failed trace:
```bash
npx @techwarq/ailens traces why 0ae20da7
```

```
🔍 ailens traces why — analyzing "video-gen-pipeline"...

🔴 [high] video-gen-pipeline

Step breakdown:
  ✓ ok               refine-prompt [llm]
  🟡 suspicious      gen-image [image-gen]
  🔴 ROOT CAUSE      gen-video [video-gen]
              Timed out after 300s — likely the image input was malformed

🔗 Root cause: "gen-video" [video-gen]
   Input:  "https://broken-url.example.com/image.png"
   Output: ""

🩺 Diagnosis:
   The video generation step timed out because it received a broken image URL
   from the previous step. The gen-image step returned a temporary signed URL
   that expired before the video job consumed it. This is a cascade failure —
   the root cause is the URL lifetime mismatch, not the video generator itself.

✏  Suggested fix:
   Download the image to a stable URL or base64 before passing it to
   the video generator. Signed CDN URLs from DALL-E expire after ~1 hour.
```

---

## Works with any LLM

Your own LLM calls are just wrapped functions — use any SDK, any model:

```ts
// OpenAI
const output = await l.run(prompt, () =>
  openai.chat.completions.create({ model: 'gpt-4o', messages: [{ role: 'user', content: prompt }] })
    .then(r => r.choices[0].message.content!)
)

// Anthropic
const output = await l.run(prompt, () =>
  anthropic.messages.create({ model: 'claude-opus-4-7', messages: [{ role: 'user', content: prompt }] })
    .then(r => r.content[0].text)
)

// Gemini
const output = await l.run(prompt, () =>
  genai.models.generateContent({ model: 'gemini-2.0-flash', contents: prompt })
    .then(r => r.text())
)

// Any other model — just return a string
const output = await l.run(prompt, () => myCustomModel.generate(prompt))
```

---

## Analysis model — any provider

The analysis model (used by `why`, `diff`, checks) is separate from your app's LLM. Configure it to use any provider:

```ts
// Anthropic (default)
const l = lens({ analysisProvider: 'anthropic', analysisApiKey: process.env.ANTHROPIC_API_KEY })

// OpenAI
const l = lens({ analysisProvider: 'openai', analysisApiKey: process.env.OPENAI_API_KEY })

// Groq (fast + cheap)
const l = lens({
  analysisProvider: 'openai-compatible',
  analysisBaseURL: 'https://api.groq.com/openai/v1',
  analysisApiKey: process.env.GROQ_API_KEY,
  analysisModel: 'llama-3.3-70b-versatile',
})

// Gemini
const l = lens({
  analysisProvider: 'openai-compatible',
  analysisBaseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
  analysisApiKey: process.env.GOOGLE_API_KEY,
  analysisModel: 'gemini-2.0-flash',
})

// Ollama (local, free)
const l = lens({
  analysisProvider: 'openai-compatible',
  analysisBaseURL: 'http://localhost:11434/v1',
  analysisApiKey: 'ollama',
  analysisModel: 'llama3.2',
})
```

Or via environment variables:
```bash
# Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
AILENS_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Any OpenAI-compatible provider
AILENS_PROVIDER=openai-compatible
AILENS_BASE_URL=https://api.groq.com/openai/v1
AILENS_API_KEY=gsk_...
AILENS_MODEL=llama-3.3-70b-versatile
```

---

## Config file

```bash
npx @techwarq/ailens init
```

Creates `.ailens/config.json`:

```json
{
  "logDir": ".ailens",
  "analysisProvider": "anthropic",
  "analysisModel": "claude-sonnet-4-6",
  "maxLogs": 1000,
  "verbose": false
}
```

Config priority: **code config > env vars > config.json**

---

## Logs stay local

All logs go to `.ailens/sessions/` on your machine. The `.gitignore` inside `.ailens/` keeps session logs out of git by default.

Your prompts and outputs **never leave your machine** unless you explicitly run `why`, `diff`, or use semantic checks — those send data to your configured analysis model.

```
.ailens/
  config.json        ← commit this
  .gitignore         ← keeps sessions/ out of git
  sessions/          ← one file per session, gitignored
  traces/            ← one file per trace, gitignored
```

---

## Full API reference

```ts
import { lens, AILens, AILensCheckError } from '@techwarq/ailens'
import type {
  AILensConfig, LensCall, CheckResult,
  RunOptions, DiffResult, WhyResult,
  Trace, TraceStep, TraceWhyResult,
} from '@techwarq/ailens'

const l = lens(config?)               // create a lens instance
l.run(prompt, fn, options?)           // wrap an LLM call
l.runWithId(prompt, fn, options?)     // same, returns { output, id }
l.runWithSystem(system, prompt, fn)   // track system prompt separately
l.feedback(id, 'good' | 'bad')        // mark a call
l.trace(name, fn)                     // run a multi-step pipeline
l.runMedia(prompt, fn, options?)      // single media generation call
l.runAsync(prompt, submitFn, pollFn)  // async media generation (polling)
l.getSession()                        // get all calls in current session
l.getSessionId()                      // get current session ID
```

`RunOptions`:
```ts
{
  tag?: string                  // group calls by tag for filtering
  meta?: Record<string, any>    // arbitrary metadata
  check?: string[]              // semantic rules to enforce
  feedback?: 'good' | 'bad'     // mark immediately on creation
}
```

---

## Roadmap

- `ailens test` — auto-generate regression tests from good outputs
- HTML report output
- CI mode (`--ci` flag, exits non-zero on regressions)
- VS Code extension

---

## License

MIT — made by [Sonali Nayak](https://github.com/techwarq)
