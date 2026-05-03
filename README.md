# ailens

> Bring real software engineering to your AI app — debug, diff, type-check, and auto-test your LLM calls.

```bash
npm install ailens
```

---

## The problem

You ship an AI feature. Something breaks. You have no idea why.

- Which prompt caused it?
- Did your last prompt change make things better or worse?
- Are outputs actually following your rules?

Normal dev tools don't understand LLM outputs. `ailens` does.

---

## Setup — 2 lines

```ts
import { lens } from 'ailens'

const l = lens() // reads config from .ailens/ and env vars

// Wrap any LLM call
const output = await l.run(prompt, () => myModel.call(prompt))
```

That's it. `ailens` silently logs every call locally to `.ailens/sessions/`.

---

## Commands

### `npx ailens why`

Diagnose recent bad outputs. Tells you *exactly* what in your prompt caused the problem.

```
🔴 Call 3f2a1b9c [high]
   2024-01-15 14:32:01
   Tag: summarizer

   📋 Prompt: "Summarize the following article in one sentence..."

   💬 Output: "This article discusses many important topics including..."

   🩺 Diagnosis:
   The prompt doesn't constrain the output format. The model
   interpreted "one sentence" loosely and produced a vague opener.

   ⚠  Prompt issues:
   • No example of what a good summary looks like
   • "One sentence" is ambiguous — no length or style constraint

   ✏  Suggested fix:
   Add: "Return exactly one sentence under 20 words. Example: 'Scientists
   discovered X by doing Y, leading to Z.'"
```

### `npx ailens diff --last`

Compare the last two sessions. Shows what *behavior* changed, not just what text changed.

```
📊 ailens diff
   Before: 3f2a1b9c (12 calls)
   After:  7d4e2f1a (15 calls)

SUMMARY
Tone shifted more formal after the system prompt rewrite. Responses are
40% longer on average. One regression: edge cases with short inputs now
return empty strings.

  Tone          more formal
  Length        +40% longer

✅ Improvements:
   + Fewer hallucinations in product descriptions
   + More consistent JSON structure

❌ Regressions:
   - Short inputs (<10 words) now return empty strings
   - Occasional broken markdown in headers
```

### `npx ailens sessions`

List all recorded sessions.

```
ID              CALLS  DATE
──────────────────────────────────────────────────
7d4e2f1a        15     1/15/2024, 2:45 PM
3f2a1b9c        12     1/15/2024, 11:20 AM (2 bad)
```

---

## Semantic checks (type-check your outputs)

```ts
const output = await l.run(prompt, () => myModel.call(prompt), {
  check: [
    'under 50 words',
    'does not contain "I cannot"',
    'valid json',
    'is polite and professional in tone',   // LLM judge
    'does not mention competitor products', // LLM judge
  ]
})
// throws AILensCheckError if any check fails
```

Fast local checks run instantly (length, regex, JSON). Semantic checks use a small LLM judge.

---

## Mark good/bad outputs

```ts
const { output, id } = await l.runWithId(prompt, () => myModel.call(prompt))

if (userThumbsUp) l.feedback(id, 'good')
if (userThumbsDown) l.feedback(id, 'bad')
```

`ailens why` focuses on bad-marked outputs first.

---

## Config

```bash
npx ailens init
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

Or via env vars:

```bash
AILENS_API_KEY=sk-...
AILENS_PROVIDER=anthropic   # or openai
AILENS_MODEL=claude-sonnet-4-6
AILENS_LOG_DIR=.ailens
```

---

## Works with any model

```ts
// OpenAI
const output = await l.run(prompt, () =>
  openai.chat.completions.create({ model: 'gpt-4o', messages: [...] })
    .then(r => r.choices[0].message.content!)
)

// Anthropic
const output = await l.run(prompt, () =>
  anthropic.messages.create({ model: 'claude-sonnet-4-6', ... })
    .then(r => r.content[0].text)
)

// Any other model — just wrap it
const output = await l.run(prompt, () => myCustomModel.generate(prompt))
```

---

## Logs stay local

All logs go to `.ailens/sessions/` on your machine. The `.gitignore` inside `.ailens/` keeps session logs out of git by default. Your prompts and outputs never leave your machine unless you use `why`/`diff` analysis (which calls your configured analysis model).

---

## Roadmap

- `ailens test` — auto-generate regression tests from good outputs
- `ailens check` — enforce semantic rules across a whole session
- HTML report output
- CI mode (`--ci` flag, exits non-zero on regressions)
- VS Code extension

---

## License

MIT
