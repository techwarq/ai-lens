# Changelog

## Unreleased

Correctness release. Several checks and analyses returned wrong results before. The fixes below change behavior, so read **Behavior changes** before upgrading.

### Behavior changes

- **Semantic checks no longer pass when the judge fails.** Previously, if the judge failed (missing or invalid API key, HTTP error, rate limit, truncated or unparseable response), the check was reported as `passed: true`. It now returns `passed: false` with an `error` message, and `run()` throws `AILensCheckError`. To keep judge outages from blocking your app, set `lens({ checkErrors: 'ignore' })`: the error is still logged, but `run()` won't throw because of it.
- **Local rules must match the whole rule.** `does not contain "x"` used to be read as `contains "x"`, which inverted the result. Compound rules such as `under 50 words unless asked for detail` were also treated as a plain word limit. These rules now go to the LLM judge instead.
- **`LensCall.model` / `provider` record your app's model.** They previously recorded the *analysis* model. Pass `{ model, provider }` in `RunOptions`; if you don't, they are logged as `'unknown'`.
- **`runWithSystem()` now enforces `check`**, and it logs calls that throw. `runMedia()` and `runAsync()` also log failed calls now. All four previously ignored checks or dropped errored calls.
- **`ailens why` only diagnoses failing calls** (bad feedback, errors or failed checks). If nothing failed, it reports that. Before, it analyzed good calls and made up problems for them.
- **Config priority is now code > env vars > `config.json` in both the SDK and the CLI.** The CLI used to let `config.json` override env vars, and the SDK ignored `config.json` completely.
- **The API key is taken from the selected provider's env var.** With `AILENS_PROVIDER=openai`, the key now comes from `OPENAI_API_KEY`. It is no longer picked up from `ANTHROPIC_API_KEY` just because that variable happens to be set.
- `ailens init` no longer overwrites an existing `config.json`. Pass `--force` to replace it.

### Fixes

- **ESM:** `import { lens } from '@techwarq/ailens'` failed with `ERR_UNSUPPORTED_DIR_IMPORT`. ESM now loads a thin wrapper around the CommonJS build, so `import` and `require` share one implementation (`instanceof AILensCheckError` works either way).
- **`diff --last` / `why`:** sessions were ordered by random UUID. They are now ordered by start time, so "last two sessions" means what it says.
- **Drift score:** TF-IDF vectors for the before and after sets were built from separate vocabularies, which made the cosine similarity meaningless. Both sets are now embedded in one shared space with smoothed IDF. The local TF-IDF fallback also works without an API key.
- **Diff results:** the measured `driftScore`, `cosineSimilarity` and `slices` are now returned in the result. Before, they were never included. The behavioral-slice classifier was also skipped whenever drift was measured, and its rates used the wrong denominator. Fixed divide-by-zero when all "before" outputs are empty.
- **Judge:** the prompt and system prompt are passed to the judge as context. Untrusted text is wrapped in tags so an output can't inject instructions. Calls use `temperature: 0`, and the returned score is clamped to 0–1. Results come back in the same order as the rules.
- **Analysis calls:** HTTP errors, API error bodies, empty and truncated responses now throw a clear error. Before, any of these could silently produce a result. The JSON parser also accepts JSON wrapped in code fences or prose. OpenAI-compatible local servers (e.g. Ollama) no longer need an API key.
- **Traces:** `t.run(name, prompt, fn)` records the prompt; the old `t.run(name, fn)` form still works. The trace now records `finalOutput`, and `success`/`error` are set when the pipeline throws outside a step. Repeated step names (agent loops) no longer collide, and step checks respect `checkErrors`.
- **Feedback:** `feedback(id, …)` works for calls from earlier sessions and returns `false` (with a warning) if the ID isn't found. Session and trace files are written atomically, and one corrupt log line no longer makes a whole session unreadable.
- **CLI:** `--version` reports the real package version, the columns in the traces list line up, and root-cause confidence is labelled as self-reported.

### Tooling

- Added a test suite (`npm test`, node:test + tsx), `npm run typecheck`, and `npm run smoke`, which checks that the built package loads via `require()` and `import` and that the CLI starts.
- Added CI on Linux, macOS and Windows with Node 18, 20 and 22.
