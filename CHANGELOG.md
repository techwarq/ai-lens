# Changelog

## Unreleased

- **G-Eval for every output type.** The new `geval` grader writes evaluation steps for a criterion once, locks them in `plan.json`, and grades each output against them from 0 to 10. It works on text, JSON, images, video and audio.
- **Tool use checks.** The new `trajectory` grader runs G-Eval over every step and tool call in a run, to check the right tools were called with the right arguments and their results were used.
- **Audio for model graders.** Audio is shown as a waveform and spectrogram, plus measured loudness and silence, so `judge`, `geval` and `reference` work on audio too.
- **Prompt edits are versions.** Uncommitted edits are fingerprinted, so each try is its own version (`a1f3c2d +edit 9e04b1`) and the report names the edited files. Untracked output files no longer mark a version as changed.
- **An honest headline.** Pass/fail checks are gates instead of quality points. Each check is tested on its own, opposite moves read **MIXED**, and a lopsided side-by-side result decides the verdict.
- **What went wrong, in plain words.** The latest change opens with 2–3 sentences your model writes from what the judge saw, then the scores, with a judge quote under each check that moved. Summaries are cached in `.ai-lens/summaries.jsonl`.
- **How much more data.** When nothing moved, the report says how many more matched inputs would show a 0.10 change.
- **A simpler report.** It leads with the latest change and its verdict, lists only the checks that moved, keeps older changes to one line each, and uses colour in a terminal (set `NO_COLOR` to turn it off).

## 0.2.0

A rewrite in Python, published on PyPI as `ailens-evals`. It succeeds the earlier TypeScript package `@techwarq/ailens`.

- **Tracing:** `@lens.trace`, `@lens.step` and `lens.log()` record inputs, prompt, input assets, output, latency, errors, git commit, tokens and cost for every run and step.
- **Multimodal:** text, JSON, image, video and audio outputs. Media is copied and fingerprinted so history can't be overwritten.
- **Planning:** `lens track "<goal>"` turns a plain-English goal into an eval plan.
- **Graders:**
  - code checks
  - LLM-as-a-judge with images and video frames
  - reference grading with a rubric learned from your best results
  - side-by-side old vs new comparison, asked in both orders to cancel position bias
- **Statistics:** versions are compared on matched inputs with 95% bootstrap intervals, and the report says "not enough data" instead of guessing.
- **Causes:** the report names the commit, model, parameter, prompt or asset change behind each regression.
- **Cost:** token and cost hotspots, including repeated identical calls.
- **Suggestions:** `lens suggest` proposes file-and-line fixes for quality and cost.
- **Bring your own model:** ai-lens ships no model and never handles API keys.
