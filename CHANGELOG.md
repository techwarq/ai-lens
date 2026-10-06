# Changelog

## 0.2.0

A rewrite in Python, published on PyPI as `ailens-evals`. It replaces the TypeScript package `@techwarq/ailens`, which is now deprecated.

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
