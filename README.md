# ai-lens

**Evals for AI apps that make text, images, video or audio.** Add one decorator, say in plain English what you care about, and ai-lens tells you whether every change made your outputs better or worse, why, and what it costs.

[![PyPI](https://img.shields.io/pypi/v/ailens-evals)](https://pypi.org/project/ailens-evals/)
[![CI](https://github.com/techwarq/ai-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/techwarq/ai-lens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```
$ lens inspect

are my videos getting better?

Latest change
  WORSE  a1f3c2d → a1f3c2d +edit 9e04b1

    The faster model breaks the motion: the cat's legs melt into the board mid-jump, and
    the fur smears whenever it moves. The old version won 9 of 12 side-by-side comparisons.

    Scores
    ↓ c3  Is the motion smooth, with no warping?  0.81 → 0.48  -0.33 ±0.08
           "Frames 2 and 3 show the cat's legs melting into the board."
    ~ c4  Does the cat stay in frame?             0.70 → 0.77  +0.07 ±0.12  could be noise
      rubric: Sharp subject -0.40

    side by side  new better 1 · old better 9 · tie 2
      "The old clip keeps the fur sharp; the new one smears it in motion."

    changed  uncommitted edits to prompts/motion.py, model gen-v3 → gen-v3-fast
    cost/run $0.0310 → $0.0170
    12 matched inputs

Versions
  version               runs  pass  c3     c4    cost/run  tokens/run  time
  a1f3c2d               12    100%  0.81   0.70  $0.0310   3.2k        41.0s
  a1f3c2d +edit 9e04b1  12    100%  0.48↓  0.77  $0.0170   3.2k        28.5s
```

## Why ai-lens

- **One decorator.** `@lens.trace` records inputs, prompt, input assets, output, latency, errors, git commit, tokens and cost for every run and step.
- **Any output.** Text, JSON, images, video (judged from sampled frames) and audio.
- **Plain-English goals.** `lens track "are my videos getting better?"` writes the eval plan for you.
- **Your best results become the bar.** Point ai-lens at examples you love. It learns a rubric from them and grades every new output against them.
- **Verdicts you can trust.** Versions are compared on the same inputs, with 95% confidence intervals. Old and new outputs are compared side by side, in both orders, so position bias can't decide. When there's too little data, it says so.
- **It names the cause.** Each regression is tied to the commit, model, parameter, prompt or asset that changed.
- **It finds token waste.** The most expensive calls, repeated identical calls, and cost per run over time.
- **It suggests fixes.** `lens suggest` reads your report, diff and code, and proposes changes with file and line numbers.
- **Your model, your keys.** ai-lens ships no model and never sees an API key. All judging runs through a function you provide.

## Install

```bash
pip install ailens-evals
```

You need Python 3.10+. For image, video and audio outputs, also install [ffmpeg](https://ffmpeg.org/download.html) (`brew install ffmpeg` or `apt install ffmpeg`). The package itself has no dependencies.

## Set up with a coding agent

Paste this into Claude Code, Cursor, Copilot or Codex, inside your project:

```
Set up ai-lens in this project by following https://raw.githubusercontent.com/techwarq/ai-lens/main/docs/setup-with-ai.md
```

The agent installs the package, traces your AI feature, wires in your own model, and checks that it works. Or follow the steps below yourself.

## Quickstart

### 1. Give ai-lens your model

ai-lens never calls a model on its own. Write one function in your project that takes a list of **parts** and returns the model's reply. Parts are text strings and JPEG image bytes, in order. Use a vision model if you generate images or video.

```python
# lens_model.py
import base64
import anthropic

client = anthropic.Anthropic()

def judge(parts):
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(part).decode()}}
        if isinstance(part, bytes) else {"type": "text", "text": part}
        for part in parts
    ]
    return client.messages.create(model="claude-opus-5-5", max_tokens=16000, messages=[{"role": "user", "content": content}])
```

<details>
<summary>OpenAI or any other model</summary>

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

def judge(parts):
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(part).decode()}}
        if isinstance(part, bytes) else {"type": "text", "text": part}
        for part in parts
    ]
    return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": content}])
```

```python
# Anything else: return the reply as a string
def judge(parts):
    return my_model.generate(parts)
```

</details>

Return a plain string, or return the Claude, OpenAI or Gemini response object as is. For Claude responses, and other models you've priced with `lens.configure(prices=...)`, the cost is counted as eval spend.

### 2. Trace your app

```python
import ai_lens as lens

lens.configure(model="lens_model:judge")

@lens.trace
def make_video(prompt: str, start_frame: str, seed: int = 42) -> str:
    script = write_script(prompt)
    lens.log(params={"seed": seed, "steps": 50}, model="gen-v3")
    return render(script, start_frame)   # e.g. "out/video.mp4"

@lens.step
def write_script(prompt):
    return client.messages.create(...)   # tokens and cost picked up automatically

@lens.step(type="tool")
def search_stock_footage(query): ...
```

Run your app as usual. Each call adds one line to `.ai-lens/runs.jsonl`.

### 3. Say what you want to know

```bash
lens track "are my videos getting better or worse with my changes?" --type video
```

```
Tracking: are my videos getting better or worse with my changes?
Output type: video
  c1  success
  c2  video_integrity
  c3  judge: Does the motion stay smooth with no warping between frames?
  c4  judge: Does the video follow the prompt and the start frame?
```

The plan is saved in `.ai-lens/plan.json`. Edit it if you want different checks.

### 4. Change your code, run it again, and inspect

```bash
lens inspect
```

Only new runs are scored, so each run is paid for once. Then you get the versions table, what changed and why, rubric scores, side-by-side results, token hotspots and failures.

### 5. Get fixes

```bash
lens suggest
```

```
1. Revert gen-v3-fast for the motion pass  [quality]
   app/render.py:42
   Problem:  smoothness fell from 0.82 to 0.49 when the model changed
   Change:   model="gen-v3" in render_motion(); keep gen-v3-fast for the thumbnail pass only
   Expected: smoothness back to ~0.8, cost +$0.004/run
```

## Tracing

| You write | ai-lens records |
|---|---|
| `@lens.trace` | A run: inputs, output, latency, status, error with traceback, git commit, branch, uncommitted changes, source line |
| `@lens.step`, `@lens.step(type="tool")` | A step inside the run, nested by parent. Steps that return an LLM response become `llm` steps with model, tokens and cost |
| `with lens.trace("batch") as run:` / `with lens.step("render"):` | The same, for code that isn't a single function |
| `lens.log(params={...})` | Settings that define a version, such as seed, steps or temperature |
| `lens.log(prompt=full_prompt)` | The exact prompt sent. By default it's taken from a `prompt` argument |
| `lens.log(assets=["style.png"])` | Extra input files. Media passed as arguments are picked up automatically |
| `lens.log(usage=response)` | Token usage from a response you didn't return |
| `lens.log(anything=value)` | Saved under `metadata` |

**Guarantees:**
- Tracing never changes what your function returns or raises.
- If saving fails, you get a warning and your app keeps running.
- Async functions and threads are supported.
- Media outputs and input assets are copied and fingerprinted (sha256), so overwriting `out/video.mp4` never corrupts history.

## Graders

| Grader | What it does | Cost |
|---|---|---|
| `success` | Did the run finish without an error? | free |
| `not_empty`, `valid_json` | Basic output checks | free |
| `image_integrity`, `video_integrity`, `audio_integrity`, `audio_silence` | Does the file decode, have frames and a non-zero length, and is it not mostly silent? | free (ffmpeg) |
| `geval` | [G-Eval](https://arxiv.org/abs/2303.16634): when you run `lens track`, your model writes evaluation steps for one criterion. They're saved in `plan.json`, so every score uses the same steps. Each output is then graded by following those steps, scored 0 to 10 against fixed bands, and saved with notes for each step | 1 call |
| `trajectory` | G-Eval over the whole run: every step and tool call, with its arguments, results and errors, in order. Checks whether tool use was correct. Added automatically when your app has `@lens.step(type="tool")` steps | 1 call |
| `judge` | Your model answers one narrow question about the output, 0 to 1, with a reason | 1 call |
| `reference` | Your model compares the output with your best results, scored per rubric criterion | 1 call |
| side by side | Old vs new output for the same inputs, asked twice with the order swapped. If the answers disagree, it's a tie | 2 calls |

Model graders see the prompt, input assets and the output. They see images directly and video as 4 frames in order. Audio is shown as a waveform and a spectrogram picture, plus measured length, loudness and silence.

To tune a G-Eval check, edit its `steps` in `.ai-lens/plan.json`.

## Grade against your best results

```python
lens.configure(references="evals/best.jsonl")
```

```jsonl
{"output": "best/cat_surfing.mp4", "inputs": {"prompt": "a cat surfing"}, "notes": "smooth motion, sharp fur"}
{"output": "best/city.png", "notes": "this is the colour grade we want"}
{"output": "Captions are under 12 words and name the subject."}
```

- **`output`** is a file or a text description of what good looks like.
- **`inputs`** (optional) limits a reference to runs made from the same inputs.
- **`notes`** (optional) say what makes it good.
- Paths are relative to the `.jsonl` file. You can also pass a list of these dicts.

`lens track` studies up to 4 references with your model and writes 3–6 specific criteria. Every new output is compared with its closest references, and the report tracks each criterion per version. If your references change, `lens inspect` tells you to run `lens track` again.

## How verdicts are made

1. **Versions.** Runs are grouped by git commit, model and `params`, plus a fingerprint of your uncommitted edits. Each set of edits you try, like a prompt change, is its own version, labelled e.g. `a1f3c2d +edit 9e04b1`. New untracked files, like your app's outputs, don't count as edits.
2. **Same inputs first.** If two versions share at least 3 inputs, they're compared only on those inputs. This way a version isn't judged worse just because it got harder prompts. Otherwise both versions are compared as a whole, and the report says "different inputs".
3. **Each check on its own.** Every scored check gets its own mean difference with a 95% bootstrap interval. A check moved only when the whole interval is on one side of zero. Pass/fail checks (success, valid JSON, file integrity) are gates: they are kept out of quality scores, and any drop in their pass rate is flagged.
4. **The headline.**
   - **WORSE** if a pass/fail check started failing
   - otherwise, a lopsided side-by-side result (at least 80% one way) decides
   - otherwise, **MIXED** if some checks went up and others down, **BETTER** or **WORSE** if they all moved one way
   - **NO CLEAR CHANGE** when nothing moved beyond noise, with how many more matched inputs you'd need to see a 0.10 change
   - **NOT ENOUGH DATA** below 3 runs
5. **Side by side.** For up to 5 matched inputs per change, your model sees both outputs and picks the better one, twice, with the order swapped. Disagreements count as ties, which cancels position bias.
6. **In plain words.** For a BETTER, WORSE or MIXED result, your model writes 2–3 sentences on what the outputs now do differently, from what the judge saw, shown above the scores. It's written once per result and cached, so it adds about one cheap call per change. Without a model, or with `--no-eval`, you get a plain summary built from the scores.
7. **Locked judge.** The plan records which model function judged it. If you switch models, `lens inspect` warns that new scores aren't comparable with old ones.

## Commands

| Command | What it does |
|---|---|
| `lens track "<goal>" [--type text\|json\|image\|video\|audio] [--model module:function]` | Creates the eval plan. The output type is detected from your runs if you leave it out |
| `lens inspect [--per-version 20] [--pairs 5] [--no-eval] [--json]` | Scores new runs (at most `--per-version` per version), runs `--pairs` side-by-side comparisons per change, and prints the report |
| `lens suggest` | Proposes file-and-line fixes for regressions, failures and token waste |

`ailens` works as an alias for `lens`.

## Configuration

| What | How |
|---|---|
| Model for planning, judging and suggestions | `lens.configure(model="module:function")` or `lens track --model module:function` |
| Reference results | `lens.configure(references="path.jsonl")`, or a list of dicts |
| Data folder | `lens.configure(path=...)` or `AI_LENS_DIR` (default `.ai-lens/`) |
| Turn tracing off | `lens.configure(enabled=False)` or `AI_LENS_DISABLED=1` |
| Don't copy media | `lens.configure(copy_artifacts=False)` |
| Prices for non-Claude models | `lens.configure(prices={"my-model": (input_per_million, output_per_million)})` |

## What's stored

```
.ai-lens/
  runs.jsonl        one line per traced run
  plan.json         goal, checks, rubric, judge
  evals.jsonl       one score per run and check
  pairwise.jsonl    side-by-side results
  config.json       where your model function lives (never a key)
  references.json   your normalised reference set
  artifacts/        outputs, by run id
  assets/           input files, by content hash
  references/       reference files, by content hash
```

Everything stays on your machine, in plain JSON you can read, diff or commit.

## Limitations

- Video is judged from 4 sampled frames, so judgements about audio and timing within a clip are limited.
- Your model can't hear audio. It sees the audio's shape: silence, clipping, noise, rhythm and pitch. It can't tell what words are spoken.
- G-Eval in the paper weights scores by token probabilities. ai-lens only gets text back from your model, so it uses the whole-number score.
- Steps inside threads you start yourself aren't attached to the run. Async code works.
- Judge scores are only as good as your model. Use your best vision model for media, and compare versions on the same inputs for a fair verdict.

## Coming from `@techwarq/ailens` on npm?

That was the earlier TypeScript version and it's no longer developed. ai-lens is now a Python package with multimodal evals, statistics and reference grading. Install it with `pip install ailens-evals`.

## Development

```bash
git clone https://github.com/techwarq/ai-lens && cd ai-lens
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/basedpyright --pythonpath .venv/bin/python ai_lens tests examples
```

To release, bump `version` in `pyproject.toml`, add a `CHANGELOG.md` entry, and push a tag like `v0.2.1`. GitHub Actions builds the package and publishes it to PyPI through trusted publishing, so no token is stored anywhere.

## License

MIT
