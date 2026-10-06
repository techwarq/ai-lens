# ai-lens

**Evals for AI apps that make text, images, video or audio.** Add one decorator, say in plain English what you care about, and ai-lens tells you whether every change made your outputs better or worse, why, and what it costs.

[![PyPI](https://img.shields.io/pypi/v/ailens-evals)](https://pypi.org/project/ailens-evals/)
[![CI](https://github.com/techwarq/ai-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/techwarq/ai-lens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```
$ lens inspect

What changed
  a1f3c2d → b72c9e1  WORSE  -0.21 [-0.29, -0.13]  (12 matched inputs)
      changed: commit "faster renderer", model gen-v3 → gen-v3-fast
      biggest drop: judge: Is the motion smooth, with no warping? -0.33
      rubric: Sharp subject -0.40
      judge: "Frames 2 and 3 show the cat's legs melting into the board."
      side by side: new better 1 · old better 9 · tie 2
        "The old clip keeps the fur sharp; the new one smears it in motion."
      cost/run $0.0310 → $0.0170

Where the tokens go (b72c9e1)
  call                       model              calls/run  in/call  out/call  cost/run  share
  write_script (1 repeated)  claude-sonnet-5-5  2.0        2.9k     310       $0.0170   100%
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
| `judge` | Your model grades one specific check from 0 to 1, with a reason. It sees the prompt, input assets and the output (images directly, video as 4 frames in order) | 1 call |
| `reference` | Your model compares the output with your best results, scored per rubric criterion | 1 call |
| side by side | Old vs new output for the same inputs, asked twice with the order swapped. If the answers disagree, it's a tie | 2 calls |

Audio content can't be judged by vision models, so audio uses the free checks.

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

1. **Versions.** Runs are grouped by git commit, uncommitted changes, model and `params`. Each group is a version.
2. **Same inputs first.** If two versions share at least 3 inputs, they're compared only on those inputs. This way a version isn't judged worse just because it got harder prompts. Otherwise both versions are compared as a whole, and the report says "different inputs".
3. **Confidence.** Each run's quality is the average of its check scores. The change is the mean difference, with a 95% bootstrap interval:
   - **BETTER** or **WORSE** only when the whole interval is on one side of zero
   - **NO CLEAR CHANGE** when it isn't
   - **NOT ENOUGH DATA** below 3 runs
4. **Side by side.** For up to 5 matched inputs per change, your model sees both outputs and picks the better one, twice, with the order swapped. Disagreements count as ties, which cancels position bias.
5. **Locked judge.** The plan records which model function judged it. If you switch models, `lens inspect` warns that new scores aren't comparable with old ones.

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
- Steps inside threads you start yourself aren't attached to the run. Async code works.
- Judge scores are only as good as your model. Use your best vision model for media, and compare versions on the same inputs for a fair verdict.

## Coming from `@techwarq/ailens` on npm?

That TypeScript package is deprecated. ai-lens is now a Python package with multimodal evals, statistics and reference grading. Install it with `pip install ailens-evals`.

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
