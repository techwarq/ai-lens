# Set up ai-lens with a coding agent

These are instructions for a coding agent (Claude Code, Cursor, Copilot, Codex and others) to add ai-lens to a Python project. Follow the steps in order. Show the user a summary at the end.

ai-lens records every run of the project's AI feature. It scores the outputs with the project's own model, and reports whether each code change made results better or worse, why, and what it costs. Package: `ailens-evals` on PyPI. Import: `import ai_lens as lens`. Command: `lens`.

## Rules

- Do not change what any function returns or raises. Tracing must be invisible to the app.
- Never write API keys into code. Reuse the client and environment variables the project already has.
- Keep changes small: a decorator, a `lens.log(...)` line, one new file `lens_model.py`, one `lens.configure(...)` call, and one `.gitignore` line.
- If something is unclear, such as which function is the main AI feature, ask the user instead of guessing.

## 1. Check the project

1. Python must be 3.10 or newer. If it's older, stop and tell the user.
2. Find how dependencies are managed (`pyproject.toml` with uv or poetry, `requirements.txt`, `Pipfile`) and which model SDK the project already uses (`anthropic`, `openai`, `google-genai` or another).
3. Find the AI feature to track: the function that turns inputs into the final output (text, JSON, an image, a video or audio). Note what it returns.

## 2. Install

Add `ailens-evals` the way the project already adds dependencies, for example:

```bash
uv add ailens-evals            # uv
poetry add ailens-evals        # poetry
pip install ailens-evals       # and add it to requirements.txt
```

If the output is an image, video or audio, check that `ffmpeg` and `ffprobe` are on the PATH (`ffmpeg -version`). If they're missing, tell the user to install ffmpeg (`brew install ffmpeg` or `apt install ffmpeg`). Don't install system packages yourself.

## 3. Trace the AI feature

Put `@lens.trace` on the main function, and `@lens.step` on its important sub-calls.

```python
import ai_lens as lens

@lens.trace
def generate_video(prompt: str, start_frame: str, seed: int = 42) -> str:
    script = write_script(prompt)
    lens.log(params={"seed": seed, "steps": 50}, model="video-model-v3")
    return render(script, start_frame)

@lens.step
def write_script(prompt: str):
    return client.messages.create(...)

@lens.step(type="tool")
def search_footage(query: str): ...
```

- **The output** should be the real result: text, a dict, or the **path to the media file** that was produced. If the function returns something else (like an upload URL), tell the user. Don't change the return value.
- **The prompt:** if the function has a `prompt` argument, it's recorded automatically. If the real prompt is built inside the function, add `lens.log(prompt=final_prompt)` right after it's built.
- **Input files:** image, video or audio paths passed as arguments are recorded automatically. For others, add `lens.log(assets=[path, ...])`.
- **Version settings:** call `lens.log(params={...})` with the settings someone might change between versions (model name, seed, steps, temperature, resolution). Call `lens.log(model="...")` for the generation model if no LLM response is returned.
- **Tokens and cost:** steps that return a Claude or OpenAI response are counted automatically. If a call's response isn't returned, add `lens.log(usage=response)`.
- **Async functions** work the same way. Don't trace small helpers or code that runs thousands of times per request.

## 4. Give ai-lens the project's model

Create `lens_model.py` at the project root, next to where the app is started from, so that `import lens_model` works. Use the SDK the project already uses, a model that can see images, and the project's existing credentials. The function takes `parts`, a list of `str` and JPEG `bytes` in order, and returns the model's reply.

Anthropic:

```python
import base64

import anthropic

client = anthropic.Anthropic()


def judge(parts):
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(part).decode()}}
        if isinstance(part, bytes)
        else {"type": "text", "text": part}
        for part in parts
    ]
    return client.messages.create(model="claude-opus-5-5", max_tokens=16000, messages=[{"role": "user", "content": content}])
```

OpenAI:

```python
import base64

from openai import OpenAI

client = OpenAI()


def judge(parts):
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(part).decode()}}
        if isinstance(part, bytes)
        else {"type": "text", "text": part}
        for part in parts
    ]
    return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": content}])
```

For any other provider or a local model, return the reply text as a `str`.

Then register it once where the app starts, such as the entry point or settings module:

```python
import ai_lens as lens

lens.configure(model="lens_model:judge")
```

Use the `"module:function"` string, not the function object, so the `lens` command can import it. If the app runs from a subfolder, use the dotted module path, for example `"myapp.lens_model:judge"`.

## 5. Optional: reference results

If the user has examples of great output, create `evals/best.jsonl`:

```jsonl
{"output": "evals/best/example.mp4", "inputs": {"prompt": "a cat surfing"}, "notes": "smooth motion, sharp subject"}
{"output": "Captions are under 12 words and name the subject."}
```

Register it next to the model: `lens.configure(model="lens_model:judge", references="evals/best.jsonl")`. Paths are relative to the `.jsonl` file. Don't invent references; only add them if the user gives you examples.

## 6. Ignore the data folder

Add this line to `.gitignore`:

```
.ai-lens/
```

## 7. Verify

1. Run the app, or call the traced function once with realistic input.
2. Check that `.ai-lens/runs.jsonl` exists and its last line has the expected `inputs`, `prompt`, `output` and `output_type`.
3. Create the plan, using the user's goal if they gave one:
   ```bash
   lens track "are my outputs getting better with my changes?" --type <text|json|image|video|audio>
   ```
   If it says no model could be reached, check that `lens_model.py` is importable from the project root and that the credentials are set.
4. Run `lens inspect` and check that the report prints without errors.

## 8. Tell the user

Summarise in a few lines:
- which functions are traced, and what is logged as prompt, params and assets
- where `lens_model.py` is and which model it uses
- the `lens track` command you ran, and the checks it created
- how to use it from now on: change code, run the app, `lens inspect`, and `lens suggest` for fixes
