import random
import time

import ai_lens as lens


@lens.step(type="tool")
def find_references(topic: str) -> list[str]:
    time.sleep(0.05)
    return [f"{topic} reference {index}" for index in range(3)]


@lens.step
def write_caption(topic: str, references: list[str]) -> dict:
    time.sleep(0.1)
    return {
        "model": "claude-sonnet-5-5",
        "content": f"A short film about {topic}, inspired by {len(references)} references.",
        "usage": {"input_tokens": random.randint(800, 1200), "output_tokens": random.randint(80, 120)},
    }


@lens.trace(tags=["example"])
def make_caption(topic: str, style: str = "cinematic") -> str:
    lens.log(params={"style": style})
    references = find_references(topic)
    response = write_caption(topic, references)
    return response["content"]


if __name__ == "__main__":
    lens.configure(references=[{"output": "A short, vivid caption that names the subject and the mood.", "notes": "our house style"}])
    for topic in ["a cat surfing", "a city at night", "a forest in fog"]:
        print(make_caption(topic))
    print('\nNow try: lens track "are my captions getting better?" && lens inspect')
