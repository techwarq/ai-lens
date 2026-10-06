import ai_lens as lens
from ai_lens.cli import main


def test_full_flow(fake_llm, capsys):
    assert main(["inspect"]) == 1
    assert "lens track" in capsys.readouterr().err

    assert main(["track", "are my answers on topic?", "--type", "text"]) == 0
    assert "Is it on topic?" in capsys.readouterr().out

    @lens.trace
    def answer(question):
        return f"answer to {question}"

    answer("why?")
    assert main(["inspect"]) == 0
    output = capsys.readouterr().out
    assert "Versions" in output
    assert "0.90" in output

    assert main(["suggest"]) == 0
    assert "Revert the model switch" in capsys.readouterr().out


def test_changing_the_judge_model_warns(fake_llm, capsys, monkeypatch):
    from ai_lens.config import settings

    assert main(["track", "goal", "--type", "text"]) == 0

    @lens.trace
    def answer(question):
        return question

    answer("x")

    def other_model(parts):
        return "{}"

    monkeypatch.setattr(settings, "model", other_model)
    assert main(["inspect", "--no-eval"]) == 0
    assert "your model changed" in capsys.readouterr().err
