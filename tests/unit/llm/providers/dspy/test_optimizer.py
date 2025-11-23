from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import pytest

from various_llm_benchmark.llm.providers.dspy import optimizer
from various_llm_benchmark.llm.providers.dspy.client import SupportsDsPyLM
from various_llm_benchmark.prompts.prompt import PromptTemplate


class _FakeTeleprompter:
    def __init__(self, *, metric: object, **kwargs: object) -> None:
        self.metric = metric
        self.kwargs = kwargs
        self.trainset: list[object] | None = None
        self.task: object | None = None

    def compile(self, task: object, trainset: Sequence[object]) -> object:
        self.task = task
        self.trainset = list(trainset)
        return _FakePredictor(output="expected")


class _FakePredictor:
    def __init__(self, *, output: str) -> None:
        self.output = output
        self.calls: list[str] = []

    def __call__(self, *, prompt: str) -> object:
        self.calls.append(prompt)
        return SimpleNamespace(output=self.output)


class _RecordingLM(SupportsDsPyLM):
    model_type = "chat"

    def __init__(self, *, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append({"prompt": prompt, "messages": messages, **kwargs})
        content = prompt if prompt is not None else ""
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], model=self.model)

    def copy(self, **_kwargs: object) -> _RecordingLM:
        return _RecordingLM(model=self.model, temperature=self.temperature)


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    """Provide a small prompt tuning dataset file."""
    data = [
        {"input": "first", "target": "expected"},
        {"input": "second", "target": "expected"},
    ]
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in data), encoding="utf-8")
    return path


def test_load_prompt_tuning_examples_reads_valid_lines(dataset_file: Path) -> None:
    """Valid JSONL lines should be parsed into examples."""
    examples = optimizer.load_prompt_tuning_examples(dataset_file)

    assert len(examples) == 2
    assert examples[0].input == "first"
    assert examples[0].target == "expected"


def test_load_prompt_tuning_examples_raises_on_invalid_line(tmp_path: Path) -> None:
    """Invalid lines should raise with context."""
    path = Path(tmp_path) / "broken.jsonl"
    path.write_text('{"input": "ok", "target": "x"}\n{"input": 123}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid dataset format"):
        optimizer.load_prompt_tuning_examples(path)


def test_optimize_prompt_uses_teleprompter_and_metric(dataset_file: Path) -> None:
    """Teleprompter output should improve the optimization score."""
    examples = optimizer.load_prompt_tuning_examples(dataset_file)
    template = PromptTemplate(system="system text")
    teleprompter: optimizer.Teleprompter = _FakeTeleprompter(metric=None)

    def build_teleprompter(**kwargs: object) -> _FakeTeleprompter:
        assert "metric" in kwargs
        teleprompter.kwargs = kwargs
        return teleprompter

    result = optimizer.optimize_prompt(
        examples,
        template,
        model="model-x",
        temperature=0.5,
        max_bootstrapped_demos=2,
        num_candidates=3,
        teleprompter_factory=build_teleprompter,
        predict_factory=lambda _: _FakePredictor(output="wrong"),
        lm_factory=_RecordingLM,
    )

    assert result.base_score == 0.0
    assert result.optimized_score == 1.0
    assert teleprompter.trainset is not None
    assert teleprompter.kwargs["max_bootstrapped_demos"] == 2
    assert teleprompter.kwargs["num_candidates"] == 3
