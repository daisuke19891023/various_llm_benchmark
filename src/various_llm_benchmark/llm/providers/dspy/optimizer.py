from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import dspy
from pydantic import BaseModel, Field, ValidationError

from various_llm_benchmark.llm.providers.dspy.client import SupportsDsPyLM

if TYPE_CHECKING:
    from various_llm_benchmark.prompts.prompt import PromptTemplate

_DSPY: Any = cast("Any", dspy)


class PromptTuningExample(BaseModel):
    """Structured example for prompt tuning."""

    input: str = Field(..., description="User input text")
    target: str = Field(..., description="Expected model output")


class PromptOptimizationResult(BaseModel):
    """Result of a prompt optimization run."""

    base_score: float = Field(..., description="Score before optimization")
    optimized_score: float = Field(..., description="Score after optimization")
    trainset_size: int = Field(..., description="Number of examples used for tuning")


class Teleprompter(Protocol):
    """Subset of the DsPy teleprompter interface used by the optimizer."""

    def compile(self, task: object, trainset: Sequence[object]) -> object:  # pragma: no cover - protocol method
        """Compile the given task with a trainset to produce an optimized module."""
        ...


TeleprompterFactory = Callable[..., Teleprompter]
PredictFactory = Callable[[type[dspy.Signature]], Callable[..., Any]]
LMFactory = Callable[..., SupportsDsPyLM]


def load_prompt_tuning_examples(path: Path) -> list[PromptTuningExample]:
    """Load prompt tuning examples from a JSONL file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        error_message = f"Dataset file not found: {dataset_path}"
        raise FileNotFoundError(error_message)

    raw_text = dataset_path.read_text(encoding="utf-8")
    examples: list[PromptTuningExample] = []
    errors: list[str] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            examples.append(PromptTuningExample.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {line_number}: {exc}")

    if errors:
        details = "\n".join(errors)
        error_message = f"Invalid dataset format in {dataset_path}:\n{details}"
        raise ValueError(error_message)
    if not examples:
        error_message = f"No valid examples found in dataset: {dataset_path}"
        raise ValueError(error_message)

    return examples


def optimize_prompt(
    examples: Sequence[PromptTuningExample],
    prompt_template: PromptTemplate,
    *,
    model: str,
    temperature: float,
    max_bootstrapped_demos: int = 4,
    num_candidates: int = 8,
    num_threads: int = 1,
    teleprompter_factory: TeleprompterFactory | None = None,
    predict_factory: PredictFactory | None = None,
    lm_factory: LMFactory | None = None,
) -> PromptOptimizationResult:
    """Run DsPy's teleprompter optimizer for the given dataset."""
    _configure_lm(model, temperature, lm_factory or _create_default_lm)
    trainset: list[Any] = [_to_dspy_example(item, prompt_template) for item in examples]
    metric = _exact_match_metric
    teleprompter = (teleprompter_factory or _default_teleprompter)(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        num_candidates=num_candidates,
        num_threads=num_threads,
    )

    predict_builder = predict_factory or _DSPY.Predict
    task = predict_builder(_PromptTuningSignature)
    optimized_task_obj = teleprompter.compile(task, trainset=trainset)
    optimized_task = cast("Callable[..., Any]", optimized_task_obj)

    base_score = _evaluate(task, trainset, metric)
    optimized_score = _evaluate(optimized_task, trainset, metric)

    return PromptOptimizationResult(
        base_score=base_score,
        optimized_score=optimized_score,
        trainset_size=len(trainset),
    )


def _configure_lm(model: str, temperature: float, factory: LMFactory) -> None:
    lm = factory(model=model, temperature=temperature)
    _DSPY.settings.configure(lm=lm)


def _to_dspy_example(example: PromptTuningExample, prompt_template: PromptTemplate) -> Any:
    combined_prompt = prompt_template.to_prompt_text(example.input)
    raw_example = _DSPY.Example(prompt=combined_prompt, target=example.target)
    return raw_example.with_inputs("prompt")


def _evaluate(
    task: Callable[..., Any],
    trainset: Sequence[Any],
    metric: Callable[[Any, Any], float],
) -> float:
    if not trainset:
        return 0.0

    scores = [_safe_score(task, item, metric) for item in trainset]
    return sum(scores) / len(scores)


def _safe_score(
    task: Callable[..., Any],
    example: Any,
    metric: Callable[[Any, Any], float],
) -> float:
    prompt_value = str(getattr(example, "prompt", ""))
    prediction = task(prompt=prompt_value)
    return float(metric(example, prediction))


def _exact_match_metric(example: Any, prediction: object, *_: object) -> float:
    predicted = _normalize_output(getattr(prediction, "output", ""))
    expected = _normalize_output(getattr(example, "target", ""))
    return float(predicted == expected)


def _normalize_output(text: str) -> str:
    return text.strip().casefold()


def _default_teleprompter(**kwargs: object) -> Teleprompter:
    teleprompter = _DSPY.teleprompt.BootstrapFewShotWithRandomSearch(**kwargs)
    return cast("Teleprompter", teleprompter)


def _create_default_lm(**kwargs: Any) -> SupportsDsPyLM:
    return cast("SupportsDsPyLM", _DSPY.LM(**kwargs))


class _PromptTuningSignature(dspy.Signature):
    prompt: Any = _DSPY.InputField(desc="Combined system prompt and user input")
    output: Any = _DSPY.OutputField(desc="Model response for the prompt")
