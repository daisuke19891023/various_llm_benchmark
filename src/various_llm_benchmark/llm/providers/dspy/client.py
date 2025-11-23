from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import dspy

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse
else:
    ModelResponse = Any

from various_llm_benchmark.llm.protocol import LLMClient
from various_llm_benchmark.models import ChatMessage, ImageInput, LLMResponse


class SupportsDsPyLM(Protocol):
    """Subset of DsPy LM interface required by the adapter."""

    model_type: str
    model: str

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Execute a language model call and return the raw response."""
        ...

    def copy(self, **kwargs: Any) -> Self:
        """Return a configured copy of the LM instance."""
        ...


LMFactory = Callable[..., SupportsDsPyLM]
NormalizedOutput = str | Mapping[str, object]
NormalizedOutputs = list[NormalizedOutput]


class DsPyLLMClient(LLMClient):
    """Adapter for DsPy LM to the LLMClient protocol."""

    def __init__(
        self,
        default_model: str,
        *,
        temperature: float = 0.7,
        lm_factory: LMFactory | None = None,
        **lm_kwargs: object,
    ) -> None:
        """Create a DsPy LM wrapper with defaults."""
        self._default_model = default_model
        self._temperature = temperature
        self._lm_factory: LMFactory = lm_factory or _default_lm_factory
        self._lm_kwargs = lm_kwargs
        self._lm: SupportsDsPyLM = self._lm_factory(
            model=default_model, temperature=temperature, **self._lm_kwargs,
        )

    def generate(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Generate a completion without history."""
        lm = self._resolve_lm(model)
        raw_response = _call_forward(lm, prompt=prompt, messages=None)
        outputs = _normalize_outputs(raw_response, getattr(lm, "model_type", "chat"))
        content = _extract_content(outputs)
        model_name = _extract_model(raw_response, model or self._default_model)
        raw = _dump_raw(raw_response, outputs)
        return LLMResponse(content=content, model=model_name, raw=raw)

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        """Generate a completion using chat-style messages."""
        lm = self._resolve_lm(model)
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        raw_response = _call_forward(lm, prompt=None, messages=formatted_messages)
        outputs = _normalize_outputs(raw_response, getattr(lm, "model_type", "chat"))
        content = _extract_content(outputs)
        model_name = _extract_model(raw_response, model or self._default_model)
        raw = _dump_raw(raw_response, outputs)
        return LLMResponse(content=content, model=model_name, raw=raw)

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Vision is not supported by the DsPy adapter."""
        message = "Vision generation is not supported for DsPy"
        raise NotImplementedError(message)

    def _resolve_lm(self, model: str | None) -> SupportsDsPyLM:
        if model is None or model == self._default_model:
            return self._lm
        return self._lm_factory(
            model=model, temperature=self._temperature, **self._lm_kwargs,
        )


def _call_forward(
    lm: SupportsDsPyLM,
    *,
    prompt: str | None,
    messages: list[dict[str, str]] | None,
) -> ModelResponse:
    return lm.forward(prompt=prompt, messages=messages)


def _normalize_outputs(response: ModelResponse, model_type: str) -> NormalizedOutputs:
    if model_type == "responses" and getattr(response, "output", None) is not None:
        return _normalize_response_outputs(response)
    return _normalize_completion_outputs(response)


def _extract_choice_text(choice: object) -> str | None:
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    text_attr = getattr(choice, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(choice, Mapping):
        mapping_choice = cast("Mapping[str, object]", choice)
        text_candidate = mapping_choice.get("text")
        if isinstance(text_candidate, str):
            return text_candidate
    return None


def _extract_choice_tool_calls(choice: object) -> object | None:
    message = getattr(choice, "message", None)
    if message is not None:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is not None:
            return tool_calls
    tool_calls_attr = getattr(choice, "tool_calls", None)
    if tool_calls_attr is not None:
        return tool_calls_attr
    if isinstance(choice, Mapping):
        mapping_choice = cast("Mapping[str, object]", choice)
        return mapping_choice.get("tool_calls")
    return None


def _normalize_completion_outputs(response: ModelResponse) -> NormalizedOutputs:
    outputs: NormalizedOutputs = []
    choices = cast("list[Any]", getattr(response, "choices", []))
    for choice in choices:
        text = _extract_choice_text(choice)
        tool_calls = _extract_choice_tool_calls(choice)
        output: dict[str, object] = {}
        if isinstance(text, str):
            output["text"] = text
        if tool_calls is not None:
            output["tool_calls"] = tool_calls
        if output:
            if len(output) == 1 and "text" in output:
                outputs.append(cast("str", output["text"]))
            else:
                outputs.append(output)
        else:
            choice_text = str(cast("object", choice))
            outputs.append(choice_text)
    return outputs


def _normalize_response_outputs(response: ModelResponse) -> NormalizedOutputs:
    output_items = getattr(response, "output", None)
    if not isinstance(output_items, Sequence):
        return []
    typed_output_items = cast("Sequence[Any]", output_items)
    text_outputs, tool_calls, reasoning = _collect_response_sections(typed_output_items)
    return _build_response_outputs(text_outputs, tool_calls, reasoning)


def _collect_response_sections(
    output_items: Sequence[object],
) -> tuple[list[str], list[object], list[str]]:
    text_outputs: list[str] = []
    tool_calls: list[object] = []
    reasoning: list[str] = []
    for item in output_items:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            text_outputs.extend(_extract_text_outputs(item))
        elif item_type == "function_call":
            tool_calls.append(_convert_value(item))
        elif item_type == "reasoning":
            reasoning.extend(_extract_reasoning(item))
    return text_outputs, tool_calls, reasoning


def _extract_text_outputs(item: object) -> list[str]:
    collected: list[str] = []
    content_items = cast("Sequence[Any]", getattr(item, "content", []) or [])
    for content_item in content_items:
        text = getattr(content_item, "text", None)
        if isinstance(text, str):
            collected.append(text)
    return collected


def _extract_reasoning(item: object) -> list[str]:
    collected: list[str] = []
    content = cast("Sequence[Any]", getattr(item, "content", None) or [])
    summary = cast("Sequence[Any]", getattr(item, "summary", None) or [])
    for reasoning_item in content or summary:
        text = getattr(reasoning_item, "text", None)
        if isinstance(text, str):
            collected.append(text)
    return collected


def _build_response_outputs(
    text_outputs: list[str], tool_calls: list[object], reasoning: list[str],
) -> NormalizedOutputs:
    result: dict[str, object] = {}
    if text_outputs:
        result["text"] = "".join(text_outputs)
    if tool_calls:
        result["tool_calls"] = tool_calls
    if reasoning:
        result["reasoning_content"] = "".join(reasoning)
    if not result:
        return []
    if len(result) == 1 and "text" in result:
        return [cast("str", result["text"])]
    return [result]


def _extract_content(outputs: NormalizedOutputs | str) -> str:
    if isinstance(outputs, str):
        return outputs
    if not outputs:
        return ""
    first_item = outputs[0]
    if isinstance(first_item, str):
        return first_item
    return _extract_from_mapping(first_item)


def _extract_from_mapping(mapping: Mapping[str, object]) -> str:
    for key in ("text", "reasoning_content"):
        value = mapping.get(key)
        if isinstance(value, str):
            return value
    return str(mapping)


def _extract_model(response: Any, fallback: str) -> str:
    model_attr = getattr(response, "model", None)
    return model_attr if isinstance(model_attr, str) else fallback


def _dump_raw(response: object, outputs: object) -> dict[str, object]:
    return {
        "response": _convert_value(response),
        "outputs": _convert_value(outputs),
    }


def _convert_value(value: Any) -> object:
    if isinstance(value, Mapping):
        mapping_value = cast("Mapping[Any, object]", value)
        return {str(key): _convert_value(item) for key, item in mapping_value.items()}
    if isinstance(value, (list, tuple, set)):
        sequence_value = cast("Sequence[Any]", value)
        return [_convert_value(item) for item in sequence_value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _convert_value(model_dump())
    if hasattr(value, "__dict__"):
        return _convert_value(vars(value))
    return value


def _default_lm_factory(**kwargs: Any) -> SupportsDsPyLM:
    return cast("SupportsDsPyLM", dspy.LM(**kwargs))
