from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from various_llm_benchmark.llm.protocol import LLMClient
from various_llm_benchmark.models import ChatMessage, ImageInput, LLMResponse

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.responses import ResponseInputParam


ReasoningEffort = Literal["none", "low", "medium", "high"]
Verbosity = Literal["low", "medium", "high"]


class ReasoningParam(TypedDict):
    """Nested reasoning configuration for Responses API."""

    effort: ReasoningEffort


class TextParam(TypedDict, total=False):
    """Nested text configuration for Responses API."""

    verbosity: Verbosity


class OpenAIRequestParams(TypedDict, total=False):
    """Payload fields for OpenAI Responses API requests."""

    model: str
    input: ResponseInputParam | str
    temperature: float
    reasoning: ReasoningParam
    text: TextParam


def _create_response(client: OpenAI, params: OpenAIRequestParams) -> Any:
    create = cast("Any", client.responses.create)
    return create(**params)


class OpenAILLMClient(LLMClient):
    """Adapter for OpenAI Responses API."""

    def __init__(self, client: OpenAI, default_model: str, *, temperature: float = 0.7) -> None:
        """Create a client wrapper with defaults."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Verbosity | None = None,
    ) -> LLMResponse:
        """Generate a completion without history."""
        completion = _create_response(
            self._client,
            _build_request_kwargs(
                model or self._default_model,
                prompt,
                self._temperature,
                reasoning_effort,
                verbosity,
            ),
        )
        content = _extract_content(completion.output)
        return LLMResponse(content=content, model=completion.model, raw=completion.model_dump())

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Verbosity | None = None,
    ) -> LLMResponse:
        """Generate a completion using chat messages."""
        openai_messages: list[dict[str, str]] = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        openai_input = cast("ResponseInputParam", openai_messages)
        completion = _create_response(
            self._client,
            _build_request_kwargs(
                model or self._default_model,
                openai_input,
                self._temperature,
                reasoning_effort,
                verbosity,
            ),
        )
        content = _extract_content(completion.output)
        return LLMResponse(content=content, model=completion.model, raw=completion.model_dump())

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Verbosity | None = None,
    ) -> LLMResponse:
        """Generate a response that combines text prompts and image data."""
        text_prompt = _merge_prompt(prompt, system_prompt)
        openai_input = cast(
            "ResponseInputParam",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text_prompt},
                        {"type": "input_image", "image_url": image.as_data_url()},
                    ],
                },
            ],
        )
        completion = _create_response(
            self._client,
            _build_request_kwargs(
                model or self._default_model,
                openai_input,
                self._temperature,
                reasoning_effort,
                verbosity,
            ),
        )
        content = _extract_content(completion.output)
        return LLMResponse(content=content, model=completion.model, raw=completion.model_dump())


def _build_request_kwargs(
    model: str,
    input_value: ResponseInputParam | str,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    verbosity: Verbosity | None,
) -> OpenAIRequestParams:
    request: OpenAIRequestParams = {
        "model": model,
        "input": input_value,
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}
    if verbosity is not None:
        request["text"] = {"verbosity": verbosity}
    return request


def _extract_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return _extract_from_mapping_content(cast("Mapping[str, Any]", content))

    if isinstance(content, list):
        return "".join(_extract_content(item) for item in cast("list[Any]", content))

    extracted_text = _extract_text_attribute(content)
    if extracted_text is not None:
        return extracted_text

    nested_content = getattr(content, "content", None)
    if nested_content is not None:
        return _extract_content(nested_content)

    return str(content)


def _extract_from_mapping_content(content: Mapping[str, Any]) -> str:
    typed_content: dict[str, Any] = dict(content)
    text_value: Any = typed_content.get("text")
    if isinstance(text_value, Mapping):
        nested_mapping = cast("Mapping[str, Any]", text_value)
        nested_text: Any = nested_mapping.get("value")
        if nested_text is not None:
            return str(nested_text)
    elif text_value is not None:
        return str(text_value)

    nested_content: Any = typed_content.get("content")
    if nested_content is not None:
        return _extract_content(nested_content)
    return str(typed_content)


def _extract_text_attribute(content: Any) -> str | None:
    text_attr = getattr(content, "text", None)
    if text_attr is None:
        return None
    value = getattr(text_attr, "value", None)
    return value if isinstance(value, str) else str(text_attr)


def extract_openai_content(content: Any) -> str:
    """Public wrapper for parsing OpenAI response content."""
    return _extract_content(content)


def _merge_prompt(prompt: str, system_prompt: str | None) -> str:
    if not system_prompt:
        return prompt
    return f"{system_prompt}\n\n{prompt}"
