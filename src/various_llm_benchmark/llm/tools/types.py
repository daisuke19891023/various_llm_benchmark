"""Typed structures for tool inputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input payload for web search tools."""

    prompt: str = Field(description="ユーザーからの検索依頼")
    model: str | None = Field(default=None, description="明示的に利用するモデル")
    use_light_model: bool | None = Field(
        default=None, description="軽量モデルを利用する場合にtrue",
    )

