from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from retrievault.config import get_settings
from retrievault.synthesize.prompt import SYSTEM_PROMPT


@dataclass(frozen=True)
class Generation:
    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class AnthropicClient:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.retrievault_synthesis_model
        self.max_tokens = settings.synthesis_max_tokens

    async def generate(self, messages: list[dict[str, Any]]) -> Generation:
        # No cache_control: every query carries different sources, so a cache write (1.25x the
        # input price) would almost never be read back. The static system prompt is also below
        # the 1,024-token minimum Sonnet 4.6 needs to cache at all.
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=0.0,
        )
        usage = response.usage
        return Generation(
            text="".join(block.text for block in response.content if block.type == "text"),
            stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        )
