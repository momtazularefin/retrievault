from typing import List, Dict, Any
from anthropic import AsyncAnthropic
from retrievault.config import get_settings

class AnthropicClient:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.retrievault_synthesis_model

    async def generate(self, system_prompt: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calls Anthropic Claude with the provided system prompt and message history.
        Filters internal LangGraph state messages down to just user/assistant messages.
        """
        anthropic_msgs = []
        for m in messages:
            if m["role"] in ["user", "assistant"]:
                anthropic_msgs.append({"role": m["role"], "content": m["content"]})
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=anthropic_msgs,
            temperature=0.0
        )
        return {
            "content": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": response.usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens or 0,
        }
