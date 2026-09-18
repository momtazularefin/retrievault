from retrievault.config import Settings

# Anthropic bills cache writes (5-minute TTL) at 1.25x and cache reads at 0.1x the input price.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def synthesis_cost_usd(
    settings: Settings,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Estimated synthesis cost. ``input_tokens`` excludes cached tokens, so all four count."""
    input_rate = settings.synthesis_input_usd_per_mtok / 1_000_000
    output_rate = settings.synthesis_output_usd_per_mtok / 1_000_000
    return (
        input_tokens * input_rate
        + cache_creation_input_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        + cache_read_input_tokens * input_rate * CACHE_READ_MULTIPLIER
        + output_tokens * output_rate
    )
