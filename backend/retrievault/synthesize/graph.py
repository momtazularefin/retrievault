"""Grounded synthesis as a small LangGraph state machine.

synthesize -> validate -> (one corrective retry) -> END

The graph receives the question and the reranked chunks; retrieval and reranking stay outside
it. The conversation in ``messages`` always starts with the user turn that holds the sources and
the question, so a retry resends the whole exchange rather than only the correction.
"""

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from retrievault.synthesize.citations import check_citations, correction_message
from retrievault.synthesize.llm_client import AnthropicClient
from retrievault.synthesize.prompt import REFUSAL_OPENING, build_user_message

MAX_RETRIES = 1


class GraphState(TypedDict, total=False):
    question: str
    chunks: list[dict[str, Any]]
    messages: Annotated[list[dict[str, Any]], operator.add]
    answer: str
    citations: list[dict[str, Any]]
    invalid_labels: list[str]
    refused: bool
    grounding: str
    stop_reason: str | None
    needs_retry: bool
    retries: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


_client: AnthropicClient | None = None


def get_client() -> AnthropicClient:
    global _client
    if _client is None:
        _client = AnthropicClient()
    return _client


async def synthesize_node(state: GraphState) -> dict[str, Any]:
    chunks = state.get("chunks", [])
    if not chunks:
        # Nothing was retrieved, so there is nothing to ground an answer in.
        return {"answer": REFUSAL_OPENING, "stop_reason": None}

    new_messages: list[dict[str, Any]] = []
    history = state.get("messages", [])
    if not history:
        new_messages.append(
            {"role": "user", "content": build_user_message(state["question"], chunks)}
        )
    generation = await get_client().generate(history + new_messages)
    new_messages.append({"role": "assistant", "content": generation.text})

    return {
        "messages": new_messages,
        "answer": generation.text,
        "stop_reason": generation.stop_reason,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "input_tokens": state.get("input_tokens", 0) + generation.input_tokens,
        "output_tokens": state.get("output_tokens", 0) + generation.output_tokens,
        "cache_creation_input_tokens": state.get("cache_creation_input_tokens", 0)
        + generation.cache_creation_input_tokens,
        "cache_read_input_tokens": state.get("cache_read_input_tokens", 0)
        + generation.cache_read_input_tokens,
    }


async def validate_node(state: GraphState) -> dict[str, Any]:
    chunks = state.get("chunks", [])
    retries = state.get("retries", 0)
    check = check_citations(state.get("answer", ""), chunks)
    feedback = correction_message(check, len(chunks))
    truncated = state.get("stop_reason") == "max_tokens"

    if feedback and not truncated and retries < MAX_RETRIES:
        return {
            "messages": [{"role": "user", "content": feedback}],
            "retries": retries + 1,
            "needs_retry": True,
        }

    if check.refused:
        grounding = "refused"
    elif truncated:
        grounding = "truncated"
    elif check.invalid_labels:
        grounding = "invalid_citations"
    elif check.uncited:
        grounding = "uncited"
    else:
        grounding = "grounded"
    return {
        "answer": check.answer,
        "citations": check.citations,
        "invalid_labels": check.invalid_labels,
        "refused": check.refused,
        "grounding": grounding,
        "retries": retries,
        "needs_retry": False,
    }


def route_validation(state: GraphState) -> str:
    return "synthesize" if state.get("needs_retry") else END


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("validate", validate_node)
    workflow.set_entry_point("synthesize")
    workflow.add_edge("synthesize", "validate")
    workflow.add_conditional_edges(
        "validate", route_validation, {"synthesize": "synthesize", END: END}
    )
    return workflow.compile()
