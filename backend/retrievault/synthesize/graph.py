import operator
from typing import TypedDict, List, Dict, Any, Annotated

from langgraph.graph import StateGraph, END

from retrievault.synthesize.prompt import build_system_prompt
from retrievault.synthesize.llm_client import AnthropicClient
from retrievault.synthesize.citations import extract_and_validate_citations

class GraphState(TypedDict):
    question: str
    chunks: List[Dict[str, Any]]
    messages: Annotated[list, operator.add]
    answer: str
    citations: List[Dict[str, Any]]
    retries: int
    input_tokens: int
    output_tokens: int

# Initialize a global client or fetch on demand
_client = None

def get_client() -> AnthropicClient:
    global _client
    if _client is None:
        _client = AnthropicClient()
    return _client

async def synthesize_node(state: GraphState):
    question = state.get("question", "")
    chunks = state.get("chunks", [])
    messages = state.get("messages", [])
    
    if not messages:
        messages = [{"role": "user", "content": question}]
        
    system_prompt = build_system_prompt(chunks)
    
    client = get_client()
    response = await client.generate(system_prompt, messages)
    
    new_message = {"role": "assistant", "content": response["content"]}
    
    current_input = state.get("input_tokens", 0)
    current_output = state.get("output_tokens", 0)
    
    return {
        "messages": [new_message],
        "answer": response["content"],
        "input_tokens": current_input + response["input_tokens"],
        "output_tokens": current_output + response["output_tokens"]
    }

async def validate_node(state: GraphState):
    answer = state.get("answer", "")
    chunks = state.get("chunks", [])
    retries = state.get("retries", 0)
    
    citations, error_msg = extract_and_validate_citations(answer, chunks)
    
    if error_msg and retries < 1:
        # Invalid citations, trigger retry
        return {
            "messages": [{"role": "user", "content": error_msg}],
            "retries": retries + 1
        }
    
    # Valid or we ran out of retries (accept as is)
    return {
        "citations": citations
    }

def route_validation(state: GraphState):
    retries = state.get("retries", 0)
    messages = state.get("messages", [])
    if not messages:
        return END
        
    last_msg = messages[-1]
    # If the last message was our automated validation error from user
    if last_msg["role"] == "user" and retries > 0:
        return "synthesize"
    return END

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("validate", validate_node)
    
    workflow.set_entry_point("synthesize")
    workflow.add_edge("synthesize", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_validation,
        {
            "synthesize": "synthesize",
            END: END
        }
    )
    
    return workflow.compile()
