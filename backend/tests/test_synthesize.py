import pytest
from retrievault.synthesize.prompt import build_system_prompt
from retrievault.synthesize.citations import extract_and_validate_citations
from retrievault.synthesize.graph import build_graph

def test_build_system_prompt():
    chunks = [
        {"file_path": "a.py", "start_line": 1, "end_line": 10, "symbol_name": "foo", "code": "def foo(): pass"}
    ]
    prompt = build_system_prompt(chunks)
    assert "[S1] a.py L1-L10 (foo)" in prompt
    assert "def foo(): pass" in prompt
    assert chunks[0]["citation_label"] == "[S1]"

def test_extract_and_validate_citations_valid():
    chunks = [
        {"citation_label": "[S1]", "file_path": "a.py", "start_line": 1, "end_line": 10}
    ]
    answer = "This is the answer [S1]."
    citations, err = extract_and_validate_citations(answer, chunks)
    assert not err
    assert len(citations) == 1
    assert citations[0]["label"] == "[S1]"
    assert citations[0]["file_path"] == "a.py"

def test_extract_and_validate_citations_hallucinated():
    chunks = [
        {"citation_label": "[S1]", "file_path": "a.py", "start_line": 1, "end_line": 10}
    ]
    answer = "This is the answer [S2]."
    citations, err = extract_and_validate_citations(answer, chunks)
    assert "You cited [S2]" in err
    assert not citations

@pytest.mark.asyncio
async def test_langgraph_valid_flow(monkeypatch):
    # Mock AnthropicClient.generate
    async def mock_generate(self, prompt, messages):
        return {
            "content": "Here is the answer [S1].",
            "input_tokens": 10,
            "output_tokens": 5
        }
    
    from retrievault.synthesize.llm_client import AnthropicClient
    monkeypatch.setattr(AnthropicClient, "generate", mock_generate)
    
    graph = build_graph()
    chunks = [
        {"file_path": "a.py", "start_line": 1, "end_line": 10, "code": "def foo(): pass"}
    ]
    
    state = {"question": "What is foo?", "chunks": chunks}
    result = await graph.ainvoke(state)
    
    assert "Here is the answer [S1]." in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["label"] == "[S1]"
    assert result.get("retries", 0) == 0

@pytest.mark.asyncio
async def test_langgraph_retry_flow(monkeypatch):
    call_count = 0
    async def mock_generate(self, prompt, messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"content": "Hallucination [S99].", "input_tokens": 10, "output_tokens": 5}
        else:
            return {"content": "Fixed [S1].", "input_tokens": 10, "output_tokens": 5}
            
    from retrievault.synthesize.llm_client import AnthropicClient
    monkeypatch.setattr(AnthropicClient, "generate", mock_generate)
    
    graph = build_graph()
    chunks = [
        {"file_path": "a.py", "start_line": 1, "end_line": 10, "code": "def foo(): pass"}
    ]
    
    state = {"question": "What is foo?", "chunks": chunks}
    result = await graph.ainvoke(state)
    
    assert result["answer"] == "Fixed [S1]."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["label"] == "[S1]"
    assert result["retries"] == 1
    assert call_count == 2
