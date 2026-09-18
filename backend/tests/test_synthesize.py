import pytest

from retrievault.synthesize import graph as graph_module
from retrievault.synthesize.citations import check_citations, correction_message, normalize_labels
from retrievault.synthesize.graph import build_graph
from retrievault.synthesize.llm_client import Generation
from retrievault.synthesize.prompt import REFUSAL_OPENING, SYSTEM_PROMPT, build_user_message

CHUNKS = [
    {
        "file_path": "fastapi/a.py",
        "start_line": 1,
        "end_line": 10,
        "symbol_name": "foo",
        "symbol_type": "function",
        "code": "def foo(): pass",
        "github_url": "https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/a.py#L1-L10",
    },
    {
        "file_path": "fastapi/b.py",
        "start_line": 5,
        "end_line": 9,
        "symbol_name": "Bar.run",
        "symbol_type": "method",
        "code": "    def run(self): ...",
        "context": "class Bar:",
        "part": 2,
        "part_count": 3,
        "github_url": "https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/b.py#L5-L9",
    },
]


def generation(text: str, stop_reason: str = "end_turn") -> Generation:
    return Generation(text, stop_reason, 100, 20, 0, 0)


class ScriptedClient:
    """Returns scripted answers and records every conversation it was sent."""

    def __init__(self, *responses: Generation):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def generate(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


@pytest.fixture
def client(monkeypatch):
    def install(*responses):
        scripted = ScriptedClient(*responses)
        monkeypatch.setattr(graph_module, "get_client", lambda: scripted)
        return scripted

    return install


def test_user_message_lists_sources_before_the_question():
    message = build_user_message("What is foo?", CHUNKS)

    assert "[S1] fastapi/a.py L1-L10 (foo)" in message
    assert "[S2] fastapi/b.py L5-L9 (Bar.run) part 2 of 3" in message
    assert "Enclosed by:\nclass Bar:" in message
    assert message.rstrip().endswith("Question: What is foo?")
    assert "citation_label" not in CHUNKS[0], "prompt building must not mutate chunks"


def test_system_prompt_is_static_and_names_the_refusal_sentence():
    assert REFUSAL_OPENING in SYSTEM_PROMPT
    assert "{" not in SYSTEM_PROMPT


def test_citations_map_by_position_in_first_appearance_order():
    check = check_citations("Bar runs [S2]. Foo exists [S1]. Again [S2].", CHUNKS)

    assert [c["label"] for c in check.citations] == ["[S2]", "[S1]"]
    assert check.citations[0]["symbol_name"] == "Bar.run"
    assert check.citations[1]["github_url"].endswith("a.py#L1-L10")
    assert check.citations[1]["snippet"] == "def foo(): pass"
    assert not check.invalid_labels and not check.refused and not check.uncited


def test_out_of_range_labels_are_invalid_and_valid_ones_are_kept():
    check = check_citations("Real [S1], made up [S7] and [S0].", CHUNKS)

    assert [c["label"] for c in check.citations] == ["[S1]"]
    assert check.invalid_labels == ["[S7]", "[S0]"]
    assert "[S7], [S0]" in correction_message(check, len(CHUNKS))


def test_grouped_labels_are_split_into_individual_labels():
    assert normalize_labels("Both [S1, S2] and [S1; S2].") == "Both [S1][S2] and [S1][S2]."
    check = check_citations("Both [S1, S2].", CHUNKS)
    assert [c["label"] for c in check.citations] == ["[S1]", "[S2]"]


def test_refusal_is_detected_and_carries_no_citations():
    check = check_citations(f"{REFUSAL_OPENING} The sources cover routing only [S1].", CHUNKS)

    assert check.refused
    assert check.citations == []
    assert correction_message(check, len(CHUNKS)) is None


def test_answer_without_citations_is_uncited_and_gets_feedback():
    check = check_citations("Foo does something.", CHUNKS)

    assert check.uncited
    assert "no [S#] citations" in correction_message(check, len(CHUNKS))


async def test_grounded_answer_needs_one_call(client):
    client(generation("Foo exists [S1]."))

    result = await build_graph().ainvoke({"question": "What is foo?", "chunks": CHUNKS})

    assert result["grounding"] == "grounded"
    assert [c["label"] for c in result["citations"]] == ["[S1]"]
    assert result["retries"] == 0 and result["llm_calls"] == 1
    assert result["input_tokens"] == 100 and result["output_tokens"] == 20


async def test_retry_resends_the_question_with_the_correction(client):
    scripted = client(generation("Made up [S9]."), generation("Fixed [S1]."))

    result = await build_graph().ainvoke({"question": "What is foo?", "chunks": CHUNKS})

    assert result["answer"] == "Fixed [S1]."
    assert result["grounding"] == "grounded"
    assert result["retries"] == 1 and result["llm_calls"] == 2
    retry_conversation = scripted.calls[1]
    assert [m["role"] for m in retry_conversation] == ["user", "assistant", "user"]
    assert "Question: What is foo?" in retry_conversation[0]["content"]
    assert retry_conversation[1]["content"] == "Made up [S9]."
    assert "[S9]" in retry_conversation[2]["content"]
    assert result["input_tokens"] == 200


async def test_persistent_invalid_labels_are_flagged_not_hidden(client):
    client(generation("Made up [S9] with real [S1]."), generation("Still made up [S8] and [S1]."))

    result = await build_graph().ainvoke({"question": "What is foo?", "chunks": CHUNKS})

    assert result["grounding"] == "invalid_citations"
    assert result["invalid_labels"] == ["[S8]"]
    assert [c["label"] for c in result["citations"]] == ["[S1]"]
    assert result["retries"] == 1


async def test_uncited_answer_is_retried_then_flagged(client):
    scripted = client(generation("Foo does things."), generation("Foo still does things."))

    result = await build_graph().ainvoke({"question": "What is foo?", "chunks": CHUNKS})

    assert result["grounding"] == "uncited"
    assert len(scripted.calls) == 2


async def test_refusal_is_accepted_without_retry(client):
    scripted = client(generation(f"{REFUSAL_OPENING} Nothing about Celery."))

    result = await build_graph().ainvoke({"question": "Celery?", "chunks": CHUNKS})

    assert result["refused"] is True
    assert result["grounding"] == "refused"
    assert result["citations"] == []
    assert len(scripted.calls) == 1


async def test_truncated_answer_is_flagged_without_retry(client):
    scripted = client(generation("Foo is long and", stop_reason="max_tokens"))

    result = await build_graph().ainvoke({"question": "What is foo?", "chunks": CHUNKS})

    assert result["grounding"] == "truncated"
    assert len(scripted.calls) == 1


async def test_no_chunks_refuses_without_calling_the_model(client):
    scripted = client()

    result = await build_graph().ainvoke({"question": "What is foo?", "chunks": []})

    assert result["refused"] is True
    assert result.get("llm_calls", 0) == 0
    assert scripted.calls == []
