from typing import Any

# A fixed opening makes a refusal machine-detectable: the API reports it as `refused` and the
# eval scores refusals on that flag instead of guessing from keywords.
REFUSAL_OPENING = "I could not find this in the retrieved FastAPI source."

SYSTEM_PROMPT = f"""You answer questions about the FastAPI codebase using only the numbered \
source chunks in the user's message. Each chunk is labelled [S1], [S2], and so on, followed by \
its file path, line range, and symbol.

Rules:
1. Use only the source chunks. Do not use outside knowledge of FastAPI, even if you are sure \
of it.
2. Cite every factual claim with the label of the chunk that supports it, placed at the end of \
the sentence. Write each label in its own brackets, for example [S1][S3], never [S1, S3].
3. Only use labels that appear in the message.
4. If the chunks do not contain the answer, reply with exactly this sentence and nothing that \
cites a source: "{REFUSAL_OPENING}" You may add one sentence saying what is missing.
5. If the chunks answer only part of the question, answer that part with citations and say \
which part the sources do not cover.
6. Be concise. Refer to code by name in backticks."""


def source_label(index: int) -> str:
    return f"[S{index}]"


def format_sources(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, 1):
        header = (
            f"{source_label(index)} {chunk.get('file_path', 'unknown')} "
            f"L{chunk.get('start_line', '?')}-L{chunk.get('end_line', '?')} "
            f"({chunk.get('symbol_name', '')})"
        )
        part_count = chunk.get("part_count", 1)
        if part_count and part_count > 1:
            header += f" part {chunk.get('part', 1)} of {part_count}"
        lines = [header]
        if chunk.get("context"):
            lines.append("Enclosed by:\n" + chunk["context"])
        lines.append(f"```python\n{chunk.get('code', '')}\n```")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[dict[str, Any]]) -> str:
    """Sources first, question last, so the question is the final thing the model reads."""
    return f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}"
