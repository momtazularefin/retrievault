import re
from dataclasses import dataclass, field
from typing import Any

from retrievault.synthesize.prompt import REFUSAL_OPENING

# Matches [S1] and grouped forms such as [S1, S3] or [S1; S3].
_GROUP = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\]")
_NUMBER = re.compile(r"\d+")


@dataclass
class CitationCheck:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    refused: bool = False

    @property
    def uncited(self) -> bool:
        return not self.refused and not self.citations


def normalize_labels(answer: str) -> str:
    """Rewrite grouped labels like [S1, S3] as [S1][S3] so every label is individually linkable."""

    def split(match: re.Match) -> str:
        return "".join(f"[S{number}]" for number in _NUMBER.findall(match.group(0)))

    return _GROUP.sub(split, answer)


def is_refusal(answer: str) -> bool:
    return answer.strip().lstrip("\"'“*_ ").startswith(REFUSAL_OPENING)


def check_citations(answer: str, chunks: list[dict[str, Any]]) -> CitationCheck:
    """Map [S#] labels to chunks by position; labels outside 1..len(chunks) are invalid.

    Citations keep the order of first appearance. A refusal carries no citations even if the
    model added labels to it.
    """
    normalized = normalize_labels(answer)
    if is_refusal(normalized):
        return CitationCheck(answer=normalized, refused=True)

    citations: list[dict[str, Any]] = []
    invalid: list[str] = []
    seen: set[int] = set()
    for number in (int(n) for n in re.findall(r"\[S(\d+)\]", normalized)):
        if number in seen:
            continue
        seen.add(number)
        label = f"[S{number}]"
        if not 1 <= number <= len(chunks):
            invalid.append(label)
            continue
        chunk = chunks[number - 1]
        citations.append(
            {
                "label": label,
                "file_path": chunk.get("file_path", ""),
                "start_line": chunk.get("start_line", 1),
                "end_line": chunk.get("end_line", 1),
                "symbol_name": chunk.get("symbol_name", ""),
                "symbol_type": chunk.get("symbol_type", ""),
                "github_url": chunk.get("github_url", ""),
                "snippet": chunk.get("code", ""),
            }
        )
    return CitationCheck(answer=normalized, citations=citations, invalid_labels=invalid)


def correction_message(check: CitationCheck, chunk_count: int) -> str | None:
    """Feedback for one corrective retry, or None when the answer needs no correction."""
    if check.invalid_labels:
        return (
            f"You cited {', '.join(check.invalid_labels)}, which do not exist; the valid labels "
            f"are [S1] to [S{chunk_count}]. Rewrite the answer using only those labels."
        )
    if check.uncited:
        return (
            "Your answer has no [S#] citations. Cite the chunk that supports each claim, or, if "
            f'the sources do not contain the answer, reply starting with "{REFUSAL_OPENING}"'
        )
    return None
