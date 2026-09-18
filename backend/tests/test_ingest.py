from dataclasses import replace

from retrievault.chunker import Chunk
from retrievault.ingest import chunk_payload, collect_chunks, generate_point_id

CHUNK = Chunk(
    file_path="fastapi/routing.py",
    symbol_name="APIRouter.include_router",
    symbol_type="method",
    start_line=1578,
    end_line=1608,
    code="    def include_router(",
    context="class APIRouter(routing.Router):",
    part=1,
    part_count=9,
)


def test_point_id_is_stable_and_span_specific():
    assert generate_point_id(CHUNK, "0.136.3") == generate_point_id(CHUNK, "0.136.3")
    moved = replace(CHUNK, start_line=1579)
    assert generate_point_id(moved, "0.136.3") != generate_point_id(CHUNK, "0.136.3")
    assert generate_point_id(CHUNK, "0.136.4") != generate_point_id(CHUNK, "0.136.3")


def test_payload_carries_span_context_and_a_line_anchored_url():
    payload = chunk_payload(CHUNK, "fastapi/fastapi", "0.136.3")

    assert payload["file_path"] == "fastapi/routing.py"
    assert (payload["start_line"], payload["end_line"]) == (1578, 1608)
    assert payload["context"] == "class APIRouter(routing.Router):"
    assert (payload["part"], payload["part_count"]) == (1, 9)
    assert payload["commit_tag"] == "0.136.3"
    assert payload["github_url"] == (
        "https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/routing.py#L1578-L1608"
    )


def test_collect_chunks_skips_tests_and_docs_and_uses_archive_relative_paths(tmp_path):
    archive_root = tmp_path / "fastapi-0.136.3"
    package = archive_root / "fastapi"
    (package / "security").mkdir(parents=True)
    (package / "tests").mkdir()
    (package / "docs").mkdir()
    (package / "routing.py").write_text("def route():\n    return 1\n", encoding="utf-8")
    (package / "security" / "http.py").write_text("class HTTPBase:\n    pass\n", encoding="utf-8")
    (package / "tests" / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    (package / "docs" / "conf.py").write_text("X = 1\n", encoding="utf-8")
    (package / "README.md").write_text("# not python\n", encoding="utf-8")

    chunks = collect_chunks(str(package), str(archive_root))

    assert sorted(c.file_path for c in chunks) == ["fastapi/routing.py", "fastapi/security/http.py"]
