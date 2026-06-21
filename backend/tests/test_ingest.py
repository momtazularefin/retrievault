from retrievault.ingest import generate_point_id
from retrievault.chunker import Chunk

def test_point_id_generation_is_stable():
    chunk = Chunk(
        file_path="fastapi/routing.py",
        symbol_name="APIRoute",
        symbol_type="class",
        start_line=120,
        end_line=168,
        code="class APIRoute:\n    pass"
    )
    
    tag = "0.136.3"
    
    id1 = generate_point_id(chunk, tag)
    id2 = generate_point_id(chunk, tag)
    
    assert id1 == id2
    
def test_chunk_payload_shape():
    chunk = Chunk(
        file_path="fastapi/routing.py",
        symbol_name="APIRoute",
        symbol_type="class",
        start_line=120,
        end_line=168,
        code="class APIRoute:\n    pass"
    )
    
    payload = chunk.to_dict()
    assert payload["file_path"] == "fastapi/routing.py"
    assert payload["symbol_name"] == "APIRoute"
    assert payload["symbol_type"] == "class"
    assert payload["start_line"] == 120
    assert payload["end_line"] == 168
    assert payload["code"] == "class APIRoute:\n    pass"
    assert payload["language"] == "python"
