from retrievault.rerank.reranker import rerank

def test_reranker_reorders_known_case():
    query = "How to declare a path parameter in FastAPI?"
    
    # doc1 is irrelevant, doc2 is highly relevant
    doc1 = {"id": "1", "code": "def solve_math_problem(a, b):\n    return a + b"}
    doc2 = {
        "id": "2", 
        "code": "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items/{item_id}')\ndef read_item(item_id: int):\n    return item_id"
    }
    doc3 = {"id": "3", "code": "print('Hello world!')"}
    
    # Pass them in a suboptimal order
    chunks = [doc1, doc2, doc3]
    
    reranked = rerank(query, chunks, top_k=2)
    
    assert len(reranked) == 2
    # The relevant doc should be pulled to the top
    assert reranked[0]["id"] == "2"
    assert "rerank_score" in reranked[0]
