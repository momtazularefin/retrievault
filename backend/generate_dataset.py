import os
import json
import uuid
import random
from qdrant_client import QdrantClient
from anthropic import Anthropic

def generate():
    qdrant = QdrantClient("http://localhost:6333")
    anthropic = Anthropic()
    
    collection_name = "retrievault_chunks"
    try:
        count = qdrant.count(collection_name).count
        print(f"Index has {count} chunks")
    except Exception as e:
        print("Qdrant not accessible?", e)
        return

    # Let's scroll to get 25 chunks
    scroll_res, _ = qdrant.scroll(collection_name=collection_name, limit=25, with_payload=True)
    
    dataset = []
    
    # Manual unanswerable
    unanswerable = [
        "How do I deploy FastAPI on AWS Lambda using Zappa?",
        "What is the best way to integrate Django ORM with FastAPI?",
        "Does FastAPI support Flask blueprints?",
        "How to configure Celery broker in FastAPI core?",
        "What is the weather in Tokyo?"
    ]
    for q in unanswerable:
        dataset.append({
            "id": str(uuid.uuid4()),
            "question": q,
            "reference_answer": "I can only answer questions related to the FastAPI codebase. I cannot assist with this.",
            "gold_files": [],
            "gold_symbols": [],
            "category": "refusal"
        })
        
    print(f"Generated {len(dataset)} refusal questions.")
    
    valid_count = 0
    for point in scroll_res:
        if valid_count >= 45:
            break
        
        payload = point.payload
        content = payload.get("content", "")
        file_path = payload.get("file_path", "")
        symbol = payload.get("symbol_name", "")
        
        prompt = f"""
        Given the following source code chunk from FastAPI:
        File: {file_path}
        Symbol: {symbol}
        Code:
        {content[:1000]}
        
        Generate exactly 2 unique questions that can be perfectly answered by this code chunk.
        Provide the output in JSON format as a list of objects with "question" and "reference_answer".
        """
        
        resp = anthropic.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            system="You are an expert developer. Output raw JSON list.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            text = resp.content[0].text
            start = text.find('[')
            end = text.rfind(']') + 1
            items = json.loads(text[start:end])
            
            for item in items:
                if valid_count >= 45:
                    break
                dataset.append({
                    "id": str(uuid.uuid4()),
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "gold_files": [file_path],
                    "gold_symbols": [symbol] if symbol else [],
                    "category": random.choice(["factual", "how-it-works"])
                })
                valid_count += 1
                
        except Exception as e:
            print("Failed to parse JSON for a chunk", e)
            
    print(f"Total dataset size: {len(dataset)}")
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/dataset.jsonl", "w") as f:
        # header
        f.write(json.dumps({
            "corpus_repo": "fastapi/fastapi",
            "corpus_tag": "0.136.3",
            "dataset_version": "1.0",
            "created": "2026-06-26T00:00:00Z"
        }) + "\n")
        
        for d in dataset:
            f.write(json.dumps(d) + "\n")
            
if __name__ == "__main__":
    generate()
