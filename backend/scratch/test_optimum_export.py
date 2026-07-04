import time
import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_DIR = "scratch/model-onnx"

def export_model():
    print(f"Exporting {MODEL_NAME} to ONNX via Optimum...")
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAME, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Saving ONNX model and tokenizer to {ONNX_DIR}...")
    model.save_pretrained(ONNX_DIR)
    tokenizer.save_pretrained(ONNX_DIR)
    print("Export completed successfully!")

def compare_results():
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    
    pairs = [
        ["Does FastAPI support blueprints?", "FastAPI uses APIRouter to group path operations similar to Flask blueprints."],
        ["Does FastAPI support blueprints?", "This is a random sentence about something else completely."]
    ]
    
    # 1. Run PyTorch CrossEncoder
    print("\n--- Running PyTorch Reranker ---")
    pt_model = CrossEncoder(MODEL_NAME)
    start = time.perf_counter()
    pt_scores = pt_model.predict(pairs).tolist()
    pt_time = time.perf_counter() - start
    print(f"PyTorch Scores: {pt_scores} (Time: {pt_time:.4f}s)")
    
    # 2. Run ONNX Reranker via ORTModel
    print("\n--- Running ONNX Reranker via ORTModel ---")
    session = ORTModelForSequenceClassification.from_pretrained(ONNX_DIR, provider="CPUExecutionProvider")
    start = time.perf_counter()
    
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = session(**inputs)
    onnx_scores = outputs.logits.squeeze(-1).tolist()
    onnx_time = time.perf_counter() - start
    print(f"ONNX Scores: {onnx_scores} (Time: {onnx_time:.4f}s)")
    
    # Compare
    diff = np.abs(np.array(pt_scores) - np.array(onnx_scores))
    print(f"\nMax difference between PyTorch and ONNX scores: {diff.max():.6f}")

if __name__ == "__main__":
    import os
    if not os.path.exists(ONNX_DIR):
        export_model()
    compare_results()
