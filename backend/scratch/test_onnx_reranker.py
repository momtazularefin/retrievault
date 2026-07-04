import os
import time
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_PATH = "scratch/model.onnx"

def export_to_onnx():
    print(f"Loading {MODEL_NAME} for ONNX export...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    # Create dummy inputs
    inputs = tokenizer([["query", "document"]], return_tensors="pt")

    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )
    print("Export completed successfully!")

def compare_results():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Test queries
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
    
    # 2. Run ONNX Reranker
    print("\n--- Running ONNX Reranker ---")
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="np")
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    outputs = session.run(None, onnx_inputs)
    onnx_scores = outputs[0].squeeze(-1).tolist()
    onnx_time = time.perf_counter() - start
    print(f"ONNX Scores: {onnx_scores} (Time: {onnx_time:.4f}s)")
    
    # Assert similarity
    diff = np.abs(np.array(pt_scores) - np.array(onnx_scores))
    print(f"\nMax difference between PyTorch and ONNX scores: {diff.max():.6f}")

if __name__ == "__main__":
    if not os.path.exists(ONNX_PATH):
        export_to_onnx()
    compare_results()
