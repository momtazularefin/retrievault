from dataclasses import dataclass

import onnxruntime as ort
from fastembed import SparseTextEmbedding, TextEmbedding

from retrievault.config import get_settings


def get_onnx_providers(execution_device: str) -> list[str]:
    available = ort.get_available_providers()
    device = execution_device.lower()
    
    if device == "cpu":
        return ["CPUExecutionProvider"]
        
    elif device == "gpu":
        gpu_providers = [p for p in ["CUDAExecutionProvider", "DirectMLExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider"] if p in available]
        if not gpu_providers:
            raise RuntimeError(
                f"GPU execution requested, but no GPU provider (CUDA, DirectML, ROCm) "
                f"is available in ONNX Runtime. Available providers: {available}"
            )
        return gpu_providers
        
    elif device == "npu":
        npu_providers = [p for p in ["DirectMLExecutionProvider", "DmlExecutionProvider", "VitisAIExecutionProvider"] if p in available]
        if not npu_providers:
            raise RuntimeError(
                f"NPU execution requested, but no NPU provider (DirectML, VitisAI) "
                f"is available in ONNX Runtime. Available providers: {available}"
            )
        return npu_providers
        
    else:
        raise ValueError(
            f"Invalid execution_device: '{execution_device}'. "
            "Must be 'cpu', 'gpu', or 'npu'."
        )


@dataclass(frozen=True)
class EncodedQuery:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class QueryEncoder:
    def __init__(
        self,
        dense_model: TextEmbedding | None = None,
        sparse_model: SparseTextEmbedding | None = None,
    ):
        settings = get_settings()
        providers = get_onnx_providers(settings.execution_device)
        self._dense = dense_model or TextEmbedding(model_name=settings.embed_model, providers=providers)
        self._sparse = sparse_model or SparseTextEmbedding(model_name=settings.sparse_model, providers=providers)

    def encode(self, query: str) -> EncodedQuery:
        dense_vec = list(self._dense.embed([query]))[0]
        sparse_vec = list(self._sparse.embed([query]))[0]
        return EncodedQuery(
            dense=dense_vec.tolist(),
            sparse_indices=sparse_vec.indices.tolist(),
            sparse_values=sparse_vec.values.tolist(),
        )
