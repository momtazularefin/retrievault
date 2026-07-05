from fastembed import TextEmbedding
import onnxruntime as ort

print("ONNX Runtime available providers:", ort.get_available_providers())
try:
    model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5", providers=["CPUExecutionProvider"])
    print("Success! TextEmbedding initialized with CPUExecutionProvider.")
except Exception as e:
    print("Failed to initialize with providers argument:", type(e), e)
