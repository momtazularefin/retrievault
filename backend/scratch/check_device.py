import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
try:
    import torch_directml
    print("DirectML available:", torch_directml.is_available())
except ImportError:
    print("torch_directml is not installed.")
