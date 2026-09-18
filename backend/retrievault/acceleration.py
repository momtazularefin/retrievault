"""Strict ONNX Runtime execution-provider selection.

``ACCELERATION`` names the hardware the ONNX models must run on. A requested accelerator that
is missing raises instead of silently running on the CPU, and that is checked twice: before a
session is created (the provider must be installed) and after (ONNX Runtime can still fall
back to the CPU when a provider fails to initialise, and only warns when it does).
"""

import onnxruntime as ort

CPU_PROVIDER = "CPUExecutionProvider"

# DirectML runs on DirectX 12 GPUs, including integrated ones. It is not an NPU path: on an
# AMD Ryzen AI laptop it runs on the Radeon iGPU, not the XDNA NPU.
GPU_PROVIDERS = ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider")

# AMD's Ryzen AI NPU is reached through the Vitis AI provider (Ryzen AI Software).
NPU_PROVIDERS = ("VitisAIExecutionProvider",)

MODES = ("none", "gpu", "npu")


def get_onnx_providers(acceleration: str) -> list[str]:
    """Return the provider list for an acceleration mode, raising if it is unavailable."""
    mode = acceleration.strip().lower()
    if mode not in MODES:
        raise ValueError(f"Invalid ACCELERATION value {acceleration!r}; expected one of {MODES}.")
    if mode == "none":
        return [CPU_PROVIDER]

    available = ort.get_available_providers()
    candidates = GPU_PROVIDERS if mode == "gpu" else NPU_PROVIDERS
    installed = [provider for provider in candidates if provider in available]
    if not installed:
        raise RuntimeError(
            f"ACCELERATION={mode} requires one of {candidates}, but ONNX Runtime only has "
            f"{available}. Install a matching onnxruntime build or set ACCELERATION=none."
        )
    # One accelerator; ONNX Runtime places operators the accelerator cannot run on the CPU.
    return [installed[0]]


def verify_session(session: ort.InferenceSession, acceleration: str, model_name: str) -> None:
    """Raise if a session did not end up on the provider its acceleration mode requires."""
    expected = get_onnx_providers(acceleration)[0]
    active = session.get_providers()
    if not active or active[0] != expected:
        raise RuntimeError(
            f"{model_name} was expected to run on {expected} but ONNX Runtime activated "
            f"{active}. The provider failed to initialise; refusing to fall back silently."
        )
