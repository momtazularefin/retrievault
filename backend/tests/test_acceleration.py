import pytest

from retrievault import acceleration
from retrievault.acceleration import get_onnx_providers, verify_session


def available(monkeypatch, providers):
    monkeypatch.setattr(acceleration.ort, "get_available_providers", lambda: providers)


def test_none_uses_cpu_without_checking_hardware(monkeypatch):
    available(monkeypatch, [])
    assert get_onnx_providers("none") == ["CPUExecutionProvider"]


def test_gpu_prefers_cuda_then_rocm_then_directml(monkeypatch):
    available(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert get_onnx_providers("gpu") == ["DmlExecutionProvider"]
    available(monkeypatch, ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    assert get_onnx_providers("GPU") == ["CUDAExecutionProvider"]


def test_npu_does_not_accept_directml(monkeypatch):
    # DirectML runs on the GPU; treating it as an NPU would report the wrong hardware.
    available(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="ACCELERATION=npu"):
        get_onnx_providers("npu")
    available(monkeypatch, ["VitisAIExecutionProvider", "CPUExecutionProvider"])
    assert get_onnx_providers("npu") == ["VitisAIExecutionProvider"]


def test_missing_gpu_raises_instead_of_falling_back(monkeypatch):
    available(monkeypatch, ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="ACCELERATION=gpu"):
        get_onnx_providers("gpu")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Invalid ACCELERATION"):
        get_onnx_providers("tpu")


class FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


def test_verify_session_detects_a_silent_cpu_fallback(monkeypatch):
    available(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    verify_session(FakeSession(["DmlExecutionProvider", "CPUExecutionProvider"]), "gpu", "model")
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        verify_session(FakeSession(["CPUExecutionProvider"]), "gpu", "model")
