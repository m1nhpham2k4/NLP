from __future__ import annotations

from typing import Any, Dict


def resolve_torch_device(requested_device: str = "auto") -> str:
    """Resolve runtime device: CUDA when available, otherwise CPU."""
    requested = (requested_device or "auto").strip().lower()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        cuda_available = False
        mps_available = False

    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda"):
        return requested if cuda_available else "cpu"
    if requested == "mps":
        return "mps" if mps_available else "cpu"
    return "cpu"


def torch_device_status(requested_device: str = "auto") -> Dict[str, Any]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        cuda_name = torch.cuda.get_device_name(0) if cuda_available and cuda_count else None
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception as exc:
        return {
            "requested_device": requested_device,
            "resolved_device": "cpu",
            "cuda_available": False,
            "cuda_count": 0,
            "cuda_name": None,
            "mps_available": False,
            "error": str(exc),
        }

    return {
        "requested_device": requested_device,
        "resolved_device": resolve_torch_device(requested_device),
        "cuda_available": cuda_available,
        "cuda_count": cuda_count,
        "cuda_name": cuda_name,
        "mps_available": mps_available,
        "error": None,
    }
