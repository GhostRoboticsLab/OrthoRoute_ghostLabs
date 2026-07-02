"""GPU backend selection seam (plan §B3).

The router's GPU surface is deliberately tiny — a *solver* object exposing
two methods (the KernelProvider protocol):

- ``find_path_fullgraph_gpu_seeds(costs, src_seeds, dst_targets, ub_hint=None,
  *, allowed_bitmap=None, use_bitmap=False) -> Optional[List[int]]``
  Multi-source/multi-sink SSSP over the full CSR graph.
- ``find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)
  -> Optional[List[int]]``
  Single-pair SSSP on an ROI subgraph (SimpleDijkstra integration point).

(The CUDA implementation's ``find_paths_on_rois`` is internal — only called
from inside ``find_path_roi_gpu``. Via-capacity kernels are NOT part of this
seam: their call sites in ``unified_pathfinder`` already carry complete
vectorized numpy fallbacks, which are the implementation on non-CUDA
backends.)

Selection: ``ORTHO_BACKEND=cuda|metal|cpu`` forces a backend (and fails
loudly if it is unavailable); otherwise probe CUDA (CuPy) first, then Metal
(MLX on macOS), else CPU.

Python 3.9 compatible.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("cuda", "metal", "cpu")


def cuda_available() -> bool:
    """True if CuPy imports and a CUDA device is usable."""
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def metal_available() -> bool:
    """True if MLX imports with a GPU default device (Apple Silicon)."""
    try:
        import mlx.core as mx
        return mx.default_device().type == mx.DeviceType.gpu
    except Exception:
        return False


def select_backend(use_gpu: bool) -> str:
    """Pick the GPU backend: env override first, then probe, else cpu.

    Args:
        use_gpu: The engine's config.use_gpu intent.

    Returns:
        One of 'cuda', 'metal', 'cpu'.

    Raises:
        RuntimeError: If ORTHO_BACKEND forces a backend that is unavailable
            (explicit intent must not be silently downgraded).
    """
    forced = os.environ.get("ORTHO_BACKEND", "").strip().lower()
    if forced:
        if forced not in VALID_BACKENDS:
            raise RuntimeError(
                f"ORTHO_BACKEND={forced!r} invalid; expected one of {VALID_BACKENDS}")
        if forced == "cuda" and not cuda_available():
            raise RuntimeError("ORTHO_BACKEND=cuda but CuPy/CUDA is not available")
        if forced == "metal" and not metal_available():
            raise RuntimeError("ORTHO_BACKEND=metal but MLX/Metal is not available")
        return forced

    if not use_gpu:
        return "cpu"
    if cuda_available():
        return "cuda"
    if metal_available():
        return "metal"
    return "cpu"


def create_gpu_solver(backend: str, graph, lattice) -> Optional[object]:
    """Instantiate the KernelProvider solver for a backend (None for cpu)."""
    if backend == "cuda":
        from .pathfinder.cuda_dijkstra import CUDADijkstra
        return CUDADijkstra(graph, lattice)
    if backend == "metal":
        from .pathfinder.metal_dijkstra import MetalDijkstra
        return MetalDijkstra(graph, lattice)
    return None
