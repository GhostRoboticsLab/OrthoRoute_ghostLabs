"""Backend seam tests (plan §B3 / §6 Phase 3).

Covers backend selection (env override semantics) and the two hazard
fixes: a GPU fast-path error must fall back to CPU routing with a loud
counter, and a backend init failure must downgrade loudly, not silently.
"""

import pytest

from orthoroute.algorithms.manhattan import backends
from orthoroute.algorithms.manhattan.unified_pathfinder import (
    PathFinderConfig,
    UnifiedPathFinder,
)

from conftest import make_two_pad_board


class TestSelectBackend:
    def test_cpu_when_gpu_disabled(self, monkeypatch):
        monkeypatch.delenv("ORTHO_BACKEND", raising=False)
        assert backends.select_backend(use_gpu=False) == "cpu"

    def test_env_override_cpu_wins(self, monkeypatch):
        monkeypatch.setenv("ORTHO_BACKEND", "cpu")
        assert backends.select_backend(use_gpu=True) == "cpu"

    def test_env_override_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("ORTHO_BACKEND", "opencl")
        with pytest.raises(RuntimeError, match="invalid"):
            backends.select_backend(use_gpu=True)

    def test_forced_unavailable_backend_raises(self, monkeypatch):
        # Explicit intent must not be silently downgraded.
        monkeypatch.setenv("ORTHO_BACKEND", "cuda")
        monkeypatch.setattr(backends, "cuda_available", lambda: False)
        with pytest.raises(RuntimeError, match="not available"):
            backends.select_backend(use_gpu=True)

    def test_probe_order_cuda_first(self, monkeypatch):
        monkeypatch.delenv("ORTHO_BACKEND", raising=False)
        monkeypatch.setattr(backends, "cuda_available", lambda: True)
        monkeypatch.setattr(backends, "metal_available", lambda: True)
        assert backends.select_backend(use_gpu=True) == "cuda"

    def test_probe_falls_to_metal(self, monkeypatch):
        monkeypatch.delenv("ORTHO_BACKEND", raising=False)
        monkeypatch.setattr(backends, "cuda_available", lambda: False)
        monkeypatch.setattr(backends, "metal_available", lambda: True)
        assert backends.select_backend(use_gpu=True) == "metal"

    def test_probe_falls_to_cpu(self, monkeypatch):
        monkeypatch.delenv("ORTHO_BACKEND", raising=False)
        monkeypatch.setattr(backends, "cuda_available", lambda: False)
        monkeypatch.setattr(backends, "metal_available", lambda: False)
        assert backends.select_backend(use_gpu=True) == "cpu"


class FaultySolver:
    """KernelProvider double whose fast path always blows up."""

    plane_size = None

    def find_path_fullgraph_gpu_seeds(self, *a, **kw):
        raise RuntimeError("injected GPU fault")

    def find_path_roi_gpu(self, *a, **kw):
        raise RuntimeError("injected GPU fault")


class TestGpuFaultFallback:
    def test_fastpath_error_falls_back_to_cpu(self, monkeypatch):
        """A GPU exception must not mark the net failed (old behavior)."""
        board = make_two_pad_board(layer_count=4)
        config = PathFinderConfig()
        config.portal_x_snap_max = 0.75
        pf = UnifiedPathFinder(config=config, use_gpu=False)
        pf.initialize_graph(board)
        pf.map_all_pads(board)
        pf.precompute_all_pad_escapes(board)
        pf.prepare_routing_runtime()

        # Inject a faulting GPU solver and force fast-path eligibility.
        pf.solver.gpu_solver = FaultySolver()
        monkeypatch.setattr(pf, "_gpu_fastpath_eligible", lambda costs: True)

        pf.route_multiple_nets(board.nets)

        assert pf.gpu_fastpath_failures >= 1, "fault was never exercised"
        path = pf.net_paths.get("TEST_NET", [])
        assert len(path) >= 2, "CPU fallback did not route the net"

    def test_backend_recorded_on_router(self):
        board = make_two_pad_board(layer_count=4)
        pf = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        pf.initialize_graph(board)
        assert pf.gpu_backend == "cpu"
        assert pf.gpu_fastpath_failures == 0
