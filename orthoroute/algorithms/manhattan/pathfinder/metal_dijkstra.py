"""Metal (Apple Silicon) wavefront SSSP solver via MLX custom kernels.

Implements the KernelProvider surface (see ``backends.py``):
``find_path_fullgraph_gpu_seeds`` and ``find_path_roi_gpu`` — drop-in
equivalents of the CUDA implementations in ``cuda_dijkstra.py``.

Design (validated by tools/spikes/mlx_sssp_spike.py on M4):

- 32-bit atomic relax: non-negative IEEE-754 floats are order-isomorphic
  to their uint32 bit patterns, so ``atomic_fetch_min_explicit(atomic_uint)``
  on ``as_type<uint>(dist)`` IS an atomic float-min — no CAS loop.
- Race-free parents WITHOUT 64-bit atomics (which MLX's Metal JIT target
  does not expose): the relax loop maintains distances only; after the
  frontier empties, one full-edge pass recovers parents by writing any
  predecessor u with dist[u] + w(u->v) == dist[v]. All edge costs are
  strictly positive, so every recovered parent strictly decreases dist and
  the parent forest is acyclic by construction. This sidesteps the CUDA
  implementation's benign-race/packed-key machinery entirely.
- Persistent state mutated in-place through const-casts on kernel inputs;
  each wave returns an activation counter that doubles as a dependency
  token so MLX's lazy graph cannot reorder mutating dispatches.
- Unified memory: CSR arrays and costs arrive as host numpy arrays and are
  copied once per call into MLX buffers; results are read back through
  zero-copy numpy views.

Python 3.9 compatible.
"""

import logging
import time
from typing import List, Optional

import numpy as np

import mlx.core as mx

logger = logging.getLogger(__name__)

INF_BITS = np.uint32(0x7F800000)  # +inf float32; uint order == float order (x >= 0)

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# One thread per node: expand frontier members, atomically min neighbor
# distances (uint-ordered float bits), mark newly improved nodes in the
# next frontier, count activations.
_RELAX_SRC = """
    uint u = thread_position_in_grid.x;
    uint N = params[0];
    uint use_bitmap = params[1];
    (void)token;
    if (u >= N) return;
    if (!(frontier_in[u >> 5] & (1u << (u & 31)))) return;

    device atomic_uint* dist_a = (device atomic_uint*)dist_bits;
    device atomic_uint* fout_a = (device atomic_uint*)frontier_out;

    float du = as_type<float>(dist_bits[u]);
    uint improved = 0;
    for (uint j = indptr[u]; j < indptr[u + 1]; ++j) {
        uint v = indices[j];
        if (use_bitmap && !(allowed[v >> 5] & (1u << (v & 31)))) continue;
        uint nd_bits = as_type<uint>(du + w[j]);
        uint old = atomic_fetch_min_explicit(&dist_a[v], nd_bits,
                                             memory_order_relaxed);
        if (nd_bits < old) {
            uint bit = 1u << (v & 31);
            uint prev = atomic_fetch_or_explicit(&fout_a[v >> 5], bit,
                                                 memory_order_relaxed);
            if (!(prev & bit)) improved++;
        }
    }
    if (improved) {
        atomic_fetch_add_explicit(&count[0], improved, memory_order_relaxed);
    }
"""

# Post-convergence parent recovery: any u with dist[u] + w == dist[v] is a
# valid shortest-path predecessor of v; plain last-writer-wins stores are
# safe because every candidate is equally valid and costs > 0 forbid ties
# from forming cycles.
_PARENT_SRC = """
    uint u = thread_position_in_grid.x;
    uint N = params[0];
    uint use_bitmap = params[1];
    (void)token;
    if (u >= N) return;
    uint du_bits = dist_bits[u];
    if (du_bits >= 0x7F800000u) return;  // unreached
    if (use_bitmap && !(allowed[u >> 5] & (1u << (u & 31)))) return;

    device atomic_uint* par_a = (device atomic_uint*)parent;
    float du = as_type<float>(du_bits);
    for (uint j = indptr[u]; j < indptr[u + 1]; ++j) {
        uint v = indices[j];
        if (as_type<uint>(du + w[j]) == dist_bits[v]) {
            atomic_store_explicit(&par_a[v], u, memory_order_relaxed);
        }
    }
"""

_relax_kernel = mx.fast.metal_kernel(
    name="ortho_relax",
    input_names=["params", "token", "indptr", "indices", "w",
                 "allowed", "dist_bits", "frontier_in", "frontier_out"],
    output_names=["count"],
    source=_RELAX_SRC,
    header=_HEADER,
    atomic_outputs=True,
)

_parent_kernel = mx.fast.metal_kernel(
    name="ortho_parents",
    input_names=["params", "token", "indptr", "indices", "w",
                 "allowed", "dist_bits", "parent"],
    output_names=["count"],
    source=_PARENT_SRC,
    header=_HEADER,
    atomic_outputs=True,
)

# Waves dispatched blind between frontier-empty readbacks (spike: readback
# every wave costs ~2.4x vs K=8..32 at 10M nodes; K=8 captures most of it
# while wasting at most 7 empty dispatches).
_CHECK_EVERY = 8
NO_PARENT = np.uint32(0xFFFFFFFF)


def _sssp(indptr, indices, weights, sources, allowed_bitmap=None):
    """Full SSSP; returns (dist float32 array, parent uint32 array).

    Args:
        indptr, indices, weights: CSR graph (host numpy; int/uint32, float32).
        sources: iterable of source node ids (dist 0).
        allowed_bitmap: optional uint32 bitmask over nodes; when given,
            relaxation into cleared nodes is blocked (sources should already
            be force-allowed by the caller).
    """
    n = len(indptr) - 1
    # MLX places inputs of <= 7 uint32 elements in the constant address
    # space, where the device-space atomic const-cast cannot compile; pad
    # every mutated buffer to >= 8 elements (extra elements are inert:
    # the kernels guard on u >= N and never set bits past node n).
    words = max((n + 31) // 32, 8)

    use_bitmap = allowed_bitmap is not None
    if use_bitmap:
        allowed = mx.array(np.ascontiguousarray(allowed_bitmap, dtype=np.uint32))
    else:
        allowed = mx.array(np.zeros(1, dtype=np.uint32))  # unread when flag=0

    params = mx.array(np.array([n, 1 if use_bitmap else 0], dtype=np.uint32))
    indptr_d = mx.array(np.ascontiguousarray(indptr, dtype=np.uint32))
    indices_d = mx.array(np.ascontiguousarray(indices, dtype=np.uint32))
    w_d = mx.array(np.ascontiguousarray(weights, dtype=np.float32))

    dist0 = np.full(max(n, 8), INF_BITS, dtype=np.uint32)
    f0 = np.zeros(words, dtype=np.uint32)
    for s in sources:
        s = int(s)
        dist0[s] = 0
        f0[s >> 5] |= np.uint32(1) << np.uint32(s & 31)
    dist_bits = mx.array(dist0)
    frontier = [mx.array(f0), mx.zeros((words,), dtype=mx.uint32)]
    mx.eval(dist_bits, frontier[0], frontier[1], indptr_d, indices_d, w_d, allowed)

    token = params  # initial dependency token
    waves = 0
    pending = []
    max_waves = n + 1  # every wave that continues must activate >= 1 node
    while waves < max_waves:
        fin = frontier[waves % 2]
        fout = mx.zeros((words,), dtype=mx.uint32)
        frontier[(waves + 1) % 2] = fout
        (count,) = _relax_kernel(
            inputs=[params, token, indptr_d, indices_d, w_d,
                    allowed, dist_bits, fin, fout],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
            init_value=0,
        )
        token = count
        pending.append(count)
        waves += 1
        if len(pending) >= _CHECK_EVERY:
            mx.eval(pending[-1])
            if int(pending[-1].item()) == 0:
                break
            pending = []

    # Race-free parent recovery over the converged distance field.
    parent = mx.array(np.full(max(n, 8), NO_PARENT, dtype=np.uint32))
    (tok,) = _parent_kernel(
        inputs=[params, token, indptr_d, indices_d, w_d,
                allowed, dist_bits, parent],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.uint32],
        init_value=0,
    )
    mx.eval(tok, dist_bits, parent)

    dist = np.array(dist_bits, copy=False).view(np.float32)[:n]
    parents = np.array(parent, copy=False)[:n]
    return dist, parents, waves


def _backtrace(parents, dist, target, n):
    """Walk parents from target to a dist==0 source; returns path or None."""
    path = [int(target)]
    v = int(target)
    for _ in range(n):
        if dist[v] == 0.0:
            path.reverse()
            return path
        u = int(parents[v])
        if u == 0xFFFFFFFF or u == v:
            logger.warning("[METAL] Backtrace hit missing parent at node %d", v)
            return None
        path.append(u)
        v = u
    logger.warning("[METAL] Backtrace exceeded %d nodes (cycle?)", n)
    return None


class MetalDijkstra:
    """KernelProvider implementation on MLX/Metal (CUDA-equivalent surface)."""

    def __init__(self, graph=None, lattice=None):
        self.graph = graph
        self.lattice = lattice
        indptr = graph.indptr
        indices = graph.indices
        self.indptr = np.asarray(indptr.get() if hasattr(indptr, "get") else indptr)
        self.indices = np.asarray(indices.get() if hasattr(indices, "get") else indices)
        self.num_nodes = len(self.indptr) - 1
        self.plane_size = (lattice.x_steps * lattice.y_steps) if lattice else None
        logger.info(f"[METAL] MetalDijkstra ready: {self.num_nodes:,} nodes, "
                    f"{len(self.indices):,} edges on {mx.default_device()}")

    # -- KernelProvider surface -----------------------------------------

    def find_path_fullgraph_gpu_seeds(self, costs, src_seeds, dst_targets,
                                      ub_hint=None, *, allowed_bitmap=None,
                                      use_bitmap=False):
        """Multi-source/multi-sink SSSP on the full graph (CUDA-equivalent).

        Args:
            costs: per-edge cost array (host numpy under the Metal backend).
            src_seeds: np.int32 node ids seeded at distance 0.
            dst_targets: np.int32 candidate target node ids (best dist wins).
            ub_hint: ignored (run-to-convergence; strictly conservative).
            allowed_bitmap: optional uint32 owner-aware node bitmap.
            use_bitmap: bitmap enable flag (implied by allowed_bitmap).

        Returns:
            Path as list of global node ids, or None.
        """
        if len(src_seeds) == 0 or len(dst_targets) == 0:
            return None
        costs_np = np.asarray(costs.get() if hasattr(costs, "get") else costs,
                              dtype=np.float32)
        bitmap = None
        if use_bitmap and allowed_bitmap is not None:
            bitmap = np.ascontiguousarray(
                np.asarray(allowed_bitmap, dtype=np.uint32)).ravel().copy()
            # Force-allow seeds and targets (mirrors the CUDA behavior).
            for s in np.concatenate([np.asarray(src_seeds, dtype=np.int64),
                                     np.asarray(dst_targets, dtype=np.int64)]):
                bitmap[s >> 5] |= np.uint32(1) << np.uint32(s & 31)

        t0 = time.perf_counter()
        dist, parents, waves = _sssp(self.indptr, self.indices, costs_np,
                                     src_seeds, allowed_bitmap=bitmap)
        targets = np.asarray(dst_targets, dtype=np.int64)
        tdist = dist[targets]
        best = int(np.argmin(tdist))
        if not np.isfinite(tdist[best]):
            logger.info("[METAL-SEEDS] No target reachable "
                        f"({waves} waves, {time.perf_counter() - t0:.3f}s)")
            return None
        path = _backtrace(parents, dist, targets[best], self.num_nodes)
        logger.info(f"[METAL-SEEDS] {waves} waves, "
                    f"{time.perf_counter() - t0:.3f}s, "
                    f"path={'%d nodes' % len(path) if path else 'BROKEN'}")
        return path

    def find_path_roi_gpu(self, src: int, dst: int, costs, roi_nodes,
                          global_to_roi) -> Optional[List[int]]:
        """Single-pair SSSP on an ROI subgraph (SimpleDijkstra-compatible)."""
        roi_nodes = np.asarray(
            roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes,
            dtype=np.int64)
        global_to_roi = np.asarray(
            global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi)
        costs_np = np.asarray(costs.get() if hasattr(costs, "get") else costs,
                              dtype=np.float32)

        roi_src = int(global_to_roi[src])
        roi_dst = int(global_to_roi[dst])
        if roi_src < 0 or roi_dst < 0:
            logger.warning("[METAL-ROI] src or dst not in ROI")
            return None

        roi_indptr, roi_indices, roi_w = self._extract_roi_csr(
            roi_nodes, global_to_roi, costs_np)

        dist, parents, _ = _sssp(roi_indptr, roi_indices, roi_w, [roi_src])
        if not np.isfinite(dist[roi_dst]):
            return None
        local = _backtrace(parents, dist, roi_dst, len(roi_nodes))
        if local is None:
            return None
        return [int(roi_nodes[i]) for i in local]

    # -- helpers ---------------------------------------------------------

    def _extract_roi_csr(self, roi_nodes, global_to_roi, costs):
        """Vectorized ROI CSR extraction (host-side, numpy)."""
        indptr, indices = self.indptr, self.indices
        starts = indptr[roi_nodes].astype(np.int64)
        counts = (indptr[roi_nodes + 1] - indptr[roi_nodes]).astype(np.int64)
        total = int(counts.sum())

        # Global edge ids of every outgoing edge of every ROI node.
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        e_ids = np.repeat(starts, counts) + (np.arange(total) - offsets)

        dst_local = global_to_roi[indices[e_ids]]
        keep = dst_local >= 0
        src_local = np.repeat(np.arange(len(roi_nodes), dtype=np.int64), counts)[keep]

        kept_dst = dst_local[keep].astype(np.uint32)
        kept_w = costs[e_ids[keep]]

        roi_indptr = np.zeros(len(roi_nodes) + 1, dtype=np.uint32)
        np.add.at(roi_indptr, src_local + 1, 1)
        roi_indptr = np.cumsum(roi_indptr, dtype=np.uint32)
        # src_local is already sorted (np.repeat order), so kept arrays are
        # in CSR order without an extra sort.
        return roi_indptr, kept_dst, kept_w
