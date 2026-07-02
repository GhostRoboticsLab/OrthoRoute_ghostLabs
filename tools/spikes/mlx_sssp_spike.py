#!/usr/bin/env python3
"""Phase 2 de-risking spike: MLX/Metal wavefront SSSP viability (plan §B5).

Answers, in risk order:
1. Can a Python wave-per-dispatch loop on mx.fast.metal_kernel sustain a
   ~10M-node wavefront SSSP, and what does frontier-empty termination cost
   (per-wave host readback vs K-batched blind dispatch)?
2. Does the 32-bit float-min atomic relax (the shipping CUDA design) work
   on Metal? (Non-negative IEEE floats are order-isomorphic to uint32, so
   atomic_fetch_min_explicit(atomic_uint) IS float-min — no CAS loop.)
3. Does the undocumented 64-bit atomic_ulong packed-key (dist<<32|parent)
   fast path compile and produce identical results? (Apple8+/M2+ only.)

Also demonstrates the two host-driver patterns the port needs:
- persistent device buffers mutated in-place via const-cast (CUDA-style),
- explicit dependency chaining (token) so MLX's lazy graph cannot reorder
  blind-batched wave dispatches that mutate the same buffers.

Run: .venv/bin/python tools/spikes/mlx_sssp_spike.py [--nodes 10000000]
"""

import argparse
import heapq
import time

import mlx.core as mx
import numpy as np

INF_BITS = 0x7F800000  # +inf as uint32; uint order == float order for x >= 0

HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# Wavefront relax: one thread per node, scan frontier bitmask.
# All mutation happens through const-casts on *inputs* (persistent buffers);
# the only true output is the atomic newly-activated counter, which doubles
# as the dependency token for chained dispatches.
RELAX_SRC = """
    uint u = thread_position_in_grid.x;
    uint N = params[0];
    (void)token;  // dependency chain only
    if (u >= N) return;
    if (!(frontier_in[u >> 5] & (1u << (u & 31)))) return;

    device atomic_uint* dist_a = (device atomic_uint*)dist_bits;
    device atomic_uint* fout_a = (device atomic_uint*)frontier_out;
    device atomic_uint* par_a  = (device atomic_uint*)parent;

    float du = as_type<float>(dist_bits[u]);
    uint improved = 0;
    for (uint j = indptr[u]; j < indptr[u + 1]; ++j) {
        uint v = indices[j];
        uint nd_bits = as_type<uint>(du + w[j]);
        uint old = atomic_fetch_min_explicit(&dist_a[v], nd_bits,
                                             memory_order_relaxed);
        if (nd_bits < old) {
            atomic_store_explicit(&par_a[v], u, memory_order_relaxed);
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

# 64-bit packed-key variant: one atomic_ulong min carries (dist_bits, parent)
# together — no separate parent store, no benign race.
RELAX64_SRC = """
    uint u = thread_position_in_grid.x;
    uint N = params[0];
    (void)token;
    if (u >= N) return;
    if (!(frontier_in[u >> 5] & (1u << (u & 31)))) return;

    device atomic_ulong* key_a = (device atomic_ulong*)dist_key;
    device atomic_uint* fout_a = (device atomic_uint*)frontier_out;

    ulong ku = atomic_load_explicit(&key_a[u], memory_order_relaxed);
    float du = as_type<float>((uint)(ku >> 32));
    uint improved = 0;
    for (uint j = indptr[u]; j < indptr[u + 1]; ++j) {
        uint v = indices[j];
        ulong nk = (((ulong)as_type<uint>(du + w[j])) << 32) | (ulong)u;
        ulong old = atomic_fetch_min_explicit(&key_a[v], nk,
                                              memory_order_relaxed);
        if (nk < old) {
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


def build_grid_csr(nx, ny, nz, seed=42):
    """6-neighbor 3D grid CSR with mildly varied positive weights (float32)."""
    rng = np.random.default_rng(seed)
    n = nx * ny * nz
    coords = np.arange(n, dtype=np.int64)
    z, rem = np.divmod(coords, nx * ny)
    y, x = np.divmod(rem, nx)

    srcs, dsts = [], []
    for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        ok = (x + dx < nx) & (y + dy < ny) & (z + dz < nz)
        u = coords[ok]
        v = u + dx + dy * nx + dz * nx * ny
        srcs.append(u); dsts.append(v)   # both directions
        srcs.append(v); dsts.append(u)
    src = np.concatenate(srcs)
    dst = np.concatenate(dsts)
    w = (1.0 + 0.5 * rng.random(len(src))).astype(np.float32)

    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.uint32)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr, dtype=np.uint32)
    return indptr, dst.astype(np.uint32), w


def cpu_dijkstra(indptr, indices, w, source, n):
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            nd = d + float(w[j])
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


class MetalSSSP:
    """Wave-per-dispatch SSSP with persistent in-place device buffers."""

    def __init__(self, indptr, indices, w, use64=False):
        self.n = len(indptr) - 1
        self.words = (self.n + 31) // 32
        self.use64 = use64
        self.indptr = mx.array(indptr)
        self.indices = mx.array(indices)
        self.w = mx.array(w)
        self.params = mx.array(np.array([self.n], dtype=np.uint32))
        names = ["params", "token", "indptr", "indices", "w",
                 "dist_key" if use64 else "dist_bits",
                 "frontier_in", "frontier_out"]
        if not use64:
            names.insert(6, "parent")
        self.kernel = mx.fast.metal_kernel(
            name="relax64" if use64 else "relax",
            input_names=names,
            output_names=["count"],
            source=RELAX64_SRC if use64 else RELAX_SRC,
            header=HEADER,
            atomic_outputs=True,
        )

    def _reset(self, source):
        n, words = self.n, self.words
        if self.use64:
            key = np.full(n, (np.uint64(INF_BITS) << np.uint64(32)),
                          dtype=np.uint64)
            key[source] = np.uint64(source)  # dist 0.0, parent self
            self.dist_key = mx.array(key)
        else:
            bits = np.full(n, INF_BITS, dtype=np.uint32)
            bits[source] = 0
            self.dist_bits = mx.array(bits)
            self.parent = mx.array(np.full(n, 0xFFFFFFFF, dtype=np.uint32))
        f0 = np.zeros(words, dtype=np.uint32)
        f0[source >> 5] = 1 << (source & 31)
        self.frontier = [mx.array(f0), mx.zeros((words,), dtype=mx.uint32)]
        mx.eval(self.frontier[0], self.frontier[1],
                self.dist_key if self.use64 else self.dist_bits)

    def _dispatch(self, token, fin, fout):
        state = [self.dist_key] if self.use64 else [self.parent, self.dist_bits]
        if not self.use64:
            inputs = [self.params, token, self.indptr, self.indices, self.w,
                      self.dist_bits, self.parent, fin, fout]
        else:
            inputs = [self.params, token, self.indptr, self.indices, self.w,
                      self.dist_key, fin, fout]
        (count,) = self.kernel(
            inputs=inputs,
            grid=(self.n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
            init_value=0,
        )
        return count

    def run(self, source, check_every=1, max_waves=100000):
        """Returns (waves dispatched, wasted dispatches, distances np array)."""
        self._reset(source)
        token = self.params  # initial dummy dependency
        waves = wasted = 0
        pending = []
        while waves < max_waves:
            fin = self.frontier[waves % 2]
            fout = self.frontier[(waves + 1) % 2]
            # frontier_out must be clean before the wave: zero it in-graph.
            # (fresh zeros; the old buffer of that slot is dropped)
            self.frontier[(waves + 1) % 2] = fout = mx.zeros_like(fout)
            count = self._dispatch(token, fin, fout)
            token = count
            pending.append(count)
            waves += 1
            if len(pending) >= check_every:
                mx.eval(pending[-1])
                recent = [int(c.item()) for c in pending]
                if recent[-1] == 0:
                    # count how many trailing dispatches were empty
                    for r in reversed(recent):
                        if r == 0:
                            wasted += 1
                        else:
                            break
                    break
                pending = []
        if self.use64:
            key = np.array(self.dist_key, copy=False)
            bits = (key >> np.uint64(32)).astype(np.uint32)
        else:
            bits = np.array(self.dist_bits, copy=False)
        dist = bits.view(np.float32).astype(np.float64)
        return waves, wasted, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=10_000_000)
    args = ap.parse_args()

    print(f"MLX {mx.__version__} on {mx.default_device()}")

    # ---- correctness, small graph -------------------------------------
    nx, ny, nz = 64, 64, 8
    indptr, indices, w = build_grid_csr(nx, ny, nz)
    n_small = nx * ny * nz
    t0 = time.perf_counter()
    ref = cpu_dijkstra(indptr, indices, w, 0, n_small)
    t_cpu = time.perf_counter() - t0

    sssp32 = MetalSSSP(indptr, indices, w, use64=False)
    waves, _, dist32 = sssp32.run(0, check_every=1)
    err32 = np.max(np.abs(dist32 - ref) / np.maximum(ref, 1e-9))
    print(f"[32-bit] small {n_small} nodes: waves={waves} "
          f"max_rel_err={err32:.2e} (cpu heapq {t_cpu:.2f}s) "
          f"-> {'OK' if err32 < 1e-5 else 'FAIL'}")

    try:
        sssp64 = MetalSSSP(indptr, indices, w, use64=True)
        waves64, _, dist64 = sssp64.run(0, check_every=1)
        err64 = np.max(np.abs(dist64 - ref) / np.maximum(ref, 1e-9))
        print(f"[64-bit] small: waves={waves64} max_rel_err={err64:.2e} "
              f"-> {'OK' if err64 < 1e-5 else 'FAIL'}")
    except Exception as e:
        print(f"[64-bit] FAILED to compile/run: {type(e).__name__}: {e}")

    # ---- scale: ~args.nodes, wave-dispatch benchmark ------------------
    side = int(round((args.nodes / 40) ** 0.5))
    nx = ny = side
    nz = 40
    indptr, indices, w = build_grid_csr(nx, ny, nz)
    n_big = nx * ny * nz
    print(f"\nbig graph: {nx}x{ny}x{nz} = {n_big:,} nodes, "
          f"{len(indices):,} directed edges")

    sssp = MetalSSSP(indptr, indices, w, use64=False)
    src = n_big // 2
    for check_every, label in ((1, "readback every wave"),
                               (8, "readback every 8"),
                               (32, "readback every 32")):
        t0 = time.perf_counter()
        waves, wasted, dist = sssp.run(src, check_every=check_every)
        dt = time.perf_counter() - t0
        reached = int(np.isfinite(dist).sum())
        print(f"[{label:22s}] waves={waves:4d} wasted={wasted:2d} "
              f"time={dt:6.2f}s  ({1e3 * dt / waves:6.2f} ms/wave) "
              f"reached={reached:,}")

    # ---- empty-frontier dispatch overhead ------------------------------
    empty = mx.zeros((sssp.words,), dtype=mx.uint32)
    token = sssp.params
    mx.eval(empty)
    t0 = time.perf_counter()
    reps = 200
    for _ in range(reps):
        token = sssp._dispatch(token, empty, empty)
    mx.eval(token)
    dt = time.perf_counter() - t0
    print(f"[empty-frontier dispatch] {1e6 * dt / reps:.0f} us/wave "
          f"(pure per-dispatch overhead incl. full-grid bitmask scan)")


if __name__ == "__main__":
    main()
