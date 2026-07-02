"""MetalDijkstra correctness vs CPU Dijkstra (plan §6 Phase 4).

Engine-level oracle per plan §B4: compare path COSTS and legality, never
exact node sequences (GPU wavefront tie-breaking differs legitimately).
Skipped wholesale when MLX/Metal is unavailable so the suite still runs
on CUDA/CI hosts.
"""

import heapq

import numpy as np
import pytest

from orthoroute.algorithms.manhattan import backends

if not backends.metal_available():
    pytest.skip("MLX/Metal unavailable", allow_module_level=True)

from orthoroute.algorithms.manhattan.pathfinder.metal_dijkstra import (  # noqa: E402
    MetalDijkstra,
    _sssp,
)


def random_csr(n, avg_deg, seed):
    """Random directed graph with strictly positive float32 weights."""
    rng = np.random.default_rng(seed)
    m = n * avg_deg
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    w = (0.1 + rng.random(len(src))).astype(np.float32)
    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.uint32)
    np.add.at(indptr, src + 1, 1)
    return np.cumsum(indptr, dtype=np.uint32), dst.astype(np.uint32), w


def cpu_dijkstra(indptr, indices, w, sources, n, blocked=None):
    dist = np.full(n, np.inf)
    pq = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, int(s)))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            v = int(indices[j])
            if blocked is not None and blocked[v]:
                continue
            nd = d + float(np.float32(np.float32(d) + w[j]) - np.float32(d)) if False else d + float(w[j])
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


class FakeGraph:
    def __init__(self, indptr, indices):
        self.indptr = indptr
        self.indices = indices


class TestSSSPCore:
    def test_distances_match_cpu(self):
        n = 5000
        indptr, indices, w = random_csr(n, 6, seed=1)
        ref = cpu_dijkstra(indptr, indices, w, [0], n)
        dist, parents, waves = _sssp(indptr, indices, w, [0])
        reach = np.isfinite(ref)
        assert np.isfinite(dist[reach]).all()
        rel = np.abs(dist[reach] - ref[reach]) / np.maximum(ref[reach], 1e-9)
        assert rel.max() < 1e-5

    def test_multi_source(self):
        n = 3000
        indptr, indices, w = random_csr(n, 5, seed=2)
        sources = [0, 17, 999]
        ref = cpu_dijkstra(indptr, indices, w, sources, n)
        dist, _, _ = _sssp(indptr, indices, w, sources)
        reach = np.isfinite(ref)
        rel = np.abs(dist[reach] - ref[reach]) / np.maximum(ref[reach], 1e-9)
        assert rel.max() < 1e-5

    def test_parents_form_valid_shortest_tree(self):
        n = 4000
        indptr, indices, w = random_csr(n, 6, seed=3)
        dist, parents, _ = _sssp(indptr, indices, w, [0])
        # Every reached non-source node must have a parent whose edge
        # exactly achieves its distance (float32 arithmetic).
        reached = np.where(np.isfinite(dist) & (dist > 0))[0]
        checked = 0
        for v in reached[:500]:
            u = int(parents[v])
            assert u != 0xFFFFFFFF, f"node {v} reached but parentless"
            row = slice(int(indptr[u]), int(indptr[u + 1]))
            dsts = indices[row]
            ws = w[row]
            match = dsts == v
            assert match.any()
            best = np.float32(dist[u]) + ws[match]
            assert np.any(best.astype(np.float32) == np.float32(dist[v]))
            checked += 1
        assert checked > 0

    def test_bitmap_blocks_nodes(self):
        # Line graph 0-1-2-3-4 plus expensive bypass 0-5-4; block node 2.
        edges = sorted([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
                        (0, 5, 10.0), (5, 4, 10.0)])  # CSR needs src order
        n = 6
        src = np.array([e[0] for e in edges])
        dst = np.array([e[1] for e in edges], dtype=np.uint32)
        w = np.array([e[2] for e in edges], dtype=np.float32)
        indptr = np.zeros(n + 1, dtype=np.uint32)
        np.add.at(indptr, src + 1, 1)
        indptr = np.cumsum(indptr, dtype=np.uint32)

        bitmap = np.full(1, 0xFFFFFFFF, dtype=np.uint32)
        bitmap[0] &= ~np.uint32(1 << 2)  # clear node 2
        dist, parents, _ = _sssp(indptr, dst, w, [0], allowed_bitmap=bitmap)
        assert dist[4] == pytest.approx(20.0)  # forced onto the bypass
        assert not np.isfinite(dist[2])


@pytest.fixture(scope="module")
def solver():
    n = 8000
    indptr, indices, w = random_csr(n, 6, seed=4)
    solver = MetalDijkstra(FakeGraph(indptr, indices))
    solver._test_w = w
    return solver


class TestKernelProviderSurface:

    def test_fullgraph_seeds_path_cost_matches_cpu(self, solver):
        w = solver._test_w
        n = solver.num_nodes
        ref = cpu_dijkstra(solver.indptr, solver.indices, w, [0, 5], n)
        targets = np.array([n - 1, n - 2], dtype=np.int32)
        path = solver.find_path_fullgraph_gpu_seeds(
            w, np.array([0, 5], dtype=np.int32), targets)
        assert path is not None
        assert path[0] in (0, 5)
        assert path[-1] in targets
        # Path cost must equal the CPU-optimal cost of its endpoint.
        cost = 0.0
        for a, b in zip(path, path[1:]):
            row = slice(int(solver.indptr[a]), int(solver.indptr[a + 1]))
            hit = np.where(solver.indices[row] == b)[0]
            assert hit.size, f"edge {a}->{b} not in graph"
            cost += float(w[int(solver.indptr[a]) + int(hit[0])])
        assert cost == pytest.approx(float(ref[path[-1]]), rel=1e-5)

    def test_roi_path(self, solver):
        w = solver._test_w
        n = solver.num_nodes
        roi_nodes = np.arange(0, n // 2, dtype=np.int64)
        global_to_roi = np.full(n, -1, dtype=np.int64)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes))

        src, dst = 0, n // 2 - 1
        blocked = np.zeros(n, bool)
        blocked[n // 2:] = True
        ref = cpu_dijkstra(solver.indptr, solver.indices, w, [src], n,
                           blocked=blocked)
        path = solver.find_path_roi_gpu(src, dst, w, roi_nodes, global_to_roi)
        if not np.isfinite(ref[dst]):
            assert path is None
            return
        assert path is not None
        assert path[0] == src and path[-1] == dst
        assert all(g < n // 2 for g in path)  # confined to ROI
        cost = 0.0
        for a, b in zip(path, path[1:]):
            row = slice(int(solver.indptr[a]), int(solver.indptr[a + 1]))
            hit = np.where(solver.indices[row] == b)[0]
            cost += float(w[int(solver.indptr[a]) + int(hit[0])])
        assert cost == pytest.approx(float(ref[dst]), rel=1e-5)
