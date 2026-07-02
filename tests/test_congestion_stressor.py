"""Congested stressor gate (plan §6 Phase 4): CPU vs Metal under contention.

Synthetic 4-layer board with crossing nets funneled through a shared
channel — the sparse fixture never stresses negotiation, this does. Both
backends must converge to 0 overuse, pass the §C2 oracle, and take a
comparable number of iterations (within the plan's ~20% band, expressed
as a small absolute slack for small iteration counts).
"""

import pytest

from orthoroute.algorithms.manhattan import backends
from orthoroute.algorithms.manhattan.route_oracle import validate_routing
from orthoroute.algorithms.manhattan.unified_pathfinder import (
    PathFinderConfig,
    UnifiedPathFinder,
)

from orthoroute.domain.models.board import Board, Component, Coordinate, Net, Pad

N_NETS = 10


def make_crossing_board():
    """N_NETS nets in an X pattern: left pad i connects to right pad N-1-i.

    All routes must funnel through the board center, forcing edge
    contention on the two inner layers (capacity 1).
    """
    board = Board(id="stress", name="Congestion Stressor")
    board.layer_count = 4
    # 1.6mm (4-step) pad spacing: dense enough to contend for the two
    # inner layers, sparse enough that every pad can claim an escape
    # portal (portal_delta_min is 3 steps).
    for i in range(N_NETS):
        y_left = 1.0 + i * 1.6
        y_right = 1.0 + (N_NETS - 1 - i) * 1.6
        p_left = Coordinate(x=1.0, y=y_left)
        p_right = Coordinate(x=9.0, y=y_right)
        cl = Component(id=f"L{i}", reference=f"L{i}", value="T", footprint="T",
                       position=p_left)
        cr = Component(id=f"R{i}", reference=f"R{i}", value="T", footprint="T",
                       position=p_right)
        pl = Pad(id=f"L{i}p", component_id=cl.id, position=p_left,
                 layer="F.Cu", size=(0.5, 0.5), net_id=None)
        pr = Pad(id=f"R{i}p", component_id=cr.id, position=p_right,
                 layer="F.Cu", size=(0.5, 0.5), net_id=None)
        cl.pads.append(pl)
        cr.pads.append(pr)
        board.add_component(cl)
        board.add_component(cr)
        board.add_net(Net(id=f"NET{i}", name=f"NET{i}", pads=[pl, pr]))
    return board


def route_stressor(backend_env, monkeypatch):
    monkeypatch.setenv("ORTHO_BACKEND", backend_env)
    board = make_crossing_board()
    config = PathFinderConfig()
    config.portal_x_snap_max = 0.75  # margin puts pads half a pitch off-grid
    pf = UnifiedPathFinder(config=config, use_gpu=False)
    pf.initialize_graph(board)
    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()

    iterations = []
    result = pf.route_multiple_nets(
        board.nets, iteration_cb=lambda it, t, v: iterations.append(it))
    return pf, board, result, (iterations[-1] if iterations else 0)


@pytest.fixture(scope="module")
def cpu_run():
    mp = pytest.MonkeyPatch()
    try:
        yield route_stressor("cpu", mp)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def metal_run():
    if not backends.metal_available():
        pytest.skip("MLX/Metal unavailable")
    mp = pytest.MonkeyPatch()
    try:
        yield route_stressor("metal", mp)
    finally:
        mp.undo()


class TestCpuBaseline:
    def test_converges(self, cpu_run):
        pf, board, result, iters = cpu_run
        total, count = pf.accounting.compute_overuse(pf)
        assert total == 0 and count == 0

    def test_oracle_passes(self, cpu_run):
        pf, board, _, _ = cpu_run
        report = validate_routing(pf, board)
        assert report.ok, report.errors[:5]
        assert report.eligible_nets == N_NETS
        assert report.routed_nets == N_NETS, report.unrouted_nets


class TestMetalUnderCongestion:
    def test_converges(self, metal_run):
        pf, board, result, iters = metal_run
        assert pf.gpu_backend == "metal"
        total, count = pf.accounting.compute_overuse(pf)
        assert total == 0 and count == 0

    def test_oracle_passes(self, metal_run):
        pf, board, _, _ = metal_run
        report = validate_routing(pf, board)
        assert report.ok, report.errors[:5]
        assert report.routed_nets == N_NETS, report.unrouted_nets

    def test_no_gpu_faults(self, metal_run):
        pf, _, _, _ = metal_run
        assert pf.gpu_fastpath_failures == 0

    def test_iterations_comparable_to_cpu(self, cpu_run, metal_run):
        _, _, _, cpu_iters = cpu_run
        _, _, _, metal_iters = metal_run
        # Plan band: within ~20%; use max(20%, +3 absolute) so tiny
        # iteration counts don't flap on legitimate tie-break divergence.
        slack = max(round(0.2 * cpu_iters), 3)
        assert metal_iters <= cpu_iters + slack, (
            f"Metal took {metal_iters} iterations vs CPU {cpu_iters}")
