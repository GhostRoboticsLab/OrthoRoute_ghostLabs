#!/usr/bin/env python3
"""Benchmark a full board route on the selected backend (plan Phase 4/5).

Reports lattice/graph scale, per-iteration progress, wall time, and the
§C2 oracle verdict. Backend comes from ORTHO_BACKEND (default: probe).

Usage: ORTHO_BACKEND=metal .venv/bin/python tools/bench_route.py BOARD [max_iters]
"""

import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.getLogger().setLevel(logging.WARNING)

from orthoroute.algorithms.manhattan.route_oracle import validate_routing  # noqa: E402
from orthoroute.algorithms.manhattan.unified_pathfinder import (  # noqa: E402
    PathFinderConfig,
    UnifiedPathFinder,
)
from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser  # noqa: E402


def main() -> int:
    board_file = sys.argv[1]
    max_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    board = KiCadFileParser().load_board(board_file)
    if board is None:
        print("FAILED to load board", file=sys.stderr)
        return 1
    print(f"board: {board.name} nets={len(board.nets)} "
          f"layers={board.layer_count}", flush=True)

    config = PathFinderConfig()
    if max_iters:
        config.max_iterations = max_iters
    pf = UnifiedPathFinder(config=config, use_gpu=False)

    t0 = time.perf_counter()
    pf.initialize_graph(board)
    print(f"backend={pf.gpu_backend} nodes={pf.lattice.num_nodes:,} "
          f"edges={len(pf.graph.indices):,} "
          f"init={time.perf_counter() - t0:.1f}s", flush=True)

    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()
    t1 = time.perf_counter()
    print(f"escapes+runtime={t1 - t0:.1f}s portals={len(pf.portals)}", flush=True)

    iters = []

    def cb(it, *_args):
        iters.append(it)
        if it % 10 == 0 or it <= 3:
            print(f"  iter {it}: {time.perf_counter() - t1:.0f}s elapsed", flush=True)

    result = pf.route_multiple_nets(board.nets, iteration_cb=cb)
    t2 = time.perf_counter()

    report = validate_routing(pf, board)
    print(f"\nRESULT backend={pf.gpu_backend}")
    print(f"  routed={report.routed_nets}/{report.eligible_nets} eligible "
          f"({len(pf.net_paths)} paths total)")
    print(f"  iterations={iters[-1] if iters else 0} "
          f"converged={result.get('converged')} "
          f"overuse={report.overuse_total}")
    print(f"  route_time={t2 - t1:.1f}s total={t2 - t0:.1f}s")
    print(f"  gpu_fastpath_failures={pf.gpu_fastpath_failures}")
    print(f"  oracle: {report.summary()}")
    for e in report.errors[:5]:
        print(f"  oracle-error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
