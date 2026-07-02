#!/usr/bin/env python3
"""Golden routing digest for refactor equivalence checks (plan §6 Phase 3).

Routes the stripped TESTBOARD fixture CPU-only through the full live call
sequence and prints a SHA-256 digest over the canonicalized result
(per-net node paths + emitted geometry, rounded). Two runs of the same
code must produce identical digests (seeded RNG, stable sorts); a seam
refactor must not change the digest either.

Usage: .venv/bin/python tools/golden_route.py [board.kicad_pcb]
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from orthoroute.algorithms.manhattan.unified_pathfinder import (  # noqa: E402
    PathFinderConfig,
    UnifiedPathFinder,
)
from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser  # noqa: E402

DEFAULT_BOARD = REPO_ROOT / "TestBoards" / "TestBackplane.kicad_pcb"


def main() -> int:
    board_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_BOARD)
    board = KiCadFileParser().load_board(board_file)
    if board is None:
        print("FAILED to load board", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    pf = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
    pf.initialize_graph(board)
    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()
    pf.route_multiple_nets(board.nets)
    tracks, vias = pf.emit_geometry(board)
    dt = time.perf_counter() - t0

    payload = pf.get_geometry_payload()
    canon = {
        "paths": {net: list(map(int, path))
                  for net, path in sorted(pf.net_paths.items())},
        "tracks": sorted(
            (t["net"], t["layer"], round(t["x1"], 6), round(t["y1"], 6),
             round(t["x2"], 6), round(t["y2"], 6), round(t["width"], 6))
            for t in payload.tracks
        ),
        "vias": sorted(
            (v["net"], round(v["x"], 6), round(v["y"], 6),
             v["from_layer"], v["to_layer"])
            for v in payload.vias
        ),
    }
    blob = json.dumps(canon, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(blob).hexdigest()

    print(f"nets_routed={len(pf.net_paths)} tracks={tracks} vias={vias} "
          f"time={dt:.1f}s")
    print(f"GOLDEN_DIGEST={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
