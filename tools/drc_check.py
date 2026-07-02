#!/usr/bin/env python3
"""§C2 DRC oracle: route a board, write copper back, run KiCad DRC.

Routes the given board headlessly (backend via ORTHO_BACKEND), writes the
emitted geometry into a routed copy with board_writer, loads the copy with
`kicad-cli pcb drc` and reports violation counts by type. Requires a local
KiCad 10 install for kicad-cli (path auto-detected on macOS).

Usage: ORTHO_BACKEND=metal .venv/bin/python tools/drc_check.py [BOARD] [OUT]
"""

import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from orthoroute.algorithms.manhattan.route_oracle import validate_routing  # noqa: E402
from orthoroute.algorithms.manhattan.unified_pathfinder import (  # noqa: E402
    PathFinderConfig,
    UnifiedPathFinder,
)
from orthoroute.infrastructure.kicad.board_writer import write_routed_board  # noqa: E402
from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser  # noqa: E402

DEFAULT_BOARD = REPO_ROOT / "TestBoards" / "testboard" / "testboard-stripped.kicad_pcb"
KICAD_CLI_CANDIDATES = (
    "/Applications/KiCad.app/Contents/MacOS/kicad-cli",
    "kicad-cli",
)


def find_kicad_cli():
    for cand in KICAD_CLI_CANDIDATES:
        path = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if path:
            return path
    return None


def main() -> int:
    board_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BOARD
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        board_file.with_name(board_file.stem + "-routed.kicad_pcb"))

    board = KiCadFileParser().load_board(str(board_file))
    if board is None:
        print("FAILED to load board", file=sys.stderr)
        return 1

    pf = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
    pf.apply_board_rules(board)
    pf.initialize_graph(board)
    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()
    pf.route_multiple_nets(board.nets)
    tracks, vias = pf.emit_geometry(board)
    payload = pf.get_geometry_payload()
    report = validate_routing(pf, board)
    print(f"backend={pf.gpu_backend} routed={report.routed_nets}/"
          f"{report.eligible_nets} tracks={tracks} vias={vias}")
    print(f"oracle: {report.summary()}")

    copper = [l.name for l in getattr(board, "layers", [])
              if l.name.endswith(".Cu")]
    n = write_routed_board(str(board_file), str(out_file),
                           payload.tracks, payload.vias, copper or None)
    # DRC must see the real design rules: carry the project file along.
    src_pro = board_file.with_suffix(".kicad_pro")
    if src_pro.exists():
        out_file.with_suffix(".kicad_pro").write_text(
            src_pro.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {n} geometry nodes -> {out_file}")

    cli = find_kicad_cli()
    if not cli:
        print("kicad-cli not found — DRC skipped", file=sys.stderr)
        return 2

    drc_json = out_file.with_suffix(".drc.json")
    proc = subprocess.run(
        [cli, "pcb", "drc", "--format", "json", "--output", str(drc_json),
         "--severity-error", str(out_file)],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 and not drc_json.exists():
        print(f"kicad-cli drc failed: {proc.stderr[:500]}", file=sys.stderr)
        return 3

    data = json.loads(drc_json.read_text())
    violations = data.get("violations", [])
    by_type = Counter(v.get("type", "?") for v in violations)
    print(f"\nDRC: {len(violations)} error-severity violations "
          f"(unconnected_items={len(data.get('unconnected_items', []))})")
    for vtype, count in by_type.most_common():
        print(f"  {vtype}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
