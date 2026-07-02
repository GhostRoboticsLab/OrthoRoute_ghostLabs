#!/usr/bin/env python3
"""Generate stripped routing test fixtures from finished KiCad boards.

Removes all top-level ``(segment ...)`` and ``(via ...)`` nodes (the routed
copper) while keeping everything else — footprints, nets, zones — byte-for-
byte identical to the source. Stripping is idempotent, so regeneration from
the same source is byte-reproducible.

Usage:
    python tools/make_fixtures.py SRC.kicad_pcb DEST.kicad_pcb
    python tools/make_fixtures.py          # regenerate the TESTBOARD fixtures

The default source boards are the vendor TESTBOARD carriers (finished,
already-routed designs used read-only; they are never modified).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orthoroute.infrastructure.kicad.sexpr import strip_top_level_nodes  # noqa: E402

STRIP_NODES = ("segment", "via")

DEFAULT_SOURCES = {
    "/path/to/local/test-boards/testboard.kicad_pcb":
        REPO_ROOT / "TestBoards" / "testboard" / "testboard-stripped.kicad_pcb",
    "/path/to/local/test-boards/testboard-mini.kicad_pcb":
        REPO_ROOT / "TestBoards" / "testboard" / "testboard-mini-stripped.kicad_pcb",
}


def strip_board(src: Path, dest: Path) -> int:
    """Strip one board file; returns the number of nodes removed."""
    text = src.read_text(encoding="utf-8")
    stripped, removed = strip_top_level_nodes(text, STRIP_NODES)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(stripped, encoding="utf-8")
    # Carry the project file (renamed to match): it holds the real design
    # rules (min track/via sizes), which KiCad DRC and rule-aware emission
    # read — without it DRC falls back to KiCad defaults.
    src_pro = src.with_suffix(".kicad_pro")
    if src_pro.exists():
        dest.with_suffix(".kicad_pro").write_text(
            src_pro.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{src.name} -> {dest.relative_to(Path.cwd()) if dest.is_relative_to(Path.cwd()) else dest}: "
          f"removed {removed} segment/via nodes"
          f"{' (+project file)' if src_pro.exists() else ''}")
    return removed


def main() -> int:
    if len(sys.argv) == 3:
        pairs = {sys.argv[1]: Path(sys.argv[2])}
    elif len(sys.argv) == 1:
        pairs = DEFAULT_SOURCES
    else:
        print(__doc__, file=sys.stderr)
        return 2

    for src, dest in pairs.items():
        src_path = Path(src)
        if not src_path.exists():
            print(f"ERROR: source not found: {src_path}", file=sys.stderr)
            return 1
        strip_board(src_path, Path(dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
