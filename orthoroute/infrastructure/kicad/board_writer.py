"""Write routed geometry back into a .kicad_pcb file (headless commit).

Injects OrthoRoute's emitted tracks/vias as ``(segment ...)`` and
``(via ...)`` nodes before the closing paren of the root ``kicad_pcb``
node, leaving all existing content byte-identical. This gives the
headless/cli path a KiCad-loadable artifact — the write-back equivalent
of the GUI's ``commit_routes()`` — and enables the §C2 DRC oracle via
``kicad-cli pcb drc``.

Net references are written name-only (``(net "NAME")``), the KiCad 10
dialect. KiCad 9 (format >= 20240101) reads name-only refs as well; for
older boards KiCad re-resolves nets on load.

Python 3.9 compatible.
"""

import uuid
from typing import Dict, List, Optional

_NAMESPACE = uuid.UUID("f086b5b2-6d43-41cf-a2a0-8f8f4d6a52a3")


def _uuid_for(kind: str, index: int, payload: str) -> str:
    """Deterministic UUIDv5 so identical routes produce identical files."""
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{index}:{payload}"))


def _fmt(value: float) -> str:
    """KiCad-style compact float formatting (up to 6 decimals, no trailing 0)."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def _escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace('"', '\\"')


def segment_sexpr(track: Dict, index: int) -> str:
    """Render one emitted track dict as a (segment ...) node."""
    payload = (f"{_fmt(track['x1'])},{_fmt(track['y1'])},"
               f"{_fmt(track['x2'])},{_fmt(track['y2'])},{track['layer']}")
    return (
        "\t(segment\n"
        f"\t\t(start {_fmt(track['x1'])} {_fmt(track['y1'])})\n"
        f"\t\t(end {_fmt(track['x2'])} {_fmt(track['y2'])})\n"
        f"\t\t(width {_fmt(track['width'])})\n"
        f"\t\t(layer \"{track['layer']}\")\n"
        f"\t\t(net \"{_escape(track['net'])}\")\n"
        f"\t\t(uuid \"{_uuid_for('segment', index, payload)}\")\n"
        "\t)\n"
    )


def via_sexpr(via: Dict, index: int, copper_layers: Optional[List[str]] = None) -> str:
    """Render one emitted via dict as a (via ...) node.

    Vias not spanning the full outer-layer pair are marked blind/buried
    (``(via blind ...)``) when the board's copper layer order is known.
    """
    from_layer, to_layer = via["from_layer"], via["to_layer"]
    kind = ""
    if copper_layers and len(copper_layers) >= 2:
        outer = {copper_layers[0], copper_layers[-1]}
        if {from_layer, to_layer} != outer:
            kind = " blind"
    payload = f"{_fmt(via['x'])},{_fmt(via['y'])},{from_layer},{to_layer}"
    return (
        f"\t(via{kind}\n"
        f"\t\t(at {_fmt(via['x'])} {_fmt(via['y'])})\n"
        f"\t\t(size {_fmt(via.get('diameter', 0.25))})\n"
        f"\t\t(drill {_fmt(via.get('drill', 0.15))})\n"
        f"\t\t(layers \"{from_layer}\" \"{to_layer}\")\n"
        f"\t\t(net \"{_escape(via['net'])}\")\n"
        f"\t\t(uuid \"{_uuid_for('via', index, payload)}\")\n"
        "\t)\n"
    )


def write_routed_board(src_path: str, dest_path: str,
                       tracks: List[Dict], vias: List[Dict],
                       copper_layers: Optional[List[str]] = None) -> int:
    """Append routed geometry to a board file; returns nodes written.

    Args:
        src_path: The (typically unrouted) source .kicad_pcb.
        dest_path: Destination file (may equal src_path).
        tracks: Emitted track dicts (net/layer/x1/y1/x2/y2/width).
        vias: Emitted via dicts (net/x/y/from_layer/to_layer/diameter/drill).
        copper_layers: Board copper names in stackup order, used to mark
            blind/buried vias; omit to write all vias as through vias.
    """
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()

    close = text.rstrip().rfind(")")
    if close < 0:
        raise ValueError(f"Not an s-expression file: {src_path}")

    parts = [text[:close]]
    for i, track in enumerate(tracks):
        parts.append(segment_sexpr(track, i))
    for i, via in enumerate(vias):
        parts.append(via_sexpr(via, i, copper_layers))
    parts.append(text[close:])

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return len(tracks) + len(vias)
