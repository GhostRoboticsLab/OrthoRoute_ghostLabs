"""Tests for the KiCad 10-capable file parser (Phase 1 gate).

Coordinates are asserted against pcbnew 10.0.4 ground truth (the parser
was verified pad-exact, <=1um, on all 329+332 pads of the TESTBOARD boards;
spot checks here pin the footprint transform: rotation and back side).
"""

import pytest

from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser


@pytest.fixture(scope="module")
def main_board(stripped_main_board):
    board = KiCadFileParser().load_board(str(stripped_main_board))
    assert board is not None
    return board


@pytest.fixture(scope="module")
def mini_board(stripped_mini_board):
    board = KiCadFileParser().load_board(str(stripped_mini_board))
    assert board is not None
    return board


class TestKiCad10MainBoard:
    def test_net_count(self, main_board):
        assert len(main_board.nets) == 90

    def test_pad_count(self, main_board):
        assert sum(len(c.pads) for c in main_board.components) == 329

    def test_copper_layer_count(self, main_board):
        # Edge.Cuts must not be counted (the old parser reported 5 here).
        assert main_board.layer_count == 4

    def test_gnd_membership(self, main_board):
        gnd = next(n for n in main_board.nets if n.name == "GND")
        assert len(gnd.pads) == 66

    def test_net_ids_are_names(self, main_board):
        assert all(n.id == n.name for n in main_board.nets)

    def test_back_rot180_pad_position(self, main_board):
        # J1.A5 (USB_CC1), footprint on B.Cu rotated 180deg — pcbnew truth.
        j1 = next(c for c in main_board.components if c.reference == "J1")
        a5 = next(p for p in j1.pads if p.id == "J1_A5")
        assert a5.position.x == pytest.approx(26.25, abs=1e-3)
        assert a5.position.y == pytest.approx(1.025, abs=1e-3)
        assert a5.net_id == "USB_CC1"

    def test_through_hole_pads_have_drill(self, main_board):
        drilled = [p for c in main_board.components for p in c.pads
                   if p.drill_size and p.drill_size > 0]
        assert len(drilled) == 129  # 123 PTH + 6 NPTH


class TestKiCad10MiniBoard:
    def test_counts(self, mini_board):
        assert len(mini_board.nets) == 90
        assert sum(len(c.pads) for c in mini_board.components) == 332

    def test_back_rot0_pad_position(self, mini_board):
        # DP5.1 (NPX_D05), WS2812B on B.Cu unrotated — pcbnew truth.
        dp5 = next(c for c in mini_board.components if c.reference == "DP5")
        p1 = next(p for p in dp5.pads if p.id == "DP5_1")
        assert p1.position.x == pytest.approx(27.335, abs=1e-3)
        assert p1.position.y == pytest.approx(24.575, abs=1e-3)
        assert p1.net_id == "NPX_D05"


LEGACY = """(kicad_pcb (version 20171130) (host pcbnew 5.0)
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (net 0 "")
  (net 1 "GND")
  (net 2 "VCC")
  (module R_0805 (layer F.Cu)
    (at 10.0 20.0 90)
    (fp_text reference "R1" (at 0 0))
    (fp_text value "10k" (at 0 0))
    (pad "1" smd rect (at -1.0 0) (size 1.0 1.3) (layers "F.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 1.0 0) (size 1.0 1.3) (layers "F.Cu") (net 2 "VCC"))
  )
)"""


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    path = tmp_path_factory.mktemp("legacy") / "legacy.kicad_pcb"
    path.write_text(LEGACY, encoding="utf-8")
    return KiCadFileParser().load_board(str(path))


class TestLegacyDialect:
    """KiCad <= 7 files: numbered net table, fp_text reference."""

    def test_module_footprints_supported(self, parsed):
        assert len(parsed.components) == 1
        assert parsed.components[0].reference == "R1"
        assert parsed.components[0].value == "10k"

    def test_numbered_nets_resolved_to_names(self, parsed):
        assert {n.name for n in parsed.nets} == {"GND", "VCC"}
        pads = parsed.components[0].pads
        assert pads[0].net_id == "GND"
        assert pads[1].net_id == "VCC"

    def test_rotated_pad_world_position(self, parsed):
        # Footprint at (10, 20) rot 90 CCW in Y-down frame:
        # pad 1 local (-1, 0) -> world (10, 21).
        p1 = parsed.components[0].pads[0]
        assert p1.position.x == pytest.approx(10.0, abs=1e-6)
        assert p1.position.y == pytest.approx(21.0, abs=1e-6)

    def test_copper_count_excludes_edge_cuts(self, parsed):
        assert parsed.layer_count == 2
