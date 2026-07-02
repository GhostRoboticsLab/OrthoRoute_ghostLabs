"""Assertions on the stripped TESTBOARD test fixtures.

The fixtures are stripped copies of finished boards (all routed copper
removed, zones kept). These tests guarantee the fixtures stay stripped and
pin the board statistics the Phase-1 parser gate asserts against.
"""

import pytest

from orthoroute.infrastructure.kicad.sexpr import (
    children,
    find_top_level_spans,
    node_name,
    parse_file,
    strip_top_level_nodes,
)


def _load_root(path):
    doc = parse_file(str(path))
    assert len(doc) == 1 and node_name(doc[0]) == "kicad_pcb"
    return doc[0]


def _net_names(root):
    """Distinct non-empty net names referenced by any (net ...) node."""
    names = set()

    def walk(node):
        if not isinstance(node, list):
            return
        if node and node[0] == "net":
            leaves = [c for c in node[1:] if isinstance(c, str)]
            if leaves:
                names.add(leaves[-1])  # KiCad 10: (net "NAME"); legacy: (net 3 "NAME")
        for c in node[1:]:
            walk(c)

    walk(root)
    names.discard("")
    return names


@pytest.fixture(scope="module")
def main_root(stripped_main_board):
    return _load_root(stripped_main_board)


@pytest.fixture(scope="module")
def mini_root(stripped_mini_board):
    return _load_root(stripped_mini_board)


class TestStrippedMainBoard:
    @pytest.fixture
    def root(self, main_root):
        return main_root

    def test_no_routed_copper(self, root):
        assert not children(root, "segment")
        assert not children(root, "via")

    def test_zones_kept(self, root):
        assert len(children(root, "zone")) == 6

    def test_kicad10_format(self, root):
        version = children(root, "version")[0][1]
        assert int(version) >= 20260000

    def test_net_count(self, root):
        assert len(_net_names(root)) == 90

    def test_gnd_membership(self, root):
        gnd_pads = 0
        for fp in children(root, "footprint"):
            for pad in children(fp, "pad"):
                net = children(pad, "net")
                if net and net[0][-1] == "GND":
                    gnd_pads += 1
        assert gnd_pads == 66

    def test_pad_count(self, root):
        pads = sum(len(children(fp, "pad")) for fp in children(root, "footprint"))
        assert pads == 329

    def test_copper_layer_count(self, root):
        layers = children(root, "layers")[0]
        copper = [l for l in layers[1:]
                  if isinstance(l, list) and str(l[1]).endswith(".Cu")]
        assert len(copper) == 4  # F, In1, In2, B — Edge.Cuts must not count


class TestStrippedMiniBoard:
    @pytest.fixture
    def root(self, mini_root):
        return mini_root

    def test_no_routed_copper(self, root):
        assert not children(root, "segment")
        assert not children(root, "via")

    def test_zones_kept(self, root):
        assert len(children(root, "zone")) == 9

    def test_same_logical_nets_as_main(self, root, stripped_main_board):
        main_nets = _net_names(_load_root(stripped_main_board))
        assert _net_names(root) == main_nets

    def test_pad_count(self, root):
        pads = sum(len(children(fp, "pad")) for fp in children(root, "footprint"))
        assert pads == 332


class TestStripReproducibility:
    def test_stripping_fixture_is_noop(self, stripped_main_board):
        text = stripped_main_board.read_text(encoding="utf-8")
        restripped, removed = strip_top_level_nodes(text, ("segment", "via"))
        assert removed == 0
        assert restripped == text

    def test_no_top_level_spans_left(self, stripped_mini_board):
        text = stripped_mini_board.read_text(encoding="utf-8")
        assert find_top_level_spans(text, ("segment", "via")) == []
