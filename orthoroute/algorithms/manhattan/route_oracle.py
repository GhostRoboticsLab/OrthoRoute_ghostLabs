"""Routing correctness oracle (plan §C2).

``overuse == 0`` is a *convergence* signal, not a *correctness* signal:
nets whose pads lack portals are silently dropped by ``_parse_requests``,
and a converged run says nothing about path legality. This module provides
the engine-level checks every routing gate must pass, independent of
overuse:

1. **Accounting** — no silent drops: every portal-eligible net is either
   routed or explicitly reported.
2. **Connectivity** — every routed net's path is a single connected node
   sequence whose endpoints are the net's portal nodes.
3. **Legality** — every lateral step follows its layer's H/V discipline;
   every layer change keeps (x, y) fixed (Manhattan via).

Backend-agnostic: paths are compared structurally, never node-by-node
against another backend (GPU tie-breaking differs legitimately).

Python 3.9 compatible.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class OracleReport:
    """Result of validate_routing; falsy when any check failed."""

    def __init__(self):
        self.eligible_nets = 0
        self.routed_nets = 0
        self.unrouted_nets: List[str] = []
        self.errors: List[str] = []
        self.overuse_total = 0
        self.overuse_edges = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def summary(self) -> str:
        status = "OK" if self.ok else f"{len(self.errors)} ERRORS"
        return (f"oracle {status}: {self.routed_nets}/{self.eligible_nets} "
                f"eligible nets routed, {len(self.unrouted_nets)} unrouted, "
                f"overuse={self.overuse_total}")


def validate_routing(pf, board) -> OracleReport:
    """Run the §C2 engine-level oracle against a routed PathFinderRouter.

    Args:
        pf: A UnifiedPathFinder after route_multiple_nets().
        board: The routed Board (source of net/pad truth).

    Returns:
        OracleReport with per-check errors (empty errors == pass).
    """
    report = OracleReport()
    lattice = pf.lattice
    plane = lattice.x_steps * lattice.y_steps

    # -- 1. accounting: no silent drops -------------------------------
    for net in board.nets:
        pads = getattr(net, "pads", [])
        if len(pads) < 2:
            continue
        portaled = [p for p in pads if _pad_portaled(pf, p)]
        if len(portaled) < 2:
            continue  # not portal-eligible (TH-only / unportaled)
        report.eligible_nets += 1
        path = pf.net_paths.get(net.name)
        if path and len(path) >= 2:
            report.routed_nets += 1
        else:
            report.unrouted_nets.append(net.name)

    # -- 2 & 3. per-path connectivity + legality ----------------------
    for net_name, path in pf.net_paths.items():
        if len(path) < 2:
            continue
        for a, b in zip(path, path[1:]):
            ax, ay, az = lattice.idx_to_coord(int(a))
            bx, by, bz = lattice.idx_to_coord(int(b))
            if az == bz:
                legal = (lattice.is_legal_planar_edge(ax, ay, az, bx, by, bz)
                         or lattice.is_legal_planar_edge(bx, by, bz, ax, ay, az))
                if not legal:
                    report.errors.append(
                        f"{net_name}: illegal lateral step "
                        f"({ax},{ay},z{az})->({bx},{by},z{bz})")
            elif (ax, ay) != (bx, by):
                report.errors.append(
                    f"{net_name}: via moves laterally "
                    f"({ax},{ay},z{az})->({bx},{by},z{bz})")

        # Endpoints must sit at the net's portal columns. Layer-agnostic:
        # multi-seed supersource routing may legally enter at a different
        # layer of the same portal column.
        expected = _portal_columns(pf, net_name)
        if expected:
            ends = set()
            for node in (path[0], path[-1]):
                x, y, _ = lattice.idx_to_coord(int(node))
                ends.add((x, y))
            if ends != expected:
                report.errors.append(
                    f"{net_name}: path endpoint columns {sorted(ends)} != "
                    f"portal columns {sorted(expected)}")

    # -- convergence (reported, not an oracle failure by itself) ------
    total, count = pf.accounting.compute_overuse(pf)
    report.overuse_total = int(total)
    report.overuse_edges = int(count)

    logger.info("[ORACLE] %s", report.summary())
    for err in report.errors[:10]:
        logger.error("[ORACLE] %s", err)
    return report


def _pad_portaled(pf, pad) -> bool:
    """Whether any portal key matches this pad (component-scoped keys)."""
    comp_id = getattr(pad, "component_id", None) or "GENERIC_COMPONENT"
    return f"{comp_id}_{pad.id}" in pf.portals


def _portal_columns(pf, net_name) -> set:
    """Expected (x, y) endpoint columns for a routed net, from its portals."""
    pad_ids = getattr(pf, "net_pad_ids", {}).get(net_name)
    if not pad_ids:
        return set()
    columns = set()
    for pad_id in pad_ids:
        portal = pf.portals.get(pad_id)
        if portal is None:
            return set()
        columns.add((portal.x_idx, portal.y_idx))
    return columns
