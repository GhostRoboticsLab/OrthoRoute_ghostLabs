## What

Replaces the regex `.kicad_pcb` loader with a real s-expression parser that
reads every KiCad dialect from 5 through 10.

## Why

KiCad 8/9/10 dropped the numbered net table and moved references into
`(property "Reference" ...)`. The old regex loader returns **0 nets / 0 pads**
on those files, so loading a board directly from disk (the `cli` and headless
paths) is effectively broken on any modern board.

## What's in it

- `sexpr.py` — a small, dependency-free balanced-paren parser (Python 3.9
  clean) plus byte-exact, idempotent top-level strip helpers.
- `file_parser.py` — structural pad/net/layer extraction across both dialects:
  legacy numbered-net + `(module ...)`, and modern net-table-free
  `(footprint ...)`/`property` with name-only `(net "NAME")` synthesis.
  Footprint rotation is CCW-positive in KiCad's Y-down frame and back-side pad
  offsets carry no extra mirror — verified pad-exact (≤1 µm) against
  pcbnew 10.0.4. Copper-layer counting excludes `Edge.Cuts`; design rules are
  merged from a sibling `.kicad_pro` when present.
- Small fixes ridden along: `pad_escape_planner` reads `drill` with a
  `drill_size` fallback; `rich_kicad_interface` imports `KiCad` from `kipy`
  (matching the rest of the codebase).

## Tests

Small, self-contained **synthetic** boards (no external board files):

- `test_sexpr` — parser round-trips + idempotent strip helpers.
- `test_file_parser` — both dialects: legacy numbered-net `(module ...)`, and
  modern net-table-free `(footprint ...)`/`property` with name-only net
  synthesis, back-side pad placement, and `Edge.Cuts` exclusion.

```
$ python -m pytest tests/test_sexpr.py tests/test_file_parser.py
27 passed
```

## Notes for review

- This replaces a core input path. Happy to gate it behind an issue/discussion
  first if you'd prefer — opened as a PR because the old path returns nothing
  on current KiCad files, so it's arguably a bug fix.
- Pad-transform correctness was validated pad-exact against pcbnew 10.0.4 on
  real multi-layer boards (front/back, rot 0/90/180/270). The shipped tests pin
  the same transform on synthetic boards so the suite stays fixture-free.

## Checklist

- [x] Tests pass locally (27 passed on a clean clone)
- [x] No new runtime dependencies (stdlib only)
- [x] Preserves KiCad 5–7 parsing (legacy dialect is tested)
- [x] Google-style docstrings on new code
