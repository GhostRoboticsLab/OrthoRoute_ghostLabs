# Changelog

## 1.1.0 — 2026-07-02

### Apple Silicon GPU support (Metal, via MLX)

- New `MetalDijkstra` backend: GPU wavefront SSSP on Apple M-series chips
  using `mlx.fast.metal_kernel` (32-bit atomic float-min relax, bit-packed
  frontiers, race-free post-convergence parent recovery — no 64-bit atomics
  required, so all M1–M5 GPUs work).
- Backend selection seam (`ORTHO_BACKEND=cuda|metal|cpu`): probes CUDA
  first, then Metal, else CPU; forcing an unavailable backend fails loudly.
  CUDA and CPU behavior is unchanged (golden-digest verified).
- GPU hazard fixes: a GPU fast-path error now falls back to CPU routing
  (previously the net was marked failed with no CPU attempt), and an
  owner-bitmap-constrained "no path" defers to cost-based CPU search.
  Backend init failures print a console-visible `[GPU-INIT]` banner.
- Install: `pip install mlx` (Python ≥ 3.10). On macOS the in-KiCad plugin
  runs KiCad's bundled Python 3.9, so Metal acceleration targets the
  headless/cli workflows run from your own venv.
- Measured on M4: 10M-node/59M-edge SSSP in 0.20–0.63 s; the TESTBOARD test
  fixture routes 36/36 portal-eligible nets to zero overuse in 6 iterations;
  TestBackplane (18 copper layers, 446k nodes, 14.3M edges) routes 512 nets
  at roughly 7 s per PathFinder iteration.

### KiCad 10 compatibility

- `.kicad_pcb` parser rewritten on a real s-expression parser
  (`orthoroute/infrastructure/kicad/sexpr.py`), supporting every dialect
  from KiCad 5 through KiCad 10 (format 20260206): name-only nets,
  `property`-based references, correct footprint rotation/back-side pad
  transforms (verified ≤ 1 µm against pcbnew 10.0.4), copper layer counting
  that no longer counts `Edge.Cuts`.
- Design rules are merged from the sibling `.kicad_pro` and applied to
  emitted geometry (track width, via diameter/drill respect board minimums).
- `build.py` derives the KiCad settings-path version dynamically (default
  10.0, `ORTHO_KICAD_VERSION` override), including the PCM metadata.
- `kicad-python` requirement bumped to `>=0.7.1,<0.8` (Python 3.12–3.14
  wheels via pynng 0.9; KiCad 10 endpoints).
- Outdated "KiCad IS CURRENTLY BROKEN with PCM" warning corrected.

### Testing & validation (first test suite in the repo)

- `tests/`: 90+ KiCad-free pytest tests — s-expression parser, lattice/CSR
  invariants, via accounting, file parser (both dialects), backend seam
  fault injection, Metal-vs-CPU SSSP oracles, congestion stressor.
- Stripped TESTBOARD routing fixtures (`TestBoards/testboard/`) with
  byte-reproducible regeneration (`tools/make_fixtures.py`).
- Routing correctness oracle (`route_oracle.py`): connectivity + Manhattan
  legality + no-silent-drops accounting, independent of `overuse==0`.
- Headless DRC oracle: `board_writer.py` writes routed copper back into a
  `.kicad_pcb` (deterministic UUIDs, blind/buried vias) and
  `tools/drc_check.py` runs `kicad-cli pcb drc` on the result.
- Repaired `--test-via` smoke test (was broken on all platforms: 2-layer
  board yields an empty routing graph) and made `cli` mode fail loudly on
  0-net parses instead of silently generating no copper.

### Known limitations (unchanged from upstream, now measured)

- Zone/plane blindness: the router does not see copper pours; routes across
  a solid GND plane pass `overuse==0` but violate DRC (~600 zone-conflict
  violations on the TESTBOARD fixture).
- Via-barrel conflicts: small residual class of via-to-via shorts.
- Board outline is not modeled (routes may approach Edge.Cuts).
- Through-hole-only nets are skipped by the escape planner (SMD-first
  design).
