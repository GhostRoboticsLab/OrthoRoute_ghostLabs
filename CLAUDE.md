# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

OrthoRoute is a GPU-accelerated PCB autorouter for KiCad 9 (IPC API plugin, Python). It routes on a 3D Manhattan lattice (alternating horizontal/vertical layers, blind/buried vias) using the PathFinder negotiated-congestion algorithm, with CUDA/CuPy acceleration of the shortest-path search.

## Commands

```bash
pip install -r requirements.txt          # numpy required; PyQt6 (GUI) and cupy (GPU) optional

python main.py                           # plugin with GUI (requires running KiCad 9 with IPC API enabled and a board open)
python main.py plugin --no-gui           # plugin headless (loads board from live KiCad)
python main.py headless board.ORP        # batch routing: .ORP in → .ORS out (the cloud-GPU workflow)
python main.py headless board.ORP -o out.ORS --max-iterations 250 --cpu-only
python main.py cli board.kicad_pcb       # route a .kicad_pcb directly — CPU-only, and does NOT save results (unimplemented)

python build.py                          # build manual-install IPC plugin zip into build/ (wipes build/ first)
python build.py --pcm                    # build the SWIG/PCM fallback package
```

There is no test suite (zero `test_*.py` files — adding pytest tests in `tests/` is the maintainer's top-wanted contribution). Validation is via smoke tests:

```bash
python main.py --test-via                # self-contained 2-layer via test, no KiCad needed — fastest sanity check
python main.py --autoroute               # headless smoke test against a live KiCad board (CPU)
python main.py --test-manhattan          # full routing test — launches the GUI with auto-start, needs KiCad
```

These global flags override any subcommand. Exit codes: 0 pass, 1 fail; output ends with `TEST PASSED` / `VIA TEST PASSED` etc.

**Logging:** console shows WARNING+ only (deliberately, to avoid flooding agent context). DEBUG/INFO go to `logs/latest.log` (deleted and rewritten every run) and `logs/run_<timestamp>.log` in the CWD. Check the log files for routing progress, e.g. `grep "\[ITER" logs/latest.log`.

## Architecture

### The live routing path (what actually runs)

The clean-architecture scaffolding (`application/` RoutingOrchestrator, commands/queries, EventBus, memory repositories, `domain/services/routing_engine.py` ABC) is **not on the live path**. The real flow:

1. `main.py` → `KiCadPlugin` (`orthoroute/presentation/plugin/kicad_plugin.py`) — the composition root; creates the **single** `UnifiedPathFinder` instance.
2. Board extraction: `RichKiCadInterface` (`orthoroute/infrastructure/kicad/rich_kicad_interface.py`) via `kipy` — returns a raw `board_data` **dict**, not a domain object. Reads `KICAD_API_SOCKET`/`KICAD_API_TOKEN` env vars set by KiCad.
3. GUI (`orthoroute/presentation/gui/main_window.py`, ~4000 lines) converts dict → domain `Board` (`_create_board_from_data`) and drives the engine. Routing runs **synchronously on the Qt thread** with `processEvents()` — `RoutingThread` is defined but never instantiated.
4. Engine call sequence (mandatory order; GUI and `headless` mode run all steps — the smoke tests and `cli` mode skip `precompute_all_pad_escapes`):
   `pf.initialize_graph(board)` → `pf.map_all_pads(board)` (legacy no-op, as is `prepare_routing_runtime`) → `pf.precompute_all_pad_escapes(board)` → `pf.prepare_routing_runtime()` → `pf.route_multiple_nets(board.nets, ...)` → `pf.emit_geometry(board)` → `pf.get_geometry_payload()`.
   Skipping escape planning silently yields zero routed nets (`_parse_requests` drops nets whose pads lack portals).
5. Write-back to KiCad lives in the **GUI layer**: `commit_routes()` in `main_window.py` builds kipy `Track`/`Via` objects and pushes a transactional commit. `main_window.py` cannot be imported without kipy.

Before routing starts, `_pathfinder_negotiation` STEP 0 runs board analysis (`board_analyzer.py`) and **parameter auto-derivation** (`parameter_derivation.py`), which overwrites hand-set config values (pres_fac schedule, via cost, hotset cap) based on the congestion ratio ρ. Env overrides: `ORTHO_PRES_FAC_MULT`, `ORTHO_PRES_FAC_MAX`, `ORTHO_HIST_GAIN`.

### The engine: one monolith, several decoys

- The live engine is `PathFinderRouter` (~3,800 lines) in `orthoroute/algorithms/manhattan/unified_pathfinder.py`, aliased at line 5816: `UnifiedPathFinder = PathFinderRouter`. It has **no base classes** — it composes helper objects (`Lattice3D`, `CSRGraph`, `EdgeAccountant`, `ROIExtractor`, `SimpleDijkstra`, `PadEscapePlanner`, `CUDADijkstra`). The docstring at the top of this file (lines 1–489) is the best architecture document in the repo, including CRITICAL INVARIANTS.
- The seven `*_mixin.py` files in `algorithms/manhattan/pathfinder/` are an **abandoned refactor** — `PathFinderRouter` does not inherit them and editing them has zero runtime effect. But they are import-time load-bearing: `pathfinder/__init__.py` imports all of them, so a syntax error in any mixin breaks the whole router.
- Live modules inside `pathfinder/`: `config.py` (constants), `kicad_geometry.py` (lattice↔world conversion), `cuda_dijkstra.py` (GPU SSSP kernels), `via_kernels.py` (via-capacity kernels).
- **Name-collision traps:** two `PathFinderRouter` classes (live: `unified_pathfinder.py:1979`; legacy: `rrg.py:225`, unused) and two `PathFinderConfig` classes (live: `unified_pathfinder.py:564`, plain class, no kwargs; decoy: `pathfinder/config.py:94`, dataclass with different defaults). Two `Portal` classes (live: `pad_escape_planner.py`).
- **Dead code — do not edit or trust:** `cuda_dijkstra_original.py`, `algorithms/base/` (nothing imports it), most of `rrg.py` (only `RoutingConfig` is imported), `real_global_grid.py`, `manhattan_router_rrg.py` (unreachable — the plugin's RRG fallback raises `RuntimeError('RRG disabled during bring-up')` in `kicad_plugin.py`, and the GUI method instantiating it has no callers — but it IS imported at module level by `main_window.py`, so keep it import-clean), `*.bak`/`*.backup` files, `infrastructure/serialization.py` (shadowed by the `serialization/` package), and `presentation/pipeline.py` (imports a nonexistent `graph_checks` module; nothing calls it). Checkpointing is also gone: `algorithms/manhattan/checkpoint.py` doesn't exist — the GUI checkpoint menu items fail with an error dialog, and headless `--checkpoint-interval`/`--resume-checkpoint` flags are parsed but ignored.

### PathFinder invariants (violating these breaks convergence)

- Edge capacity is 1; `emit_geometry()` only produces clean KiCad-exportable geometry when total overuse == 0 — otherwise only provisional GUI-preview geometry exists.
- Present usage is rebuilt from committed nets every iteration; costs are updated once per iteration so all nets see the same cost landscape (`SEQUENTIAL_ALL=1` forces per-net updates for debugging).
- The hotset contains only nets touching overused edges — rerouting clean nets causes thrashing.
- Pad escapes ("portals"): every SMD pad gets an F.Cu stub + a via down to an inner layer; routing then happens portal-to-portal on inner layers. Escape geometry is private (not in the routing graph) and always merged at emission.
- Determinism relies on seeded RNG (seed 42) and stable sort orders — don't change pad/net iteration order casually.
- Cost-function edits are flagged HIGH-RISK in code comments (small changes cause 20%+ convergence regressions). ~300–500 via-barrel DRC violations on large boards are a **known limitation** — do not add aggressive late-phase barrel penalties (documented overuse explosion).

### Units and coordinates

Everything internal is **float mm** plus grid indices (default pitch 0.4mm); flat node index = `layer*(x_steps*y_steps) + y*x_steps + x`. KiCad IPC uses integer **nanometers** — conversion (`/1_000_000`) happens only at the boundary (`rich_kicad_interface.py`, `commit_routes()`, `swig_adapter.py`). Layers use KiCad names (`F.Cu`, `In1.Cu`, …); odd inner layers route horizontal-only, even vertical-only. Emitted geometry is re-quantized to the grid so via centers exactly match track endpoints.

### Configuration reality

The engine ignores `orthoroute.json` (that file feeds `ApplicationSettings`, which only the ceremonial layer reads). Real engine config is the `PathFinderConfig` in `unified_pathfinder.py` plus env vars: `USE_GPU`, `ORTHO_CPU_ONLY`, `SEQUENTIAL_ALL`, `ORTHO_NO_SCREENSHOTS` / `ORTHO_SCREENSHOT_FREQ` / `ORTHO_SCREENSHOT_SCALE` (GUI writes per-iteration screenshots to `debug_output/` — disable for long runs). Note: the GUI checkbox writes `ORTHO_GPU` but nothing reads it. GPU falls back to CPU **silently** at several layers (CuPy missing, per-call exceptions, small ROIs) — check `[GPU-INIT]`/`[GPU]` log lines before trusting performance numbers.

### ORP/ORS files and the cloud workflow

`.ORP` (board export) and `.ORS` (solution) are gzip-compressed JSON with a `format_version` field; readers/writers live in `orthoroute/infrastructure/serialization/` (`orp_exporter.py`, `ors_exporter.py`). Workflow: GUI Ctrl+E exports `.ORP` → `python main.py headless board.ORP` on a GPU box (see `docs/cloud_gpu_setup.md`) → GUI Ctrl+I imports the `.ORS` → "Apply to KiCad". ORP v1.0 does not serialize existing traces, keepouts, or component data.

### Known KiCad issues

- The IPC API only works when KiCad's "Select Items" tool is active and nothing is selected.
- PCM installation of IPC plugins crashes KiCad on Windows (KiCad GitLab issue #19465) — hence `build.py` produces both a manual-install IPC zip (primary) and a SWIG PCM zip (fallback). Users must enable Preferences → Plugins → Enable Python API.
- Layer-count detection needs KiCad ≥ 9.0.5; on failure it logs ERROR/WARNING and non-fatally defaults to 2 layers instead of aborting.

## Conventions (from docs/contributing.md)

- PEP 8; Google-style docstrings required on public functions/classes; comments explain *why* (document magic numbers).
- Tests: pytest, `tests/test_*.py`. Priority targets: lattice building, CSR construction, via pooling accounting, portal escape planning, pad mapping.
- Dependencies: testing/docs/dev tools OK; ask before core routing/GUI/GPU deps; **PyTorch is explicitly banned** (CuPy is the standard).
- Don't attempt to refactor `unified_pathfinder.py` wholesale in one PR.
- GPU changes must keep working CPU fallbacks and be tested on both paths.
