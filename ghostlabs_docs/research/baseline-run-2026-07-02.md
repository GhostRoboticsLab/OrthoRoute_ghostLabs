# Baseline Run on Apple M4 / Python 3.14 (2026-07-02)

> Investigation report supporting `ghostlabs_docs/plan.md` (2026-07-02).

# OrthoRoute CPU-only baseline on Apple Silicon (M4) + Python 3.14.6 — 2026-07-02

## 1. Environment setup — SUCCESS

- `.gitignore` already covers `.venv` (line 84 of `/Volumes/MacSSD1/DeveloperWorkspace/Projects/OrthoRoute_ghostLabs/.gitignore`); no edit needed. It also gitignores `logs/`, so log files won't pollute the repo.
- Venv created at `/Volumes/MacSSD1/DeveloperWorkspace/Projects/OrthoRoute_ghostLabs/.venv` with `/opt/homebrew/bin/python3 -m venv` (Python 3.14.6, pip 26.1.2, 295 MB after installs). This is the deliverable left behind.
- **Zero wheel build failures.** Every package resolved to a prebuilt arm64 or universal2 wheel; nothing compiled from source on Python 3.14/arm64.
- `kicad-python` installed trivially: resolved to **kicad-python 0.7.1** (pure-Python wheel) with binary deps `pynng 0.9.0` (cp314 universal2 wheel) and `protobuf 5.29.6`. `import kipy` succeeds on Python 3.14.
- Import sanity: numpy 2.5.0, psutil 7.2.2, PyQt6 (Qt runtime 6.11.0 reported by QtCore), and kipy all import cleanly.

### pip freeze (exact versions)
```
attrs==26.1.0
cffi==2.0.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kicad-python==0.7.1
numpy==2.5.0
protobuf==5.29.6
psutil==7.2.2
pycparser==3.0
pynng==0.9.0
PyQt6==6.11.0
PyQt6-Qt6==6.11.1
PyQt6_sip==13.11.1
referencing==0.37.0
rpds-py==2026.6.3
sniffio==1.3.1
```
(cupy intentionally skipped — no CUDA on Apple Silicon.)

## 2. Baseline test 1: `main.py --test-via` — FAIL (exit 1)

Console tail:
```
Starting tiny 2-layer via test...
WARNING - [BOUNDS] No pads found via board.nets, falling back to board._kicad_bounds + 3.0mm
ERROR - [BOUNDS] No bounds available, using default 100x100mm
ERROR - [VIA-TEST] Test failed with exception: No edges
VIA TEST FAILED: Exception occurred: No edges
```
Real exit code: 1. No Python traceback is printed — the test's `except Exception` handler (`main.py:281-284`) reduces it to the message. The exception is `ValueError("No edges")` raised at `orthoroute/algorithms/manhattan/unified_pathfinder.py:686` (`CSRGraph.finalize`, empty edge list path).

Root-cause chain (all pure logic, NOT a Python 3.14/arm64 issue):
1. The test builds a 2-layer board (`main.py:212`, `board.layer_count = 2`).
2. `unified_pathfinder.py:1141`: `routing_layers = list(range(1, layer_count - 1))` → empty for layer_count=2 → log shows `[VIA-PAIRS] layer_count=2, routing_layers=0` → **0 via pairs, 0 via edges**.
3. Lateral track edges are also only built on inner layers — `unified_pathfinder.py:1199` and `:1214` both loop `for z in range(1, self.layers - 1)` → empty for 2 layers → **0 track edges**.
4. Total pre-allocated edges = 0 (`Pre-allocating for 0 edges (0 via edges for 0 pairs)`), CSRGraph takes the list path and `finalize()` raises `ValueError("No edges")` at line 686.

Secondary defect in the test itself: `run_tiny_via_test` (`main.py:197-284`) creates `Pad` objects but never attaches them to the board (only `net.pad_ids` is set), so the bounds pass finds no pads and falls back to a default 100×100 mm area (`unified_pathfinder.py:2277` warning path).

Conclusion: the repo's "fastest sanity check" is broken by the current inner-layers-only routing architecture on any platform; the 2-layer test board is architecturally incompatible with it.

## 3. Baseline test 2: CLI vs KiCad 10 board (`testboard-mini.kicad_pcb`) — FAIL (exit 1, silent empty parse)

Command: `.venv/bin/python main.py cli .../testboard-mini.kicad_pcb -o /tmp/orthobaseline`

- No crash, no traceback. Runs the full pipeline in ~2 s and exits 1 via `main.py:676-677` (`[CLI] No copper generated`).
- **The regex parser does NOT handle the KiCad 10 format.** Parse result: `Loaded board: Untitled Board with 0 nets`, `Mapped 0 pads (from ~0)`, `=== Route 0 nets ===`, `0 tracks, 0 vias`.
- Isolated parser test confirms: **components parsed: 0, pads parsed: 0, nets parsed: 0** (layers DO parse: F.Cu, In1.Cu, In2.Cu, B.Cu found).
- The engine itself works fine downstream on Python 3.14: builds a 251×251×5 lattice (315,005 nodes), 1,888,524-edge CSR, CPU sort in 0.6 s, via metadata, Manhattan validation — all pass on the (empty) board.

Exact parser failures in `orthoroute/infrastructure/kicad/file_parser.py` vs the file format (`(version 20260206)`, `generator_version "10.0"`):
- **line 225** `net_pattern = r'\(net\s+(\d+)\s+"([^"]*)"\)'` requires a numeric net id. The KiCad 10 file contains **zero** numbered net declarations (no top-level net table at all); pad net refs are name-only: `(net "GND")`. → `_extract_nets` returns 0 nets. Same numeric-id assumption at line 210 for pad nets.
- **line 130** reference extraction expects legacy `(fp_text reference "...")`; KiCad 8+/10 uses `(property "Reference" "J1" ...)`. → every component's reference is empty, and **line 155** `if component['reference']:` drops all 79 footprints.
- **line 111** `footprint_pattern = r'\(footprint\s+"([^"]+)"\s+(.*?)\n\s*\)'` matches 79 footprints but the non-greedy body capture truncates at the first nested closing paren (~397 chars of a multi-KB footprint), so pads could never be extracted even if references matched.
- **line 385** `layer_count=len([l for l in layers if 'Cu' in l.get('name','')])` — substring match counts `Edge.Cuts` as copper, so this 4-copper-layer board is reported as **5 layers** (`Using 5 layers from board` in the log).
- Failure is silent by design: `load_board` (line 21-28) swallows exceptions, and nothing errors on a 0-net board until the final "No copper generated" check.

## 4. Baseline test 3: full board (`testboard.kicad_pcb`) — identical FAIL

Same result: `Loaded board: Untitled Board with 0 nets`, exit 1, `No copper generated`. The failure is format-level (KiCad 10 s-expression dialect), **not board-specific**.

## 5. Summary: what works / what breaks today

WORKS on Apple Silicon + Python 3.14.6:
- Venv + all deps as prebuilt wheels (numpy, psutil, PyQt6, kicad-python 0.7.1 incl. pynng cp314 wheel) — no compilations, no failures.
- The entire orthoroute import chain and engine machinery: config load, logging (console WARNING+, full logs in `logs/latest.log`), lattice build, CSR construction/sort (1.89M edges in 0.6s CPU), via metadata, CPU-only mode detection (`[GPU] CPU-only mode: config.use_gpu=False, CuPy not installed` — graceful, no crash without cupy; `CUDA_DIJKSTRA_AVAILABLE=True` just means the module imports).
- Layer-table parsing in the file parser (copper layer names are found).

BREAKS (both pre-existing logic/format issues, neither platform-related):
- `--test-via`: `ValueError("No edges")` (`unified_pathfinder.py:686`) because 2-layer boards produce empty `routing_layers` (`unified_pathfinder.py:1141`) and no lateral edges (`:1199`, `:1214`).
- `cli` mode on KiCad 10 `.kicad_pcb`: `KiCadFileParser` regexes target a pre-KiCad-8 dialect (`file_parser.py:111/130/155/210/225/385`) → 0 components/pads/nets parsed silently → router routes nothing → exit 1.

NOT TESTED (out of scope/needs live KiCad): `plugin` mode (requires running KiCad with IPC API + open board), `headless` ORP mode (no .ORP available), GUI rendering. No actual routing iteration was therefore exercised — no net ever made it into the router.

Artifacts: run outputs in the scratchpad (`via_test.out`, `cli_mini.out`, `cli_full.out`); no `/tmp/orthobaseline*` output was ever created because the CLI exits before writing (and per CLAUDE.md, cli mode "does NOT save results (unimplemented)" even on success).

## Key verified facts

- /Volumes/MacSSD1/DeveloperWorkspace/Projects/OrthoRoute_ghostLabs/.gitignore line 84 already contains '.venv' — no edit needed
- Venv created at /Volumes/MacSSD1/DeveloperWorkspace/Projects/OrthoRoute_ghostLabs/.venv with Python 3.14.6 (/opt/homebrew/bin/python3); size 295 MB
- pip freeze: numpy==2.5.0, psutil==7.2.2, PyQt6==6.11.0, PyQt6-Qt6==6.11.1, PyQt6_sip==13.11.1, kicad-python==0.7.1, protobuf==5.29.6, pynng==0.9.0, cffi==2.0.0, jsonschema==4.26.0, attrs==26.1.0, rpds-py==2026.6.3, referencing==0.37.0, jsonschema-specifications==2025.9.1, pycparser==3.0, sniffio==1.3.1
- Zero wheels failed to build on Python 3.14/arm64 — every dependency resolved to a prebuilt arm64 or universal2 wheel (pynng 0.9.0 ships a cp314 universal2 wheel)
- kicad-python resolves to 0.7.1 and installs cleanly on Python 3.14; 'import kipy' succeeds in the venv
- main.py --test-via FAILS with exit code 1: 'VIA TEST FAILED: Exception occurred: No edges'; the ValueError('No edges') is raised at orthoroute/algorithms/manhattan/unified_pathfinder.py:686 (CSRGraph.finalize)
- Via-test root cause: unified_pathfinder.py:1141 'routing_layers = list(range(1, layer_count - 1))' is empty for the test's 2-layer board, and lateral edges are only built for inner layers at unified_pathfinder.py:1199 and :1214 — total edges = 0 ('Pre-allocating for 0 edges (0 via edges for 0 pairs)' in logs/latest.log)
- run_tiny_via_test (main.py:197-284) never attaches its Pad objects to the board (only net.pad_ids is set), triggering '[BOUNDS] No pads found via board.nets' and the default 100x100mm bounds fallback (unified_pathfinder.py:2277)
- CLI baseline (main.py cli testboard-mini.kicad_pcb) exits 1 with '[CLI] No copper generated' (main.py:676-677) after parsing 0 nets, 0 pads, 0 components from the KiCad 10 board; log shows 'Loaded board: Untitled Board with 0 nets' and '=== Route 0 nets ==='
- The test boards are KiCad 10 format: '(version 20260206)' '(generator_version "10.0")'; they contain NO numbered net declarations — pad net references are name-only '(net "GND")'
- file_parser.py:225 net_pattern r'\(net\s+(\d+)\s+"([^"]*)"\)' requires a numeric net id, so _extract_nets returns 0 nets on KiCad 10 files (same numeric assumption for pad nets at file_parser.py:210)
- file_parser.py:130 extracts references via legacy '(fp_text reference "...")'; KiCad 8+/10 uses '(property "Reference" "J1" ...)', so all 79 matched footprints have empty references and are dropped by 'if component[reference]:' at file_parser.py:155
- file_parser.py:111 footprint_pattern r'\(footprint\s+"([^"]+)"\s+(.*?)\n\s*\)' truncates the footprint body at the first nested closing paren (captured only 397 chars of the first footprint), so pads can never be parsed from KiCad 8+ files
- file_parser.py:385 counts copper layers via substring 'Cu' in layer name, counting 'Edge.Cuts' — the 4-copper-layer board is reported as 5 layers ('Using 5 layers from board')
- Retry with the full board testboard.kicad_pcb produced the identical failure (0 nets, exit 1) — the parse failure is format-level, not board-specific
- KiCadFileParser.load_board (file_parser.py:21-28) swallows all exceptions and returns None/empty data; the empty-board condition produces no error until the final copper check — the parse failure is silent
- CPU-only engine machinery works on Python 3.14/arm64: 251x251x5 lattice (315,005 nodes), 1,888,524-edge CSR built and sorted in 0.6 s on CPU, '[GPU] CPU-only mode: config.use_gpu=False, CuPy not installed' logged with graceful fallback (no crash without cupy)
- No routing iteration was ever exercised in any baseline test — zero nets reached the router in all three runs

## Recommendations

- Treat the file parser (orthoroute/infrastructure/kicad/file_parser.py) as the primary blocker for CPU-only CLI routing on this machine: it needs KiCad 8+/10 s-expression support (property "Reference" syntax, name-only '(net "NAME")' pad refs with no numbered net table, and balanced-paren footprint body extraction instead of the non-greedy regex at line 111). A proper S-expression tokenizer would fix all four defects at once.
- Plan to derive the net table from pad-level '(net "NAME")' occurrences on KiCad 10 boards, since the numbered top-level net table no longer exists — net ids must be synthesized.
- Fix the copper-layer count at file_parser.py:385 to match layer type/name exactly (e.g. name.endswith('.Cu')) instead of substring 'Cu', which currently counts Edge.Cuts.
- Do not use 'main.py --test-via' as the platform sanity check: it is broken by design for 2-layer boards (routing_layers = range(1, layer_count-1) is empty at unified_pathfinder.py:1141; lateral edges also skip outer layers at :1199/:1214). Either the test should use a >=4-layer board or the engine needs a 2-layer mode; a plan should pick one explicitly.
- Also fix run_tiny_via_test in main.py to attach pads to the board (it only sets net.pad_ids), otherwise bounds fall back to a default 100x100mm area even after the layer issue is addressed.
- Add an early hard failure in run_cli when the parsed board has 0 nets or 0 pads, so parser regressions surface as errors instead of a 2-second silent no-op pipeline.
- For a real routed-baseline on this Mac, the most viable near-term path is plugin/headless via kipy (kicad-python 0.7.1 installs and imports fine on Python 3.14) against a live KiCad 10 instance, or a .ORP export — the CLI file-parser path cannot produce a routing baseline until the parser is rewritten.
- No platform work is needed for the CPU engine itself: imports, numpy 2.5, PyQt6 6.11 and CSR construction all work on Python 3.14/arm64; cupy remains correctly optional with graceful CPU fallback.
- Expect exit-code semantics to be trustworthy (1 on failure) but console output to be sparse — any automation should parse logs/latest.log rather than stdout, per the repo's WARNING+-only console policy.
