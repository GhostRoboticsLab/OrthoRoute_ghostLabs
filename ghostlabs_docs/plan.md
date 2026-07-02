# ghostLabs Master Technical Plan — Apple Silicon GPU Support + KiCad 10 Compatibility

**Repo:** OrthoRoute_ghostLabs (fork of bbenchoff/OrthoRoute)
**Date:** 2026-07-02 · **Status:** PLANNING — no implementation started
**Dev machine:** Apple M4 (GPU family Apple9), macOS 26.5.1, Python 3.14.6 (Homebrew), KiCad 10.0.4 installed, IPC API already enabled in KiCad preferences.

---

## 1. Mission

Upgrade OrthoRoute so that it:

1. **Routes on Apple M-series GPUs** — the engine is currently CUDA/CuPy-only; Apple Silicon users get silent CPU fallback. **Runtime-scope decision (see §B1, Risk 4):** MLX requires Python ≥ 3.10, but the in-KiCad macOS plugin runs KiCad's bundled Python 3.9 — so `import mlx` fails inside the plugin. The Metal backend therefore targets the **headless / ORP / `cli` path first** (dev venv + local/cloud GPU box, Python 3.14). In-plugin GPU acceleration on macOS is a follow-on that depends on resolving the interpreter (an external ≥ 3.10 venv or a subprocess worker); until then the in-KiCad GUI stays CPU on macOS.
2. **Works with KiCad 10** — upstream targets KiCad 9; KiCad 10.0.x changed the board file format and the plugin install paths.
3. Stays **upstream-contributable** — changes should be shaped so bbenchoff/OrthoRoute can merge them (see §7 constraints; notably **PyTorch is banned** as a dependency by the upstream maintainer).

Test fixtures: the vendor TESTBOARD carrier boards at
`/path/to/local/test-boards/`
(`testboard.kicad_pcb` 50×80 mm, `testboard-mini.kicad_pcb` 50×64 mm — both 4-layer, KiCad 10 format). See §5 for what they can and cannot test.

---

## 2. Verified baseline (2026-07-02, this machine)

Everything below was measured, not assumed. Environment setup is **done**: `.venv/` exists at repo root (gitignored) with Python 3.14.6 and all deps as prebuilt arm64/universal2 wheels — zero build failures:

```
numpy==2.5.0  psutil==7.2.2  PyQt6==6.11.0  kicad-python==0.7.1
protobuf==5.29.6  pynng==0.9.0  (cupy intentionally omitted — no CUDA here)
```

| Check | Result | Root cause |
|---|---|---|
| Imports: full orthoroute chain, kipy, PyQt6 | ✅ works on Py3.14/arm64 | — |
| Engine machinery CPU-only (lattice 315k nodes, 1.89M-edge CSR in 0.6 s, graceful no-CuPy fallback) | ✅ works | — |
| `main.py --test-via` | ❌ FAIL `ValueError("No edges")` | **Pre-existing bug, all platforms**: 2-layer boards yield empty `routing_layers = range(1, layer_count-1)` (`unified_pathfinder.py:1141`) and no lateral edges (`:1199,:1214`). Test board also never attaches its pads to the board. |
| `main.py cli <testboard board>` | ❌ FAIL, exit 1, **silent** | `file_parser.py` regexes target a pre-KiCad-8 dialect → parses **0 nets, 0 pads, 0 components** from KiCad 10 files, then "No copper generated". |
| Actual routing iteration | ⚠️ never exercised | zero nets reached the router in all baseline runs |

**Implication:** there is currently *no* working smoke test on this machine. Phase 0 must create one before any port work can be validated.

---

## 3. Workstream A — KiCad 10 compatibility

### A0. Facts established (with sources)

- KiCad 10.0.0 released 2026-03-20; current 10.0.4 (2026-06-21). **Target ≥ 10.0.1** — 10.0.0 shipped a regression where IPC plugins didn't appear in the toolbar (fixed 10.0.1, commit `761bdf55`).
- **IPC wire protocol is compatible**: KiCad's API policy is additive-only protobuf changes; a 9.0-built client still talks to 10.x. `KICAD_API_SOCKET`/`KICAD_API_TOKEN` and the macOS socket (`/tmp/kicad/api.sock`) are unchanged.
- **kicad-python (kipy):** bump to `>=0.7.1,<0.8`. 0.6.0 moved to pynng 0.9.0 (ships cp314 wheels — this is the Python 3.14 fix); 0.7.x adds KiCad-10-only endpoints (`get_items_by_net`, `get_connected_items`, …) which must be gated on a runtime KiCad version check. Verified installing and importing cleanly on Python 3.14.
- **macOS KiCad 10.0.4 bundles Python 3.9** and uses it for IPC-plugin venvs by default → **plugin code must stay 3.9-compatible** (no 3.10+ syntax like `match` or `X | Y` annotations in runtime code).
- **Board file format changed hard** (version `20260206`): the numbered net table is **gone**; nets are referenced by name only — `(net "GND")` on pads/tracks/zones. Component references moved from `(fp_text reference …)` to `(property "Reference" …)` (since KiCad 8).
- **The PCM story is stale**: GitLab issue #19465 was a Windows *load failure* (not a crash) and was fixed **before KiCad 9.0.0 final**. PCM formally supports IPC plugins via `runtime: "ipc"` since 9.0.1 and prompts users to enable the API server. PCM packaging schema v2 exists for KiCad 10 (v1 still accepted).
- Plugin dir on macOS: `~/Documents/KiCad/10.0/plugins/` (exists, empty, on this machine). `plugin.json` API schema is still v1 — no manifest change needed.

### A1. Fix the `.kicad_pcb` file parser (blocker for `cli` mode and the File fallback adapter)

`orthoroute/infrastructure/kicad/file_parser.py` fails on four independent counts against KiCad 10 files (all verified by executing its exact regexes):

1. `:225` net table regex requires `(net <int> "name")` — zero matches (table no longer exists). Net IDs must be **synthesized** from name-only pad refs.
2. `:130/:155` reference extraction expects `(fp_text reference …)` → all footprints dropped (KiCad 8+ uses `(property "Reference" …)`).
3. `:111` non-greedy footprint body capture truncates at the first nested `)` (~400 chars of a multi-KB footprint) → pads unreachable even if 1–2 were fixed.
4. `:385` counts copper layers by substring `'Cu'` — **counts `Edge.Cuts`**, reporting 5 layers on a 4-copper board (this reached the engine in baseline: it built a 5-layer lattice).

**Plan:** replace the regex approach with a small **balanced-paren s-expression tokenizer** (one class, no dependency) and rebuild the extractors on top. Support both dialects (numbered nets + fp_text for old files; name-nets + property for 8/9/10 files). Add a hard failure in `run_cli` when a parsed board has 0 nets or 0 pads — today the failure is completely silent.

### A2. Packaging / install fixes

- `build.py` hardcodes `Documents/KiCad/9.0/` in **~14 places** (lines 130/136/142/152-154/158/168/172/177/265/266/267 — the original six-line list was non-exhaustive), **including the functionally load-bearing `build.py:313` `"kicad_version": "9.0"` in the PCM metadata JSON** → derive the version segment dynamically; default to 10.0.
- `requirements.txt`: `kicad-python>=0.7.1,<0.8` (pulls pynng≥0.9 with cp314 wheels); keep numpy/psutil; cupy stays optional.
- Reconsider PCM distribution (see A0) — optional stretch goal; manual install keeps working meanwhile.
- Update README/docs: the "KiCad IS CURRENTLY BROKEN with PCM" warning and issue-19465 rationale are outdated.

### A3. Live IPC verification against KiCad 10.0.4

The wire protocol is compatible on paper; verify in practice (plugin mode with a board open):
- `RichKiCadInterface.connect()` + `get_board_data()` — pads/tracks/stackup extraction, layer-count detection ladder (needs ≥9.0.5 APIs — confirm they respond in 10.0.4).
- `commit_routes()` write-back — kipy `Track`/`Via` creation, `begin_commit/push_commit`, `VT_BLIND_BURIED` padstack behavior.
- Known repo quirk to re-test on 10: IPC only works when the "Select Items" tool is active and nothing is selected.
- Fix the `from kicad import KiCad` vs `from kipy import KiCad` inconsistency in `rich_kicad_interface.py` while in there.

### A4. Repo test-harness repairs (platform-independent bugs found in baseline)

- `--test-via` is broken by design: 2-layer boards produce zero edges. Either build the synthetic test board with 4 layers or add an explicit 2-layer mode decision. Also attach the test's `Pad` objects to the board (currently only `net.pad_ids` is set → bogus 100×100 mm bounds fallback).
- These fixes matter because `--test-via` is the only KiCad-free smoke test — it must pass before it can gate the Metal port.

---

## 4. Workstream B — Apple M-series GPU support

### B0. Porting surface (from full code audit)

The scary number — ~4,100 lines of embedded CUDA C — collapses on inspection:

- **Only ~780 CUDA-C lines actually execute**: `wavefront_expand_active` (the #1 hot kernel, 217 lines), `wavefront_expand_all` (239), `compact_mask_to_list` (35), `backtrace_paths` (105), `validate_parents` (35), plus 3 via kernels (~148) in `via_kernels.py`.
- ~1,530 lines of kernels are **disabled or dead** (3 persistent-kernel variants, delta stepping, procedural relax, accountant kernel) — do **not** port them. `cuda_dijkstra_original.py` and the mixins are dead weight too. (Arithmetic check: live ~780 + dead-kernel-source ~1,530 + the remaining ~1,800 of comments, host-side glue, and the ~494 lines of kernels stranded in the two abandoned mixins ≈ the ~4,100 headline.)
- The live GPU **call surface is 5 methods** (the plan originally said 6): `CUDADijkstra.{find_path_fullgraph_gpu_seeds, find_path_roi_gpu, find_paths_on_rois}` (the third only called *indirectly*, from inside `find_path_roi_gpu`) + `ViaKernelManager.{hard_block_via_edges, apply_via_penalties}`. **`detect_barrel_conflicts_gpu` is dead — zero callers anywhere; its kernel is compiled but never dispatched. Live barrel-conflict detection is the numpy `PathFinderRouter._detect_barrel_conflicts` (`unified_pathfinder.py:5373`, called at :3774).** So the Metal port needs **5** GPU methods, not 6, and the via layer has only **2** live kernels (see B4).
- Array math already uses the `xp = cupy-or-numpy` pattern (`CSRGraph`, `EdgeAccountant`, ~24 sites) — the entire PathFinder cost model is portable array code. The gaps are **~175** `hasattr(x,'get')`-shaped duck-typing sites (measured, not ~110; plus ~600 `.get(` calls needing per-site triage — device readback vs. plain `dict.get`) and **3–4** `hasattr(…,'device')` gates (`unified_pathfinder.py:3107, 3170, 4468`, plus `via_col_pres` at :3201), not one. This is a per-site audit, not a mechanical find-and-replace: a single misclassified site is a silent wrong-backend transfer, and there is no test net to catch it.
- CuPy host API in use is minimal: null stream only, no multi-stream, no memory pools on the live path, no cub/thrust, no ElementwiseKernel. One `as_strided` stride-0 trick that disappears naturally on Metal (kernels already take stride params).
- Known pre-existing wart: `unified_pathfinder.py` imports cupy twice with contradictory fallbacks (`cp = np` at :506 vs `cp = None` at :521) — unify during the seam refactor.

### B1. Backend decision

**Primary: MLX** (`mlx.core` for arrays + `mx.fast.metal_kernel` for raw-MSL kernels). Rationale (researched 2026-07):
- Actively maintained (0.31.2, Apr 2026; releases every ~3-4 weeks), Python 3.10–3.14, the only CuPy-shaped option on Apple GPUs. **⚠ The 3.10 floor collides with the KiCad-bundled 3.9 plugin runtime — see the runtime-scope decision in §1 and Risk 4; the Metal backend is headless-first on macOS.**
- `mx.fast.metal_kernel` takes raw MSL body + arbitrary header → full MSL intrinsics reachable, JIT-cached, atomics supported.
- Unified memory: zero-copy NumPy views (`np.array(a, copy=False)`); the existing per-net 20 MB parent-array readback and per-iteration `.get()→modify→asarray` round-trips become nearly free — a genuine architectural win over the CUDA original.
- Strategic: MLX now has a CUDA backend (`mlx[cuda]`), so the array layer could eventually serve NVIDIA users too.
- Known MLX constraints to design around: lazy evaluation (call `mx.eval()` once per iteration), no boolean-mask reads / `nonzero()` / single-arg `where()` (the kernels already use bit-packed frontiers, so this maps naturally), float64 CPU-only (engine is float32/int32 anyway).

**Fallback: wgpu-py** (WGSL + wgpu-native extension `shader-int64-atomic-min-max`, available on Metal for Apple8+Mac2/Apple9). Cost: no array layer at all — every reduction becomes a hand-written kernel. Keep as plan B only.

**Rejected** (with cause): PyTorch MPS (banned upstream), Taichi (development halted 2024), jax-metal (dead since Oct 2024), Vulkan/MoltenVK (no 64-bit atomics), cuPyNumeric (CPU-only on macOS), Mojo/MAX (Apple-GPU atomics not ready; re-evaluate in 12 months).

### B2. Two GPU-porting constraints: the real one (lazy-eval dispatch) and the optional one (64-bit atomics)

**Plan of record (single, unambiguous): ship the 32-bit path everywhere first.** The #1 hot kernel `wavefront_expand_active` is *already* 32-bit in shipping CUDA (32-bit float-min CAS + separate parent write, benign race, CPU-side cycle detection as safety net); it works on **all** M-series and needs no 64-bit atomics. Add the packed-u64 `atomicMin` `(float-cost-bits << 32) | parent` key later as an **Apple8+ (M2+) fast path**, not a prerequisite.

- **The 64-bit atomics question is an optimization, not "the one real hardware constraint."** It affects only M1 exclusion for the *non-hot* packed-key path. The dev machine (M4 = Apple9) has the full 64-bit set, so this can't even be exercised as a blocker here.
  - Metal family map: M1 = Apple7, M2 = Apple8, M3/M4 = Apple9, M5 = Apple10.
  - **Apple8 (M2, macOS) already has `ulong` atomic min/max — exactly the needed primitive; the full 64-bit set arrives at Apple9+ (M3/M4/M5); M1 (Apple7) has none, not even CAS** (Metal Feature Set Tables, fn.7 — externally confirmed). So the packed-u64 fast path uses **native** `atomic_fetch_min_explicit(atomic_ulong)` gated on **Apple8+ (M2+)**, not "Apple9+".
- **The actual primary-vs-fallback determinant (de-risk this FIRST — see B5):** whether MLX's lazy-eval graph + a Python wave-per-dispatch loop can detect frontier-empty termination **without a per-wave host readback** (a readback every wave forces `mx.eval` + sync and can dominate runtime); whether the ~32-arg kernel launch survives Metal's **31-buffer bind table** (pack into a constant struct — net-new code with no CUDA analog, plus re-laying the strided persistent pools `dist_val_pool`/`parent_val_pool`/`best_key_pool`); and whether `compact_mask_to_list`'s warp intrinsics hold on Apple simdgroups (`simd_ballot`/`popcount`/`simd_shuffle`; Apple simdgroups are 32-wide, matching the warp-32 assumptions).
- No cooperative-groups equivalent on Metal — irrelevant: every kernel needing grid sync is already disabled upstream ("Hangs on cooperative kernel launch"). The shipping design (Python loop of dispatches + sync) maps 1:1 to Metal command buffers. `Device().mem_info` → `MTLDevice.recommendedMaxWorkingSetSize` for stamp-pool sizing.

### B3. Architecture: backend seam

Introduce two protocols at the **existing** composition points (`unified_pathfinder.py:2132-2164` and `:2178-2197`). This is **not** a one-liner: `KernelProvider` wraps cleanly at those points, but `ArrayModule` touches ~175 hot-path sites — treat it as a discuss-first refactor (contributing.md), scoped as small as the audit allows:

1. **ArrayModule** — formalize `xp` + add `to_cpu(x)` / `is_device_array(x)` helpers replacing the ~175 `hasattr(x,'get')` sites and the 3–4 `hasattr(…,'device')` gates; unify the two contradictory `cp` fallbacks (`:506` `cp=np` vs `:521` `cp=None`). **Build this only if the Phase-2 spike selects MLX** — the wgpu-py fallback has no array layer, which would make ArrayModule dead work. Land it with the array-layer tests written *first* (Phase 0) and a grep lint that fails CI on raw `.get()`/hasattr-get in engine code.
2. **KernelProvider** — the **5-method** live surface from B0, implemented by today's `CUDADijkstra`/`ViaKernelManager` (unchanged behavior) and new `MetalDijkstra`/`MetalViaKernelManager`. (`detect_barrel_conflicts_gpu` is *not* part of the live surface — barrel detection stays numpy, so no Metal twin is needed.)

Selection: `use_gpu` → probe CUDA first, then Metal (`ORTHO_BACKEND=cuda|metal|cpu` env override for testing). Two porting-adjacent hazards get fixed as part of the seam because they would mask new-backend bugs:
- GPU fast-path failure currently marks the net failed and **skips CPU fallback** (`unified_pathfinder.py:4520-4529`) → add fallback or a loud counter.
- Backend init failure currently downgrades to CPU **silently** (`:2184-2197`) → surface `[GPU-INIT]` status in test output / a startup banner.

### B4. Correctness oracle

GPU wavefront is deliberately not cost-ordered and injects jitter/round-robin bias — GPU and CPU produce *different but equally valid* paths. Therefore:
- **Engine-level oracle:** compare per-net path cost, path legality (H/V discipline, via adjacency), and iteration-level overuse convergence vs `--cpu-only` runs — never exact node sequences. Determinism within a backend via seed 42.
- **Via kernels:** the CPU twins in `via_kernels.py:570-699` (`hard_block_via_edges_cpu`, `apply_via_penalties_cpu`, `detect_barrel_conflicts_cpu`) *functionally* mirror the GPU kernels, so an equivalence oracle is supported — with two caveats. (a) Only **2** of the three GPU kernels are actually live (`hard_block`, `apply_via_penalties`); `detect_barrel_conflicts_gpu` is dead (B0), so there is nothing on the GPU side to compare its twin against. (b) The twins are **not line-for-line and not wired** — they use 2D indexing and drop the `Ny` arg vs. the GPU's flattened `col_idx = xu*Ny + yu`, and nothing imports them; a comparison test must adapt input shapes and import them directly. Demand bit-exactness only after confirming each kernel's reduction is integer/order-independent; otherwise define a tolerance.

### B5. De-risking spike (before committing to the seam refactor) — ~3–5 days, not 1

Realistically several independent efforts across two shader toolchains; budget 3–5 days and run them in **risk order** so the primary-vs-fallback call is made early and cheaply:
1. **(Primary risk) MLX lazy-eval dispatch loop + frontier-empty readback cost** on a ~10M-node wavefront: prove you can loop-dispatch waves and detect termination without a per-wave host readback; measure `mx.eval` placement and the 31-buffer constant-struct bind. This — not atomics — decides whether MLX is viable. Compare against a NumPy baseline harness.
2. **(Guaranteed path) 32-bit float-min CAS relax kernel in MLX** — the portable path that ships first (plan of record, B2).
3. **(Optional optimization) MLX `atomic_ulong` fetch-min** — validate the undocumented-but-expressible 64-bit path with a ~20-line kernel; test *both* expressibility routes (`atomic_outputs=True` typed as `ulong` vs. reinterpret-cast a plain output buffer to `atomic_ulong*`). Time-boxed; not on the critical path.
4. **(Fallback only if MLX fails items 1–2) wgpu-py** relax kernel with `shader-int64-atomic-min-max` requested. Do **not** build this speculatively — standing up a second framework/shader-language (WGSL) is ~a day on its own, and selecting wgpu means an **all-or-nothing framework swap** (no array layer — every reduction becomes a hand-written kernel; MLX arrays and wgpu buffers don't share memory).

---

## 5. Workstream C — Test fixtures (TESTBOARD carriers)

### C0. What the analysis found (measured, not eyeballed)

| | carrier (main) | carrier-mini |
|---|---|---|
| Size / layers | 50×80 mm, 4 Cu (F, In1=**GND plane**, In2, B) | 50×64 mm, same stackup |
| Pads | **329 = 200 SMD (175 F.Cu/25 B.Cu) + 123 PTH + 6 NPTH** | **332 = 203 SMD (95 F.Cu/108 B.Cu) + 123 PTH + 6 NPTH** |
| Nets | 90 (GND 66 pads, VBUS_PIX 27, …) | **same 90 logical nets, different physical pads/placement** |
| **Already routed** | **yes** — 1,150 segments + 56 vias + zones | **yes** — 1,005 segments + 25 vias |
| Lattice @0.4 mm | 100k nodes (1.25% of OrthoRoute's 8M design target) | 80k nodes |
| Congestion ρ (doc formula) | 0.495 (sparse, **plane-blind**) | 0.482 |

Hard constraints discovered:
- **Both boards are finished designs.** ORP export silently drops existing traces/keepouts → fixtures must be **stripped copies** (all segments/vias deleted).
- **43 of 90 nets are 100% through-hole** (PGA module ↔ headers, 2.54 mm pin grid). The escape planner skips every pad with `drill > 0` (`pad_escape_planner.py:239-241`), so those nets are unroutable **by design** — only ~33 all-SMD nets (+14 mixed, SMD-ends only) will route. Assertions must target that subset, not 90.
- `_get_pad_layer` hardcodes F.Cu (`pad_escape_planner.py:766-769`): the mini's **108 back-side SMD pads** (WS2812B array) would get escape stubs emitted on the wrong copper (`[ESCAPE-LAYER-BUG]`).
- In1.Cu is the design's solid GND plane; OrthoRoute can't see zones, so it would happily route signals across it. Honoring the plane leaves the Manhattan scheme without a horizontal inner layer (honest ρ ≈ 0.96–0.99).

### C1. Fixture plan

1. `ghostlabs_docs/fixtures/` (or `TestBoards/testboard/`): **stripped** copies of both boards (tracks/vias removed; keep zones for future zone-awareness work), produced by a small script (kipy against live KiCad, or the new s-expr parser once written) so regeneration is reproducible. Originals in the vendor project are never modified.
2. **Main board = pipeline / regression smoke gate (NOT a routing-correctness gate).** After the KiCad-10 parser fix, an ORP-headless route of the stripped board exercises parse → lattice → escape → route → emit → commit end-to-end and should converge (0 overuse) for the portal-eligible SMD-net subset in seconds on CPU. **Caveat — this is plane-blind:** In1.Cu is a solid GND pour OrthoRoute can't see (ORP v1.0 doesn't serialize zones), so a converged route here lays signals *across* the plane; `overuse==0` proves convergence, not electrical validity (§C2). It gates *plumbing and CPU-vs-Metal self-consistency* (iteration counts within ~20% — §B4), not routing correctness. (The B.Cu-escape bug also trips on this board's 25 B.Cu SMD pads, so even the "positive" subset must be restricted to all-F.Cu nets until `_get_pad_layer` is fixed.)
   **A true "positive" correctness gate requires one of:** (a) plane-aware lattice — mark plane layers non-routing (needs zone extraction the plan doesn't yet have; then honest ρ ≈ 0.96–0.99 and the "seconds / 0-overuse" bar must be restated), or (b) a genuinely plane-free signal-only fixture (or synthetic board) paired with the connectivity + DRC oracle of §C2. Decide this before relying on any board as a correctness gate.
3. **Mini board = negative/regression fixture**: pins down the B.Cu-escape gap (expect `[ESCAPE-LAYER-BUG]`) — becomes a positive test only if/when B.Cu escapes are implemented.
4. **These boards never serve as performance benchmarks** — 1% of design-target size, ρ≈0.5. For Metal perf claims, use the repo's `TestBoards/TestBackplane.kicad_pcb` or a synthetic large lattice.
5. Also keep the repaired `--test-via` (A4) as the zero-dependency gate.

### C2. Correctness oracle (overuse==0 is convergence, not correctness)

`overuse==0` only means every edge's capacity (=1) is satisfied. It does **not** prove the route is electrically valid or complete. Two blind spots the fixtures inherit:

- **Connectivity:** nets whose pads lack portals are silently dropped by `_parse_requests`, so "33–47 nets routed" can include partially- or un-connected nets with nothing asserting otherwise.
- **DRC:** clean, converged runs still carry ~300–500 known via-barrel violations (CLAUDE.md), plus any shorts from routing across the invisible In1.Cu GND plane.

**Add two hard oracles, independent of overuse, to every routing gate:**

1. **Connectivity** — every routed net forms a single connected component spanning all its portal-eligible pads.
2. **DRC** — run KiCad 10 DRC (or a standalone clearance/short check) on committed geometry; record the violation count as a tracked metric with an explicit tolerance for the known via-barrel class.

State everywhere that `overuse==0` is a **convergence** signal, not a **correctness** signal. Note also that power nets (GND 66 pads, VBUS_PIX 27) have no net-class / width / clearance model here — either scope them out of the routable set or record that "router validation" excludes power/plane/diff-pair/length-matched nets (out of scope).

---

## 6. Phased roadmap

Each phase gates the next; nothing lands without its gate passing.

| Phase | Work | Gate (must pass) |
|---|---|---|
| **0. Harness + first tests** | Fix `--test-via` (A4); build stripped TESTBOARD fixtures (C1) + assert they're correctly stripped; add 0-nets hard-fail to `cli`; write the KiCad-free pytest scaffolding (lattice / CSR / via-accounting / oracle harness — also upstream's #1 ask) | `--test-via` passes on this Mac CPU-only; stripped fixtures have 0 tracks / 0 vias, zones kept, regeneration byte-reproducible; `cli` exits non-zero with a clear message on a 0-net board; pytest scaffolding runs green |
| **1. KiCad 10** | S-expr parser rewrite (A1); build.py/requirements bumps (A2, incl. `build.py:313` PCM `kicad_version`); live IPC verify vs 10.0.4 (A3); confirm which interpreter KiCad 10.0.4 uses for plugin venvs (§1 runtime-scope) | **Automatable (CI/pytest):** `cli` on stripped main board parses 90 nets / 329 pads, **asserts 4 copper layers** (F/In1/In2/B, Edge.Cuts excluded) and spot-checks net membership (e.g. GND = 66 pads). **Manual acceptance (tracked separately, live KiCad):** plugin connects to 10.0.4, extracts board, commits a route |
| **2. Metal spike** | B5 prototypes, in risk order (lazy-eval dispatch/readback first, atomics last) | measured lazy-eval dispatch + readback + wavefront numbers on ~10M nodes; primary-vs-fallback backend decision recorded here in plan.md |
| **3. Backend seam** | KernelProvider protocol (5 live GPU methods — B0); ArrayModule **only if the spike chose MLX**; silent-fallback hazard fixes **in a separate behavior-changing PR** with a fault-injecting `KernelProvider` test; CUDA path re-verified via reproducible golden-diff | CPU-only routing of main fixture identical pre/post refactor; **golden ORS diff** (`.ORP` → `main.py headless` on any CUDA host before/after → `ORS_before == ORS_after`); forced-GPU-failure test asserts CPU fallback + counter |
| **4. Metal port** | `MetalDijkstra` (~600 lines MSL **kernel bodies** + ~1,500–2,500 lines MLX **host driver** — budget both) + `MetalViaKernelManager` (2 live kernels vs shape-adapted CPU twins; barrel detection stays numpy) | via kernels bit-exact (or order-tolerance-defined) vs CPU; **on a congested stressor (TestBackplane / synthetic ρ≈1), not just the sparse fixture:** Metal converges (0 overuse), iterations within ~20% of CPU, **and passes the §C2 connectivity + DRC oracle** |
| **5. Validation + upstream** | Oracle comparisons (B4 + §C2) across fixtures + TestBackplane; perf measurement; split into upstream-sized PRs | **Gate = validation result:** oracle (per-net cost / legality / convergence + connectivity + DRC) matches CPU across both fixtures **and** TestBackplane; via kernels bit-exact; perf recorded. (Opening PRs is the deliverable, not the gate.) |

Suggested PR granularity for upstream, as a **dependency-ordered stack** (not all independent): (1) file-parser rewrite + tests, (2) KiCad-10 packaging/paths + kipy bump, (3) test-via fix — these three are independently mergeable. Then (4) backend seam and (5) Metal backend are a **2-PR sequence: PR 5 implements the KernelProvider protocol introduced in PR 4 and cannot compile or merge without it.** The seam is *not* a "pure refactor" to wave through — it threads a protocol through ~175 hot-path sites in the **~5.8k-line file / ~3.9k-line `PathFinderRouter` class**, which the maintainer explicitly flags as discuss-first; **open an issue before writing PR 4** (contributing.md requires it for large refactors and new GPU deps). Realistic expectation: PRs 1–3 are the contributions upstream will take; the seam is discuss-first; the Metal backend (PR 5, `extras_require={'metal': ['mlx']}`, MLX "ask-first" conversation) is likely **fork-carried** unless the maintainer opts in.

## 7. Constraints & risks

**Upstream constraints** (docs/contributing.md): PEP 8, Google docstrings, pytest tests in `tests/` (repo currently has zero — adding them is the top-wanted contribution), **no PyTorch**, ask before new core/GPU deps (MLX will need that conversation — raise it as a pre-PR issue and flag it in the **Metal-backend PR (PR 5)**, the only PR that introduces MLX; the seam PR stays dep-free), GitHub-only communication.

**Risks**
1. **MLX lazy-eval + Python wave-per-dispatch loop is the primary viability risk** — detecting frontier-empty termination may force a per-wave host readback (`mx.eval` + sync) that dominates runtime, and the ~32-arg launch must pack into Metal's 31-buffer bind table. Spike item 1 (§B5) de-risks this first; if it fails, MLX is not viable and the fallback is a full wgpu-py rewrite.
2. `mx.fast.metal_kernel` 64-bit `atomic_ulong` is not an officially documented path (header escape hatch makes it expressible) → **optional optimization**, not on the critical path; the 32-bit variant ships first and works on all M-series. Spike item 3 time-boxes it.
3. KiCad 10 IPC compat is verified on paper, not yet against 10.0.4 live → Phase 1 gate does it early, before any GPU work depends on it.
4. **macOS plugin runtime is Python 3.9 (KiCad-bundled), but MLX requires ≥ 3.10 → the Metal backend cannot `import` inside the in-KiCad plugin.** Decision (§1): Metal is **headless-first** on macOS; the in-plugin GPU path needs an external ≥ 3.10 interpreter (separate venv / subprocess) — verify what KiCad 10.0.4 actually uses for plugin venvs in Phase 1. Non-MLX runtime code stays 3.9-compatible regardless (CI/lint rule).
5. No routing has ever been exercised on this machine (baseline never got nets in) → Phase 0/1 gates fix observability first (0-nets hard fail, `[GPU-INIT]` surfacing).
6. Upstream may reject MLX as a dep → `extras_require={'metal': ['mlx']}` removes install-time imposition but is **not sufficient for merge**: contributing.md requires "ask first" for GPU deps and flags large refactors, and a CUDA/NVIDIA maintainer would own ~600 lines of MSL + host driver they can't test. Honest expectation: the Metal backend is **fork-carried** unless the maintainer opts in. CUDA/CPU stay untouched regardless.

## 8. Reference material

Full investigation reports live in `ghostlabs_docs/research/`:

- `gpu-porting-surface.md` — complete CUDA/CuPy touchpoint inventory with the kernel-by-kernel portability table
- `apple-gpu-backends.md` — MLX / wgpu-py / rejected-options research with URLs and Metal feature-table citations
- `kicad10-changes.md` — KiCad 10 release/IPC/kipy/file-format research with sources
- `testboard-fixture-analysis.md` — measured board statistics and fixture verdicts
- `baseline-run-2026-07-02.md` — the baseline run's exact environment, failures, and root causes

Companion doc: repo-wide architecture map in `CLAUDE.md` (written and adversarially verified 2026-07-02).
