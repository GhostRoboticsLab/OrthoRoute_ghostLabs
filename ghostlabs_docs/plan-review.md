# Adversarial Review — `ghostlabs_docs/plan.md`

**Date:** 2026-07-02 · **Method:** 20-agent adversarial workflow — every checkable claim re-verified against the actual code, the baseline re-run on this machine, external hardware/software facts web-checked, five strategic critics, and an independent skeptic re-checking each refutation. **Reviewing:** the Apple-Silicon-GPU + KiCad-10 master plan.

---

## Bottom line

The plan is **unusually well-grounded on facts and dangerously optimistic on scope, sequencing, and success criteria.** Almost every concrete, checkable assertion in it holds up: all four `file_parser.py` bugs, the baseline failures, the CUDA kernel line counts, the board statistics (including the exact 43/33/14 net split), and every external hardware fact (Metal feature tables, MLX, wgpu, KiCad-10 format `20260206`, the stale issue #19465) were **independently confirmed**. If you were worried the plan was built on hallucinated line numbers, it isn't.

The problems are of a different kind, and they matter more:

1. **The primary deliverable has an unresolved blocker the plan never reconciles** — MLX needs Python ≥3.10, the in-KiCad macOS plugin runs Python 3.9. As written, the Metal backend *cannot load in the plugin*.
2. **The main validation gate rests on a board OrthoRoute cannot route correctly** — the In1.Cu GND plane is flagged four times but never resolved, and `overuse==0` is treated as "correct" when it only means "converged."
3. **The de-risking spike targets the wrong risk** — "64-bit atomics" is elevated to "the one real hardware constraint" but is not load-bearing; the real integration unknowns get one under-emphasized line.
4. **The Metal-port effort is under-budgeted ~3–4×** — "~600 lines of MSL" counts only kernel bodies and silently drops the host-driver rewrite where the actual work lives.
5. **Several phase gates don't gate what they claim to.**

None of this is fatal to the *project*; all of it should change the *plan* before Phase 0 starts. Details below, tiered by how much they should change the plan.

---

## What survived the audit (credit where due)

These are verified — build on them without re-checking:

| Claim | Verdict |
|---|---|
| `file_parser.py` bugs at :225 (int netcode regex), :130/:155 (`fp_text reference`), :111 (non-greedy truncation), :385 (`'Cu'` substring counts `Edge.Cuts`→5 layers) | **All 4 VERIFIED** (truncation is ~499 chars not "~400", mechanism identical) |
| Baseline: Py 3.14.6, import chain OK, `--test-via` → `ValueError("No edges")` (raise at `unified_pathfinder.py:686`), `cli` → silent 0-net failure | **All reproduced honestly on this machine** |
| CUDA kernel line counts (active 217 / all 239 / compact 35 / backtrace 105 / validate 35 / 3 via kernels 148) | **VERIFIED exact** |
| Board stats: `20260206`, 4 Cu layers, 90 nets, already-routed (1150 seg/59 via main; 1005/31 mini), **43 TH-only / 33 SMD-only / 14 mixed = 90** | **VERIFIED exact** |
| Escape-planner bugs: drill>0 skip at `pad_escape_planner.py:239-241`; F.Cu hardcode at :766-769 | **VERIFIED** |
| Hazard anchors: cupy double-import (:506 `cp=np` vs :521 `cp=None`), GPU-fail-skips-CPU (:4520-4529), composition points (:2132-2164, :2178-2197), `routing_layers=range(1,layer_count-1)` (:1141/:1199/:1214), class/alias lines (:564/:1979/:5816) | **VERIFIED** |
| External facts: Metal atomics family map (M1=Apple7 none / M2=Apple8 ulong min-max macOS / M3-M4=Apple9 full / M5=Apple10); MLX 0.31.2, Py3.10-3.14, `mx.fast.metal_kernel` raw MSL + atomics, float64 CPU-only, `mlx[cuda]`; wgpu `shader-int64-atomic-min-max` on Metal Apple9/Apple8+Mac2; KiCad 10.0.0 (2026-03-20) toolbar regression fixed 10.0.1; `SEXPR_BOARD_FILE_VERSION=20260206`, netcodes dropped at 20251028; #19465 was a pre-9.0.0 Windows *load* failure not a PCM crash; Taichi halted; jax-metal dead | **VERIFIED** (2 items "needs-citation": exact kipy 0.6.0→pynng mapping, PCM `runtime:"ipc"` since 9.0.1) |

---

## Tier 1 — Load-bearing problems (fix before implementing)

### 1.1 · The Metal backend can't run where the mission says it must — Python 3.9 vs MLX 3.10+
**HIGH.** Plan §A0/Risk 4 states the macOS KiCad plugin runtime is bundled **Python 3.9**; §B1 states MLX requires **Python 3.10–3.14** (externally confirmed). The router runs in-process on the Qt thread inside the plugin (CLAUDE.md). So under KiCad's own interpreter the `import mlx` fails and Apple-GPU acceleration is **unreachable for real plugin users** — only the dev venv (3.14) and the headless/cloud box benefit. Yet Mission #1 is *"Routes on Apple M-series GPUs"* for the plugin. The plan never reconciles this.

**Do:** Make an explicit decision and put it in the mission: either (a) the Metal backend is **headless/ORP-only** on macOS (state it plainly — the in-KiCad GUI stays CPU), or (b) specify how a ≥3.10 interpreter reaches the plugin (external venv / subprocess worker). Verify what interpreter KiCad 10.0.4 actually uses for IPC-plugin venvs on *this* Mac during Phase 1 — this gates whether the headline feature is even deliverable in-plugin.

### 1.2 · The "positive smoke fixture" is self-contradictory — the In1.Cu GND plane has no plan
**HIGH.** §C0 notes In1.Cu is a solid GND pour the router can't see; §C1 then makes that same board the *positive* fixture that must route 33–47 nets with **0 overuse in seconds**. These are incompatible. The lattice builder unconditionally lays H/V edges on every inner layer; there is no "this layer is a plane, don't route on it" mechanism. So either the router lays signals **across the GND pour** (electrically invalid copper that only looks converged because overuse counts edges, not planes), or you honor the plane and lose the only horizontal inner layer → honest **ρ ≈ 0.96–0.99** (the plan's own number) that will *not* converge in seconds. And ORP v1.0 doesn't serialize zones, so the headless validation path **can't even tell the router In1.Cu is a plane.** The plan flags this four times and resolves it zero times, then builds Phase 1 and the entire CPU-vs-Metal regression gate on top of it.

**Do:** Before calling any board a "positive" gate, pick one: (a) mark plane layers non-routing in the lattice (needs zone extraction the plan doesn't have; then restate the "seconds/0-overuse" bar because honest ρ likely fails it), or (b) use a genuinely plane-free signal-only fixture (or synthetic board) for the positive gate and demote both TESTBOARD boards to **parse/plumbing** smoke tests. Do not let `overuse==0` stand in for "electrically valid" while a plane is invisible.

### 1.3 · `overuse==0` is not a correctness oracle — nothing validates the router's actual job
**HIGH.** Every gate in §5/§6 is phrased as "converges, 0 overuse." Per the invariants, that only means edge-capacity(=1) is satisfied. It says nothing about (i) whether each net is actually electrically connected — nets whose pads lack portals are *silently dropped* by `_parse_requests`, so "33–47 nets routed" can include partially/unconnected nets with no assertion catching it — or (ii) DRC (CLAUDE.md documents 300–500 known via-barrel violations on clean runs). The plan never runs KiCad DRC on committed output and never asserts pad-to-pad connectivity. Every downstream Metal-vs-CPU comparison inherits this blind spot.

**Do:** Add two hard oracles independent of overuse: (1) per-net connectivity (every routed net = one connected component spanning all its portal-eligible pads); (2) KiCad-10 DRC (or standalone clearance/short check) on committed geometry, recorded as a tracked metric with an explicit tolerance for the known via-barrel class. State that `overuse==0` is a convergence signal, not a correctness signal.

### 1.4 · The de-risking spike de-risks the wrong thing — "64-bit atomics" is not load-bearing
**HIGH.** §B2 is titled *"The 64-bit atomics question (the one real hardware constraint)"*, Risk #1 is the undocumented `atomic_ulong` path, and the headline spike item is validating `atomic_ulong` on the M4. The plan's own analysis dismantles the premise three ways: (a) the **default rollout is "32-bit everywhere first … works on all M-series"** — needs zero 64-bit atomics; (b) the dev machine is an **M4 = Apple9 with the full 64-bit set**, so the "constraint" can't even be exercised as a blocker here; (c) the **#1 hot kernel `wavefront_expand_active` is already 32-bit in shipping CUDA** (the packed-u64 `atomicMin` is a different, non-hot path). So packed-u64 is a *later-stage optimization affecting only M1 exclusion*, not a prerequisite for a working backend. Meanwhile the **real** primary-vs-fallback determinant gets one line (spike item 4): whether MLX's lazy-eval graph + a Python wave-per-dispatch loop can detect frontier-empty termination without a per-wave host readback (a readback every wave can dominate runtime), whether the ~32-arg kernel signatures survive Metal's **31-buffer bind limit**, and whether `simd_ballot`/shuffle assumptions hold on Apple simdgroups.

**Do:** Demote 64-bit atomics to "an Apple8+ optimization for the non-hot packed-key path." Make the spike's *primary* measurement the lazy-eval dispatch loop + frontier-empty readback cost on a ~10M-node wavefront, plus the 31-buffer struct-packing and simdgroup semantics. Keep the `atomic_ulong` prototype as a secondary, time-boxed check. (Also: the packed-u64 fast path works on **Apple8+/M2+**, not "Apple9+" as §B2 line 121 says — the plan under-gates its own optimization by a GPU generation.)

### 1.5 · The Metal port is under-budgeted ~3–4× — "~600 lines of MSL" is kernel text only
**HIGH.** The 5 live Dijkstra kernel *bodies* already total ~636 CUDA-C lines by direct count, so "~600 lines MSL" is a 1:1 translation of the kernel text and **budgets nothing for the host-side driver**, which is where the porting effort lives. That layer in `cuda_dijkstra.py` is large and deeply CuPy-coupled: `find_path_fullgraph_gpu_seeds` (284 lines), `find_path_roi_gpu` (239), `find_paths_on_rois`, plus three `_expand_wavefront_*` dispatchers, with ~144 CuPy host-API calls and ~179 references to persistent stamp/pool buffers — none of it MSL, all of it re-implemented against MLX's buffer/eval/readback model. Realistic `MetalDijkstra` scope is **~1,500–2,500 lines of new host Python plus the ~600 of kernel text.**

**Do:** Re-scope Phase 4 into two budgets: (a) MSL kernel bodies (~600, plausible) and (b) MLX host-driver reimplementation (estimate separately — the larger number). Put the combined figure in the Phase 4 gate. Prototype the 31-buffer struct-bind in the Phase 3 spike (it interacts with 1.4).

---

## Tier 2 — Ordering & gating defects

### 2.1 · The spike is scheduled *after* the seam it's supposed to de-risk (circular)
**HIGH.** §B5 is literally titled *"De-risking spike (before committing to the seam refactor)"*, but the §6 table orders Phase 2 (ArrayModule + KernelProvider seam) **before** Phase 3 (the spike). This is load-bearing, not cosmetic: `ArrayModule` presupposes a CuPy-shaped array backend, which is only true for the MLX "primary." If the spike selects the **wgpu-py fallback** ("no array layer at all"), the entire ArrayModule half of the Phase-2 seam is **dead work.**
**Do:** Move Phase 3 before Phase 2 (matching B5's own text). At minimum split the seam — land the backend-agnostic `KernelProvider` in Phase 2, defer `ArrayModule` until the spike confirms MLX.

### 2.2 · Phase 4's Metal gate runs only on the fixture the plan says can't stress the kernels
**HIGH.** Gate = "via kernels bit-exact; main fixture converges (0 overuse), iterations within ~20% of CPU." But §5 says the main fixture is 100k nodes (**1.25%** of design target) at **ρ≈0.5 (sparse)** and "never serve[s] as a performance benchmark." A sparse board generates little edge overuse and few via conflicts, so the negotiated-congestion loop, the hotset, and the racy wavefront atomics — exactly where a new Metal kernel goes wrong — are **barely exercised.** The Metal port can pass Phase 4 green while broken under real congestion; the congested test is deferred to Phase 5, i.e. after the port is declared done.
**Do:** Pull a congested stressor (TestBackplane or a synthetic ρ≈1 lattice) into the **Phase 4** gate. A correctness gate for a race-prone parallel kernel must include a board that actually produces overuse and via conflicts.

### 2.3 · Phase 5's gate is paperwork, not validation
**HIGH.** Phase 5's *work* is oracle comparisons + perf; its *gate* is "PRs opened per contributing.md." You can open well-formatted PRs on top of failing oracle runs. The exit criterion is disconnected from the phase's goal.
**Do:** Make the gate the validation result (oracle matches across both fixtures **and** TestBackplane; via kernels bit-exact; perf recorded). Opening PRs is the deliverable, not the gate.

### 2.4 · The test suite is scheduled *after* the phases whose gates depend on it
**MEDIUM.** §6 says "nothing lands without its gate passing," and Phases 2–4 gates are convergence/oracle comparisons — but the pytest suite that makes those trustworthy is parked in Phase 5 ("upstream's #1 ask"). Until then the safety net for a 110-site seam refactor plus a full kernel rewrite is one repaired `--test-via` and a single hand-run fixture.
**Do:** Move the lattice/CSR/via-accounting/oracle-harness pytest tests to Phase 0–1 (they need no GPU and are the upstream-wanted deliverable). Let them *be* the gates.

### 2.5 · Phase 2 is billed "pure refactor / CUDA-neutral" but bundles behavior changes and leans on a one-shot cloud run
**MEDIUM.** The Phase-2 gate demands "CPU-only routing identical pre/post," yet the same phase changes fallback semantics (GPU-fail → add CPU fallback at :4520-4529; silent init downgrade → surfaced banner). Those are functional changes; "identical" only holds because the CPU-only gate never takes the GPU path being changed — so the changed code is **untested by the gate that certifies it.** The neutrality proof is "rent a cloud GPU for one run," a manual, non-reproducible check for a 110+-site change.
**Do:** Split the fallback/observability fixes into their own small PR with a fault-injecting `KernelProvider` test that forces a GPU failure and asserts CPU fallback + counter. Replace the ad-hoc rental with a reproducible golden-diff: `.ORP` → `main.py headless` on any CUDA host (free Colab T4/spot) before/after, assert `ORS_before == ORS_after`.

*(Also flagged, lower severity: Phase 0's gate validates only `--test-via`, not the stripped fixtures or the 0-net hard-fail it also produces — the fixtures are the actual regression gate for the GPU work and go unvalidated until Phase 2. Phase 1's parser gate counts 90 nets/329 pads but doesn't assert the 4-copper-layer fix or net-membership correctness — the exact bugs A1 found. Phase 1's "commits a route" half is manual/live-KiCad-only and can't be a pytest gate.)*

---

## Tier 3 — Factual corrections to the plan (verified against code)

### 3.1 · The "6-method live GPU surface" is really **5** — `detect_barrel_conflicts_gpu` is dead code
**REFUTED, confirmed by skeptic.** Grep finds **zero callers** of `ViaKernelManager.detect_barrel_conflicts_gpu` anywhere in the live engine; its kernel is compiled but never dispatched. Live barrel-conflict detection is the **numpy** `PathFinderRouter._detect_barrel_conflicts` (`unified_pathfinder.py:5373`, called at :3774). Also `find_paths_on_rois` has no direct call in `unified_pathfinder.py` — it's only invoked indirectly inside `find_path_roi_gpu`. **Impact:** the KernelProvider needs to port **5** GPU methods, not 6; the barrel kernel doesn't need a Metal twin at all; and the §B4 via-kernel "bit-exact vs CPU twin" oracle only has **2** live kernels to exercise (`hard_block`, `apply_via_penalties`), not 3.

### 3.2 · TESTBOARD pad counts don't reconcile
**HIGH (internal contradiction).** §C0 table: main "329 (200 SMD: 175 F.Cu/25 B.Cu; 123 TH)" — but 175+25=200 and 200+123=**323**, not 329 (off by 6). Mini "332 (203 SMD: 95 F.Cu/108 B.Cu; 123 TH)" — 95+108=203 and 203+123=**326**, not 332. Independently, the pad-type parse found 129 thru_hole + 6 np_thru_hole + 200 smd = **335** tokens vs 329 `(pad` lines. The counts are unreliable, yet the **Phase-1 gate hard-asserts "329."** A correct parser returning a different number would fail the gate.
**Do:** Re-count from the parser once written, identify the ~6 stray pads (likely NPTH/mounting/edge-connector), and make the gate assert whatever the parser is actually expected to return. Also drop or qualify "identical netlist" for the mini — the 175/25 vs 95/108 F/B split proves the boards are physically different.

### 3.3 · The `~110` duck-typing count is a ~1.6× undercount; "one device gate" is wrong
**REFUTED, confirmed by skeptic.** `hasattr(...,'get')`-shaped sites in `algorithms/manhattan` = ~126–175 (not ~110); `.get(` calls ~600. `hasattr(costs,'device')` appears at **3** sites (:3107, :3170, :4468) plus a fourth `hasattr(via_col_pres,'device')` at :3201 — not "one." **Impact:** the seam refactor (Tier-1 scope concern) is bigger than stated, and — worse — the migration is mis-framed as a mechanical helper-swap when each `.get()` needs per-site triage (device readback vs. plain `dict.get`); a single misclassification is a silent wrong-backend transfer, and there's no test suite to catch it.

### 3.4 · "Silent" backend-init downgrade isn't silent
**PARTIAL, confirmed by skeptic.** The cited init-**failure** branch (`unified_pathfinder.py:2184-2197`) *does* log `logger.warning("[GPU] Failed to initialize CUDA Dijkstra")` at :2194 — console-visible. Only the separate non-failure "CPU-only mode" path logs at INFO (:2205, console-suppressed). The hazard is real (there's no startup banner) but the "silent / no WARNING" framing is inaccurate for the branch cited.

### 3.5 · Via CPU "twins" are not "line-for-line" and are not "wired"
**PARTIAL, confirmed by skeptic.** Three CPU twins do exist at `via_kernels.py:570-699` and *functionally* mirror the GPU kernels (so an equivalence oracle is supported *in principle*) — but they use 2D indexing and drop the `Ny` arg vs. the GPU's flattened `col_idx=xu*Ny+yu` layout, so an oracle must **adapt input shapes** (not a literal copy). And **nothing imports or calls the `_cpu` twins** — they're standalone/dead, "importable" not "wired." A real comparison also needs GPU hardware. Combined with 3.1, only 2 of the 3 twins have a live GPU counterpart to compare against.

### 3.6 · Number hygiene (LOW, but fix for a plan that will be read by the maintainer)
- **Embedded-CUDA totals don't sum internally:** 780 live + 1,530 dead = 2,310, leaving ~1,790 of the "~4,100" unexplained. (The ~4,100 denominator itself *is* defensible — it needs the ~494 lines of mixin kernels counted, which reconciles to ~4,271 — but the plan should show the arithmetic.)
- **Monolith size:** "5.8k-line" (whole file) vs contributing.md's "3,936-line" / CLAUDE.md's "~3,800" (the class). State "~5.8k-line file / ~3.9k-line class."
- **ρ quoted as both** 0.495/0.482 (plane-blind) **and** 0.96–0.99 (plane-honored); the positive gate silently uses the sparse figure. Attach the honest-ρ caveat to the gate, not just to C0.
- **A2-1 gap:** the plan's build.py line list is accurate but non-exhaustive — it misses `build.py:313` `"kicad_version": "9.0"` in the **PCM metadata JSON**, which is functionally more important than the human-readable INSTALL.txt strings it does list. Add it to the fix list.

---

## Tier 4 — Backend & correctness gaps worth a paragraph in the plan

- **Vulkan/MoltenVK is rejected for "no 64-bit atomics"** — but the plan's *default* design uses only 32-bit + float atomics, which MoltenVK supports. The rejection outcome may still be right (translation overhead, no array layer, no zero-copy), but the stated *reason* is the same non-load-bearing 64-bit framing from §B2. Restate on real merits. (MEDIUM)
- **Cross-backend oracle is too weak and its pass/fail is undefined.** FP reduction order differs between NumPy/Accelerate-on-ARM and MLX's async command buffers → different tie-breaks → different overused-edge sets → legitimately >20% iteration differences **without a bug**. The plan asserts "determinism within a backend via seed 42" but never establishes Metal is even *run-to-run* deterministic under lazy eval, and the "bit-exact via kernels" bar may be unachievable if any kernel does an unordered float reduction. Define the tolerance, its justification, and the action on exceedance; verify Metal determinism empirically before relying on it. (MEDIUM)
- **No net-class / width / clearance awareness.** VBUS_PIX (27 pads) and GND (66 pads) are power nets; routing them as unit-width Manhattan traces on signal layers is electrically wrong, and high-fanout nets need MST/Steiner decomposition the plan never mentions (and whose determinism feeds the oracle concern). Scope power/plane nets out explicitly, or state net-class as a known non-feature so "router validation" isn't overclaimed. (HIGH/MEDIUM)
- **Obstacle/keepout coverage unstated.** 123 TH pads, mounting holes, and board cutouts may not become blocked lattice nodes; if not, signal traces can run through them and `overuse==0` accepts it. ORP v1.0 also drops keepouts, so the headless path can't carry them. Verify/assert, or route obstacle checks through live-IPC. (MEDIUM)
- **The B.Cu-escape bug also poisons the "positive" main fixture** — it has 25 B.Cu SMD pads, and `_get_pad_layer` hardcodes F.Cu, so the "cleanly positive" main / "cleanly negative" mini partition doesn't hold. (MEDIUM)
- **`torch.mps.compile_shader`** (stable since ~2.10) is the most battle-tested raw-MSL-from-Python path and a real hedge against the undocumented-MLX-atomics risk — correctly banned upstream, but Risk 6 already contemplates a fork-only Metal backend, where the ban is self-imposed. Worth one line as a fork-only contingency. (LOW)

---

## Recommended concrete edits to `plan.md`

1. **§1 Mission / §B / Risk 4:** resolve Python 3.9-vs-MLX-3.10 — declare Metal headless-only, or specify the interpreter path. (Tier 1.1)
2. **§C1:** stop calling the plane-bearing TESTBOARD board a "positive" gate; either add plane-handling + restate the ρ/seconds bar, or pick a plane-free/synthetic positive fixture and demote TESTBOARD to parse smoke tests. (Tier 1.2)
3. **§5/§6 gates:** add connectivity + DRC oracles; state `overuse==0` is convergence, not correctness. (Tier 1.3)
4. **§B2/§B5:** demote 64-bit atomics; make the spike measure lazy-eval dispatch/readback + 31-buffer packing first; fix the Apple8-vs-Apple9 gate wording; state one plan of record (32-bit-everywhere first, u64 as an Apple8+ fast path later). (Tier 1.4, 1.5)
5. **§6 table:** move the spike (Phase 3) before the seam (Phase 2); make Phase 4's gate include a congested stressor; make Phase 5's gate the validation result; pull the pytest suite to Phase 0–1. (Tier 2)
6. **§B0/§B3/§B4:** correct the live GPU surface to **5 methods** (barrel detection is numpy), the duck-typing count to ~175 / 3-4 device gates, and the via-oracle to 2 live kernels; note the twins need shape-adaptation and aren't wired. (Tier 3.1/3.3/3.5)
7. **§C0/§6:** reconcile the pad counts (323/326 vs 329/332) and make the Phase-1 gate assert the real number; qualify "identical netlist." (Tier 3.2)
8. **§7/§6:** drop "independently mergeable" for PR 5 (it requires PR 4); state the honest upstream expectation (parser/packaging/test-via are the realistic contributions; the seam is discuss-first; the Metal backend is likely fork-carried); move the MLX "ask-first" to the Metal PR, not the "CUDA-neutral" seam PR. (Tier 3/4)
9. **Number hygiene:** fix the CUDA-line arithmetic note, the 5.8k/3.9k monolith figure, the dual-ρ presentation, and add `build.py:313` to the packaging fix list. (Tier 3.6)

---

*Method note: 6 code-verification clusters (each refutation re-checked by an independent skeptic), 1 baseline reproduction run in the repo venv, 1 web-sourced external-fact check, 5 strategic critics. 20 agents, 0 errors, ~609k tokens. Skeptics reversed 2 first-pass downgrades (build.py `9.0` and the 780/4100 count) back to VERIFIED — reflected above. Full structured output: `tasks/wpjfr0527.output`.*
