# GPU Porting Surface Audit (CUDA/CuPy → Metal)

> Investigation report supporting `ghostlabs_docs/plan.md` (2026-07-02).

# OrthoRoute GPU-Porting Surface Map (CUDA/CuPy → Metal audit)

All paths relative to `/Volumes/MacSSD1/DeveloperWorkspace/Projects/OrthoRoute_ghostLabs`. No files were modified.

## 1. Complete CUDA/CuPy touchpoint inventory

### 1a. Live GPU modules (on the runtime path)

| File | Lines | Role |
|---|---|---|
| `orthoroute/algorithms/manhattan/unified_pathfinder.py` | 5,818 | Engine (`PathFinderRouter`, aliased `UnifiedPathFinder` at :5816). Dual `import cupy` guards at :502-507 (`CUPY_AVAILABLE`, sets `cp = np` on failure!) and :517-522 (`GPU_AVAILABLE`, sets `cp = None`). `CUDADijkstra` import guard :529-534 (`CUDA_DIJKSTRA_AVAILABLE`). |
| `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` | 5,751 | `CUDADijkstra` — 11 `cp.RawKernel` compilations in `__init__` (:151-1921), ~1,717 lines of embedded CUDA C, plus ~3,800 lines of host driver code (much dead). |
| `orthoroute/algorithms/manhattan/pathfinder/via_kernels.py` | 699 | `ViaKernelManager` — 3 compiled RawKernels (:169-180) + 1 orphaned kernel string; ~193 CUDA-C lines; NumPy CPU fallbacks (:570-699). |
| `orthoroute/algorithms/manhattan/pathfinder/persistent_kernel.py` | 334 | 1 cooperative-groups RawKernel (~205 CUDA-C lines), compiled with `enable_cooperative_groups=True` (:236). **Disabled** — gate `_enable_persistent_kernel = False` at `cuda_dijkstra.py:145`. |
| `orthoroute/infrastructure/gpu/cuda_provider.py` | 212 | `CUDAProvider(GPUProvider)` — memory pool (`cp.get_default_memory_pool().set_limit`, :48-56), `cp.cuda.runtime.memGetInfo` (:52), null-stream sync (:202). Instantiated by `kicad_plugin.py:51` and `main_window.py:3251` but **never handed to the engine** — availability-logging ceremony only. |
| `orthoroute/infrastructure/gpu/cpu_fallback.py` | 170 | NumPy no-op provider (`CPUFallbackProvider`/`CPUProvider`). |
| `orthoroute/application/interfaces/gpu_provider.py` | 57 | `GPUProvider` ABC (10 abstract methods: is_available/initialize/cleanup/get_device_info/get_memory_info/create_array/copy_array/to_cpu/to_gpu/synchronize). Not used by the live engine. |

Touchpoints inside `unified_pathfinder.py` (live engine):
- **CSRGraph** :647-793 — `self.xp = cp if use_gpu else np` (:652); GPU radix sort `cp.argsort(kind='stable')` + `.get()` during graph build (:700-712); GPU upload of CSR + edge_layer/edge_kind (:777-785).
- **EdgeAccountant** :799-1068 — pure xp array math (`self.xp` at :805; `present/present_ema/history/capacity` :808-811; `update_costs` :973-1043; `update_history` :898-961; `update_present_ema` :963-971). Per-element scalar GPU indexing in `commit_path`/`clear_path` (:821-836).
- **ROIExtractor** :1331+ — takes `use_gpu` but the live `extract_roi_geometric` is pure Python/NumPy sets and loops (:1337-1520).
- **SimpleDijkstra** (CPU oracle) :1770-1972 — copies CSR to CPU via `.get()` (:1775-1776); optional GPU delegation `self.gpu_solver.find_path_roi_gpu(...)` when ROI > 1,000 nodes (:1790-1804).
- **Engine init** :2069-2206 — GPU gates (see §4); via capacity arrays allocated as `cp` arrays (:2137-2154); `cp.cuda.Device().mem_info` (:2189-2190); `ViaKernelManager` (:2164); `CUDADijkstra` attach (:2186); `ROIExtractor` (:2206).
- **Per-iteration cost pipeline** — via pooling penalties GPU kernel / CPU fallback (:3198-3316); via hard-block GPU/CPU (:3402-3478); barrel-conflict GPU detection + GPU penalty write (:3793-3798); dozens of `accounting.use_gpu` branches with `.get()` / `cp.asarray()` round-trips (:3232, :3315-3316, :3377-3378, :3437-3478, :3576-3578, :4239-4240, :4333-4334, :4744-4756, :4794, :4913-4915, :5178-5202).
- **Per-net routing loop `_route_all`** :4367-4700 — GPU supersource fast path (:4460-4529) calling `find_path_fullgraph_gpu_seeds` (:4484); gate is `hasattr(costs, 'device')` (:4468).
- **Via metadata → GPU** :5337-5345 (`cp.asarray` of indices/xy/z_lo/z_hi).

### 1b. Dead/decoy GPU code (import-time load-bearing only — must stay import-clean but not ported)
- `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra_original.py` — 5,271 lines, 11 kernel strings (~1,662 CUDA-C lines). Dead per CLAUDE.md.
- The 7 `*_mixin.py` files (11,115 lines total; all `try: import cupy` at :14) — abandoned refactor, imported by `pathfinder/__init__.py:145-151` so syntax must stay valid. GPU-relevant ones: `roi_extractor_mixin.py` (4 RawKernels: `roi_extraction_kernel` :206, `near_far_multi_roi` :1305, `astar_expansion_kernel` :2261, `multi_roi_astar_kernel` :2588; `cp.cuda.runtime.getLastError` :286, CUDA events :390-393), `negotiation_mixin.py` (`apply_mult_kernel` :1360), `pathfinding_mixin.py`, `graph_builder_mixin.py`, `lattice_builder_mixin.py`, `geometry_mixin.py`, `diagnostics_mixin.py`.
- Ceremony: `orthoroute/__init__.py:32-33` (imports CUDAProvider at package import), `shared/configuration/settings.py:92-122` (`GPUSettings` — cuda_streams etc., read by nothing live), `shared/exceptions/base_exceptions.py:76-87` (`GPUError`), GUI checkbox writes `ORTHO_GPU` env that nothing reads (`main_window.py:1784`).

## 2. Classification of touchpoints

**(a) Generic xp array math (ports nearly free to any NumPy-like backend):**
- `EdgeAccountant` in its entirety (~270 lines) — the PathFinder cost model (`update_costs`, `update_history`, `update_present_ema`, overuse computation) is pure `xp.maximum/minimum/where/arange` math.
- `CSRGraph` arrays and the CPU-side CSR build; GPU is only used for one `argsort` (:708) and array upload (:777-785).
- ~24 `self.xp.` sites, all in `unified_pathfinder.py` (:652-1333 region).

**(b) CuPy-specific host API calls (need backend shims):**
- `cp.RawKernel` compile/launch: `cuda_dijkstra.py` (11 kernels), `via_kernels.py:169-180`, `persistent_kernel.py:233-237` (the only `enable_cooperative_groups=True`).
- Device/memory queries: `cp.cuda.Device().mem_info` — `cuda_dijkstra.py:2304, 2410, 3750`; `unified_pathfinder.py:2189`; `cp.cuda.runtime.memGetInfo` — `cuda_provider.py:52`. Used to size the K_pool stamp pools (`cuda_dijkstra.py:2299-2355`).
- Memory pool: only `cuda_provider.py:48-56` (ceremonial layer).
- Streams: **only the null stream** — `cp.cuda.Stream.null.synchronize()` at `via_kernels.py:259,338,400`; `cuda_dijkstra.py:3483,3961,4057,4200,4236,4641,4667`; `persistent_kernel.py:327`; `cuda_provider.py:202`. No multi-stream, no async pipelining.
- CUDA events for timing: `cuda_dijkstra.py:3074-3122, 3201-3315` (+dead mixin :390-393).
- `cupy.lib.stride_tricks.as_strided` stride-0 broadcast for shared CSR (`cuda_dijkstra.py:2372-2386`) — load-bearing zero-copy trick; kernels take explicit `*_stride` params (0 = shared), so a Metal port can pass one buffer + stride scalars instead.
- `cp.unpackbits` (:2661-2662, :3115), `cp.nonzero`, `cp.count_nonzero` (hot-loop termination checks :2940, :3066, :5679-5715), `cp.unique`, `cp.asnumpy` (:3145).
- Host↔device transfers: ~70 `.get()` sites in `unified_pathfinder.py`, ~40 in `cuda_dijkstra.py` (duck-typed as `hasattr(x, "get")` — this idiom is the de-facto backend detector throughout).
- `cupyx.scipy.sparse`: imported (`cuda_dijkstra.py:13`) but used only in dead code (`find_paths_bidirectional_batch`, :5429). **No cub/thrust, no ElementwiseKernel/ReductionKernel/cp.fuse anywhere.**

**(c) Hand-written CUDA C kernels:** see §3.

## 3. RawKernel catalog

### cuda_dijkstra.py (11 kernels, ~1,717 CUDA-C lines)

| # | Kernel (compile site) | Computes | ~Lines | CUDA-specific features | Status |
|---|---|---|---|---|---|
| 1 | `relax_edges_parallel` (:151-194) | Classic per-ROI single-node Dijkstra edge relax; packed 64-bit dist‖parent | 43 | **native `atomicMin` on `unsigned long long`**, `__float_as_int`/`__int_as_float` | Compiled, **never launched** (host method `_relax_edges_parallel` :1968 uses CuPy vectorized ops instead) |
| 2 | `wavefront_expand_all` (:199-438) | Full-scan BFS/A* wavefront, one block per ROI, grid-stride over all nodes; bit-packed frontier; stride-0 shared CSR; dual mode: legacy float-min or cycle-proof 64-bit atomic keys | 239 | `atomicCAS`-loop float-min, **CAS-loop `atomicMin64` on 64-bit keys** (Iter≥2 path), `atomicExch`, `atomicOr`, A* Manhattan heuristic, H/V discipline validation | **LIVE** — dense-frontier path of `_expand_wavefront_full_scan` (:3358-3491) |
| 3 | `wavefront_expand_active` (:444-661) | Compacted active-list wavefront: one thread per frontier node; ROI bbox + owner-bitmap gates, round-robin layer bias, deterministic hash jitter | 217 | `atomicCAS`-loop float-min (32-bit only — **no 64-bit atomics**), `atomicExch` parent (known benign race), `atomicOr` frontier, `atomicAdd` counter, `__ldg` | **LIVE — the #1 hot kernel** (launched from `_expand_wavefront_compacted` :3302 whenever frontier sparsity < 50%, which is nearly always) |
| 4 | `wavefront_expand_procedural` (:666-857) | CSR-free 6-neighbor arithmetic relax from direction weight tables | 191 | float-min CAS, `__ldg`, `atomicExch/Or` | Compiled, **never launched** |
| 5 | `assign_nodes_to_buckets` (:862-916) | Delta-stepping bucket assignment | 54 | `__ldg`, `__float2int_rd`, `atomicOr` | **Dead** — delta stepping hard-disabled (`if False:` at :2814; `GPUConfig.USE_DELTA_STEPPING = False` at `unified_pathfinder.py:555`) |
| 6 | `sssp_persistent_cooperative` (:921-1149) | Grid-resident persistent SSSP with device ping-pong queues | 228 | **cooperative groups `cg::grid_group`/`grid.sync()`**, `__launch_bounds__(256)`, float-min CAS, `atomicAdd` queue counters | **Disabled** (`GPUConfig.USE_PERSISTENT_KERNEL = False`, `unified_pathfinder.py:553` — comment: "Hangs on cooperative kernel launch") |
| 7 | `sssp_persistent_stamped` (:1154-1684) | Persistent SSSP + uint16 stamp pools + device-side backtrace + via segment-pooling prefix sums + RR bias + jitter | 530 | `__shared__` memory, **device `printf`** (:1409), CAS-loop `atomicMin64`, `__float2uint_rn` scaled-int keys, `atomicAdd/Exch/Or`, stamp accessors | **Dead** — only launcher `route_batch_persistent` (:3718) has zero callers |
| 8 | `compact_mask_to_list` (:1689-1724) | Warp-aggregated stream compaction of bitset frontier → index list | 35 | **warp intrinsics: `__ballot_sync`, `__popc`, `__shfl_sync`**, `atomicAdd` | **LIVE hot path** — every wavefront iteration (:3101) |
| 9 | `accountant_update` (:1728-1768) | Per-edge present/history/total-cost update | 40 | none beyond `powf/fmaxf/fminf` | Compiled, **never launched** (cost updates run as xp array math in `EdgeAccountant.update_costs`) |
| 10 | `validate_parents` (:1774-1809) | Parent-pointer vs CSR consistency check (diagnostics) | 35 | `atomicAdd` | LIVE (called in `_reconstruct_paths` :4191 on the ROI-batch path) |
| 11 | `backtrace_paths` (:1816-1921) | Per-ROI GPU backtrace sink→source w/ cycle detection, bitmap validation, optional 64-bit key parent decode | 105 | `int visited[4096]` thread-local array (heavy register/local memory), reads uint64 keys | LIVE for ROI batches with `max_roi_size > 100_000` (:4150, launch :4224); the full-graph seeds path backtraces on **CPU** instead (`parent_val_pool[0].get()` :5730) |

### via_kernels.py (4 kernel strings, ~193 CUDA-C lines)

| Kernel | Computes | ~Lines | Features | Status |
|---|---|---|---|---|
| `hard_block_via_capacity` (:34-86) | Sets `total_cost = inf` for via edges at column/segment capacity | 52 | `__int_as_float(0x7f800000)`, `atomicAdd` counter | **LIVE per iteration** (`unified_pathfinder.py:3402-3420`) |
| `apply_via_pooling_penalties` (:93-139) | Adds congestion penalty to via edge costs | 46 | **`atomicAdd(float*)`** on cost array | **LIVE per iteration** (`unified_pathfinder.py:3212-3221`) |
| `block_via_keepouts_owner_aware` (:413-458) | Owner-aware planar-edge blocking around via barrels | 45 | none notable | **Orphan** — string never compiled (`_compile_kernels` :166-180 skips it), no references |
| `detect_barrel_conflicts` (:465-515) | Counts committed edges touching other nets' via barrels | 50 | `atomicAdd`; quirk: `nullptr` flag arg passed as `cp.int32(0)` (:397) | **LIVE per iteration** (`unified_pathfinder.py:3768+` via `detect_barrel_conflicts_gpu` :346) |

### persistent_kernel.py
- `persistent_sssp_kernel` (:19-224, ~205 lines): full-graph persistent SSSP; cooperative groups `grid.sync()` (~12 sync points), CAS `atomicMin64`, float-min CAS, packed 64-bit keys. Compiled with `enable_cooperative_groups=True` (:236); launched with fixed `num_blocks = 80` ("80 SMs on RTX 4090", :296). **Disabled** at runtime.

### Metal portability read on the live kernels
- The **hot kernel** (`wavefront_expand_active`) needs only 32-bit atomics (CAS float-min emulation, or, exch, add) — all expressible in MSL (`atomic_uint` CAS loop; `atomic_fetch_or/add_explicit`). Its 40-parameter signature exceeds Metal's 31-buffer bind table → scalars must be packed into a constant struct (natural in MSL anyway).
- `compact_mask_to_list` maps cleanly to Metal SIMD-group ops (`simd_ballot`→`__ballot_sync`, `popcount`→`__popc`, `simd_broadcast`/`simd_shuffle`→`__shfl_sync`); Apple GPUs use 32-wide simdgroups, matching the warp-width-32 assumptions.
- The 64-bit atomic-key path (`wavefront_expand_all` Iter≥2, `backtrace_paths` atomic mode, all persistent kernels) is the main Metal hazard: MSL has no general 64-bit atomicCAS; `atomic_ulong` min/max exists only on newer Apple GPU families (M3/A17+, MSL 3.1 tier). Portable workarounds: quantize cost to 32 bits and pack cost‖parent into 2×32-bit with a retry loop, or accept the legacy 32-bit float-min + separate parent write (already the shipping behavior of the hot kernel, with CPU-side cycle detection in backtrace as the safety net).
- Cooperative-groups grid sync has **no Metal equivalent** — irrelevant in practice because every kernel needing it is already disabled/dead (the multi-launch Python loop + null-stream sync is the shipping design, and it maps 1:1 to Metal command-buffer dispatch loops).
- Device `printf` (dead kernel :1409) has no MSL equivalent — dead code anyway.

## 4. Abstraction seam analysis

**Where the code branches on GPU today (decision points):**
1. Import guards: `unified_pathfinder.py:502-507` & :517-522 (`GPU_AVAILABLE`), :529-534 (`CUDA_DIJKSTRA_AVAILABLE`); `cuda_dijkstra.py:11-21`; `via_kernels.py:16-25` (both define `_DummyCuPy` stubs).
2. Config/env: `PathFinderConfig.use_gpu = True` (:609); `USE_GPU` env override (:1990-1992); `main.py:452-461` (`--cpu-only`, `ORTHO_CPU_ONLY`); GUI `GPUConfig.GPU_MODE` (`main_window.py:98, 2771`).
3. Constructor-time composition (the real seam): `use_gpu=self.config.use_gpu and GPU_AVAILABLE` for `CSRGraph` (:2113), `EdgeAccountant` (:2121), `ViaKernelManager` (:2132-2164), `ROIExtractor` (:2206); `use_gpu_solver = config.use_gpu and GPU_AVAILABLE and CUDA_DIJKSTRA_AVAILABLE` → `self.solver.gpu_solver = CUDADijkstra(self.graph, self.lattice)` (:2178-2197, falls back to `None` on any exception — silent CPU fallback).
4. Call-time duck-typing: `hasattr(costs, 'device')` (:4468), `hasattr(x, 'get')` (everywhere), `self.accounting.use_gpu` branches, per-call try/except downgrades (:1802-1804, :2083-2086, :3223-3224, :4526-4529).

**The live GPU call surface is remarkably small.** A `MetalDijkstra` needs exactly three entry points to cover the hot path: `find_path_fullgraph_gpu_seeds` (`cuda_dijkstra.py:5467`), `find_path_roi_gpu` (:4909, reached via `SimpleDijkstra.find_path_roi` :1792-1797 for ROIs > 1,000 nodes), and transitively `find_paths_on_rois` (:2014, only caller is `find_path_roi_gpu` :4987). A `MetalViaKernelManager` needs three: `hard_block_via_edges` (:186), `apply_via_penalties` (:267), `detect_barrel_conflicts_gpu` (:346) — plus the two module helpers `convert_via_metadata_to_gpu`/`ensure_gpu_array` (:522-563). Everything else in `cuda_dijkstra.py` is dead weight.

**ArrayModule + KernelProvider difficulty: moderate-low.** The xp pattern already exists (`CSRGraph.xp`, `EdgeAccountant.xp`); the gaps are (a) the `hasattr(x,'get')`/`.get()` idiom (needs a `to_cpu()` helper), (b) `hasattr(costs,'device')` gate, (c) the two module-level `cp` singletons in `unified_pathfinder.py` that alias differently on failure (`cp = np` at :506 vs `cp = None` at :521 — a pre-existing inconsistency to clean up), (d) K_pool sizing from `Device().mem_info` (Metal: `MTLDevice.recommendedMaxWorkingSetSize`), (e) `as_strided` stride-0 broadcast (unnecessary on Metal — kernels already take stride scalars).

**CPU fallback as correctness oracle: yes, with caveats.** `SimpleDijkstra` (:1770-1972) is pure NumPy+heapq and, with `--cpu-only`, drives the entire pipeline (EdgeAccountant and via fallbacks all run on np); determinism is seeded (seed 42). Caveats: the GPU wavefront is explicitly *not* a strict Dijkstra ("WARNING: ignores cost ordering", :2821) and injects kernel-side jitter/RR-bias (:592-621), and Iter-1 relaxes H/V discipline — so GPU and CPU produce *different but equally valid* paths. The oracle must therefore compare **path cost, path legality (H/V/via adjacency), and iteration-level overuse convergence**, not exact node sequences. `via_kernels.py` is the exception where exact equivalence testing is possible: each kernel has a line-for-line CPU twin (`hard_block_via_edges_cpu` :570, `apply_via_penalties_cpu` :617, `detect_barrel_conflicts_cpu` :657).

## 5. Relative sizes & hot/cold split

- **Embedded CUDA C total:** ~4,100 lines across the repo; ~2,115 in live files (`cuda_dijkstra.py` 1,717 + `via_kernels.py` 193 + `persistent_kernel.py` 205); ~1,981 in dead files (`cuda_dijkstra_original.py` 1,662 + mixins ~319).
- **CUDA C that actually executes:** ~780 lines — `wavefront_expand_active` (217) + `wavefront_expand_all` (239) + `compact_mask_to_list` (35) + `backtrace_paths` (105) + `validate_parents` (35) + 3 via kernels (148).
- **Disabled/dead kernels:** ~1,530 lines — persistent×3 (963), procedural (191), delta bucket (54), relax (43), accountant (40), owner-aware blocking (45), plus dead-mixin kernels (~319).
- **Host xp/array code:** EdgeAccountant ~270 lines + CSRGraph ~150 + ~100 scattered `.get()`/`asarray` branch lines in `unified_pathfinder.py`; `cuda_dijkstra.py` host code ~4,000 lines of which roughly half serves dead entry points (`_run_delta_stepping` :4360-4674, `route_batch_persistent` :3718-4076, bidirectional :5148-5466, multisource-GPU :4761-4908, `find_path_batch` :2088-2216).
- **Hot path per net (GPU mode):** `_route_all` :4401 loop → `find_path_fullgraph_gpu_seeds` (K=1 full graph, `use_atomic_parent_keys=True` :5624) → Python iteration loop (≤2000) of `_expand_wavefront_parallel` :3044 → `compact_kernel` + `wavefront_expand_active` (or `wavefront_expand_all` when dense) → periodic `cp.count_nonzero`/`cp.min` sync checks (:5694-5718) → CPU backtrace via 20 MB `parent_val_pool.get()` (:5730). Note the compacted hot kernel runs the **32-bit legacy atomics path** (no 64-bit keys in its signature) even though the data dict requests atomic keys.
- **Hot path per iteration:** via hard-block + penalty kernels, barrel-conflict kernel, `EdgeAccountant.update_costs` xp math, plus several full-array `.get()`→modify→`cp.asarray` CPU round-trips (:3232-3316, :3427-3478) that are already CPU-bound and would port trivially.
- **ROI wavefront path** (`find_paths_on_rois`/`_run_near_far` :2770, stamp pools sized from GPU memory :2299-2355): live but secondary — reached only when the supersource fast path is unavailable (no portals / costs on CPU / no gpu_solver), via `SimpleDijkstra.find_path_roi` GPU delegation for ROIs > 1,000 nodes. When the GPU fast path fails for a net it is marked failed and CPU fallback is *skipped* for that net (:4520-4529).

## Key verified facts

- unified_pathfinder.py imports cupy twice with inconsistent fallbacks: line 506 sets cp = np on ImportError (CUPY_AVAILABLE), line 521 sets cp = None (GPU_AVAILABLE)
- cuda_dijkstra.py compiles 11 cp.RawKernel objects in CUDADijkstra.__init__ (lines 151-1921) totaling ~1,717 lines of embedded CUDA C; via_kernels.py compiles 3 (lines 169-180, ~193 CUDA-C lines); persistent_kernel.py compiles 1 with enable_cooperative_groups=True (line 236, ~205 lines)
- Only ~780 CUDA-C lines actually execute at runtime: wavefront_expand_active (cuda_dijkstra.py:444-661), wavefront_expand_all (:199-438), compact_mask_to_list (:1689-1724), backtrace_paths (:1816-1921), validate_parents (:1774-1809), and via_kernels.py's hard_block_via_capacity/apply_via_pooling_penalties/detect_barrel_conflicts
- Kernels compiled but never launched: relax_edges_parallel, wavefront_expand_procedural, accountant_update, assign_nodes_to_buckets (delta stepping hard-disabled via 'if False:' at cuda_dijkstra.py:2814); persistent kernels disabled via GPUConfig.USE_PERSISTENT_KERNEL=False (unified_pathfinder.py:553, comment 'Hangs on cooperative kernel launch') and _enable_persistent_kernel=False (cuda_dijkstra.py:145); route_batch_persistent (cuda_dijkstra.py:3718) has zero callers; OWNER_AWARE_BLOCKING_KERNEL (via_kernels.py:413) is never compiled
- The hot per-net GPU path is find_path_fullgraph_gpu_seeds (cuda_dijkstra.py:5467), called from _route_all at unified_pathfinder.py:4484 gated by use_portals + gpu_solver + hasattr(costs,'device') (:4462-4468); it drives a Python loop over _expand_wavefront_parallel (:3044) using compact_mask_to_list + wavefront_expand_active, and backtraces on CPU via parent_val_pool.get() (:5730)
- The hot kernel wavefront_expand_active uses only 32-bit atomics (atomicCAS float-min loop, atomicExch, atomicOr, atomicAdd) and __ldg — no 64-bit atomics; 64-bit CAS-loop atomicMin64 on packed cost|parent keys is used in wavefront_expand_all's Iter>=2 path, backtrace_paths' atomic mode, and all (disabled) persistent kernels; relax_edges_parallel (dead) uses native atomicMin on unsigned long long
- compact_mask_to_list uses warp intrinsics __ballot_sync/__popc/__shfl_sync (cuda_dijkstra.py:1707-1717); cooperative-groups grid.sync() appears only in the three disabled persistent kernels; device printf only in dead sssp_persistent_stamped (:1409); __shared__ memory only in sssp_persistent_stamped (:1404-1405)
- CuPy host API surface: only null stream (cp.cuda.Stream.null.synchronize at via_kernels.py:259,338,400; cuda_dijkstra.py:3483,4200,4236 etc.), cp.cuda.Event timing (:3074-3122,3201-3315), cp.cuda.Device().mem_info for K_pool sizing (:2304,2410; unified_pathfinder.py:2189), cupy.lib.stride_tricks.as_strided stride-0 shared-CSR broadcast (:2372-2386), cp.unpackbits/nonzero/count_nonzero/argsort; memory pool only in ceremonial cuda_provider.py:48-56; cupyx.scipy.sparse used only in dead code (:5429); no cub/thrust/ElementwiseKernel/ReductionKernel/fuse anywhere
- The xp = cp-or-np pattern covers CSRGraph (unified_pathfinder.py:652), EdgeAccountant (:805), ROIExtractor (:1333); the entire PathFinder cost model (update_costs :973-1043, update_history :898-961) is portable xp array math; ~70 .get() duck-typed transfer sites in unified_pathfinder.py
- GPU composition seam: unified_pathfinder.py:2178-2197 attaches CUDADijkstra as self.solver.gpu_solver (None on any exception = silent CPU fallback); ViaKernelManager at :2164; live GPU call surface is 3 solver methods (find_path_fullgraph_gpu_seeds, find_path_roi_gpu, find_paths_on_rois) + 3 via-kernel methods (hard_block_via_edges, apply_via_penalties, detect_barrel_conflicts_gpu)
- SimpleDijkstra (unified_pathfinder.py:1770-1972) is pure NumPy+heapq, delegates to gpu_solver.find_path_roi_gpu only when ROI > 1,000 nodes (:1790-1797); via_kernels.py has exact CPU twins for all 3 live kernels (:570-699); GPU wavefront is explicitly not cost-ordered ('WARNING: ignores cost ordering', cuda_dijkstra.py:2821) and adds kernel-side jitter and round-robin layer bias, so GPU vs CPU paths differ legitimately
- When the GPU supersource fast path fails for a net, the net is marked failed and CPU fallback is skipped within that iteration (unified_pathfinder.py:4520-4529)
- Dead GPU code that must stay import-clean: cuda_dijkstra_original.py (5,271 lines, 11 kernels) and the 7 mixins (11,115 lines; roi_extractor_mixin.py alone has 4 RawKernels at :206, :1305, :2261, :2588) — pathfinder/__init__.py:145-151 imports all mixins at module load
- CUDAProvider/GPUProvider (infrastructure/gpu/, application/interfaces/gpu_provider.py) is instantiated by kicad_plugin.py:51 and main_window.py:3251 but never passed to the engine — it is not the porting surface
- wavefront_expand_active takes 40 kernel parameters (cuda_dijkstra.py:487-529), exceeding Metal's 31-entry buffer argument table — scalars must be packed into a constant struct for an MSL port

## Recommendations

- Scope the Metal port to the ~780 lines of live CUDA C (wavefront_expand_active, wavefront_expand_all, compact_mask_to_list, backtrace_paths, validate_parents, 3 via kernels) — do NOT port the ~1,530 lines of disabled/dead kernels (persistent x3, delta stepping, procedural, relax, accountant) or anything in cuda_dijkstra_original.py/mixins
- Introduce the backend seam at the existing composition points: a KernelProvider protocol with exactly 6 methods (find_path_fullgraph_gpu_seeds, find_path_roi_gpu, find_paths_on_rois; hard_block_via_edges, apply_via_penalties, detect_barrel_conflicts_gpu) implemented by CUDADijkstra/ViaKernelManager today and MetalDijkstra/MetalViaKernelManager tomorrow, selected at unified_pathfinder.py:2178-2197 and :2132-2164
- For ArrayModule: formalize the existing xp pattern plus a to_cpu(x)/is_device_array(x) helper pair to replace the ~110 hasattr(x,'get')/.get() and hasattr(costs,'device') duck-typing sites; unify the two contradictory cp fallbacks (cp=np at :506 vs cp=None at :521) while doing so
- Handle the 64-bit atomic-key problem by either (a) quantizing cost to 32 bits (the dead stamped kernel already demonstrates __float2uint_rn scaling at 1e6) and requiring Apple9-family atomic_ulong min only on M3+, or (b) shipping the legacy 32-bit float-min-CAS + separate parent write path that the hot kernel (wavefront_expand_active) already uses, keeping backtrace cycle detection as the safety net — option (b) requires no new atomics beyond MSL-standard 32-bit ops
- Map warp intrinsics in compact_mask_to_list to Metal simdgroup ops (simd_ballot/popcount/simd_broadcast; Apple simdgroups are 32-wide, matching the kernel's warp-32 assumptions); alternatively replace compaction with a two-pass prefix-sum, but the direct mapping is straightforward
- Pack the 30-40 scalar kernel parameters into a single constant struct per kernel for Metal (31-buffer bind limit); the stride-0 shared-CSR as_strided trick disappears naturally since kernels already take explicit stride parameters — pass one buffer + stride=0
- Replace CUDA-runtime dependencies with Metal equivalents at exactly these points: Device().mem_info → MTLDevice.recommendedMaxWorkingSetSize (K_pool sizing, cuda_dijkstra.py:2299-2355), Stream.null.synchronize → commandBuffer waitUntilCompleted, cp.cuda.Event → MTLCommandBuffer GPU timestamps; no multi-stream or memory-pool work is needed (engine uses null stream only)
- Consider MLX (mx.fast.metal_kernel accepts raw MSL source and MLX arrays are NumPy-like) as the CuPy analog — it can serve as both ArrayModule and RawKernel host; PyTorch is explicitly banned per docs/contributing.md, so avoid torch/MPS
- Use --cpu-only headless runs (python main.py headless board.ORP --cpu-only) as the oracle harness: compare per-net path cost, H/V/via legality, and iteration-level overuse convergence rather than exact node sequences (GPU wavefront is not cost-ordered and adds kernel-side jitter/RR bias); for via_kernels.py demand exact equivalence against the existing CPU twins (hard_block_via_edges_cpu etc.)
- Fix two porting-adjacent hazards while touching the seam: (1) GPU fast-path failure currently skips CPU fallback and marks the net failed (unified_pathfinder.py:4520-4529) — a new backend's early bugs will silently drop nets; add a fallback or at least a loud counter; (2) silent CPU downgrade at :2184-2197 means a broken Metal init would go unnoticed — surface [GPU-INIT] status in test output
- Budget for the per-iteration CPU round-trips that already exist (.get() → modify → asarray at :3232-3316, :3427-3478, and the 20 MB per-net parent backtrace transfer at cuda_dijkstra.py:5730): unified memory on Apple Silicon makes these nearly free, which is a genuine architectural advantage of the Metal port worth exploiting (zero-copy MTLBuffer sharing instead of explicit transfers)
