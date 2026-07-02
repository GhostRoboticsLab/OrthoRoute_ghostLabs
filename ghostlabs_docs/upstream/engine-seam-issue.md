# Issue: pluggable GPU backend seam + two GPU-fallback correctness fixes

_(Open this as a GitHub **issue** on bbenchoff/OrthoRoute before sending a PR —
per CONTRIBUTING, core/GPU changes should be discussed first. It bundles three
separable things; you can take them in any order.)_

## Context

I've been porting the router to Apple Silicon (Metal via MLX). Doing that, I
factored the GPU surface behind a tiny seam and found two fallback bugs. The
seam and the bug fixes are useful upstream **independent of any non-CUDA
backend**, so I'm raising them here first.

## 1. Two GPU-fallback bugs (worth fixing regardless of backend)

In `unified_pathfinder`'s GPU fast path:

- **(a)** an exception in the fast path (e.g. a CuPy import/kernel error) marked
  the net **failed** with no CPU attempt — so on any machine where the GPU path
  throws mid-run, nets get dropped instead of falling back to the working CPU
  router.
- **(b)** an owner-bitmap-constrained "no path" was treated as a hard failure
  instead of deferring to the cost-based CPU search.

Both now fall through to CPU. These are small and self-contained. Diffs ready.

## 2. Backend-selection seam

A ~90-line `backends.py`:

- `select_backend()` — `ORTHO_BACKEND=cuda|cpu` override, else probe CuPy, else
  CPU. Forcing an unavailable backend **fails loudly** instead of silently
  downgrading (which used to mask GPU-init bugs).
- `create_gpu_solver()` — the router's GPU surface is just two solver methods
  (the `KernelProvider` protocol), so this is a small, **CUDA-behavior-
  preserving** refactor. I verified a golden routing digest is byte-identical
  before/after the seam.

It also gives the project a clean place to register additional backends.

## 3. Optional: a backend-agnostic correctness oracle

`route_oracle.py` validates a routed board structurally (connectivity +
Manhattan legality + no-silent-drops accounting) independent of
`overuse == 0`, plus a headless DRC harness that writes routed copper back to a
`.kicad_pcb` and runs `kicad-cli pcb drc`. Happy to send as its own PR.

## What I'm **not** proposing to upstream

The Metal/MLX backend implementation itself. You can't compile or run Metal on
a CUDA box, and I don't want you maintaining code you can't execute — it lives
in my fork and plugs into the seam in (2). If the seam lands, anyone can add
Metal / ROCm / etc. the same way.

**Question:** would you take (1) + (2) as a PR, and (3) as a separate one?
