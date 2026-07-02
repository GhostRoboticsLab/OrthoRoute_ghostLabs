## What

Adds the first unit-test suite to the repo, and repairs the two smoke-test
entry points that were broken on every platform.

## Why

`CONTRIBUTING.md` lists tests as the **Critical Priority** ("The project has
zero unit tests. This is the biggest blocker to refactoring."). This starts
that suite — KiCad-free, CPU-only, **no new runtime dependencies** — and along
the way fixes two commands that silently did the wrong thing:

- `python main.py --test-via` built a **2-layer** board, which produces an
  empty routing graph (no via layers), so the self-test raised before it could
  test anything. Now uses a 4-layer board and the real engine call sequence.
- `cli` mode printed "No copper generated" and exited **0** when a board parsed
  to 0 nets, and skipped the mandatory `precompute_all_pad_escapes` step (which
  silently drops every net). Now it runs the full sequence and hard-fails
  (exit 1) on a 0-net parse.

It also fixes `.gitignore`, which ignored `test_*.py` and `tests/*` — actively
preventing *any* test suite from being tracked.

## Tests

- `test_lattice` — node count, H/V layer discipline, legal planar edges, legal
  via pairs.
- `test_via_accounting` — `EdgeAccountant` commit/clear symmetry, usage floored
  at 0, refresh-from-canonical, overuse counting at capacity.
- `test_engine_smoke` — the mandatory engine call sequence routes a trivial
  two-pad board end to end.
- `conftest.py` — shared fixtures + `make_two_pad_board` factory.

```
$ python -m pytest tests/
25 passed
```

No KiCad, no GPU, no network required.

## Checklist

- [x] Tests pass locally (25 passed on a clean clone)
- [x] No new runtime dependencies
- [x] PEP 8 / Google-style docstrings on new code
- [x] Fixes `.gitignore` so a fresh clone can collect the suite
