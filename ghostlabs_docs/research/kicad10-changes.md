# KiCad 9 → 10 Changes Affecting OrthoRoute

> Investigation report supporting `ghostlabs_docs/plan.md` (2026-07-02).

# KiCad 9 → 10 changes affecting an IPC-API Python plugin (OrthoRoute) — research as of 2026-07-02

## 1. KiCad 10 releases and IPC API compatibility

**Release timeline** (all from kicad.org blog):
- 10.0.0 — 2026-03-20 ("released on the spring equinox"): https://www.kicad.org/blog/2026/03/Version-10.0.0-Released/
- 10.0.1 — 2026-04-15: https://www.kicad.org/blog/2026/04/KiCad-10.0.1-Release/
- 10.0.2 — 2026-05-09: https://www.kicad.org/blog/2026/05/KiCad-10.0.2-Release/
- 10.0.3 — 2026-05-15: https://www.kicad.org/blog/2026/05/KiCad-10.0.3-Release/
- 10.0.4 — 2026-06-21 (current point release as of 2026-07-02; matches the locally installed 10.0.4): https://www.kicad.org/blog/2026/06/KiCad-10.0.4-Release/

**Compat policy** (https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/index.html, fetched 2026-07-02): the API follows protobuf best practices — "new versions of KiCad may introduce new messages and fields, but will not modify the meaning of existing messages and fields" and "deprecated messages and fields will be supported for at least one major version of KiCad after the deprecation is announced." So a client built against the 9.0 proto surface still talks to 10.x at the wire level. Transport is unchanged: protobuf over NNG, Unix domain socket on macOS/Linux (`/tmp/kicad/api.sock`, PID appended if multiple instances), `KICAD_API_SOCKET`/`KICAD_API_TOKEN` env vars still documented as-is (https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/for-addon-developers/index.html).

**Real-world 9→10 breakage found**: KiCad 10.0.0 shipped a regression where IPC plugins that worked in 9 did not appear in the toolbar (forum thread 2026-03-30/04-01, https://forum.kicad.info/t/kicad-ipc-api-attached-to-kicad-9-but-not-to-kicad-10/68030; tracked as https://gitlab.com/kicad/code/kicad-python/-/work_items/99, closed 2026-04-01). Fixed by Jon Evans, commit 761bdf5531e01d23b47aef3f549b7d716581bef7 ("Fix IPC plugins not showing up in toolbar anymore", authored 2026-03-24, cherry-picked into 10.0.1). 10.0.1 also: "Report plugin action errors to status bar", "Report plugin load errors to the user", "Allow relative paths in entry points", "Prevent Python plug ins from disappearing (#23861)", plus new API endpoints: dimensions read/update, barcodes, reference images, connected items (#22990), title block set, footprint jumper settings, net-tie-group deserialization fix (#23539). Per dev-docs: as of 10.0.1 plugin stdout/stderr is surfaced in the editor's warning system. **Practical minimum for an IPC plugin is KiCad 10.0.1, not 10.0.0.**

Other API facts: SWIG bindings still exist in 9 and 10 but are removed in KiCad 11; IPC API in 9/10 requires a running GUI (headless via kicad-cli arrives in 11); no plot/export via IPC until 11 (dev-docs for-addon-developers).

## 2. kicad-python (kipy) releases

From https://pypi.org/pypi/kicad-python/json (fetched 2026-07-02):
- 0.5.0 — 2025-10-13 (KiCad 9 era; pins pynng >=0.8,<0.9)
- 0.6.0 — 2026-03-15: **moved to pynng 0.9.0** (this is the Python 3.14 fix), adds `locked` on Track/ArcTrack, `Board.get_layer_name` (KiCad 9.0.8)
- 0.7.0 / 0.7.1 — 2026-04-17/18 (latest): `Board.get_items_by_id`, `get_groups` (requires KiCad 10); `get_barcodes`, `get_reference_images`, `set_title_block_info`, `get_connected_items`, `get_items_by_net`, `get_items_by_netclass` (require KiCad 10.0.1); 0.7.1 fixes `KiCad.run_action`.

No breaking API rewrite found in 0.5→0.7 — changes are additive; README (https://gitlab.com/kicad/code/kicad-python) says "requires a suitable version of KiCad (9.0 or higher)". Deps: `requires_python >=3.9`; protobuf >=5.29,<6; pynng >=0.9.0,<0.10; jsonschema >=4.23,<5.

**Python 3.14**: supported with kipy >=0.6.0. pynng 0.9.0 (uploaded 2026-02-04) ships cp314 and cp314t wheels including `macosx_10_15_universal2` (https://pypi.org/pypi/pynng/0.9.0/json). kicad-python 0.5.0 fails on 3.14 because KiCad's venv installer uses `pip --only-binary :all:` and pynng <0.9 has no 3.14 wheels — that was https://gitlab.com/kicad/code/kicad-python/-/issues/91 ("IPC Plugins can't load with Python 3.14"), closed 2026-03-02. Caveat: **on macOS KiCad 10.0.4 bundles Python 3.9** (`/Applications/KiCad.app/Contents/Frameworks/Python.framework/Versions/` contains only 3.9; dev-docs: "On macOS and Windows, KiCad will continue to ship with its own Python interpreter, which will be used by the IPC system by default"), so plugin code launched by KiCad runs under 3.9 unless the user repoints the interpreter.

## 3. GitLab issue #19465

https://gitlab.com/kicad/code/kicad/-/issues/19465 — title is "IPC Python Plugin Loading Broken in Windows" (not a PCM crash): plugin.json-format plugins failed to load / identifier-vs-folder-name complaints on Windows, 9.0.0-rc1. **Closed 2025-01-03, milestone 9.0, labels priority::high + status::fix-committed** (GitLab API). So it was fixed before KiCad 9.0.0 final — OrthoRoute's stated reason for avoiding PCM is stale and the citation doesn't actually describe a crash. PCM distribution of IPC plugins is now the supported path: since KiCad 9.0.1 PCM metadata has a `runtime` field ("may be set to either 'ipc' or 'swig' … if not set, SWIG assumed") per https://dev-docs.kicad.org/en/addons/index.html, and PCM prompts the user to enable the API server when installing an IPC plugin (https://gitlab.com/kicad/code/kicad/-/issues/20062, closed 2025-02-23, milestone 9.0.1, label 10.0).

## 4. Plugin discovery/installation in KiCad 10

- Directory: `${KICAD_DOCUMENTS_HOME}/<version>/plugins` (dev-docs) → **`~/Documents/KiCad/10.0/plugins/` on macOS**; confirmed locally: that directory exists (currently empty) on this machine. A KiCad-10 user in work item #99 likewise used `Documents\KiCad\10.0\plugins\`.
- plugin.json IPC schema: **still https://go.kicad.org/api/schemas/v1** per dev-docs (fetched 2026-07-02) — no v2 for the *API* plugin schema. (Separate thing: the *PCM* packaging schema gained v2 in KiCad 10, https://go.kicad.org/pcm/schemas/v2, which allows new package types; v1 still fine for `plugin` type.)
- Venvs: unchanged model — KiCad auto-creates a venv per plugin and installs declared deps; cache at `${KICAD_CACHE_HOME}/python-environments/<plugin_identifier>` → `~/Library/Caches/KiCad/10.0/...` on macOS (dev-docs; `~/Library/Caches/KiCad/10.0` confirmed present locally).
- "Enable KiCad API" preference: still exists in 10.0 (docs.kicad.org/10.0: "When enabled, you can use plugins that interact with KiCad's IPC API. If this option is not enabled, such plugins will not function"), and the **default is still `false`** — 10.0 branch source `common/settings/common_settings.cpp:470-471`: `PARAM<bool>( "api.enable_server", &m_Api.enable_server, false )`. Note: on this machine it is already enabled (`~/Library/Preferences/kicad/10.0/kicad_common.json` → `"api": {"enable_server": true, "interpreter_path": ".../KiCad.app/.../python3"}`).

## 5. Board file format 9→10 — this breaks OrthoRoute's regex parser

- Version stamp: 9.0 branch `SEXPR_BOARD_FILE_VERSION = 20241229`; 10.0 branch = **20260206** (both from `pcbnew/pcb_io/kicad_sexpr/pcb_io_kicad_sexpr.h` on the respective branches).
- **Netcodes are gone** (format 20251028 "Stop writing netcodes; they're an internal implementation detail"). Verified in 10.0-branch writer `pcb_io_kicad_sexpr.cpp`: tracks (line 2856), pads (1849), zones (2870) and shapes (1109) now emit `(net "NET_NAME")` with a quoted string, and `formatNetInformation()` — the header-level `(net N "NAME")` table that 9.0 wrote (9.0 branch line 671/678) — has been **removed entirely**. OrthoRoute's `orthoroute/infrastructure/kicad/file_parser.py` (lines 212-233) parses integer netcodes from both — it will find zero nets in any board saved by KiCad 10. (dev-docs file-format page at https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/ still documents the old `(net NET_NUMBER)` form — it lags the code.)
- Other new/changed tokens within 10.0 (from the version history in the 10.0 header): rounded rectangles (20250829), PCB points (20250901), table UUIDs (20250907), footprint unit metadata (20250909), `pcb_barcode` objects (20250914), via types split into blind/buried/through (20250926), pad-to-die delay scaling (20251027), backdrill/tertiary drill (20251101), PCB variants with per-footprint overrides (20260101, 20260206). Design blocks were extended to the PCB editor in 10 (release blog). Padstacks were already a KiCad 9 format feature, not new in 10.
- Coordinates: still mm in the file, nm (int64) in the IPC API — no change found (UNCONFIRMED only in the sense that no source states "unchanged"; no evidence of any change).
- Files saved by KiCad 9 remain readable by 10, but once re-saved they get the new format; docs note KiCad is backwards- but not forwards-compatible.

## 6. macOS-specific KiCad 10 IPC quirks

- Socket path `/tmp/kicad/api.sock` (PID suffix when multiple instances) — unchanged (dev-docs for-addon-developers).
- Bundled Python for IPC plugin venvs on macOS is **3.9** in 10.0.4 (verified locally) — plugin code must stay 3.9-compatible.
- GitLab issue search (labels=macos, search=API): **no open macOS-specific IPC API issues found**. Adjacent macOS 10.x fixes: PCM-with-proxy crash fixed in 10.0.4 (#24087), fresh-install startup crash #24180 (closed 2026-05-29), sleep/wake crash #23999 fixed in 10.0.2, another sleep crash #24620 still open (not IPC). The one confirmed 10.0.0 IPC bug (toolbar registration) was cross-platform, fixed in 10.0.1. UNCONFIRMED: absence of macOS-specific IPC bugs is based on issue-tracker search, not exhaustive.

## UNCONFIRMED items
- Root cause of the 10.0.0 toolbar regression (1-line fix; commit message gives no explanation).
- Whether PCM install of an IPC plugin ever *crashed* KiCad 9 as OrthoRoute's docs claim — issue 19465 describes load failure on Windows, not a crash, and was closed pre-9.0.0-final.
- Exact release date of kicad-python 0.7.1 (PyPI shows 2026-04-17/18).
- GitLab notes on issues 91/99 could not be read (API 401 for work-item notes), so resolution details are inferred from the dependency history (pynng 0.9.0 cp314 wheels + kipy 0.6.0 bump) and the linked fix commit.

## Key verified facts

- KiCad 10.0.0 released 2026-03-20; point releases 10.0.1 (2026-04-15), 10.0.2 (2026-05-09), 10.0.3 (2026-05-15), 10.0.4 (2026-06-21, current as of 2026-07-02) — https://www.kicad.org/blog/2026/03/Version-10.0.0-Released/ and sibling blog posts
- IPC API stability policy: additive-only changes, deprecated messages/fields supported for at least one major version — https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/index.html (fetched 2026-07-02); a 9.0-built client still talks to 10.x at the wire level
- KiCad 10.0.0 had a regression where IPC plugins did not appear in the toolbar; fixed in 10.0.1 via commit 761bdf5531e01d23b47aef3f549b7d716581bef7 (Jon Evans, 2026-03-24) — https://forum.kicad.info/t/kicad-ipc-api-attached-to-kicad-9-but-not-to-kicad-10/68030 and https://gitlab.com/kicad/code/kicad-python/-/work_items/99
- kicad-python latest is 0.7.1 (2026-04-17/18); 0.6.0 (2026-03-15) bumped to pynng>=0.9.0,<0.10; 0.7.0 adds KiCad 10 / 10.0.1 endpoints (get_groups, get_barcodes, get_connected_items, get_items_by_net) — https://pypi.org/pypi/kicad-python/json
- Python 3.14 works with kicad-python >=0.6.0: pynng 0.9.0 ships cp314 wheels incl. macosx_10_15_universal2 (uploaded 2026-02-04, https://pypi.org/pypi/pynng/0.9.0/json); kicad-python 0.5.0 fails on 3.14 (issue https://gitlab.com/kicad/code/kicad-python/-/issues/91, closed 2026-03-02) because KiCad installs venv deps with pip --only-binary :all:
- GitLab issue 19465 ('IPC Python Plugin Loading Broken in Windows') was closed 2025-01-03, milestone 9.0, status::fix-committed — fixed before KiCad 9.0.0 final; it describes a Windows load failure, not a PCM crash — https://gitlab.com/kicad/code/kicad/-/issues/19465 (via GitLab API)
- PCM distribution of IPC plugins is supported: PCM metadata 'runtime' field ('ipc'/'swig') exists since KiCad 9.0.1, and PCM prompts users to enable the API server when installing an IPC plugin (issue 20062, closed 2025-02-23, milestone 9.0.1) — https://dev-docs.kicad.org/en/addons/index.html and https://gitlab.com/kicad/code/kicad/-/issues/20062
- KiCad 10 plugin directory on macOS is ~/Documents/KiCad/10.0/plugins/ (pattern ${KICAD_DOCUMENTS_HOME}/<version>/plugins per https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/for-addon-developers/index.html); confirmed present on the local machine; venv cache at ~/Library/Caches/KiCad/10.0/python-environments/<id>
- plugin.json for IPC plugins still uses schema https://go.kicad.org/api/schemas/v1 in KiCad 10 (dev-docs, fetched 2026-07-02); the separate PCM packaging schema gained v2 for KiCad 10 (https://go.kicad.org/pcm/schemas/v2)
- 'Enable KiCad API' preference default is still false in KiCad 10: 10.0 branch common/settings/common_settings.cpp lines 470-471: PARAM<bool>("api.enable_server", &m_Api.enable_server, false); the local machine already has it enabled in ~/Library/Preferences/kicad/10.0/kicad_common.json
- Board file format version: KiCad 9.0 branch SEXPR_BOARD_FILE_VERSION=20241229, KiCad 10.0 branch=20260206 (pcbnew/pcb_io/kicad_sexpr/pcb_io_kicad_sexpr.h on branches 9.0 and 10.0)
- KiCad 10 no longer writes netcodes (format 20251028 'Stop writing netcodes'): 10.0-branch pcb_io_kicad_sexpr.cpp writes (net "NET_NAME") for tracks (line 2856), pads (1849), zones (2870), shapes (1109), and the header net table writer formatNetInformation() ((net %d %s), 9.0-branch line 671/678) is removed
- OrthoRoute's parser depends on the removed netcode format: orthoroute/infrastructure/kicad/file_parser.py lines 212-233 parse integer netcodes; requirements.txt pins kicad-python>=0.5.0; build.py lines 136/142/168/172/266-267 hardcode Documents/KiCad/9.0/ paths
- Local install verified: /Applications/KiCad.app is 10.0.4 (CFBundleVersion) and bundles only Python 3.9 (Contents/Frameworks/Python.framework/Versions/); dev-docs: on macOS/Windows KiCad's bundled interpreter is used for IPC plugins by default
- macOS IPC transport unchanged: Unix socket /tmp/kicad/api.sock (PID appended for extra instances), KICAD_API_SOCKET/KICAD_API_TOKEN env vars unchanged (dev-docs for-addon-developers); no open macOS-specific IPC API issues found in the KiCad tracker (labels=macos search, 2026-07-02)
- SWIG bindings still present in KiCad 9 and 10, removed in KiCad 11; IPC API requires a running GUI in 9/10 (headless and plot/export support arrive in KiCad 11) — https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/for-addon-developers/index.html

## Recommendations

- Target KiCad >=10.0.1 (ideally the installed 10.0.4), not 10.0.0: 10.0.0 had the IPC-plugin toolbar registration regression; gate any KiCad-10-only kipy calls (get_items_by_net, get_connected_items, get_barcodes) behind a runtime KiCad version check
- Bump requirements to kicad-python>=0.7.1,<0.8 (or at minimum >=0.6.0): this pulls pynng>=0.9.0 with cp314 wheels, fixing the Python 3.14 venv-install failure (kicad-python issue #91), and adds the KiCad 10 endpoints; note the protobuf>=5.29,<6 dependency
- Do NOT assume Python 3.14 at plugin runtime on macOS: KiCad 10.0.4 bundles Python 3.9 and uses it for IPC plugin venvs by default — keep plugin code 3.9-compatible or document repointing Preferences > Plugins > Path to Python interpreter
- Update all install paths from ~/Documents/KiCad/9.0/plugins/ to ~/Documents/KiCad/10.0/plugins/ (build.py lines 136, 142, 168, 172, 266-267); better, derive the version segment dynamically instead of hardcoding; plugin.json schema stays v1 (go.kicad.org/api/schemas/v1) — no manifest change needed
- Fix or retire the regex .kicad_pcb parser (orthoroute/infrastructure/kicad/file_parser.py:212-233): KiCad 10 boards (format 20260206) have no netcode table and reference nets by quoted name — (net "NAME") — on tracks/pads/zones; support both forms, or preferably read nets via the IPC API (kipy 0.7.x Board.get_items_by_net / get_connected_items) so the file format no longer matters
- Reconsider PCM distribution: issue 19465 was fixed before KiCad 9.0.0 final and PCM now formally supports IPC plugins via the runtime:"ipc" metadata field (since 9.0.1) and prompts users to enable the API server (issue 20062); target the v2 PCM schema if publishing for KiCad 10+
- Keep KICAD_API_SOCKET/KICAD_API_TOKEN handling as-is — unchanged in KiCad 10; on macOS the fallback socket is /tmp/kicad/api.sock with a PID suffix when multiple KiCad instances run
- Document that users must enable Preferences > Plugins > Enable KiCad API (still default-off in 10.0 per common_settings.cpp); on this development machine it is already enabled, so local testing against the running 10.0.4 instance can start immediately
- For debugging on 10.0.1+, use the new surfacing of plugin stdout/stderr and load errors in the editor status-bar warning system instead of only ~/Documents/KiCad/10.0/logs/
