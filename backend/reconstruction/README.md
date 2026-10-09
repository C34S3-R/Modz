# Reconstruction engine (`backend/reconstruction/`)

CPU-only, checkpointed photogrammetry for this server: video or photographs →
frames → SIFT → verified matches → incremental SfM → sparse point cloud →
plane-sweep dense depth maps.
It is a **self-contained package with a CLI**; wiring into the web pipeline
(`backend/pipeline/`, `job_manager`) is a later phase and is not done yet.

Design priority order: correctness, low memory, recoverability — **not** speed.

## Stages

| # | stage | module | writes |
|---|-------|--------|--------|
| 1 | `input_validation` | `frames.py` | ffprobe metadata in `logs/ffprobe.log` |
| 2 | `frame_extraction` | `frames.py` | `frames/*.png`, `frames/extraction_manifest.json` |
| 3 | `frame_selection` | `frames.py` | quality-ranked selection |
| 4 | `feature_extraction` | `features.py` | `features/<frame>.npz` |
| 5 | `feature_matching` | `matching.py` | `matches/<a>__<b>.npz`, `matches/matches_meta.json` |
| 6 | `sfm` | `sfm.py` | `sparse/*.ply`, `sparse/sparse_preview.png`, `sfm/sfm.json`, `reconstruction_report.json` |
| 7 | `dense` | `dense.py` | `dense/<frame>.npz` (depth/ncc/agreement), `dense/dense_point_cloud.ply`, `dense/dense_meta.json` |
| 8 | `mesh` | `mesh.py` | `mesh/raw/mesh.ply` (vertices+faces+colours), `mesh/mesh_meta.json` |
| 9 | `lowpoly` | `lowpoly.py` | `lowpoly/lowpoly.ply` (decimated + colours), `lowpoly/lowpoly_meta.json` |
| 10 | `texture` | `texture.py` | `texture/texture.png`, `texture/textured.obj`, `texture/textured.mtl`, `texture/texture_meta.json` |
| 11 | `blender` | `blender.py` | `blender/prepared_textured.obj` + sidecars, `blender/blender_meta.json` |
| 12 | `bussid` | `bussid.py` | `bussid/*` (model+sidecars+manifest.json), `exports/<id>.bussidmod`, `exports/result.json`, `bussid/bussid_meta.json` |

Each stage records start/end, command, exit code, stdout/stderr, memory and a
human-readable failure reason in `logs/stage_<name>.json`; the engine
checkpoints at `reconstruction/project_state.json`. The root
`project_state.json` stays owned by the web backend.

## Per-project layout

```text
projects/<id>/
  input/                         originals (never deleted, never modified)
  frames/                        extracted frames + extraction_manifest.json
  features/  matches/            SIFT + per-pair verified correspondences
  sfm/sfm.json                   cameras, poses, focal, observations
  sparse/                        sparse_point_cloud.ply, cameras.ply, preview.png
  dense/                         per-view depth maps (.npz), fused coloured
                                 point cloud, dense_meta.json
  mesh/  lowpoly/  texture/      raw mesh, decimated mesh, atlas + OBJ/MTL
  blender/                       prepared model (Blender output or logged
                                 model-copy fallback) + blender_meta.json
  bussid/                        final model + sidecars + manifest.json
  exports/                       <id>.bussidmod package + result.json
  logs/                          per-stage records + pipeline.log
  reconstruction/project_state.json   engine checkpoint (restartable)
  reconstruction_report.json     final summary: registration, focal, errors,
                                 dropped cameras, coverage messages, exports
```

## CPU-only policy

`enforce_cpu_only()` clears `CUDA_VISIBLE_DEVICES`, disables OpenCV's OpenCL
runtime and clamps threads (`RECON_THREADS`, default 2, **max 4**). Every
stage logs `CPU-ONLY MODE ENABLED` and the actual algorithm list. Requesting a
CUDA algorithm fails loudly with the CPU alternative named — there is no silent
GPU fallback on this host (i5 M560, 6 GB RAM, no NVIDIA).

Heavy stages run one at a time (the default concurrency gate is 1); frames and
videos are streamed to disk, never held in RAM.

## CLI

```bash
./vehicle-reconstruct input.mp4 --cpu-only \
    [--threads 2] [--max-image-size 1200] [--features 3000] \
    [--frame-interval 0.5] [--quality low] [--target-triangles 30000]
./vehicle-reconstruct resume <project_id>      # continue from checkpoints
./vehicle-reconstruct status <project_id>      # stage states + report summary
```

Environment overrides: `RECON_THREADS`, `RECON_QUALITY`, `RECON_INITIAL_FOV`
(horizontal FOV guess, default 60°), `RECON_FOCAL_MULTISTART` (default on),
`RECON_FOCAL_OVERRIDE` (pin the focal length in px; skips probing),
`RECON_MATCH_WINDOW`, `RECON_MIN_REGISTERED_RATIO`, `RECON_TEXTURE_SIZE`,
`RECON_DEBUG_PNP` (dump per-camera PnP diagnostics),
and for the dense sweep: `RECON_DENSE_MAX_SIZE` (working resolution long
side, default 480), `RECON_DENSE_STEP_PX` (max disparity step between depth
hypotheses, default 1.5), `RECON_DENSE_MAX_PLANES` (default 160),
`RECON_DENSE_MIN_PARALLAX` (default 0.02), `RECON_DENSE_BLOCK` (default 16),
`RECON_DENSE_MIN_NCC` (default 0.30), and for meshing:
`RECON_MESH_CROP` (subject-ball radius multiplier, default 1.5),
`RECON_MESH_RESOLUTION` (grid cells on the longest axis, default 384),
`RECON_MESH_TRUNC_VOXELS` / `RECON_MESH_TRUNC_REL` (fusion truncation band,
defaults 4.0 voxels / 1% of distance), `RECON_MESH_MIN_WEIGHT` (views that
must agree per voxel, default 4), `RECON_MESH_SUPPORT_REL` (vertex prune
threshold as a share of the project's validated-surface agreement level,
default 0.5), `RECON_MESH_COLOR_TOL` (default 3.0 voxels), and
`RECON_TARGET_TRIANGLES` (low-poly budget; defaults to the quality preset:
20 000 / 30 000 / 60 000). External tooling: `BLENDER_PATH` (Blender
executable for the cleanup stage; empty means auto-resolve from `PATH`) and
`REQUIRE_EXTERNAL_TOOLS` (`true` makes the Blender stage fail instead of
degrading to its model-copy fallback).

## Verification

There is no repository test suite; this module verifies itself against a
synthetic scene with known ground truth:

```bash
PYTHONPATH=backend .venv/bin/python backend/reconstruction/selftest.py render projects/synth_selftest
./vehicle-reconstruct resume synth_selftest
PYTHONPATH=backend .venv/bin/python backend/reconstruction/selftest.py evaluate projects/synth_selftest
```

`render` draws a textured box from 48 known viewpoints (65° FOV, focal
502.3 px) and stores `input/gt.json`. `evaluate` aligns the reconstruction to
those poses (Umeyama, scale-free) and fails unless **all four sparse
checks** hold:

- registered images ≥ 80 % of frames
- mean reprojection < 3 px
- mean camera position error < 12 % of scene size
- mean relative rotation error < 4°

When `mesh/raw/mesh.ply` exists, **four mesh checks** join (the mesh is
back-aligned to GT and scored against the exact box SDF), and once
`lowpoly/lowpoly.ply` exists **five more** (the same four on the
decimated mesh plus the triangle budget):

- bounding-box dimensions within 20 % of GT (p0.5–p99.5 extents)
- surface-distance p90 ≤ 0.5
- coverage: p95 of GT-surface samples to nearest vertex ≤ 0.4 — samples
  only cover GT faces at least one camera could see (the box is convex so
  visibility is exact); unobserved faces are reported as
  `unseen_gt_faces`, never silently passed
- junk share (|sdf| > 0.15) ≤ 50 %
- lowpoly faces ≤ 1.1 × target triangles

Once `texture/texture_meta.json` exists, **four texture checks** join:

- `texture.png`, `textured.obj`, `textured.mtl` present (from the meta's
  output paths)
- baked + dilated atlas share ≥ 15 %
- `mapping_check` p50 ≤ 12 with ≥ 30 samples (frame → atlas round trip)
- `cross_view_check` p50 ≤ 25 with ≥ 30 samples (other views,
  depth-occlusion gated)

The four mesh thresholds are a **calibration point, not aspirations**:
each sits ≥ 1.9× from the nearest known-bad variant we have actually
produced (ungated synth: dims +158…+695 %, surface p90 3.20, junk 0.97;
support-only fragment synth: coverage p95 1.27) while leaving ≥ 1.3×
headroom over the three-gate measurements. Regressions to any earlier
failure mode still fail loudly. The texture thresholds are calibrated
the same way, measured with the stage's own check code against known-bad
atlases: a spatially shifted atlas (simulating a broken bbox → pack →
UV chain) scores mapping p50 58.4 / cross-view p50 60.3, a black atlas
~105, and an atlas that was never baked fills ~0.000 — each threshold
sits ≥ 1.9× inside those bad values. The checks gate on p50, not the
mean: on a high-frequency texture the mean carries a long alignment
tail (good mean 8.1 / 34.3) that would overlap the bad variants.

The real-video check is `projects/realvid` (built from
`projects/ss/input/vid.mp4`, 478 x 850, 50.2 s): it must clear the 35 %
registration gate with all six exports present (`sparse/sparse_point_cloud.ply`,
`sparse/cameras.ply`, `sparse/sparse_preview.png`, `sfm/sfm.json`,
`reconstruction_report.json`, `reconstruction/project_state.json`). It
currently registers **31/80 (38.8 %)** at 0.97 px mean reprojection with 0
dropped cameras, peak stage memory 0.19 GB, and reports honestly that the
left, rear and roof of the vehicle were never observed in this clip.

The dense stage validates itself against the sparse reconstruction on every
run (`sparse_check` in `dense/dense_meta.json`): dense depth sampled at the
sparse points' pixels versus those points' own depth. On `realvid`:
**31/31 views swept**, median relative depth error **1.9 %** (per-view range
1.2–7.6 %), coverage 83.6 % of pixels at NCC ≥ 0.3 (median NCC 0.82–0.84),
327 s wall, 0.45 GB peak RSS, fused cloud 343 k points.

Mesh + low-poly on `synth_selftest` (quality low, 37 views) — the full
gate stack (**weight ≥ 4 → relative support → sparse-anchored
components**) takes the raw 6.35 M vertex / 12.16 M face noise ball to
**46,467 vertices / 84,736 faces**, then the QEM stage lands on
**19,999 faces** (target 20 000, ~2 s). Final scores: bbox
4.05 × 2.04 × 1.79 vs GT 4.2 × 1.8 × 1.6 (max err 13.3 %), surface
p50/p90 0.109/0.260, coverage p50/p95 0.120/0.313, junk 31.9 % —
**13/13 checks green**. Per-stage memory peaks stay ≤ 0.6 GB.

On `realvid` the same stack takes the 352,759 gate-4 vertices through
relative support (baseline 0.54 → threshold 0.27, keeps 61 %) and
anchoring (4,346 → 425 components) to **130,183 vertices / 225,483
faces**; QEM lands on **exactly 20,000 faces** in ~2 s. Projected into
`frame_000011` with that view's own intrinsics the vertices land on the
hood edge, bumper, grille and number plate — registration is exact, and
multi-view renders show coherent vehicle/ground slabs with real colours.
Residual floating sheets remain over the rubble (honest for night,
specular footage at 31/80 views with rear/roof unobserved); that is the
known next geometry-side target, not a registration defect.

Texture on `synth_selftest`: **7,277 charts** packed into the 1024²
atlas at scale 1.0 (utilisation 29 %), 17,687 faces painted
depth-checked / 724 projection-only / 1,588 never seen (7.9 %, fallback
cell), 6,520 faces painted from a margin-joined chart view, 17,494
baked, 33.3 % of the atlas filled (bake + dilation); `mapping_check`
p50 **2.23**, `cross_view_check` p50 **14.3** (220 samples, 430
occluded corners skipped), 54 s, 0.11 GB peak RSS. With the packaging
checks below the full evaluation runs **20/20 checks green** (4 sparse +
4 mesh + 5 low-poly + 4 texture + 3 package).

Texture on `realvid`: 7,791 charts, packing auto-shrinks the scale to
0.61 for the larger vehicle panels (utilisation 60 %), 12,862 faces
depth-checked / 829 projection-only / 6,309 never seen (31.5 % — the
rear/left/roof the clip never observes, honestly parked in the fallback
cell), 12,740 baked, 61.9 % filled; `mapping_check` p50 **0.78**
(sub-texel), `cross_view_check` p50 **15.6** (283 samples, 365 occluded
skipped), 45 s, 0.11 GB. Rendered back through a real pose the plate
reads `KAP·852F` correctly, the textured draw covers 100 % of the
mesh's geometry with zero black texels, and the black patches in the
render are missing geometry (glass, unobserved bodywork) — present
identically in the vertex-colour render, i.e. a mesh-completeness
honesty issue, not a texture defect.

Blender + BUSSID on `synth_selftest`: no Blender is installed on this
host, so the cleanup stage takes its documented fallback — a WARNING in
`logs/pipeline.log` plus `fallback: true` in `blender_meta.json`, model
copied through unchanged — and the BUSSID stage assembles the
OBJ → MTL → texture chain into `bussid/` with a `manifest.json`, then
packages it as `exports/synth_selftest.bussidmod` (1.3 MB, 3 assets).
On `realvid` the package is `realvid.bussidmod` (1.6 MB), `testzip()`
clean, root manifest `format: bussidmod` with every listed asset
resolving inside the archive. The evaluation gains 3 structural checks —
`bussid_files_ok`, `bussid_package_ok`, `bussid_textures_ok` — bringing
the total to **20/20 green**.

All output-producing stages (mesh → bussid) are **deterministic and
restartable**: clearing the stages and re-running reproduces
byte-identical outputs (metas identical modulo timing/memory fields) —
including a byte-identical `.bussidmod`, since zip entries carry a fixed
timestamp instead of file mtimes — and an immediate follow-up resume
reuses the checkpoints without re-running (0.5 s for the whole
pipeline).

Run `python3 -m py_compile $(find backend -name '*.py' -type f | sort)` after
edits.

## Design notes (why the code looks like this)

These are measurements, not opinions:

- **Bundle adjustment tolerances.** SciPy's `ftol`/`xtol` are *relative*
  stopping tests. At `1e-4` the solver stopped after a couple of trust-region
  steps while `optimality` was still ~5.7e3 — a silent no-op BA (measured:
  2.66 → 2.37 px instead of converging to the 0.5 px noise floor in ~35
  evaluations). `ftol=xtol=1e-8` is deliberate; going below ~1e-12 makes `trf`
  divide by a collapsed trust-region radius. A no-op BA is invisible in a cost
  field, so every BA now reports `mean_reprojection_before/after_px`,
  `solver_status` and `solver_message`.
- **Verification is K-free.** Matches are verified with the fundamental matrix
  in *pixel* coordinates. Fitting an essential matrix through an assumed
  focal length is not harmless: the principal-point term breaks the scale
  invariance of the epipolar constraint, so RANSAC keeps a K-dependent subset
  of correspondences and every track inherits the bias. Measured on the
  synthetic scene: the same sequence landed **14 % or 49 %** off ground truth
  depending only on which FOV the matcher assumed. Poses are re-derived in
  `sfm.py` from the focal the reconstruction settles on.
- **Focal multi-start.** Reprojection error is nearly flat in focal length —
  the structure absorbs the error — so joint refinement from a wrong start
  always reports the starting value as best. Instead, `run_sparse_multistart`
  rebuilds a contiguous 12-frame sub-model at each candidate focal
  (0.8/0.9/1.0/1.15 × the FOV prior), extends the grid while the winner sits
  on an edge, and runs the real SfM at the winner. Honest limit: this rejects
  a *badly* wrong prior and lands within roughly 10 % of the true focal; it is
  not a calibration. Use `RECON_FOCAL_OVERRIDE` when the real value is known.
- **Determinism.** OpenCV RANSAC draws from a global stream, so matching and
  SfM reseed from their stage fingerprints. Two runs of identical input
  produce identical output, which is what makes A/B comparisons of a change
  meaningful.
- **Iterative observation filtering.** BA's robust loss down-weights an
  observation it cannot explain but never removes it, so a wrong match keeps
  tugging on its camera until the trajectory silently deforms. After the final
  BA the engine drops observations still above
  `track_max_reprojection_error`, forgets points left with a single view, and
  refits — up to `ba_filter_rounds` times.
- **Honest reporting.** Unregistered images, disconnected camera components
  and dropped cameras are reported by name with reasons; unseen geometry is
  never invented. Coverage messages say so plainly, e.g. *"Rear geometry
  insufficiently observed."* Every unregistered image gets its own entry in
  `unregistered_reasons` - the failure message points readers there, so an
  empty list would send them nowhere - and the failure text states the
  component split it actually observed instead of guessing at the cause.
- **The seed decides which part of the video gets reconstructed.** Registration
  can only grow inside the component the seed starts in, so candidates are
  tried largest-component-first. Measured on `projects/realvid` (`vid.mp4`,
  80 frames): the match graph splits **31 / 25 / 24** images, and seeding the
  25-frame component registered 25/80 (31 %, below the 35 % gate) while the
  31-frame component registers 31/80 (38.8 %) and passes. The quality gates
  are unchanged, so a large component that cannot produce a usable seed still
  falls through to a smaller one.
- **What actually split that graph.** Frames 54-58 (t ≈ 33-35 s) fall to
  387-950 SIFT features against ~2000 for healthy frames. The bridge pair
  (55, 56) yields 32 Lowe-0.75 matches and **11** F-RANSAC inliers against the
  25-inlier gate, while (54, 55) has 26 and (56, 57) has 31. One failed pair
  at the blurriest moment splits the sequence in two - which is why the
  failure message reports component sizes instead of "not enough orbit
  overlap".
- **Checkpoint fingerprints are computed twice, identically.** A stage record
  stores its fingerprint through the same helper the next run compares against
  (`_stage_fingerprint`). The frames digest (`frames/selected`, names + sizes)
  used to be attached only at comparison time, and is empty on a first run when
  no frames exist yet: every stage therefore recorded an unsuffixed
  fingerprint and was re-run - with everything downstream - on the very next
  resume, measured as a full extra matching pass (1047 s first run, 615 s
  re-verifying unchanged work). `frame_extraction` is excluded from that digest
  because it is its *downstream* output. Verified: consecutive resumes now
  skip every stage in 0.0 s.
- **Notes keep both ends.** `ctx.note` kept only the last 50 entries, which
  discarded the seed pair, the focal-probe scores and the calibration
  decisions - exactly what a failed run needs diagnosed. It now keeps the
  first 60 and the last 180 (240 bound, ~30 KB).
- **Bundle adjustment reports `parameter_shift`.** Equal
  `mean_reprojection_before/after_px` on their own are ambiguous: "already at
  the robust-loss optimum" and "the solver never moved" look identical from
  the residual alone. The displacement of the parameter vector tells them
  apart, so a silent no-op BA is visible rather than inferred.

### Dense stage (why the sweep looks like this)

- **A global log-spaced depth range destroys NCC before it starts.** The
  first sweep prototype used one range for the whole image (3–97th
  percentile of the in-frustum sparse points), log-spaced. That range has to
  cover the vehicle *and* the background — 5 to 109 units on `realvid` — so
  adjacent planes were ~4 % apart. At baseline 3.7 that is a ~6 px disparity
  error between hypotheses, wider than the NCC window: NCC at the *true*
  depth fell to 0.11 and depth error was 35–47 % median with *negative*
  correlation against ground truth. The geometry was verified exact first
  (a point projected directly versus through `H(d)` agreed to 1.1e-6 px), so
  the failure was purely sampling.
- **Per-pixel inverse-depth hypotheses sized by the disparity budget fix
  it.** Each 16 px block gets its own `[z_lo, z_hi]` from the sparse points
  landing in it (empty blocks inherit a neighbour's bounds), planes are
  uniform in *inverse* depth (constant disparity step), and the plane count
  is `ceil(span / dense_step_px)` where `span = f·B·Δ(1/z)` — the actual
  disparity the sweep must cover. Measured on three reference views: median
  depth error **2.2 / 1.8 / 2.6 %** and correlation **+0.88 / +0.95 / +0.91**
  where the global sweep gave 35 % and +0.14. Production runs land at 1.9 %
  across 31 views.
- **Neighbour selection is two scale-free gates, not absolute distances.**
  Reconstruction scale is arbitrary, so a neighbour is usable when its
  baseline is *at least* `dense_min_parallax × median depth` (below that it
  contributes no depth signal) and *at most* the plane budget can resolve
  (above that the match breaks — measured: baseline 8.9 units collapsed NCC
  at true depth to 0.14 even before the range fix). Baselines between the
  gates add coverage, not error: widening the window from 2 to 4 neighbours
  raised coverage 74 → 83 % at unchanged 2.3 % accuracy.
- **The plane budget uses a p95 block span, not the max.** One `realvid`
  view (frame 27) contains a sparse point at z=0.66 next to background at
  z=75; its block span of 1.5 (vs ~0.15 typical) made the span gate reject
  *every* neighbour and the view was skipped. p95 lets those few extreme
  blocks be swept coarser — 31/31 views instead of 30/31, same accuracy.
- **Optimisation: float32 + explicit projection + cached reference terms.**
  The first implementation ran float64 and BLAS matmuls per plane and took
  931 s. Profiling showed 14 ms in `R @ cam` and 7 ms in `K @ xs` for a
  3×129600 product, plus 6 box filters per plane of which the reference-side
  two never change. Explicit elementwise projection in float32 (exact to
  ≪0.001 px at these magnitudes) and precomputing `blur(a)`, `blur(a²)` per
  view: **327 s**, identical quality (1.82 → 1.89 %, within run noise).
  Unmasked box filters are used where the full NCC support window is valid
  and windows touching invalid pixels are rejected outright instead of mixed
  with zero padding.
- **`agreement` counts only confident neighbours.** The first version
  divided by all sources, so a source whose best NCC never cleared the gate
  (its per-plane argmin stuck at plane 0) dragged every pixel toward
  disagreement: mean 0.30. Counting only sources with a confident match
  raised it to 0.44 and made it mean what it says — for fusion, a pixel
  where every confident neighbour picked the same depth neighbourhood.
- **The exported cloud sizes voxels from the bulk, not the extremes.**
  A few confident-looking far pixels (sky inside a propagated depth range)
  stretched the bounding box until voxel = 1.18 units — the 4-unit vehicle
  became ~4 voxels across and the cloud collapsed to 47 k points. Sizing the
  voxel from p1–p99 bounds (dropping 94 k outliers) gives voxel 0.31 and
  343 k points. The mesh stage will re-sample the depth maps anyway; this
  cloud is the inspection artefact.

### Mesh stage (volume, consensus, pruning)

- **The fusion volume follows the subject, not the scene.** Full-scene
  extent on `realvid` is 106 units diagonal, so the 4-unit vehicle would be
  10 voxels across even at 256³. The volume is cropped to a ball around the
  *look-at point* — the least-squares intersection of the camera optical
  axes (the point an orbit scan is of; handheld rays miss it by a median
  2.5 units but converge fine) — with radius `mesh_crop` (1.5) times the
  median camera distance. That yields extent 36 units: 42 vehicle voxels at
  384³, keeping 66 % of raw dense points. The crop can only ever drop
  geometry the camera-hull logic calls background; if the solve is
  degenerate or would keep almost nothing, the stage falls back to the full
  extent and says so (`volume.crop.applied` in the meta).
- **TSDF voxels start "empty" (+1), and the truncation band grows with
  depth:** `max(4 voxels, 1 % of camera distance)`. Plane-sweep noise is
  ~2 % of depth, so a fixed voxel-sized band would reject far samples;
  unfused space sitting at the iso-level (0) would grow phantom surfaces,
  so the +1 init matters as much as the band.
- **Consensus gates the surface — photometric confidence cannot.** Fusion
  admits every confident pixel, but on real footage (night, specular paint,
  residual pose error) samples are wrong *and wrong differently per view*.
  Measured on `realvid`: 5.1 M samples → 64 % of covered voxels hit by
  exactly one view, and the ungated zero level set was 12 156 units² of
  surface — ~10x the scene's real area (ground disk + vehicle ≈ 900) and
  visually a solid ball of noise. Gating by agreeing-view count:
  weight ≥ 2 → 3 776, ≥ 3 → 1 929, **≥ 4 → 1 218**, ≥ 6 → 598 units².
  The stored per-pixel `ncc`/`agreement` do *not* predict cross-view
  consistency here (P(consistent) flat at 0.44–0.51 across every decile),
  which is why the gate is multi-view consensus instead.
- **Vertices carry a support fraction and get pruned by it, relative to
  validated surface.** Support = share of observing views whose depth
  confirms the vertex inside the fusion band. The threshold cannot be an
  absolute number: the *sparse* points (reprojection-checked true
  surface) themselves reach median support 0.54 on `realvid` but only
  0.35 on `synth_selftest` (10% focal error biases each view's depths
  differently), while mesh vertices sit below that (0.27 / 0.20). So the
  stage measures the sparse baseline first and keeps vertices at
  `mesh_support_rel` (0.5) times it, floored at 0.1 - honest on both
  datasets. Pruning drops the faces that touch a pruned vertex and
  compacts orphans; if fewer than 1 000 faces survive the stage fails
  honestly instead of shipping a fragment.
- **A third gate drops unanchored pieces.** Support judges vertices one
  at a time, so self-consistent junk (sheets, blobs) survives it: on
  `synth_selftest` 64 % of vertices lived in **10 373 tiny components**
  floating 0.3–2 units from the box. Connected components over face
  adjacency are kept only when some vertex lies within
  `mesh_anchor_rel` × the sparse cloud's median spacing of validated
  surface — anchored at the scale this project's surface is actually
  validated at. That rule took the bounding box from +56/+163/+273 %
  error to −4/+13/+12 % while retaining 73 % of true surface; on
  `realvid` (junk denser and attached to the main blobs) it removes the
  4 346-fragment tail and keeps the anchored bodies whole.
- Fusion is slab-chunked elementwise float32 (no BLAS, fixed view order)
  so reruns are bit-identical; the 33.6 M-voxel `realvid` grid fuses in
  ~300 s and the whole stage peaked at 0.72 GB RSS.

### Low-poly stage (why decimation is a stage, not a flag)

- **The deliverable is a light BUSSID mod, not a film asset.** The raw
  marching-cubes surface exists as an honest intermediate that this stage
  collapses to the preset's triangle budget (`low` 20 k, `medium` 30 k,
  `high` 60 k — `RECON_TARGET_TRIANGLES` overrides). Fewer triangles means
  a smaller `.bussidmod`, a faster import, and less for the device that
  runs BUSSID to chew on — so aggressive decimation is a feature.
- **QEM via `fast_simplification`** (C++, single pass, no GPU): quadric
  error metrics keep planar vehicle panels crisp at very low counts,
  which matters because silhouette beats micro-relief on a game model.
  The target is part of the stage fingerprint, so changing the budget
  re-runs only this stage.
- **Degenerate faces are dropped first.** A noisy TSDF can produce
  zero-area triangles; their quadric error is zero, which would let them
  dominate collapse decisions.
- **Colours transfer by nearest source vertex** (KD-tree): the decimator
  moves vertices, frame colours live on the source mesh, and the mean
  transfer distance lands far below a pixel of the baked texture, so the
  swap is invisible. The distance p95 is recorded in the meta.
- **Honest passthrough:** a mesh already at budget is copied through with
  `decimated: false` instead of being "simplified" to the same size.

### Texture stage (hand-rolled atlas, because xatlas is not available)

- **Dominant view per triangle, two quality tiers.** A frame may paint a
  triangle only if it is front-facing, fully in frame and covers more
  than a square pixel; where the depth map exists the triangle centroid
  must also agree with it inside the same fusion band the mesh used
  (`RECON_MESH_TRUNC_VOXELS`/`_REL`), so occluded geometry never leaks
  into the texture. The largest passing projection wins. Faces that
  clear only the projection test are counted separately
  (`projection_only`) and faces no camera ever sees are counted as
  `never_seen` — the meta reports why each texel exists instead of
  hiding the fallback.
- **Charts by BFS** over shared edges: the seed's view must be a
  *candidate* of the neighbour (passes projection and depth gates and
  covers ≥ 60 % of the neighbour's own best projection) and its normal
  must stay within 40° of the seed. The candidate margin is what keeps
  a flat panel in one chart - triangles of a panel rank the near-tie
  views in slightly different orders, and exact per-face winners
  shattered 19 k visible faces into 11.5 k single-face charts (thousands
  of seams); with the margin the same mesh lands at 7.3 k charts and
  6.5 k faces were painted from a chart view that was not their own
  per-face best (`margin_joined` in the meta). Splits that remain are
  honest: 21 % of disagreeing neighbour pairs fail the depth gate
  itself, and the meta's tier counts describe the paint, not the
  intent.
- **Shelf packing is deterministic** (sort by height, width, index): a
  single area-derived scale (capped at 1 and at the atlas edge, shrunk
  by ×0.8 until it fits) maps chart rectangles into
  `RECON_TEXTURE_SIZE` (default 1024²) with 2 texels of padding.
- **Barycentric bake.** Each chart triangle rasterizes into its packed
  rectangle; the texel's 3D position comes from the barycentric weights,
  is re-projected into the dominant view and samples that frame
  bilinearly, depth-gated again. Nearest chart wins overlaps; unsampied
  texels (seams, interior holes) are dilated inwards for 4 passes so
  linear filtering never reads black, and never-seen faces take a small
  fallback cell coloured from their own vertices.
- **Two self-checks** (both in the meta, like the dense stage's
  `sparse_check`). `mapping_check` compares each sampled face's centroid
  texel against a fresh sample of the very frame that painted it - the
  bake samples at the 3D point's projection while the UV maps through
  the chart bounding box, so this validates the whole bbox → pack → UV
  chain (synth p50 2.23). `cross_view_check` re-projects the same
  texels into *other* views, gated exactly like the bake (front-facing,
  in frame, depth-consistent) so occluded corners are skipped instead
  of blaming the bake for a correct pixel (synth: 220 samples, 430
  occluded skipped, p50 14.3). On the synthetic set lighting is
  world-fixed, so the deltas measure bake correctness rather than
  exposure changes.
- **OBJ with per-corner UVs** (`textured.obj` + `textured.mtl`), so
  chart seams need no vertex splitting and Blender (Phase 8) can import
  the textured mesh directly.

### Blender and BUSSID stages (honest fallback, package that cannot lie)

- **Blender is optional on purpose.** The stage resolves `BLENDER_PATH`
  (or `blender` on `PATH`) and, when it finds one, runs
  `blender_scripts/finalize_model.py` headless: import the textured OBJ,
  recalculate outward normals, re-export — **without welding**, because
  the atlas stores per-corner UVs and chart seams share vertex positions,
  so merge-by-distance would destroy the layout the texture stage just
  built. When nothing resolves, it copies the model through unchanged and
  says so: a `WARNING` in `logs/pipeline.log`, `fallback: true` in
  `blender_meta.json`, the fallback command recorded in the stage record.
  `REQUIRE_EXTERNAL_TOOLS=true` turns that degradation into an honest
  failure (exit 127) instead. The resolved binary is part of the stage
  fingerprint, so installing Blender later re-runs the stage instead of
  reusing a fallback checkpoint forever.
- **The engine pins the OBJ → MTL → texture chain itself** (`_finalize_sidecars`):
  exporters differ in how they name the MTL and write `map_Kd` (absolute
  paths, `//` prefixes, sometimes nothing), and a broken reference only
  surfaces when BUSSID loads the mod — exactly where nothing can report
  the problem. The files are rewritten only when the content differs, so
  a re-run never bumps mtimes. Light objects are **not** here: they
  belong to the web pipeline's database-backed analysis/creation stages.
- **`bussid/` mirrors `pipeline/bussid.py`**: copy model + every
  referenced sidecar (resolved from the file's own `mtllib`/`map_*`
  lines, flattened to basenames with the MTL rewritten to match) and a
  `manifest.json` (`format: bussid`, `lights: []`). The directory is
  rebuilt from scratch so stale files from an earlier layout cannot leak
  into a package.
- **`exports/<id>.bussidmod` mirrors `pipeline/exporter.py`**: root
  `manifest.json` (`format: bussidmod`), assets under `assets/bussid/…`,
  written to `.part` and atomically renamed, `exports/result.json` for
  consumers (`download_url: null` — the web backend builds its own URL).
  Two deliberate differences: only the deliverables are packaged (the web
  exporter sweeps whole workspace roots because it must package whatever
  previous jobs registered; intermediates must never reach a mod a phone
  has to import), and **every zip entry carries a fixed 1980-01-01
  timestamp** — `zipfile` would otherwise record each source mtime, so
  re-packaging identical bytes would produce a different archive and
  fail the determinism check. A 512 MB in-write cap fails honestly if
  intermediates ever do leak in (web default is 8 GiB because it packs
  whole roots).
- **Three structural evaluation checks** (`bussid_files_ok`,
  `bussid_package_ok`, `bussid_textures_ok`) guard the failures that
  only surface on the phone: truncated archive, wrong manifest format,
  or a model without its material/texture (loads grey, silently).

### Phase 10: the web hook (the `meshroom` stage runs this engine)

- **One pipeline, no second backend.** `backend/pipeline/meshroom.py`
  keeps its stage name, its place in the 13-stage sequence and its spot
  inside the single job worker, but `reconstruct()` now calls
  `run_pipeline(..., control=ctx.control, on_progress=..., through="texture")`.
  The `through=` slice exists so the web keeps owning what comes after
  geometry/texture — `mesh_validation → blender → lights → bussid_prep →
  export → package` — with its own stages (DB registrations, download
  URLs), while the standalone CLI runs the same engine through its own
  blender/bussid stages.  The old external-Meshroom command path stays
  available behind `MESHROOM_BACKEND=external`.
- **Two checkpoint systems that agree.** The web records the whole
  `meshroom` stage as one unit (done/pending in the project state); the
  engine keeps per-stage checkpoints in
  `reconstruction/project_state.json`.  A cancel or crash inside the
  stage leaves the web stage `pending` and the engine mid-plan
  resumable, so the next run re-enters exactly where the work stopped.
- **Cancel is not a failure.** The engine's `StateStore.stage()` lets
  `ProcessCancelled` pass through to a `cancel()` that resets the stage
  record to `pending` with `result.reason = "cancelled"` (no failure
  reason, nothing red).  The runner's existing
  `except ProcessCancelled` then sets the web stage back to `pending`
  and the job to `cancelled` — verified end-to-end.
- **Engine sub-stage visibility.** `on_progress` mirrors every engine
  stage transition onto the project log as
  `[MESHROOM] <operation>` lines, so the live UI shows which internal
  stage is working instead of a silent bar for ten minutes.
- **The bridge contract is the filesystem + registrations.** After the
  engine finishes, the hook copies `textured.obj`, `textured.mtl` and
  `texture.png` **verbatim** into `reconstruction/` (same names, so
  `mtllib`/`map_Kd` resolve — `mesh_validation` fails the stage if a
  material reference dangles) and registers them, plus the mesh/lowpoly
  point clouds and the report for the file browser.  `model_file()`
  prefers `blender/ > bussid/ > reconstruction/ > input/`, so the
  bridged model is what blender/lights/bussid/export consume without
  any of them knowing the engine exists.

- **End-to-end verification (web run, `projects/phase10_e2e`).** The same
  478 x 850 reference clip driven through the UI: validation → extraction
  (100 frames @ 2 fps) → filtering → `meshroom` (engine: matching 908 s,
  SfM 828 s, dense 31/31 views, mesh, texture — every sub-stage visible as
  `[MESHROOM]` lines in `logs/pipeline.log`) → `mesh_validation`
  **12,034 vertices / 20,000 faces / 1 material** → blender fallback →
  bussid → package **4,798,976 bytes, 10 assets**, download HTTP 200 and
  byte-identical to the on-disk `.bussidmod`.  Three failure modes were
  exercised along the way: a mid-engine cancel (job `cancelled`, web stage
  `pending`, no failure reason), a host power loss during matching (stage
  records and the 83 valid pair files survived the reboot; resume
  recomputed the three zero-byte writes and continued), and a service
  restart (job marked failed, engine checkpoints intact).
- **The delivered package is self-contained.** The blender fallback and
  the bussid stage carry the model's *referenced* material chain
  (`mtllib` → `map_Kd`) via `referenced_sidecars()` in
  `pipeline/common.py` instead of relying on a stem-name match alone, so
  `assets/bussid/prepared_textured.obj` resolves `textured.mtl` and
  `texture.png` **inside the zip** — verified by reading the archive, not
  just the directory listing.

## Forcing a re-run

- **Config change** → the stage fingerprint changes and the stage re-runs by
  itself (matching, for example, deletes its stale pair files).
- **Code change only** → fingerprints do not change, so delete the stage from
  `reconstruction/project_state.json` and unlink `logs/stage_<name>.json`,
  then `./vehicle-reconstruct resume <id>`.
- **New input/naming** → delete the whole `projects/<id>/` directory.
- **Web stages** (`mesh_validation` onward) → SQLite is authoritative:
  the `projects.state` column holds the stage records and the project's
  `project_state.json` is only a mirror of it.  The retry API only
  resets failed/incomplete stages, so re-running a *completed* web stage
  means setting its record back to `pending` in the database before
  `POST /api/projects/<id>/process`.

## Not built yet

Phase 10 (the web hook) is built and verified end-to-end — see the
verification bullets above.  What remains unexercised on this host: the
real Blender execution path (no Blender binary here — the logged,
WARNING-visible fallback is what runs), the legacy
`MESHROOM_BACKEND=external` path (the machine's Meshroom build is
CUDA-only, and this host has no NVIDIA GPU), and timestamp-fixed zip
entries on the *web* exporter (the engine's own package path fixes
every zip timestamp to 1980-01-01 for byte-determinism; the web
exporter keeps real mtimes, so web packages are content-stable but not
byte-identical across runs).
