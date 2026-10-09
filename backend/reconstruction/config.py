"""CPU-only reconstruction configuration and hardware guards.

The whole engine is designed for a Dell Latitude E6410: 2 cores / 4 threads,
6 GB RAM, no NVIDIA GPU.  Every knob that affects memory or CPU time lives
here so it can be raised later without touching algorithm code.

Absolute rules enforced by this module:
  * no CUDA, no NVIDIA, no GPU acceleration, ever;
  * no silent GPU fallback - an algorithm that needs CUDA must fail loudly
    and name its CPU-compatible alternative;
  * never assume more than RAM_LIMIT_GB (default 4.0 of the 6 GB physical,
    leaving room for the OS and the FastAPI server);
  * thread count is configurable (default 2, 4 allowed).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CPU_ONLY = True  # global switch; see enforce_cpu_only()

# Human-readable list of what actually runs. Printed once at startup so an
# operator can see there is no hidden GPU path.
CPU_ALGORITHMS = {
    "frame_extraction": "FFmpeg (libx264 decode, CPU)",
    "frame_quality": "OpenCV Laplacian blur score, luminance, aHash duplicates",
    "feature_detection": "OpenCV SIFT (CPU, patent-free since 4.4)",
    "feature_matching": "OpenCV BFMatcher + Lowe ratio test (CPU)",
    "geometric_verification": "OpenCV RANSAC essential/fundamental matrix (CPU)",
    "pose_estimation": "OpenCV solvePnPRansac (CPU)",
    "bundle_adjustment": "SciPy sparse least_squares, Huber loss (CPU)",
    "dense_depth": "plane-sweep multi-view NCC over neighbours (CPU)",
    "meshing": "TSDF fusion + marching cubes (scikit-image, CPU)",
    "decimation": "quadric error metric simplification (fast_simplification, CPU)",
    "blender": "Blender headless --background (CPU)",
}


class ReconstructionError(RuntimeError):
    """A stage failed with a message a human can act on.

    Always carries a readable ``message``; ``suggestion`` names the fix and
    ``details`` holds machine-readable context for the report.
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        exit_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}
        self.exit_code = exit_code

    def human(self) -> str:
        text = self.message
        if self.suggestion:
            text += f" Suggestion: {self.suggestion}"
        return text


def enforce_cpu_only(enabled: bool = True) -> List[str]:
    """Disable every GPU pathway process-wide and log what will run.

    Returns the algorithm list so the caller can persist it in the report.
    Raises ReconstructionError if CPU-only mode is somehow switched off -
    this machine has no CUDA device, so a GPU request can only ever fail
    later, deep inside a stage, which is exactly what we refuse to allow.
    """
    global CPU_ONLY
    if not enabled:
        raise ReconstructionError(
            "CPU_ONLY=false is not supported on this host",
            suggestion="Set CPU_ONLY=true (this machine has no CUDA device); "
                       "GPU acceleration is intentionally not implemented.",
        )
    CPU_ONLY = True
    # Hide any GPU from any library that goes looking for one.  These are
    # read by OpenCV/OpenCL/CUDA runtimes at import time, so set them before
    # heavy work starts; they are harmless if the libraries ignore them.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    os.environ["OPENCV_FOR_THREADS_NUM"] = str(cpu_threads())
    try:
        import cv2

        cv2.setUseOptimized(True)
        cv2.setNumThreads(cpu_threads())
        # Belt and braces: if OpenCV was built with OpenCL, force it off.
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        # cv2 may not be importable in a unit test of the config alone.
        pass
    return [f"{name}: {desc}" for name, desc in CPU_ALGORITHMS.items()]


def require_cpu_algorithm(name: str, available: bool, alternative: str) -> None:
    """Guard used before running anything that could be GPU-backed."""
    if not available:
        raise ReconstructionError(
            f"Algorithm '{name}' requires CUDA/GPU support which is disabled "
            f"in CPU-only mode.",
            suggestion=f"Use the CPU-compatible alternative: {alternative}",
            details={"cpu_only": True, "algorithm": name},
        )


def cpu_threads() -> int:
    # Default 2 (2 physical cores); 4 is the documented optional maximum
    # (4 logical threads).  Higher values only cause cache thrash here.
    try:
        return max(1, min(4, int(os.getenv("RECON_THREADS", "2"))))
    except (TypeError, ValueError):
        return 2


def resolve_blender(configured: str) -> Optional[str]:
    """Resolve a Blender executable without invoking a shell.

    Mirrors the web backend's ``config.resolve_tool``: an explicit path wins
    (expanded, then ``PATH``-looked-up), otherwise ``blender`` on ``PATH``.
    Returns ``None`` when nothing is installed - the stage then degrades to
    its logged model-copy fallback unless the strict policy is set.
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        return shutil.which(configured)
    return shutil.which("blender")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Quality presets.  "medium" is the default for full-orbit vehicle work:
# it keeps enough frames and matches to register front + sides + rear on
# 6 GB RAM / 2 cores.  "low" stays for quick checks, "high" for cleaner
# geometry, and "dense" for maximum coverage (slower, higher memory).
# Runtime is explicitly NOT an optimization target - these exist to keep
# the process inside memory, not to finish quickly.
PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "max_frames": 120,
        "max_features": 3000,
        "max_image_size": 1100,
        "ba_max_points": 12000,
        "dense_neighbors": 5,
        "target_triangles": 30000,
    },
    "medium": {
        "max_frames": 220,
        "max_features": 4000,
        "max_image_size": 1300,
        "ba_max_points": 20000,
        "dense_neighbors": 6,
        "target_triangles": 40000,
    },
    "high": {
        "max_frames": 320,
        "max_features": 5000,
        "max_image_size": 1400,
        "ba_max_points": 32000,
        "dense_neighbors": 8,
        "target_triangles": 80000,
    },
    "dense": {
        "max_frames": 400,
        "max_features": 6000,
        "max_image_size": 1500,
        "ba_max_points": 32000,
        "dense_neighbors": 10,
        "target_triangles": 100000,
    },
}


@dataclass
class ReconConfig:
    """Everything the engine needs to run one project, CLI or server driven."""

    project_dir: Path

    # --- hardware policy -------------------------------------------------
    cpu_only: bool = True
    threads: int = 2                 # default 2; 4 allowed via --threads
    ram_limit_gb: float = 4.0        # of 6 GB physical; reduce batch if above
    quality: str = "low"             # preset: low | medium | high

    # --- phase 1: video -> frames ----------------------------------------
    # Denser default (0.3 s instead of 0.5 s): a walk-around video needs
    # overlapping views of front, sides, and rear or the match graph splits
    # and only one part (e.g. the back) registers.
    frame_interval: float = 0.3      # seconds between extracted frames
    max_frames: int = 220            # hard cap after extraction
    image_format: str = "jpg"
    jpeg_quality: int = 2            # ffmpeg -q:v scale (2 = visually good)
    # Frame quality thresholds (lenient on purpose: a slightly imperfect
    # frame may be the only view of the rear of the matatu).  Duplicate
    # rejection requires BOTH signals to agree - see frames.select_frames.
    min_blur_score: float = 12.0     # variance of Laplacian (lower = blurrier)
    min_brightness: float = 16.0     # 0..255 mean luminance
    max_brightness: float = 240.0
    duplicate_hamming_max: int = 2   # of 64 aHash bits (essentially identical)
    duplicate_corr_max: float = 0.992  # 16x16 thumbnail correlation

    # --- phase 2: features + matching -------------------------------------
    # Wider sequential window (5 instead of 3) plus the keyframe hop keeps
    # the orbit connected across blurry bridge frames; without it the match
    # graph splits and SfM seeds only one component (the "back only" model).
    max_image_size: int = 1300       # longest side before SIFT
    max_features: int = 4000         # SIFT nfeatures per image
    sift_octave_layers: int = 3      # SIFT nOctaveLayers
    sift_contrast_threshold: float = 0.02
    match_window: int = 5            # N vs N+1..N+5 sequential matches
    ratio_test: float = 0.75         # Lowe ratio
    min_matches: int = 25            # before geometric verification
    # RANSAC on 8-20 points happily "verifies" pairs that share no real
    # overlap at all; such pairs then poison feature tracks and no camera
    # can be registered from them.  Measured on the synthetic test set:
    # every pair with >= 31 inliers was geometrically correct, every pair
    # with <= 20 was not, so 25 keeps the trustworthy middle ground.
    min_inliers: int = 25            # after RANSAC
    ransac_threshold: float = 1.0    # pixels
    keyframe_skip: int = 4           # also match N vs N+skip to bridge loops

    # --- phase 3: SfM ------------------------------------------------------
    initial_fov_degrees: float = 60.0   # horizontal FOV guess (video, no EXIF)
    initial_focal_scale: float = 1.2    # fallback f ~= scale * max(w,h) if FOV=0
    # Focal multi-start.  A wrong FOV guess is not a small blemish: the
    # structure absorbs it, reprojection error stays excellent while the
    # geometry drifts (measured on the synthetic set: +10% focal error kept
    # the median reprojection at 1.1 px and still put camera positions 48%
    # off ground truth), and neither joint bundle adjustment nor a profile
    # walk can escape - the wrong basin is a genuine local minimum with a
    # steep wall around it.  So the guess is *tested*: a small contiguous
    # sub-model is rebuilt at several candidate focal lengths and the one
    # that best explains its observations is used for the real run.
    focal_multistart: bool = True
    focal_probe_frames: int = 12        # cameras in each probe sub-model
    focal_probe_scales: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.15)
    focal_override_px: float = 0.0      # chosen focal; 0 = derive from FOV
    sfm_frame_limit: int = 0            # probes: only a contiguous block
    ba_refine_intrinsics: bool = True
    ba_focal_prior: float = 0.25        # relative sigma when refining focal
    ba_max_points: int = 8000        # observation cap per BA round (memory)
    ba_max_nfev: int = 60            # iterations per round (RAM guard)
    # Final refit rounds: bundle adjustment's robust loss *down-weights* an
    # observation it cannot explain instead of removing it, so a wrong match
    # never actually leaves the model - it keeps tugging on its camera and
    # its neighbours and the error accumulates into a smooth drift along the
    # orbit (measured: 0.5 m in the middle of the sequence but 11 m at both
    # ends, even at the correct focal length).  Throwing away the
    # observations that stay bad after a refit, then refitting, breaks that
    # loop; each round also drops points left with a single view.
    ba_filter_rounds: int = 3
    min_pnp_correspondences: int = 8     # 2D-3D pairs required for PnP
    min_pnp_inliers: int = 8          # RANSAC inliers required
    # A correct pose often explains only part of the correspondence set:
    # points triangulated from a short baseline are imprecise, and a focal
    # length that bundle adjustment has not refined yet adds a systematic
    # few-pixel error.  What matters is a real consensus (>= 8 inliers)
    # plus a low mean error over those inliers afterwards.
    min_pnp_inlier_ratio: float = 0.15  # and this share of them
    max_reprojection_error: float = 4.0   # px, for point acceptance
    track_max_reprojection_error: float = 8.0  # px, drops a poisoned track pixel
    min_parallax_deg: float = 1.0         # triangulation baseline gate
    min_registered_ratio: float = 0.35    # reject sparse recon below this
    min_registered_images: int = 6
    # Pose sanity.  RANSAC PnP can "explain" a handful of nearly collinear
    # points with the camera at a nonsense distance; such a camera then
    # survives because bundle adjustment's robust loss down-weights exactly
    # the residuals that would expose it.  The already-registered cameras
    # define the working scale, so a candidate landing far outside it is
    # refused, and any camera that still ends up badly fitted after the
    # final bundle adjustment is dropped and reported.
    camera_radius_ratio_max: float = 3.0   # vs median registered distance
    camera_radius_ratio_min: float = 0.5   # vs structure radius (inside it?)
    camera_max_reprojection_error: float = 8.0  # px, per-camera mean
    camera_min_observations: int = 10      # too few to judge reliably
    # Vehicle front for coverage reporting (section 6).  "first" labels the
    # sector of the first registered view as front (walk-around videos usually
    # start there), a number labels that azimuth in degrees, and "unknown"
    # keeps neutral sector names instead of guessing.
    front_azimuth: Any = "first"

    # --- dense / mesh / texture stages ------------------------------------
    # Heavier defaults: more neighbours vote per pixel (6), higher working
    # resolution (600 px), and a larger plane budget (200) so sides and roof
    # get depth instead of only the best-observed panel.  NCC gate 0.25 keeps
    # more pixels than the old 0.30 without admitting noise.
    dense_neighbors: int = 6
    # Working resolution of the plane sweep (long side, px).  600 stays
    # affordable on CPU while giving the mesh stage markedly more to work
    # with.  Never upscales smaller frames.
    dense_max_size: int = 600
    # Depth hypotheses are spaced so that neighbouring planes displace the
    # image by at most this many pixels.  Wider spacing breaks the NCC match
    # outright - measured on realvid: 4% log-spacing at baseline 3.7 left
    # NCC at the true depth at 0.11 (depth error 35%), while 1.5 px
    # inverse-depth spacing gives 2.3% median error.
    dense_step_px: float = 1.5
    dense_max_planes: int = 200        # hard cap (cost volume = planes x pixels)
    dense_min_parallax: float = 0.02   # B / median depth: skip near-duplicates
    dense_block: int = 16              # px grid for per-pixel depth ranges
    dense_ncc_window: int = 9          # NCC support window (odd)
    dense_min_ncc: float = 0.25        # confidence gate for a depth pixel
    target_triangles: int = 40000
    texture_size: int = 1024
    # --- foreground masking ------------------------------------------------
    # Paint selected-frame backgrounds black before feature extraction so a
    # small subject owns the matches instead of the table/street behind it.
    # Env MASK_BACKGROUND=0 disables; MASK_BACKEND=rembg|keying|auto picks
    # the segmenter (rembg needs `pip install rembg onnxruntime`).
    mask_background: bool = True
    mask_backend: str = "auto"
    # rembg model for masking (u2netp = tiny/fast, u2net = large/slower).
    # Env MASK_MODEL overrides.
    mask_model: str = "u2netp"

    # --- external tooling (Phases 8-9) ------------------------------------
    # Empty means auto-resolve: BLENDER_PATH if set, else `blender` on PATH
    # (see resolve_blender).  When nothing resolves, the Blender stage
    # degrades to a logged model-copy fallback - unless the strict policy
    # is on, which fails the stage instead of shipping an unprepared model.
    blender_path: str = ""
    require_external_tools: bool = False

    # --- mesh (TSDF fusion + marching cubes) ------------------------------
    # The fusion volume is cropped to a ball around the point all cameras
    # look at (the orbit subject), radius = mesh_crop x median camera
    # distance to it.  A full-scene volume would spend almost all of its
    # voxels on distant background: measured on realvid the un-cropped
    # scene has diagonal 106 units, so the 4-unit vehicle would be 10
    # voxels across at 256^3; inside the ball the diagonal is 36 units and
    # the vehicle is 42 voxels across at 384^3.  If the crop removes too
    # much the stage falls back to the full extent and says so in the meta.
    mesh_crop: float = 1.5             # ball radius multiplier
    mesh_resolution: int = 448         # grid cells along the longest axis
    # Truncation band for a depth sample: max(voxels * size, fraction of
    # camera distance).  The relative term covers plane-sweep noise that
    # grows with depth (~2% of z); the voxel term keeps near geometry
    # tight.  Samples outside the band neither pull nor push the surface.
    mesh_trunc_voxels: float = 4.0
    mesh_trunc_rel: float = 0.01
    mesh_color_tol_voxels: float = 3.0  # vertex colour: |depth - z| gate
    # Multi-view consensus gates.  Depth fusion admits every confident
    # pixel, but on real footage (night, specular paint, residual pose
    # error) many samples are wrong *and* wrong differently in each view:
    # measured on realvid, 64% of covered voxels were hit by a single view
    # only, and the surface area at gate 1 was 10x the scene's real area.
    # Requiring mesh_min_weight views to agree per voxel, then pruning
    # vertices that confirm worse than mesh_support_rel times the
    # *validated-surface baseline* (how well the depth maps agree with the
    # reprojection-checked sparse points - the honest true-surface level
    # for THIS project; fixed absolutes fail across datasets: 0.54 on
    # realvid vs 0.35 on synth), removes the floating sheets while real
    # geometry survives.
    mesh_min_weight: int = 5
    mesh_support_rel: float = 0.5
    # Component anchoring: drop connected components that do not touch
    # validated surface.  Radius = mesh_anchor_rel * (median spacing of
    # sparse points), i.e. "anchored at the scale this project's surface
    # is validated at".  Measured on synth_selftest: 64% of vertices live
    # in 10k+ tiny fragments floating 0.3-2 units from the box; anchoring
    # at the sparse spacing removes them, taking bounding-box error from
    # +56/+163/+273% to -4/+13/+12% while retaining 73% of true surface.
    mesh_anchor_rel: float = 1.0

    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    @classmethod
    def for_project(
        cls,
        project_dir: Path | str,
        *,
        quality: Optional[str] = None,
        threads: Optional[int] = None,
        frame_interval: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_features: Optional[int] = None,
        target_triangles: Optional[int] = None,
        mask_background: Optional[bool] = None,
        mask_backend: Optional[str] = None,
        mask_model: Optional[str] = None,
        cpu_only: bool = True,
    ) -> "ReconConfig":
        project_dir = Path(project_dir).expanduser().resolve()
        preset_name = (quality or os.getenv("RECON_QUALITY") or "medium").lower()
        if preset_name not in PRESETS:
            raise ReconstructionError(
                f"Unknown quality preset '{preset_name}'",
                suggestion="Choose one of: " + ", ".join(sorted(PRESETS)),
            )
        preset = PRESETS[preset_name]
        cfg = cls(
            project_dir=project_dir,
            quality=preset_name,
            threads=max(1, min(4, threads if threads is not None else cpu_threads())),
            ram_limit_gb=_f("RECON_RAM_LIMIT_GB", 4.0),
            frame_interval=max(0.05, float(
                frame_interval if frame_interval is not None
                else _f("RECON_FRAME_INTERVAL", 0.3))),
            max_frames=_i("RECON_MAX_FRAMES", int(preset["max_frames"])),
            max_image_size=int(
                max_image_size if max_image_size is not None
                else preset["max_image_size"]),
            max_features=int(
                max_features if max_features is not None
                else preset["max_features"]),
            ba_max_points=int(preset["ba_max_points"]),
            dense_neighbors=int(preset["dense_neighbors"]),
            target_triangles=int(
                target_triangles if target_triangles is not None
                else _i("RECON_TARGET_TRIANGLES",
                        int(preset["target_triangles"]))),
            texture_size=_i("RECON_TEXTURE_SIZE", 1024),
            mask_background=(bool(mask_background) if mask_background is not None
                             else os.getenv("MASK_BACKGROUND", "1").strip().lower()
                             not in {"0", "false", "no", "off"}),
            mask_backend=(str(mask_backend or os.getenv("MASK_BACKEND", "auto")).strip().lower()
                          or "auto"),
            mask_model=(str(mask_model or os.getenv("MASK_MODEL", "u2netp")).strip()
                        or "u2netp"),
            blender_path=os.getenv("BLENDER_PATH", "").strip(),
            require_external_tools=os.getenv(
                "REQUIRE_EXTERNAL_TOOLS", "").strip().lower()
            in {"1", "true", "yes", "on"},
            dense_max_size=_i("RECON_DENSE_MAX_SIZE", 600),
            dense_step_px=_f("RECON_DENSE_STEP_PX", 1.5),
            dense_max_planes=_i("RECON_DENSE_MAX_PLANES", 200),
            dense_min_parallax=_f("RECON_DENSE_MIN_PARALLAX", 0.02),
            dense_block=_i("RECON_DENSE_BLOCK", 16),
            dense_min_ncc=_f("RECON_DENSE_MIN_NCC", 0.25),
            mesh_crop=_f("RECON_MESH_CROP", 1.5),
            mesh_resolution=_i("RECON_MESH_RESOLUTION", 448),
            mesh_trunc_voxels=_f("RECON_MESH_TRUNC_VOXELS", 4.0),
            mesh_trunc_rel=_f("RECON_MESH_TRUNC_REL", 0.01),
            mesh_color_tol_voxels=_f("RECON_MESH_COLOR_TOL", 3.0),
            mesh_min_weight=_i("RECON_MESH_MIN_WEIGHT", 5),
            mesh_support_rel=_f("RECON_MESH_SUPPORT_REL", 0.5),
            mesh_anchor_rel=_f("RECON_MESH_ANCHOR_REL", 1.0),
            match_window=_i("RECON_MATCH_WINDOW", 5),
            min_registered_ratio=_f("RECON_MIN_REGISTERED_RATIO", 0.35),
            initial_fov_degrees=_f("RECON_INITIAL_FOV", 60.0),
            focal_multistart=(os.getenv("RECON_FOCAL_MULTISTART", "1")
                              not in {"0", "false", "no"}),
            # An operator who already knows the camera's focal length (or who
            # is validating the automatic estimate against a calibration) can
            # pin it; a pinned focal skips the probe grid entirely so the
            # number used is exactly the number supplied.
            focal_override_px=_f("RECON_FOCAL_OVERRIDE", 0.0),
            cpu_only=bool(cpu_only),
        )
        return cfg

    # ---------------------------------------------------------------------
    @property
    def input_dir(self) -> Path:
        return self.project_dir / "input"

    def d(self, *parts: str) -> Path:
        """A path inside the project (created on demand by ensure_layout)."""
        return self.project_dir.joinpath(*parts)

    def ensure_layout(self) -> None:
        """Create the isolated per-project directory tree (spec section 28)."""
        for parts in (
            ("input",),
            ("frames", "original"),
            ("frames", "selected"),
            ("frames", "rejected"),
            ("features",),
            ("matches",),
            ("sfm",),
            ("sparse",),
            ("depth",),
            ("dense",),
            ("mesh", "raw"),
            ("mesh", "cleaned"),
            ("mesh", "lowpoly"),
            ("textures",),
            ("blender",),
            ("lights",),
            ("bussid",),
            ("logs",),
            ("metadata",),
            ("reconstruction",),
        ):
            self.d(*parts).mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        # The web backend owns <project>/project_state.json with its own
        # schema, so engine checkpoints live in reconstruction/ beside it.
        return self.d("reconstruction", "project_state.json")

    @property
    def report_path(self) -> Path:
        return self.project_dir / "reconstruction_report.json"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["project_dir"] = str(self.project_dir)
        return data

    def fingerprint(self) -> str:
        """Cheap change detector: same fingerprint => safe to skip redoing work."""
        keys = (
            "quality", "threads", "frame_interval", "max_frames",
            "max_image_size", "max_features", "sift_octave_layers",
            "sift_contrast_threshold", "match_window", "ratio_test",
            "mask_background", "mask_backend", "mask_model",
        )
        return "|".join(f"{k}={getattr(self, k)}" for k in keys)

    def feature_fingerprint(self) -> str:
        """Invalidates SIFT .npz files (image pixels / descriptor settings)."""
        keys = (
            "max_image_size", "max_features", "sift_octave_layers",
            "sift_contrast_threshold",
        )
        return "|".join(f"{k}={getattr(self, k)}" for k in keys)

    def match_fingerprint(self) -> str:
        """Invalidates pair .npz files (matching + verification settings).

        Kept separate from the feature fingerprint so that widening the
        ratio test or the sequential window does not throw away expensive
        SIFT descriptors.
        """
        keys = (
            "max_image_size", "max_features", "sift_octave_layers",
            "sift_contrast_threshold", "match_window", "ratio_test",
            "min_matches", "min_inliers", "ransac_threshold",
            "keyframe_skip", "initial_fov_degrees",
        )
        return "|".join(f"{k}={getattr(self, k)}" for k in keys)

    def sfm_fingerprint(self) -> str:
        """Invalidates the SfM stage (pose/track/BA settings)."""
        return self.match_fingerprint() + "|" + "|".join(
            f"{k}={getattr(self, k)}" for k in (
                "min_parallax_deg", "max_reprojection_error",
                "track_max_reprojection_error", "ba_max_nfev",
                "ba_max_points", "ba_filter_rounds", "ba_refine_intrinsics",
                "ba_focal_prior",
                "min_pnp_correspondences", "min_pnp_inliers",
                "min_pnp_inlier_ratio", "min_registered_ratio",
                "min_registered_images", "front_azimuth",
                "camera_radius_ratio_max", "camera_radius_ratio_min",
                "camera_max_reprojection_error", "camera_min_observations",
                "focal_multistart", "focal_probe_frames",
                "focal_probe_scales", "focal_override_px",
                "sfm_frame_limit",
            ))

    def dense_fingerprint(self) -> str:
        """Invalidates depth maps when poses *or* sweep settings change.

        The dense stage reads the sparse reconstruction as its ground truth
        (poses define the warps, the sparse cloud defines every pixel's depth
        range), so it inherits ``sfm_fingerprint`` and adds only its own
        sweep parameters.
        """
        return self.sfm_fingerprint() + "|dense|" + "|".join(
            f"{k}={getattr(self, k)}" for k in (
                "dense_neighbors", "dense_max_size", "dense_step_px",
                "dense_max_planes", "dense_min_parallax", "dense_block",
                "dense_ncc_window", "dense_min_ncc",
                "mask_background", "mask_backend", "mask_model",
            ))

    def mesh_fingerprint(self) -> str:
        """Invalidates the mesh when depth maps *or* fusion settings change.

        The mesh stage consumes the dense depth maps (poses project the
        volume, depth maps fill the TSDF), so it inherits
        ``dense_fingerprint`` and adds only its own fusion parameters.
        """
        return self.dense_fingerprint() + "|mesh|" + "|".join(
            f"{k}={getattr(self, k)}" for k in (
                "mesh_crop", "mesh_resolution", "mesh_trunc_voxels",
                "mesh_trunc_rel", "mesh_color_tol_voxels",
                "mesh_min_weight", "mesh_support_rel", "mesh_anchor_rel",
            ))

    def lowpoly_fingerprint(self) -> str:
        """Invalidates the decimated mesh when the raw mesh or the target
        triangle budget changes; inherits everything upstream of the mesh.
        """
        return self.mesh_fingerprint() + "|lowpoly|" + "|".join(
            f"{k}={getattr(self, k)}" for k in ("target_triangles",))

    def texture_fingerprint(self) -> str:
        """Invalidates the bake when the decimated mesh or atlas size
        changes; inherits everything upstream of the low-poly stage.
        """
        return self.lowpoly_fingerprint() + "|texture|" + "|".join(
            f"{k}={getattr(self, k)}" for k in ("texture_size",))

    def blender_fingerprint(self) -> str:
        """Invalidates the Blender stage when the textured model, the tool
        itself, or the strict-tools policy changes.

        The *resolved* binary is part of the fingerprint on purpose:
        installing Blender after a fallback run must redo the stage instead
        of reusing the model-copy checkpoint forever.
        """
        return (self.texture_fingerprint() + "|blender|"
                f"tool={resolve_blender(self.blender_path)}"
                f"|require_external_tools={self.require_external_tools}")

    def bussid_fingerprint(self) -> str:
        """Invalidates BUSSID prep/package when its inputs (the prepared
        model) or the package layout change; inherits everything upstream.
        """
        return self.blender_fingerprint() + "|bussid|package=v1"

    def stage_fingerprint(self, stage: str) -> Optional[str]:
        """Fingerprint that decides whether a completed stage is still valid.

        Checkpoints alone would happily replay stale work forever; pairing
        them with a fingerprint means tuning a parameter actually redoes the
        affected stage and nothing else (section 24).
        """
        mapping = {
            "input_validation": None,
            "frame_extraction": "fingerprint",
            "frame_selection": "fingerprint",
            "masking": "fingerprint",
            "feature_extraction": "feature_fingerprint",
            "feature_matching": "match_fingerprint",
            "sfm": "sfm_fingerprint",
            "dense": "dense_fingerprint",
            "mesh": "mesh_fingerprint",
            "lowpoly": "lowpoly_fingerprint",
            "texture": "texture_fingerprint",
            "blender": "blender_fingerprint",
            "bussid": "bussid_fingerprint",
        }
        name = mapping.get(stage)
        return getattr(self, name)() if name else None
