"""Windowed observed-flow evidence, applied only to Gaussian strain weights."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

from modal_gaussians.camera_geometry import project_camera
from modal_gaussians.flow import spectrum
from modal_gaussians.flow.storage import open_array
from modal_gaussians.iteration_cache import cached, module_revision
from modal_gaussians.progress import Progress, report_progress
from . import edge_coherence


@dataclass(frozen=True)
class RigidityRefinementConfig:
    baseline_work_dir: str
    window_seconds: float = 10.0
    hop_seconds: float = 2.0
    strength: float = 0.5
    amplitude_floor_fraction: float = 0.05
    minimum_windows: int = 3
    minimum_effective_windows: float = 3.0

    def to_dict(self):
        if not isinstance(self.baseline_work_dir, str) or not Path(self.baseline_work_dir).is_absolute():
            raise ValueError("Rigidity baseline_work_dir must be an absolute path")
        for name in ("window_seconds", "hop_seconds", "minimum_effective_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Rigidity {name} must be finite and positive")
        if self.hop_seconds > self.window_seconds:
            raise ValueError("Rigidity hop cannot exceed window duration")
        for name in ("strength", "amplitude_floor_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"Rigidity {name} must lie in (0,1)")
        if type(self.minimum_windows) is not int or self.minimum_windows < 3:
            raise ValueError("Rigidity refinement requires at least three usable windows")
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        result = cls(**value)
        if result.to_dict() != dict(value):
            raise ValueError("Rigidity refinement configuration must be fully resolved")
        return result


def _view_responses(prepared, arrays, camera, view, slots, config):
    """Read each occupied spatial flow tile once, sharing it across windows/modes."""
    record = prepared.manifest["flows"][view]
    metadata = record["manifest"]["arrays"]["flow"]
    flow = open_array(Path(record["path"]) / metadata["file"])
    if flow.shape != tuple(metadata["shape"]) or np.dtype(flow.dtype) != np.float32:
        raise ValueError("Flow dimensions/type differ from the prepared observation contract")
    frames, height, width, components = flow.shape
    if components != 2:
        raise ValueError("Windowed flow requires two displacement components")
    fps = float(record["manifest"]["fps_hz"])
    window, hop = round(config.window_seconds * fps), round(config.hop_seconds * fps)
    if window < 3 or hop < 1 or window > frames:
        raise ValueError("Window/hop cannot be represented by this flow sequence")
    windows = 1 + (frames - window) // hop
    if windows < config.minimum_windows:
        raise ValueError("Too few time windows for rigidity refinement")
    points = arrays["g_points"]
    transform = camera.world_to_camera.detach().cpu().numpy()
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = project_camera(xyz, camera.K.detach().cpu().numpy(), camera.radial_distortion)
    projectable = np.isfinite(uv).all(1) & (xyz[:, 2] > 0)
    pixels = np.zeros((len(points), 2), np.int64)
    pixels[projectable] = np.rint(uv[projectable]).astype(np.int64)
    projectable &= ((pixels >= 0).all(1) & (pixels[:, 0] < width) & (pixels[:, 1] < height))
    rows = np.flatnonzero(projectable)
    mask = prepared.arrays[f"v{view}_mask"]
    projectable[rows] &= mask[pixels[rows, 1], pixels[rows, 0]]
    # Existing depth/alpha/contribution gates; these are not flow tracking confidence.
    visible = (arrays["u_surface_visible"][:, view] & projectable)
    usable = (arrays["observation_view_mask"][slots, :, view]
              & arrays["u_own_field_mask"][slots] & visible[None, :])
    selected = np.flatnonzero(usable.any(0))
    response = np.zeros((len(slots), len(points), windows, 2), np.complex64)
    tile_h, tile_w = flow.chunks[1:3]
    tile_columns = (width + tile_w - 1) // tile_w
    tile_ids = (pixels[selected, 1] // tile_h) * tile_columns + pixels[selected, 0] // tile_w
    order = np.argsort(tile_ids, kind="stable")
    selected, tile_ids = selected[order], tile_ids[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(tile_ids)) + 1, len(selected)] if len(selected) else np.array([0])
    progress = Progress(f"rigidity flow windows {prepared.source['views'][view]['label']}",
                        len(boundaries) - 1, unit="tiles")
    for tile_index, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ids = selected[lo:hi]
        tile = int(tile_ids[lo])
        top, left = (tile // tile_columns) * tile_h, (tile % tile_columns) * tile_w
        block = np.asarray(flow[:, top:min(top + tile_h, height), left:min(left + tile_w, width), :])
        values = block[:, pixels[ids, 1] - top, pixels[ids, 0] - left, :]
        for local, slot in enumerate(slots):
            frequency = float(prepared.source["modes"][slot]["frequency_hz"])
            response[local, ids] = edge_coherence.window_responses(
                values, fps, frequency, config.window_seconds, config.hop_seconds)
        progress.update(tile_index + 1)
    response *= usable[:, :, None, None]
    return response, usable, pixels


def build_rigidity_factors(prepared, arrays, cameras, values, slots, timer):
    config = RigidityRefinementConfig.from_dict(values)
    baseline = json.loads((Path(config.baseline_work_dir) / "manifest.json").read_text(encoding="utf-8"))
    contract = {"implementation": "window_phase_rigidity_v1", "baseline_run": baseline["run_identity"],
                "prepared": prepared.manifest["prepared_identity"], "config": config.to_dict(),
                "source_mode_slots": list(slots),
                "code": module_revision(edge_coherence, spectrum, sys.modules[__name__])}

    def compute():
        edges = arrays["g_edge_index"]
        shape = (len(prepared.source["modes"]), len(edges))
        numerator, evidence = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        for view, camera in enumerate(cameras):
            response, usable, pixels = _view_responses(prepared, arrays, camera, view, slots, config)
            distinct = (pixels[edges[:, 0]] != pixels[edges[:, 1]]).any(1)
            for local, slot in enumerate(slots):
                eligible = usable[local, edges].all(1) & distinct
                rows = np.flatnonzero(eligible)
                inconsistency, weight, count = edge_coherence.edge_phase_evidence(
                    response[local], edges[rows], amplitude_floor_fraction=config.amplitude_floor_fraction,
                    minimum_windows=config.minimum_windows,
                    minimum_effective_windows=config.minimum_effective_windows)
                numerator[slot, rows] += inconsistency
                evidence[slot, rows] += weight
                report_progress(f"rigidity evidence view={view} mode={slot}: windows={response.shape[2]} "
                                f"visible_points={int(usable[local].sum())} candidate_edges={len(rows)} "
                                f"usable_edges={int((weight > 0).sum())}")
            del response
        disagreement = np.divide(numerator, evidence, out=np.zeros_like(numerator), where=evidence > 0)
        # ponytail: availability gates are not calibrated tracking confidence; unknown edges stay unchanged.
        availability = np.minimum(evidence, 1.0)
        factors = np.clip(1.0 - config.strength * availability * disagreement, 1.0 - config.strength, 1.0)
        for slot in slots:
            quantiles = np.quantile(factors[slot], [0, .1, .5, .9, 1]).tolist()
            report_progress(f"rigidity factors mode={slot}: evidence_edges={int((evidence[slot] > 0).sum())} "
                            f"factor_min_p10_median_p90_max={quantiles}")
        return {"rigidity_edge_factor": factors.astype(np.float32),
                "rigidity_edge_coherence": (1.0 - disagreement).astype(np.float32),
                "rigidity_edge_evidence": availability.astype(np.float32)}

    return cached(prepared.cache_dir / "rigidity_evidence", contract, compute, timer, "rigidity_evidence")
