"""Hard modal-gradient cuts with static occlusion used only as an evidence gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Sequence

import numpy as np

from modal_gaussians.camera_geometry import (
    project_camera, radial_path_length_bound, validate_radial_views,
)
from modal_gaussians.motion.common.geometry_ops import (
    bilinear_sample_float64 as _bilinear, pixel_valid as _pixel_valid,
)
from modal_gaussians.progress import Progress
from .geometry_graph import GeometryGraph, _components


# Unknown evidence never cuts an edge, including long edges.
REASONS = {
    0: "eligible_below_threshold", 1: "gradient_cut", 2: "endpoint_not_visible",
    3: "view_unavailable", 4: "projected_length_below_one_pixel",
    5: "path_outside_image", 6: "path_modal_or_coverage_invalid",
    7: "path_occluded",
}


@dataclass(frozen=True)
class ModalGradientConfig:
    gradient_threshold: float = 0.05
    amplitude_percentile: float = 99.0
    alpha_minimum: float = 0.05
    profile_min_samples: int = 5
    profile_max_step_pixels: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(self.gradient_threshold) or self.gradient_threshold <= 0:
            raise ValueError("Modal gradient threshold must be finite and positive")
        if not math.isfinite(self.amplitude_percentile) or not 0 < self.amplitude_percentile <= 100:
            raise ValueError("Modal amplitude percentile must lie in (0,100]")
        if not math.isfinite(self.alpha_minimum) or not 0 < self.alpha_minimum <= 1:
            raise ValueError("Modal alpha minimum must lie in (0,1]")
        if (isinstance(self.profile_min_samples, bool)
                or not isinstance(self.profile_min_samples, int) or self.profile_min_samples < 5):
            raise ValueError("Modal paths require at least five samples")
        if not math.isfinite(self.profile_max_step_pixels) or not 0 < self.profile_max_step_pixels <= 1:
            raise ValueError("Modal path sample spacing must lie in (0,1] pixels")


def _gradient(field: np.ndarray, mask: np.ndarray, percentile: float):
    """Joint complex U/V central differences; invalid stencils carry no evidence."""
    valid = mask & np.isfinite(field).all(axis=-1)
    clean = np.where(valid[..., None], field, 0)
    amplitude = np.linalg.norm(clean, axis=-1)
    scale = float(np.percentile(amplitude[valid], percentile)) if np.any(valid) else 0.0
    stencil = np.zeros(mask.shape, dtype=bool)
    stencil[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1]
                           & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:])
    gradient = np.zeros(mask.shape, dtype=np.float64)
    if math.isfinite(scale) and scale > 0:
        # One shared view scale retains spatial amplitude differences and phase.
        dx = (clean[1:-1, 2:] / scale - clean[1:-1, :-2] / scale) * 0.5
        dy = (clean[2:, 1:-1] / scale - clean[:-2, 1:-1] / scale) * 0.5
        gradient[1:-1, 1:-1] = np.sqrt(np.sum(np.abs(dx) ** 2 + np.abs(dy) ** 2, axis=-1))
        gradient[~stencil] = 0
    return gradient, stencil, scale


def filter_modal_gradient_graph(
    graph: GeometryGraph, *, Ks: np.ndarray, world_to_cameras: np.ndarray,
    rendered_depths: Sequence[np.ndarray], rendered_alphas: Sequence[np.ndarray],
    endpoint_thresholds: np.ndarray, modal_fields: Sequence[np.ndarray],
    masks: Sequence[np.ndarray], radial_coefficients: np.ndarray | None = None,
    config: ModalGradientConfig | None = None,
) -> tuple[GeometryGraph, dict[str, np.ndarray], dict]:
    """Cut a full KNN candidate edge if any observable view crosses a strong gradient.

    Depth never supplies negative evidence by itself: invisible endpoints,
    invalid paths and nearer occluders make a view unknown. A deeper rendered
    surface is not a contradiction because KNN segments may pass through air.
    Retained weights are copied unchanged; degree/components reflect actual cuts.
    """
    settings = config or ModalGradientConfig()
    settings.validate()
    edges = np.asarray(graph.edge_index)
    if not np.array_equal(edges, graph.candidate_edge_index):
        raise ValueError("Modal-gradient pruning requires the full unfiltered KNN candidate graph")
    points = np.asarray(graph.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Modal-gradient graph points must be nonempty finite [G,3]")
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= len(points))):
        raise ValueError("Modal-gradient edges must be valid [E,2] indices")
    view_count = len(modal_fields)
    if not view_count or any(len(values) != view_count for values in (masks, rendered_depths, rendered_alphas)):
        raise ValueError("Modal fields, masks, depths and alphas must have matching nonempty views")
    Ks, poses = np.asarray(Ks, dtype=np.float64), np.asarray(world_to_cameras, dtype=np.float64)
    if (Ks.shape != (view_count, 3, 3) or poses.shape != (view_count, 4, 4)
            or not np.isfinite(Ks).all() or not np.isfinite(poses).all()
            or np.any(Ks[:, (0, 1), (0, 1)] <= 0)):
        raise ValueError("Modal-gradient cameras must be finite [V,3,3]/[V,4,4] with positive focal lengths")
    endpoint = np.asarray(endpoint_thresholds, dtype=np.float64)
    if endpoint.shape != (view_count,) or np.any(np.isinf(endpoint)) or np.any(endpoint < 0):
        raise ValueError("Endpoint tolerances must be nonnegative [V], with NaN for unavailable")
    radial = validate_radial_views(radial_coefficients, view_count)
    evidence = np.zeros((len(edges), view_count), dtype=np.int8)
    reasons = np.full(evidence.shape, 3, dtype=np.uint8)
    scores = np.full(evidence.shape, np.nan, dtype=np.float32)
    visibility = np.zeros((len(points), view_count), dtype=bool)
    scales = np.zeros(view_count, dtype=np.float64)
    view_summaries = []
    for view in range(view_count):
        field, mask = np.asarray(modal_fields[view]), np.asarray(masks[view])
        depth = np.asarray(rendered_depths[view], dtype=np.float64)
        alpha = np.asarray(rendered_alphas[view], dtype=np.float64)
        if (field.ndim != 3 or field.shape[-1] != 2 or not np.iscomplexobj(field)
                or min(field.shape[:2]) < 3 or mask.dtype != bool or mask.shape != field.shape[:2]
                or depth.shape != mask.shape or alpha.shape != mask.shape):
            raise ValueError("Modal views require complex [H,W,2], bool mask and matching depth/alpha (H,W>=3)")
        if (not np.isfinite(depth).all() or not np.isfinite(alpha).all()
                or np.any(depth < 0) or np.any(alpha < 0) or np.any(alpha > 1)):
            raise ValueError("Rendered depth/alpha must be finite, depth >=0 and alpha in [0,1]")
        gradient, stencil, scale = _gradient(field, mask, settings.amplitude_percentile)
        scales[view] = scale
        if not math.isfinite(scale) or scale <= 0 or not np.isfinite(endpoint[view]):
            view_summaries.append({"view": view, "available": False, "amplitude_scale": scale})
            continue
        reasons[:, view] = 2
        camera = points @ poses[view, :3, :3].T + poses[view, :3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = project_camera(camera, Ks[view], radial[view])
        in_frame = _pixel_valid(pixels, mask.shape) & (camera[:, 2] > 0)
        nodes = np.flatnonzero(in_frame)
        sampled_depth = _bilinear(depth, pixels[nodes])
        visible = ((_bilinear(alpha, pixels[nodes]) >= settings.alpha_minimum)
                   & (_bilinear(mask.astype(np.float64), pixels[nodes]) >= 1 - 1e-8)
                   & (sampled_depth > 0)
                   & (np.abs(camera[nodes, 2] - sampled_depth)
                      <= (endpoint[view] + 1e-8) * np.maximum(sampled_depth, 1e-8)))
        visibility[nodes[visible], view] = True
        selected = np.flatnonzero(visibility[edges[:, 0], view] & visibility[edges[:, 1], view])
        projected_length = np.linalg.norm(pixels[edges[selected, 1]] - pixels[edges[selected, 0]], axis=1)
        reasons[selected[projected_length < 1], view] = 4
        selected, projected_length = selected[projected_length >= 1], projected_length[projected_length >= 1]
        if radial[view] and len(selected):
            projected_length = radial_path_length_bound(pixels[edges[selected, 0]], pixels[edges[selected, 1]], Ks[view], radial[view])
        sample_counts = np.maximum(settings.profile_min_samples, np.ceil(projected_length / settings.profile_max_step_pixels).astype(np.int64) + 1)
        stencil_float = stencil.astype(np.float64)
        progress = Progress(f"Modal gradient view {view + 1}/{view_count}", len(selected), unit="edges")
        completed = 0
        for sample_count in np.unique(sample_counts):
            matching = selected[sample_counts == sample_count]
            fractions = np.linspace(0, 1, int(sample_count))[None, :, None]
            batch_size = max(1, min(4096, 1_000_000 // int(sample_count)))
            for start in range(0, len(matching), batch_size):
                rows = matching[start:start + batch_size]
                a, b = camera[edges[rows, 0]][:, None], camera[edges[rows, 1]][:, None]
                # Uniform pinhole-image fractions -> true 3D segment fractions.
                # Projecting these points also handles curved SIMPLE_RADIAL paths.
                t = fractions * a[..., 2:3] / ((1 - fractions) * b[..., 2:3] + fractions * a[..., 2:3])
                path_camera = a * (1 - t) + b * t
                path = project_camera(path_camera, Ks[view], radial[view])
                inside = _pixel_valid(path, mask.shape).all(axis=1)
                reasons[rows, view] = 5
                completed += len(rows)
                rows, path, path_camera = rows[inside], path[inside], path_camera[inside]
                if len(rows):
                    xy = path.reshape(-1, 2)
                    path_depth = _bilinear(depth, xy).reshape(len(rows), -1)
                    coverage = ((_bilinear(alpha, xy).reshape(len(rows), -1) >= settings.alpha_minimum)
                                & (_bilinear(stencil_float, xy).reshape(len(rows), -1) >= 1 - 1e-8)
                                & (path_depth > 0)).all(axis=1)
                    reasons[rows, view] = 6
                    nearer = (path_camera[..., 2] - path_depth > (endpoint[view] + 1e-8) * np.maximum(path_depth, 1e-8)).any(axis=1)
                    reasons[rows[coverage & nearer], view] = 7
                    eligible = coverage & ~nearer
                    eligible_rows = rows[eligible]
                    if len(eligible_rows):
                        values = _bilinear(gradient, path[eligible].reshape(-1, 2)).reshape(len(eligible_rows), -1).max(axis=1)
                        cut = values > settings.gradient_threshold
                        scores[eligible_rows, view] = values
                        evidence[eligible_rows, view] = np.where(cut, -1, 1)
                        reasons[eligible_rows, view] = cut.astype(np.uint8)
                progress.update(completed)
        view_summaries.append({
            "view": view, "available": True, "amplitude_scale": scale,
            "visible_nodes": int(visibility[:, view].sum()),
            "reason_counts": {label: int(np.count_nonzero(reasons[:, view] == code)) for code, label in REASONS.items()},
        })
    removed = np.any(evidence == -1, axis=1)
    keep = ~removed
    degree, components, sizes = _components(len(points), edges[keep])
    result = replace(
        graph, edge_index=edges[keep], edge_length=graph.edge_length[keep],
        edge_weight=graph.edge_weight[keep], edge_view_evidence=evidence[keep],
        edge_evidence_kind=np.any(evidence[keep] == 1, axis=1).astype(np.int8),
        node_visible_view_mask=visibility, degree=degree, component_index=components,
        component_size=sizes, candidate_view_evidence=evidence,
    )
    diagnostics = {"candidate_scores": scores, "candidate_reason": reasons,
                   "candidate_removed": removed, "view_amplitude_scale": scales}
    summary = {"config": asdict(settings), "fusion": "any_observable_view",
               "candidate_edges": len(edges), "removed_edges": int(removed.sum()),
               "retained_edges": int(keep.sum()), "components": len(sizes),
               "reason_codes": REASONS, "views": view_summaries}
    return result, diagnostics, summary
