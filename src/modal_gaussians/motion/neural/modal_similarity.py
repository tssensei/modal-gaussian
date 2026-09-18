"""Build trusted KNN connections from visible endpoint modal similarity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Sequence

import numpy as np

from modal_gaussians.camera_geometry import project_camera, validate_radial_views
from modal_gaussians.motion.common.geometry_ops import (
    bilinear_sample_float64 as _bilinear, pixel_valid as _pixel_valid,
)
from modal_gaussians.progress import Progress
from .geometry_graph import GeometryGraph, _components


REASONS = {
    0: "similar", 1: "different", 2: "endpoint_not_visible",
    3: "view_unavailable", 4: "projected_distance_exceeds_limit",
    5: "patch_unreliable", 6: "both_low_signal", 7: "ambiguous_similarity",
}
STATUSES = {0: "retained", 1: "difference_rejected", 2: "unsupported"}


@dataclass(frozen=True)
class ModalSimilarityConfig:
    similarity_threshold: float = 0.20
    difference_threshold: float = 0.30
    amplitude_floor_fraction: float = 0.02
    amplitude_percentile: float = 99.0
    max_pixel_distance: float = 32.0
    patch_radius: int = 1
    patch_relative_dispersion_max: float = 0.30
    alpha_minimum: float = 0.05

    def validate(self) -> None:
        for name in ("similarity_threshold", "difference_threshold", "amplitude_floor_fraction",
                     "max_pixel_distance", "patch_relative_dispersion_max"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Modal similarity {name} must be finite and positive")
        if self.similarity_threshold >= self.difference_threshold:
            raise ValueError("Similarity threshold must be below difference threshold")
        if not math.isfinite(self.amplitude_percentile) or not 0 < self.amplitude_percentile <= 100:
            raise ValueError("Modal amplitude percentile must lie in (0,100]")
        if not math.isfinite(self.alpha_minimum) or not 0 < self.alpha_minimum <= 1:
            raise ValueError("Modal alpha minimum must lie in (0,1]")
        if isinstance(self.patch_radius, bool) or not isinstance(self.patch_radius, int) or self.patch_radius < 1:
            raise ValueError("Modal endpoint patch radius must be a positive integer")


def _endpoint_modes(field, mask, depth, alpha, pixels, z, visible, tolerance, floor, settings):
    """Robust local complex samples; depth visibility is a proxy, not splat ownership."""
    modes = np.zeros((len(pixels), 2), dtype=np.complex128)
    reliable = np.zeros(len(pixels), dtype=bool)
    nodes = np.flatnonzero(visible)
    radius = settings.patch_radius
    offsets = np.stack(np.meshgrid(np.arange(-radius, radius + 1),
                                  np.arange(-radius, radius + 1)), axis=-1).reshape(-1, 2)
    progress = Progress("Modal endpoint patches", len(nodes), unit="nodes")
    batch_size = max(1, min(32768, 300_000 // len(offsets)))
    for start in range(0, len(nodes), batch_size):
        rows = nodes[start:start + batch_size]
        xy = np.rint(pixels[rows]).astype(np.int64)[:, None, :] + offsets[None]
        valid = ((xy[..., 0] >= 0) & (xy[..., 0] < mask.shape[1])
                 & (xy[..., 1] >= 0) & (xy[..., 1] < mask.shape[0]))
        x, y = xy[..., 0].clip(0, mask.shape[1] - 1), xy[..., 1].clip(0, mask.shape[0] - 1)
        patch_depth, values = depth[y, x], field[y, x]
        valid &= (mask[y, x] & np.isfinite(values).all(axis=-1)
                  & (alpha[y, x] >= settings.alpha_minimum) & (patch_depth > 0)
                  & (np.abs(z[rows, None] - patch_depth)
                     <= (tolerance + 1e-8) * np.maximum(patch_depth, 1e-8)))
        enough = valid.sum(axis=1) >= 3
        if np.any(enough):
            rows, values, valid = rows[enough], values[enough], valid[enough]
            # Componentwise medians tolerate minority outliers in the local patch.
            median = (np.nanmedian(np.where(valid[..., None], values.real, np.nan), axis=1)
                      + 1j * np.nanmedian(np.where(valid[..., None], values.imag, np.nan), axis=1))
            deviation = np.linalg.norm(values - median[:, None], axis=-1)
            dispersion = (np.nanmedian(np.where(valid, deviation, np.nan), axis=1)
                          / np.maximum(np.linalg.norm(median, axis=1), floor))
            modes[rows] = median
            reliable[rows] = dispersion <= settings.patch_relative_dispersion_max
        progress.update(min(start + batch_size, len(nodes)))
    return modes, reliable


def build_modal_similarity_graph(
    graph: GeometryGraph, *, Ks: np.ndarray, world_to_cameras: np.ndarray,
    rendered_depths: Sequence[np.ndarray], rendered_alphas: Sequence[np.ndarray],
    endpoint_thresholds: np.ndarray, modal_fields: Sequence[np.ndarray],
    masks: Sequence[np.ndarray], radial_coefficients: np.ndarray | None = None,
    config: ModalSimilarityConfig | None = None,
) -> tuple[GeometryGraph, dict[str, np.ndarray], dict]:
    """Keep a candidate only with similar-motion evidence and no reliable conflict.

    Compare robust complex U/V endpoint patches independently in each view.
    No coverage or gradient is sampled along the virtual edge between them.
    Unknown observations never establish a connection or veto positive evidence.
    """
    settings = config or ModalSimilarityConfig()
    settings.validate()
    edges = np.asarray(graph.edge_index)
    if not np.array_equal(edges, graph.candidate_edge_index):
        raise ValueError("Modal similarity requires the full unfiltered KNN candidate graph")
    points = np.asarray(graph.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Modal similarity points must be nonempty finite [G,3]")
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= len(points))):
        raise ValueError("Modal similarity edges must be valid [E,2] indices")
    view_count = len(modal_fields)
    if not view_count or any(len(values) != view_count for values in (masks, rendered_depths, rendered_alphas)):
        raise ValueError("Modal fields, masks, depths and alphas must have matching nonempty views")
    Ks, poses = np.asarray(Ks, dtype=np.float64), np.asarray(world_to_cameras, dtype=np.float64)
    if (Ks.shape != (view_count, 3, 3) or poses.shape != (view_count, 4, 4)
            or not np.isfinite(Ks).all() or not np.isfinite(poses).all()
            or np.any(Ks[:, (0, 1), (0, 1)] <= 0)):
        raise ValueError("Modal cameras must be finite [V,3,3]/[V,4,4] with positive focal lengths")
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
        depth, alpha = np.asarray(rendered_depths[view]), np.asarray(rendered_alphas[view])
        if (field.ndim != 3 or field.shape[-1] != 2 or not np.iscomplexobj(field)
                or min(field.shape[:2]) < 3 or mask.dtype != bool or mask.shape != field.shape[:2]
                or depth.shape != mask.shape or alpha.shape != mask.shape):
            raise ValueError("Modal views require complex [H,W,2], bool mask and matching depth/alpha (H,W>=3)")
        if (not np.isfinite(depth).all() or not np.isfinite(alpha).all()
                or np.any(depth < 0) or np.any(alpha < 0) or np.any(alpha > 1)):
            raise ValueError("Rendered depth/alpha must be finite, depth >=0 and alpha in [0,1]")
        valid_modal = mask & np.isfinite(field).all(axis=-1)
        scale = (float(np.percentile(np.linalg.norm(field[valid_modal].astype(np.complex128), axis=-1),
                                     settings.amplitude_percentile)) if np.any(valid_modal) else 0.)
        scales[view] = scale
        available = math.isfinite(scale) and scale > 0 and np.isfinite(endpoint[view])
        reliable = np.zeros(len(points), dtype=bool)
        if available:
            reasons[:, view] = 2
            camera = points @ poses[view, :3, :3].T + poses[view, :3, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                pixels = project_camera(camera, Ks[view], radial[view])
            nodes = np.flatnonzero(_pixel_valid(pixels, mask.shape) & (camera[:, 2] > 0))
            sampled_depth = _bilinear(depth, pixels[nodes])
            visible = ((_bilinear(alpha, pixels[nodes]) >= settings.alpha_minimum)
                       & (_bilinear(mask.astype(np.float64), pixels[nodes]) >= 1 - 1e-8)
                       & (sampled_depth > 0)
                       & (np.abs(camera[nodes, 2] - sampled_depth)
                          <= (endpoint[view] + 1e-8) * np.maximum(sampled_depth, 1e-8)))
            visibility[nodes[visible], view] = True
            floor = settings.amplitude_floor_fraction * scale
            modes, reliable = _endpoint_modes(field, mask, depth, alpha, pixels, camera[:, 2],
                                              visibility[:, view], endpoint[view], floor, settings)
            selected = np.flatnonzero(visibility[edges[:, 0], view] & visibility[edges[:, 1], view])
            near = (np.linalg.norm(pixels[edges[selected, 0]] - pixels[edges[selected, 1]], axis=1)
                    <= settings.max_pixel_distance)
            reasons[selected, view] = 4
            selected = selected[near]
            reasons[selected, view] = 5
            selected = selected[reliable[edges[selected, 0]] & reliable[edges[selected, 1]]]
            amplitude = np.linalg.norm(modes, axis=1)
            a, b = edges[selected].T
            high_a, high_b = amplitude[a] >= floor, amplitude[b] >= floor
            reasons[selected, view] = 6
            signal = high_a | high_b
            selected, a, b = selected[signal], a[signal], b[signal]
            values = np.linalg.norm(modes[a] - modes[b], axis=1) / np.maximum(np.maximum(amplitude[a], amplitude[b]), floor)
            scores[selected, view] = values
            reasons[selected, view] = 7
            similar = (values <= settings.similarity_threshold) & high_a[signal] & high_b[signal]
            different = values >= settings.difference_threshold
            evidence[selected[similar], view] = 1
            evidence[selected[different], view] = -1
            reasons[selected[similar], view] = 0
            reasons[selected[different], view] = 1
        view_summaries.append({
            "view": view, "available": bool(available), "amplitude_scale": scale,
            "visible_nodes": int(visibility[:, view].sum()), "reliable_patch_nodes": int(reliable.sum()),
            "supported_edges": int(np.count_nonzero(evidence[:, view] == 1)),
            "different_edges": int(np.count_nonzero(evidence[:, view] == -1)),
            "unknown_edges": int(np.count_nonzero(evidence[:, view] == 0)),
            "reason_counts": {label: int(np.count_nonzero(reasons[:, view] == code)) for code, label in REASONS.items()},
        })
    supported, conflicted = np.any(evidence == 1, axis=1), np.any(evidence == -1, axis=1)
    keep = supported & ~conflicted
    status = np.where(conflicted, 1, np.where(keep, 0, 2)).astype(np.uint8)
    degree, components, sizes = _components(len(points), edges[keep])
    result = replace(
        graph, edge_index=edges[keep], edge_length=graph.edge_length[keep], edge_weight=graph.edge_weight[keep],
        edge_view_evidence=evidence[keep], edge_evidence_kind=np.ones(keep.sum(), dtype=np.int8),
        node_visible_view_mask=visibility, degree=degree, component_index=components,
        component_size=sizes, candidate_view_evidence=evidence,
    )
    diagnostics = {"candidate_scores": scores, "candidate_reason": reasons, "candidate_status": status,
                   "candidate_removed": ~keep, "view_amplitude_scale": scales}
    summary = {
        "config": asdict(settings), "fusion": "any_support_without_conflict",
        "score": "norm(mi-mj)/max(norm(mi),norm(mj),amplitude_floor)",
        "visibility": "projected_center_depth_alpha_proxy", "candidate_edges": len(edges),
        "retained_edges": int(keep.sum()), "removed_edges": int((~keep).sum()),
        "difference_rejected_edges": int(conflicted.sum()), "unsupported_edges": int((status == 2).sum()),
        "all_unknown_edges": int((~supported & ~conflicted).sum()),
        "supported_but_conflicted_edges": int((supported & conflicted).sum()),
        "components": len(sizes), "largest_component_nodes": int(sizes.max()),
        "singleton_components": int((sizes == 1).sum()), "connected_nodes": int((degree > 0).sum()),
        "reason_codes": REASONS, "status_codes": STATUSES, "views": view_summaries,
    }
    return result, diagnostics, summary
