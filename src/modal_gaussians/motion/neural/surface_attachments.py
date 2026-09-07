"""v11 host-only fields with size-aware, spatially varying surface attachments.

All decisions are frozen geometry operations. Gaussian covariance describes a
finite support approximation, not a recovered physical surface. Ambiguous or
distant points remain explicit unresolved points; no foreground labels change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .fragment_propagation import FragmentPropagationConfig, build_attachments, core_mask
from .geometry_graph import build_control_graph

VERSION = 11
METHOD = "neural_field_with_surface_attachments"
STRATEGY = "surface"
STATUS = {0: "host", 1: "attached", 2: "outside_search_radius", 3: "support_gap_too_large",
          4: "ambiguous_host", 5: "outside_selected_host_support"}
INPUT_NAMES = {"a_covariance", "a_view_visible", "a_view_depth", "a_view_depth_tolerance"}
ARRAY_NAMES = INPUT_NAMES | {
    "f_candidate_component_mask", "f_component_extent", "f_core_mask", "f_component_status", "f_host_component",
    "a_provisional_control_count", "a_legacy_preserved", "a_point_status", "a_point_host",
    "a_nearest_host_distance", "a_anchor_indptr", "a_anchor_indices", "a_anchor_weights",
    "t_host_gaussian_index", "t_motion_component_index", "t_interpolation_indptr",
    "t_interpolation_indices", "t_interpolation_weights",
}


@dataclass(frozen=True)
class SurfaceAttachmentConfig:
    strategy: str = STRATEGY
    min_host_nodes: int = 101
    min_host_controls: int = 2
    search_radius_fraction: float = 0.03
    support_sigma: float = 2.0
    max_support_gap: float = 0.008
    patch_radius: float = 0.008
    core_preference: float = 1.25
    ambiguity_ratio: float = 1.25
    smoothing_steps: int = 4
    smoothing_strength: float = 0.25
    preserve_legacy: bool = True
    legacy_config: dict = field(default_factory=lambda: FragmentPropagationConfig().to_dict())

    def validate(self):
        if self.strategy != STRATEGY:
            raise ValueError("Unknown surface attachment strategy")
        for name in ("min_host_nodes", "min_host_controls"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Surface {name} must be a positive integer")
        if type(self.smoothing_steps) is not int or self.smoothing_steps < 0:
            raise ValueError("Surface smoothing_steps must be a nonnegative integer")
        for name in ("search_radius_fraction", "support_sigma", "max_support_gap", "patch_radius", "core_preference", "ambiguity_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Surface {name} must be finite and positive")
        if self.ambiguity_ratio < 1 or self.core_preference < 1:
            raise ValueError("Surface preference and ambiguity ratios must be >=1")
        if (isinstance(self.smoothing_strength, bool) or not math.isfinite(self.smoothing_strength)
                or not 0 <= self.smoothing_strength < 1):
            raise ValueError("Surface smoothing_strength must be in [0,1)")
        if type(self.preserve_legacy) is not bool:
            raise ValueError("Surface preserve_legacy must be boolean")
        FragmentPropagationConfig.from_dict(self.legacy_config)

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        result = cls(**value)
        if result.to_dict() != dict(value):
            raise ValueError("Surface configuration must contain all resolved settings")
        return result


def scene_covariance(scene):
    active = scene.foreground.active()
    scales = active["scales"].detach().cpu().numpy().astype(np.float64)
    quats = active["quaternions"].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(quats).all():
        raise ValueError("Surface attachment requires finite positive Gaussian scales and quaternions")
    rotation = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()  # gsplat uses wxyz
    return (rotation * scales[:, None, :] ** 2) @ rotation.transpose(0, 2, 1)


def observation_inputs(scene, cameras, depths, alphas, endpoint_thresholds, contribution_mass, contribution_threshold):
    """Use persisted static depth and actual contribution; occlusion is unknown."""
    from modal_gaussians.camera_geometry import project_camera
    points = scene.foreground.active()["means"].detach().cpu().numpy().astype(np.float64)
    G, V = len(points), len(cameras)
    visible = np.zeros((G, V), dtype=bool)
    camera_depth = np.zeros((G, V), dtype=np.float64)
    tolerances = np.where(np.isfinite(endpoint_thresholds), endpoint_thresholds, 0.).astype(np.float64)
    for v, camera in enumerate(cameras):
        w2c = camera.world_to_camera.detach().cpu().numpy()
        camera_points = points @ w2c[:3, :3].T + w2c[:3, 3]
        camera_depth[:, v] = camera_points[:, 2]
        uv = project_camera(camera_points, camera.K.detach().cpu().numpy(), camera.radial_distortion)
        valid = np.isfinite(uv).all(1) & (camera_points[:, 2] > 0)
        xy = np.zeros((G, 2), dtype=np.int64)
        xy[valid] = np.rint(uv[valid]).astype(np.int64)
        H, W = depths[v].shape
        valid &= (xy >= 0).all(1) & (xy[:, 0] < W) & (xy[:, 1] < H)
        rows = np.flatnonzero(valid)
        z = depths[v][xy[rows, 1], xy[rows, 0]]
        opacity = alphas[v][xy[rows, 1], xy[rows, 0]]
        visible[rows, v] = ((z > 0) & np.isfinite(z) & (opacity >= .05) & (tolerances[v] > 0)
                            & (np.abs(camera_points[rows, 2] - z) <= tolerances[v])
                            & (contribution_mass[rows, v] > contribution_threshold[v]))
    return {"a_covariance": scene_covariance(scene), "a_view_visible": visible,
            "a_view_depth": camera_depth, "a_view_depth_tolerance": tolerances}


def validate_inputs(inputs, G):
    if set(inputs) != INPUT_NAMES:
        raise ValueError("Surface attachment input inventory differs")
    cov = inputs["a_covariance"]
    visible = inputs["a_view_visible"]
    if (cov.dtype != np.float64 or cov.shape != (G, 3, 3) or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.transpose(0, 2, 1), atol=1e-14, rtol=1e-12)
            or np.any(np.linalg.eigvalsh(cov) <= 0)):
        raise ValueError("Surface covariance must be symmetric positive definite float64 [G,3,3]")
    if visible.dtype != np.bool_ or visible.ndim != 2 or visible.shape[0] != G:
        raise ValueError("Surface visibility must be boolean [G,V]")
    for name, shape in (("a_view_depth", visible.shape), ("a_view_depth_tolerance", (visible.shape[1],))):
        if inputs[name].dtype != np.float64 or inputs[name].shape != shape or not np.isfinite(inputs[name]).all():
            raise ValueError(f"Surface {name} shape/dtype/finiteness differs")
    if np.any(inputs["a_view_depth_tolerance"] < 0):
        raise ValueError("Surface depth tolerance must be nonnegative")


def support_distance(points, covariance, left, right, sigma):
    """Center-line projected ellipsoid support gap, with a separate distance cap."""
    delta = points[right] - points[left]
    distance = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(distance[..., None], 1e-30)
    radius_left = np.sqrt(np.maximum(np.einsum('...i,...ij,...j->...', direction, covariance[left], direction), 0))
    radius_right = np.sqrt(np.maximum(np.einsum('...i,...ij,...j->...', direction, covariance[right], direction), 0))
    return distance, np.maximum(0, distance - sigma * (radius_left + radius_right))


def _normalize(matrix):
    matrix = matrix.tocsr()
    total = np.asarray(matrix.sum(1)).ravel()
    matrix.data /= np.repeat(np.maximum(total, 1e-30), np.diff(matrix.indptr))
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def build_surface_controls(graph, *, geometry_config, fragment_config, scene_scale, attachment_inputs):
    from .training_fragments import host_subgraph
    settings = SurfaceAttachmentConfig.from_dict(fragment_config)
    p = graph.points.astype(np.float64); c = graph.component_index; sizes = graph.component_size
    G, C = len(p), len(sizes)
    validate_inputs(attachment_inputs, G)
    cov = attachment_inputs["a_covariance"]
    radius = settings.search_radius_fraction * scene_scale
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Surface search radius must be positive")
    eligible = np.flatnonzero((sizes >= settings.min_host_nodes)[c])
    if not len(eligible):
        raise ValueError("Surface attachment has no host component meeting min_host_nodes")
    # Count controls at the unchanged h, without letting candidates consume the final budget.
    provisional = build_control_graph(host_subgraph(graph, eligible),
        config=replace(geometry_config, max_controls=max(geometry_config.max_controls, len(eligible))), scene_scale=scene_scale)
    control_count = np.bincount(c[eligible[provisional.control_point_index]], minlength=C).astype(np.int64)
    main = (sizes >= settings.min_host_nodes) & (control_count >= settings.min_host_controls)
    hosts = np.flatnonzero(main[c]).astype(np.int64)
    if not len(hosts):
        raise ValueError("Surface attachment has no host component meeting min_host_controls")
    host_graph = host_subgraph(graph, hosts)
    controls = build_control_graph(host_graph, config=geometry_config, scene_scale=scene_scale)
    edges = graph.edge_index
    adjacency = csr_matrix((np.tile(graph.edge_length, 2),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(G, G))
    core = core_mask(adjacency, 3)
    ordered = np.argsort(c, kind="stable"); offsets = np.r_[0, np.cumsum(sizes)]
    members = [ordered[lo:hi] for lo, hi in zip(offsets[:-1], offsets[1:])]
    extent = np.array([np.linalg.norm(np.ptp(p[m], axis=0)) for m in members])
    legacy = (build_attachments({"g_"+k:v for k,v in graph.as_dict().items()},
                               FragmentPropagationConfig.from_dict(settings.legacy_config)) if settings.preserve_legacy else None)
    host_tree = cKDTree(p[hosts])
    nearest = host_tree.query(p)[0].astype(np.float64)
    point_status = np.zeros(G, np.int8); point_host = c.copy()
    point_status[~main[c]] = 2; point_host[~main[c]] = -1
    component_status = np.zeros(C, np.int8); owner = np.full(C, -1, np.int64)
    preserved = np.zeros(C, bool)
    anchor_rows, anchor_cols, anchor_data = [], [], []
    visible = attachment_inputs["a_view_visible"]; depth = attachment_inputs["a_view_depth"]
    tolerance = attachment_inputs["a_view_depth_tolerance"]
    local_graphs = {}
    patches = {}

    def patch_for(seed):
        # Work in the host component, not a G-sized Dijkstra buffer per query.
        host = int(c[seed])
        if seed not in patches:
            if host not in local_graphs:
                nodes = members[host]
                local_graphs[host] = adjacency[nodes][:, nodes]
            nodes = members[host]
            local = int(np.searchsorted(nodes, seed))
            distance = dijkstra(local_graphs[host], directed=False, indices=local, limit=settings.patch_radius)
            patches[seed] = nodes[distance <= settings.patch_radius]
        return patches[seed]

    def append(row, anchors, weights):
        anchor_rows.extend([int(row)] * len(anchors)); anchor_cols.extend(anchors.tolist()); anchor_data.extend(weights.tolist())

    for comp in np.flatnonzero(~main):
        m = members[comp]
        # Keep the old successful patch exactly when its host remains eligible and support is valid.
        if legacy is not None and legacy['f_component_status'][comp] == 1:
            host = int(legacy['f_host_component'][comp])
            lo, hi = legacy['f_anchor_indptr'][comp:comp+2]
            anchors = legacy['f_anchor_indices'][lo:hi]; beta = legacy['f_anchor_weights'][lo:hi]
            d, gap = support_distance(p, cov, m[:,None], anchors[None], settings.support_sigma)
            if main[host] and np.all(d <= radius) and np.all(gap <= settings.max_support_gap):
                for i in m: append(i, anchors, beta)
                point_status[m] = 1; point_host[m] = host; owner[comp] = host; component_status[comp] = 1
                preserved[comp] = True
                continue
        # Score each host by point coverage, then center distance. Visibility/depth
        # resolves close alternatives only; an unobserved view is never a veto.
        options = {}; point_candidates = {}
        for i in m:
            candidates = hosts[host_tree.query_ball_point(p[i], radius)]
            if not len(candidates): continue
            point_status[i] = 3
            d, gap = support_distance(p, cov, i, candidates, settings.support_sigma)
            keep = gap <= settings.max_support_gap
            candidates, d = candidates[keep], d[keep]
            if not len(candidates): continue
            point_candidates[int(i)] = candidates
            for host in np.unique(c[candidates]):
                group = candidates[c[candidates] == host]
                distances = d[c[candidates] == host]
                seed = int(group[np.lexsort((group, distances))[0]])
                score = int(np.sum(visible[i] & visible[seed] & (np.abs(depth[i]-depth[seed]) <= tolerance)))
                options.setdefault(int(host), []).append((int(i), seed, float(distances.min()), score))
        if not options:
            component_status[comp] = 2 if np.all(point_status[m] == 2) else 3
            continue
        ordered_hosts = sorted(options, key=lambda h: (-len(options[h]), np.median([x[2] for x in options[h]]), h))
        best = ordered_hosts[0]
        baseline_distance = np.median([x[2] for x in options[best]])
        alternatives = [h for h in ordered_hosts if len(options[h]) == len(options[best])
                        and np.median([x[2] for x in options[h]]) <= settings.ambiguity_ratio * max(baseline_distance, 1e-12)]
        if len(alternatives) > 1:
            votes = {h: sum(x[3] for x in options[h]) for h in alternatives}
            winners = [h for h in alternatives if votes[h] == max(votes.values())]
            if len(winners) != 1:
                point_status[list(point_candidates)] = 4; component_status[comp] = 4
                continue
            best = winners[0]
        owner[comp] = best; component_status[comp] = 1
        for i in point_candidates: point_status[i] = 5
        for i, seed, _, _ in options[best]:
            patch = patch_for(seed)
            d, gap = support_distance(p, cov, i, patch, settings.support_sigma)
            keep = (d <= radius) & (gap <= settings.max_support_gap)
            patch, d = patch[keep], d[keep]
            # The accepted seed is always retained, including degree-1 boundary points.
            beta = np.exp(-np.square((d - d.min()) / settings.patch_radius))
            beta *= np.where(core[patch], settings.core_preference, 1.)
            beta /= beta.sum()
            append(i, patch, beta)
            point_status[i] = 1; point_host[i] = best

    A = coo_matrix((anchor_data, (anchor_rows, anchor_cols)), shape=(G, G)).tocsr()
    # Smooth *geometric weights*, not trainable motion, and never cross hosts or
    # extend support to rejected points. Preserved legacy rows stay bitwise fixed.
    selected_edges = edges[(point_status[edges[:,0]] == 1) & (point_status[edges[:,1]] == 1)
                           & (point_host[edges[:,0]] == point_host[edges[:,1]]) & ~preserved[c[edges[:,0]]]]
    W = coo_matrix((np.ones(2*len(selected_edges)),
                   (np.r_[selected_edges[:,0], selected_edges[:,1]], np.r_[selected_edges[:,1], selected_edges[:,0]])), shape=(G,G)).tocsr()
    W = _normalize(W)
    movable = np.diff(W.indptr) > 0
    for _ in range(settings.smoothing_steps):
        mixed = (diags(np.where(movable, 1-settings.smoothing_strength, 1.)) @ A
                 + settings.smoothing_strength * W @ A).tocsr()
        rows = np.repeat(np.arange(G), np.diff(mixed.indptr))
        d, gap = support_distance(p, cov, rows, mixed.indices, settings.support_sigma)
        keep = (d <= radius) & (gap <= settings.max_support_gap)
        mixed.data[~keep] = 0
        # Normalize only updated rows: do not perturb preserved old beta by rounding.
        normalized = _normalize(mixed)
        A = diags((~movable).astype(float)) @ A + diags(movable.astype(float)) @ normalized
        A = A.tocsr(); A.eliminate_zeros(); A.sort_indices()
    host_N = csr_matrix((controls.interpolation_weights, controls.interpolation_indices, controls.interpolation_indptr),
                        shape=(len(hosts), len(controls.positions)))
    effective = (A[:,hosts] @ host_N).tocoo()
    local = host_N.tocoo()
    N = coo_matrix((np.r_[effective.data, local.data], (np.r_[effective.row, hosts[local.row]], np.r_[effective.col, local.col])),
                   shape=(G,len(controls.positions))).tocsr()
    N.sum_duplicates(); N.eliminate_zeros(); N.sort_indices()
    expected = point_status <= 1
    if not np.allclose(np.asarray(N.sum(1)).ravel(), expected.astype(float), rtol=1e-12, atol=1e-12):
        raise RuntimeError("Surface interpolation has missing or invalid support")
    return {**{"c_"+k:v for k,v in controls.as_dict().items()}, **attachment_inputs,
        "f_candidate_component_mask": ~main, "f_component_extent": extent, "f_core_mask": core,
        "f_component_status": component_status, "f_host_component": owner,
        "a_provisional_control_count": control_count, "a_legacy_preserved": preserved,
        "a_point_status": point_status, "a_point_host": point_host, "a_nearest_host_distance": nearest,
        "a_anchor_indptr": A.indptr.astype(np.int64), "a_anchor_indices": A.indices.astype(np.int64), "a_anchor_weights": A.data.astype(np.float64),
        "t_host_gaussian_index": hosts, "t_motion_component_index": point_host.copy(),
        "t_interpolation_indptr": N.indptr.astype(np.int64), "t_interpolation_indices": N.indices.astype(np.int64),
        "t_interpolation_weights": N.data.astype(np.float64)}


def diagnostics(arrays):
    status = arrays['a_point_status']; c = arrays['g_component_index']
    return {"host_components": int((~arrays['f_candidate_component_mask']).sum()),
            "host_gaussians": len(arrays['t_host_gaussian_index']), "host_controls": len(arrays['c_positions']),
            "preserved_legacy_components": int(arrays['a_legacy_preserved'].sum()),
            "point_status": {name: int((status == code).sum()) for code,name in STATUS.items()},
            "component_status": {name: int((arrays['f_component_status'] == code).sum()) for code,name in STATUS.items()},
            "unresolved_geometry_gaussians": int((status > 1).sum()),
            "candidate_nearest_host_distance_quantiles": np.quantile(arrays['a_nearest_host_distance'][status != 0], [.5,.9,.99]).tolist() if np.any(status!=0) else [],
            "per_mode_propagated_gaussians": (arrays['support_class'] == 3).sum(1).tolist() if 'support_class' in arrays else [],
            "per_mode_unsupported_host_components": [len(np.unique(c[(status==0)&(r==0)])) for r in arrays.get('support_class', [])]}
