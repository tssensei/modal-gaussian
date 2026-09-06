"""Shared immutable observation sources and the historical alpha input adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from modal_gaussians.static import load_static_scene
from modal_gaussians.topology import load_observation_topology
from modal_gaussians.measurements import load_gaussian_measurements

def load_observed_sources(
    *, scene_dir: str | Path, topology_dir: str | Path,
    measurements_dir: str | Path, graph_dir: str | Path,
) -> tuple[dict[str, Any], Any, Any, Any, Any]:
    """Validate shared scene/topology/measurement identities, without solver settings.

    The observed graph is still a required historical input for both pipelines.
    Its format validator is imported only when this source contract is used.
    """
    from modal_gaussians.motion.rigid.structure_graph import load_observed_structure_graph

    scene = load_static_scene(scene_dir, "cpu")
    topology = load_observation_topology(topology_dir)
    measurements = load_gaussian_measurements(measurements_dir)
    graph = load_observed_structure_graph(graph_dir)
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")
    scene_identity = scene_manifest["static_scene_identity"]
    foreground_identity = scene_manifest["foreground_identity"]
    topology_identity = topology.manifest["topology_identity"]
    for name, value in (
        ("topology static scene", topology.manifest["static_scene_identity"]),
        ("graph static scene", graph.manifest["static_scene_identity"]),
    ):
        if value != scene_identity:
            raise ValueError(f"{name} identity does not match the static scene")
    for name, value in (
        ("topology foreground", topology.manifest["foreground_identity"]),
        ("graph foreground", graph.manifest["foreground_identity"]),
    ):
        if value != foreground_identity:
            raise ValueError(f"{name} identity does not match the static foreground")
    if measurements.manifest["topology_identity"] != topology_identity:
        raise ValueError("Measurements do not belong to the supplied topology")
    if graph.manifest["topology_identity"] != topology_identity:
        raise ValueError("Observed graph does not belong to the supplied topology")

    topology_views = topology.manifest["views"]
    measurement_views = measurements.manifest["views"]
    graph_views = graph.manifest["views"]
    if not (
        len(topology_views) == len(measurement_views) == len(graph_views)
    ):
        raise ValueError("Solver inputs have different view counts")
    views: list[dict[str, Any]] = []
    for index, (topology_view, measurement_view, graph_view) in enumerate(
        zip(topology_views, measurement_views, graph_views)
    ):
        label = topology_view["label"]
        for source, view in (
            ("measurement", measurement_view),
            ("graph", graph_view),
        ):
            if view["index"] != index or view["label"] != label:
                raise ValueError(f"{source} view order does not match topology")
            if view["flow_identity"] != topology_view["flow_identity"]:
                raise ValueError(f"{source} flow identity for {label!r} differs")
            if view["shape_hw"] != topology_view["shape_hw"]:
                raise ValueError(f"{source} shape for {label!r} differs")
        if graph_view["camera_identity"] != topology_view["camera_identity"]:
            raise ValueError(f"Graph camera identity for {label!r} differs")
        views.append(
            {
                "index": index,
                "label": label,
                "shape_hw": list(topology_view["shape_hw"]),
                "camera_identity": topology_view["camera_identity"],
                "flow_identity": topology_view["flow_identity"],
                "sample_count": int(measurement_view["sample_count"]),
            }
        )
    modes = [
        {
            "mode_slot": int(mode["mode_slot"]),
            "candidate_index": int(mode["candidate_index"]),
            "frequency_hz": float(mode["frequency_hz"]),
        }
        for mode in measurements.manifest["modes"]
    ]
    source = {
        "static_scene": str(scene_dir),
        "static_scene_identity": scene_identity,
        "foreground_identity": foreground_identity,
        "topology": str(topology.path),
        "topology_identity": topology_identity,
        "measurements": str(measurements.path),
        "gaussian_measurements_identity": measurements.manifest[
            "gaussian_measurements_identity"
        ],
        "observed_structure_graph": str(graph.path),
        "observed_structure_graph_identity": graph.manifest[
            "observed_structure_graph_identity"
        ],
        "modes": modes,
        "views": views,
    }
    return source, scene, topology, measurements, graph


@dataclass(frozen=True)
class FixedAlignmentArtifact:
    """Only the fixed complex gains and identifiability exposed to neural fitting."""
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


def load_fixed_alignment(path: str | Path) -> FixedAlignmentArtifact:
    """Read the existing rigid artifact through its strict compatibility loader.

    The persisted source identity is preserved. Rigid motion/trust arrays are
    not exposed as neural initialization or training data.
    """
    from modal_gaussians.motion.rigid.rigid import load_rigid_modes

    artifact = load_rigid_modes(path)
    arrays = {key: artifact.arrays[key] for key in ("alphas", "alpha_identifiable_mask")}
    return FixedAlignmentArtifact(artifact.path, artifact.manifest, arrays)
