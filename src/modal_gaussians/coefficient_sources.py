"""SEA-RAFT inputs for coefficient preparation; historical identities stay intact."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from modal_gaussians.flow.artifact import FlowAnalysisArtifact, flow_artifact_identity
from modal_gaussians.flow.storage import open_array
from modal_gaussians.iteration_cache import identity, sha256
from modal_gaussians.scene_store import resolve_path
from modal_gaussians.spectrum_cache import _source


@dataclass(frozen=True)
class CoordinateFlow:
    path: Path
    manifest: dict
    arrays: SimpleNamespace
    identity: str
    reference_identity: str
    image_directory: Path


def coordinate_flow_identity(flow):
    return flow.identity if isinstance(flow, CoordinateFlow) else flow_artifact_identity(flow)


def load_coordinate_flow(path):
    """Read headers/mask and open lazy SEA flow; never load old Farneback arrays."""
    root, source = _source(path)
    reference = resolve_path(source["stabilization_source"], strict=True)
    metadata = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "modal_gaussians.flow_analysis" or metadata.get("version") not in (6, 7):
        raise ValueError("SEA-RAFT geometry reference must contain historical flow metadata")
    for new, old in (("frames", "frame_names"), ("fps_hz", "fps_hz"),
                     ("reference_frame_index", "reference_frame_index"),
                     ("reference_frame_name", "reference_frame_name")):
        if source[new] != metadata[old]:
            raise ValueError(f"SEA-RAFT/reference {new} differs")
    if source["flow_shape"] != metadata["arrays"]["flow"]["shape"]:
        raise ValueError("SEA-RAFT/reference flow shapes differ")
    names = source["frames"]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or Path(n).name != n for n in names):
        raise ValueError("SEA-RAFT frame names must be unique basenames")
    hashes = {name: metadata["arrays"][name]["sha256"] for name in ("flow", "mask_union", "spectrum")}
    reference_identity = flow_artifact_identity(FlowAnalysisArtifact(reference, metadata, None, hashes))
    mask_path = (reference / metadata["arrays"]["mask_union"]["file"]).resolve(strict=True)
    if not mask_path.is_relative_to(reference) or sha256(mask_path) != hashes["mask_union"]:
        raise ValueError("Geometry reference mask binding differs")
    mask = np.load(mask_path, allow_pickle=False)
    if mask.dtype != np.bool_ or list(mask.shape) != source["flow_shape"][1:3]:
        raise ValueError("Geometry reference mask shape/dtype differs")
    stable = metadata.get("stabilized_sequence")
    if stable is not None:
        stable_root = (reference / stable["path"]).resolve(strict=True)
        if not stable_root.is_relative_to(reference):
            raise ValueError("Stabilized RGB directory escapes its reference")
        saved = json.loads((stable_root / "manifest.json").read_text(encoding="utf-8"))
        if (saved["frames"] != names or saved["fps_hz"] != source["fps_hz"]
                or saved["reference_frame"] != source["reference_frame_name"]):
            raise ValueError("Stabilized RGB timing differs from SEA-RAFT")
        images = stable_root / "images"
    else:
        images = resolve_path(metadata["inputs"]["sequence"]["image_directory"], strict=True)
    recorded_images = source.get("inference_images") or source.get("stabilized_images") or source["images"]
    if resolve_path(recorded_images, strict=True) != images or not images.is_dir():
        raise ValueError("SEA-RAFT images differ from the recorded camera/reference geometry")
    flow = open_array(root / source["flow_file"])
    if list(flow.shape) != source["flow_shape"] or flow.dtype != np.float32:
        raise ValueError("SEA-RAFT array header differs from manifest")
    # Like the shared FFT cache, bind immutable SEA data by its original source manifest.
    binding = identity({"source": source, "reference_identity": reference_identity,
                        "mask_sha256": hashes["mask_union"], "adapter": "coefficient_sea_v1"})
    manifest = {**source, "frame_names": names}
    return CoordinateFlow(root, manifest, SimpleNamespace(flow=flow, mask_union=mask),
                          binding, reference_identity, images)
