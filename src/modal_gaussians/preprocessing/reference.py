"""Immutable video timing and pixel grid, independent of optical flow and FFT."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

from modal_gaussians.preprocessing.sequence import validate_image_mask_sequence, read_binary_mask
from modal_gaussians.common.cache import atomic_json, identity, sha256
from modal_gaussians.common.scene_store import resolve_path

FORMAT = "modal_gaussians.sequence_reference"


@dataclass
class SequenceReference:
    path: Path
    manifest: dict
    arrays: SimpleNamespace


def reference_identity(reference):
    return identity(reference.manifest)


def load_reference(path):
    root = resolve_path(path, strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("version") != 1:
        raise ValueError("Expected sequence_reference v1; prepare a new reference")
    names, index = manifest["frame_names"], manifest["reference_frame_index"]
    if (len(names) < 3 or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n or Path(n).name != n or n in (".", "..") for n in names)
            or type(index) is not int or not 0 <= index < len(names)
            or names[index] != manifest["reference_frame_name"]
            or not np.isfinite(manifest["fps_hz"]) or manifest["fps_hz"] <= 0):
        raise ValueError("Invalid sequence timing or reference")
    if sha256(root / "mask_union.npy") != manifest["mask_sha256"]:
        raise ValueError("Sequence mask checksum differs")
    mask = np.load(root / "mask_union.npy", allow_pickle=False)
    if mask.dtype != bool or list(mask.shape) != manifest["shape_hw"] or not mask.any():
        raise ValueError("Invalid sequence mask")
    return SequenceReference(root, manifest, SimpleNamespace(mask_union=mask))


def prepare_reference(*, images, masks, fps, reference_frame, output_dir, stabilize=False):
    """Publish a new reference; all derived pixels stay inside its directory."""
    output = resolve_path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    sequence = validate_image_mask_sequence(image_dir=resolve_path(images), mask_dir=resolve_path(masks),
        fps_hz=fps, reference_frame_name=reference_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        mask = np.zeros((sequence.height, sequence.width), bool)
        stable = None
        if stabilize:
            from modal_gaussians.preprocessing.stabilization import StabilizationSettings, stabilize_image_sequence, write_stabilized_sequence
            with tempfile.TemporaryDirectory(prefix="stabilization-", dir=temporary) as scratch:
                result = stabilize_image_sequence(sequence, StabilizationSettings(), Path(scratch))
                try:
                    stable = write_stabilized_sequence(temporary / "stabilized_sequence", sequence, result)
                    for i in range(sequence.frame_count):
                        mask |= np.asarray(result.masks[i])
                finally:
                    for array in (result.frames_gray, result.masks, result.valid_mask):
                        array.store.close()
        else:
            for path in sequence.mask_paths:
                mask |= read_binary_mask(path, mask.shape)
        np.save(temporary / "mask_union.npy", mask, allow_pickle=False)
        digest = hashlib.sha256()
        for image_path, mask_path in zip(sequence.image_paths, sequence.mask_paths):
            digest.update(image_path.stem.encode('utf-8'))
            digest.update(bytes.fromhex(sha256(image_path)))
            digest.update(bytes.fromhex(sha256(mask_path)))
        manifest = {"format": FORMAT, "version": 1,
            "inputs": {"sequence": {"image_directory": str(sequence.image_dir),
                                    "mask_directory": str(sequence.mask_dir)}},
            "frame_names": list(sequence.frame_names), "fps_hz": sequence.fps_hz,
            "reference_frame_name": sequence.reference_frame_name,
            "reference_frame_index": sequence.reference_frame_index,
            "shape_hw": [sequence.height, sequence.width], "input_pixels_sha256": digest.hexdigest(),
            "mask_sha256": sha256(temporary / "mask_union.npy"),
            "stabilized_sequence": stable}
        atomic_json(temporary / "manifest.json", manifest)
        load_reference(temporary)
        os.rename(temporary, output)
    except BaseException:
        if temporary.resolve().parent != output.parent.resolve() or not temporary.name.startswith(f".{output.name}."):
            raise RuntimeError("Temporary publication directory escaped its output parent")
        shutil.rmtree(temporary)
        raise
    return load_reference(output)
