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


def _pixels_digest(names, images, masks):
    digest = hashlib.sha256()
    for name, image, mask in zip(names, images, masks):
        digest.update(name.encode('utf-8'))
        digest.update(bytes.fromhex(sha256(image)))
        digest.update(bytes.fromhex(sha256(mask)))
    return digest.hexdigest()


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
    if manifest.get("format") != FORMAT or manifest.get("version") != 2:
        raise ValueError("Expected sequence_reference v2; prepare a new reference")
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
    if manifest.get('capture') not in ('tripod', 'stabilized'):
        raise ValueError('Reference must declare tripod or stabilized capture')
    valid = np.load(root/'valid_mask.npy', allow_pickle=False)
    if (sha256(root/'valid_mask.npy') != manifest['valid_sha256'] or valid.dtype != bool
            or valid.shape != mask.shape or not valid.any() or np.any(mask & ~valid)):
        raise ValueError('Invalid reference support')
    stable = manifest['stabilized_sequence']
    if (stable is None) != (manifest['capture'] == 'tripod'):
        raise ValueError('Reference capture/derived pixels disagree')
    if stable is not None:
        from modal_gaussians.geometry.scene import Camera
        Camera.from_manifest_record(stable['target_camera'])
        if stable.get('method') != 'background_pnp_reference_depth_v1' or stable.get('version') != 2:
            raise ValueError('Unsupported stabilization method')
        for name, digest in stable['files'].items():
            path = root/'stabilized_sequence'/name
            if not path.resolve().is_relative_to(root/'stabilized_sequence') or sha256(path) != digest:
                raise ValueError('Stabilized artifact checksum differs')
    raw = manifest['inputs']['sequence']
    images, masks = resolve_path(raw['image_directory']), resolve_path(raw['mask_directory'])
    if _pixels_digest(names, [images/(n+'.png') for n in names], [masks/(n+'.png') for n in names]) != manifest['input_pixels_sha256']:
        raise ValueError('Original sequence pixels changed')
    return SequenceReference(root, manifest, SimpleNamespace(mask_union=mask, valid_mask=valid))


def prepare_reference(*, images, masks, fps, reference_frame, output_dir, tripod=False,
                      scene_dir=None, label=None, settings=None, device='cuda'):
    """Publish a new reference; all derived pixels stay inside its directory."""
    if type(tripod) is not bool or (not tripod and (scene_dir is None or not label)):
        raise ValueError('Stabilization is default: provide --scene and --view, or explicitly declare --tripod')
    if tripod and (scene_dir is not None or label is not None or settings is not None):
        raise ValueError('Tripod capture does not take stabilization scene/view/settings')
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
        valid = np.ones(mask.shape, bool)
        if not tripod:
            from modal_gaussians.preprocessing.stabilization import stabilize_image_sequence
            stable, mask, valid = stabilize_image_sequence(sequence, temporary/'stabilized_sequence',
                scene_dir=scene_dir, label=label, settings=settings, device=device)
        else:
            for path in sequence.mask_paths:
                mask |= read_binary_mask(path, mask.shape)
        np.save(temporary / "mask_union.npy", mask, allow_pickle=False)
        np.save(temporary / "valid_mask.npy", valid, allow_pickle=False)
        digest = _pixels_digest(sequence.frame_names, sequence.image_paths, sequence.mask_paths)
        manifest = {"format": FORMAT, "version": 2, 'capture': 'tripod' if tripod else 'stabilized',
            "inputs": {"sequence": {"image_directory": str(sequence.image_dir),
                                    "mask_directory": str(sequence.mask_dir)}},
            "frame_names": list(sequence.frame_names), "fps_hz": sequence.fps_hz,
            "reference_frame_name": sequence.reference_frame_name,
            "reference_frame_index": sequence.reference_frame_index,
            "shape_hw": [sequence.height, sequence.width], "input_pixels_sha256": digest,
            "mask_sha256": sha256(temporary / "mask_union.npy"),
            "valid_sha256": sha256(temporary / "valid_mask.npy"),
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
