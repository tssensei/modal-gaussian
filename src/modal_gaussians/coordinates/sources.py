"""Bind saved SEA-RAFT flow to the exact reference pixel grid for coefficient fitting."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from modal_gaussians.preprocessing.reference import reference_identity
from modal_gaussians.flow.storage import open_array
from modal_gaussians.flow.reference_selection import motion_reference
from modal_gaussians.common.cache import identity
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.spectrum.cache import _source


@dataclass(frozen=True)
class CoordinateFlow:
    path: Path
    manifest: dict
    arrays: SimpleNamespace
    identity: str
    reference_identity: str
    image_directory: Path


def coordinate_flow_identity(flow):
    return flow.identity


def load_coordinate_flow(path):
    from modal_gaussians.preprocessing.reference import load_reference
    from modal_gaussians.flow.reference_selection import sequence_metadata
    root, source = _source(path)
    reference = load_reference(source['stabilization_source'])
    from modal_gaussians.common.cache import sha256
    if source.get('sequence_reference_identity') != reference_identity(reference):
        raise ValueError('Flow/reference identity differs')
    support = np.load(root/'valid_mask.npy', allow_pickle=False)
    if (sha256(root/'valid_mask.npy') != source['valid_mask_sha256'] or support.dtype != bool
            or support.shape != reference.arrays.valid_mask.shape or np.any(support & ~reference.arrays.valid_mask)):
        raise ValueError('Invalid flow support')
    metadata = reference.manifest
    for new, old in (('frames', 'frame_names'), ('fps_hz', 'fps_hz')):
        if source[new] != metadata[old]:
            raise ValueError(f'SEA-RAFT/reference {new} differs')
    motion_reference(source, metadata)
    if source['flow_shape'] != [len(metadata['frame_names']), *metadata['shape_hw'], 2]:
        raise ValueError('SEA-RAFT/reference shape differs')
    images = sequence_metadata(reference.path)['image_dir']
    if resolve_path(source['inference_images'], strict=True) != images:
        raise ValueError('SEA-RAFT images differ from the reference pixel grid')
    flow = open_array(root / source['flow_file'])
    if list(flow.shape) != source['flow_shape'] or flow.dtype != np.float32:
        flow.store.close()
        raise ValueError('SEA-RAFT array header differs from manifest')
    ref_id = reference_identity(reference)
    binding = identity({'source': source, 'reference_identity': ref_id, 'adapter': 'coefficient_sea_v2'})
    return CoordinateFlow(root, {**source, 'frame_names': source['frames']},
        SimpleNamespace(flow=flow, mask_union=reference.arrays.mask_union & support, valid_mask=support), binding, ref_id, images)
