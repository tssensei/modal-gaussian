"""Explicit v16 component-field import; no old training or general legacy reader."""
import argparse
import json
from pathlib import Path
import numpy as np

from modal_gaussians.common.cache import identity, sha256
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.motion import training
from modal_gaussians.coordinates.reference import publish_reference


def load_v16(path, *, validate=True):
    root = resolve_path(path, strict=True)
    m = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (m.get('format') != 'modal_gaussians.completed_modes' or m.get('version') != 16
            or m.get('config', {}).get('training_fragment_config', {}).get('strategy') != 'component_field'
            or m.get('arrays_file') != 'completed_modes.npz' or m.get('networks_file') != 'networks.pt'):
        raise ValueError('This import accepts only saved v16 component fields')
    if identity(training._artifact_identity_payload(m)) != m['completed_modes_identity']:
        raise ValueError('v16 model identity differs')
    if (sha256(root/m['arrays_file']) != m['arrays_file_sha256']
            or sha256(root/m['networks_file']) != m['networks_sha256']):
        raise ValueError('v16 array/network checksum differs')
    source = {k:m[k] for k in ('static_scene_identity','foreground_identity','topology_identity',
        'gaussian_measurements_identity','observed_structure_graph_identity','alignment_identity',
        'complex_2d_modes_identity','modes','views')}
    source['modes'] = m.get('source_modes',m['modes'])
    if 'selected_modal_supervision' in m:
        source['selected_modal_supervision'] = m['selected_modal_supervision']
    if source != m['source_identity']:
        raise ValueError('v16 source identity differs')
    with np.load(root/m['arrays_file'],allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    if (training._arrays_identity(arrays) != m['arrays_identity']
            or m['arrays'] != {k:dict(dtype=v.dtype.name,shape=list(v.shape)) for k,v in arrays.items()}
            or any(not np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fc')):
        raise ValueError('v16 array inventory/identity differs')
    if any(k.startswith('r_') for k in arrays):
        raise ValueError('Residual-field imports are not supported')
    for mode in range(len(m['modes'])):
        training._field_geometry(arrays, mode, validate=True)
    return training.NeuralModesArtifact(root,m,arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('scene','modes','output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    print(publish_reference(scene_dir=args.scene, completed_modes_dir=args.modes,
                            output_dir=args.output, source_loader=load_v16))


if __name__ == '__main__':
    main()
