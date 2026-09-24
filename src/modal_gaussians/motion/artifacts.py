"""Load current baked displacement and angular fields."""
import json
import numpy as np
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.motion import training as nm
from modal_gaussians.motion.component_field import VERSION, METHOD, support_roles


def _validate_arrays(arrays, manifest, *, check_field_geometry=True):
    K, G, V = (manifest['counts'][key] for key in ('modes', 'foreground_gaussians', 'views'))
    C = manifest['counts']['controls']
    if K != 1 or G < 1 or V < 1 or C < 1:
        raise ValueError('Current models contain one frequency and nonempty geometry/views')
    for key, shape in {'phi': (K,G,3), 'rotation': (K,G,3),
                       'control_displacement': (K,C,3), 'alphas': (K,V)}.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.complex64 or not np.isfinite(value).all():
            raise ValueError(f'Invalid complex field {key}')
    if arrays['g_points'].shape != (G,3) or arrays['c_positions'].shape != (C,3):
        raise ValueError('Geometry/control counts differ')
    if arrays['observation_view_mask'].shape != (K,G,V) or arrays['observation_view_mask'].dtype != bool:
        raise ValueError('Observation view mask differs')
    roles, supported = support_roles(arrays, arrays['observation_view_mask'])
    if not np.array_equal(roles, arrays['support_class']) or np.any(arrays['phi'][~supported] != 0):
        raise ValueError('Support roles or unresolved fields differ')
    if check_field_geometry:
        nm._field_geometry(arrays, 0, validate=True)


def load_neural_completed_modes(path, *, validate=False):
    root = resolve_path(path, strict=True)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('format') != nm.COMPLETED_MODES_FORMAT or manifest.get('version') != VERSION
            or manifest.get('completion_method') != METHOD):
        raise ValueError('Unsupported model; rebuild using the current pipeline')
    nm.NeuralModesConfig.from_dict(manifest['config'])
    if manifest.get('arrays_file') != nm.ARRAYS_FILENAME or manifest.get('networks_file') != nm.MODELS_FILENAME:
        raise ValueError('Model filenames differ')
    if nm._identity(nm._artifact_identity_payload(manifest)) != manifest.get('completed_modes_identity'):
        raise ValueError('Model identity differs')
    with np.load(root / nm.ARRAYS_FILENAME, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if manifest.get('arrays') != {name: {'dtype': a.dtype.name, 'shape': list(a.shape)} for name, a in arrays.items()}:
        raise ValueError('Model array inventory differs')
    _validate_arrays(arrays, manifest, check_field_geometry=validate)
    if validate:
        if (nm._sha256(root / nm.ARRAYS_FILENAME) != manifest['arrays_file_sha256']
                or nm._sha256(root / nm.MODELS_FILENAME) != manifest['networks_sha256']
                or nm._arrays_identity(arrays) != manifest['arrays_identity']
                or nm._source_identity(manifest) != manifest['source_identity']):
            raise ValueError('Model checksums or source metadata differ')
        for name, array in arrays.items():
            if array.dtype.kind in 'fc' and not np.isfinite(array).all():
                raise ValueError(f'Non-finite model array: {name}')
    return nm.NeuralModesArtifact(root, manifest, arrays, arrays['rotation'], arrays['control_displacement'])
