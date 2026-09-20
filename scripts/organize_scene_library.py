"""One-time import of the reviewed 2026-09-19 retention plan; never deletes sources.

Run with the existing metadata inventory and cleanup analysis JSON files. Large
immutable payloads use same-volume hard links; mutable work and metadata are copied.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import time


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def transfer(source, target, link):
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    if target.exists():
        if target.stat().st_size != before.st_size:
            raise FileExistsError(f'Destination differs: {target}')
        if link and not os.path.samefile(source, target):
            raise FileExistsError(f'Destination is not this payload: {target}')
        if not link and target.stat().st_mtime_ns != before.st_mtime_ns:
            raise FileExistsError(f'Source or destination changed: {target}')
        return
    if link:
        os.link(source, target)
    else:
        temporary = target.with_name(target.name + '.importing')
        shutil.copy2(source, temporary)
        if source.stat().st_mtime_ns != before.st_mtime_ns:
            raise RuntimeError(f'Source changed during import: {source}')
        temporary.replace(target)


def publish_catalogs(destination, registry):
    """Index saved results and cache contracts using JSON metadata only."""
    def relocated(value):
        path = Path(value).resolve()
        for old, new in registry['locations']:
            if path.is_relative_to(Path(old).resolve()):
                return Path(new) / path.relative_to(Path(old).resolve())
        raise ValueError(f'Result has no imported location: {path}')

    for scene, record in registry['scenes'].items():
        modes = []
        experiment = Path(record['assets']['baseline']).parent
        inputs = [(experiment, 'accepted_baseline')]
        if scene == 'bush':
            batch = destination / record['assets']['batch40']
            state = json.loads((batch / 'batch_state.json').read_text(encoding='utf-8'))
            inputs += [(relocated(job['result_dir']), 'completed_uniform60')
                       for job in state['jobs'] if job['state'] == 'complete']
            record['batch_status'] = {'state': state['status'], 'completed': state['completed'],
                                     'total': len(state['jobs'])}
        for experiment, status in inputs:
            outputs = json.loads((destination / experiment / 'outputs.json').read_text(encoding='utf-8'))
            model = relocated(outputs['completed_modes'])
            manifest = json.loads((destination / model / 'manifest.json').read_text(encoding='utf-8'))
            iteration = json.loads((destination / experiment / 'iteration.json').read_text(encoding='utf-8'))
            for mode in manifest['modes']:
                modes.append({'frequency_hz': mode['frequency_hz'], 'status': status,
                              'experiment': experiment.as_posix(), 'completed_modes': model.as_posix(),
                              'completed_modes_identity': manifest['completed_modes_identity'],
                              'prepared': relocated(outputs['prepared']).as_posix(),
                              'graph': relocated(iteration['external_geometry_graph']['path']).as_posix(),
                              'checkpoint': f'{scene}/checkpoints/{model.name}'})
        record['modes'] = sorted(modes, key=lambda row: row['frequency_hz'])
        contracts = []
        for manifest_path in sorted((destination / scene / 'cache').glob('*/*/manifest.json')):
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            contracts.append({'path': manifest_path.parent.relative_to(destination).as_posix(),
                              'contract': manifest.get('contract')})
        catalog_path = destination / scene / 'catalog.json'
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
        write_json(catalog_path, {**catalog, **record, 'cache_entries': contracts})
        write_json(destination / scene / 'results/index.json', record['modes'])


def organize(inventory, analysis, destination):
    snapshot = json.loads(inventory.read_text(encoding='utf-8'))
    reviewed = json.loads(analysis.read_text(encoding='utf-8'))
    source = Path(reviewed['root']).resolve(strict=True)
    destination = destination.resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Library must be outside the old outputs tree')
    if (destination / 'registry.json').exists():
        raise FileExistsError('Library is already published; do not re-import')
    mappings = {
        'bush_neural_soft_rigidity003_rotation0_0744_001': 'bush/experiments/baseline_0744',
        'corn_neural_soft_rigidity003_rotation0_0225_001': 'corn/experiments/baseline_0225',
        'bush_uniform60_modes_001': 'bush/experiments/uniform60/initial',
        'bush_uniform60_modes_001/modal_images': 'bush/modal_images/uniform60',
        'bush_uniform60_modes_parallel_001': 'bush/experiments/uniform60/parent_2000',
        'bush_uniform60_modes_5000_002': 'bush/experiments/uniform60/current_5000',
        'bush_spectrum_001': 'bush/spectrum/main', 'corn_spectrum_001': 'corn/spectrum/main',
        'bush_spectrum_uniform60_001': 'bush/selections/uniform60',
        'bush_spectrum_greedy60_001': 'bush/selections/greedy60',
        'bush_neural_modal_similarity_0744_001/prepared': 'bush/geometry/prepared',
        'bush_neural_dense_controls_001/static_scene': 'bush/geometry/static',
        'corn_subject_selection_001/prepared': 'corn/geometry/prepared',
        'corn_subject_selection_001/static_scene': 'corn/geometry/static',
        'bush_neural_soft_fixed_controls_0744_001/graph': 'bush/graphs/baseline_0744',
        'corn_subject_selection_001/graph_modal_soft_002': 'corn/graphs/baseline_0225',
        'models': '_shared/tools/models', 'third_party': '_shared/tools/third_party',
    }
    for scene, count, frequency in [('bush', 3, '0744'), ('corn', 2, '0225')]:
        for index in range(1, count + 1):
            mappings[f'{scene}{index}_sea_raft_{frequency}_001'] = f'{scene}/flow/view{index}'
            old = (f'bush_neural_001/flow_stabilized/view{index}' if scene == 'bush'
                   else f'corn_local_001/flow/view{index}')
            mappings[old] = f'{scene}/references/view{index}'
    for root in reviewed['protected_minimal']:
        if any(root == key or root.startswith(key + '/') for key in mappings):
            continue
        if root.startswith('_cache/'):
            reason = reviewed['protected'][root]
            owners = [scene for scene in ('bush', 'corn') if scene in reason.lower()]
            if len(owners) != 1:
                raise ValueError(f'Cache has ambiguous scene: {root}: {reason}')
            _, kind, key = root.split('/')
            category = {'trained_modes': 'results/models', 'neural_work': 'checkpoints'}.get(kind, 'cache/' + kind)
            mappings[root] = f'{owners[0]}/{category}/{key}'
        else:
            scene = next((name for name in ('bush', 'corn') if root.startswith(name)), None)
            mappings[root] = f'{scene}/geometry/ancestors/{root}' if scene else f'_shared/history/{root}'
    # Reports are part of the audit trail, not model dependencies.
    for path in source.glob('cleanup_*20260919*'):
        mappings[path.name] = '_shared/history/' + path.name
    protected = set(reviewed['protected_minimal']) | {p.name for p in source.glob('cleanup_*20260919*')}

    def match(path, keys):
        current = Path(path)
        for parent in (current, *current.parents):
            if parent.as_posix() in keys:
                return parent.as_posix()
        return None

    # Scan filenames only. New files inside retained roots are preserved too;
    # unreviewed files elsewhere are reported, never labelled safe to delete.
    files = []
    unknown = []
    candidate_roots = {row['path'] for row in reviewed['candidates']}
    for folder, dirs, names in os.walk(source):
        for name in names:
            path = Path(folder) / name
            relative = path.relative_to(source).as_posix()
            if match(relative, protected) is not None:
                prefix = match(relative, mappings)
                if prefix is None:
                    raise ValueError(f'No destination for {relative}')
                target = Path(mappings[prefix]) / Path(relative).relative_to(prefix)
                if path.is_symlink() or not path.resolve().is_relative_to(source):
                    raise ValueError(f'Unexpected link outside input tree: {path}')
                size = path.stat().st_size
                mutable = '/neural_work/' in '/' + relative or target.parts[1:2] == ('checkpoints',)
                payload = '.zarr/' in relative or path.suffix.lower() in {'.npy', '.npz', '.pt', '.pth', '.png', '.jpg', '.jpeg', '.bin'}
                files.append((relative, target.as_posix(), size, payload and not mutable))
            elif relative not in snapshot['files'] or match(relative, candidate_roots) is None:
                unknown.append(relative)
    write_json(destination / 'import_plan.json', {'source': str(source), 'files': files, 'unreviewed': unknown})
    started = time.monotonic()
    for index, (old, new, size, link) in enumerate(files, 1):
        transfer(source / old, destination / new, link)
        if index % 1000 == 0:
            print(f'Imported {index}/{len(files)} files, elapsed {time.monotonic()-started:.0f}s', flush=True)
    locations = [[str(source / old), new] for old, new in sorted(mappings.items(), key=lambda item: -len(item[0]))]
    scenes = {}
    for scene, freq in [('bush', '0744'), ('corn', '0225')]:
        assets = {key: f'{scene}/{value}' for key, value in {
            'prepared': 'geometry/prepared', 'static': 'geometry/static', 'spectrum': 'spectrum/main',
            'graph': f'graphs/baseline_{freq}', 'baseline': f'experiments/baseline_{freq}/experiment/preview',
            'experiments': 'experiments', 'cache': 'cache',
        }.items()}
        for index in range(1, (4 if scene == 'bush' else 3)):
            assets[f'flow{index}'] = f'{scene}/flow/view{index}'
            assets[f'reference{index}'] = f'{scene}/references/view{index}'
        if scene == 'bush':
            assets.update({key: f'bush/{value}' for key, value in {
                'uniform60': 'selections/uniform60', 'greedy60': 'selections/greedy60',
                'modal_images': 'modal_images/uniform60', 'batch40': 'experiments/uniform60/current_5000',
            }.items()})
        geometry = [new for old, new in mappings.items() if old.startswith('_cache/geometry/') and new.startswith(scene + '/')]
        if len(geometry) != 1:
            raise ValueError(f'Expected one candidate graph for {scene}')
        assets['candidate_graph'] = geometry[0]
        control_geometry = [new for new in mappings.values() if new.startswith(scene + '/cache/control_geometry/')]
        if len(control_geometry) == 1:
            assets['control_geometry'] = control_geometry[0]
        identities = sorted({data['source']['static_scene_identity'] for name, data in snapshot['metadata'].items()
                             if isinstance(data, dict) and name.startswith(scene) and isinstance(data.get('source'), dict)
                             and 'static_scene_identity' in data['source']})
        scenes[scene] = {'cache': scene + '/cache', 'assets': assets, 'static_scene_identities': identities}
    record = {'format': 'modal_gaussians.scene_library', 'version': 1, 'locations': locations,
              'local_routes': [[f'{scene}/cache/{kind}', f'{scene}/{target}'] for scene in scenes
                               for kind, target in [('trained_modes', 'results/models'), ('neural_work', 'checkpoints')]],
              'scenes': scenes}
    for scene in scenes:
        scene_files = [row for row in files if row[1].startswith(scene + '/')]
        write_json(destination / scene / 'catalog.json', {**scenes[scene], 'files': len(scene_files),
                   'logical_bytes': sum(row[2] for row in scene_files)})
    with (destination / 'old_outputs_classification.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['old_path', 'category', 'new_path', 'bytes'])
        for old, new, size, _ in files:
            writer.writerow([old, 'relocated_copy_or_link', new, size])
        for row in reviewed['candidates']:
            writer.writerow([row['path'], 'previously_reviewed_legacy', '', row['bytes']])
        for path in unknown:
            writer.writerow([path, 'unreviewed_do_not_delete', '', ''])
    write_json(destination / 'import_summary.json', {'files': len(files), 'logical_bytes': sum(row[2] for row in files),
               'hardlinked_bytes': sum(row[2] for row in files if row[3]),
               'copied_bytes': sum(row[2] for row in files if not row[3]),
               'seconds': time.monotonic()-started, 'unreviewed_files': unknown,
               'source_deletions': 0, 'verification': 'file size/mtime or hardlink identity only; no numerical validation'})
    publish_catalogs(destination, record)
    write_json(destination / 'registry.json', record)  # Publish only after the whole import succeeds.
    print(json.dumps(json.loads((destination / 'import_summary.json').read_text()), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    organize(args.inventory, args.analysis, args.destination)
