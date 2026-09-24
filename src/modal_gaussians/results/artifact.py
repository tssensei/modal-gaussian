"""Identity-bound materialization of one complete modal Gaussian result."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
from typing import Any, Mapping, Sequence

from modal_gaussians import __version__
from modal_gaussians.coordinates.direct import DIRECT_COORDINATES_FORMAT, DirectModalCoordinatesArtifact, load_direct_modal_coordinates
from modal_gaussians.motion.common.completed_modes import CompletedModesArtifact, load_completed_modes

from modal_gaussians.coordinates.rgb import RGB_COORDINATES_FORMAT, RGBModalCoordinatesArtifact, load_rgb_modal_coordinates
from modal_gaussians.coordinates.design import RenderedModalDesignArtifact, load_rendered_modal_design
from modal_gaussians.geometry.scene import ForegroundBackgroundScene, load_static_scene


MODAL_RESULT_FORMAT = "modal_gaussians.modal_result"
MODAL_RESULT_VERSION = 3
MANIFEST_FILENAME = "manifest.json"

STORAGE_CONVENTION = {
    "layout": "linked_immutable_artifacts",
    "large_arrays_copied": False,
    "source_paths": "absolute",
    "validation": "reload_and_verify_every_source_identity",
}

DEFORMATION_CONVENTION = {
    "foreground": "means(t)=means_static+sum_k real(q_view(t,k)*phi(k))",
    "background": "static",
    "phi": "completed_modes.phi[K,G_fg,3]_complex64",
    "q": "coordinates[sum(T_view),K]_complex64",
    "coordinate_index": "views[view].frame_offset+local_frame_index",
    "flow_comparison": "q(t)-q(reference_frame_index)",
    "gaussian_index": "static_scene_foreground_order",
    "units": "normalized_scene_coordinates",
}


CoordinateArtifact = (
    DirectModalCoordinatesArtifact
    | RGBModalCoordinatesArtifact
)
COORDINATE_IDENTITY_NAMES = {
    "direct": "direct_coordinates_identity",
    "rgb": "rgb_coordinates_identity",
    "refined_rgb": "refined_coordinates_identity",
    "sweep_rgb": "sweep_coordinates_identity",
}


@dataclass(frozen=True)
class ModalResultArtifact:
    """Expose one validated result and its already verified linked sources."""

    path: Path
    manifest: dict[str, Any]
    scene: ForegroundBackgroundScene
    completed_modes: CompletedModesArtifact
    coordinates: CoordinateArtifact
    rendered_design: RenderedModalDesignArtifact | None
    direct_coordinates: DirectModalCoordinatesArtifact | None
    coordinate_artifacts: tuple = ()


def _canonical_json(value: Any) -> bytes:
    """Encode one path-independent identity payload deterministically."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select the immutable scientific bindings and playback convention."""

    return {
        "format": MODAL_RESULT_FORMAT,
        "version": MODAL_RESULT_VERSION,
        "storage": manifest["storage"],
        "static_scene_identity": manifest["static_scene_identity"],
        "foreground_identity": manifest["foreground_identity"],
        "background_identity": manifest["background_identity"],
        "completed_modes_identity": manifest["completed_modes_identity"],
        "rendered_design_identity": manifest["rendered_design_identity"],
        "coordinate_source": manifest["coordinate_source"],
        "coordinate_sources": manifest["coordinate_sources"],
        "modes": manifest["modes"],
        "views": manifest["views"],
        "counts": manifest["counts"],
        "deformation": manifest["deformation"],
        "quality_gate": manifest["quality_gate"],
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one JSON object from an artifact directory."""

    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest does not exist: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact manifest must be a JSON object: {manifest_path}")
    return value


def _source_path(sources: Mapping[str, Any], name: str) -> Path:
    """Resolve one required absolute linked-source path from a result manifest."""

    record = sources.get(name)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"Modal result source {name!r} is invalid")
    path = resolve_path(record["path"], strict=True)
    if not path.is_dir():
        raise FileNotFoundError(f"Modal result source is not a directory: {path}")
    return path


def _load_coordinate_artifact(path: Path) -> tuple[str, CoordinateArtifact]:
    """Auto-detect and strictly load direct or RGB coordinates."""

    path = resolve_path(path, strict=True)
    artifact_format = _read_manifest(path).get("format")
    if artifact_format == DIRECT_COORDINATES_FORMAT:
        return "direct", load_direct_modal_coordinates(path)
    if artifact_format == RGB_COORDINATES_FORMAT:
        return "rgb", load_rgb_modal_coordinates(path)
    from modal_gaussians.coordinates.sweep import SWEEP_FORMAT, load_sweep_coordinates
    if artifact_format == SWEEP_FORMAT:
        return "sweep_rgb", load_sweep_coordinates(path)
    from modal_gaussians.coordinates.refinement_artifacts import COORDINATES_FORMAT, load_refined_coordinates
    if artifact_format == COORDINATES_FORMAT:
        return "refined_rgb", load_refined_coordinates(path)
    raise ValueError(
        "Result coordinates must be a direct- or RGB-coordinate artifact"
    )


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    """Raise one focused error when an identity or ordered contract differs."""

    if actual != expected:
        raise ValueError(f"Modal result {name} differs across linked artifacts")


def _select_views(views: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> list[dict[str, Any]]:
    """Copy selected view records into a local index/frame layout; never mutate sources."""
    by_label = {view["label"]: view for view in views}
    if (len(by_label) != len(views) or not labels or len(set(labels)) != len(labels)
            or any(label not in by_label for label in labels)):
        raise ValueError("Selected view labels must be unique and present in the source")
    selected, offset = [], 0
    for index, label in enumerate(labels):
        view = {**by_label[label], "index": index}
        if "frame_offset" in view:
            view["frame_offset"] = offset
            offset += view["frame_count"]
        selected.append(view)
    return selected


def _coordinate_view_records(
    coordinate_views: Sequence[Mapping[str, Any]],
    design_views: Sequence[Mapping[str, Any]],
    completed_views: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the view chain and retain only runtime frame/camera metadata."""

    if not (
        len(coordinate_views) == len(design_views) == len(completed_views)
    ):
        raise ValueError("Modal result linked artifacts have different view counts")
    records: list[dict[str, Any]] = []
    frame_offset = 0
    for index, (coordinate, design, completed) in enumerate(
        zip(coordinate_views, design_views, completed_views)
    ):
        for field in ("index", "label", "flow_identity", "shape_hw"):
            _require_equal(
                f"view {index} {field}", design.get(field), completed.get(field)
            )
            _require_equal(
                f"coordinate view {index} {field}",
                coordinate.get(field),
                design.get(field),
            )
        for coordinate_name, design_name in (
            ("frame_count", "frame_count"),
            ("fps_hz", "fps_hz"),
            ("reference_frame_name", "flow_reference_frame_name"),
            ("reference_frame_index", "flow_reference_frame_index"),
            ("sample_count", "sample_count"),
        ):
            _require_equal(
                f"coordinate view {index} {coordinate_name}",
                coordinate.get(coordinate_name),
                design.get(design_name),
            )
        _require_equal(
            f"coordinate view {index} frame offset",
            coordinate.get("frame_offset"),
            frame_offset,
        )
        frame_count = int(coordinate["frame_count"])
        record = {
            "index": index,
            "label": coordinate["label"],
            "camera_name": design["camera_name"],
            "camera_identity": design["camera_identity"],
            "flow_identity": coordinate["flow_identity"],
            "shape_hw": list(coordinate["shape_hw"]),
            "fps_hz": float(coordinate["fps_hz"]),
            "frame_offset": frame_offset,
            "frame_count": frame_count,
            "frame_names": list(coordinate["frame_names"]),
            "reference_frame_name": coordinate["reference_frame_name"],
            "reference_frame_index": int(coordinate["reference_frame_index"]),
        }
        records.append(record)
        frame_offset += frame_count
    return records


def _load_sources(
    *,
    scene_dir: str | Path,
    completed_modes_dir: str | Path,
    coordinates_dir: str | Path,
) -> tuple[
    Path,
    ForegroundBackgroundScene,
    CompletedModesArtifact,
    str,
    CoordinateArtifact,
    RenderedModalDesignArtifact,
    DirectModalCoordinatesArtifact | None,
    list[dict[str, Any]],
]:
    """Load and cross-check the complete static/mode/design/coordinate chain."""

    scene_path = resolve_path(scene_dir, strict=True)
    completed_path = resolve_path(completed_modes_dir, strict=True)
    coordinate_path = resolve_path(coordinates_dir, strict=True)
    scene = load_static_scene(scene_path, "cpu")
    completed = load_completed_modes(completed_path)
    coordinate_kind, coordinates = _load_coordinate_artifact(coordinate_path)
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")

    _require_equal(
        "completed static scene identity",
        completed.manifest.get("static_scene_identity"),
        scene_manifest["static_scene_identity"],
    )
    _require_equal(
        "completed foreground identity",
        completed.manifest.get("foreground_identity"),
        scene_manifest["foreground_identity"],
    )
    _require_equal(
        "coordinate completed-mode identity",
        coordinates.manifest.get("completed_modes_identity"),
        completed.manifest["completed_modes_identity"],
    )
    _require_equal(
        "coordinate mode order",
        coordinates.manifest.get("modes"),
        completed.manifest["modes"],
    )

    if coordinate_kind in ("refined_rgb", "sweep_rgb"):
        from modal_gaussians.coordinates.sequences import validate_sequences
        cm = coordinates.manifest
        _require_equal("coordinate scene", cm['static_scene_identity'], scene_manifest['static_scene_identity'])
        views = cm['views']
        validate_sequences(views, cm['images'], scene_manifest, frame_count=len(coordinates.coordinates))
        original = {v['label']:v for v in completed.manifest['views']}
        for view in views:
            if view['kind'] == 'fixed':
                if view['label'] not in original:
                    raise ValueError('Refined fixed recording not in observation sources')
                _require_equal('refined mode camera', view['camera_identity'], original[view['label']]['camera_identity'])
        if coordinate_kind == 'refined_rgb':
            if completed.manifest.get('version') != 19 or scene_manifest.get('version') != 7:
                raise ValueError('Refined coordinates require derived scene and mode bank')
            for key in ('preparation_identity','run_identity'):
                _require_equal(f'refined {key}',cm[key],completed.manifest[key])
        _require_equal('coordinate field shape', completed.arrays['phi'].shape,
                       (len(cm['modes']),scene.foreground.count,3))
        if not np.array_equal(completed.arrays['g_points'],scene.foreground.params['means'].detach().numpy()):
            raise ValueError('Coordinate Gaussian order differs')
        return scene_path,scene,completed,coordinate_kind,coordinates,None,None,views

    design_source = coordinates.manifest.get("rendered_design")
    if not isinstance(design_source, str) or not design_source:
        raise ValueError("Coordinate artifact does not name its rendered design")
    design = load_rendered_modal_design(design_source)
    _require_equal(
        "rendered-design identity",
        coordinates.manifest.get("rendered_design_identity"),
        design.manifest["rendered_design_identity"],
    )
    for name, expected in (
        ("static_scene_identity", scene_manifest["static_scene_identity"]),
        ("foreground_identity", scene_manifest["foreground_identity"]),
        ("completed_modes_identity", completed.manifest["completed_modes_identity"]),
        ("modes", completed.manifest["modes"]),
    ):
        _require_equal(f"rendered-design {name}", design.manifest.get(name), expected)
    _require_equal(
        "rendered-design foreground count",
        design.manifest["counts"].get("foreground_gaussians"),
        scene.foreground.count,
    )

    design_views = design.manifest["views"]
    completed_views = completed.manifest["views"]
    direct: DirectModalCoordinatesArtifact | None = None
    if coordinate_kind == "rgb":
        direct_source = coordinates.manifest.get("direct_coordinates")
        if not isinstance(direct_source, str) or not direct_source:
            raise ValueError(f"{coordinate_kind} coordinates do not name their direct source")
        direct = load_direct_modal_coordinates(direct_source)
        _require_equal(
            f"{coordinate_kind} direct-coordinate identity",
            coordinates.manifest.get("direct_coordinates_identity"),
            direct.manifest["direct_coordinates_identity"],
        )
        for name in ("rendered_design_identity", "completed_modes_identity", "modes"):
            _require_equal(
                f"{coordinate_kind}/direct {name}",
                coordinates.manifest.get(name),
                direct.manifest.get(name),
            )
        direct_views = direct.manifest["views"]
        coordinate_views = coordinates.manifest["views"]
        if coordinate_kind == "rgb":
            # Check the complete source chain before mapping a fitted subset by label.
            _coordinate_view_records(direct_views, design_views, completed_views)
            labels = [view["label"] for view in coordinate_views]
            direct_views = _select_views(direct_views, labels)
            design_views = _select_views(design_views, labels)
            completed_views = _select_views(completed_views, labels)
        if len(coordinate_views) != len(direct_views):
            raise ValueError(f"{coordinate_kind}/direct coordinate view counts differ")
        runtime_fields = (
            "index",
            "label",
            "flow_identity",
            "shape_hw",
            "fps_hz",
            "frame_offset",
            "frame_count",
            "frame_names",
            "reference_frame_name",
            "reference_frame_index",
            "sample_count",
        )
        for index, (coordinate_view, direct_view) in enumerate(
            zip(coordinate_views, direct_views)
        ):
            for name in runtime_fields:
                _require_equal(
                    f"{coordinate_kind}/direct view {index} {name}",
                    coordinate_view.get(name),
                    direct_view.get(name),
                )

    views = _coordinate_view_records(
        coordinates.manifest["views"],
        design_views,
        completed_views,
    )
    expected_shape = (
        sum(record["frame_count"] for record in views),
        len(completed.manifest["modes"]),
    )
    _require_equal("coordinate array shape", coordinates.coordinates.shape, expected_shape)
    phi = completed.arrays["phi"]
    _require_equal(
        "completed phi shape",
        phi.shape,
        (len(completed.manifest["modes"]), scene.foreground.count, 3),
    )
    return (
        scene_path,
        scene,
        completed,
        coordinate_kind,
        coordinates,
        design,
        direct,
        views,
    )


def _source_record(path: Path, identity_name: str, identity: str) -> dict[str, str]:
    """Describe one linked artifact without putting its path in result identity."""

    return {
        "path": str(path.resolve()),
        "identity_name": identity_name,
        "identity": identity,
    }


def _assemble_result(scene_dir, completed_modes_dir, coordinate_paths, result_path):
    from modal_gaussians.coordinates.sequences import bind_fixed_view, reindex_views, validate_sequences
    paths = list(coordinate_paths)
    if not paths: raise ValueError('Result needs coordinates')
    parts = [_load_sources(scene_dir=scene_dir,completed_modes_dir=completed_modes_dir,coordinates_dir=p) for p in paths]
    _,scene,completed = parts[0][:3]
    sources = dict(static_scene=_source_record(resolve_path(scene_dir),'static_scene_identity',scene.manifest['static_scene_identity']),
                   completed_modes=_source_record(completed.path,'completed_modes_identity',completed.manifest['completed_modes_identity']))
    views,images,arrays,bindings,artifacts = [],[],[],[],[]
    for i,(_,_,_,kind,coord,design,direct,records) in enumerate(parts):
        key=f'coordinates_{i}'
        name=COORDINATE_IDENTITY_NAMES[kind]
        sources[key]=_source_record(coord.path,name,coord.manifest[name])
        if design is not None:
            sources[f'rendered_design_{i}']=_source_record(design.path,'rendered_design_identity',design.manifest['rendered_design_identity'])
        if direct is not None:
            sources[f'direct_coordinates_{i}']=_source_record(direct.path,'direct_coordinates_identity',direct.manifest['direct_coordinates_identity'])
        bindings.append(dict(source=key,kind=kind,format=coord.manifest['format'],identity_name=name,
                             identity=coord.manifest[name],direct_coordinates_identity=direct.manifest['direct_coordinates_identity'] if direct else None))
        source_images=coord.manifest.get('images',[])
        if kind in ('rgb','direct'):
            by_label={v['label']:v for v in source_images}
            # Direct coordinates have no RGB target record; they remain playback-only.
            records=[bind_fixed_view(v,by_label[v['label']] if kind=='rgb' else
                     {'files':[{'sha256':None}]*v['frame_count']}) for v in records]
        views.extend(records);images.extend(source_images);arrays.append(coord.coordinates);artifacts.append(coord)
    if len({v['label'] for v in views}) != len(views):
        raise ValueError('Duplicate result sequence labels')
    kinds=[b['kind'] for b in bindings]
    if 'direct' in kinds and len(kinds)>1:
        raise ValueError('Direct playback coordinates cannot be mixed with RGB recordings')
    views=reindex_views(views)
    q=np.concatenate(arrays)
    if kinds!=['direct']: validate_sequences(views,images,scene.manifest,frame_count=len(q))
    kind=kinds[0] if len(set(kinds))==1 else 'mixed_rgb'
    cm=dict(views=views,images=images)
    merged=RGBModalCoordinatesArtifact(Path(result_path),cm,q)
    sm=scene.manifest
    manifest=dict(format=MODAL_RESULT_FORMAT,version=MODAL_RESULT_VERSION,storage=dict(STORAGE_CONVENTION),
        sources=sources,static_scene_identity=sm['static_scene_identity'],foreground_identity=sm['foreground_identity'],
        background_identity=sm['background_identity'],completed_modes_identity=completed.manifest['completed_modes_identity'],
        rendered_design_identity=parts[0][5].manifest['rendered_design_identity'] if len(parts)==1 and parts[0][5] else None,
        coordinate_sources=bindings,coordinate_source=dict(kind=kind,identity=hashlib.sha256(_canonical_json(bindings)).hexdigest()),
        modes=completed.manifest['modes'],views=views,
        counts=dict(foreground_gaussians=scene.foreground.count,background_gaussians=scene.background.count,
                    modes=q.shape[1],views=len(views),frames=len(q)),deformation=dict(DEFORMATION_CONVENTION),
        quality_gate=dict(required=True,status='modal_result_candidate_unapproved',
                          inherited_from=[a.manifest['quality_gate']['status'] for a in artifacts]))
    manifest['modal_result_identity']=hashlib.sha256(_canonical_json(_identity_payload(manifest))).hexdigest()
    return ModalResultArtifact(Path(result_path),manifest,scene,completed,merged,
                               parts[0][5] if len(parts)==1 else None,parts[0][6] if len(parts)==1 else None,tuple(artifacts))


def load_modal_result(path):
    root=resolve_path(path,strict=True)
    manifest=_read_manifest(root)
    if manifest.get('format')!=MODAL_RESULT_FORMAT or manifest.get('version')!=MODAL_RESULT_VERSION:
        raise ValueError('Unsupported modal result; materialize a new v3 result')
    sources=manifest['sources']
    result=_assemble_result(_source_path(sources,'static_scene'),_source_path(sources,'completed_modes'),
                            [_source_path(sources,b['source']) for b in manifest['coordinate_sources']],root)
    for key,value in result.manifest.items():
        if key=='sources':
            if set(sources)!=set(value): raise ValueError('Result source inventory differs')
            for name,record in value.items():
                for field in ('identity','identity_name'):
                    _require_equal(f'{name} {field}',sources[name][field],record[field])
                _require_equal(f'{name} path',_source_path(sources,name),resolve_path(record['path'],strict=True))
        else:
            _require_equal(key,manifest.get(key),value)
    return ModalResultArtifact(root,manifest,result.scene,result.completed_modes,result.coordinates,
                               result.rendered_design,result.direct_coordinates,result.coordinate_artifacts)


def materialize_modal_result(*,scene_dir,completed_modes_dir,coordinates_dir,output_dir,command=()):
    from modal_gaussians.coordinates.preparation import _publish
    destination=resolve_path(output_dir)
    paths=[coordinates_dir] if isinstance(coordinates_dir,(str,Path)) else list(coordinates_dir)
    result=_assemble_result(scene_dir,completed_modes_dir,paths,destination)
    if any(destination.is_relative_to(resolve_path(s['path'])) for s in result.manifest['sources'].values()):
        raise ValueError('Result output must be outside immutable sources')
    if any(destination.is_relative_to(resolve_path(s['directory'])) for s in result.coordinates.manifest['images']):
        raise ValueError('Result output must be outside immutable input frames')
    result.manifest['producer']=dict(project_version=__version__,created_utc=datetime.now(timezone.utc).isoformat(),command=list(command))
    with _publish(destination) as work:
        (work/MANIFEST_FILENAME).write_text(json.dumps(result.manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        load_modal_result(work)
    return load_modal_result(destination)
