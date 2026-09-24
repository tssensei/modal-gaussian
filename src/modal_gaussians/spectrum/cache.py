"""Shared FFT grids and lossless selected-mode exports, independent of training."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
import re
import shutil
import tempfile
import time

import numpy as np

from modal_gaussians.spectrum.transform import temporal_rfft_tiles
from modal_gaussians.flow.storage import create_array, open_array
from modal_gaussians.common.cache import atomic_json, identity, sha256
from modal_gaussians.spectrum.modes import TRANSFORM_CONVENTION
from modal_gaussians.common.progress import Progress, report_progress

FORMAT = "modal_gaussians.shared_spectrum"
EXPORT_FORMAT = "modal_gaussians.spectrum_selected_frequency"
SELECTION_FORMAT = "modal_gaussians.spectrum_selection"
REGIONS = ("selected_box", "full_frame")


def _json(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


@dataclass
class SpectrumCache:
    path: Path
    manifest: dict
    frequencies: np.ndarray
    arrays: dict = field(default_factory=dict, repr=False)


def _index(value, size, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < size:
        raise ValueError(f"Invalid {label}: {value}")
    return int(value)


def _file(cache, relative):
    path = (cache.path / relative).resolve()
    if not path.is_relative_to(cache.path):
        raise ValueError("Spectrum file is outside its cache")
    return path


def load_spectrum(path):
    """Open only metadata and the small frequency table, never scan numerical data."""
    root = resolve_path(path, strict=True)
    manifest = _json(root / "manifest.json")
    if (manifest.get("format") != FORMAT or manifest.get("version") != 2
            or manifest.get("status") != "complete"):
        raise ValueError("Spectrum cache is incomplete or unsupported")
    length, fps = manifest["fft_length"], float(manifest["fps_hz"])
    if (isinstance(length, bool) or not isinstance(length, int) or length < 3
            or not math.isfinite(fps) or fps <= 0 or not manifest.get("views")
            or manifest.get("transform") != TRANSFORM_CONVENTION):
        raise ValueError("Invalid spectrum metadata")
    labels = [view.get("label", "") for view in manifest["views"]]
    if (len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels)):
        raise ValueError("Spectrum view labels must be unique safe directory names")
    frequencies = np.load(root / "frequencies.npy", allow_pickle=False)
    expected = np.fft.rfftfreq(length, d=1 / fps)
    if frequencies.dtype != np.float64 or not np.array_equal(frequencies, expected):
        raise ValueError("Spectrum frequency table differs from its declared grid")
    return SpectrumCache(root, manifest, frequencies)


def frequency_bin(cache, frequency):
    """Accept a grid frequency exactly (apart from decimal floating-point roundoff)."""
    value = float(frequency)
    if not math.isfinite(value) or value < 0 or value > cache.frequencies[-1]:
        raise ValueError("Frequency is outside the cached range")
    step = cache.manifest["fps_hz"] / cache.manifest["fft_length"]
    index = round(value / step)
    if index >= len(cache.frequencies) or not math.isclose(value, float(cache.frequencies[index]),
                                                         rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Frequency must lie on the {step:g} Hz grid; selection was not changed")
    return index


def read_mode(cache, view_index, bin_index):
    view_index = _index(view_index, len(cache.manifest["views"]), "view")
    bin_index = _index(bin_index, len(cache.frequencies), "frequency bin")
    record = cache.manifest["views"][view_index]
    if view_index not in cache.arrays:
        array = open_array(_file(cache, record["spectrum_file"]))
        if array.shape != (len(cache.frequencies), *record["shape_hw"], 2) or array.dtype != np.complex64:
            raise ValueError("Spectrum storage shape or dtype differs")
        cache.arrays[view_index] = array
    return np.asarray(cache.arrays[view_index][bin_index], dtype=np.complex64)


def read_region(cache, view_index, region):
    view_index = _index(view_index, len(cache.manifest["views"]), "view")
    record = cache.manifest["views"][view_index]
    if region not in REGIONS:
        raise ValueError("Unknown spectrum region")
    result = np.load(_file(cache, record['valid_file'] if region == 'full_frame' else record["region_file"]), allow_pickle=False)
    if result.dtype != bool or result.shape != tuple(record["shape_hw"]) or not result.any():
        raise ValueError("Selected-box region is empty or invalid")
    return result


def read_curves(cache, view_index):
    view_index = _index(view_index, len(cache.manifest["views"]), "view")
    with np.load(_file(cache, cache.manifest["views"][view_index]["curves_file"]), allow_pickle=False) as data:
        result = {name: data[name] for name in REGIONS}
    if any(values.shape != cache.frequencies.shape or not np.isfinite(values).all()
           for values in result.values()):
        raise ValueError("Invalid cached spectrum curve")
    return result


def _source(path):
    root = resolve_path(path, strict=True)
    source = _json(root / "manifest.json")
    if (source.get("format") != "modal_gaussians.sea_raft_flow"
            or source.get("version") != 3 or source.get("status") != "complete"
            or source.get("flow_file") != "flow.zarr" or source.get("flow_dtype") != "float32"
            or source.get("flow_direction") != "reference_to_frame"
            or source.get("flow_units") != "input_pixels"
            or source.get("transform") != TRANSFORM_CONVENTION):
        raise ValueError(f"Expected complete SEA-RAFT reference flow: {root}")
    shape = source["flow_shape"]
    frames, reference = source["frames"], source["reference_frame_index"]
    if (len(shape) != 4 or shape[-1] != 2 or min(shape) <= 0 or shape[0] != len(frames)
            or len(frames) < 3 or not math.isfinite(float(source["fps_hz"])) or source["fps_hz"] <= 0
            or not 0 <= reference < len(frames) or frames[reference] != source["reference_frame_name"]):
        raise ValueError(f"Invalid SEA-RAFT timing or shape: {root}")
    support = np.load(root/'valid_mask.npy', allow_pickle=False)
    if (sha256(root/'valid_mask.npy') != source['valid_mask_sha256'] or support.dtype != bool
            or list(support.shape) != shape[1:3] or not support.any()):
        raise ValueError('Invalid SEA-RAFT support')
    return root, source


def _scene_regions(scene_dir, sources):
    """Use camera/box metadata on CPU; do not load any Gaussian tensors."""
    from modal_gaussians.geometry.scene import cameras_from_scene_manifest
    from modal_gaussians.geometry.selection import projected_box_pixels

    root = resolve_path(scene_dir, strict=True)
    scene = _json(root / "manifest.json")
    partition = scene.get("partition", {})
    if (partition.get("method") != "manual_subject_selection_v1"
            or partition.get("mapping_file") != "partition.npz"):
        raise ValueError("Spectrum selected-box region requires a manually selected subject scene")
    with np.load(root / "partition.npz", allow_pickle=False) as data:
        box = tuple(data[name] for name in ("box_position", "box_wxyz", "box_dimensions"))
    cameras = {camera.label: camera for camera in cameras_from_scene_manifest(scene)
               if camera.role == "reference"}
    regions = []
    for label, _, source in sources:
        camera = cameras.get(label)
        if camera is None or [camera.height, camera.width] != source["flow_shape"][1:3]:
            raise ValueError(f"Reference camera/flow dimensions differ: {label}")
        pixels = projected_box_pixels(camera, *box)
        region = np.zeros((camera.height, camera.width), dtype=bool)
        region[pixels[:, 1], pixels[:, 0]] = True
        if not region.any():
            raise ValueError(f"Selected box is outside reference view: {label}")
        regions.append(region)
    return regions, {"path": str(root), "scene_identity": scene["static_scene_identity"],
                     "box": [value.tolist() for value in box]}


def build_spectrum(*, views, scene_dir=None, fft_length, output_dir, region_paths=None):
    """Compute every view once and atomically publish a common-frequency cache."""
    started = time.perf_counter()
    if (not views or len({label for label, _ in views}) != len(views)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label, _ in views)):
        raise ValueError("View labels must be unique letters, digits, underscores or hyphens")
    sources = [(label, *_source(path)) for label, path in views]
    fps = float(sources[0][2]["fps_hz"])
    if any(float(source["fps_hz"]) != fps for _, _, source in sources):
        raise ValueError("Shared FFT cache requires equal sampling rates across views")
    if (isinstance(fft_length, bool) or not isinstance(fft_length, int)
            or fft_length < max(len(source["frames"]) for _, _, source in sources)):
        raise ValueError("FFT length must cover every source sequence without truncation")
    if region_paths is None:
        if scene_dir is None:
            raise ValueError("Provide a subject scene or per-view analysis regions")
        regions, scene_record = _scene_regions(scene_dir, sources)
    else:
        import hashlib
        if scene_dir is not None or len(region_paths) != len(sources) or dict(region_paths).keys() != {v[0] for v in sources}:
            raise ValueError("Provide exactly one analysis region per view, without --scene")
        regions, records = [], []
        for label, _, source in sources:
            path = Path(dict(region_paths)[label]).resolve()
            region = np.load(path, allow_pickle=False)
            if region.dtype != bool or list(region.shape) != source["flow_shape"][1:3] or not region.any():
                raise ValueError(f"Invalid analysis region: {path}")
            regions.append(region)
            records.append({"label": label, "path": str(path),
                            "sha256": hashlib.sha256(region.tobytes()).hexdigest()})
        scene_record = {"analysis_regions": records, "cache_clipped": False}
    supports = [np.load(path/'valid_mask.npy', allow_pickle=False) for _, path, _ in sources]
    regions = [region & support for region, support in zip(regions, supports)]
    if any(not r.any() for r in regions):
        raise ValueError('No valid spectrum analysis pixels')
    contract = {"implementation": "shared_zero_padded_rfft_valid_v2", "fft_length": fft_length,
                "fps_hz": fps, "transform": TRANSFORM_CONVENTION, "scene": scene_record,
                "views": [{"label": label, "path": str(path), "source_identity": identity(source)}
                          for label, path, source in sources]}
    destination = resolve_path(output_dir)
    if destination.exists():
        cached = load_spectrum(destination)
        if cached.manifest.get("contract") != contract:
            raise FileExistsError(f"Existing spectrum has a different source/grid: {destination}")
        report_progress(f"Reusing complete spectrum: {destination}")
        return cached
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-writing-", dir=destination.parent))
    frequencies = np.fft.rfftfreq(fft_length, d=1 / fps)
    manifest = {"format": FORMAT, "version": 2, "status": "running", "contract": contract,
                "spectrum_identity": identity(contract), "fft_length": fft_length, "fps_hz": fps,
                "transform": TRANSFORM_CONVENTION, "views": [], "validation": False,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    np.save(temporary / "frequencies.npy", frequencies, allow_pickle=False)
    atomic_json(temporary / "manifest.json", manifest)
    try:
        for index, ((label, source_path, source), region) in enumerate(zip(sources, regions)):
            tick = time.perf_counter()
            relative = Path(f"view_{index:03d}")
            folder = temporary / relative
            folder.mkdir()
            flow = open_array(source_path / "flow.zarr")
            if list(flow.shape) != source["flow_shape"] or flow.dtype != np.float32:
                raise ValueError(f"SEA-RAFT flow storage differs from metadata: {label}")
            _, height, width, _ = flow.shape
            support = supports[index]
            spectrum = create_array(folder / "spectrum.zarr", (len(frequencies), height, width, 2), np.complex64)
            sums = {name: np.zeros(len(frequencies), dtype=np.float64) for name in REGIONS}
            progress = Progress(f"shared FFT {label}", height * width, unit="pixels")
            completed = 0
            try:
                for rows, columns, values in temporal_rfft_tiles(flow, fft_length=fft_length, block_width=64):
                    spectrum[:, rows, columns, :] = values
                    amplitude = np.sqrt(np.abs(values[..., 0]) ** 2 + np.abs(values[..., 1]) ** 2)
                    sums["full_frame"] += amplitude[:, support[rows, columns]].sum(axis=1, dtype=np.float64)
                    local = region[rows, columns]
                    if local.any():
                        sums["selected_box"] += amplitude[:, local].sum(axis=1, dtype=np.float64)
                    completed += local.size
                    progress.update(completed)
            finally:
                flow.store.close()
                spectrum.store.close()
            sums["full_frame"] /= int(support.sum())
            sums["selected_box"] /= int(region.sum())
            np.savez(folder / "curves.npz", **sums)
            np.save(folder / "region.npy", region, allow_pickle=False)
            np.save(folder / "valid.npy", support, allow_pickle=False)
            shutil.copyfile(resolve_path(source["reference_image"]), folder / "reference.png")
            record = {"label": label, "shape_hw": [height, width], "source_path": str(source_path),
                      "source_manifest": source, "reference_image": (relative / "reference.png").as_posix(),
                      "spectrum_file": (relative / "spectrum.zarr").as_posix(),
                      "region_file": (relative / "region.npy").as_posix(),
                      "valid_file": (relative / "valid.npy").as_posix(),
                      "curves_file": (relative / "curves.npz").as_posix(),
                      "selected_box_pixels": int(region.sum()), "seconds": time.perf_counter() - tick}
            manifest["views"].append(record)
            atomic_json(temporary / "manifest.json", manifest)
            report_progress(f"{label} spectrum complete in {record['seconds']:.1f}s")
        manifest.update(status="complete", seconds=time.perf_counter() - started)
        atomic_json(temporary / "manifest.json", manifest)
        os.rename(temporary, destination)
    except BaseException:
        report_progress(f"Incomplete spectrum retained at {temporary}; it is not a usable cache")
        raise
    report_progress(f"Shared spectrum complete: {destination} | seconds={manifest['seconds']:.1f}")
    return SpectrumCache(destination, manifest, frequencies)


def save_selection(cache, bins, destination, *, preserve_order=False):
    selected = list(dict.fromkeys(_index(value, len(cache.frequencies), "frequency bin") for value in bins))
    if not preserve_order:
        selected.sort()
    if 0 in selected:
        raise ValueError("DC can be inspected but cannot be selected for training")
    output = resolve_path(destination)
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, {"format": SELECTION_FORMAT, "version": 1,
                        "spectrum_path": str(cache.path), "spectrum_identity": cache.manifest["spectrum_identity"],
                        "bins": selected, "frequencies_hz": [float(cache.frequencies[k]) for k in selected]})
    return output


def export_selection(cache, selection_path, output_dir):
    """Copy selected complex slices; never consult flow or compute another transform."""
    selection = _json(selection_path)
    if (selection.get("format") != SELECTION_FORMAT or selection.get("version") != 1
            or selection.get("spectrum_identity") != cache.manifest["spectrum_identity"]):
        raise ValueError("Selection does not belong to this spectrum cache")
    bins = selection.get("bins", [])
    if not isinstance(bins, list):
        raise ValueError("Selection bins must be a list")
    selected = [_index(value, len(cache.frequencies), "frequency bin") for value in bins]
    if not selected or 0 in selected or len(set(selected)) != len(selected):
        raise ValueError("Export requires distinct positive frequency bins")
    frequencies = [float(cache.frequencies[k]) for k in selected]
    if selection.get("frequencies_hz") != frequencies:
        raise ValueError("Selection frequencies differ from cached bin frequencies")
    output = resolve_path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-writing-", dir=output.parent))
    records = []
    for bin_index, frequency in zip(selected, frequencies):
        for view_index, view in enumerate(cache.manifest["views"]):
            relative = Path(f"bin_{bin_index:04d}") / view["label"]
            folder = temporary / relative
            folder.mkdir(parents=True)
            values = read_mode(cache, view_index, bin_index)
            np.save(folder / "modal_image.npy", values[None], allow_pickle=False)
            np.save(folder / 'valid_mask.npy', read_region(cache, view_index, 'full_frame'), allow_pickle=False)
            source = view["source_manifest"]
            manifest = {name: source.get(name) for name in (
                "images", "stabilization_source", "stabilized_images", "inference_images", "reference_image",
                "reference_frame_name", "reference_frame_index", "fps_hz", "frames", "flow_units",
                "flow_direction", "smoothing", "transform")}
            if "reference_selection" in source:
                manifest["reference_selection"] = source["reference_selection"]
            manifest.update(format=EXPORT_FORMAT, version=2, status="complete", frequency_hz=frequency,
                            valid_mask_sha256=sha256(folder/'valid_mask.npy'),
                            modes_file="modal_image.npy", modes_shape=[1, *view["shape_hw"], 2],
                            modes_dtype="complex64", full_spectrum=False, validation=False,
                            source_flow_path=view["source_path"],
                            spectrum_source={"path": str(cache.path), "identity": cache.manifest["spectrum_identity"],
                                             "fft_length": cache.manifest["fft_length"], "bin_index": bin_index,
                                             "frequency_step_hz": cache.manifest["fps_hz"] / cache.manifest["fft_length"]})
            atomic_json(folder / "manifest.json", manifest)
            records.append({"bin_index": bin_index, "frequency_hz": frequency, "view": view["label"],
                            "path": relative.as_posix()})
    atomic_json(temporary / "manifest.json", {"format": "modal_gaussians.spectrum_export", "version": 1,
                                            "status": "complete", "selection": selection, "modes": records})
    os.rename(temporary, output)
    return output
