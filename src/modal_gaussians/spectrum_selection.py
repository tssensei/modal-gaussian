"""Uniform bins and grouped greedy flow reconstruction from cached FFT fields."""
from pathlib import Path
from modal_gaussians.scene_store import resolve_path
import json
import os
import tempfile
import time

import numpy as np

from modal_gaussians.flow.storage import open_array, read_pixels
from modal_gaussians.frequency import _ViewStatistics, _fit, _selected_columns, GREEDY_TIE_TOLERANCE
from modal_gaussians.iteration_cache import atomic_json
from modal_gaussians.progress import Progress, report_progress
from modal_gaussians.spectrum_cache import load_spectrum, save_selection


def uniform_bins(candidate_count, count):
    if not 1 <= count <= candidate_count:
        raise ValueError("Selection count exceeds the positive FFT bins")
    # Equal intervals over (0, Nyquist]; rounding is explicit grid sampling,
    # not the manual GUI's forbidden automatic frequency snapping.
    return np.rint(np.arange(1, count + 1) * candidate_count / count).astype(np.int64)


def _statistics(cache, topology_dir):
    root = resolve_path(topology_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "modal_gaussians.observation_topology":
        raise ValueError("Expected observation topology metadata")
    views = manifest["views"]
    if [v["label"] for v in views] != [v["label"] for v in cache.manifest["views"]]:
        raise ValueError("Topology and spectrum view order differ")
    with np.load(root / "topology.npz", allow_pickle=False) as data:
        sample_views, all_pixels = data["sample_view_index"], data["sample_pixels_xy"]
    statistics = []
    for view_index, (view, record) in enumerate(zip(views, cache.manifest["views"])):
        source = record["source_manifest"]
        if (resolve_path(source["stabilization_source"]) != resolve_path(view["flow_artifact"])
                or source["reference_frame_name"] != view["flow_reference_frame_name"]
                or record["shape_hw"] != view["shape_hw"]):
            raise ValueError(f"Reference geometry differs for {view['label']}")
        pixels = all_pixels[sample_views == view_index]
        if not len(pixels) or len(pixels) != view["sample_count"]:
            raise ValueError("Topology contains inconsistent or empty samples")
        report_progress(f"greedy {view['label']}: reading cached modes and SEA-RAFT targets at {len(pixels)} existing topology pixels")
        spectrum = open_array(cache.path / record["spectrum_file"])
        flow = open_array(Path(record["source_path"]) / source["flow_file"])
        try:
            modes = read_pixels(spectrum, slice(1, None), pixels)
            raw = read_pixels(flow, slice(None), pixels)
        finally:
            spectrum.store.close()
            flow.store.close()
        columns, frames = 2 * len(modes), len(raw)
        gram = np.zeros((columns, columns), np.float64)
        cross = np.zeros((columns, frames), np.float64)
        energy = np.zeros(frames, np.float64)
        progress = Progress(f"greedy statistics {view['label']}", len(pixels), unit="pixels")
        for start in range(0, len(pixels), 1024):
            stop = min(start + 1024, len(pixels))
            values = modes[:, start:stop].reshape(len(modes), -1).T
            design = np.empty((values.shape[0], columns), np.float64)
            design[:, 0::2], design[:, 1::2] = values.real, -values.imag
            target = raw[:, start:stop].astype(np.float64)
            target -= target[source["reference_frame_index"]:source["reference_frame_index"] + 1].copy()
            target = target.reshape(frames, -1).T
            if not np.isfinite(design).all() or not np.isfinite(target).all():
                raise ValueError("Non-finite cached modal fields or flow targets")
            gram += design.T @ design
            cross += design.T @ target
            energy += np.sum(target * target, axis=0)
            progress.update(stop)
        del raw, modes
        flow_energy = float(energy.sum())
        if flow_energy <= np.finfo(float).eps:
            raise ValueError(f"No flow energy for {view['label']}")
        diagonal = np.diag(gram)
        scales = np.sqrt((diagonal[::2] + diagonal[1::2]) / (2 * len(pixels)))
        scales[scales <= np.finfo(float).eps] = 1
        statistics.append(_ViewStatistics(view["label"], len(pixels), gram, cross, energy, flow_energy, scales))
    return statistics, manifest["topology_identity"]


def greedy_cached(statistics, frequencies, count):
    """Same equal-view grouped LS objective as legacy greedy, using Schur residuals.

    Project all candidate pairs against the current selected span together. Only
    2x2 eigensolves are needed for candidate gains, rather than a growing dense
    solve per candidate. Refresh residuals from original statistics every round.
    """
    if not statistics or not 1 <= count <= len(frequencies):
        raise ValueError("Invalid greedy count or empty views")
    normalized = []
    for stat in statistics:
        scales = np.repeat(stat.pair_scales, 2)
        normalized.append((stat.gram / scales[:, None] / scales[None, :], stat.cross_flow / scales[:, None]))
    selected, history = [], []
    progress = Progress("cached greedy selection", count, unit="frequencies")
    pairs = np.arange(len(frequencies)) * 2
    for step in range(count):
        scores = np.zeros(len(frequencies), np.float64)
        for stat, (gram, cross) in zip(statistics, normalized):
            if selected:
                columns = _selected_columns(selected)
                small = gram[np.ix_(columns, columns)]
                eigenvalues, vectors = np.linalg.eigh((small + small.T) * .5)
                tolerance = np.finfo(float).eps * max(len(columns), 2 * stat.pixel_count) * max(eigenvalues[-1], 0)
                active = eigenvalues > tolerance
                basis = vectors[:, active] / np.sqrt(eigenvalues[active])
                projection = gram[:, columns] @ basis
                residual = gram - projection @ projection.T
                response = cross - projection @ (basis.T @ cross[columns])
            else:
                residual, response = gram, cross
            blocks = np.empty((len(frequencies), 2, 2))
            blocks[:, 0, 0] = residual[pairs, pairs]
            blocks[:, 1, 1] = residual[pairs + 1, pairs + 1]
            blocks[:, 0, 1] = blocks[:, 1, 0] = (residual[pairs, pairs + 1] + residual[pairs + 1, pairs]) * .5
            eigenvalues, vectors = np.linalg.eigh(blocks)
            tolerance = np.finfo(float).eps * max(2 * (step + 1), 2 * stat.pixel_count) * max(1., float(np.diag(gram).max()))
            inverse = np.zeros_like(eigenvalues)
            np.divide(1., eigenvalues, out=inverse, where=eigenvalues > tolerance)
            for start in range(0, len(frequencies), 128):
                stop = min(start + 128, len(frequencies))
                responses = response[2 * start:2 * stop].reshape(stop - start, 2, -1)
                projected = np.einsum("fji,fjt->fit", vectors[start:stop], responses)
                scores[start:stop] += np.sum(projected**2 * inverse[start:stop, :, None], axis=(1, 2)) / stat.flow_energy / len(statistics)
        scores[selected] = -np.inf
        # Near-dependent pairs can differ by roundoff after Schur subtraction.
        # Resolve close candidates with the original rank-aware full solve and
        # its original lowest-frequency tie rule.
        best = None
        best_score = -np.inf
        best_fits = None
        for index in np.flatnonzero(scores >= np.max(scores) - 1e-7):
            fits = [_fit(stat, [*selected, int(index)]) for stat in statistics]
            score = float(np.mean([fit.r2 for fit in fits]))
            if score > best_score + GREEDY_TIE_TOLERANCE:
                best, best_score, best_fits = int(index), score, fits
        if best is None:
            raise RuntimeError("No finite greedy candidate")
        selected.append(best)
        fits = best_fits
        macro = float(np.mean([fit.r2 for fit in fits]))
        previous = history[-1]["macro_r2"] if history else 0.
        if macro < previous - 1e-9:
            raise ValueError("Greedy reconstruction objective decreased")
        history.append({"step": step + 1, "bin": best + 1, "frequency_hz": float(frequencies[best]),
                        "macro_r2": macro, "marginal_macro_r2_gain": max(0., macro - previous),
                        "view_r2": [fit.r2 for fit in fits], "numerical_rank": [fit.numerical_rank for fit in fits]})
        progress.update(step + 1, f"Hz={frequencies[best]:.6g} macro_r2={macro:.6f} gain={macro - previous:.6f}", force=True)
    return np.asarray(selected, np.int64) + 1, history


def select_spectrum(*, spectrum_dir, method, count, output_dir, topology_dir=None):
    started = time.perf_counter()
    cache = load_spectrum(spectrum_dir)
    if not 1 <= count < len(cache.frequencies) or method not in ("uniform", "greedy"):
        raise ValueError("Invalid selection method/count")
    output = resolve_path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-writing-", dir=output.parent))
    metadata = {"method": method, "count": count, "spectrum_path": str(cache.path),
                "spectrum_identity": cache.manifest["spectrum_identity"], "validation": False}
    if method == "uniform":
        bins = uniform_bins(len(cache.frequencies) - 1, count)
        metadata["sampling"] = "equally spaced targets over (0, Nyquist], explicitly rounded to cached bins"
    else:
        if topology_dir is None:
            raise ValueError("Greedy selection requires the existing observation topology")
        statistics, topology_identity = _statistics(cache, topology_dir)
        # Small sufficient statistics preserve selection inputs, not another
        # copy of the full flow or FFT cache.
        np.savez(temporary / "statistics.npz", **{
            f"view_{i}_{name}": getattr(stat, name) for i, stat in enumerate(statistics)
            for name in ("gram", "cross_flow", "energy_per_frame", "pair_scales")})
        bins, history = greedy_cached(statistics, cache.frequencies[1:], count)
        metadata.update(objective="equal_view_macro_r2_grouped_complex_pair",
                        solver="batched_schur_residual_pairs", exact_finalist_margin=1e-7,
                        target="reference_relative_raw_SEA_RAFT_flow",
                        topology_path=str(resolve_path(topology_dir)), topology_identity=topology_identity,
                        view_labels=[stat.label for stat in statistics],
                        pixel_counts=[stat.pixel_count for stat in statistics],
                        flow_energy=[stat.flow_energy for stat in statistics], history=history)
    save_selection(cache, bins, temporary / "selection.json", preserve_order=True)
    metadata.update(bins=bins.tolist(), frequencies_hz=cache.frequencies[bins].tolist(),
                    seconds=time.perf_counter() - started, status="complete")
    atomic_json(temporary / "report.json", metadata)
    os.rename(temporary, output)
    report_progress(f"Saved {method} {count} frequencies: {output} ({metadata['seconds']:.1f}s)")
    return output
