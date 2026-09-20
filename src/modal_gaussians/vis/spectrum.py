"""Shared-FFT and saved-mode visualization, without legacy flow reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable

import cv2
import numpy as np
import viser
import viser.uplot

from modal_gaussians.iteration_cache import identity
from modal_gaussians.scene_store import resolve_path
from modal_gaussians.spectrum_cache import load_spectrum, read_curves, read_region

PREVIEW_PERCENTILE = 99.0
NORMALIZATIONS = ("per mode", "all saved modes")


@dataclass(frozen=True)
class SpectrumViewState:
    index: int
    label: str
    pixels_xy: np.ndarray
    reference_rgb: np.ndarray
    raw_frequencies_hz: np.ndarray
    raw_power: np.ndarray
    reconstructed_power: np.ndarray
    alphas: np.ndarray
    alpha_identifiable: np.ndarray
    frequency_limits: tuple[float, float]
    selection_pixels_xy: np.ndarray
    magnitude_hi: np.ndarray


def _json(path):
    return json.loads(resolve_path(path, strict=True).read_text(encoding="utf-8"))


def _hsv_rgb(values: np.ndarray, magnitude_hi: float) -> np.ndarray:
    """Convert complex phase to hue and amplitude to brightness."""

    phase = (np.angle(values) + np.pi) / (2.0 * np.pi)
    value = np.clip(np.abs(values) / max(float(magnitude_hi), 1.0e-12), 0.0, 1.0)
    h6 = np.mod(phase, 1.0) * 6.0
    sector = np.floor(h6).astype(np.int64)
    fraction = h6 - sector
    p = np.zeros_like(value)
    q = value * (1.0 - fraction)
    t = value * fraction
    candidates = (
        np.stack((value, t, p), axis=-1),
        np.stack((q, value, p), axis=-1),
        np.stack((p, value, t), axis=-1),
        np.stack((p, q, value), axis=-1),
        np.stack((t, p, value), axis=-1),
        np.stack((value, p, q), axis=-1),
    )
    rgb = np.zeros(values.shape + (3,), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        rgb[sector == index] = candidate[sector == index]
    return rgb


class SpectrumComparisonController:
    """Display a fixed mode bank using its exact exported bins and training alphas."""

    def __init__(self, result):
        self.result = result
        bank = result.completed_modes.manifest
        if bank.get("version") != 17:
            raise ValueError("The shared-spectrum panel requires a fixed-frequency mode bank (v17)")
        self.frequencies_hz = np.asarray([m["frequency_hz"] for m in bank["modes"]])
        self.design_views = result.rendered_design.manifest["views"]
        self.available_view_ids = tuple(v["label"] for v in self.design_views)
        if len(set(self.available_view_ids)) != len(self.available_view_ids):
            raise ValueError("Spectrum view labels must be unique")
        self._images = {label: [] for label in self.available_view_ids}
        self._alphas = np.zeros((len(self.frequencies_hz), len(self.design_views)), np.complex64)
        self._identifiable = np.zeros(self._alphas.shape, bool)
        self.cache = None
        if len(bank["sources"]) != len(self.frequencies_hz):
            raise ValueError("Mode bank source count differs from modes")
        for k, source in enumerate(bank["sources"]):
            root = resolve_path(source["path"], strict=True)
            model = _json(root / "manifest.json")
            slot = source["slot"]
            if (model["completed_modes_identity"] != source["identity"]
                    or model["static_scene_identity"] != bank["static_scene_identity"]
                    or model["foreground_identity"] != bank["foreground_identity"]
                    or model["modes"][slot]["frequency_hz"] != self.frequencies_hz[k]):
                raise ValueError("Mode bank source identity/frequency differs")
            dense = _json(resolve_path(model["complex_2d_modes"]) / "manifest.json")
            if dense["complex_2d_modes_identity"] != model["complex_2d_modes_identity"]:
                raise ValueError("Saved modal image source identity differs")
            dense_views = {v["label"]: v for v in dense["views"]}
            model_views = {v["label"]: i for i, v in enumerate(model["views"])}
            array_path = (root / model["arrays_file"]).resolve(strict=True)
            if not array_path.is_relative_to(root):
                raise ValueError("Saved alignment must remain inside its model artifact")
            # Read only tiny saved alignment arrays, never the network or baked Phi again.
            with np.load(array_path, allow_pickle=False) as arrays:
                alphas = arrays["alphas"][slot]
                identifiable = arrays["alpha_identifiable_mask"][slot]
            for v, design_view in enumerate(self.design_views):
                label = design_view["label"]
                view = dense_views[label]
                if (view["camera_identity"] != design_view["camera_identity"]
                        or view["shape_hw"] != design_view["shape_hw"]):
                    raise ValueError("Modal image camera/shape differs from rendered design")
                selected = view["selected_source"]
                exported = _json(resolve_path(selected["path"]) / "manifest.json")
                if (identity(exported) != selected["manifest_identity"]
                        or exported.get("format") != "modal_gaussians.spectrum_selected_frequency"
                        or exported.get("status") != "complete"):
                    raise ValueError("Modal image must bind a complete cached-bin export")
                grid = exported["spectrum_source"]
                if self.cache is None:
                    self.cache = load_spectrum(grid["path"])
                if (grid["identity"] != self.cache.manifest["spectrum_identity"]
                        or resolve_path(grid["path"]) != self.cache.path
                        or grid["fft_length"] != self.cache.manifest["fft_length"]):
                    raise ValueError("Modal images must share the exact saved FFT cache")
                index = grid["bin_index"]
                if (type(index) is not int or not 0 <= index < len(self.cache.frequencies)
                        or not math.isclose(self.cache.frequencies[index], self.frequencies_hz[k], rel_tol=0, abs_tol=1e-9)
                        or exported["frequency_hz"] != self.frequencies_hz[k]):
                    raise ValueError("Modal frequency is not its exact shared FFT bin")
                cache_view = next(item for item in self.cache.manifest["views"] if item["label"] == label)
                timing = cache_view["source_manifest"]
                if (resolve_path(exported["source_flow_path"]) != resolve_path(cache_view["source_path"])
                        or exported["reference_frame_name"] != design_view["flow_reference_frame_name"]
                        or exported["reference_frame_index"] != design_view["flow_reference_frame_index"]
                        or exported["frames"] != timing["frames"]
                        or exported["fps_hz"] != design_view["fps_hz"]
                        or exported["modes_shape"] != [1, *design_view["shape_hw"], 2]):
                    raise ValueError("Spectrum/export reference timing or source differs")
                image_path = resolve_path(view["modes_file"], strict=True)
                if image_path != resolve_path(selected["path"]) / exported["modes_file"]:
                    raise ValueError("Modal image path differs from its export")
                self._images[label].append(image_path)
                self._alphas[k, v] = alphas[model_views[label]]
                self._identifiable[k, v] = identifiable[model_views[label]]
        if not np.isfinite(self._alphas).all():
            raise ValueError("Saved view alignment contains non-finite values")
        self._states = {}
        self.component_index = self.reconstructed_index = 0
        self.amplitude_normalization = "per mode"
        self.original_image_region = "Cached region"
        self.select_view(self.available_view_ids[0])

    def _raw(self, label, k, pixels):
        field = np.load(self._images[label][k], mmap_mode="r", allow_pickle=False)
        try:
            v = self.available_view_ids.index(label)
            if field.dtype != np.complex64 or field.shape != (1, *self.design_views[v]["shape_hw"], 2):
                raise ValueError("Modal image must be complex64 [1,H,W,2]")
            return np.asarray(field[0, pixels[:, 1], pixels[:, 0], :]).copy()
        finally:
            field._mmap.close()

    def _projection(self, v, k):
        design = self.result.rendered_design
        lo, hi = design.samples["view_sample_offsets"][v:v + 2]
        matrix = design.design[lo:hi]
        if not self._identifiable[k, v]:
            return np.zeros((hi - lo, 2), np.complex64)
        return self._alphas[k, v] * (matrix[:, :, 2*k] - 1j * matrix[:, :, 2*k+1])

    def _prepare_view(self, label):
        v = self.available_view_ids.index(label)
        cache_index = next(i for i, item in enumerate(self.cache.manifest["views"]) if item["label"] == label)
        record = self.cache.manifest["views"][cache_index]
        design = self.result.rendered_design
        lo, hi = design.samples["view_sample_offsets"][v:v + 2]
        pixels = np.asarray(design.samples["sample_pixels_xy"][lo:hi], dtype=np.int64)
        h, w = record["shape_hw"]
        if len(pixels) == 0 or np.any(pixels < 0) or np.any(pixels >= [w, h]):
            raise ValueError("Spectrum model-support pixels are empty or invalid")
        region = read_region(self.cache, cache_index, "selected_box")
        y, x = np.nonzero(region)
        selection = np.column_stack((x, y))
        reference = cv2.imread(str(resolve_path(self.cache.path / record["reference_image"], strict=True)))
        if reference is None or reference.shape != (h, w, 3):
            raise ValueError("Cached spectrum reference image has invalid dimensions")
        power = np.zeros(len(self.frequencies_hz), np.float32)
        maximum = np.zeros(2)
        for k in range(len(self.frequencies_hz)):
            raw, projected = self._raw(label, k, pixels), self._projection(v, k)
            power[k] = np.linalg.norm(projected, axis=1).mean()
            maximum = np.maximum(maximum, np.percentile(np.concatenate((np.abs(raw), np.abs(projected))),
                                                         PREVIEW_PERCENTILE, axis=0))
        f = self.cache.frequencies
        margin = 0.05 * max(float(np.ptp(self.frequencies_hz)), float(f[1] - f[0]))
        return SpectrumViewState(v, label, pixels, cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), f,
            read_curves(self.cache, cache_index)["selected_box"], power, self._alphas[:, v],
            self._identifiable[:, v], (max(float(f[0]), float(self.frequencies_hz.min()) - margin),
            min(float(f[-1]), float(self.frequencies_hz.max()) + margin)), selection,
            np.where(maximum > 0, maximum, 1.0))

    def select_view(self, label):
        if label not in self.available_view_ids:
            raise ValueError(f"Unknown spectrum view: {label!r}")
        if label not in self._states:
            self._states[label] = self._prepare_view(label)
        self.view_id, self.state = label, self._states[label]
        self._refresh_products()

    def select_mode(self, index):
        if not 0 <= int(index) < len(self.frequencies_hz):
            raise ValueError("Mode index is outside the saved bank")
        self.reconstructed_index = int(index)
        self._refresh_products()

    def select_component(self, component):
        if component.upper() not in ("U", "V"):
            raise ValueError("Modal component must be U or V")
        self.component_index = 0 if component.upper() == "U" else 1
        self._refresh_products()

    def select_amplitude_normalization(self, normalization):
        if normalization not in NORMALIZATIONS:
            raise ValueError("Unknown modal image normalization")
        self.amplitude_normalization = normalization
        self._refresh_products()

    def select_original_image_region(self, region):
        if region not in ("Model support", "Cached region"):
            raise ValueError("Unknown modal image region")
        self.original_image_region = region
        self._refresh_products()

    def _modal_image(self, values, high, pixels, radius=0):
        output = 0.35 * self.state.reference_rgb.astype(np.float32) / 255
        colors = _hsv_rgb(values, high)
        h, w = output.shape[:2]
        x, y = pixels.T
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                output[np.clip(y + dy, 0, h-1), np.clip(x + dx, 0, w-1)] = colors
        return np.clip(np.rint(output * 255), 0, 255).astype(np.uint8)

    def _refresh_products(self):
        k, c = self.reconstructed_index, self.component_index
        pixels = self.state.pixels_xy
        raw = self._raw(self.view_id, k, pixels)[:, c]
        projected = self._projection(self.state.index, k)[:, c]
        high = (float(np.percentile(np.concatenate((np.abs(raw), np.abs(projected))), PREVIEW_PERCENTILE))
                if self.amplitude_normalization == "per mode" else float(self.state.magnitude_hi[c]))
        self.modal_image_magnitude_hi = high if math.isfinite(high) and high > 0 else 1.0
        if self.original_image_region == "Cached region":
            original_pixels = self.state.selection_pixels_xy
            raw = self._raw(self.view_id, k, original_pixels)[:, c]
        else:
            original_pixels = pixels
        self.raw_modal_image = self._modal_image(raw, self.modal_image_magnitude_hi, original_pixels)
        self.reconstructed_modal_image = self._modal_image(projected, self.modal_image_magnitude_hi, pixels, radius=1)
        self.status = (
            f"**View:** {self.view_id} · **Frequency:** {self.frequencies_hz[k]:.6f} Hz · "
            f"**Component:** {'U' if c == 0 else 'V'}  \n"
            "**Source:** saved SEA-RAFT shared FFT and exact exported bin.  \n"
            "**Projection:** fixed 3D mode with saved training view alignment; no display refit.  \n"
            "**Curve regions:** input = cached analysis region; projection = model-support samples. "
            "Their spatial averaging domains differ.  \n"
            "**Image brightness:** shared model-support percentile; pixels outside displayed support are dim reference RGB."
            + ("" if self.state.alpha_identifiable[k] else "  \n**Projection unavailable:** saved view alignment is unidentifiable.")
        )

    @property
    def raw_frequencies_hz(self): return self.state.raw_frequencies_hz
    @property
    def raw_power(self): return self.state.raw_power
    @property
    def reconstructed_power(self): return self.state.reconstructed_power
    @property
    def frequency_limits(self): return self.state.frequency_limits

    def current_modal_phase_display_context(self, mode_index, component_index, amplitude_normalization):
        if (mode_index != self.reconstructed_index or component_index != self.component_index
                or amplitude_normalization != self.amplitude_normalization):
            raise ValueError("3D phase controls and spectrum panel are not synchronized")
        return (self.view_id, complex(self.state.alphas[mode_index]),
                bool(self.state.alpha_identifiable[mode_index]), self.modal_image_magnitude_hi)


def _spectrum_plot_data(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    selected_frequency_hz: float,
    marker_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an ordered uPlot spectrum plus a narrow selected-frequency marker."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    values = np.asarray(power, dtype=np.float64)
    order = np.argsort(frequencies)
    frequencies, values = frequencies[order], values[order]
    spacing = float(np.min(np.diff(frequencies))) if len(frequencies) > 1 else 1.0
    half_width = max(1.0e-9, 1.0e-3 * spacing)
    marker_x = selected_frequency_hz + np.asarray([-half_width, 0.0, half_width])
    plot_x = np.unique(np.concatenate((frequencies, marker_x)))
    plot_power = np.interp(plot_x, frequencies, values)
    marker = np.full(plot_x.shape, np.nan)
    indices = [int(np.argmin(np.abs(plot_x - value))) for value in marker_x]
    marker[indices] = (0.0, marker_height, 0.0)
    return plot_x, plot_power, marker


class ModalSpectrumPanel:
    """Show cached spectra and selected fixed-mode projections in a floating panel."""

    def __init__(
        self,
        server: viser.ViserServer,
        controller: SpectrumComparisonController,
        *,
        on_view_selected: Callable[[str], None],
        on_mode_selected: Callable[[int], None],
        on_component_selected: Callable[[str], None],
        on_normalization_selected: Callable[[str], None],
        on_solo_selected: Callable[[int], None],
        on_enable_all: Callable[[], None],
    ) -> None:
        self.controller = controller
        self._updating = False
        self._frequency_order = tuple(
            int(value) for value in np.argsort(controller.frequencies_hz, kind="stable")
        )
        display = np.empty(len(self._frequency_order), dtype=np.int64)
        display[np.asarray(self._frequency_order)] = np.arange(len(display))
        self._display_indices = tuple(int(value) for value in display)
        self.view = server.gui.add_dropdown(
            "Spectrum view",
            options=controller.available_view_ids,
            initial_value=controller.view_id,
        )
        self.frequency_range = server.gui.add_dropdown(
            "Frequency range",
            options=("full spectrum", "selected modes"),
            initial_value="full spectrum",
        )
        self.component = server.gui.add_dropdown(
            "Modal image component",
            options=("U", "V"),
            initial_value="U" if controller.component_index == 0 else "V",
        )
        self.normalization = server.gui.add_dropdown(
            "Amplitude normalization",
            options=NORMALIZATIONS,
            initial_value=controller.amplitude_normalization,
        )
        self.original_region = server.gui.add_dropdown(
            "Original image region",
            options=("Cached region", "Model support"),
            initial_value=controller.original_image_region,
        )
        self.mode = server.gui.add_slider(
            "Selected mode",
            min=0,
            max=len(display) - 1,
            step=1,
            initial_value=self._display_indices[controller.reconstructed_index],
        )
        self.frequency = server.gui.add_number(
            "Selected frequency (Hz)",
            initial_value=float(controller.frequencies_hz[controller.reconstructed_index]),
            disabled=True,
        )
        solo = server.gui.add_button("Solo selected mode")
        enable_all = server.gui.add_button("Enable all modes")
        self.status = server.gui.add_markdown(controller.status)
        server.gui.add_markdown(
            "Input: saved full FFT curve over its cached analysis region. "
            "Projection: saved frequencies on model-support pixels, with training alignment. "
            "This is a modal projection, not a coefficient-video FFT."
        )
        maximum = self._shared_power_max()
        self.raw_plot = server.gui.add_uplot(
            data=_spectrum_plot_data(
                controller.raw_frequencies_hz,
                controller.raw_power,
                float(controller.frequencies_hz[controller.reconstructed_index]),
                maximum,
            ),
            series=self._series("Cached-region mean amplitude", "#4c9aff"),
            title=f"Original {controller.view_id} spectrum",
            scales=self._scales(maximum),
            legend=viser.uplot.Legend(show=True),
            height=260,
        )
        self.reconstructed_plot = server.gui.add_uplot(
            data=_spectrum_plot_data(
                controller.frequencies_hz,
                controller.reconstructed_power,
                float(controller.frequencies_hz[controller.reconstructed_index]),
                maximum,
            ),
            series=self._series("Model-support projected amplitude", "#ff9f43"),
            title=f"Projected modes {controller.view_id} spectrum",
            scales=self._scales(maximum),
            legend=viser.uplot.Legend(show=True),
            height=260,
        )
        self.raw_image = server.gui.add_image(
            controller.raw_modal_image,
            label="Original modal image",
            format="jpeg",
            jpeg_quality=90,
        )
        self.reconstructed_image = server.gui.add_image(
            controller.reconstructed_modal_image,
            label="Projected 3D modal image",
            format="jpeg",
            jpeg_quality=90,
        )

        @self.view.on_update
        def _(_) -> None:
            if not self._updating:
                controller.select_view(str(self.view.value))
                self._refresh()
                on_view_selected(str(self.view.value))

        @self.component.on_update
        def _(_) -> None:
            if not self._updating:
                on_component_selected(str(self.component.value))

        @self.frequency_range.on_update
        def _(_) -> None:
            if not self._updating:
                self._refresh()

        @self.normalization.on_update
        def _(_) -> None:
            if not self._updating:
                on_normalization_selected(str(self.normalization.value))

        @self.original_region.on_update
        def _(_) -> None:
            if not self._updating:
                controller.select_original_image_region(str(self.original_region.value))
                self._refresh()

        @self.mode.on_update
        def _(_) -> None:
            if not self._updating:
                on_mode_selected(self._frequency_order[int(self.mode.value)])

        @solo.on_click
        def _(_) -> None:
            on_solo_selected(self._frequency_order[int(self.mode.value)])

        @enable_all.on_click
        def _(_) -> None:
            on_enable_all()

    @staticmethod
    def _series(label: str, color: str) -> tuple[viser.uplot.Series, ...]:
        return (
            viser.uplot.Series(label="Frequency (Hz)"),
            viser.uplot.Series(label=label, stroke=color, width=2),
            viser.uplot.Series(label="Selected frequency", stroke="#ff3b30", width=1),
        )

    def _frequency_limits(self) -> tuple[float, float]:
        if self.frequency_range.value == "full spectrum":
            frequencies = self.controller.raw_frequencies_hz
            return float(frequencies[0]), float(frequencies[-1])
        return self.controller.frequency_limits

    def _shared_power_max(self) -> float:
        lower, upper = self._frequency_limits()
        raw_visible = (
            (self.controller.raw_frequencies_hz >= lower)
            & (self.controller.raw_frequencies_hz <= upper)
        )
        reconstructed_visible = (
            (self.controller.frequencies_hz >= lower)
            & (self.controller.frequencies_hz <= upper)
        )
        maxima = [0.0]
        if np.any(raw_visible):
            maxima.append(float(np.max(self.controller.raw_power[raw_visible])))
        if np.any(reconstructed_visible):
            maxima.append(
                float(
                    np.max(
                        self.controller.reconstructed_power[reconstructed_visible]
                    )
                )
            )
        maximum = max(maxima)
        return 1.05 * maximum if maximum > 0.0 else 1.0

    def _scales(self, maximum: float) -> dict[str, viser.uplot.Scale]:
        return {
            "x": viser.uplot.Scale(
                time=False, range=self._frequency_limits()
            ),
            "y": viser.uplot.Scale(range=(0.0, maximum)),
        }

    def _refresh(self) -> None:
        index = self.controller.reconstructed_index
        frequency = float(self.controller.frequencies_hz[index])
        maximum = self._shared_power_max()
        self._updating = True
        try:
            self.mode.value = self._display_indices[index]
            self.frequency.value = frequency
            self.view.value = self.controller.view_id
            self.component.value = "U" if self.controller.component_index == 0 else "V"
            self.normalization.value = self.controller.amplitude_normalization
            self.original_region.value = self.controller.original_image_region
        finally:
            self._updating = False
        self.status.content = self.controller.status
        self.raw_plot.data = _spectrum_plot_data(
            self.controller.raw_frequencies_hz,
            self.controller.raw_power,
            frequency,
            maximum,
        )
        self.raw_plot.title = f"Original {self.controller.view_id} spectrum"
        self.raw_plot.scales = self._scales(maximum)
        self.reconstructed_plot.data = _spectrum_plot_data(
            self.controller.frequencies_hz,
            self.controller.reconstructed_power,
            frequency,
            maximum,
        )
        self.reconstructed_plot.title = (
            f"Projected modes {self.controller.view_id} spectrum"
        )
        self.reconstructed_plot.scales = self._scales(maximum)
        self.raw_image.image = self.controller.raw_modal_image
        self.reconstructed_image.image = self.controller.reconstructed_modal_image

    def set_mode_index(self, index: int) -> None:
        self.controller.select_mode(index)
        self._refresh()

    def set_component(self, component: str) -> None:
        self.controller.select_component(component)
        self._refresh()

    def set_amplitude_normalization(self, normalization: str) -> None:
        self.controller.select_amplitude_normalization(normalization)
        self._refresh()


__all__ = [
    "ModalSpectrumPanel",
    "SpectrumComparisonController",
]
