"""Shared-FFT and saved-mode visualization, without legacy flow reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import math
from typing import Callable

import cv2
import numpy as np
import viser
import viser.uplot

from modal_gaussians.common.cache import identity
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.spectrum.cache import load_spectrum, read_curves, read_region
from modal_gaussians.vis.projections import ViewerProjections

PREVIEW_PERCENTILE = 99.0
NORMALIZATIONS = ("per mode", "all saved modes")


@dataclass
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
    display_pixels_xy: np.ndarray


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
    """Display saved modal images and projections with their training alphas."""

    def __init__(self, result, *, projections=None):
        self.result = result
        self.projections = projections or ViewerProjections(result)
        self.frequencies_hz = np.asarray([m.frequency for m in result.modes], np.float64)
        self.design_views = result.observation_views
        self._lock = threading.RLock()
        self._executor = None
        self._pending = set()
        self._errors = {}
        self._on_update = None
        self._closed = False
        self.available_view_ids = tuple(v["label"] for v in self.design_views)
        if len(set(self.available_view_ids)) != len(self.available_view_ids):
            raise ValueError("Spectrum view labels must be unique")
        self._images = {label: [] for label in self.available_view_ids}
        self._alphas = np.zeros((len(self.frequencies_hz), len(self.design_views)), np.complex64)
        self._identifiable = np.zeros(self._alphas.shape, bool)
        self.cache = None
        for k, mode in enumerate(result.modes):
            model, slot, arrays = mode.artifact.manifest, mode.slot, mode.artifact.arrays
            alphas, identifiable = arrays["alphas"][slot], arrays["alpha_identifiable_mask"][slot]
            dense_views = mode.modal_views
            model_views = {v["label"]: i for i, v in enumerate(model["views"])}
            for v, design_view in enumerate(self.design_views):
                label = design_view["label"]
                view = dense_views[label]
                if (view["camera_identity"] != design_view["camera_identity"]
                        or view["shape_hw"] != design_view["shape_hw"]):
                    raise ValueError("Modal image camera/shape differs from rendered design")
                selected = view["selected_source"]
                exported = mode.exports[label]
                if (identity(exported) != selected["manifest_identity"]
                        or exported.get("format") != "modal_gaussians.spectrum_selected_frequency"
                        or exported.get("status") != "complete"):
                    raise ValueError("Modal image must bind a complete selected-frequency export")
                motion_reference = design_view.get("motion_reference", {
                    "reference_frame_name": design_view["flow_reference_frame_name"],
                    "reference_frame_index": design_view["flow_reference_frame_index"]})
                if (not math.isclose(exported["frequency_hz"], self.frequencies_hz[k], rel_tol=0, abs_tol=1e-9)
                        or exported["reference_frame_name"] != motion_reference["reference_frame_name"]
                        or exported["reference_frame_index"] != motion_reference["reference_frame_index"]
                        or exported.get("reference_selection", {}).get("identity") != motion_reference.get("selection_identity")
                        or exported["fps_hz"] != design_view["fps_hz"]
                        or exported["modes_shape"] != [1, *design_view["shape_hw"], 2]):
                    raise ValueError("Modal export frequency, reference timing or shape differs")
                grid = exported["spectrum_source"]
                if self.cache is None:
                    self.cache = load_spectrum(grid["path"])
                if (grid["identity"] != self.cache.manifest["spectrum_identity"]
                        or resolve_path(grid["path"]) != self.cache.path
                        or grid["fft_length"] != self.cache.manifest["fft_length"]):
                    raise ValueError("Modal images must share the exact saved FFT cache")
                index = grid["bin_index"]
                if (type(index) is not int or not 0 <= index < len(self.cache.frequencies)
                        or not math.isclose(self.cache.frequencies[index], self.frequencies_hz[k], rel_tol=0, abs_tol=1e-9)):
                    raise ValueError("Modal frequency is not its exact shared FFT bin")
                cache_view = next(item for item in self.cache.manifest["views"] if item["label"] == label)
                if (resolve_path(exported["source_flow_path"]) != resolve_path(cache_view["source_path"])
                        or exported["frames"] != cache_view["source_manifest"]["frames"]):
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
        self.full_spectrum_available = True
        self.image_regions = (('Cached region', 'Model support'))
        self.original_image_region = self.image_regions[0]
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
        value = self._available(k, self.available_view_ids[v])
        if value is None:
            return None
        return (self._alphas[k, v] * value.values if self._identifiable[k, v]
                else np.zeros_like(value.values))

    def _record_projection(self, label, k):
        state = self._states[label]
        if np.isfinite(state.reconstructed_power[k]):
            return
        value = self._available(k, label)
        if value is None:
            return
        raw = self._raw(label, k, value.pixels)
        projected = self._projection(state.index, k)
        state.reconstructed_power[k] = np.linalg.norm(projected, axis=1).mean()
        state.magnitude_hi[:] = np.maximum(state.magnitude_hi,
            np.percentile(np.concatenate((np.abs(raw), np.abs(projected))), PREVIEW_PERCENTILE, axis=0))

    def _prepare_view(self, label):
        v = self.available_view_ids.index(label)
        h, w = self.design_views[v]["shape_hw"]
        cache_index = next(i for i, item in enumerate(self.cache.manifest["views"]) if item["label"] == label)
        record = self.cache.manifest["views"][cache_index]
        y, x = np.nonzero(read_region(self.cache, cache_index, "selected_box"))
        selection = np.column_stack((x, y))
        reference_path = resolve_path(self.cache.path / record["reference_image"], strict=True)
        reference = cv2.imread(str(reference_path))
        if reference is None or reference.shape != (h, w, 3):
            raise ValueError("Cached spectrum reference image has invalid dimensions")
        f = self.cache.frequencies
        spacing = float(np.min(np.diff(np.sort(f)))) if len(f) > 1 else max(float(abs(f[0])), 1e-3)
        margin = 0.05 * max(float(np.ptp(self.frequencies_hz)), spacing)
        limits = (max(0., float(self.frequencies_hz.min()) - margin), float(self.frequencies_hz.max()) + margin)
        raw_power = np.full(len(f), np.nan, np.float32)
        raw_power = read_curves(self.cache, cache_index)["selected_box"]
        limits = (max(float(f[0]), limits[0]), min(float(f[-1]), limits[1]))
        empty = np.empty((0, 2), np.int64)
        return SpectrumViewState(v, label, empty, cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), f,
            raw_power, np.full(len(self.frequencies_hz), np.nan, np.float32), self._alphas[:, v],
            self._identifiable[:, v], limits, selection, np.zeros(2), empty)

    def start(self, on_update):
        """Start only after the 3D viewer and panel handles exist."""
        self._on_update = on_update
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="modal-projection")
        self._refresh_products()

    def close(self):
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()

    def _available(self, k, label):
        try:
            return self.projections.get(k, label)
        except (OSError, ValueError, KeyError) as error:
            self._errors[k, label] = str(error)
            return None

    def _request(self, k, label):
        key = k, label
        if self._closed or self._executor is None or key in self._pending or key in self._errors:
            return
        self._pending.add(key)
        def compute():
            try:
                self.projections.get(k, label, compute=True)
            except Exception as error:
                self._errors[key] = str(error)
            finally:
                with self._lock:
                    self._pending.discard(key)
                    if not self._closed:
                        if key not in self._errors:
                            self._record_projection(label, k)
                        # Read current selection here; an old job never restores it.
                        self._refresh_products()
                        if self._on_update is not None:
                            self._on_update()
        self._executor.submit(compute)

    def complete_view(self):
        with self._lock:
            for k in range(len(self.frequencies_hz)):
                self._errors.pop((k, self.view_id), None)
                if self._available(k, self.view_id) is None:
                    self._request(k, self.view_id)
            self._refresh_products()

    def select_view(self, label):
        with self._lock:
            if label not in self.available_view_ids:
                raise ValueError(f"Unknown spectrum view: {label!r}")
            if label not in self._states:
                self._states[label] = self._prepare_view(label)
                for k in range(len(self.frequencies_hz)):
                    self._record_projection(label, k)
            self.view_id, self.state = label, self._states[label]
            if self.amplitude_normalization == "all saved modes":
                self.complete_view()
            self._refresh_products()

    def select_mode(self, index):
        with self._lock:
            if not 0 <= int(index) < len(self.frequencies_hz):
                raise ValueError("Mode index is outside the saved bank")
            self.reconstructed_index = int(index)
            self._refresh_products()

    def select_component(self, component):
        with self._lock:
            if component.upper() not in ("U", "V"):
                raise ValueError("Modal component must be U or V")
            self.component_index = 0 if component.upper() == "U" else 1
            self._refresh_products()

    def select_amplitude_normalization(self, normalization):
        with self._lock:
            if normalization not in NORMALIZATIONS:
                raise ValueError("Unknown modal image normalization")
            self.amplitude_normalization = normalization
            if normalization == "all saved modes":
                self.complete_view()
            self._refresh_products()

    def select_original_image_region(self, region):
        with self._lock:
            if region not in self.image_regions:
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
        with self._lock:
            self._refresh_current()

    def _refresh_current(self):
        k, c = self.reconstructed_index, self.component_index
        value = self._available(k, self.view_id)
        ready = value is not None
        if ready:
            pixels = value.pixels
            self._record_projection(self.view_id, k)
            projected = self._projection(self.state.index, k)[:, c]
            raw = self._raw(self.view_id, k, pixels)[:, c]
            all_ready = np.isfinite(self.state.reconstructed_power).all()
            high = (float(self.state.magnitude_hi[c]) if self.amplitude_normalization == "all saved modes" and all_ready
                    else float(np.percentile(np.concatenate((np.abs(raw), np.abs(projected))), PREVIEW_PERCENTILE)))
            h, w = self.design_views[self.state.index]["shape_hw"]
            support = np.zeros((h, w), np.uint8)
            support[pixels[:, 1], pixels[:, 0]] = 1
            y, x = np.nonzero(cv2.dilate(support, np.ones((3, 3), np.uint8)))
            self.state.pixels_xy = pixels
            self.state.display_pixels_xy = np.column_stack((x, y))
        else:
            pixels = np.empty((0, 2), np.int64)
            projected = np.empty(0, np.complex64)
            high = 1.0
            self._request(k, self.view_id)
        self.modal_image_magnitude_hi = high if math.isfinite(high) and high > 0 else 1.0
        if self.original_image_region == "Cached region":
            original_pixels = self.state.selection_pixels_xy
        elif self.original_image_region == "Full image" or not ready:
            y, x = np.indices(self.state.reference_rgb.shape[:2])
            original_pixels = np.column_stack((x.ravel(), y.ravel()))
        else:
            original_pixels = self.state.display_pixels_xy
        raw = self._raw(self.view_id, k, original_pixels)[:, c]
        self.raw_modal_image = self._modal_image(raw, self.modal_image_magnitude_hi, original_pixels)
        self.reconstructed_modal_image = self._modal_image(projected, self.modal_image_magnitude_hi, pixels, radius=1)
        count = int(np.isfinite(self.state.reconstructed_power).sum())
        error = self._errors.get((k, self.view_id))
        self.status = (
            f"**View:** {self.view_id} · **Frequency:** {self.frequencies_hz[k]:.6f} Hz · "
            f"**Component:** {'U' if c == 0 else 'V'}  \n"
            f"**Projections available:** {count}/{len(self.frequencies_hz)}. Missing frequencies are gaps.  \n"
            "**Projection:** fixed 3D mode with saved training view alignment; no display refit.  \n"
            + ('**Curve regions:** input = cached analysis region; projection = per-mode model-support samples.  \n')
            + (f"**Projection failed:** {error}" if error else ("" if ready else "**Projection pending.**"))
            + ("" if self.state.alpha_identifiable[k] else "  \n**Projection unavailable:** saved view alignment is unidentifiable.")
            + ("  \n**Shared brightness pending:** waiting for all frequencies." if
               self.amplitude_normalization == "all saved modes" and count < len(self.frequencies_hz) else "")
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
                bool(self.state.alpha_identifiable[mode_index]) and self._available(mode_index, self.view_id) is not None,
                self.modal_image_magnitude_hi)


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
        self.view = server.gui.add_dropdown(
            "Spectrum view",
            options=controller.available_view_ids,
            initial_value=controller.view_id,
        )
        self.frequency_range = server.gui.add_dropdown(
            "Frequency range",
            options=(('full spectrum', 'selected modes')),
            initial_value='full spectrum',
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
            options=controller.image_regions,
            initial_value=controller.original_image_region,
        )
        self.mode = server.gui.add_slider(
            "Selected mode",
            min=0,
            max=len(controller.frequencies_hz) - 1,
            step=1,
            initial_value=controller.reconstructed_index,
        )
        self.frequency = server.gui.add_number(
            "Selected frequency (Hz)",
            initial_value=float(controller.frequencies_hz[controller.reconstructed_index]),
            disabled=True,
        )
        solo = server.gui.add_button("Solo selected mode")
        enable_all = server.gui.add_button("Enable all modes")
        complete = server.gui.add_button("Complete projections for this view")
        @complete.on_click
        def _(_):
            controller.complete_view()
            self._refresh()
        self.status = server.gui.add_markdown(controller.status)
        server.gui.add_markdown(
            'Input: saved full FFT curve over its cached analysis region. Projection: saved frequencies on model-support pixels, with training alignment. This is a modal projection, not a coefficient-video FFT.'
        )
        self._raw_title = 'Original spectrum'
        maximum = self._shared_power_max()
        frequency = float(controller.frequencies_hz[controller.reconstructed_index])
        self.raw_plot, self.reconstructed_plot = (
            server.gui.add_uplot(data=_spectrum_plot_data(frequencies, power, frequency, maximum),
                series=self._series(label, color), title=title, scales=self._scales(maximum),
                legend=viser.uplot.Legend(show=True), height=260)
            for frequencies, power, label, color, title in self._curves()
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
                on_mode_selected(int(self.mode.value))

        @solo.on_click
        def _(_) -> None:
            on_solo_selected(int(self.mode.value))

        @enable_all.on_click
        def _(_) -> None:
            on_enable_all()

    @staticmethod
    def _series(label: str, color: str) -> tuple[viser.uplot.Series, ...]:
        return (
            viser.uplot.Series(label="Frequency (Hz)"),
            viser.uplot.Series(label=label, stroke=color, width=2, spanGaps=False),
            viser.uplot.Series(label="Selected frequency", stroke="#ff3b30", width=1),
        )

    def _frequency_limits(self) -> tuple[float, float]:
        if self.frequency_range.value == "full spectrum":
            frequencies = self.controller.raw_frequencies_hz
            return float(frequencies[0]), float(frequencies[-1])
        return self.controller.frequency_limits

    def _curves(self):
        """Keep plot creation, scaling and refresh on the same two curve definitions."""
        c = self.controller
        return (
            (c.raw_frequencies_hz, c.raw_power,
             'Cached-region mean amplitude',
             "#4c9aff", f"{self._raw_title} · {c.view_id}"),
            (c.frequencies_hz, c.reconstructed_power, "Model-support projected amplitude",
             "#ff9f43", f"Projected modes {c.view_id} spectrum"),
        )

    def _shared_power_max(self) -> float:
        lower, upper = self._frequency_limits()
        maximum = max(float(np.max(power[(frequencies >= lower) & (frequencies <= upper) & np.isfinite(power)], initial=0.))
                      for frequencies, power, *_ in self._curves())
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
            self.mode.value = index
            self.frequency.value = frequency
            self.view.value = self.controller.view_id
            self.component.value = "U" if self.controller.component_index == 0 else "V"
            self.normalization.value = self.controller.amplitude_normalization
            self.original_region.value = self.controller.original_image_region
        finally:
            self._updating = False
        self.status.content = self.controller.status
        for plot, (frequencies, power, _, _, title) in zip((self.raw_plot, self.reconstructed_plot), self._curves()):
            plot.data = _spectrum_plot_data(frequencies, power, frequency, maximum)
            plot.title = title
            plot.scales = self._scales(maximum)
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
