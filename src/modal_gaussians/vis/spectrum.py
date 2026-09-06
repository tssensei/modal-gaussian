"""Original-versus-reconstructed modal spectrum controls for Viser."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import viser
import viser.uplot

from modal_gaussians.flow.storage import DenseArray, read_pixels

from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    flow_artifact_identity,
    load_flow_analysis_artifact,
)
from modal_gaussians.measurements import load_gaussian_measurements
from modal_gaussians.modes import Complex2DModesArtifact, load_complex_2d_modes
from modal_gaussians.motion.common.mode_mapping import resolve_source_mode_slots
from modal_gaussians.result import ModalResultArtifact
from modal_gaussians.topology import load_observation_topology


PIXEL_CHUNK_SIZE = 4096
PREVIEW_PERCENTILE = 99.0


@dataclass(frozen=True)
class SpectrumViewState:
    """Cache the compact quantities needed by one spectrum-panel view."""

    index: int
    label: str
    flow: FlowAnalysisArtifact
    pixels_xy: np.ndarray
    reference_rgb: np.ndarray
    raw_frequencies_hz: np.ndarray
    raw_power: np.ndarray
    reconstructed_power: np.ndarray
    alphas: np.ndarray
    alpha_identifiable: np.ndarray
    frequency_limits: tuple[float, float]


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


def _read_reference_rgb(flow: FlowAnalysisArtifact) -> np.ndarray:
    """Load the exact reference RGB named by a flow artifact."""

    sequence = flow.manifest["inputs"]["sequence"]
    image_directory = Path(sequence["image_directory"]).expanduser().resolve(
        strict=True
    )
    frame_name = str(flow.manifest["reference_frame_name"])
    image_path = image_directory / f"{frame_name}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load flow reference image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != flow.arrays.mask_union.shape:
        raise ValueError("Flow reference image shape differs from flow arrays")
    return rgb


def _mean_image_plane_power(
    spectrum: DenseArray, pixels_xy: np.ndarray
) -> np.ndarray:
    """Average vector amplitude over candidate pixels without a large temporary."""

    result = np.zeros(spectrum.shape[0], dtype=np.float64)
    for lower in range(0, len(pixels_xy), PIXEL_CHUNK_SIZE):
        pixels = pixels_xy[lower : lower + PIXEL_CHUNK_SIZE]
        values = read_pixels(spectrum, slice(None), pixels)
        result += np.sum(
            np.sqrt(np.abs(values[..., 0]) ** 2 + np.abs(values[..., 1]) ** 2),
            axis=1,
            dtype=np.float64,
        )
    result /= float(len(pixels_xy))
    if not np.isfinite(result).all():
        raise ValueError("Raw modal spectrum contains non-finite power")
    return result.astype(np.float32)


def _completed_dense_modes(completed: Any) -> Complex2DModesArtifact:
    """Bind display targets to the method's declared dense modal source."""

    if completed.manifest.get("version") in (8, 9):
        method = ("neural_fragment_motion_propagation" if completed.manifest["version"] == 9
                  else "neural_complex_displacement_field")
        if completed.manifest.get("completion_method") != method:
            raise ValueError("Unsupported neural spectrum source contract")
        source = completed.manifest.get("complex_2d_modes")
        if not isinstance(source, str) or not source:
            raise ValueError("Neural completed modes do not identify dense 2D modes")
        dense = load_complex_2d_modes(source)
        if dense.manifest["complex_2d_modes_identity"] != completed.manifest.get("complex_2d_modes_identity"):
            raise ValueError("Neural Viewer dense-mode identity differs")
        if dense.manifest["topology_identity"] != completed.manifest.get("topology_identity"):
            raise ValueError("Neural Viewer dense-mode topology identity differs")
    else:
        measurement_path = Path(completed.manifest["measurements"])
        measurements = load_gaussian_measurements(measurement_path)
        dense = load_complex_2d_modes(measurements.manifest["complex_2d_modes"])
        topology = load_observation_topology(measurements.manifest["topology"])
        if measurements.manifest["gaussian_measurements_identity"] != completed.manifest["gaussian_measurements_identity"]:
            raise ValueError("Viewer measurements differ from completed modes")
        if dense.manifest["complex_2d_modes_identity"] != measurements.manifest["complex_2d_modes_identity"]:
            raise ValueError("Viewer dense-mode identity differs from measurements")
        if dense.manifest["modes"] != measurements.manifest["modes"]:
            raise ValueError("Viewer dense-mode order differs from measurements")
        if topology.manifest["topology_identity"] != completed.manifest["topology_identity"]:
            raise ValueError("Viewer topology identity differs from modal result")
    resolve_source_mode_slots(completed.manifest["modes"], dense.manifest["modes"])
    return dense


class SpectrumComparisonController:
    """Compare bound dense flow spectra with rendered 3D modal projections."""

    def __init__(self, result: ModalResultArtifact) -> None:
        self.result = result
        self.frequencies_hz = np.asarray(
            [mode["frequency_hz"] for mode in result.manifest["modes"]],
            dtype=np.float64,
        )
        completed = result.completed_modes
        dense = _completed_dense_modes(completed)
        self._source_mode_slots = resolve_source_mode_slots(
            result.manifest["modes"], dense.manifest["modes"]
        )
        self.dense_modes: Complex2DModesArtifact = dense
        self._states: dict[str, SpectrumViewState] = {}
        self._global_magnitude_cache: dict[tuple[str, int], float] = {}
        self.component_index = 0
        self.reconstructed_index = 0
        self.amplitude_normalization = "per mode"

        flow_sources = result.coordinates.manifest.get("flow_artifacts")
        if not isinstance(flow_sources, list):
            raise ValueError("Modal coordinates do not bind flow artifacts")
        design_views = result.rendered_design.manifest["views"]
        if not (
            len(flow_sources)
            == len(design_views)
            == len(dense.manifest["views"])
        ):
            raise ValueError("Viewer spectrum source view counts differ")
        self.available_view_ids = tuple(view["label"] for view in design_views)
        for index, (source, design_view, dense_view) in enumerate(
            zip(flow_sources, design_views, dense.manifest["views"])
        ):
            flow = load_flow_analysis_artifact(source)
            identity = flow_artifact_identity(flow)
            if (
                design_view["index"] != index
                or dense_view["index"] != index
                or design_view["label"] != dense_view["label"]
                or identity != design_view["flow_identity"]
                or identity != dense_view["flow_identity"]
            ):
                raise ValueError("Viewer spectrum view order or flow identity differs")
            self._states[design_view["label"]] = self._prepare_view(
                index, design_view["label"], flow
            )
        self.select_view(self.available_view_ids[0])

    def _prepare_view(
        self, index: int, label: str, flow: FlowAnalysisArtifact
    ) -> SpectrumViewState:
        """Fit one complex display alpha per mode for one rendered view."""

        design = self.result.rendered_design
        offsets = design.samples["view_sample_offsets"]
        lower, upper = int(offsets[index]), int(offsets[index + 1])
        pixels = np.asarray(
            design.samples["sample_pixels_xy"][lower:upper], dtype=np.int64
        )
        if len(pixels) == 0:
            raise ValueError(f"Rendered design has no samples for {label!r}")
        matrix = design.design[lower:upper]
        dense = self.dense_modes.view_modes[index]
        alphas = np.zeros(len(self.frequencies_hz), dtype=np.complex64)
        identifiable = np.zeros(len(self.frequencies_hz), dtype=bool)
        reconstructed_power = np.zeros(len(self.frequencies_hz), dtype=np.float32)
        for mode_index in range(len(self.frequencies_hz)):
            projected = (
                np.asarray(matrix[:, :, 2 * mode_index], dtype=np.float64)
                - 1j
                * np.asarray(matrix[:, :, 2 * mode_index + 1], dtype=np.float64)
            )
            raw = np.asarray(
                dense[self._source_mode_slots[mode_index], pixels[:, 1], pixels[:, 0], :],
                dtype=np.complex128,
            )
            denominator = float(np.vdot(projected, projected).real)
            if (
                not math.isfinite(denominator)
                or denominator <= np.finfo(np.float64).tiny
            ):
                if self.result.completed_modes.manifest.get("version") in (8, 9):
                    if math.isfinite(denominator):
                        # Unresolved neural modes remain explicit zero fields.
                        continue
                raise ValueError(
                    "Rendered projected mode has zero image-plane energy for "
                    f"mode {mode_index} at "
                    f"{float(self.frequencies_hz[mode_index]):.9g} Hz"
                )
            alpha = np.vdot(projected, raw) / denominator
            if not math.isfinite(alpha.real) or not math.isfinite(alpha.imag):
                raise ValueError(f"Viewer fitted alpha for {label!r} is non-finite")
            reconstructed = alpha * projected
            alphas[mode_index] = np.complex64(alpha)
            identifiable[mode_index] = True
            reconstructed_power[mode_index] = np.float32(
                np.mean(
                    np.sqrt(
                        np.abs(reconstructed[:, 0]) ** 2
                        + np.abs(reconstructed[:, 1]) ** 2
                    )
                )
            )
        raw_frequencies = np.fft.rfftfreq(
            flow.arrays.flow.shape[0], d=1.0 / float(flow.manifest["fps_hz"])
        )
        raw_power = _mean_image_plane_power(flow.arrays.spectrum, pixels)
        selected_min = float(np.min(self.frequencies_hz))
        selected_max = float(np.max(self.frequencies_hz))
        raw_step = float(raw_frequencies[1] - raw_frequencies[0])
        margin = 0.05 * max(selected_max - selected_min, raw_step)
        limits = (
            max(float(raw_frequencies[0]), selected_min - margin),
            min(float(raw_frequencies[-1]), selected_max + margin),
        )
        return SpectrumViewState(
            index=index,
            label=label,
            flow=flow,
            pixels_xy=pixels,
            reference_rgb=_read_reference_rgb(flow),
            raw_frequencies_hz=raw_frequencies,
            raw_power=raw_power,
            reconstructed_power=reconstructed_power,
            alphas=alphas,
            alpha_identifiable=identifiable,
            frequency_limits=limits,
        )

    def select_view(self, label: str) -> None:
        """Switch all spectrum and image products to one bound fixed view."""

        if label not in self._states:
            raise ValueError(f"Unknown Viewer spectrum view: {label!r}")
        self.view_id = label
        self.state = self._states[label]
        self._refresh_products()

    def select_mode(self, index: int) -> None:
        """Select one greedy mode slot without changing mode order."""

        if not 0 <= int(index) < len(self.frequencies_hz):
            raise ValueError("Viewer spectrum mode index is outside the result")
        self.reconstructed_index = int(index)
        self._refresh_products()

    def select_component(self, component: str) -> None:
        """Select the U or V complex image component."""

        value = str(component).upper()
        if value not in ("U", "V"):
            raise ValueError(f"Unknown modal image component: {component!r}")
        self.component_index = 0 if value == "U" else 1
        self._refresh_products()

    def select_amplitude_normalization(self, normalization: str) -> None:
        """Choose per-mode or shared entire-spectrum phase-image brightness."""

        if normalization not in ("per mode", "entire spectrum"):
            raise ValueError(f"Unknown amplitude normalization: {normalization!r}")
        self.amplitude_normalization = normalization
        self._refresh_products()

    def _selected_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact raw and rendered reconstructed modes at panel pixels."""

        index = self.reconstructed_index
        pixels = self.state.pixels_xy
        raw = np.asarray(
            self.dense_modes.view_modes[self.state.index][
                self._source_mode_slots[index], pixels[:, 1], pixels[:, 0], :
            ],
            dtype=np.complex64,
        )
        design = self.result.rendered_design
        offsets = design.samples["view_sample_offsets"]
        lower, upper = int(offsets[self.state.index]), int(offsets[self.state.index + 1])
        matrix = design.design[lower:upper]
        projected = (
            np.asarray(matrix[:, :, 2 * index], dtype=np.float32)
            - 1j * np.asarray(matrix[:, :, 2 * index + 1], dtype=np.float32)
        ).astype(np.complex64)
        reconstructed = self.state.alphas[index] * projected
        return raw, reconstructed

    def _entire_spectrum_magnitude_hi(self) -> float:
        """Compute the shared raw/reconstructed percentile scale lazily."""

        key = (self.view_id, self.component_index)
        cached = self._global_magnitude_cache.get(key)
        if cached is not None:
            return cached
        pixels = self.state.pixels_xy
        spectrum = self.state.flow.arrays.spectrum
        maximum = 0.0
        for lower in range(0, spectrum.shape[0], 8):
            block = read_pixels(
                spectrum, slice(lower, lower + 8), pixels, self.component_index
            )
            maximum = max(maximum, float(np.percentile(np.abs(block), PREVIEW_PERCENTILE)))
        for mode_index in range(len(self.frequencies_hz)):
            previous = self.reconstructed_index
            self.reconstructed_index = mode_index
            _, reconstructed = self._selected_modes()
            self.reconstructed_index = previous
            maximum = max(
                maximum,
                float(
                    np.percentile(
                        np.abs(reconstructed[:, self.component_index]),
                        PREVIEW_PERCENTILE,
                    )
                ),
            )
        if not math.isfinite(maximum) or maximum <= 0.0:
            maximum = 1.0
        self._global_magnitude_cache[key] = maximum
        return maximum

    def _modal_image(self, values: np.ndarray, magnitude_hi: float) -> np.ndarray:
        """Overlay phase-HSV samples on the bound reference RGB image."""

        base = self.state.reference_rgb.astype(np.float32) / 255.0
        output = 0.35 * base
        colors = _hsv_rgb(values, magnitude_hi)
        x = self.state.pixels_xy[:, 0]
        y = self.state.pixels_xy[:, 1]
        height, width = output.shape[:2]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx = np.clip(x + dx, 0, width - 1)
                yy = np.clip(y + dy, 0, height - 1)
                output[yy, xx] = colors
        return np.clip(np.rint(255.0 * output), 0.0, 255.0).astype(np.uint8)

    def _refresh_products(self) -> None:
        """Refresh selected modal images, brightness, and status text."""

        raw, reconstructed = self._selected_modes()
        raw_values = raw[:, self.component_index]
        reconstructed_values = reconstructed[:, self.component_index]
        if self.amplitude_normalization == "per mode":
            magnitude_hi = float(
                np.percentile(
                    np.concatenate((np.abs(raw_values), np.abs(reconstructed_values))),
                    PREVIEW_PERCENTILE,
                )
            )
            if not math.isfinite(magnitude_hi) or magnitude_hi <= 0.0:
                magnitude_hi = 1.0
        else:
            magnitude_hi = self._entire_spectrum_magnitude_hi()
        self.modal_image_magnitude_hi = magnitude_hi
        self.raw_modal_image = self._modal_image(raw_values, magnitude_hi)
        self.reconstructed_modal_image = self._modal_image(
            reconstructed_values, magnitude_hi
        )
        frequency = float(self.frequencies_hz[self.reconstructed_index])
        raw_index = int(np.argmin(np.abs(self.state.raw_frequencies_hz - frequency)))
        component = "U" if self.component_index == 0 else "V"
        identifiable = bool(self.state.alpha_identifiable[self.reconstructed_index])
        self.status = (
            f"**View:** `{self.view_id}` &nbsp; **candidate pixels:** "
            f"{len(self.state.pixels_xy)}  \n"
            f"**Selected:** {frequency:.6f} Hz &nbsp; **raw rFFT bin:** "
            f"{self.state.raw_frequencies_hz[raw_index]:.6f} Hz &nbsp; "
            f"**component:** {component}  \n"
            f"**Amplitude normalization:** `{self.amplitude_normalization}`"
            + ("" if identifiable else "  \n**Reconstruction unavailable:** zero projection energy.")
        )

    @property
    def raw_frequencies_hz(self) -> np.ndarray:
        return self.state.raw_frequencies_hz

    @property
    def raw_power(self) -> np.ndarray:
        return self.state.raw_power

    @property
    def reconstructed_power(self) -> np.ndarray:
        return self.state.reconstructed_power

    @property
    def frequency_limits(self) -> tuple[float, float]:
        return self.state.frequency_limits

    def current_modal_phase_display_context(
        self,
        mode_index: int,
        component_index: int,
        amplitude_normalization: str,
    ) -> tuple[str, complex, bool, float]:
        """Return the exact view alpha and brightness shared with 3D coloring."""

        if (
            int(mode_index) != self.reconstructed_index
            or int(component_index) != self.component_index
            or amplitude_normalization != self.amplitude_normalization
        ):
            raise ValueError("Viewer phase controls and spectrum panel are not synchronized")
        return (
            self.view_id,
            complex(self.state.alphas[self.reconstructed_index]),
            bool(self.state.alpha_identifiable[self.reconstructed_index]),
            float(self.modal_image_magnitude_hi),
        )


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
    """Mirror the old floating Viser spectrum panel against new artifacts."""

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
        self.component = server.gui.add_dropdown(
            "Modal image component",
            options=("U", "V"),
            initial_value="U" if controller.component_index == 0 else "V",
        )
        self.normalization = server.gui.add_dropdown(
            "Amplitude normalization",
            options=("per mode", "entire spectrum"),
            initial_value=controller.amplitude_normalization,
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
        maximum = self._shared_power_max()
        self.raw_plot = server.gui.add_uplot(
            data=_spectrum_plot_data(
                controller.raw_frequencies_hz,
                controller.raw_power,
                float(controller.frequencies_hz[controller.reconstructed_index]),
                maximum,
            ),
            series=self._series("Original power", "#4c9aff"),
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
            series=self._series("Reconstructed power", "#ff9f43"),
            title=f"Reconstructed {controller.view_id} spectrum",
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
            label="Reconstructed modal image",
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

        @self.normalization.on_update
        def _(_) -> None:
            if not self._updating:
                on_normalization_selected(str(self.normalization.value))

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

    def _shared_power_max(self) -> float:
        lower, upper = self.controller.frequency_limits
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
                time=False, range=self.controller.frequency_limits
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
            f"Reconstructed {self.controller.view_id} spectrum"
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
