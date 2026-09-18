"""Manual playback of RGB-refined shapes with their saved per-video responses."""
from __future__ import annotations

import math
import threading

import numpy as np
import torch

from modal_gaussians.motion.common.rotations import rotate_gaussian_quaternions
from modal_gaussians.motion.neural.rgb_refinement import load_rgb_refinement
from modal_gaussians.static import load_static_scene, cameras_from_scene_manifest, _load_gsplat_rasterization
from modal_gaussians.vis.playback_panel import add_gui_playback_group
from modal_gaussians.vis.viewer import DRIVE_MANUAL, ModalViserViewer, ViewerCamera

DRIVE_PLAYBACK = "RGB-fitted playback"
INITIAL_SHAPE = "Initial shape"
REFINED_SHAPE = "Refined shape"


class RGBRefinementViewerData:
    def __init__(self, path, device="cuda"):
        self.device = torch.device(device)
        self.manifest, motion = load_rgb_refinement(path)
        self.scene = load_static_scene(self.manifest["static_scene"], self.device).eval()
        self.scene.requires_grad_(False)
        if self.scene.manifest["static_scene_identity"] != self.manifest["static_scene_identity"]:
            raise ValueError("RGB refinement and static scene identities differ")
        self.phi0 = torch.as_tensor(motion["phi0"], dtype=torch.complex64, device=self.device)
        self.phi = torch.as_tensor(motion["phi"], dtype=torch.complex64, device=self.device)
        self.rotation = None if motion["rotation"] is None else torch.as_tensor(
            motion["rotation"], dtype=torch.complex64, device=self.device)
        self.frequencies_hz = torch.as_tensor(motion["frequencies_hz"]).cpu().numpy()
        expected = (len(self.manifest["modes"]), self.scene.foreground.count, 3)
        if (self.phi.shape != expected or self.phi0.shape != expected
                or (self.rotation is not None and self.rotation.shape != expected)
                or self.frequencies_hz.shape != (expected[0],)):
            raise ValueError("RGB motion arrays differ from the scene or mode domain")
        self.coefficients = [torch.as_tensor(c, dtype=torch.complex64, device=self.device)
                             for c in motion["coefficients"]]
        self.times = [torch.as_tensor(t).cpu().numpy() for t in motion["times"]]
        views = self.manifest["views"]
        if not views or len(self.coefficients) != len(views) or len(self.times) != len(views):
            raise ValueError("RGB response and view counts differ")
        references = [c for c in cameras_from_scene_manifest(self.scene.manifest) if c.role == "reference"]
        by_label = {c.label: c for c in references}
        if len(by_label) != len(references) or len({v["label"] for v in views}) != len(views):
            raise ValueError("RGB playback and reference camera labels must be unique")
        cameras = []
        for view, coefficients, times in zip(views, self.coefficients, self.times):
            if (coefficients.shape != (len(view["frame_names"]), expected[0])
                    or times.shape != (len(view["frame_names"]),) or not len(times)
                    or not math.isfinite(view["fps_hz"]) or view["fps_hz"] <= 0):
                raise ValueError("RGB response frames differ from their video metadata")
            camera = by_label.get(view["label"])
            if camera is None:
                raise ValueError(f"Missing RGB reference camera: {view['label']}")
            cameras.append(ViewerCamera(label=view["label"], camera=camera,
                c2w=np.linalg.inv(camera.world_to_camera.detach().cpu().numpy()).astype(np.float64),
                fov=2 * math.atan(0.5 * camera.height / float(camera.K[1, 1])),
                aspect=camera.width / camera.height))
        self.cameras = tuple(cameras)

    def deformed(self, coordinate, *, refined=True, scale=1.0, rotate=True):
        """Both shape choices share the selected drive and fixed rotation field."""
        q = torch.as_tensor(coordinate, dtype=self.phi.dtype, device=self.device) * float(scale)
        if q.shape != (len(self.frequencies_hz),) or not bool(torch.isfinite(q).all()):
            raise ValueError("RGB Viewer requires one finite complex coefficient per mode")
        active = self.scene.foreground.active()
        if not bool(torch.any(q)):
            return active["means"], None
        phi = self.phi if refined else self.phi0
        means = active["means"] + torch.einsum("k,kgd->gd", q, phi).real
        quaternions = None
        if rotate and self.rotation is not None:
            angles = torch.einsum("k,kgd->gd", q, self.rotation).real
            quaternions = rotate_gaussian_quaternions(active["quaternions"], angles)
        return means, quaternions


class RGBRefinementViserViewer(ModalViserViewer):
    """Reuse camera navigation and render scheduling without modal diagnostics."""

    @property
    def views(self):
        return self.data.manifest["views"]

    def _active_frame_count(self):
        return len(self.data.coefficients[self._active_view_index()])

    def _on_playback_view(self, event):
        self.playback[5].value = float(self.views[self._active_view_index()]["fps_hz"])
        super()._on_playback_view(event)

    def _on_drive(self, event):
        manual = self.drive.value == DRIVE_MANUAL
        self.motion_scale.disabled = not manual
        self.playback_scale.disabled = manual
        for mode in self.mode_controls:
            mode["gain"].disabled = not manual
            mode["phase"].disabled = not manual
        self.request_render(event)

    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown("**RGB mode refinement**\n\nInitial and refined shapes share the selected drive "
            "and fixed rotation field. Manual oscillator uses per-mode gain and phase; RGB-fitted "
            "playback uses the video's learned response. Manual time = Timestep / FPS; in playback, "
            "FPS changes playback speed only.")
        self.viewer_resolution = gui.add_slider("Viewer Res", min=64, max=2048, step=1,
                                                initial_value=self._viewer_resolution)
        self.playback_view = gui.add_dropdown("Playback view", options=tuple(v["label"] for v in self.views),
                                             initial_value=self.views[0]["label"])
        fps = float(self.views[0]["fps_hz"])
        self.playback = add_gui_playback_group(self.server, self._active_frame_count(),
            max_fps=max(60., *(float(v["fps_hz"]) for v in self.views)),
            initial_fps=fps, num_frames_getter=self._active_frame_count)
        self.timestep = self.playback[0]
        self.timestep.disabled = False
        self.canonical = gui.add_checkbox("Canonical", False)
        self.shape = gui.add_dropdown("Mode shape", options=(REFINED_SHAPE, INITIAL_SHAPE),
                                      initial_value=REFINED_SHAPE)
        self.drive = gui.add_dropdown("Drive", options=(DRIVE_MANUAL, DRIVE_PLAYBACK),
                                      initial_value=DRIVE_MANUAL)
        self.motion_scale = gui.add_slider("Motion scale", min=0., max=1., step=.001, initial_value=.04)
        self.playback_scale = gui.add_slider("Playback scale", min=0., max=2., step=.01,
                                             initial_value=1., disabled=True)
        self.rotate_ellipsoids = gui.add_checkbox("Rotate Gaussian ellipsoids", self.data.rotation is not None,
                                                 disabled=self.data.rotation is None)
        disable_all = gui.add_button("Turn off all modes")
        self.mode_controls = tuple(dict(
            frequency=float(frequency),
            enabled=gui.add_checkbox(f"Mode {index} ({frequency:.3f} Hz) enable", True),
            gain=gui.add_slider(f"Mode {index} gain", min=0., max=5., step=.01, initial_value=1.),
            phase=gui.add_slider(f"Mode {index} phase", min=-math.pi, max=math.pi, step=.01, initial_value=0.),
        ) for index, frequency in enumerate(self.data.frequencies_hz))
        for mode in self.mode_controls:
            for name in ("enabled", "gain", "phase"):
                mode[name].on_update(self.request_render)
        disable_all.on_click(self._disable_all_modes)
        self.hide_background = gui.add_checkbox("Hide background", False,
                                                disabled=self.data.scene.background.count == 0)
        for handle in (self.viewer_resolution, self.timestep, self.shape, self.motion_scale,
                       self.playback_scale, self.playback[5], self.rotate_ellipsoids, self.hide_background):
            handle.on_update(self.request_render)
        self.drive.on_update(self._on_drive)
        self.playback_view.on_update(self._on_playback_view)
        self.canonical.on_update(self._on_canonical)
        self._build_camera_controls()
        self._build_modal_comparison()

    def _build_modal_comparison(self):
        gui = self.server.gui
        self._modal_comparison = None
        self._modal_comparison_lock = threading.Lock()
        self._modal_images_loaded = False
        self.modal_image_window = gui.add_panel()
        with self.modal_image_window.add_tab("Modal images"):
            gui.add_markdown("**Modal image comparison**\n\nLeft: **Input** | Middle: "
                "**Initial reconstruction** | Right: **Refined reconstruction**\n\n"
                "Hue = phase; brightness = amplitude, with one shared scale. "
                "Fixed reference-camera projections of mode shapes; independent of playback and manual gain.")
            self.modal_image_view = gui.add_dropdown("Comparison view",
                options=tuple(v["label"] for v in self.views), initial_value=self.views[0]["label"])
            self._modal_mode_labels = tuple(f"{i}: {f:.3f} Hz" for i, f in enumerate(self.data.frequencies_hz))
            self.modal_image_mode = gui.add_dropdown("Comparison frequency", options=self._modal_mode_labels,
                                                     initial_value=self._modal_mode_labels[0])
            self.modal_image_component = gui.add_dropdown("Component", options=("U", "V"), initial_value="U")
            self.load_modal_images = gui.add_button("Load modal image comparison")
            self.modal_image_status = gui.add_markdown("Load once; view, frequency and component selections then update the comparison.")
            self.modal_images = gui.add_image(np.zeros((1, 3, 3), dtype=np.uint8),
                label="Input | Initial reconstruction | Refined reconstruction", format="jpeg", jpeg_quality=95)
            self.modal_images.visible = False
        self.modal_image_window.float(x=16., y=16., width=1280., height=680.)
        self.load_modal_images.on_click(self._update_modal_comparison)

        def selection_changed(event):
            if self._modal_images_loaded:
                self._update_modal_comparison(event)

        for handle in (self.modal_image_view, self.modal_image_mode, self.modal_image_component):
            handle.on_update(selection_changed)

    def _update_modal_comparison(self, event=None):
        # Only a user-requested view/mode is projected; playback never calls this.
        with self._modal_comparison_lock:
            self.load_modal_images.disabled = True
            self.modal_images.visible = False
            self.modal_image_status.content = "Loading selected modal images..."
            try:
                if self._modal_comparison is None:
                    from modal_gaussians.vis.rgb_modal_comparison import RGBModalComparison
                    self._modal_comparison = RGBModalComparison(self.data)
                image, status = self._modal_comparison.compare(
                    str(self.modal_image_view.value), self._modal_mode_labels.index(self.modal_image_mode.value),
                    ("U", "V").index(self.modal_image_component.value))
                self.modal_images.image = image
                self.modal_image_status.content = status
                self.modal_images.visible = True
                self._modal_images_loaded = True
            except Exception as error:
                self.modal_image_status.content = f"**Modal image comparison unavailable:** {error}"
            finally:
                self.load_modal_images.disabled = False

    @torch.inference_mode()
    def _render(self, client):
        if self.drive.value == DRIVE_MANUAL:
            coordinate, scale = self._manual_coordinate()
        else:
            coordinate = self.data.coefficients[self._active_view_index()][int(self.timestep.value)]
            scale = float(self.playback_scale.value)
        coordinate = torch.as_tensor(coordinate, dtype=self.data.phi.dtype, device=self.data.device)
        coordinate = coordinate * torch.tensor([m["enabled"].value for m in self.mode_controls],
                                                device=self.data.device)
        means, quaternions = self.data.deformed(coordinate,
            refined=self.shape.value == REFINED_SHAPE,
            scale=0. if self.canonical.value else scale,
            rotate=bool(self.rotate_ellipsoids.value))
        rendered = self.data.scene.render_deformed(self._render_camera(client), means,
            foreground_quaternions=quaternions, include_background=not self.hide_background.value)["rgb"]
        return rendered.clamp(0, 1).mul(255).round().byte().cpu().numpy()


def run_rgb_refinement_viewer(*, refinement_dir, work_dir, host="127.0.0.1", port=8080,
                              viewer_resolution=2048):
    _load_gsplat_rasterization()
    data = RGBRefinementViewerData(refinement_dir)
    viewer = RGBRefinementViserViewer(data, work_dir=work_dir, host=host, port=port,
                                      viewer_resolution=viewer_resolution)
    viewer.wait()
