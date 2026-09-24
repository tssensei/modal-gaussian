"""Select a static scene's motion subject with an oriented 3D box."""
from __future__ import annotations

from modal_gaussians.common.scene_store import resolve_path

import numpy as np
import torch

from modal_gaussians.common.camera_rendering import rasterize_cameras
from modal_gaussians.geometry.scene import _load_gsplat_rasterization, cameras_from_scene_manifest, load_static_scene
from modal_gaussians.geometry.selection import _box_values, load_subject_selection, points_in_box, save_subject_selection
from modal_gaussians.vis.viewer import ModalViserViewer, ViewerCamera


class SubjectSelectionData:
    def __init__(self, scene_dir, selection_path=None, device="cuda"):
        self.device = torch.device(device)
        self.scene_path = resolve_path(scene_dir, strict=True)
        self.scene = load_static_scene(self.scene_path, self.device).eval()
        self.scene.requires_grad_(False)
        self.active = self.scene._active_for("all")
        self.points = self.active["means"].detach().cpu().numpy()
        # Old foreground only places the initial box; selection always includes old background.
        foreground = self.points[:self.scene.foreground.count]
        lower, upper = foreground.min(axis=0), foreground.max(axis=0)
        dimensions = np.maximum((upper - lower) * 1.1, 1e-4)
        self.initial_box = _box_values((lower + upper) / 2, (1, 0, 0, 0), dimensions)
        self.box = (load_subject_selection(selection_path, self.scene)
                    if selection_path is not None else self.initial_box)
        cameras = cameras_from_scene_manifest(self.scene.manifest)
        references = [camera for camera in cameras if camera.role == "reference"] or list(cameras[:1])
        self.cameras = tuple(ViewerCamera.from_camera(camera) for camera in references)

    @torch.inference_mode()
    def render_selection(self, camera, selected, *, highlight=True, only_selected=False):
        """Tint both original partitions with one shared depth order; never edit parameters."""
        active = self.active
        mask = torch.as_tensor(selected, device=self.device, dtype=torch.bool)
        colors = active["colors"]
        if highlight:
            tint = colors.new_tensor((0.15, 0.95, 1.0))
            colors = torch.where(mask[:, None], 0.35 * colors + 0.65 * tint, 0.4 * colors)
        opacities = active["opacities"] * mask if only_selected else active["opacities"]
        rendered, _, _ = rasterize_cameras(_load_gsplat_rasterization(), [camera],
            means=active["means"], quats=active["quaternions"], scales=active["scales"],
            opacities=opacities, colors=colors,
            viewmats=camera.world_to_camera.to(self.device)[None], Ks=camera.K.to(self.device)[None],
            width=camera.width, height=camera.height, packed=False,
            backgrounds=torch.ones((1, 3), device=self.device), render_mode="RGB",
            rasterize_mode="classic", camera_model="pinhole")
        return rendered[0].clamp(0, 1).mul(255).round().byte().cpu().numpy()


class SubjectSelectionViewer(ModalViserViewer):
    """Reuse the existing static camera navigation and coalesced rendering."""

    def _build_gui(self):
        gui = self.server.gui
        self._box_state = self.data.box
        position, wxyz, dimensions = self._box_state
        self._orbit_center = position.copy()
        gui.add_markdown("**3D subject selection**\n\nDrag the axes to move the box; drag the rings to rotate. "
                         "Adjust its local X/Y/Z sizes below. Cyan Gaussians are selected, "
                         "including any from the old background. Selection uses Gaussian centers.")
        self.viewer_resolution = gui.add_slider(
            "Viewer Res", min=64, max=max(2048, self._viewer_resolution), step=1,
            initial_value=self._viewer_resolution)
        self.highlight = gui.add_checkbox("Highlight selection", True)
        self.only_selected = gui.add_checkbox("Only selected", False)
        self.show_box = gui.add_checkbox("Show selection box", True)
        self.selection_summary = gui.add_markdown("")
        self.box_controls = self.server.scene.add_transform_controls(
            "/subject_box", position=position, wxyz=wxyz, scale=float(dimensions.max()) * 0.4,
            depth_test=False, translation_limits=((-1e10, 1e10),) * 3)
        self.box_outline = self.server.scene.add_box(
            "/subject_box/outline", dimensions=dimensions, wireframe=True, color=(40, 240, 255))
        step = max(min(dimensions.min(), self.data.initial_box[2].min()) * 0.001, 1e-7)
        # ponytail: cap sliders at 4x subject size for fine control; add editable limits for larger selections.
        maximum = 4 * max(float(dimensions.max()), float(self.data.initial_box[2].max()))
        self.box_sizes = tuple(gui.add_slider(
            f"Box size {axis}", min=step, max=maximum, step=step, initial_value=float(value))
            for axis, value in zip("XYZ", dimensions))
        reset = gui.add_button("Reset box to old foreground bounds")
        save = gui.add_button("Save selection")
        self.save_status = gui.add_markdown("Selection is not saved. Saving creates a separate NPZ; "
                                             "the source scene and existing experiments remain unchanged.")

        async def update_box(_):
            self._refresh_box()

        self.box_controls.on_update(update_box)
        for handle in self.box_sizes:
            handle.on_update(update_box)

        @reset.on_click
        async def _reset(_):
            position, wxyz, dimensions = self.data.initial_box
            with self.server.atomic():
                self.box_controls.position = position
                self.box_controls.wxyz = wxyz
                for handle, value in zip(self.box_sizes, dimensions):
                    handle.value = float(value)
            self._refresh_box()

        @self.show_box.on_update
        async def _show(_):
            self.box_controls.visible = bool(self.show_box.value)

        @save.on_click
        def _save(_):
            try:
                path = save_subject_selection(self.work_dir, self.data.scene_path, self.data.scene,
                                              self.data.points, *self._box_state)
                self.save_status.content = f"Saved selection: `{path}`"
                print(f"Subject selection saved: {path}")
            except (OSError, ValueError) as error:
                self.save_status.content = f"Could not save selection: {error}"

        for handle in (self.viewer_resolution, self.highlight, self.only_selected):
            handle.on_update(self.request_render)
        self._build_camera_controls()

    def _refresh_box(self):
        self._box_state = _box_values(self.box_controls.position, self.box_controls.wxyz,
                                     [handle.value for handle in self.box_sizes])
        self.box_outline.dimensions = tuple(self._box_state[2])
        self.save_status.content = "Box changed; press Save selection to keep this selection."
        self.request_render()

    @torch.inference_mode()
    def _render(self, client):
        selected = points_in_box(self.data.points, *self._box_state)
        old_fg = int(selected[:self.data.scene.foreground.count].sum())
        old_bg = int(selected[self.data.scene.foreground.count:].sum())
        self.selection_summary.content = (f"**Selected: {old_fg + old_bg:,} / {len(selected):,}**\n\n"
                                          f"From old foreground: {old_fg:,}; old background: {old_bg:,}.")
        return self.data.render_selection(self._render_camera(client), selected,
            highlight=bool(self.highlight.value), only_selected=bool(self.only_selected.value))


def run_subject_selection_viewer(*, scene_dir, work_dir, selection_path=None,
                                 host="127.0.0.1", port=8080, viewer_resolution=2048):
    data = SubjectSelectionData(scene_dir, selection_path)
    _load_gsplat_rasterization()
    viewer = SubjectSelectionViewer(data, work_dir=work_dir, host=host, port=port,
                                    viewer_resolution=viewer_resolution)
    viewer.wait()
