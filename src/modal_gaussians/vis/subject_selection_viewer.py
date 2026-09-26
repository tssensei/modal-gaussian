"""Select a static scene's motion subject with an oriented 3D box."""
from __future__ import annotations

from modal_gaussians.common.scene_store import resolve_path

import numpy as np
import torch

from modal_gaussians.common.camera_rendering import rasterize_cameras
from modal_gaussians.geometry.scene import _load_gsplat_rasterization, cameras_from_scene_manifest, load_static_scene
from modal_gaussians.geometry.selection import _box_values, load_subject_selection, points_in_boxes, save_subject_selection
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
        # Initial editing bounds only: all Gaussians remain selectable.
        lower, upper = np.percentile(foreground, [5, 95], axis=0)
        dimensions = np.maximum((upper - lower) * 1.1, 1e-4)
        self.initial_box = _box_values((lower + upper) / 2, (1, 0, 0, 0), dimensions)
        self.boxes = (load_subject_selection(selection_path, self.scene)
                      if selection_path is not None else [self.initial_box])
        cameras = cameras_from_scene_manifest(self.scene.manifest)
        references = [camera for camera in cameras if camera.role == "reference"] or list(cameras[:1])
        self.cameras = tuple(ViewerCamera.from_camera(camera) for camera in references)

    @torch.inference_mode()
    def render_selection(self, camera, selected, *, highlight=True, only_selected=False):
        """Tint both original partitions with one shared depth order; never edit parameters."""
        active = self.active
        mask = torch.as_tensor(selected, device=self.device, dtype=torch.bool)
        colors = self.scene.view_colors([camera], active["means"])[0]
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
        self._boxes = {f"Box {i+1}": box for i, box in enumerate(self.data.boxes)}
        self._next_box_id = len(self._boxes) + 1
        self._syncing_box = False
        position, wxyz, dimensions = self.data.boxes[0]
        self._orbit_center = position.copy()
        gui.add_markdown("**3D subject selection — box union**\n\n"
                         "Only Gaussian centers inside ANY box are selected (cyan), including old background. "
                         "Choose Active box, drag its axes/rings, and adjust its X/Y/Z sizes. "
                         "Add box duplicates the active box; move or resize it to include another part of the subject.")
        self.viewer_resolution = gui.add_slider(
            "Viewer Res", min=64, max=max(2048, self._viewer_resolution), step=1,
            initial_value=self._viewer_resolution)
        self.highlight = gui.add_checkbox("Highlight selection", True)
        self.only_selected = gui.add_checkbox("Only selected", False)
        self.show_box = gui.add_checkbox("Show selection boxes", True)
        self.selection_summary = gui.add_markdown("")
        self.active_box = gui.add_dropdown("Active box", options=tuple(self._boxes), initial_value="Box 1")
        add = gui.add_button("Add box")
        self.remove_box = gui.add_button("Remove active box", disabled=len(self._boxes) == 1)
        self.box_controls = self.server.scene.add_transform_controls(
            "/subject_box", position=position, wxyz=wxyz, scale=float(dimensions.max()) * 0.4,
            depth_test=False, translation_limits=((-1e10, 1e10),) * 3)
        self._outlines = {label: self._add_outline(label, box) for label, box in self._boxes.items()}
        # Each active box gets its own editing range; switching/resetting refreshes it.
        self.box_sizes = tuple(gui.add_slider(
            f"Box size {axis}", min=1e-7, max=float(2 * span),
            step=max(float(span) * 1e-4, 1e-7), initial_value=float(value))
            for axis, value, span in zip("XYZ", dimensions, dimensions))
        reset = gui.add_button("Reset active box to robust subject bounds")
        save = gui.add_button("Save selection")
        self.save_status = gui.add_markdown("Selection is not saved. Saving creates a separate NPZ; "
                                             "the source scene and existing experiments remain unchanged.")

        async def update_box(_):
            self._refresh_box()

        self.box_controls.on_update(update_box)
        for handle in self.box_sizes:
            handle.on_update(update_box)

        @self.active_box.on_update
        async def _select(_):
            self._load_active_box()

        @add.on_click
        async def _add(_):
            label = f"Box {self._next_box_id}"
            self._next_box_id += 1
            self._boxes[label] = tuple(value.copy() for value in self._boxes[self.active_box.value])
            self._outlines[label] = self._add_outline(label, self._boxes[label])
            self.active_box.options = tuple(self._boxes)
            self.active_box.value = label
            self.remove_box.disabled = False
            self._load_active_box()
            self.save_status.content = "Box added; press Save selection to keep all boxes."

        @self.remove_box.on_click
        async def _remove(_):
            if len(self._boxes) == 1:
                return
            label = self.active_box.value
            self._outlines.pop(label).remove()
            del self._boxes[label]
            self.active_box.options = tuple(self._boxes)
            self.active_box.value = next(iter(self._boxes))
            self.remove_box.disabled = len(self._boxes) == 1
            self._load_active_box()
            self.save_status.content = "Box removed; press Save selection to keep this union."

        @reset.on_click
        async def _reset(_):
            self._boxes[self.active_box.value] = tuple(value.copy() for value in self.data.initial_box)
            self._load_active_box()
            self.save_status.content = "Active box reset; press Save selection to keep this union."

        @self.show_box.on_update
        async def _show(_):
            self.box_controls.visible = bool(self.show_box.value)
            for outline in self._outlines.values():
                outline.visible = bool(self.show_box.value)

        @save.on_click
        def _save(_):
            try:
                path = save_subject_selection(self.work_dir, self.data.scene_path, self.data.scene,
                                              self.data.points, list(self._boxes.values()))
                self.save_status.content = f"Saved selection: `{path}`"
                print(f"Subject selection saved: {path}")
            except (OSError, ValueError) as error:
                self.save_status.content = f"Could not save selection: {error}"

        for handle in (self.viewer_resolution, self.highlight, self.only_selected):
            handle.on_update(self.request_render)
        self._build_camera_controls()

    def _add_outline(self, label, box):
        position, wxyz, dimensions = box
        return self.server.scene.add_box(f"/subject_boxes/{label.replace(' ', '_')}",
            position=position, wxyz=wxyz, dimensions=dimensions, wireframe=True,
            color=(40, 240, 255), visible=bool(self.show_box.value))

    def _load_active_box(self):
        self._syncing_box = True
        try:
            position, wxyz, dimensions = self._boxes[self.active_box.value]
            with self.server.atomic():
                self.box_controls.position = position
                self.box_controls.wxyz = wxyz
                for handle, value in zip(self.box_sizes, dimensions):
                    handle.max = float(2 * value)
                    handle.step = max(float(value) * 1e-4, 1e-7)
                    handle.value = float(value)
                for label, outline in self._outlines.items():
                    box = self._boxes[label]
                    outline.position, outline.wxyz, outline.dimensions = box[0], box[1], tuple(box[2])
                    outline.color = (255, 200, 40) if label == self.active_box.value else (40, 240, 255)
        finally:
            self._syncing_box = False
        self.request_render()

    def _refresh_box(self):
        if self._syncing_box:
            return
        label = self.active_box.value
        box = _box_values(self.box_controls.position, self.box_controls.wxyz,
                          [handle.value for handle in self.box_sizes])
        if all(np.array_equal(a, b) for a, b in zip(box, self._boxes[label])):
            return
        self._boxes[label] = box
        outline = self._outlines[label]
        outline.position, outline.wxyz, outline.dimensions = box[0], box[1], tuple(box[2])
        self.save_status.content = "Box changed; press Save selection to keep this union."
        self.request_render()

    @torch.inference_mode()
    def _render(self, client):
        selected = points_in_boxes(self.data.points, list(self._boxes.values()))
        old_fg = int(selected[:self.data.scene.foreground.count].sum())
        old_bg = int(selected[self.data.scene.foreground.count:].sum())
        self.selection_summary.content = (f"**Union of {len(self._boxes)} boxes: {old_fg + old_bg:,} / {len(selected):,}**\n\n"
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
