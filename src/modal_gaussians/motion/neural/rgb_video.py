"""Lazy RGB supervision in the same reference pixel domain as prepared flow."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from modal_gaussians.data.sequence import read_binary_mask, read_color_image
from modal_gaussians.static import Camera, cameras_from_scene_manifest


@dataclass
class RGBVideoView:
    label: str
    camera: Camera
    frame_names: list[str]
    fps_hz: float
    reference_frame_index: int
    times: np.ndarray
    duration_seconds: float
    metadata: dict[str, Any]
    image_directory: Path
    mask_directory: Path
    original_hw: tuple[int, int]
    scale: float
    background_weight: float
    homographies: np.ndarray | None = None

    def read_frame(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one frame; return RGB and loss weights, including valid borders."""
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.frame_names):
            raise IndexError(f"RGB frame index outside {self.label}: {index!r}")
        name = self.frame_names[index]
        image = read_color_image(self.image_directory / f"{name}.png")
        if image.shape[:2] != self.original_hw:
            raise ValueError(f"RGB frame dimensions differ from the reference camera: {name}")
        mask = read_binary_mask(self.mask_directory / f"{name}.png", self.original_hw)
        weight = np.where(mask, 1.0, self.background_weight).astype(np.float32)
        if self.homographies is not None:
            matrix = np.asarray(self.homographies[index])
            if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
                raise ValueError(f"Invalid stabilization homography for {self.label}/{name}")
            # Saved RGB uses replicated borders; these are not video observations.
            valid = cv2.warpPerspective(np.ones(self.original_hw, np.uint8), matrix,
                (self.original_hw[1], self.original_hw[0]), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            weight *= valid
        if self.scale != 1.0:
            transform = np.array([[self.scale, 0, 0], [0, self.scale, 0]], dtype=np.float32)
            size = (self.camera.width, self.camera.height)
            image = cv2.warpAffine(image, transform, size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            weight = cv2.warpAffine(weight, transform, size, flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        device = self.camera.K.device
        return (torch.as_tensor(rgb, device=device, dtype=torch.float32) / 255.0,
                torch.as_tensor(weight, device=device, dtype=torch.float32))


def _directory(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label} directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} directory must be absolute: {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _scaled_camera(camera: Camera, max_width: int, device: str | torch.device) -> tuple[Camera, float]:
    scale = min(1.0, max_width / camera.width) if max_width else 1.0
    K = camera.K.clone()
    K[:2] *= scale
    parameters = list(camera.camera_parameters)
    if camera.camera_model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        parameters[:3] = [value * scale for value in parameters[:3]]
    elif camera.camera_model == "PINHOLE":
        parameters[:4] = [value * scale for value in parameters[:4]]
    resized = replace(camera, width=max(1, round(camera.width * scale)),
        height=max(1, round(camera.height * scale)), K=K,
        camera_parameters=tuple(parameters)).to(device)
    _ = resized.radial_distortion
    return resized, scale


def build_video_views(prepared_manifest: dict[str, Any], scene: Any, *, max_width: int = 0,
                      device: str | torch.device = "cpu", background_weight: float = 0.05) -> list[RGBVideoView]:
    """Read small metadata only; never load flow arrays or scan/decode a sequence."""
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 0:
        raise ValueError("RGB max_width must be nonnegative; zero preserves original resolution")
    if not np.isfinite(background_weight) or not 0 <= background_weight <= 1:
        raise ValueError("RGB background_weight must lie in [0, 1]")
    source_views, flows = prepared_manifest["source"]["views"], prepared_manifest["flows"]
    if not source_views or len(source_views) != len(flows):
        raise ValueError("RGB prepared view and flow counts must agree and be nonempty")
    labels = [view["label"] for view in source_views]
    if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("RGB source view labels must be unique nonempty strings")
    references = [c for c in cameras_from_scene_manifest(scene.manifest) if c.role == "reference"]
    by_label = {c.label: c for c in references}
    if len(by_label) != len(references):
        raise ValueError("Scene reference camera labels must be unique")
    result = []
    for source_view, record in zip(source_views, flows):
        label = source_view["label"]
        if label not in by_label:
            raise ValueError(f"No reference camera for RGB view {label!r}")
        camera, manifest = by_label[label], record["manifest"]
        if (not isinstance(source_view.get("flow_identity"), str) or not source_view["flow_identity"]
                or source_view["flow_identity"] != record.get("identity")):
            raise ValueError(f"RGB flow belongs to a different source view: {label}")
        if source_view.get("camera_identity") != camera.to_manifest_record()["camera_identity"]:
            raise ValueError(f"RGB camera belongs to a different source view: {label}")
        names = manifest.get("frame_names")
        if (not isinstance(names, list) or len(names) < 3
            or any(not isinstance(name, str) or not name or name in (".", "..")
                   or any(character in name for character in "/\\:") for name in names)
            or len(set(names)) != len(names)):
            raise ValueError(f"Invalid RGB frame stems for {label}")
        fps, reference = float(manifest.get("fps_hz", np.nan)), manifest.get("reference_frame_index")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"RGB FPS must be finite and positive for {label}")
        if (isinstance(reference, bool) or not isinstance(reference, int) or not 0 <= reference < len(names)
            or names[reference] != manifest.get("reference_frame_name")):
            raise ValueError(f"Invalid RGB reference frame for {label}")
        shape = manifest.get("arrays", {}).get("flow", {}).get("shape")
        if shape != [len(names), camera.height, camera.width, 2]:
            raise ValueError(f"RGB flow and reference-camera dimensions differ for {label}")
        stabilization = manifest.get("parameters", {}).get("stabilization", {})
        method = stabilization.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError(f"Missing RGB stabilization pixel-domain information for {label}")
        homographies = None
        if method == "none":
            if manifest.get("stabilized_sequence") is not None:
                raise ValueError(f"Contradictory RGB stabilization metadata for {label}")
            sequence = manifest.get("inputs", {}).get("sequence", {})
            images = _directory(sequence.get("image_directory"), "RGB image")
            masks = _directory(sequence.get("mask_directory"), "RGB mask")
        else:
            stabilized = manifest.get("stabilized_sequence")
            if not isinstance(stabilized, dict) or stabilized.get("path") != "stabilized_sequence":
                raise ValueError(f"Stabilized RGB sequence is missing for {label}; original frames cannot substitute")
            root = _directory(record.get("path"), "RGB flow") / "stabilized_sequence"
            with (root / "manifest.json").open(encoding="utf-8") as stream:
                sequence = json.load(stream)
            if (sequence.get("format") != "modal_gaussians.stabilized_image_mask_sequence"
                or sequence.get("version") != 2 or sequence.get("frames") != names
                or sequence.get("fps_hz") != fps or sequence.get("reference_frame") != names[reference]
                or (sequence.get("height"), sequence.get("width")) != (camera.height, camera.width)):
                raise ValueError(f"Stabilized RGB pixel-domain metadata differs for {label}")
            if sequence.get("homographies", {}).get("file") != "homographies_frame_to_reference.npy":
                raise ValueError(f"Missing stabilized RGB homography information for {label}")
            homographies = np.load(root / "homographies_frame_to_reference.npy", allow_pickle=False)
            if homographies.shape != (len(names), 3, 3) or homographies.dtype.kind != "f":
                raise ValueError(f"Invalid stabilized RGB homography array for {label}")
            if not np.allclose(homographies[reference], np.eye(3), rtol=0, atol=1e-8):
                raise ValueError(f"Stabilized RGB reference is not anchored to its camera for {label}")
            images = _directory(str(root / "images"), "stabilized RGB image")
            masks = _directory(str(root / "masks"), "stabilized RGB mask")
        resized, scale = _scaled_camera(camera, max_width, device)
        times = np.arange(len(names), dtype=np.float32) / fps
        metadata = {"label": label, "flow_artifact": record.get("path"), "pixel_domain": method,
            "image_directory": str(images), "mask_directory": str(masks), "frame_names": list(names),
            "fps_hz": fps, "reference_frame_index": reference, "original_hw": [camera.height, camera.width],
            "render_hw": [resized.height, resized.width], "image_scale": scale,
            "time_origin": "first_video_frame", "background_weight": float(background_weight),
            "invalid_stabilization_border_weight": 0.0}
        result.append(RGBVideoView(label, resized, list(names), fps, reference, times,
            float((len(names) - 1) / fps), metadata, images, masks, (camera.height, camera.width),
            scale, float(background_weight), homographies))
    return result
