"""Independent static foreground/background 3D Gaussian scene primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


Composition = Literal["all", "foreground", "background"]
CameraRole = Literal["sweep", "reference"]
GAUSSIAN_FIELDS = (
    "means",
    "quaternions",
    "log_scales",
    "color_logits",
    "opacity_logits",
)


@dataclass(frozen=True)
class SceneNormalization:
    """Describe the single raw-COLMAP to normalized-world transform."""

    center: np.ndarray
    rotation: np.ndarray
    scale: float

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Transform raw COLMAP points into the normalized static world."""

        return ((points - self.center[None]) @ self.rotation.T) / self.scale

    def normalize_world_to_camera(self, world_to_camera: np.ndarray) -> np.ndarray:
        """Express a raw COLMAP world-to-camera matrix in normalized world units."""

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = -self.rotation @ self.center
        normalized = world_to_camera @ np.linalg.inv(transform)
        normalized[:3, 3] /= self.scale
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Serialize the normalization without Python-specific objects."""

        return {
            "format": "modal_gaussians.scene_normalization",
            "version": 1,
            "formula": "x_normalized = R @ (x_raw - center) / scale",
            "center": self.center.astype(float).tolist(),
            "rotation": self.rotation.astype(float).tolist(),
            "scale": float(self.scale),
            "source": "registered_sweep_cameras_and_sparse_points",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SceneNormalization":
        """Reconstruct a normalization from a static-scene manifest."""

        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            rotation=np.asarray(payload["rotation"], dtype=np.float64),
            scale=float(payload["scale"]),
        )


@dataclass(frozen=True)
class Camera:
    """Store one immutable training/render camera in raw and normalized worlds."""

    name: str
    role: CameraRole
    label: str | None
    width: int
    height: int
    K: Tensor
    raw_world_to_camera: Tensor
    world_to_camera: Tensor
    camera_model: str
    camera_parameters: tuple[float, ...]
    image_relative_path: str
    mask_relative_path: str
    image_sha256: str
    mask_sha256: str

    def to(self, device: torch.device | str) -> "Camera":
        """Copy only camera tensors to the requested compute device."""

        return replace(
            self,
            K=self.K.to(device),
            raw_world_to_camera=self.raw_world_to_camera.to(device),
            world_to_camera=self.world_to_camera.to(device),
        )

    def to_manifest_record(self) -> dict[str, Any]:
        """Serialize camera geometry and immutable source identities."""

        raw_w2c = self.raw_world_to_camera.detach().cpu().numpy().astype(np.float64)
        w2c = self.world_to_camera.detach().cpu().numpy().astype(np.float64)
        record = {
            "name": self.name,
            "role": self.role,
            "width": self.width,
            "height": self.height,
            "camera_model": self.camera_model,
            "camera_parameters": list(self.camera_parameters),
            "distortion_applied": False,
            "raw_K": self.K.detach().cpu().numpy().astype(float).tolist(),
            "K": self.K.detach().cpu().numpy().astype(float).tolist(),
            "raw_world_to_camera": raw_w2c.astype(float).tolist(),
            "raw_camera_to_world": np.linalg.inv(raw_w2c).astype(float).tolist(),
            "world_to_camera": w2c.astype(float).tolist(),
            "camera_to_world": np.linalg.inv(w2c).astype(float).tolist(),
            "image_relative_path": self.image_relative_path,
            "mask_relative_path": self.mask_relative_path,
            "image_sha256": self.image_sha256,
            "mask_sha256": self.mask_sha256,
        }
        if self.label is not None:
            record["label"] = self.label
        record["camera_identity"] = _sha256_json(record)
        return record

    @classmethod
    def from_manifest_record(cls, payload: Mapping[str, Any]) -> "Camera":
        """Load a camera record embedded in a static-scene manifest."""

        identity_payload = dict(payload)
        recorded_identity = identity_payload.pop("camera_identity", None)
        if recorded_identity != _sha256_json(identity_payload):
            raise ValueError(f"Camera identity does not match manifest record: {payload.get('name')}")
        role = str(payload["role"])
        if role not in ("sweep", "reference"):
            raise ValueError(f"Unsupported camera role: {role}")
        if not np.allclose(
            np.asarray(payload.get("raw_K"), dtype=np.float64),
            np.asarray(payload["K"], dtype=np.float64),
        ):
            raise ValueError(f"Static v1 requires unchanged raw/normalized K: {payload.get('name')}")
        return cls(
            name=str(payload["name"]),
            role=role,
            label=str(payload["label"]) if "label" in payload else None,
            width=int(payload["width"]),
            height=int(payload["height"]),
            K=torch.tensor(payload["K"], dtype=torch.float32),
            raw_world_to_camera=torch.tensor(
                payload["raw_world_to_camera"], dtype=torch.float32
            ),
            world_to_camera=torch.tensor(
                payload["world_to_camera"], dtype=torch.float32
            ),
            camera_model=str(payload["camera_model"]),
            camera_parameters=tuple(
                float(value) for value in payload["camera_parameters"]
            ),
            image_relative_path=str(payload["image_relative_path"]),
            mask_relative_path=str(payload["mask_relative_path"]),
            image_sha256=str(payload["image_sha256"]),
            mask_sha256=str(payload["mask_sha256"]),
        )


@dataclass(frozen=True)
class StaticDataset:
    """Validated joint-COLMAP inputs ready for static Gaussian training."""

    root: Path
    cameras: tuple[Camera, ...]
    raw_points: np.ndarray
    point_colors: np.ndarray
    normalization: SceneNormalization
    dataset_identity: str
    file_identities: Mapping[str, str]

    def load_rgb(self, camera: Camera) -> Tensor:
        """Read one canonical RGB image as float32 RGB in [0, 1]."""

        path = self.root / camera.image_relative_path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image.copy()).float() / 255.0

    def load_binary_mask(self, camera: Camera) -> np.ndarray:
        """Read one semantic mask with positive pixels treated as foreground."""

        path = self.root / camera.mask_relative_path
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        return mask > 0


class GaussianSet(nn.Module):
    """Own one independently indexed set of trainable static Gaussians."""

    def __init__(
        self,
        means: Tensor,
        quaternions: Tensor,
        log_scales: Tensor,
        color_logits: Tensor,
        opacity_logits: Tensor,
    ) -> None:
        """Create a Gaussian parameter set after strict shape validation."""

        super().__init__()
        tensors = {
            "means": means,
            "quaternions": quaternions,
            "log_scales": log_scales,
            "color_logits": color_logits,
            "opacity_logits": opacity_logits,
        }
        _validate_gaussian_tensors(tensors)
        self.params = nn.ParameterDict(
            {name: nn.Parameter(value.float().contiguous()) for name, value in tensors.items()}
        )

    @property
    def count(self) -> int:
        """Return the current number of independently indexed Gaussians."""

        return int(self.params["means"].shape[0])

    def active(self) -> dict[str, Tensor]:
        """Return activated tensors expected by the rasterizer."""

        return {
            "means": self.params["means"],
            "quaternions": F.normalize(self.params["quaternions"], dim=-1),
            "scales": torch.exp(self.params["log_scales"]),
            "colors": torch.sigmoid(self.params["color_logits"]),
            "opacities": torch.sigmoid(self.params["opacity_logits"]),
        }

    def raw_tensors(self, *, cpu: bool = False) -> dict[str, Tensor]:
        """Return detached raw parameters for checkpointing or identity hashing."""

        tensors = {name: value.detach() for name, value in self.params.items()}
        if cpu:
            tensors = {name: value.cpu().contiguous() for name, value in tensors.items()}
        return tensors

    def replace_parameter(self, name: str, value: Tensor) -> nn.Parameter:
        """Replace one resized parameter while preserving its canonical field name."""

        if name not in GAUSSIAN_FIELDS:
            raise KeyError(name)
        parameter = nn.Parameter(value.contiguous())
        self.params[name] = parameter
        return parameter


class ForegroundBackgroundScene(nn.Module):
    """Render foreground and background separately or with one shared depth order."""

    def __init__(
        self,
        foreground: GaussianSet,
        background: GaussianSet,
        *,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a static scene without any dynamic or modal state."""

        super().__init__()
        if foreground.count < 2 or background.count < 2:
            raise ValueError("Static scene requires at least two FG and two BG Gaussians")
        self.foreground = foreground
        self.background = background
        self.manifest = dict(manifest) if manifest is not None else None

    @property
    def count(self) -> int:
        """Return the combined foreground/background Gaussian count."""

        return self.foreground.count + self.background.count

    def _active_for(self, composition: Composition) -> dict[str, Tensor]:
        """Select or concatenate Gaussian tensors for one rasterization call."""

        if composition == "foreground":
            return self.foreground.active()
        if composition == "background":
            return self.background.active()
        if composition != "all":
            raise ValueError(f"Unsupported composition: {composition}")
        foreground = self.foreground.active()
        background = self.background.active()
        return {
            name: torch.cat([foreground[name], background[name]], dim=0).contiguous()
            for name in foreground
        }

    def render_batch(
        self,
        cameras: Sequence[Camera],
        *,
        composition: Composition = "all",
        retain_screen_grad: bool = False,
    ) -> tuple[dict[str, Tensor], Mapping[str, Tensor]]:
        """Rasterize same-resolution cameras and return RGB, alpha, and expected depth."""

        if not cameras:
            raise ValueError("At least one camera is required")
        width, height = cameras[0].width, cameras[0].height
        if any(camera.width != width or camera.height != height for camera in cameras):
            raise ValueError("render_batch cameras must share one image resolution")
        active = self._active_for(composition)
        device = active["means"].device
        Ks = torch.stack([camera.K.to(device) for camera in cameras], dim=0)
        viewmats = torch.stack(
            [camera.world_to_camera.to(device) for camera in cameras], dim=0
        )
        rasterization = _load_gsplat_rasterization()
        rendered, alphas, info = rasterization(
            means=active["means"],
            quats=active["quaternions"],
            scales=active["scales"],
            opacities=active["opacities"],
            colors=active["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
            backgrounds=torch.ones((len(cameras), 3), device=device),
            render_mode="RGB+ED",
            rasterize_mode="classic",
            camera_model="pinhole",
        )
        if retain_screen_grad:
            means2d = info.get("means2d")
            if means2d is None or not means2d.requires_grad:
                raise RuntimeError("gsplat did not expose differentiable means2d")
            means2d.retain_grad()
        rgb = rendered[..., :3]
        alpha = alphas[..., 0]
        depth = rendered[..., 3]
        depth = torch.where(alpha > 1e-8, depth, torch.zeros_like(depth))
        return {"rgb": rgb, "alpha": alpha, "expected_depth": depth}, info

    def render(
        self,
        camera: Camera,
        *,
        composition: Composition = "all",
        outputs: Sequence[str] = ("rgb", "alpha", "expected_depth"),
    ) -> dict[str, Tensor]:
        """Render one camera through the stable public static-scene interface."""

        allowed = {"rgb", "alpha", "expected_depth"}
        unknown = set(outputs) - allowed
        if unknown:
            raise ValueError(f"Unsupported render outputs: {sorted(unknown)}")
        rendered, _ = self.render_batch([camera], composition=composition)
        return {name: rendered[name][0] for name in outputs}

    def render_deformed(
        self,
        camera: Camera,
        foreground_means: Tensor,
        *,
        foreground_colors: Tensor | None = None,
        include_background: bool = True,
    ) -> dict[str, Tensor]:
        """Render deformed foreground and static background with one depth order.

        The modal viewer supplies only foreground means and, optionally, display
        colors.  Every other Gaussian parameter remains the trained static value.
        Foreground and background are concatenated before rasterization so their
        occlusion is identical to the public static ``composition="all"`` path.
        """

        foreground = self.foreground.active()
        background = self.background.active()
        device = foreground["means"].device
        means = foreground_means.to(
            device=device, dtype=foreground["means"].dtype
        ).contiguous()
        if means.shape != foreground["means"].shape:
            raise ValueError("foreground_means must have shape [G_foreground,3]")
        if not bool(torch.isfinite(means).all().item()):
            raise ValueError("foreground_means contain non-finite values")
        if foreground_colors is None:
            colors = foreground["colors"]
        else:
            colors = foreground_colors.to(
                device=device, dtype=foreground["colors"].dtype
            ).contiguous()
            if colors.shape != foreground["colors"].shape:
                raise ValueError("foreground_colors must have shape [G_foreground,3]")
            if not bool(torch.isfinite(colors).all().item()):
                raise ValueError("foreground_colors contain non-finite values")
            colors = colors.clamp(0.0, 1.0)

        active = {
            "means": means,
            "quaternions": foreground["quaternions"],
            "scales": foreground["scales"],
            "colors": colors,
            "opacities": foreground["opacities"],
        }
        if include_background:
            combined: dict[str, Tensor] = {}
            for name, value in active.items():
                background_value = background[name]
                if name == "colors" and foreground_colors is not None:
                    background_value = torch.full_like(background_value, 0.5)
                combined[name] = torch.cat(
                    [value, background_value], dim=0
                ).contiguous()
            active = combined

        rasterization = _load_gsplat_rasterization()
        rendered, alphas, _ = rasterization(
            means=active["means"],
            quats=active["quaternions"],
            scales=active["scales"],
            opacities=active["opacities"],
            colors=active["colors"],
            viewmats=camera.world_to_camera.to(device)[None],
            Ks=camera.K.to(device)[None],
            width=camera.width,
            height=camera.height,
            packed=False,
            backgrounds=torch.ones((1, 3), device=device),
            render_mode="RGB+ED",
            rasterize_mode="classic",
            camera_model="pinhole",
        )
        alpha = alphas[0, ..., 0]
        depth = torch.where(
            alpha > 1.0e-8,
            rendered[0, ..., 3],
            torch.zeros_like(rendered[0, ..., 3]),
        )
        return {"rgb": rendered[0, ..., :3], "alpha": alpha, "expected_depth": depth}

    def render_features(
        self,
        camera: Camera,
        features: Tensor,
        *,
        composition: Composition = "foreground",
    ) -> tuple[Tensor, Tensor]:
        """Rasterize arbitrary per-Gaussian features with a zero background."""

        active = self._active_for(composition)
        if features.ndim != 2 or features.shape[0] != active["means"].shape[0]:
            raise ValueError(
                "features must have shape [selected_gaussians, feature_channels]"
            )
        if features.shape[1] < 1:
            raise ValueError("features must contain at least one channel")
        device = active["means"].device
        values = features.to(device=device, dtype=active["means"].dtype).contiguous()
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("features contain non-finite values")
        rasterization = _load_gsplat_rasterization()
        rendered, alphas, _ = rasterization(
            means=active["means"],
            quats=active["quaternions"],
            scales=active["scales"],
            opacities=active["opacities"],
            colors=values,
            viewmats=camera.world_to_camera.to(device)[None],
            Ks=camera.K.to(device)[None],
            width=camera.width,
            height=camera.height,
            packed=False,
            backgrounds=torch.zeros((1, values.shape[1]), device=device),
            render_mode="RGB",
            rasterize_mode="classic",
            camera_model="pinhole",
        )
        image = rendered[0]
        alpha = alphas[0, ..., 0]
        if image.shape != (camera.height, camera.width, values.shape[1]):
            raise RuntimeError("gsplat returned an unexpected feature-image shape")
        if alpha.shape != (camera.height, camera.width):
            raise RuntimeError("gsplat returned an unexpected feature-alpha shape")
        return image, alpha

    def tensor_dictionary(self) -> dict[str, Tensor]:
        """Export the class-free tensor dictionary used by the public bundle."""

        tensors: dict[str, Tensor] = {}
        for part_name, part in (
            ("foreground", self.foreground),
            ("background", self.background),
        ):
            for field, value in part.raw_tensors(cpu=True).items():
                tensors[f"{part_name}.{field}"] = value
        return tensors


@lru_cache(maxsize=1)
def _load_gsplat_rasterization() -> Any:
    """Import gsplat, removing two invalid MSVC flags from its Windows JIT call."""

    if os.name != "nt":
        try:
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise RuntimeError("Static rendering requires gsplat==1.5.3") from error
        return rasterization

    import torch.utils.cpp_extension as cpp_extension

    _prepare_windows_extension_environment()

    original_jit_compile = cpp_extension._jit_compile

    def windows_jit_compile(*args: Any, **kwargs: Any) -> Any:
        """Translate gsplat's unconditional GCC optimization flags for MSVC."""

        positional = list(args)
        flags = positional[2] if len(positional) > 2 else kwargs.get("extra_cflags")
        if flags is not None:
            translated = []
            for flag in flags:
                if flag == "-Wno-attributes":
                    continue
                translated.append({"-O3": "/O2", "-O0": "/Od"}.get(flag, flag))
            if len(positional) > 2:
                positional[2] = translated
            else:
                kwargs["extra_cflags"] = translated
        return original_jit_compile(*positional, **kwargs)

    cpp_extension._jit_compile = windows_jit_compile
    try:
        from gsplat.rendering import rasterization

        backend_module = importlib.import_module("gsplat.cuda._backend")
        _gsplat_cuda_backend = getattr(backend_module, "_C", None)

        if _gsplat_cuda_backend is None:
            raise RuntimeError("gsplat CUDA backend did not initialize")
    except ImportError as error:
        raise RuntimeError("Static rendering requires gsplat==1.5.3") from error
    finally:
        cpp_extension._jit_compile = original_jit_compile
    return rasterization


def _prepare_windows_extension_environment() -> None:
    """Expose conda Ninja and the detected MSVC environment to gsplat's JIT build."""

    path_value = os.environ.get("PATH", "")
    if shutil.which("ninja") is None:
        scripts = Path(sys.executable).parent / "Scripts"
        if (scripts / "ninja.exe").is_file():
            path_value = f"{scripts};{path_value}"
            os.environ["PATH"] = path_value
    if shutil.which("cl") is not None:
        return
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = [
        *program_files.glob(
            "Microsoft Visual Studio/2022/*/VC/Auxiliary/Build/vcvars64.bat"
        ),
        *program_files.glob(
            "Microsoft Visual Studio/18/*/VC/Auxiliary/Build/vcvars64.bat"
        ),
    ]
    if not candidates:
        raise RuntimeError(
            "gsplat's Windows JIT build requires Visual Studio C++ Build Tools"
        )
    process = subprocess.run(
        f'call "{candidates[0]}" >nul && set',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Could not activate Visual Studio C++ Build Tools for gsplat: "
            + process.stderr.strip()
        )
    vc_environment: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            normalized_key = key.upper()
            if normalized_key not in vc_environment or key == normalized_key:
                vc_environment[normalized_key] = value
    os.environ.update(vc_environment)
    if shutil.which("cl") is None:
        raise RuntimeError("Visual Studio environment did not expose cl.exe")


def _read_exact(stream: Any, size: int, path: Path) -> bytes:
    """Read an exact number of COLMAP binary bytes or fail with context."""

    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of COLMAP binary file: {path}")
    return value


def _read_c_string(stream: Any, path: Path) -> str:
    """Read one null-terminated UTF-8 name from a COLMAP binary file."""

    value = bytearray()
    while True:
        byte = _read_exact(stream, 1, path)
        if byte == b"\x00":
            break
        value.extend(byte)
    return value.decode("utf-8").replace("\\", "/")


def _read_registered_images_binary(path: Path) -> dict[int, str]:
    """Read only image IDs and names needed to validate a COLMAP model."""

    images: dict[int, str] = {}
    with path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8, path))[0]
        for _ in range(count):
            image_id = struct.unpack("<i", _read_exact(stream, 4, path))[0]
            _read_exact(stream, 8 * 7, path)
            _read_exact(stream, 4, path)
            name = _read_c_string(stream, path)
            point_count = struct.unpack("<Q", _read_exact(stream, 8, path))[0]
            _read_exact(stream, int(point_count) * 24, path)
            if image_id in images:
                raise ValueError(f"Duplicate COLMAP image id: {image_id}")
            if name in images.values():
                raise ValueError(f"Duplicate COLMAP image name: {name}")
            images[image_id] = name
    if not images:
        raise ValueError(f"COLMAP model contains no registered images: {path}")
    return images


def _read_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read finite XYZ/RGB values from the authoritative COLMAP sparse cloud."""

    points: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    with path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8, path))[0]
        for _ in range(count):
            record = struct.unpack("<QdddBBBdQ", _read_exact(stream, 51, path))
            points.append((record[1], record[2], record[3]))
            colors.append((record[4], record[5], record[6]))
            track_length = int(record[8])
            _read_exact(stream, track_length * 8, path)
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.float32) / 255.0
    if xyz.shape != (count, 3) or count == 0:
        raise ValueError(f"COLMAP model contains no sparse points: {path}")
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError(f"COLMAP sparse point cloud contains non-finite values: {path}")
    return xyz, rgb


def _sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 identity for one immutable input file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible value with stable key and whitespace ordering."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_rigid_matrix(matrix: np.ndarray, label: str) -> None:
    """Require a finite right-handed 4x4 rigid transform."""

    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError(f"{label} rotation is not right-handed")


def _rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a stable rotation that maps one nonzero vector onto another."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("Cannot align a zero-length direction vector")
    source /= source_norm
    target /= target_norm
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    x, y, z = cross
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def compute_scene_normalization(
    raw_points: np.ndarray,
    sweep_world_to_cameras: np.ndarray,
) -> SceneNormalization:
    """Compute the accepted point-centered, Z-up COLMAP normalization."""

    if raw_points.ndim != 2 or raw_points.shape[1] != 3 or len(raw_points) == 0:
        raise ValueError("Scene normalization requires non-empty points shaped [N,3]")
    if (
        sweep_world_to_cameras.ndim != 3
        or sweep_world_to_cameras.shape[1:] != (4, 4)
        or len(sweep_world_to_cameras) == 0
    ):
        raise ValueError("Scene normalization requires sweep poses shaped [C,4,4]")
    center = raw_points.mean(axis=0)
    low = np.quantile(raw_points - center, 0.05, axis=0)
    high = np.quantile(raw_points - center, 0.95, axis=0)
    scale = float(np.max(high - low) / 2.0)
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"Invalid scene scale: {scale}")
    original_up = -sweep_world_to_cameras[:, 1, :3].mean(axis=0)
    rotation = _rotation_aligning_vectors(original_up, np.asarray([0.0, 0.0, 1.0]))
    return SceneNormalization(center=center, rotation=rotation, scale=scale)


def _validate_image_mask_pair(image_path: Path, mask_path: Path, width: int, height: int) -> None:
    """Require canonical PNG RGB and binary mask files at the recorded size."""

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(image_path)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"RGB input must be uint8 [H,W,3]: {image_path}")
    if image.shape[:2] != (height, width) or mask.shape[:2] != (height, width):
        raise ValueError(f"Recorded dimensions do not match RGB/mask pair: {image_path}")
    if mask.ndim == 3:
        first = mask[..., 0]
        if not all(np.array_equal(first, mask[..., index]) for index in range(1, mask.shape[2])):
            raise ValueError(f"Mask channels differ: {mask_path}")
        mask = first
    if mask.ndim != 2:
        raise ValueError(f"Mask must be grayscale or replicated grayscale: {mask_path}")
    values = np.unique(mask)
    if values.size > 2 or (values.size == 2 and values[0] != 0):
        raise ValueError(f"Mask must use zero background and one positive foreground value: {mask_path}")


def _role_png_names(root: Path, parent: str, role: str) -> set[str]:
    """List one flat role directory as COLMAP-style relative PNG names."""

    directory = root / parent / role
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    entries = list(directory.iterdir())
    invalid = [path.name for path in entries if not path.is_file() or path.suffix.lower() != ".png"]
    if invalid:
        raise ValueError(f"{directory} must contain only flat PNG files: {invalid[:5]}")
    return {f"{role}/{path.name}" for path in entries}


def load_static_dataset(root: str | Path) -> StaticDataset:
    """Validate one joint-COLMAP directory and materialize its static data contract."""

    root = Path(root).expanduser().resolve(strict=True)
    cameras_path = root / "cameras.json"
    points_path = root / "sparse" / "0" / "points3D.bin"
    images_path = root / "sparse" / "0" / "images.bin"
    colmap_cameras_path = root / "sparse" / "0" / "cameras.bin"
    ply_path = root / "point_cloud.ply"
    for path in (cameras_path, points_path, images_path, colmap_cameras_path, ply_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    payload = json.loads(cameras_path.read_text(encoding="utf-8"))
    if payload.get("format") != "modal_gaussians_colmap" or payload.get("version") != 1:
        raise ValueError(f"Unsupported joint-COLMAP manifest: {cameras_path}")
    registered = _read_registered_images_binary(images_path)
    registered_by_name = {name: image_id for image_id, name in registered.items()}
    raw_points, point_colors = _read_points3d_binary(points_path)
    records = [*payload.get("frames", []), *payload.get("references", [])]
    if not records:
        raise ValueError("Joint-COLMAP cameras.json contains no training cameras")
    names: set[str] = set()
    labels: set[str] = set()
    pending: list[dict[str, Any]] = []
    file_identities: dict[str, str] = {}
    for path in (cameras_path, points_path, images_path, colmap_cameras_path, ply_path):
        file_identities[path.relative_to(root).as_posix()] = _sha256_file(path)
    for record in records:
        name = str(record["image_name"]).replace("\\", "/")
        role = str(record["role"])
        if role not in ("sweep", "reference"):
            raise ValueError(f"Unsupported camera role for {name}: {role}")
        expected_prefix = "sweep/" if role == "sweep" else "references/"
        if not name.startswith(expected_prefix):
            raise ValueError(f"Camera role/path mismatch: {name}")
        if name in names:
            raise ValueError(f"Duplicate camera name: {name}")
        names.add(name)
        if name not in registered_by_name:
            raise ValueError(f"Camera is absent from sparse/0/images.bin: {name}")
        if int(record["image_id"]) != registered_by_name[name]:
            raise ValueError(f"COLMAP image id mismatch for {name}")
        label = str(record["label"]) if role == "reference" else None
        if label is not None:
            if label in labels:
                raise ValueError(f"Duplicate reference label: {label}")
            labels.add(label)
        width = int(record["image_width"])
        height = int(record["image_height"])
        K = np.asarray(record["K"], dtype=np.float64)
        raw_w2c = np.asarray(record["world_to_camera"], dtype=np.float64)
        raw_c2w = np.asarray(record["camera_to_world"], dtype=np.float64)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 0] <= 0.0
            or K[1, 1] <= 0.0
            or not np.isclose(K[2, 2], 1.0)
        ):
            raise ValueError(f"Invalid K for {name}")
        _validate_rigid_matrix(raw_w2c, f"world_to_camera for {name}")
        _validate_rigid_matrix(raw_c2w, f"camera_to_world for {name}")
        if not np.allclose(np.linalg.inv(raw_w2c), raw_c2w, atol=1e-5):
            raise ValueError(f"w2c/c2w mismatch for {name}")
        model = str(record["camera_model"])
        parameters = tuple(float(value) for value in record["camera_parameters"])
        if model != "SIMPLE_RADIAL" or len(parameters) != 4:
            raise ValueError(f"Static v1 requires SIMPLE_RADIAL cameras, got {model} for {name}")
        image_relative = f"images/{name}"
        mask_relative = f"masks/{name}"
        image_file = root / image_relative
        mask_file = root / mask_relative
        _validate_image_mask_pair(image_file, mask_file, width, height)
        image_hash = _sha256_file(image_file)
        mask_hash = _sha256_file(mask_file)
        file_identities[image_relative] = image_hash
        file_identities[mask_relative] = mask_hash
        pending.append(
            {
                "name": name,
                "role": role,
                "label": label,
                "width": width,
                "height": height,
                "K": K,
                "raw_w2c": raw_w2c,
                "model": model,
                "parameters": parameters,
                "image_relative": image_relative,
                "mask_relative": mask_relative,
                "image_hash": image_hash,
                "mask_hash": mask_hash,
            }
        )
    expected_names = {entry["name"] for entry in pending}
    image_names = _role_png_names(root, "images", "sweep") | _role_png_names(
        root, "images", "references"
    )
    mask_names = _role_png_names(root, "masks", "sweep") | _role_png_names(
        root, "masks", "references"
    )
    if image_names != mask_names or image_names != expected_names:
        raise ValueError(
            "Joint-COLMAP RGB, mask, and cameras.json names must match exactly: "
            f"rgb_only={sorted(image_names - mask_names)[:5]}, "
            f"mask_only={sorted(mask_names - image_names)[:5]}, "
            f"unrecorded={sorted(image_names - expected_names)[:5]}, "
            f"missing={sorted(expected_names - image_names)[:5]}"
        )
    sweep_poses = [entry["raw_w2c"] for entry in pending if entry["role"] == "sweep"]
    if not sweep_poses:
        raise ValueError("Joint-COLMAP input must contain at least one sweep camera")
    sweep_w2cs = np.stack(sweep_poses)
    normalization = compute_scene_normalization(raw_points, sweep_w2cs)
    cameras: list[Camera] = []
    for entry in pending:
        cameras.append(
            Camera(
                name=entry["name"],
                role=entry["role"],
                label=entry["label"],
                width=entry["width"],
                height=entry["height"],
                K=torch.from_numpy(entry["K"].astype(np.float32)),
                raw_world_to_camera=torch.from_numpy(
                    entry["raw_w2c"].astype(np.float32)
                ),
                world_to_camera=torch.from_numpy(
                    normalization.normalize_world_to_camera(entry["raw_w2c"]).astype(
                        np.float32
                    )
                ),
                camera_model=entry["model"],
                camera_parameters=entry["parameters"],
                image_relative_path=entry["image_relative"],
                mask_relative_path=entry["mask_relative"],
                image_sha256=entry["image_hash"],
                mask_sha256=entry["mask_hash"],
            )
        )
    authoritative_files = {
        name: digest
        for name, digest in file_identities.items()
        if name != "point_cloud.ply"
    }
    identity_payload = {
        "format": "modal_gaussians.static_dataset_identity",
        "version": 1,
        "files": sorted(authoritative_files.items()),
        "normalization": normalization.to_dict(),
    }
    return StaticDataset(
        root=root,
        cameras=tuple(cameras),
        raw_points=raw_points,
        point_colors=point_colors,
        normalization=normalization,
        dataset_identity=_sha256_json(identity_payload),
        file_identities=file_identities,
    )


def classify_sparse_points(
    dataset: StaticDataset,
    *,
    erosion_kernel_size: int = 3,
    chunk_size: int = 16_384,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Classify raw sparse points by eroded FG/BG votes from every training view."""

    if erosion_kernel_size <= 0 or erosion_kernel_size % 2 == 0:
        raise ValueError("erosion_kernel_size must be positive and odd")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    points = dataset.raw_points
    fg_counts = np.zeros(len(points), dtype=np.int32)
    bg_counts = np.zeros(len(points), dtype=np.int32)
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), dtype=np.uint8)
    for camera in dataset.cameras:
        foreground = dataset.load_binary_mask(camera).astype(np.uint8)
        background = 1 - foreground
        foreground = cv2.erode(foreground, kernel, iterations=1) > 0
        background = cv2.erode(background, kernel, iterations=1) > 0
        K = camera.K.numpy().astype(np.float64)
        w2c = camera.raw_world_to_camera.numpy().astype(np.float64)
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]
        for start in range(0, len(points), chunk_size):
            end = min(start + chunk_size, len(points))
            camera_points = points[start:end] @ rotation.T + translation[None]
            projected = camera_points @ K.T
            z = camera_points[:, 2]
            safe_z = np.where(z > 1e-8, projected[:, 2], 1.0)
            u = np.rint(projected[:, 0] / safe_z).astype(np.int64)
            v = np.rint(projected[:, 1] / safe_z).astype(np.int64)
            valid = (
                (z > 1e-8)
                & (u >= 0)
                & (u < camera.width)
                & (v >= 0)
                & (v < camera.height)
            )
            if not valid.any():
                continue
            local = np.flatnonzero(valid)
            global_indices = local + start
            fg_counts[global_indices[foreground[v[local], u[local]]]] += 1
            bg_counts[global_indices[background[v[local], u[local]]]] += 1
    observed = (fg_counts + bg_counts) > 0
    foreground_mask = (fg_counts >= bg_counts) & observed
    foreground_mask |= ~observed
    background_mask = (bg_counts > fg_counts) & observed
    summary = {
        "foreground_points": int(foreground_mask.sum()),
        "background_points": int(background_mask.sum()),
        "unobserved_points": int((~observed).sum()),
    }
    if summary["foreground_points"] < 2 or summary["background_points"] < 2:
        raise ValueError(
            "Sparse-point mask voting requires at least two points in each set; "
            f"got {summary}"
        )
    return foreground_mask, background_mask, summary


def _initialize_gaussian_set(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    maximum_count: int,
    rng: np.random.Generator,
    torch_generator: torch.Generator,
) -> GaussianSet:
    """Initialize direct-RGB 3D Gaussians from one classified point subset."""

    if maximum_count <= 0:
        raise ValueError("maximum_count must be positive")
    if len(points) < 2:
        raise ValueError("Gaussian initialization requires at least two points")
    if len(points) > maximum_count:
        indices = rng.choice(len(points), maximum_count, replace=False)
        points = points[indices]
        colors = colors[indices]
    try:
        spatial_module = importlib.import_module("scipy.spatial")
    except ImportError as error:
        raise RuntimeError("Static Gaussian initialization requires scipy") from error
    ckdtree_type = getattr(spatial_module, "cKDTree", None)
    if ckdtree_type is None:
        raise RuntimeError("scipy.spatial.cKDTree is unavailable")
    neighbor_count = min(4, len(points))
    distances, _ = ckdtree_type(points).query(
        points, k=neighbor_count, workers=-1
    )
    if neighbor_count == 2:
        mean_distance = distances[:, 1]
    else:
        mean_distance = distances[:, 1:].mean(axis=1)
    low, high = np.quantile(mean_distance, [0.05, 0.95])
    scales = np.clip(mean_distance, low, high)
    colors = np.clip(colors, 1e-4, 1.0 - 1e-4)
    count = len(points)
    quaternions = torch.rand((count, 4), generator=torch_generator)
    return GaussianSet(
        means=torch.from_numpy(points.astype(np.float32)),
        quaternions=quaternions,
        log_scales=torch.from_numpy(
            np.log(scales.astype(np.float32))[:, None].repeat(3, axis=1)
        ),
        color_logits=torch.logit(torch.from_numpy(colors.astype(np.float32))),
        opacity_logits=torch.logit(torch.full((count,), 0.7)),
    )


def initialize_static_scene(
    dataset: StaticDataset,
    *,
    num_foreground: int = 40_000,
    num_background: int = 100_000,
    seed: int = 42,
) -> tuple[ForegroundBackgroundScene, dict[str, int]]:
    """Build independently indexed FG/BG Gaussians from the classified sparse cloud."""

    foreground_mask, background_mask, classification = classify_sparse_points(dataset)
    normalized_points = dataset.normalization.normalize_points(dataset.raw_points)
    rng = np.random.default_rng(seed)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed)
    foreground = _initialize_gaussian_set(
        normalized_points[foreground_mask],
        dataset.point_colors[foreground_mask],
        maximum_count=num_foreground,
        rng=rng,
        torch_generator=torch_generator,
    )
    background = _initialize_gaussian_set(
        normalized_points[background_mask],
        dataset.point_colors[background_mask],
        maximum_count=num_background,
        rng=rng,
        torch_generator=torch_generator,
    )
    classification = {
        **classification,
        "initial_foreground_gaussians": foreground.count,
        "initial_background_gaussians": background.count,
    }
    return ForegroundBackgroundScene(foreground, background), classification


def _validate_gaussian_tensors(tensors: Mapping[str, Tensor]) -> None:
    """Enforce the class-free static Gaussian tensor schema."""

    if set(tensors) != set(GAUSSIAN_FIELDS):
        raise ValueError(f"Gaussian fields must be exactly {GAUSSIAN_FIELDS}")
    count = int(tensors["means"].shape[0])
    expected = {
        "means": (count, 3),
        "quaternions": (count, 4),
        "log_scales": (count, 3),
        "color_logits": (count, 3),
        "opacity_logits": (count,),
    }
    if count < 2:
        raise ValueError("A Gaussian set must contain at least two entries")
    for name, value in tensors.items():
        if tuple(value.shape) != expected[name]:
            raise ValueError(f"Invalid {name} shape: {tuple(value.shape)} != {expected[name]}")
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {value.dtype}")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains non-finite values")


def tensor_dictionary_identity(tensors: Mapping[str, Tensor], prefix: str) -> str:
    """Hash ordered tensor names, shapes, dtypes, and raw values."""

    digest = hashlib.sha256()
    for name in sorted(key for key in tensors if key.startswith(prefix)):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_static_scene(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> ForegroundBackgroundScene:
    """Load and verify a pure-tensor static scene bundle without legacy classes."""

    path = Path(path).expanduser().resolve(strict=True)
    manifest_path = path / "manifest.json"
    tensors_path = path / "tensors.pt"
    if not manifest_path.is_file() or not tensors_path.is_file():
        raise FileNotFoundError(f"Incomplete static scene bundle: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "modal_gaussians.static_scene" or manifest.get("version") != 1:
        raise ValueError(f"Unsupported static scene manifest: {manifest_path}")
    if _sha256_file(tensors_path) != manifest.get("tensors_sha256"):
        raise ValueError("Static tensors.pt SHA-256 does not match manifest")
    loaded = torch.load(tensors_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not all(isinstance(value, Tensor) for value in loaded.values()):
        raise ValueError("tensors.pt must contain a plain tensor dictionary")
    expected_keys = {
        f"{part}.{field}" for part in ("foreground", "background") for field in GAUSSIAN_FIELDS
    }
    if set(loaded) != expected_keys:
        raise ValueError(f"Unexpected static tensor keys: {sorted(set(loaded) ^ expected_keys)}")
    if tensor_dictionary_identity(loaded, "foreground.") != manifest.get(
        "foreground_identity"
    ):
        raise ValueError("Foreground tensor identity does not match manifest")
    if tensor_dictionary_identity(loaded, "background.") != manifest.get(
        "background_identity"
    ):
        raise ValueError("Background tensor identity does not match manifest")
    static_identity_payload = {
        "dataset_identity": manifest["dataset"]["dataset_identity"],
        "foreground_identity": manifest["foreground_identity"],
        "background_identity": manifest["background_identity"],
        "normalization": manifest["scene_normalization"],
        "representation": "vanilla_3dgs_direct_rgb",
    }
    if _sha256_json(static_identity_payload) != manifest.get("static_scene_identity"):
        raise ValueError("Static scene identity does not match manifest contents")
    parts: dict[str, GaussianSet] = {}
    for part in ("foreground", "background"):
        raw = {field: loaded[f"{part}.{field}"].float() for field in GAUSSIAN_FIELDS}
        _validate_gaussian_tensors(raw)
        parts[part] = GaussianSet(**raw)
    scene = ForegroundBackgroundScene(
        parts["foreground"], parts["background"], manifest=manifest
    )
    return scene.to(device)


def cameras_from_scene_manifest(manifest: Mapping[str, Any]) -> tuple[Camera, ...]:
    """Load the ordered camera contract embedded in a static bundle manifest."""

    records = manifest.get("cameras")
    if not isinstance(records, list) or not records:
        raise ValueError("Static scene manifest contains no cameras")
    cameras = tuple(Camera.from_manifest_record(record) for record in records)
    if len({camera.name for camera in cameras}) != len(cameras):
        raise ValueError("Static scene manifest contains duplicate camera names")
    return cameras


__all__ = [
    "Camera",
    "ForegroundBackgroundScene",
    "GaussianSet",
    "SceneNormalization",
    "StaticDataset",
    "cameras_from_scene_manifest",
    "classify_sparse_points",
    "compute_scene_normalization",
    "initialize_static_scene",
    "load_static_dataset",
    "load_static_scene",
    "tensor_dictionary_identity",
]
