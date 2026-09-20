"""Scene-owned storage and I/O relocation; saved artifact identities stay unchanged."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "scene_library"


def library_root() -> Path:
    return Path(os.environ.get("MODAL_GAUSSIANS_LIBRARY", DEFAULT_ROOT)).expanduser().resolve()


@lru_cache(maxsize=8)
def _read_registry(filename: str, stamp: tuple[int, int]) -> dict:
    record = json.loads(Path(filename).read_text(encoding="utf-8"))
    if record.get("format") != "modal_gaussians.scene_library" or record.get("version") != 1:
        raise ValueError("Unsupported scene library registry")
    root = Path(filename).parent
    for key in ("locations", "local_routes"):
        compiled = []
        for old, new in sorted(record.get(key, []), key=lambda row: -len(row[0])):
            target = (root / new).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Scene library location escapes its root")
            source = (Path(old) if key == "locations" else root / old).expanduser().resolve()
            compiled.append((source, target, old))
        record["_" + key] = compiled
    return record


def registry() -> dict:
    path = library_root() / "registry.json"
    if not path.is_file():
        return {}
    stat = path.stat()
    return _read_registry(str(path), (stat.st_mtime_ns, stat.st_size))


def _relative(path: Path, parent: Path):
    try:
        return path.relative_to(parent)
    except ValueError:
        return None


def _destination(relative: str) -> Path:
    root = library_root()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Scene library location escapes its root")
    return path


def resolve_path(value, *, strict=False) -> Path:
    """Resolve physical I/O locations without editing strings in source manifests."""
    path = Path(value).expanduser().resolve()
    record = registry()
    # Longest prefix wins (e.g. a modal-image export moved out of its old batch).
    for key in ("_locations", "_local_routes"):
        for old, new, _ in record.get(key, []):
            relative = _relative(path, old)
            if relative is not None:
                return (new / relative).resolve(strict=strict)
    return path.resolve(strict=strict)


def logical_path(value) -> Path:
    """The original identity-bearing location of an imported artifact."""
    path = resolve_path(value)
    for _, new, old in sorted(registry().get("_locations", []), key=lambda x: -len(str(x[1]))):
        relative = _relative(path, new)
        if relative is not None:
            return Path(old) / relative
    return path


def scene_cache(source: dict, fallback) -> Path:
    key = source.get("static_scene_identity")
    for scene in registry().get("scenes", {}).values():
        if key in scene.get("static_scene_identities", []):
            return _destination(scene["cache"])
    return resolve_path(fallback)


def scene_record(name: str) -> dict:
    try:
        return registry()["scenes"][name]
    except KeyError:
        raise ValueError(f"Scene is not registered: {name}") from None


def asset_path(scene: str, name: str) -> Path:
    if name.startswith("mode:"):
        frequency = float(name.split(":", 1)[1])
        modes = [mode for mode in scene_record(scene).get("modes", [])
                 if abs(mode["frequency_hz"] - frequency) < 1e-9]
        if len(modes) != 1:
            raise ValueError(f"No unique saved mode at {frequency:g} Hz for {scene}")
        return _destination(modes[0]["completed_modes"])
    assets = scene_record(scene)["assets"]
    if name not in assets:
        raise ValueError(f"Unknown {scene} asset {name!r}; choose from {', '.join(sorted(assets))}")
    return _destination(assets[name])


def list_scene(scene: str | None = None) -> dict:
    names = [scene] if scene else sorted(registry().get("scenes", {}))
    result = {}
    for name in names:
        record = scene_record(name)
        result[name] = {**record,
            "assets": {key: str(_destination(value)) for key, value in record["assets"].items()},
            "cache": str(_destination(record["cache"]))}
    return result


def expand_arguments(scene: str, arguments: list[str]) -> list[str]:
    """@prepared, @spectrum, @baseline, etc. are explicit scene asset references."""
    result = []
    for word in arguments:
        if word.startswith("@"):
            asset, _, suffix = word[1:].replace("\\", "/").partition("/")
            base = asset_path(scene, asset)
            path = (base / suffix).resolve()
            if not path.is_relative_to(base):
                raise ValueError("Scene asset reference escapes its root")
            word = str(path)
        result.append(word)
    return result
