"""A CPU-only browser for a completed shared-grid spectrum cache."""

from __future__ import annotations

import base64
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import cv2
import numpy as np

from modal_gaussians.spectrum_cache import (
    export_selection,
    load_spectrum,
    read_curves,
    read_mode,
    read_region,
    save_selection,
)


def _png_url(rgb: np.ndarray) -> str:
    image = np.clip(rgb, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the spectrum preview")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _preview_images(mode: np.ndarray, region: np.ndarray, reference: np.ndarray) -> dict:
    """Share one U/V scale; display masking never changes cached coefficients."""
    magnitude = np.abs(mode)
    scale = max(float(np.percentile(magnitude[region], 99)), 1e-12)
    images = {"reference": _png_url(reference)}
    for component, name in enumerate(("u", "v")):
        hsv = np.stack((np.mod(np.angle(mode[..., component]) * 180 / np.pi + 180, 360),
                        np.ones(region.shape, dtype=np.float32),
                        np.clip(magnitude[..., component] / scale, 0, 1)), axis=-1)
        rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB) * 255
        rgb[~region] = reference[~region] * 0.35
        images[name] = _png_url(rgb)
    amplitude = np.linalg.norm(magnitude, axis=-1)
    amplitude_scale = max(float(np.percentile(amplitude[region], 99)), 1e-12)
    gray = np.repeat(np.clip(amplitude / amplitude_scale, 0, 1)[..., None], 3, axis=-1) * 255
    gray[~region] = reference[~region] * 0.35
    images["amplitude"] = _png_url(gray)
    return {"images": images, "uv_scale": scale, "amplitude_scale": amplitude_scale}


def run_spectrum_viewer(*, spectrum_dir: str | Path, work_dir: str | Path,
                        host: str = "127.0.0.1", port: int = 8110) -> None:
    """Serve cached slices and explicit user selections; never compute a transform."""
    cache = load_spectrum(spectrum_dir)
    views = cache.manifest["views"]
    frequencies = cache.frequencies
    work_dir = Path(work_dir).expanduser().resolve()
    page = Path(__file__).with_name("spectrum_web.html").read_bytes()

    @lru_cache(maxsize=2)
    def mode_at(view: int, bin_index: int) -> np.ndarray:
        return read_mode(cache, view, bin_index)

    @lru_cache(maxsize=None)
    def reference_at(view: int) -> np.ndarray:
        path = cache.path / views[view]["reference_image"]
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read the cached reference image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @lru_cache(maxsize=None)
    def region_at(view: int, region: str) -> np.ndarray:
        result = read_region(cache, view, region)
        if not np.any(result):
            raise ValueError(f"The {region} region is empty in {views[view]['label']}")
        return result

    @lru_cache(maxsize=None)
    def curves_at(view: int) -> dict:
        return {key: value.tolist() for key, value in read_curves(cache, view).items()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            pass

        def reply(self, data, *, status=200, content_type="application/json"):
            payload = (json.dumps(data, allow_nan=False).encode("utf-8")
                       if content_type == "application/json" else data)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        @staticmethod
        def index(query, name, limit):
            value = int(query[name][0])
            if not 0 <= value < limit:
                raise ValueError(f"{name} is outside the cached range")
            return value

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                if url.path == "/":
                    self.reply(page, content_type="text/html; charset=utf-8")
                elif url.path == "/api/config":
                    self.reply({"frequencies": frequencies.tolist(),
                                "views": [{"label": v["label"], "shape_hw": v["shape_hw"]} for v in views],
                                "identity": cache.manifest["spectrum_identity"],
                                "initial_bin": 18 if len(frequencies) > 18 else min(1, len(frequencies) - 1)})
                elif url.path in ("/api/view", "/api/pixel"):
                    view = self.index(query, "view", len(views))
                    bin_index = self.index(query, "bin", len(frequencies))
                    mode = mode_at(view, bin_index)
                    if url.path == "/api/view":
                        region = query["region"][0]
                        if region not in ("selected_box", "full_frame"):
                            raise ValueError("Unknown spectrum region")
                        payload = _preview_images(mode, region_at(view, region), reference_at(view))
                        payload.update({"curve": curves_at(view)[region], "view": view,
                                        "bin": bin_index, "frequency_hz": float(frequencies[bin_index])})
                    else:
                        y = self.index(query, "y", mode.shape[0])
                        x = self.index(query, "x", mode.shape[1])
                        payload = {"x": x, "y": y, "components": {
                            name: {"real": float(z.real), "imag": float(z.imag),
                                   "amplitude": float(abs(z)), "phase_rad": float(np.angle(z))}
                            for name, z in zip(("u", "v"), mode[y, x])}}
                    self.reply(payload)
                else:
                    self.reply({"error": "Unknown endpoint"}, status=404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, IndexError, TypeError) as error:
                self.reply({"error": str(error)}, status=400)
            except Exception as error:
                self.reply({"error": str(error)}, status=500)

        def do_POST(self):
            try:
                if self.path not in ("/api/save", "/api/export"):
                    self.reply({"error": "Unknown endpoint"}, status=404)
                    return
                origin = self.headers.get("Origin")
                if ((origin and urlsplit(origin).netloc != self.headers.get("Host")) or
                        self.headers.get("Sec-Fetch-Site") == "cross-site"):
                    self.reply({"error": "Cross-origin requests are not allowed"}, status=403)
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(size))
                bins = request["bins"]
                if (not isinstance(bins, list) or not bins or len(bins) > len(frequencies) or
                        any(type(b) is not int or not 0 < b < len(frequencies) for b in bins)):
                    raise ValueError("Select valid nonzero integer bins")
                # Each click gets a unique destination; no client-supplied filesystem paths.
                run_dir = work_dir / ("selection_" + uuid4().hex[:12])
                run_dir.mkdir(parents=True, exist_ok=False)
                selection = save_selection(cache, sorted(set(bins)), run_dir / "selection.json")
                result = {"selection": str(selection)}
                if self.path == "/api/export":
                    result["export"] = str(export_selection(cache, selection, run_dir / "modal_images"))
                self.reply(result)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, IndexError, TypeError) as error:
                self.reply({"error": str(error)}, status=400)
            except Exception as error:
                self.reply({"error": str(error)}, status=500)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Spectrum viewer: http://{host}:{port} (Ctrl+C to stop)", flush=True)
    print(f"Selections and exports: {work_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
