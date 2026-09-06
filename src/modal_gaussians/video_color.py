"""Deterministic HDR base-layer conversion for SDR PNG preparation.

BT.2100 HLG (1000-nit reference display) / ST.2084 PQ are decoded in linear
light, converted to BT.709 primaries, and mapped with a fixed Hable shoulder.
No frame statistics, auto-exposure, or Dolby Vision dynamic metadata are used.
FFmpeg supplies full-range, transfer-encoded RGB48; zscale is not required.

Transfer references: ITU-R BT.2100 (HLG inverse OETF and OOTF), SMPTE ST.2084
(PQ EOTF). The Hable curve follows the rational filmic operator also used by
FFmpeg's vf_tonemap, normalized here to a fixed 1000-nit white.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


_TO_709 = np.array([
    [1.6604910, -0.5876411, -0.0728499],
    [-0.1245505, 1.1328999, -0.0083494],
    [-0.0181508, -0.1005789, 1.1187297],
], dtype=np.float32)
_HLG_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
_UNKNOWN = {None, '', 'unknown', 'unspecified', 'reserved'}


@dataclass(frozen=True)
class VideoColorConversion:
    requested_mode: str
    transfer: str | None
    primaries: str = 'bt2020'
    matrix: str = 'bt2020nc'
    source_range: str = 'limited'
    dolby_vision: bool = False

    def decode_filters(self, height: int | None, fps: float) -> list[str]:
        if self.transfer is None:
            # Preserve the previous SDR conversion, filter order, and precision.
            return ([f'scale=-1:{height}'] if height is not None else []) + [
                f'fps={fps:.12g}', 'format=rgb24',
            ]
        size = f'w=-1:h={height}' if height is not None else 'w=iw:h=ih'
        # swscale changes size and the YCbCr matrix/range only. Keep the HDR
        # transfer and wide primaries until the floating-point conversion below.
        return [f'fps={fps:.12g}',
                f'scale={size}:in_color_matrix={self.matrix}:in_range={self.source_range}:out_range=full',
                'format=rgb48be']

    def metadata(self) -> dict[str, Any]:
        return {
            'requested_mode': self.requested_mode,
            'applied': self.transfer is not None,
            'algorithm': 'bt2100_hable_srgb_v1' if self.transfer else 'legacy_rgb24',
            'source_transfer': self.transfer,
            'source_primaries': self.primaries if self.transfer else None,
            'decode_matrix': self.matrix if self.transfer else None,
            'decode_range': self.source_range if self.transfer else None,
            'hdr_reference_white_nits': 1000 if self.transfer else None,
            'sdr_reference_white_nits': 100 if self.transfer else None,
            'output_color_space': 'sRGB' if self.transfer else 'unchanged',
            'dolby_vision_base_layer_only': self.dolby_vision and self.transfer is not None,
            'frame_adaptive': False,
        }


def resolve_video_color(info: dict[str, Any], mode: str = 'auto') -> VideoColorConversion:
    """Detect HDR from transfer/DOVI metadata, never from filename or bit depth."""
    if mode not in {'auto', 'off', 'hlg', 'pq'}:
        raise ValueError('Video color mode must be auto, off, hlg, or pq')
    if mode == 'off':
        return VideoColorConversion(mode, None)
    stream = info['streams'][0]
    transfer = {'arib-std-b67': 'hlg', 'smpte2084': 'pq'}.get(stream.get('color_transfer'))
    dovi = next((s for s in stream.get('side_data_list', []) if
                 s.get('side_data_type') == 'DOVI configuration record'), None)
    if dovi:
        profile = int(dovi.get('dv_profile', -1))
        compatibility = int(dovi.get('dv_bl_signal_compatibility_id', -1))
        if profile != 8 or compatibility not in {1, 2, 4}:
            raise ValueError('This Dolby Vision profile has no supported HLG/HDR10/SDR base layer; '
                             'export an SDR or HLG-compatible video before extraction')
        base_transfer = {1: 'pq', 2: None, 4: 'hlg'}[compatibility]
        if mode == 'auto':
            if stream.get('color_transfer') not in _UNKNOWN and transfer != base_transfer:
                raise ValueError('Video transfer metadata conflicts with its Dolby Vision base layer')
            transfer = base_transfer
    if mode != 'auto':
        transfer = mode
    if transfer is None:
        return VideoColorConversion(mode, None, dolby_vision=bool(dovi))
    primaries = stream.get('color_primaries')
    if primaries in _UNKNOWN:
        primaries = 'bt2020'  # Explicit HLG/PQ or profile-8 compatibility evidence.
    if primaries not in {'bt2020', 'bt709'}:
        raise ValueError(f'Unsupported HDR color primaries: {primaries}')
    matrix = stream.get('color_space')
    if matrix in _UNKNOWN:
        matrix = 'bt2020nc' if primaries == 'bt2020' else 'bt709'
    if matrix not in {'bt2020nc', 'bt709'}:
        raise ValueError(f'Unsupported HDR YCbCr matrix: {matrix}')
    color_range = stream.get('color_range')
    if color_range not in _UNKNOWN | {'tv', 'pc'}:
        raise ValueError(f'Unsupported HDR video range: {color_range}')
    return VideoColorConversion(mode, transfer, primaries, matrix,
                                'full' if color_range == 'pc' else 'limited', bool(dovi))


def hdr_linear_nits(encoded: np.ndarray, transfer: str, primaries: str = 'bt2020') -> np.ndarray:
    """BT.2100 HLG EOTF with OOTF, or absolute ST.2084 EOTF, black at zero."""
    value = np.asarray(encoded, dtype=np.float32)
    if transfer == 'hlg':
        a = 0.17883277
        b, c = 1 - 4 * a, 0.5 - a * np.log(4 * a)
        scene = np.where(value <= 0.5, value * value / 3,
                         (np.exp((value - c) / a) + b) / 12)
        luma_weights = _HLG_LUMA if primaries == 'bt2020' else np.array([0.2126, 0.7152, 0.0722])
        luminance = scene @ luma_weights.astype(np.float32)
        return 1000 * scene * np.maximum(luminance, 0)[..., None] ** 0.2
    if transfer == 'pq':
        m1, m2 = 2610 / 16384, 2523 / 32
        c1, c2, c3 = 3424 / 4096, 2413 / 128, 2392 / 128
        power = value ** (1 / m2)
        return 10000 * (np.maximum(power - c1, 0) / (c2 - c3 * power)) ** (1 / m1)
    raise ValueError(f'Unknown HDR transfer: {transfer}')


def _hable(value: np.ndarray | float) -> np.ndarray | float:
    return ((value * (0.15 * value + 0.05) + 0.004)
            / (value * (0.15 * value + 0.5) + 0.06)) - 1 / 15


def tone_map_rgb16(rgb: np.ndarray, conversion: VideoColorConversion) -> np.ndarray:
    """Convert one full-range RGB uint16 frame to uint8 sRGB without temporal state."""
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('HDR conversion requires a three-channel uint16 RGB frame')
    if conversion.transfer is None:
        raise ValueError('HDR conversion needs an HLG or PQ source')
    linear = hdr_linear_nits(rgb.astype(np.float32) / 65535, conversion.transfer, conversion.primaries) / 100
    if conversion.primaries == 'bt2020':
        linear = linear @ _TO_709.T
    # Clip negative out-of-sRGB-gamut components. A common RGB scale preserves
    # the remaining linear hue ratios, unlike separate per-channel tone curves.
    np.maximum(linear, 0, out=linear)
    peak = np.max(linear, axis=-1, keepdims=True)
    mapped = _hable(peak) / _hable(10.0)  # Fixed 1000-nit white, never per-frame maxima.
    linear *= mapped / np.maximum(peak, 1e-8)
    np.clip(linear, 0, 1, out=linear)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * linear ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)
