"""Select evenly spaced positive bins on the shared FFT grid."""
import numpy as np
from modal_gaussians.common.scene_store import resolve_path
from modal_gaussians.spectrum.cache import load_spectrum, save_selection

def uniform_bins(candidate_count, count):
    if not 1 <= count <= candidate_count:
        raise ValueError("Selection count exceeds the positive FFT bins")
    # Equal intervals over (0, Nyquist]; rounding is explicit grid sampling,
    # not the manual GUI's forbidden automatic frequency snapping.
    return np.rint(np.arange(1, count + 1) * candidate_count / count).astype(np.int64)

def select_spectrum(*, spectrum_dir, count, output_dir, max_frequency_hz=None):
    cache = load_spectrum(spectrum_dir)
    limit = cache.frequencies[-1] if max_frequency_hz is None else float(max_frequency_hz)
    if not np.isfinite(limit) or not 0 < limit <= cache.frequencies[-1]:
        raise ValueError('Frequency limit must be positive and at most Nyquist')
    candidate_count = int(np.searchsorted(cache.frequencies, limit, side='right')) - 1
    bins = uniform_bins(candidate_count, count)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_selection(cache, bins, output / 'selection.json', preserve_order=True)
    return output
