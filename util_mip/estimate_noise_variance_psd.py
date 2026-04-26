import numpy as np
from scipy.signal import periodogram

def estimate_noise_variance_psd(y, high_freq_range=(0.5, 1.0)):
    r"""
    Estimates observational white noise variance using the high-frequency tail
    of the Power Spectral Density (PSD).
    """

    r = np.asarray(y['dff'])
    fs = y['frame_rate']
    f, Pxx = periodogram(r, fs=fs)
    nyquist = fs / 2.0

    f_lower = nyquist * high_freq_range[0]
    f_upper = nyquist * high_freq_range[1]

    high_freq_mask = (f >= f_lower) & (f <= f_upper)
    mean_high_freq_power = np.mean(Pxx[high_freq_mask])

    # Estimate variance: mean power * Nyquist bandwidth
    return mean_high_freq_power * nyquist
