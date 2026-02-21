"""Shared utilities for the Paulstretch algorithm."""

import numpy as np
import scipy.io.wavfile


def load_wav_mono(filename: str) -> tuple[int, np.ndarray] | None:
    """Load a WAV file and return it as a mono audio array.

    If the input file is stereo, the channels are averaged to produce mono.

    Args:
        filename: Path to the input WAV file.

    Returns:
        A ``(samplerate, samples)`` tuple where *samples* is a 1-D float64
        array normalised to the range ``[-1.0, 1.0]``, or ``None`` if the
        file could not be read.
    """
    try:
        wavedata = scipy.io.wavfile.read(filename)
        samplerate = int(wavedata[0])
        smp = wavedata[1] * (1.0 / 32768.0)
        if len(smp.shape) > 1:  # convert to mono
            smp = (smp[:, 0] + smp[:, 1]) * 0.5
        return (samplerate, smp)
    except Exception:
        print("Error loading wav: " + filename)
        return None


def load_wav_stereo(filename: str) -> tuple[int, np.ndarray] | None:
    """Load a WAV file and return it as a stereo audio array.

    If the input file is mono, it is duplicated into two identical channels.

    Args:
        filename: Path to the input WAV file.

    Returns:
        A ``(samplerate, samples)`` tuple where *samples* is a 2-D float64
        array of shape ``(channels, num_samples)`` normalised to
        ``[-1.0, 1.0]``, or ``None`` if the file could not be read.
    """
    try:
        wavedata = scipy.io.wavfile.read(filename)
        samplerate = int(wavedata[0])
        smp = wavedata[1] * (1.0 / 32768.0)
        smp = smp.transpose()
        if len(smp.shape) == 1:  # convert to stereo
            smp = np.tile(smp, (2, 1))
        return (samplerate, smp)
    except Exception:
        print("Error loading wav: " + filename)
        return None


def optimize_windowsize(n: int) -> int:
    """Return the smallest integer >= *n* whose only prime factors are 2, 3, or 5.

    Using such a *highly composite* number as the FFT window size improves
    transform efficiency.

    Args:
        n: The minimum desired window size.

    Returns:
        The optimised window size.
    """
    orig_n = n
    while True:
        n = orig_n
        while (n % 2) == 0:
            n //= 2
        while (n % 3) == 0:
            n //= 3
        while (n % 5) == 0:
            n //= 5

        if n < 2:
            break
        orig_n += 1
    return orig_n
