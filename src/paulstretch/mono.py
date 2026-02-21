"""Mono implementation of the Paulstretch algorithm with a CLI entry point."""

import argparse
import sys
import wave

import numpy as np

from paulstretch.core import load_wav_mono


def paulstretch(
    samplerate: int,
    smp: np.ndarray,
    stretch: float,
    windowsize_seconds: float,
    outfilename: str,
) -> None:
    """Stretch a mono audio signal using the Paulstretch algorithm.

    The input samples are read from *smp* and the stretched audio is written
    to *outfilename* as a 16-bit PCM WAV file.

    Args:
        samplerate: Sample rate of the input audio in Hz.
        smp: 1-D float64 array of audio samples normalised to ``[-1.0, 1.0]``.
        stretch: Stretch factor (``1.0`` = no stretch, ``8.0`` = 8× slower).
        windowsize_seconds: Analysis window size in seconds.
        outfilename: Path to the output WAV file.
    """
    outfile = wave.open(outfilename, "wb")
    outfile.setsampwidth(2)
    outfile.setframerate(samplerate)
    outfile.setnchannels(1)

    # Make sure that windowsize is even and larger than 16.
    windowsize = int(windowsize_seconds * samplerate)
    if windowsize < 16:
        windowsize = 16
    windowsize = int(windowsize / 2) * 2
    half_windowsize = int(windowsize / 2)

    # Correct the end of the smp.
    end_size = int(samplerate * 0.05)
    if end_size < 16:
        end_size = 16
    smp[len(smp) - end_size : len(smp)] *= np.linspace(1, 0, end_size)

    # Compute the displacement inside the input file.
    start_pos = 0.0
    displace_pos = (windowsize * 0.5) / stretch

    # Create Hann window.
    window = (
        0.5
        - np.cos(np.arange(windowsize, dtype="float") * 2.0 * np.pi / (windowsize - 1))
        * 0.5
    )

    old_windowed_buf = np.zeros(windowsize)
    hinv_sqrt2 = (1 + np.sqrt(0.5)) * 0.5
    hinv_buf = hinv_sqrt2 - (1.0 - hinv_sqrt2) * np.cos(
        np.arange(half_windowsize, dtype="float") * 2.0 * np.pi / half_windowsize
    )

    while True:
        # Get the windowed buffer.
        istart_pos = int(np.floor(start_pos))
        buf = smp[istart_pos : istart_pos + windowsize]
        if len(buf) < windowsize:
            buf = np.append(buf, np.zeros(windowsize - len(buf)))
        buf = buf * window

        # Get the amplitudes of the frequency components and discard the phases.
        freqs = np.abs(np.fft.rfft(buf))

        # Randomize the phases by multiplication with a random complex number
        # with modulus = 1.
        ph = np.random.uniform(0, 2 * np.pi, len(freqs)) * 1j
        freqs = freqs * np.exp(ph)

        # Do the inverse FFT.
        buf = np.fft.irfft(freqs)

        # Window again the output buffer.
        buf *= window

        # Overlap-add the output.
        output = buf[0:half_windowsize] + old_windowed_buf[half_windowsize:windowsize]
        old_windowed_buf = buf

        # Remove the resulting amplitude modulation.
        output *= hinv_buf

        # Clamp the values to -1..1.
        output[output > 1.0] = 1.0
        output[output < -1.0] = -1.0

        # Write the output to the WAV file.
        outfile.writeframes(np.int16(output * 32767.0).tobytes())

        start_pos += displace_pos
        if start_pos >= len(smp):
            print("100 %")
            break
        sys.stdout.write("%d %% \r" % int(100.0 * start_pos / len(smp)))
        sys.stdout.flush()

    outfile.close()


def main() -> None:
    """Parse command-line arguments and run the mono Paulstretch algorithm."""
    print("Paul's Extreme Sound Stretch (Paulstretch) - Python version 20141220")
    print("Mono version")
    print("by Nasca Octavian PAUL, Targu Mures, Romania\n")

    parser = argparse.ArgumentParser(
        description="Paul's Extreme Sound Stretch (mono version)",
        usage="%(prog)s [options] input_wav output_wav",
    )
    parser.add_argument(
        "-s",
        "--stretch",
        dest="stretch",
        help="stretch amount (1.0 = no stretch)",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "-w",
        "--window_size",
        dest="window_size",
        help="window size (seconds)",
        type=float,
        default=0.25,
    )
    parser.add_argument("input_wav", help="input WAV file")
    parser.add_argument("output_wav", help="output WAV file")
    args = parser.parse_args()

    if args.stretch <= 0.0 or args.window_size <= 0.001:
        parser.error("stretch must be > 0 and window_size must be > 0.001")

    print("stretch amount = %g" % args.stretch)
    print("window size = %g seconds" % args.window_size)

    result = load_wav_mono(args.input_wav)
    if result is None:
        sys.exit(1)
    samplerate, smp = result

    paulstretch(samplerate, smp, args.stretch, args.window_size, args.output_wav)


if __name__ == "__main__":
    main()
