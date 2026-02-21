"""Stereo implementation of the Paulstretch algorithm with a CLI entry point."""

import argparse
import sys
import wave

import numpy as np

from paulstretch.core import load_wav_stereo, optimize_windowsize


def paulstretch(
    samplerate: int,
    smp: np.ndarray,
    stretch: float,
    windowsize_seconds: float,
    outfilename: str,
) -> None:
    """Stretch a stereo audio signal using the Paulstretch algorithm.

    The input samples are read from *smp* and the stretched audio is written
    to *outfilename* as a 16-bit PCM WAV file.

    Args:
        samplerate: Sample rate of the input audio in Hz.
        smp: 2-D float64 array of shape ``(channels, num_samples)`` normalised
            to ``[-1.0, 1.0]``.
        stretch: Stretch factor (``1.0`` = no stretch, ``8.0`` = 8× slower).
        windowsize_seconds: Analysis window size in seconds.
        outfilename: Path to the output WAV file.
    """
    nchannels = smp.shape[0]

    outfile = wave.open(outfilename, "wb")
    outfile.setsampwidth(2)
    outfile.setframerate(samplerate)
    outfile.setnchannels(nchannels)

    # Make sure that windowsize is even and larger than 16.
    windowsize = int(windowsize_seconds * samplerate)
    if windowsize < 16:
        windowsize = 16
    windowsize = optimize_windowsize(windowsize)
    windowsize = int(windowsize / 2) * 2
    half_windowsize = int(windowsize / 2)

    # Correct the end of the smp.
    nsamples = smp.shape[1]
    end_size = int(samplerate * 0.05)
    if end_size < 16:
        end_size = 16

    smp[:, nsamples - end_size : nsamples] *= np.linspace(1, 0, end_size)

    # Compute the displacement inside the input file.
    start_pos = 0.0
    displace_pos = (windowsize * 0.5) / stretch

    # Create window (approximates a Hann window with steeper roll-off).
    window = np.power(
        1.0 - np.power(np.linspace(-1.0, 1.0, windowsize), 2.0), 1.25
    )

    old_windowed_buf = np.zeros((nchannels, windowsize))

    while True:
        # Get the windowed buffer.
        istart_pos = int(np.floor(start_pos))
        buf = smp[:, istart_pos : istart_pos + windowsize]
        if buf.shape[1] < windowsize:
            buf = np.append(buf, np.zeros((nchannels, windowsize - buf.shape[1])), 1)
        buf = buf * window

        # Get the amplitudes of the frequency components and discard the phases.
        freqs = np.abs(np.fft.rfft(buf))

        # Randomize the phases by multiplication with a random complex number
        # with modulus = 1.
        ph = np.random.uniform(0, 2 * np.pi, (nchannels, freqs.shape[1])) * 1j
        freqs = freqs * np.exp(ph)

        # Do the inverse FFT.
        buf = np.fft.irfft(freqs)

        # Window again the output buffer.
        buf *= window

        # Overlap-add the output.
        output = (
            buf[:, 0:half_windowsize]
            + old_windowed_buf[:, half_windowsize:windowsize]
        )
        old_windowed_buf = buf

        # Clamp the values to -1..1.
        output[output > 1.0] = 1.0
        output[output < -1.0] = -1.0

        # Write the output to the WAV file.
        outfile.writeframes(np.int16(output.ravel("F") * 32767.0).tobytes())

        start_pos += displace_pos
        if start_pos >= nsamples:
            print("100 %")
            break
        sys.stdout.write("%d %% \r" % int(100.0 * start_pos / nsamples))
        sys.stdout.flush()

    outfile.close()


def main() -> None:
    """Parse command-line arguments and run the stereo Paulstretch algorithm."""
    print("Paul's Extreme Sound Stretch (Paulstretch) - Python version 20141220")
    print("by Nasca Octavian PAUL, Targu Mures, Romania\n")

    parser = argparse.ArgumentParser(
        description="Paul's Extreme Sound Stretch",
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

    result = load_wav_stereo(args.input_wav)
    if result is None:
        sys.exit(1)
    samplerate, smp = result

    paulstretch(samplerate, smp, args.stretch, args.window_size, args.output_wav)


if __name__ == "__main__":
    main()
