"""New-method Paulstretch implementation with onset detection and a CLI entry point."""

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
    onset_level: float,
    outfilename: str,
    plot_onsets: bool = False,
) -> None:
    """Stretch a stereo audio signal using the new Paulstretch method.

    Unlike the standard method, this variant detects onsets in the frequency
    domain and temporarily increases the effective stretch around them to
    preserve transient detail.

    Args:
        samplerate: Sample rate of the input audio in Hz.
        smp: 2-D float64 array of shape ``(channels, num_samples)`` normalised
            to ``[-1.0, 1.0]``.
        stretch: Stretch factor (``1.0`` = no stretch, ``8.0`` = 8× slower).
        windowsize_seconds: Analysis window size in seconds.
        onset_level: Onset detection sensitivity threshold in the range
            ``[0.0, 1.0]``.  Lower values make the detector more sensitive.
        outfilename: Path to the output WAV file.
        plot_onsets: When ``True``, display a matplotlib plot of the detected
            onset strengths after processing.  Requires *matplotlib* to be
            installed.
    """
    onsets: list[float] = []

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
    displace_pos = windowsize * 0.5

    # Create Hann window.
    window = (
        0.5
        - np.cos(
            np.arange(windowsize, dtype="float") * 2.0 * np.pi / (windowsize - 1)
        )
        * 0.5
    )

    old_windowed_buf = np.zeros((nchannels, windowsize))
    hinv_sqrt2 = (1 + np.sqrt(0.5)) * 0.5
    hinv_buf = 2.0 * (
        hinv_sqrt2
        - (1.0 - hinv_sqrt2)
        * np.cos(
            np.arange(half_windowsize, dtype="float") * 2.0 * np.pi / half_windowsize
        )
    ) / hinv_sqrt2

    freqs = np.zeros((nchannels, half_windowsize + 1))
    old_freqs = freqs

    num_bins_scaled_freq = 32
    freqs_scaled = np.zeros(num_bins_scaled_freq)
    old_freqs_scaled = freqs_scaled

    displace_tick = 0.0
    displace_tick_increase = 1.0 / stretch
    if displace_tick_increase > 1.0:
        displace_tick_increase = 1.0
    extra_onset_time_credit = 0.0
    get_next_buf = True

    while True:
        if get_next_buf:
            old_freqs = freqs
            old_freqs_scaled = freqs_scaled

            # Get the windowed buffer.
            istart_pos = int(np.floor(start_pos))
            buf = smp[:, istart_pos : istart_pos + windowsize]
            if buf.shape[1] < windowsize:
                buf = np.append(
                    buf, np.zeros((nchannels, windowsize - buf.shape[1])), 1
                )
            buf = buf * window

            # Get the amplitudes of the frequency components and discard phases.
            freqs = np.abs(np.fft.rfft(buf))

            # Scale down the spectrum to detect onsets.
            freqs_len = freqs.shape[1]
            if num_bins_scaled_freq < freqs_len:
                freqs_len_div = freqs_len // num_bins_scaled_freq
                new_freqs_len = freqs_len_div * num_bins_scaled_freq
                freqs_scaled = np.mean(
                    np.mean(freqs, 0)[:new_freqs_len].reshape(
                        [num_bins_scaled_freq, freqs_len_div]
                    ),
                    1,
                )
            else:
                freqs_scaled = np.zeros(num_bins_scaled_freq)

            # Process onsets.
            m = 2.0 * np.mean(freqs_scaled - old_freqs_scaled) / (
                np.mean(np.abs(old_freqs_scaled)) + 1e-3
            )
            if m < 0.0:
                m = 0.0
            if m > 1.0:
                m = 1.0
            if plot_onsets:
                onsets.append(m)
            if m > onset_level:
                displace_tick = 1.0
                extra_onset_time_credit += 1.0

        cfreqs = (freqs * displace_tick) + (old_freqs * (1.0 - displace_tick))

        # Randomize the phases by multiplication with a random complex number
        # with modulus = 1.
        ph = np.random.uniform(0, 2 * np.pi, (nchannels, cfreqs.shape[1])) * 1j
        cfreqs = cfreqs * np.exp(ph)

        # Do the inverse FFT.
        buf = np.fft.irfft(cfreqs)

        # Window again the output buffer.
        buf *= window

        # Overlap-add the output.
        output = (
            buf[:, 0:half_windowsize]
            + old_windowed_buf[:, half_windowsize:windowsize]
        )
        old_windowed_buf = buf

        # Remove the resulting amplitude modulation.
        output *= hinv_buf

        # Clamp the values to -1..1.
        output[output > 1.0] = 1.0
        output[output < -1.0] = -1.0

        # Write the output to the WAV file.
        outfile.writeframes(np.int16(output.ravel("F") * 32767.0).tobytes())

        if get_next_buf:
            start_pos += displace_pos

        get_next_buf = False

        if start_pos >= nsamples:
            print("100 %")
            break
        sys.stdout.write("%d %% \r" % int(100.0 * start_pos / nsamples))
        sys.stdout.flush()

        if extra_onset_time_credit <= 0.0:
            displace_tick += displace_tick_increase
        else:
            # 0.5 multiplier ensures credit_get < displace_tick_increase
            credit_get = 0.5 * displace_tick_increase
            extra_onset_time_credit -= credit_get
            if extra_onset_time_credit < 0:
                extra_onset_time_credit = 0
            displace_tick += displace_tick_increase - credit_get

        if displace_tick >= 1.0:
            displace_tick = displace_tick % 1.0
            get_next_buf = True

    outfile.close()

    if plot_onsets:
        import matplotlib.pyplot as plt

        plt.plot(onsets)
        plt.show()


def main() -> None:
    """Parse command-line arguments and run the new-method Paulstretch algorithm."""
    print("Paul's Extreme Sound Stretch (Paulstretch) - Python version 20141220")
    print("new method: using onsets information")
    print("by Nasca Octavian PAUL, Targu Mures, Romania\n")

    parser = argparse.ArgumentParser(
        description="Paul's Extreme Sound Stretch (new method with onset detection)",
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
    parser.add_argument(
        "-t",
        "--onset",
        dest="onset",
        help="onset sensitivity (0.0=max, 1.0=min)",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--plot-onsets",
        dest="plot_onsets",
        help="plot the onset strengths after processing (requires matplotlib)",
        action="store_true",
        default=False,
    )
    parser.add_argument("input_wav", help="input WAV file")
    parser.add_argument("output_wav", help="output WAV file")
    args = parser.parse_args()

    if args.stretch <= 0.0 or args.window_size <= 0.001:
        parser.error("stretch must be > 0 and window_size must be > 0.001")

    print("stretch amount = %g" % args.stretch)
    print("window size = %g seconds" % args.window_size)
    print("onset sensitivity = %g" % args.onset)

    result = load_wav_stereo(args.input_wav)
    if result is None:
        sys.exit(1)
    samplerate, smp = result

    paulstretch(
        samplerate,
        smp,
        args.stretch,
        args.window_size,
        args.onset,
        args.output_wav,
        plot_onsets=args.plot_onsets,
    )


if __name__ == "__main__":
    main()
