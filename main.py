# coding: utf-8
"""Very quick hack of a time series cleaning script.

To use with a time series in a file, set the path of the file to load
on line 146

Note: that with irregular time series the maximum frequency and frequency
increment used to generate the spectrum are hard to define programmatically
so some experimentation will likely be needed for these parameters.

For very poorly sampled time series it will likely also be necessary to
decrease CLEAN gain and up number of CLEAN iterations.

usage:
    $ python3 main.py

.. moduleauthor:: Benjamin Mort <benjamin.mort@oerc.ox.ac.uk>
"""

import logging
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from spectclean.time_series import load_times, sim_time_series

# Require python3
if sys.version_info[0] != 3 or sys.version_info[1] < 5:
    print('This script requires Python version >= 3.5')
    sys.exit(1)


def get_input_time_series():
    """Obtain time series to analyse."""
    log = logging.getLogger()

    # Generate test time series.
    length_days = 4000
    min_sample_interval = 0.1
    sample_fraction = 0.01
    noise_std = 0.1
    duty_period = 50
    duty_length = 20
    num_samples = (((length_days / min_sample_interval) * sample_fraction) *
                   (duty_period / duty_length))
    num_samples = int(num_samples)

    # Amplitude, frequency (days^-1), phase (fraction of 2pi)
    signals = [
        dict(amp=0.15, freq=0.01, phase=0.0),
        dict(amp=0.5, freq=0.01333, phase=0.0),
        dict(amp=0.8, freq=0.02, phase=-0.2),
        dict(amp=0.2, freq=0.04, phase=0.0),
        dict(amp=1.5, freq=0.05, phase=0.25),
    ]
    log.info('* Generating time series...')
    times, values = sim_time_series(length_days, min_sample_interval,
                                    num_samples, signals, noise_std)

    duty_samples = np.ones_like(times)
    duty_samples[np.mod(range(times.size), duty_period) > duty_length] = 0
    times = times[duty_samples == 1]
    values = values[duty_samples == 1]
    log.debug('- No. times after duty sampling = %i', times.size)

    return times, values, signals


def plot_time_series(axis, times, values):
    """Plot the time series.

    Args:
        axis: matplotlib axis object.
        times (numpy.ndarray): Time series times
        values (numpy.ndarray): Time series amplitudes
    """
    axis.plot(times, values, 'k.', ms=3)
    axis.set_xlabel('Time [days]')
    axis.set_ylabel('Time series amplitude')
    axis.grid(True)


def plot_dirty_spectrum(axis, freqs, amps, signals=None):
    """Plot the dirty spectrum.

    Args:
        axis: matplotlib axis object
        freqs (numpy.ndarray): Array of spectrum frequencies
        amps (numpy.ndarray): Array of spectrum amplitudes
        signals (list of dict): List of signal dictionaries as defined in the
                                get_input_time_series() function.
    """
    idx0 = freqs.size // 2
    axis.plot(freqs[idx0:], np.abs(amps[idx0:]), 'k-',
              label='abs(dirty spectrum)')
    axis.set_ylabel('Spectral amplitude')
    axis.set_xlabel('Frequency [days$^{-1}$]')
    if signals is not None:
        for signal_ in signals:
            axis.plot([signal_['freq'], signal_['freq']],
                      [0, signal_['amp'] / 2], '--', color='g', label='signal')
            axis.plot([signal_['freq']], [signal_['amp'] / 2], '+', color='g',
                      ms=10)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[0:2], labels[0:2], loc='best')
    axis.grid(True)


def plot_restored_clean_spectrum(axis, freqs, amps, signals=None):
    """Plot the restored clean spectrum.

    Args:
        axis: matplotlib axis object
        freqs (numpy.ndarray): Array of spectrum frequencies
        amps (numpy.ndarray): Array of spectrum amplitudes
        signals (list of dict): List of signal dictionaries as defined in the
                                get_input_time_series() function.
    """
    idx0 = freqs.size // 2
    axis.plot(freqs[idx0:], np.abs(amps[idx0:]), 'k-',
              label='abs(clean + last residual spectrum)')
    if signals is not None:
        for signal_ in signals:
            axis.plot([signal_['freq'], signal_['freq']], [0, signal_['amp']],
                      '--', color='g', label='signal')
            axis.plot([signal_['freq']], [signal_['amp']], '+', color='g',
                      ms=10)
    axis.set_ylabel('Spectral amplitude')
    axis.set_xlabel('Frequency [days$^{-1}$]')
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[0:2], labels[0:2], loc='best')
    axis.grid(True)


def main():
    """Main function."""

    # Create logger.
    log = logging.getLogger()

    # Obtain the time series to analyse.
    file_path = None
    # file_path = os.path.join('data', 'all_sort.dat')
    if file_path is not None:
        signals = None
        times, values = load_times(os.path.join('data', 'all_sort.dat'))
    else:
        times, values, signals = get_input_time_series()

    # Time differences.
    delta_time = np.diff(times)
    length_days = times[-1] - times[0]

    max_frequency_median = 1 / np.median(delta_time)
    max_frequency_mean = 1 / np.mean(delta_time)
    max_frequency = min(max_frequency_median, max_frequency_mean)
    over_sample = 3
    freq_inc = 1 / (over_sample * length_days)

    log.info('')
    log.info('* Time series stats:')
    log.info('  - Mean sample separation   = %.4f days', np.mean(delta_time))
    log.info('  - Median sample separation = %.4f days', np.median(delta_time))
    log.info('  - STD sample separation    = %.4f days', np.std(delta_time))

    log.info('* Inferred spectrum parameters:')
    log.info('  - Max frequency (median) = %.4f days^-1', max_frequency_median)
    log.info('  - Max frequency (mean)   = %.4f days^-1', max_frequency_mean)
    log.info('  - Frequency increment    = %.4f days^-1', freq_inc)

    max_frequency = float(input('> Enter maximum frequency '
                                '(default = {:.4f} days^-1): '
                                .format(max_frequency)) or max_frequency)
    freq_inc = float(input('> Enter frequency increment'
                           '(default = {:.4f} days^-1): '
                           .format(freq_inc)) or freq_inc)

    min_frequency = 0
    freq_range = max_frequency - min_frequency
    num_freqs = math.ceil(freq_range / freq_inc)

    log.info(' - No. (positive) frequencies = %i', num_freqs)
    response = input('> Continue (y/n)? (default = y)') or 'y'
    if response != 'y':
        sys.exit(1)

    # DFT (Dirty spectrum)
    log.info('* Generating dirty spectrum ...')
    start_time = time.time()
    # -fs -> fs
    freqs = np.arange(-num_freqs, num_freqs + 1) * freq_inc
    amps = np.zeros_like(freqs, dtype='c16')
    for i, freq in enumerate(freqs):
        phase = np.exp(-1j * 2 * math.pi * freq * times)
        amps[i] += np.sum(values * phase)
    amps /= times.size
    log.info('* Done (%.4f s)', (time.time() - start_time))

    # DFT (PSF)
    log.info('* Generating PSF ...')
    freqs_psf = np.arange(-num_freqs * 2, num_freqs * 2 + 1) * freq_inc
    psf = np.zeros_like(freqs_psf, dtype='c16')
    for i, freq in enumerate(freqs_psf):
        phase = np.exp(-1j * 2 * math.pi * freq * times)
        psf[i] += np.sum(phase)
    psf /= times.size
    log.info('* Done (%.4f s)', (time.time() - start_time))

    # Simple CLEAN
    gain = float(input('> Enter CLEAN gain (default = 0.1): ') or 0.1)
    num_iter = int(input('> Enter no. CLEAN iterations '
                         '(default = 500): ') or 500)
    # plot_ = input('> Interactive plotting (y/n)? (default = n):') or 'n'
    # plot = True if plot_ == 'y' else False
    log.info('* Starting CLEAN ... (niter = %i, gain = %f)', num_iter, gain)
    start_time = time.time()
    clean_components = np.zeros_like(amps)
    residual = np.copy(amps)
    idx0 = residual.size // 2  # Centre index
    for _ in range(num_iter):
        # Find maximum peak in the spectrum for freq > 0
        idx_max = idx0 + np.argmax(residual[idx0 + 1:]) + 1

        # Amp of the residual at the peak
        res_max = residual[idx_max]

        # Value of PSF contribution from the -ve frequency peak.
        psf_at_max = psf[idx_max * 2]

        # Amplitude of the peak, correcting for the PSF contribution of the -ve
        # frequency peak
        amp_max = (res_max * (1 - psf_at_max)) / (1 - np.abs(psf_at_max)**2)

        # Append to clean component
        clean_components[idx_max] += amp_max * gain

        # Update residual spectrum by subtracting the PSF for the +'ve and -'ve
        # frequency peaks.
        psf_plus = psf[range(freqs.size) - idx_max + freqs.size - 1]
        psf_minus = psf[range(freqs.size) + idx_max]
        residual -= ((amp_max * psf_plus + np.conj(amp_max) * psf_minus) * gain)

    log.info('* Done (%.4f s)', (time.time() - start_time))

    # Make the clean spectrum ...
    idx0 = psf.size // 2  # Centre index
    # Find the approx half-width @ half max of the PSF and calculate the
    # clean beam STD
    hwhm = np.argmax(np.abs(psf[idx0:]) <= 0.5)
    sigma = (2 * hwhm) / 2.355

    extent = hwhm * 5
    x_data = np.arange(-extent, extent + 1)
    clean_beam = np.exp(-x_data**2 / (2 * sigma**2))

    # Generate the clean spectrum
    clean_spectrum = np.convolve(clean_components, clean_beam, mode='same')

    # Scale by factor of 2 due to power in real and imag parts.
    clean_spectrum *= 2.0

    # Add back last residual
    clean_spectrum += residual

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.97,
                        hspace=0.3, wspace=0.0)
    plot_time_series(ax1, times, values)
    plot_dirty_spectrum(ax2, freqs, amps, signals)
    plot_restored_clean_spectrum(ax3, freqs, clean_spectrum, signals)
    plt.show()


if __name__ == '__main__':
    LOG = logging.getLogger()
    LOG.addHandler(logging.StreamHandler(sys.stdout))
    LOG.setLevel('DEBUG')
    main()
