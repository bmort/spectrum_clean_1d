# coding: utf-8
"""Very quick hack of a time series cleaning script.

To use with a time series in a file, uncomment the load_times() function on
line ~64 and remove plotting references to the 'signals' list.

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


def main():
    """Main function."""

    # Create logger.
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel('DEBUG')

    # Generate test time series.
    length_days = 2000
    min_sample_interval = 0.2
    sample_fraction = 0.02
    noise_std = 0.1
    num_samples = (length_days / min_sample_interval) * sample_fraction
    num_samples = int(num_samples)

    # Amplitude, frequency (days^-1), phase (fraction of 2pi)
    signals = [
        dict(amp=1.0, freq=1/20, phase=0.25),
        dict(amp=0.5, freq=1/75, phase=0.0),
        dict(amp=0.5, freq=0.04, phase=0.0),
        dict(amp=0.15, freq=0.01, phase=0.0),
        dict(amp=1.0, freq=1/50, phase=-0.2),
    ]
    log.info('* Generating time series...')
    times, values = sim_time_series(length_days, min_sample_interval,
                                    num_samples, signals, noise_std)
    # times, values = load_times(os.path.join('data', 'all_sort.dat'))
    delta_time = np.diff(times)
    log.info('')

    max_frequency_median = 1 / np.median(delta_time)
    max_frequency_mean = 1 / np.mean(delta_time)
    max_frequency = min(max_frequency_median, max_frequency_mean)
    over_sample = 3
    freq_inc = 1 / (over_sample * length_days)

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

    log.info(' - No. (positive) frequencies = {}'.format(num_freqs))
    ok = input('> Continue (y/n)? (default = y)') or 'y'
    if ok != 'y':
        sys.exit(1)

    # DFT (Dirty spectrum)
    log.info('* Generating dirty spectrum ...')
    t0 = time.time()
    # -fs -> fs
    freqs = np.arange(-num_freqs, num_freqs + 1) * freq_inc
    amps = np.zeros_like(freqs, dtype='c16')
    for i, f in enumerate(freqs):
        phase = np.exp(-1j * 2 * math.pi * f * times)
        amps[i] += np.sum(values * phase)
    amps /= times.size
    log.info('* Done (%.4f s)', (time.time() - t0))

    # DFT (PSF)
    log.info('* Generating PSF ...')
    freqs_psf = np.arange(-num_freqs * 2, num_freqs * 2 + 1) * freq_inc
    psf = np.zeros_like(freqs_psf, dtype='c16')
    for i, f in enumerate(freqs_psf):
        phase = np.exp(-1j * 2 * math.pi * f * times)
        psf[i] += np.sum(phase)
    psf /= times.size
    log.info('* Done (%.4f s)', (time.time() - t0))

    # Simple CLEAN
    gain = float(input('> Enter CLEAN gain (default = 0.1): ') or 0.1)
    num_iter = int(input('> Enter no. CLEAN iterations '
                         '(default = 500): ') or 500)
    # plot_ = input('> Interactive plotting (y/n)? (default = n):') or 'n'
    # plot = True if plot_ == 'y' else False
    log.info('* Starting CLEAN ... (niter = %i, gain = %f)', num_iter, gain)
    t0 = time.time()
    clean_components = np.zeros_like(amps)
    residual = np.copy(amps)
    for i in range(num_iter):
        # 1. Find maximum residual
        c = amps.shape[0] // 2
        idx_max = c + np.argmax(residual[c:])

        # Amp of the residual at the peak
        res_max = residual[idx_max]

        # Magnitude (abs) of the PSF contribution from the negative frequency
        # peak.
        psf_at_max = psf[idx_max * 2]

        temp = 1 - np.abs(psf_at_max)**2
        if temp == 0:
            log.warning('Exiting CLEAN at iter %i!', i)
            break

        # Amplitude of the peak, correcting for the PSF contribution of the -ve
        # frequency peak
        amp_max = (res_max * (1 - psf_at_max)) / temp

        # Append to clean component
        clean_components[idx_max] += amp_max * gain

        # Update residual spectrum by subtracting the PSF
        sub = np.zeros_like(residual)
        for j in range(freqs.size):
            psf_plus = psf[j - idx_max + (freqs.size - 1)]
            psf_minus = psf[j + idx_max]
            sub[j] = (amp_max * psf_plus + np.conj(amp_max) * psf_minus) * gain

        residual -= sub

    log.info('* Done (%.4f s)', (time.time() - t0))

    # Make the clean spectrum ...
    c = psf.size // 2
    # Find the approx half-width @ half max of the PSF and calculate the
    # clean beam STD
    hwhm = np.argmax(np.abs(psf[c:]) <= 0.5)
    sigma = (2 * hwhm) / 2.355

    extent = hwhm * 5
    x = np.arange(-extent, extent + 1)
    clean_beam = np.exp(-x**2 / (2 * sigma**2))
    # fig, ax = plt.subplots()
    # ax.plot(x, np.abs(psf[c - extent:c + extent + 1]), 'b-')
    # ax.plot(x, y, 'r--')
    # ax.grid(True)
    # plt.show()

    # Generate the clean spectrum
    clean_spectrum = np.convolve(clean_components, clean_beam, mode='same')

    # Scale by factor of 2 due to power in real and imag parts.
    clean_spectrum *= 2.0

    # Add back last residual
    # clean_spectrum += residual # FIXME(BM)

    # Plotting
    c = freqs.size // 2
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.97,
                        hspace=0.3, wspace=0.0)

    # Time series
    ax1.plot(times, values, 'k.', ms=3)
    ax1.set_xlabel('Time [days]')
    ax3.set_ylabel('Time series amplitude')
    ax1.grid(True)

    # Dirty spectrum
    ax2.plot(freqs[c:], np.abs(amps[c:]), 'k-', label='abs(dirty spectrum)')
    ax2.set_ylabel('Spectral amplitude')
    ax2.set_xlabel('Frequency [days$^{-1}$]')
    for k, s in enumerate(signals):
        ax2.plot([s['freq'], s['freq']], [0, s['amp'] / 2], '--',
                 color='g', label='signal')
        ax2.plot([s['freq']], [s['amp'] / 2], '+', color='g', ms=10)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[0:2], labels[0:2], loc='best')
    ax2.grid(True)

    ax3.plot(freqs[c:], np.abs(clean_spectrum[c:]), 'k-',
             label='abs(clean + last residual spectrum)')
    for k, s in enumerate(signals):
        ax3.plot([s['freq'], s['freq']], [0, s['amp']], '--', color='g',
                 label='signal')
        ax3.plot([s['freq']], [s['amp']], '+', color='g', ms=10)
    ax3.set_ylabel('Spectral amplitude')
    ax3.set_xlabel('Frequency [days$^{-1}$]')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[0:2], labels[0:2], loc='best')
    ax3.grid(True)

    plt.show()


if __name__ == '__main__':
    main()