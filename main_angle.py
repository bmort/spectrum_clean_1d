# coding: utf-8
"""Very quick hack of a time series cleaning script for angle data

usage:
    $ python3 main_angle.py

.. moduleauthor:: Benjamin Mort <benjamin.mort@oerc.ox.ac.uk>
"""

import logging
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Require python3
if sys.version_info[0] != 3 or sys.version_info[1] < 5:
    print('This script requires Python version >= 3.5')
    sys.exit(1)


def gauss(x, *p):
    """Gaussian fitting function."""
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2 / (2*sigma**2))


def main():
    """Main function."""

    # Create logger.
    log = logging.getLogger()

    # Load data.
    amp_type = 'Angle'
    simulate = False
    data = np.loadtxt(os.path.join('data', 'angleAndBeta_Bluejet_Redjet.txt'))
    data = data[data[:, 0].argsort()]
    date = data[:, 0]
    times = date - date[0]
    length_days = date[-1] - date[0]
    angle = data[:, 2]
    values = angle - np.mean(angle)
    values_raw = angle

    # Correct for reflections in the amplitudes.
    reflect_limits = [[721, 776.7],
                      [1044, 1102],
                      [1200, 1250],
                      [1373, 1423]]
    corrected_values = np.copy(values)
    for i, t in enumerate(times):
        for r_lim in reflect_limits:
            if r_lim[0] < t < r_lim[1]:
                print('correcting amp... t=%f' % t)
                temp = -(corrected_values[i] - np.min(values)) + np.min(values)
                corrected_values[i] = temp
    values = corrected_values

    delta_time = np.diff(times)
    log.info('')

    # Simulate using time values in the data.
    if simulate:
        signals = [
            dict(amp=0.4, freq=1 / 163, phase=-0.18),
            dict(amp=0.03, freq=1 / 6, phase=0.0),
        ]
        sim_values = np.zeros_like(times)
        log.debug('- Adding signals:')
        for i, s in enumerate(signals):
            log.debug('  [%02i] amp=%f, freq=%f, phase=%f',
                      i, s['amp'], s['freq'], s['phase'])
            arg = 2 * math.pi * times * s['freq'] + s['phase'] * math.pi
            sim_values += s['amp'] * np.cos(arg)
        fig, ax = plt.subplots(nrows=1, figsize=(12, 6))
        ax.plot(date, values_raw, 'r+', ms=3, label='raw data')
        ax.plot(date, sim_values + np.mean(values_raw), 'bx', ms=2,
                label='simulated', lw=0.5)
        if amp_type == 'Angle':
            ax.plot(date, corrected_values + np.mean(values_raw), 'k.', ms=2,
                    label='corrected')
            for r in np.array(reflect_limits).flatten():
                ax.plot([r + date[0], r + date[0]], [-0.5, 0.5], '--',
                        color='green', lw=1.0,
                        label='correction zone')
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:5], labels[:5], loc='best')
        ax.set_ylabel('%s' % amp_type)
        ax.set_xlabel('Julian Date')
        plt.savefig('%s_time_series_compare_2.png' % amp_type)
        plt.show()
        plt.close()
        values = sim_values
    else:
        signals = None

    max_frequency_median = 1 / np.median(delta_time)
    max_frequency_mean = 1 / np.mean(delta_time)
    over_sample = 3
    freq_inc = 1 / (over_sample * length_days)

    log.info('* Time series stats:')
    log.info('  - Length                   = %.4f days', length_days)
    log.info('  - Mean sample separation   = %.4f days', np.mean(delta_time))
    log.info('  - Median sample separation = %.4f days', np.median(delta_time))
    log.info('  - STD sample separation    = %.4f days', np.std(delta_time))

    log.info('* Inferred spectrum parameters:')
    log.info('  - Max frequency (median) = %.4f days^-1', max_frequency_median)
    log.info('  - Max frequency (mean)   = %.4f days^-1', max_frequency_mean)
    log.info('  - Frequency increment    = %.4f days^-1', freq_inc)

    max_frequency = 0.2  # Maximum frequency, days^-1
    freq_inc = 1e-5  # Frequency increment, days^-1

    min_frequency = 0
    freq_range = max_frequency - min_frequency
    num_freqs = math.ceil(freq_range / freq_inc)

    log.info(' - No. (positive) frequencies = %i', num_freqs)

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
    gain = 0.05
    num_iter = 2000
    log.info('* Starting CLEAN ... (niter = %i, gain = %f)', num_iter, gain)
    t0 = time.time()
    clean_components = np.zeros_like(amps)
    residual = np.copy(amps)
    c = residual.size // 2
    for _ in range(num_iter):
        # Find maximum peak in the spectrum for freq > 0
        idx_max = c + np.argmax(residual[c+1:]) + 1

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
        residual -= (amp_max * psf_plus + np.conj(amp_max) * psf_minus) * gain

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

    # Generate the clean spectrum
    clean_spectrum = np.convolve(clean_components, clean_beam, mode='same')

    # Scale by factor of 2 due to power in real and imag parts.
    clean_spectrum *= 2.0

    # Add back last residual
    restored_spectrum = clean_spectrum + residual * 2.0

    # Plotting
    c = freqs.size // 2
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.95,
                        hspace=0.5, wspace=0.0)

    # Time series
    ax1.plot(times, values, 'k.', ms=3)
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('%s' % amp_type)
    if simulate:
        ax1.set_title('Simulated time series')
    else:
        ax1.set_title('Time series')
    ax1.grid(True)

    # Dirty spectrum
    ax2.plot(freqs[c:], np.abs(amps[c:]), 'k-', label='abs(dirty spectrum)')
    ax2.set_ylabel('Spectral amplitude')
    ax2.set_xlabel('Frequency [days$^{-1}$]')
    ax2.set_title('Dirty spectrum')
    if signals is not None:
        for s in signals:
            ax2.plot([s['freq'], s['freq']], ax2.get_ylim(), '--',
                     color='g', label='signal')
            if simulate:
                ax2.plot([s['freq']], [s['amp'] / 2], '+', color='g', ms=10)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[0:2], labels[0:2], loc='best')
    ax2.grid(True)

    ax3.plot(freqs[c:], np.abs(restored_spectrum[c:]), '-', color='0.5',
             label='abs(restored spectrum)')

    ax3.plot(freqs[c:], np.abs(clean_spectrum[c:]), 'k-',
             label='abs(clean spectrum)')
    if signals is not None:
        for s in signals:
            ax3.plot([s['freq'], s['freq']], ax3.get_ylim(), '--', color='g',
                     label='%.2f days' % (1/s['freq']))
            if simulate:
                ax3.plot([s['freq']], [s['amp']], '+', color='g', ms=10)
    ax3.set_title('Clean spectrum')
    ax3.set_ylabel('Spectral amplitude')
    ax3.set_xlabel('Frequency [days$^{-1}$]')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[:], labels[:], loc='best')
    ax3.grid(True)
    if simulate:
        plt.savefig('%s_simuluated.png' % amp_type)
    else:
        plt.savefig('%s_data_2.png' % amp_type)

    plt.show()
    plt.close()

    p0 = [0.2, 1/163, 0.0007]  # initial guess for the fit
    coeff, var_matrix = curve_fit(gauss, freqs[c:],
                                  np.abs(clean_spectrum[c:]), p0=p0)
    print('clean', coeff, var_matrix, 1/coeff[1])

    coeff, var_matrix = curve_fit(gauss, freqs[c:],
                                  np.abs(restored_spectrum[c:]), p0=p0)
    print('restored', coeff, var_matrix, 1/coeff[1])

    # Get the fitted curve
    x_fit = np.linspace(coeff[1] - 10 * coeff[2], coeff[1] + 10 * coeff[2],
                        1000)
    peak_fit = gauss(x_fit, *coeff)

    fig, ax2 = plt.subplots(nrows=1, figsize=(10, 5))
    ax2.plot(freqs[c:], np.abs(restored_spectrum[c:]), '-', color='k',
             label='abs(restored spectrum)')
    ax2.plot(x_fit, peak_fit, 'r:',
             label='Fitted data mu=%.4e, sigma=%.3e' %
             (coeff[1], coeff[2]))
    f1 = coeff[1]-coeff[2]
    f2 = coeff[1]+coeff[2]
    p0 = 1 / coeff[1]
    p1 = 1 / f1
    p2 = 1 / f2
    delta_p = p1 - p2
    print(p0, p1, p2, delta_p, delta_p/2)
    ax2.plot([coeff[1], coeff[1]], ax2.get_ylim(), '--', color='g',
             label=r'1/mu = %.4f $\pm$ %.1f days' % (1/coeff[1], delta_p/2))
    ax2.set_title('Clean spectrum')
    ax2.set_ylabel('Spectral amplitude')
    ax2.set_xlabel(r'Frequency [days$^{-1}$]')
    ax2.grid(True)
    ax2.set_xlim(0, f2 + 30 * coeff[2])
    ax2.legend(loc='best')
    plt.savefig('%s_data_fit.png' % amp_type)
    plt.show()


if __name__ == '__main__':
    LOG = logging.getLogger()
    LOG.addHandler(logging.StreamHandler(sys.stdout))
    LOG.setLevel('DEBUG')
    main()
