# coding: utf-8
"""Very quick hack of a time series cleaning script for angle data

usage:
    $ python3 main_beta.py

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
    """Fitting function"""
    A, mu, sigma = p
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def dft(num_freqs, freq_inc, times, values):
    """Generate spectra."""
    log = logging.getLogger(__name__)

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

    return freqs, amps, freqs_psf, psf


def clean(amps, psf, gain=0.05, num_iter=2000, freqs=None, f0=None, f1=None):
    """."""
    log = logging.getLogger(__name__)
    log.info('* Starting CLEAN ... (niter = %i, gain = %f)', num_iter, gain)
    t0 = time.time()
    clean_components = np.zeros_like(amps)
    residual = np.copy(amps)
    num_freqs = amps.size
    c = residual.size // 2
    idx0 = c + 1
    idx1 = residual.size
    if freqs is not None:
        if f0 is not None:
            idx0 = np.argmax(freqs >= f0)
        if f1 is not None:
            idx1 = np.argmax(freqs >= f1)
    print('* CLEAN search range: %i %i (c + 1 = %i, size = %i)' %
          (idx0, idx1, c + 1, residual.size))
    for _ in range(num_iter):
        # Find maximum peak in the spectrum for freq > 0
        idx_max = idx0 + np.argmax(residual[idx0:idx1 + 1])

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
        psf_plus = psf[range(num_freqs) - idx_max + num_freqs - 1]
        psf_minus = psf[range(num_freqs) + idx_max]
        residual -= (amp_max * psf_plus + np.conj(amp_max) * psf_minus) * gain

    log.info('* Done (%.4f s)', (time.time() - t0))

    return clean_components, residual


def make_clean_spectrum(psf, clean_components, residual):
    """."""
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
    restored_spectrum = clean_spectrum + (residual * 2.0)

    return clean_spectrum, restored_spectrum


def plot_times(ax, times, values, y_label, title, xlabel=None):
    """."""
    # Time series
    ax.plot(times, values, 'k.', ms=3)
    ax.set_ylabel(y_label)
    ax.grid(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    # ax.set_ylim(-0.1, 0.1)
    ax.set_ylim(np.array(ax.get_ylim()) * 1.6)
    ax.text(0.01, 0.95, title,
            horizontalalignment='left',
            verticalalignment='top',
            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5),
            transform=ax.transAxes)


def plot_title_box(ax, title):
    """."""
    ax.text(0.01, 0.95, title,
            horizontalalignment='left',
            verticalalignment='top',
            bbox=dict(boxstyle='square', facecolor='white', alpha=0.8),
            transform=ax.transAxes)


def plot_signals(ax, signals, signal_amps=False):
    """."""
    for k, s in enumerate(signals):
        if signal_amps:
            ax.plot([s['freq'], s['freq']], [0, s['amp']], '.-',
                    label='%.2f days' % (1 / s['freq']))
        else:
            ax.plot([s['freq'], s['freq']], ax.get_ylim(), '--',
                    label='%.2f days' % (1 / s['freq']))


def plot_dirty(ax, freqs, amps, signals, signal_amps=False, title=None,
               xlabel=None):
    """ Plot dirty spectrum """
    c = amps.size // 2
    ax.plot(freqs[c:], np.abs(amps[c:]), 'k-', label='abs(dirty spectrum)')
    # ax.plot(freqs, np.abs(amps), 'k-', label='abs(dirty spectrum)')
    if signals is not None:
        plot_signals(ax, signals, signal_amps)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='best')
    ax.grid(True)
    ax.set_ylabel('Spectral amplitude')
    if xlabel:
        ax.set_xlabel(xlabel)
    plot_title_box(ax, title)


def plot_psf(ax, freqs, values, title=None, xlabel=None):
    """ Plot dirty spectrum """
    c = values.size // 2
    x1 = c - c // 4
    x2 = c + c // 4
    ax.plot(freqs[x1:x2], np.abs(values[x1:x2]), 'k-')
    # ax.plot(freqs, np.abs(values), 'k-')
    ax.grid(True)
    ax.set_ylabel('Amplitude')
    if xlabel:
        ax.set_xlabel(xlabel)
    plot_title_box(ax, title)


def plot_clean_spectrum(ax, freqs, restored_spectrum, clean_spectrum, signals,
                        signal_amps=False, title=None, xlabel=None):
    """."""
    c = freqs.size // 2
    ax.plot(freqs[c:], np.abs(restored_spectrum[c:]), '-', color='0.7',
            label='abs(residual spectrum)')
    ax.plot(freqs[c:], np.abs(clean_spectrum[c:]), 'k-',
            label='abs(clean spectrum)')
    if signals is not None:
        plot_signals(ax, signals, signal_amps)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:], labels[:], loc='best')
    ax.grid(True)
    ax.set_ylabel('Spectral amplitude')
    plot_title_box(ax, title)
    if xlabel:
        ax.set_xlabel(xlabel)


def main():
    """Main function."""

    # Create logger.
    log = logging.getLogger()

    # Load data
    simulate = False
    data = np.loadtxt(os.path.join('data', 'angleAndBeta_Bluejet_Redjet.txt'))
    data = data[data[:, 0].argsort()]
    date = data[:, 0]
    times = date - date[0]
    beta = data[:, 1]
    raw_values = beta
    values = beta - np.mean(beta)

    # Filter the data in various ways.
    limit_times = False
    remove_times = False
    remove_large_values = False
    if remove_large_values:
        std_ = np.std(values)
        print('STD:', std_)
        idx = np.logical_and(values < std_ * 2, values > -std_ * 2)
        times = times[idx]
        raw_values = raw_values[idx]
        values = values[idx]
        date = date[idx]
    if limit_times:
        # time_limit = [600, 2000]
        # time_limit = [600, 2000]
        time_limit = [900, 1600]
        idx_min = np.argmax(times >= time_limit[0])
        idx_max = np.argmax(times >= time_limit[1])
        if idx_max == 0:
            idx_max = times.size
        times = times[idx_min:idx_max]
        values = values[idx_min:idx_max]
        raw_values = raw_values[idx_min:idx_max]
        date = date[idx_min:idx_max]
    if remove_times:
        delete_range = [900, 1300]
        idx_min = np.argmax(times >= delete_range[0])
        idx_max = np.argmax(times >= delete_range[1])
        if idx_max == 0:
            idx_max = times.size()
        times = np.delete(times, range(idx_min, idx_max + 1))
        values = np.delete(values, range(idx_min, idx_max + 1))
        raw_values = np.delete(raw_values, range(idx_min, idx_max + 1))
        date = np.delete(date, range(idx_min, idx_max + 1))

    values -= np.mean(values)
    times -= times[0]

    # Simulate signal using time sampling from data set
    if simulate:
        sim_values = np.zeros_like(times)
        signals = [
            dict(amp=0.03, freq=1 / 13.0865, phase=0.0),
            dict(amp=0.04, freq=1 / 11.2384, phase=0.0),
            # dict(amp=0.1, freq=0.0769, phase=0.0),
        ]
        log.debug('- Adding signals:')
        for i, s in enumerate(signals):
            log.debug('  [%02i] amp=%f, freq=%f, phase=%f',
                      i, s['amp'], s['freq'], s['phase'])
            arg = (2 * math.pi * times * s['freq']) + (s['phase'] * math.pi)
            sim_values += s['amp'] * np.cos(arg)
    else:
        signals = None

    log.info('')
    min_frequency = 0
    # IMPORTANT!!!!!!!!
    # CLEAN is *very* sensitive to the position of the peak with a horrible
    # PSF therefore with a poor PSF the spectrum oversample has to be very
    # large to not miss the peak position at all.
    #
    max_frequency = 0.2  # days^-1
    freq_inc = 5e-6  # days^-1
    freq_range = max_frequency - min_frequency
    num_freqs = math.ceil(freq_range / freq_inc)
    log.info('* No. (positive) frequencies = %i', num_freqs)

    if simulate:
        freqs, sim_amps, freqs_psf, sim_psf = dft(num_freqs, freq_inc, times,
                                                  sim_values)
        sim_clean_components, sim_residual = clean(sim_amps, sim_psf,
                                                   gain=0.1, num_iter=1000,
                                                   freqs=freqs, f0=1 / 180)
        sim_clean_spectrum, sim_restored_spectrum = \
            make_clean_spectrum(sim_psf, sim_clean_components, sim_residual)
    else:
        freqs, amps, freqs_psf, psf = dft(num_freqs, freq_inc, times, values)
        clean_components, residual = clean(amps, psf, gain=0.1, num_iter=1000,
                                           freqs=freqs, f0=1 / 180)
        clean_spectrum, restored_spectrum = \
            make_clean_spectrum(psf, clean_components, residual)

    # Plotting
    amp_type = 'Beta'
    c = freqs.size // 2
    fig, ax = plt.subplots(nrows=4, sharex=False, figsize=(10, 8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.95,
                        hspace=0.4, wspace=0.0)
    if simulate:
        plot_times(ax[0], times, sim_values, y_label=amp_type,
                   title='Simulated signal', xlabel=r'\Delta Date [days]')
        plot_psf(ax[1], freqs_psf, sim_psf, title='PSF')
        plot_dirty(ax[2], freqs, sim_amps, signals,
                   title='Dirty spectrum (simulated)')
        plot_clean_spectrum(ax[3], freqs, sim_restored_spectrum,
                            sim_clean_spectrum, signals, signal_amps=True,
                            title='CLEAN spectrum (simulated)')
    else:
        plot_times(ax[0], times, values, y_label=amp_type,
                   title='Time series', xlabel='Date [days]')
        plot_psf(ax[1], freqs_psf, psf, title='PSF')
        plot_dirty(ax[2], freqs, amps, signals, title='Dirty spectrum.')
        plot_clean_spectrum(ax[3], freqs, residual, clean_spectrum, signals,
                            signal_amps=False, title='CLEAN spectrum')

        # initial guess at fit parameters
        p0 = [0.0055, 0.076, 0.0006]
        p1 = [0.0085, 0.088, 0.0006]
        # p1 = [0.015, 0.0885, 0.0003]

        c = freqs.size // 2
        full_search_range = True
        search_range = 10
        try:
            if full_search_range:
                coeff0, _ = curve_fit(gauss, freqs[c:],
                                      np.abs(clean_spectrum[c:]),
                                      p0=p0)
                coeff1, _ = curve_fit(gauss, freqs[c:],
                                      np.abs(restored_spectrum[c:]),
                                      p0=p0)
            else:
                idx0 = np.argmax(freqs >= p0[1]-search_range*p0[2])
                idx1 = np.argmax(freqs >= p0[1]+search_range*p0[2])
                coeff0, _ = curve_fit(gauss, freqs[idx0:idx1+1],
                                      np.abs(clean_spectrum[idx0:idx1 + 1]),
                                      p0=p0)
                coeff1, _ = curve_fit(gauss, freqs[idx0:idx1+1],
                                      np.abs(restored_spectrum[idx0:idx1 + 1]),
                                      p0=p0)

            if abs(1 / coeff1[1] - 1 / p0[1]) < 2 and coeff1[2] / p0[2] < 4:
                if coeff1[0] > np.std(np.abs(restored_spectrum)) * 2:
                    x_fit = np.linspace(coeff1[1] - search_range * coeff1[2],
                                        coeff1[1] + search_range * coeff1[2],
                                        1000)
                    delta_p = (1 / (coeff1[1] - coeff1[2]) -
                               1 / (coeff1[1] + coeff1[2]))
                    peak_fit = gauss(x_fit, *coeff1)
                    ax[3].plot(x_fit, peak_fit, 'r-',
                               label=r'Fit: %.4f $\pm$ %.1f days' %
                               (1 / coeff1[1], delta_p / 2))
            elif abs(1 / coeff0[1] - 1 / p0[1]) < 2 and coeff0[2] / p0[2] < 10:
                if coeff0[0] > np.std(np.abs(restored_spectrum)) * 2:
                    x_fit = np.linspace(coeff0[1] - 5 * coeff0[2],
                                        coeff0[1] + 5 * coeff0[2], 1000)
                    peak_fit = gauss(x_fit, *coeff0)
                    delta_p = (1 / (coeff0[1] - coeff0[2]) -
                               1 / (coeff0[1] + coeff0[2]))
                    ax[3].plot(x_fit, peak_fit, 'r-',
                               label=r'Fit: %.4f $\pm$ %.1f days' %
                               (1 / coeff0[1], delta_p / 2))
        except RuntimeError:
            pass

        try:
            if full_search_range:
                coeff2, _ = curve_fit(gauss, freqs[c:],
                                      np.abs(clean_spectrum[c:]),
                                      p0=p1)
                coeff3, _ = curve_fit(gauss, freqs[c:],
                                      np.abs(restored_spectrum[c:]),
                                      p0=p1)
            else:
                idx0 = np.argmax(freqs >= p1[1]-search_range*p1[2])
                idx1 = np.argmax(freqs >= p1[1]+search_range*p1[2])
                coeff2, _ = curve_fit(gauss, freqs[idx0:idx1+1],
                                      np.abs(clean_spectrum[idx0:idx1+1]),
                                      p0=p1)
                coeff3, _ = curve_fit(gauss, freqs[idx0:idx1+1],
                                      np.abs(restored_spectrum[idx0:idx1+1]),
                                      p0=p1)

            if abs(1 / coeff3[1] - 1 / p1[1]) < 2 and coeff3[2] / p1[2] < 10:
                x_fit = np.linspace(coeff3[1] - search_range * coeff3[2],
                                    coeff3[1] + search_range * coeff3[2], 1000)
                peak_fit = gauss(x_fit, *coeff3)
                delta_p = (1 / (coeff3[1] - coeff3[2])
                           - 1 / (coeff3[1] + coeff3[2]))
                ax[3].plot(x_fit, peak_fit, 'g-',
                           label=r'Fit: %.4f $\pm$ %.1f days' %
                           (1 / coeff3[1], delta_p / 2))
            elif abs(1 / coeff2[1] - 1 / p1[1]) < 2 and coeff2[2] / p1[2] < 5:
                x_fit = np.linspace(coeff2[1] - 5 * coeff2[2],
                                    coeff2[1] + 5 * coeff2[2], 1000)
                peak_fit = gauss(x_fit, *coeff2)
                delta_p = (1 / (coeff2[1] - coeff2[2])
                           - 1 / (coeff2[1] + coeff2[2]))
                ax[3].plot(x_fit, peak_fit, 'g-',
                           label=r'Fit: %.4f $\pm$ %.1f days' %
                           (1 / coeff2[1], delta_p / 2))
        except RuntimeError:
            pass

    ax[3].legend(loc='best', fontsize='small')
    amp_limit_str = '2std' if remove_large_values else 'all'
    if simulate:
        amp_type = 'simulated_%s' % amp_type
    if limit_times:
        plt.savefig('%s_times_%03.0f-%03.0f_amps_%s.png' %
                    (amp_type, time_limit[0],
                     time_limit[1], amp_limit_str))
    else:
        plt.savefig('%s_times_all_amps_%s.png' % (amp_type, amp_limit_str))

    plt.show()
    plt.close()


if __name__ == '__main__':
    LOG = logging.getLogger()
    LOG.addHandler(logging.StreamHandler(sys.stdout))
    LOG.setLevel('DEBUG')
    main()
