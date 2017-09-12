# coding: utf-8
"""Very quick hack of a time series folding script.

.. moduleauthor:: Benjamin Mort <benjamin.mort@oerc.ox.ac.uk>
"""

import logging
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Require python3
if sys.version_info[0] != 3 or sys.version_info[1] < 5:
    print('This script requires Python version >= 3.5')
    sys.exit(1)


def cos_fit(x, *p):
    """Fitting function."""
    A, B, phi = p
    return A + B * np.cos(2 * math.pi * x + phi * math.pi)


def main():
    """Main function."""

    # Create logger.
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel('DEBUG')

    data = np.loadtxt(os.path.join('data', 'angleAndBeta_Bluejet_Redjet.txt'))
    data = data[data[:, 0].argsort()]
    date = data[:, 0]
    times = date - date[0]
    angle = data[:, 2]


    values = angle
    reflect_limits = [[721, 776.7],
                      [1044, 1102],
                      [1200, 1250],
                      [1373, 1423]]
    corrected_values = np.copy(values)
    for i, t in enumerate(times):
        for r_lim in reflect_limits:
            if r_lim[0] < t < r_lim[1]:
                # print('correcting amp... t=%f' % t)
                temp = -(corrected_values[i] - np.min(values)) + np.min(values)
                corrected_values[i] = temp

    # Simulate signal using times defined in the loaded data
    simulate = False
    if simulate:
        signals = [
            # dict(amp=0.35, freq=1 / 163.6105, phase=-0.14),
            dict(amp=0.35, freq=1 / 163.4, phase=-0.14),
            # dict(amp=0.35, freq=1 / 164.485, phase=-0.12),
        ]
        sim_values = np.zeros_like(times)
        log.debug('- Adding signals:')
        for i, s in enumerate(signals):
            log.debug('  [%02i] amp=%f, freq=%f, phase=%f',
                      i, s['amp'], s['freq'], s['phase'])
            arg = (2 * math.pi * times * s['freq']) + (s['phase'] * math.pi)
            sim_values += s['amp'] * np.cos(arg)
        corrected_values = sim_values

    # Period in days and number of bins used to fold the data.
    fold_period = 163.4
    # fold_period = 163
    # fold_period = 11.24
    # fold_period = 5.8
    num_bins = 30

    phase = date / fold_period  # Convert to phase
    phase = phase % 1  # Only keep   fractional phase values.
    # Sort by phase value
    sort_idx = phase.argsort()
    phase = phase[sort_idx]
    phase_value = corrected_values[sort_idx]

    bin_sum = np.zeros(num_bins)
    bin_count = np.zeros(num_bins)
    bin_width = 1 / num_bins
    for i in range(phase_value.size):
        n = int(phase[i] / bin_width)
        bin_sum[n] += phase_value[i]
        bin_count[n] += 1
    print('bin_width:', bin_width)
    bin_mean = bin_sum / bin_count

    bin_std = np.zeros(num_bins)
    for i in range(phase_value.size):
        n = int(phase[i] / bin_width)
        bin_std[n] += (phase_value[i] - bin_mean[n])**2
    bin_std = np.sqrt(bin_std / bin_count)

    bin_edges = np.arange(num_bins) * bin_width
    bin_centres = bin_edges + bin_width / 2

    # Fit to raw phase
    p0 = [0.2, 0.4, 0]
    coeff, _ = curve_fit(cos_fit, phase, phase_value, p0=p0)
    x_fit = np.linspace(0, 1, 200)
    y_fit = cos_fit(x_fit, *coeff)

    # Fit to binned phase
    p0 = [0.2, 0.4, 0]
    coeff_bin, _ = curve_fit(cos_fit, bin_centres, bin_mean, p0=p0)
    x_fit_bin = np.linspace(0, 1, 200)
    y_fit_bin = cos_fit(x_fit, *coeff)

    fig, ax = plt.subplots(nrows=3, figsize=(10, 8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.95,
                        hspace=0.4, wspace=0.0)
    ax[0].plot(date, corrected_values, '.', ms=3)
    ax[0].set_xlabel('Julian date')
    ax[0].set_ylabel('Signal amplitude')
    ax[0].set_title('Angles')
    ax[0].grid(True)

    ax[1].plot(phase, phase_value, '.', ms=3)
    ax[1].plot(x_fit, y_fit, 'r--',
               label=r'$%.3f %+.2f cos(2 \pi x %+.3f \pi)$' %
               (coeff[0], coeff[1], coeff[2]))
    ax[1].legend(loc='best', fontsize='x-small')
    ax[1].set_xlabel(r'Phase / $2\pi$')
    ax[1].set_ylabel('Signal amplitude')
    ax[1].set_title('Folded period = %.4f' % fold_period)
    ax[1].grid(True)

    ax[2].errorbar(bin_centres, bin_mean,
                   yerr=bin_std, linestyle='none',
                   marker='.')
    ax[2].plot(x_fit_bin, y_fit_bin, 'r--',
               label=r'$%.3f %+.2f cos(2 \pi x %+.3f \pi)$' %
               (coeff_bin[0], coeff_bin[1], coeff_bin[2]))
    ax[2].legend(loc='best', fontsize='x-small')
    ax[2].set_xlabel(r'Phase / $2\pi$')
    ax[2].set_ylabel('Signal amplitude')
    ax[2].set_title('Binned folded period = %.4f' % fold_period)
    ax[2].grid(True)

    plt.savefig('angle_folded_%03.5f_2.png' % fold_period)
    plt.show()


if __name__ == '__main__':
    main()
