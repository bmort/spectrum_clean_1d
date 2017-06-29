# coding: utf-8
"""Module with functions for handling time series.

.. moduleauthor:: Benjamin Mort <benjamin.mort@oerc.ox.ac.uk>
"""

import logging
import math
import random

import numpy as np


def sim_time_series(length_days=2000, min_sample_interval=0.5,
                    num_samples=500, signals=None, noise_std=0.0):
    """Simulate a time series for testing

    Args:
        length_days (float): Length of the time series, in days.
        min_sample_interval (float): minimum sample interval, in days.
        num_samples (int): Number of (time) samples to generate.
        signals (list(dict)): List of signal definition dictionaries
            (see example below).
        noise_std (float): Noise amplitude STD.

    Signals in the time series are specified with a list of dictionaries, where
    each dictionary represents a different signal component.

    The signal definition dictionary contains the following fields:
    - 'amp': the signal amplitude,
    - 'freq': signal frequency, in days^-1
    - 'phase': signal phase, in units of pi.

    For example to specify two signal components the following list could be
    used:

        signals = [dict(amp=1.0, freq=0.05, phase=0.0),
                   dict(amp=0.5, freq=0.1, phase=0.0)]

    Returns:
        tuple (numpy.array, numpy.array): times, values
        Array of times and values.
    """
    log = logging.getLogger(__name__)

    if signals is None:
        signals = [dict(amp=1.0, freq=1/20, phase=0.0)]

    total_times = length_days / min_sample_interval
    times = np.arange(total_times) * min_sample_interval
    times = random.sample(times.tolist(), num_samples)
    times = np.sort(times)

    log.debug('- Total possible times = %i', total_times)
    log.debug('- No. times generated = %i', times.size)
    log.debug('- Min. sampling interval = %f', min_sample_interval)
    log.debug('- Sample fraction = %f', (num_samples / total_times))

    # Add signals
    values = np.zeros_like(times)
    log.debug('- Adding signals:')
    for i, s in enumerate(signals):
        log.debug('  [%02i] amp=%f, freq=%f, phase=%f',
                  i, s['amp'], s['freq'], s['phase'])
        arg = 2 * math.pi * times * s['freq'] + s['phase'] * math.pi
        values += s['amp'] * np.cos(arg)

    # Add noise
    values += np.random.randn(values.size) * noise_std

    # # Scale to mean and std (if required)
    # target_mean = 2.0
    # target_std = 1.0
    # values *= target_std / np.std(values)
    # values += target_mean - np.mean(values)

    return times, values


def load_times(file_name):
    """Load a time series.

    Assumes 2 columns space separated data.

    Args:
        file_name (str): File name (path) to load.

    Returns:
        tuple (numpy.array, numpy.array): times, values
        Array of times and values.
    """
    data = np.loadtxt(file_name)
    times = data[:, 0]
    values = data[:, 1]

    # Remove the mean amplitude and shift time origin
    times -= times[0]
    values -= np.mean(values)

    return times, values

