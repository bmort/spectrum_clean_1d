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

    Example
            A signal definition dictionary contains the following fields:
            'amp': the signal amplitude,
            'freq': signal frequency in days^-1
            'phase': signal phase as fraction of 2pi.

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
        arg = 2 * math.pi * (times * s['freq'] + s['phase'])
        values += s['amp'] * np.cos(arg)

    # Add noise
    values += np.random.randn(values.size) * noise_std

    # # Scale to mean and std (if required)
    # target_mean = 2.0
    # target_std = 1.0
    # values *= target_std / np.std(values)
    # values += target_mean - np.mean(values)

    return times, values


def load_times(file_name, zero=True):
    """Load a time series.

    Assumes 2 columns of data, space separated.

    Args:
        file_name (str): File name to load.
        zero (bool, optional): If true, zero time series.
    """
    data = np.loadtxt(file_name)
    time = data[:, 0]
    amp = data[:, 1]
    if zero:
        time = time - time[0]
    return time, amp

