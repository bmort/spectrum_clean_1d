# coding: utf-8
"""Example of using the time series generator.

Run from repo root dir with:
    $ python3 -m spectclean.test.test_time_series
"""

import sys
import logging

import matplotlib.pyplot as plt

from spectclean.time_series import sim_time_series

if __name__ == '__main__':

    # Construct root logger.
    log = logging.getLogger()
    handler = log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel('DEBUG')

    # Input variables
    length_days = 2000
    sample_interval = 0.2
    sample_fraction = 0.01
    num_samples = (length_days / sample_interval) * sample_fraction
    num_samples = int(num_samples)
    signals = [dict(amp=1.0, freq=1/500, phase=0.0)]

    # Generate test time series
    times, values = sim_time_series(length_days=length_days,
                                    num_samples=num_samples,
                                    min_sample_interval=sample_interval,
                                    signals=signals)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(times, values, '.')
    ax.set_title('Time series. {} values'.format(times.size))
    plt.show()
