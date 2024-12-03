"""
spike density function using a gaussian convolution
see Dayan and Abbott 2001, pp.13 1.11
see Szucs 2000
(sdf.m from Carlson lab)

efish lab
Hokkaido University
Author: Matasaburo Fukutomi
Email: mfukurow@gmail.com
"""

import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt


def sdf(
    t_EOD: np.ndarray, width: float, max_tim: float = None, bin_siz=1
) -> np.ndarray:
    """
    spike density function

    Args:
        t_EOD (np.ndarray): EOD time
        width (float):
        max_tim (float, optional): Defaults to None.
        bin_siz (int, optional): Defaults to 1 (ms)

    Returns:
        np.ndarray: spike density function (F) and the corresponding time (t_F)
    """
    if max_tim is None:
        max_tim = max(t_EOD)

    # standard deviation
    sd = width / 4

    # make a gaussian kernel
    x = np.arange(-1000, 1001)
    y = np.exp(-0.5 * ((x / sd) ** 2))
    y /= np.sqrt(2 * np.pi) * sd
    y /= np.sum(y) / 1000

    # bin EOD time
    T = bin_spikes(t_EOD, bin_siz)
    z_pad = int((np.ceil(max_tim * 1000) / bin_siz) - len(T))
    T = np.concatenate((T, np.zeros(z_pad)))

    # gaussian convolution
    F = convolve(T, y, mode="same")
    t_F = np.arange(1, len(F) + 1) / 1000

    return F, t_F


def bin_spikes(seqinput, binsize):
    tmax = max(seqinput) * 1000
    timems = np.array(seqinput) * 1000
    binmax = int(np.ceil(tmax / binsize) * binsize)
    binspot = np.arange(
        0 + (0.5 * binsize), binmax - (0.5 * binsize) + binsize, binsize
    )
    binned, _ = np.histogram(timems, bins=binspot)
    return binned
