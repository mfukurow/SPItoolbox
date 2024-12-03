"""
tools for load and save

efish lab
Hokkaido University
Author: Matasaburo Fukutomi
Email: mfukurow@gmail.com
"""

import labchart_econ as le
import numpy as np
from scipy.io import loadmat


def load_labmat(filepath: str = None, ch_trig: int = 0) -> np.ndarray:
    """
    load labchart matlab file and extract time, EOD, and trigger traces

    Args:
        filepath (str, optional): full path to labchart matlab file. Defaults to None.
        ch_trig (int, optional): channel of trigger pulses. Defaults to 0.

    Returns:
        np.ndarray: time information (time), EOD recording trace (rec_EOD),
                    trigger pulse trace (rec_trig)
    """
    if filepath is None:
        filepath = le.select_matfile()

    data = loadmat(filepath)
    time = np.array(data["ticktimes_block1"][0,])
    rec = data["data_block1"]
    rec_trig = np.array(rec[ch_trig,])
    rec_EOD = np.vstack((rec[:ch_trig], rec[(ch_trig + 1) :]))

    return time, rec_EOD, rec_trig
