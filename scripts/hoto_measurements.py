import numpy as np
import pandas as pd
from matplotlib.widgets import Button
import tkinter as tk
import sys
import skrf as rf
from pathlib import Path
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from time import time_ns, sleep, localtime, strftime

k = keith


def iv_hoto (
    # values for pandas file
    samplename,
    padname,
    number_sweeps = 10,

    # values for sweeps
    V_set = 1.2,
    V_reset = -1.2,
    nplc=1,
    V_step=0.02,

):
    '''run a measurement during which the Keithley2600 applies a constants voltage and measures the current. 
    Pulses applied during this measurement are also recorded. '''
    number_of_events =0
    data = {}
    timestamp = strftime("%Y.%m.%d-%H.%M.%S", localtime())
    data['timestamp'] = timestamp
    data['padname'] = padname
    data['samplename'] = samplename
    data['num_sweeps'] = number_sweeps
    data['V_set'] = V_set
    data['V_reset'] = V_reset 
    data['V_step'] = V_step
    data['nplc'] = nplc
    

    # functions for sweeps
    def reset():
        k._iv_lua(
            tri(V_reset, 0.02), Irange=1e-2, Ilimit=1e-2,
            Plimit=V_reset*1e-3, nplc=nplc, Vrange=V_reset
        )
        while not k.done():
            sleep(0.01)
        return k.get_data()
    def set():
        k._iv_lua(
            tri(V_set, 0.02), Irange=1e-3, Ilimit=3e-4, 
            Plimit=V_set*1e-3, nplc=nplc, Vrange=V_set
        )
        while not k.done():
            sleep(0.01)
        return k.get_data()

    # create list for sets and resets
    data['sets'] = []
    data['resets'] = []
    
    # now in HRS we do {number_sweeps}
    for i in range(number_sweeps):
        data['sets'].append(set())
        # data[f'set_{i+1}_state'] = get_current_resistance()
        data['resets'].append(reset())
        # data[f'reset_{i+1}_state'] = get_current_resistance()

    # save results
    datafolder = os.path.join('C:\\Messdaten', padname, samplename)
    i=1
    # f"{timestamp}_pulsewidth={pulse_width:.2e}s_attenuation={attenuation}dB_points={points:.2e}_{i}"
    filepath = os.path.join(datafolder, f"{timestamp}_{i}.s")
    while os.path.isfile(filepath + '.s'):
        i +=1
        filepath = os.path.join(datafolder, f"{timestamp}_{i}.s")
    io.write_pandas_pickle(meta.attach(data), filepath)
    # print(len(data))
    return data    
