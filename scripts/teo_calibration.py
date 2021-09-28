import logging
import os
import time
from datetime import datetime

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ivtools.analyze as analyze
import ivtools.instruments as instruments
import ivtools.measure as measure
import ivtools.settings

log = logging.getLogger('instruments')

# todo: test LF

"""
TeoSystem calibration using Digipot, Picoscope and Keitheley.

Returns
-------

"""

'''
Things to calibrate:
    High Frequency:
        "HFV": Desired Voltage -> Applied Voltage
        "V_MONITOR": Read Voltage -> Applied Voltage
        "HFV_INT": Read Voltage -> Applied Voltage
        "HF_LIMITED_BW": Read Current -> Real Current
        "HFI_INT": Read Current -> Real Current
        "HF_FULL_BW": Read Current -> Real Current
    Low Frequency:
        ...
    
Amp1 uses all of the bits, for possible step settings 0-31
each step corresponds to 1dB (1.122×)
step = 0 is the highest gain setting
Used on HF_FULL_BW

Amp2 uses the two LSB, for possible step settings 0,1,2,3
each step corresponds to 6dB (2×)
Used on HF_LIMITED_BW and HFI_INT

Connections:
Teo 'HFV' -> Picoscope 'A' -> Resistor 'Input'  (Signal divider)
Resistor 'Output' -> Teo 'HFI'
Teo 'V_MONITOR' -> Picoscope 'B'
Teo 'HF_FULL_BW' -> Picoscope 'C'
Teo 'HF_LIMITED_BW' -> Picoscope 'D'
Teo 'TRIG1' -> Picoscope 'Aux Trigger'
Teo 'SW1' swithch in internal mode (Gray position)
'''

### Parameters ###
save_folder = "X:\emrl\Pool\Bulletin\Handbücher.Docs\TS_Memory_Tester\calibration"
set_up = 'Teo_Digipot_Picoscope'
V = 10        # Amplitude of the triangle pulse, positive and negative
SR = 100_000  # Sweep rate of the waveform. Lower sweep rates make more precise calibrations. 1000 takes about 20 mins
                # and 100_000 about a minute.
R = 47_000    # Resistor used to measure
check = True  # If True: the existing calibration will be used so you can check it, otherwise it will
                # perform an actual calibration
nplc = 10     # Number of power line cycles to use on low frequency mode


def pico_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    for digipot circuit with device voltage probe
    gain is in A/V, in case you put an amplifier on the output
    Simultaneous sampling is faster when not using adjacent channels (i.e. A&B)
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them
    HFV_channel = 'A'
    V_monitor_channel = 'B'
    HF_full_channel = 'C'
    HF_limited_channel = 'D'

    gain_step = teo.gain()

    chs_sampled = [ch for ch in ['A', 'B', 'C', 'D'] if ch in datain]
    if not chs_sampled:
        log.error('No picoscope data detected')
        return datain
    dataout = datain
    # If data is raw, convert it here
    if datain[chs_sampled[0]].dtype == np.int8:
        datain = measure.raw_to_V(datain, dtype=dtype)
    if 'units' not in dataout:
        dataout['units'] = {}
    if HFV_channel in datain:
        V = datain[HFV_channel]
        dataout['HFV'] = V  # Subtract voltage on output?  Don't know what it is necessarily.
        if datain['COUPLINGS'][HFV_channel] == 'DC50':  # Don't let this happen
            raise Exception("If you are using a signal divider in channel {HFV_channel} and you set 'DC50' only half "
                            "of the current will go to digipot")
        dataout['units']['HFV'] = 'V'
    if V_monitor_channel in datain:
        V = datain[V_monitor_channel]
        if datain['COUPLINGS'][V_monitor_channel] == 'DC50':
            V *= 2
        if teo.calibration is not None:
            V = np.polyval(teo.calibration.loc[gain_step, 'V_MONITOR'], V)
        dataout['V_MONITOR'] = V
        dataout['units']['V_monitor'] = 'V'
    if HF_limited_channel in datain:
        I = datain[HF_limited_channel]
        if teo.calibration is not None:
            I = np.polyval(teo.calibration.loc[gain_step, 'HF_LIMITED_BW'], I)
        dataout['HF_LIMITED_BW'] = I
        dataout['units']['HF_limited'] = 'A'
    if HF_full_channel in datain:
        I = datain[HF_full_channel]
        if teo.calibration is not None:
            I = np.polyval(teo.calibration.loc[gain_step, 'HF_FULL_BW'], I)
        dataout['HF_FULL_BW'] = I
        dataout['units']['HF_full'] = 'A'
    return dataout


teo = instruments.TeoSystem()
dp = instruments.WichmannDigipot()
ps = instruments.Picoscope()


teo.HF_mode()
dp.set_R(0)
wfm = teo.tri([0, V, -V, 0], sweep_rate=SR)
wfm_dur = (V * 4) / SR
wfm_samples = len(wfm)
wfm_name = f"V{V}_SR{SR}"
teo.upload_wfm(wfm, name=wfm_name)

ps_channels = ['A', 'B', 'C', 'D']
ps.offset = dict(A=0, B=0, C=0, D=0)
ps.atten = dict(A=1, B=1, C=1, D=1)

ps.coupling.a = 'DC'  # If DC50 half of the voltage is going to Pisoscope and half to Digipot, due to the signal divider
ps.coupling.b = 'DC50'
ps.coupling.c = 'DC50'
ps.coupling.d = 'DC50'

ps.range.a = V
ps.range.b = V/10
ps.range.c = 1
ps.range.d = 1

teo_gain_steps = np.arange(32)


def measure(gain_step):
    s = gain_step

    teo.gain(s)

    ps.capture(ch=ps_channels, freq=None, duration=wfm_dur, nsamples=wfm_samples,
               trigsource='TriggerAux', triglevel=0.5, timeout_ms=30000, direction='Rising',
               pretrig=0.0, delay=0)
    time.sleep(0.1)
    teo.output_wfm(wfm_name)
    psd = ps.get_data(ch=['A', 'B', 'C', 'D'], raw=False)
    psd = pico_to_iv(psd)
    teod = teo.get_data(raw=False)
    log.debug(f"Picoscope t: ({min(psd['t'])}, {max(psd['t'])}) ; len = {len(psd['t'])}")
    log.debug(f"Teo t      : ({min(teod['t'])}, {max(teod['t'])}) ; len = {len(teod['t'])}")
    smoothing_window_teo = int(wfm_samples / 4 / 50)
    decimate_factor_teo = len(teod['t']) // 1000
    smoothing_window_ps = int(len(psd['t']) / 4 / 50)
    decimate_factor_ps = len(psd['t']) // 1000
    log.debug(f"smoothing_window_teo = {smoothing_window_teo}\ndecimate_factor_teo = {decimate_factor_teo}")
    log.debug(f"smoothing_window_teo = {smoothing_window_ps}\ndecimate_factor_teo = {decimate_factor_ps}")
    psd = analyze.smoothimate(psd, smoothing_window_ps, decimate_factor_ps)
    teod = analyze.smoothimate(teod, smoothing_window_teo, decimate_factor_teo)
    log.debug(f"Picoscope t: ({min(psd['t'])}, {max(psd['t'])}) ; len = {len(psd['t'])}")
    log.debug(f"Teo t      : ({min(teod['t'])}, {max(teod['t'])}) ; len = {len(teod['t'])}")

    # Data can have small differences in length for slow sweep rates usually less that 5 samples, so I'm just removing
    # the last values from the longer data set. If for whatever reason you set faster sweep rate the differences can
    # be huge.
    len_diff = len(teod['t']) - len(psd['t'])
    log.debug(f"len_diff = {len_diff}")
    if len_diff > 0:
        for k in ['V', 'I', 't', 'Vwfm']:
            teod[k] = teod[k][:-len_diff]
    elif len_diff < 0:
        for k in ['A', 'B', 'C', 'D', 't', 'HFV', 'V_MONITOR', 'HF_LIMITED_BW', 'HF_FULL_BW']:
            psd[k] = psd[k][:len_diff]

    log.debug(f"Picoscope t: ({min(psd['t'])}, {max(psd['t'])}) ; len = {len(psd['t'])}")
    log.debug(f"Teo t      : ({min(teod['t'])}, {max(teod['t'])}) ; len = {len(teod['t'])}")

    d = pd.Series(dict(HFV_INT=teod['V'], HFV=psd['HFV'], HFI_INT=teod['I'],
                       V_MONITOR=psd['V_MONITOR'], HF_LIMITED_BW=psd['HF_LIMITED_BW'], HF_FULL_BW=psd['HF_FULL_BW'],
                       t=psd['t'], V_wfm=teod['Vwfm']))

    d.name = s

    log.debug(f"Dataframe t: ({min(d['t'])}, {max(d['t'])}) ; len = {len(d['t'])}")

    return d

# # # # # # # # # # # # # # # # # # # # Measurement to estimate ranges # # # # # # # # # # # # # # # # # # # # #
teo.calibration = None
d_test = measure(31)

max_HF_FULL_BW = max(abs(d_test['HF_FULL_BW']))
max_HF_LIMITED_BW = max(abs(d_test['HF_LIMITED_BW']))
max_V_MONITOR = max(abs(d_test['V_MONITOR']))

if check is True:
    if os.path.isfile(ivtools.settings.teo_calibration_file):
        teo.calibration = pd.read_pickle(ivtools.settings.teo_calibration_file)
        timestamp_chek = teo.calibration.attrs['timestamp']
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]

    else:
        log.warning('Calibration file not found!')
        teo.calibration = None
else:
    teo.calibration = None
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]

if max_V_MONITOR > 5:
    ps.coupling.b = 'DC'
else:
    ps.coupling.b = 'DC50'
ps.range.b = max_V_MONITOR

# # # # # # # # # # # # # # # # # # # # # #  Measuring # # # # # # # # # # # # # # # # # # # # # #

data = pd.DataFrame(dict())
data.index.name = 'Gain Step'

log.info('_' * 50)
log.info('_' * 50)

for s in teo_gain_steps[::-1]:
    log.info(f"Step {s}")

    full_step = s
    limited_step = s % 4

    # Setting Picoscope ranges #

    max_D = max_HF_LIMITED_BW * 2 ** (4 - limited_step)
    if max_D > 5:
        ps.coupling.d = 'DC'
    else:
        ps.coupling.d = 'DC50'
    ps.range.d = max_D

    max_C = max_HF_FULL_BW * (101 / 90) ** (31 - full_step)  # 101 / 90 = 1.12222222...
    if max_C > 5:
        ps.coupling.c = 'DC'
    else:
        ps.coupling.c = 'DC50'
    ps.range.c = max_C

    d = measure(s)

    data = data.append(d)

    log.info('_' * 50)


data.attrs['timestamp'] = timestamp
data.attrs['resistor'] = R
data.attrs['sweep_rate'] = SR
data.attrs['wfm_samples'] = wfm_samples
data.attrs['set_up'] = set_up

teo.LF_mode(external=False)
LF_I_range = 45e-6
LF_V_read = LF_I_range * R
LF_wfm = np.concatenate([np.linspace(0, LF_V_read, 5),
                         np.linspace(LF_V_read, -LF_V_read, 10)[1:-1],
                         np.linspace(-LF_V_read, 0, 5)])

LF_data = teo.measureLF(LF_wfm, NPLC=nplc)

# # # # # # # # # # # # # # # # # # # # # #  Fitting # # # # # # # # # # # # # # # # # # # # # #

fits = pd.DataFrame(dict())
fits.index.name = 'Gain Step'
for s, d in data.iterrows():

    nsamples = len(d['t'])
    V_wfm = d['V_wfm']
    I_expected = d['HFV'] / R
    HFV = tuple(np.polyfit(V_wfm, d['HFV'], 1))
    HFV_INT = tuple(np.polyfit(d['HFV'], d['HFV_INT'], 1))
    V_MONITOR = tuple(np.polyfit(d['HFV'], d['V_MONITOR'], 1))
    HFI_INT = tuple(np.polyfit(I_expected, d['HFI_INT'], 1))
    HF_LIMITED_BW = tuple(np.polyfit(I_expected, d['HF_LIMITED_BW'], 1))
    HF_FULL_BW = tuple(np.polyfit(I_expected, d['HF_FULL_BW'], 1))

    f = pd.Series(dict(HFV=HFV, HFV_INT=HFV_INT, HFI_INT=HFI_INT,
                       V_MONITOR=V_MONITOR, HF_LIMITED_BW=HF_LIMITED_BW, HF_FULL_BW=HF_FULL_BW))
    f.name = s
    fits = fits.append(f)


# HFV, HFV_INT, V_MONITOR don't have steps, so they are the same fit 31 times, here I average:
HFV_slope = np.mean([fits.loc[s]['HFV'][0] for s in teo_gain_steps])
HFV_offset = np.mean([fits.loc[s]['HFV'][1] for s in teo_gain_steps])
fits['HFV'] = [(HFV_slope, HFV_offset)] * 32

HFV_INT_slope = np.mean([fits.loc[s]['HFV_INT'][0] for s in teo_gain_steps])
HFV_INT_offset = np.mean([fits.loc[s]['HFV_INT'][1] for s in teo_gain_steps])
fits['HFV_INT'] = [(HFV_INT_slope, HFV_INT_offset)] * 32

V_MONITOR_slope = np.mean([fits.loc[s]['V_MONITOR'][0] for s in teo_gain_steps])
V_MONITOR_offset = np.mean([fits.loc[s]['V_MONITOR'][1] for s in teo_gain_steps])
fits['V_MONITOR'] = [(V_MONITOR_slope, V_MONITOR_offset)] * 32


# And HF_LIMITED_BW and HFI_INT only have 4 steps, so there are many fits repeated, here I average:
HF_LIMITED_BW_slope_0 = np.mean([fits.loc[s]['HF_LIMITED_BW'][0] for s in teo_gain_steps[0::4]])
HF_LIMITED_BW_offset_0 = np.mean([fits.loc[s]['HF_LIMITED_BW'][1] for s in teo_gain_steps[0::4]])
HF_LIMITED_BW_slope_1 = np.mean([fits.loc[s]['HF_LIMITED_BW'][0] for s in teo_gain_steps[1::4]])
HF_LIMITED_BW_offset_1 = np.mean([fits.loc[s]['HF_LIMITED_BW'][1] for s in teo_gain_steps[1::4]])
HF_LIMITED_BW_slope_2 = np.mean([fits.loc[s]['HF_LIMITED_BW'][0] for s in teo_gain_steps[2::4]])
HF_LIMITED_BW_offset_2 = np.mean([fits.loc[s]['HF_LIMITED_BW'][1] for s in teo_gain_steps[2::4]])
HF_LIMITED_BW_slope_3 = np.mean([fits.loc[s]['HF_LIMITED_BW'][0] for s in teo_gain_steps[3::4]])
HF_LIMITED_BW_offset_3 = np.mean([fits.loc[s]['HF_LIMITED_BW'][1] for s in teo_gain_steps[3::4]])
fits['HF_LIMITED_BW'] = [(HF_LIMITED_BW_slope_3, HF_LIMITED_BW_offset_3),
                         (HF_LIMITED_BW_slope_2, HF_LIMITED_BW_offset_2),
                         (HF_LIMITED_BW_slope_1, HF_LIMITED_BW_offset_1),
                         (HF_LIMITED_BW_slope_0, HF_LIMITED_BW_offset_0)] * 8

HFI_INT_slope_0 = np.mean([fits.loc[s]['HFI_INT'][0] for s in teo_gain_steps[0::4]])
HFI_INT_offset_0 = np.mean([fits.loc[s]['HFI_INT'][1] for s in teo_gain_steps[0::4]])
HFI_INT_slope_1 = np.mean([fits.loc[s]['HFI_INT'][0] for s in teo_gain_steps[1::4]])
HFI_INT_offset_1 = np.mean([fits.loc[s]['HFI_INT'][1] for s in teo_gain_steps[1::4]])
HFI_INT_slope_2 = np.mean([fits.loc[s]['HFI_INT'][0] for s in teo_gain_steps[2::4]])
HFI_INT_offset_2 = np.mean([fits.loc[s]['HFI_INT'][1] for s in teo_gain_steps[2::4]])
HFI_INT_slope_3 = np.mean([fits.loc[s]['HFI_INT'][0] for s in teo_gain_steps[3::4]])
HFI_INT_offset_3 = np.mean([fits.loc[s]['HFI_INT'][1] for s in teo_gain_steps[3::4]])
fits['HFI_INT'] = [(HFI_INT_slope_3, HFI_INT_offset_3),
                   (HFI_INT_slope_2, HFI_INT_offset_2),
                   (HFI_INT_slope_1, HFI_INT_offset_1),
                   (HFI_INT_slope_0, HFI_INT_offset_0)] * 8

# Now I go with the Low frequency mode
LFV = tuple(np.polyfit(LF_wfm, LF_data['V'], 1))
I_expected = LF_data['V'] / R
LFI = tuple(np.polyfit(I_expected, LF_data['I'], 1))
fits['LFV'] = np.repeat([LFV], 32)
fits['LFI'] = np.repeat([LFI], 32)

calibration = fits.copy()

for s, v in fits.iterrows():
    for column in ['HFV', 'HFV_INT', 'V_MONITOR', 'HFI_INT', 'HF_LIMITED_BW', 'HF_FULL_BW', 'LFV', 'LFI']:
        fit = fits.loc[s, column]
        calibration.loc[s, column] = np.array([1 / fit[0], -fit[1] / fit[0]])


calibration.attrs['timestamp'] = timestamp
calibration.attrs['resistor'] = R
calibration.attrs['sweep_rate'] = SR
calibration.attrs['wfm_samples'] = wfm_samples
calibration.attrs['set_up'] = set_up

# # # # # # # # # # # # # # # # # # # # # #  Saving results # # # # # # # # # # # # # # # # # # # # # #
if teo.calibration is None:
    save_folder = os.path.join(save_folder, f'{timestamp}')
else:
    save_folder = os.path.join(save_folder, f'{timestamp_chek}_check_{timestamp}')
os.makedirs(save_folder)
os.makedirs(os.path.join(save_folder, f'plots'))

calibration.to_pickle(os.path.join(save_folder, 'teo_calibration.df'))
data.to_pickle(os.path.join(save_folder, 'calibration_data.df'))


mpl.use('Agg')
for s, d in data.iterrows():
    log.info(f"Plotting Gain Step = {s}")
    ## Ploting fits ##
    V_wfm = d['V_wfm']
    I_expected = d['HFV'] / R

    size_factor = 3
    fig = plt.figure("a", figsize=(4 * size_factor, 3 * size_factor), dpi=200)
    fig.clf()
    axs = fig.subplots(2, 3)

    ax = axs[0, 0]  # HFV fit
    ax.axhline(0, color='gray', alpha=0.7)
    ax.axvline(0, color='gray', alpha=0.7)
    ax.plot(V_wfm, d['HFV'], label="Measurement", linewidth=3)
    ax.plot(V_wfm, np.polyval(fits.loc[s, 'HFV'], V_wfm), label="Fit")
    ax.set_xlabel("Programmed voltage [V]")
    ax.set_ylabel("HFV [V]")
    ax.legend(loc="upper left")

    ax = axs[0, 1]  # HFV_INT fit
    ax.axhline(0, color='gray', alpha=0.7)
    ax.axvline(0, color='gray', alpha=0.7)
    ax.plot(d['HFV'], d['HFV_INT'], label="Measurement", linewidth=3)
    ax.plot(d['HFV'], np.polyval(fits.loc[s, 'HFV_INT'], d['HFV']), label="Fit")
    ax.set_xlabel("Picoscope HFV [V]")
    ax.set_ylabel("HFV_INT [V]")

    ax = axs[0, 2]  # V_MONITOR fit
    ax.axhline(0, color='gray', alpha=0.7)
    ax.axvline(0, color='gray', alpha=0.7)
    ax.plot(d['HFV'], d['V_MONITOR'], label="Measurement", linewidth=3)
    ax.plot(d['HFV'], np.polyval(fits.loc[s, 'V_MONITOR'], d['HFV']), label="Fit")
    ax.set_xlabel("Picoscope HFV [V]")
    ax.set_ylabel("V_MONITOR [V]")

    ax = axs[1, 0]  # HFI_INT fit
    ax.axhline(0, color='gray', alpha=0.7)
    ax.axvline(0, color='gray', alpha=0.7)
    ax.plot(I_expected, d['HFI_INT'], label="Measurement", linewidth=3)
    ax.plot(I_expected, np.polyval(fits.loc[s, 'HFI_INT'], I_expected), label="Fit")
    ax.set_xlabel("Expected current [A]")
    ax.set_ylabel("HFI_INT [A]")

    ax = axs[1, 1]  # HF_LIMITED_BW fit
    ax.axhline(0, color='gray', alpha=0.7)
    ax.axvline(0, color='gray', alpha=0.7)
    ax.plot(I_expected, d['HF_LIMITED_BW'], label="Measurement", linewidth=3)
    ax.plot(I_expected, np.polyval(fits.loc[s, 'HF_LIMITED_BW'], I_expected), label="Fit")
    ax.set_xlabel("Expected current [A]")
    ax.set_ylabel("HF_LIMITED_BW [A]")

    ax = axs[1, 2]  # HF_FULL_BW fit
    ax.axhline(0, color='gray', alpha=0.7)
    ax.axvline(0, color='gray', alpha=0.7)
    ax.plot(I_expected, d['HF_FULL_BW'], label="Measurement", linewidth=3)
    ax.plot(I_expected, np.polyval(fits.loc[s, 'HF_FULL_BW'], I_expected), label="Fit")
    ax.set_xlabel("Expected current [A]")
    ax.set_ylabel("HF_FULL_BW [A]")

    title = f"Gain step = {s} ; " \
            f"Timestamp = {timestamp} ; " \
            f"Resistor = {R} ; " \
            f"Sweep rate = {SR}\n" \
            f"Wfm samples = {wfm_samples} ; " \
            f"Set up = {set_up}"

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f'plots/s{s}_fits.png'))

    ## Ploting measurements ##

    fig = plt.figure("a", figsize=(4 * size_factor, 3 * size_factor), dpi=200)
    fig.clf()
    axs = fig.subplots(2, 3)

    ax = axs[0, 0]  # ps_HFV
    ax.axhline(0, color='gray', alpha=0.7)
    ax.plot(d['t'], d['HFV'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HFV [V]")

    ax = axs[0, 1]  # teo_HFV
    ax.axhline(0, color='gray', alpha=0.7)
    ax.plot(d['t'], d['HFV_INT'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HFV_INT [V]")

    ax = axs[0, 2]  # V_MONITOR
    ax.axhline(0, color='gray', alpha=0.7)
    ax.plot(d['t'], d['V_MONITOR'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("V_MONITOR [V]")

    ax = axs[1, 0]  # HFI
    ax.axhline(0, color='gray', alpha=0.7)
    ax.plot(d['t'], d['HFI_INT'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HFI_INT [A]")

    ax = axs[1, 1]  # HF_LIMITED_BW
    ax.axhline(0, color='gray', alpha=0.7)
    ax.plot(d['t'], d['HF_LIMITED_BW'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HF_LIMITED_BW [A]")

    ax = axs[1, 2]  # HF_FULL_BW
    ax.axhline(0, color='gray', alpha=0.7)
    ax.plot(d['t'], d['HF_FULL_BW'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HF_FULL_BW [A]")

    title = f"Gain step = {s} ; " \
            f"Timestamp = {timestamp} ; " \
            f"Resistor = {R} ; " \
            f"Sweep rate = {SR}\n" \
            f"Wfm samples = {wfm_samples} ; " \
            f"Set up = {set_up}"

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f'plots/s{s}_measurements.png'))

## Ploting LF measurement ##
I_expected = LF_data['V'] / R

fig = plt.figure("a", figsize=(6 * size_factor, 3 * size_factor), dpi=200)
fig.clf()
axs = fig.subplots(2, 2)

ax = axs[0, 0]  # LFV
ax.axhline(0, color='gray', alpha=0.7)
ax.plot(LF_data['t'], LF_data['LFV'])
ax.set_xlabel("Time [s]")
ax.set_ylabel("LFV [V]")

ax = axs[0, 1]  # LFI
ax.axhline(0, color='gray', alpha=0.7)
ax.plot(LF_data['t'], LF_data['LFI'])
ax.set_xlabel("Time [s]")
ax.set_ylabel("LFI [A]")

ax = axs[1, 0]  # LFV fit
ax.axhline(0, color='gray', alpha=0.7)
ax.axvline(0, color='gray', alpha=0.7)
ax.plot(LF_wfm, LF_data['LFV'], label="Measurement", linewidth=3)
ax.plot(LF_wfm, np.polyval(fits.loc[s, 'LFV'], LF_wfm), label="Fit")
ax.set_xlabel("Programmed voltage [V]")
ax.set_ylabel("LFV [V]")
ax.legend(loc="upper left")

ax = axs[1, 1]  # LFI fit
ax.axhline(0, color='gray', alpha=0.7)
ax.axvline(0, color='gray', alpha=0.7)
ax.plot(I_expected, LF_data['LFI'], label="Measurement", linewidth=3)
ax.plot(I_expected, np.polyval(fits.loc[s, 'LFI'], I_expected), label="Fit")
ax.set_xlabel("Expected current [A]")
ax.set_ylabel("LFI [A]")

title = f"Timestamp = {timestamp} ; " \
        f"Resistor = {R} ; " \
        f"Data points = {len(LF_wfm)} ; " \
        f"NPLC = {nplc}" \
        f"Set up = {set_up}"

fig.suptitle(title)
fig.tight_layout()
fig.savefig(os.path.join(save_folder, f'plots/LF_mode.png'))

if check is False:
    if os.path.isfile(ivtools.settings.teo_calibration_file):
        teo.calibration = pd.read_pickle(ivtools.settings.teo_calibration_file)  # so you don't get crazy trying to
        # apply calibration later

log.info(f"Done! Results saved at:\n{save_folder}")
