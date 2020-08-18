from ivtools.instruments import *
from ivtools.analyze import *
from ivtools.plot import *

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

teo = instruments.TeoSystem()
self = teo

if 0:
    # Check strange things that gain settings do

    minHFgain = int(teo.HF_Measurement.HF_Gain.GetMinValue()) # -8
    print('HF_Gain.GetMinValue() = ', minHFgain)
    maxHFgain = int(teo.HF_Measurement.HF_Gain.GetMaxValue()) # 10
    print('HF_Gain.GetMaxValue() = ', maxHFgain)

    # Try using SetValue and see what comes back
    gainvals = range(minHFgain - 2, 32)
    SetValues = gainvals
    GetValues = []
    GetSteps = []
    for val in gainvals:
        teo.HF_Measurement.HF_Gain.SetValue(val)
        hfgain = teo.HF_Measurement.HF_Gain.GetValue()
        GetValues.append(hfgain)
        step = teo.HF_Measurement.HF_Gain.GetStep()
        GetSteps.append(step)
    result = pd.DataFrame(dict(SetValue=SetValues, GetValue=GetValues, GetStep=GetSteps))
    print(result)


    # Try using SetStep and see what comes back
    gainvals = range(0, 31)
    SetSteps = gainvals
    GetValues = []
    GetSteps = []
    for val in gainvals:
        teo.HF_Measurement.HF_Gain.SetStep(val)
        hfgain = teo.HF_Measurement.HF_Gain.GetValue()
        GetValues.append(hfgain)
        step = teo.HF_Measurement.HF_Gain.GetStep()
        GetSteps.append(step)
    result = pd.DataFrame(dict(SetStep=SetSteps, GetValue=GetValues, GetStep=GetSteps))
    print(result)

# test a variety of short waveforms
# using only teo supplied functions, so that we can send the results for support if needed
# I am interested in the jitter, waveform size restrictions, and strange appended 0s

data = []
for n in range(10, 50000, 501):
    # random walky thing
    #wfm = np.sin(np.linspace(0, 2*np.pi, n)) * 0.1 * np.cumsum(np.random.randn(n))
    sin = np.sin(np.linspace(0, 2*np.pi, n))
    # actually, best to use a square pulse so we can detect the jitter
    wfm = sin > .5
    #trig1 = sin > .4 # a different set of problems emerges when trig1 is not all true
    trig1 = np.ones_like(wfm, bool)
    trig2 = np.ones_like(wfm, bool)
    if 1: # use my software padding
        wfm, trig1, trig2 = teo._pad_wfms(wfm, trig1, trig2)
        n = len(wfm)
    name = 'test'
    wf = teo.AWG_WaveformManager.CreateWaveform(name)
    wf.AddSamples(wfm, trig1, trig2)
    shots = 1
    # repeat everything 10x
    for _ in range(10):
        success = teo.AWG_WaveformManager.Run(name, shots)
        wf00 = teo.AWG_WaveformManager.GetLastResult(0)
        wf01 = teo.AWG_WaveformManager.GetLastResult(1)
        V0 = np.array(wf00.GetWaveformDataArray())
        V1 = np.array(wf01.GetWaveformDataArray())

        d = dict(nsamples=n, ncapture=sum(trig1), V0=V0, V1=V1, wfm=wfm, triggedwfm=wfm[trig1])
        data.append(d)
df = pd.DataFrame(data)

# Now do some data analysis on the results
df['chunks'] = 1 + df.nsamples // 2048 # number of chunks that should have been used
df['lenV0'] = df.V0.apply(len)
df['lenV1'] = df.V1.apply(len)
df['numzeros'] = df.V0.apply(lambda x: np.sum(x == 0), 1)
df['extra_samples'] = df.lenV0 - df.ncapture

def trigger(x):
    w = np.where(x > 0.5)[0]
    if any(w):
        return w[0]
    else:
        return np.nan
df['V0trig'] = df.V0.apply(trigger)
df['wfmtrig'] = df.triggedwfm.apply(trigger)

df['trigdiff'] = df.V0trig - df.wfmtrig

def zeros_at_end(x):
    notzeros = np.where(x != 0)[0]
    if any(notzeros):
        return len(x) - np.max(notzeros) - 1
    else:
        return 0

def zeros_at_beginning(x):
    notzeros = np.where(x != 0)[0]
    if any(notzeros):
        return np.min(notzeros)
    else:
        return 0

df['V0_zeros_prepended'] = df.V0.apply(zeros_at_beginning)
df['V1_zeros_prepended'] = df.V1.apply(zeros_at_beginning)
df['V0_zeros_appended'] = df.V0.apply(zeros_at_end)
df['V1_zeros_appended'] = df.V1.apply(zeros_at_end)

# both waveforms are USUALLY affected the same, but I have seen this be false!
df.V1_zeros_appended == df.V0_zeros_appended

# Any waveform below 1024 samples is messed up
# there are lots of zeros prepended. the number seems random but is approximately
# min(n, 550)
plotiv(df[df.n < 1024], None, 'V0', labels='n')
plt.legend(title='Number of samples')
#plotiv(df[df.n < 1024], None, 'wfm', labels='n')

# ignoring the really short waveforms which are totally messed up
df2 = df[df.nsamples > 1024]
plt.figure()
plt.scatter(df2.trig - df2.wfmtrig, df2.V0_zeros_appended)
plt.xlabel('V0 edge - wfm edge (samples)')
plt.ylabel('zeros appended to end of waveform')

plt.figure()
plt.scatter(df2.extra_samples, df2.V0_zeros_appended)
plt.xlabel('# extra samples returned')
plt.ylabel('zeros appended to end of waveform')

if 0:
    # Can we read back the waveforms and triggers that we upload?
    n = 2048
    wfm = np.sin(np.linspace(0, 2*np.pi, n))
    trig1 = np.ones_like(wfm, bool)
    trig1[0::3] = 0
    trig1 = np.repeat(trig1[::2], 2) # convert to 250 MHz!
    trig2 = np.ones_like(wfm, bool)
    trig2[1::3] = 0
    trig2 = np.repeat(trig1[::2], 2) # convert to 250 MHz!
    name = 'test'
    wf = teo.AWG_WaveformManager.CreateWaveform(name)
    wf.AddSamples(wfm, trig1, trig2)

    data_back = teo.AWG_WaveformManager.GetWaveform(name)
    wfm_back = data_back.AllSamples()
    trig1_back = data_back.All_ADC_Gates()
    trig2_back = data_back.All_BER_Gates()

    print('wfm read back correctly?: ', all(wfm == wfm_back))
    print('')

    print('trig1 read back correctly? ', all(trig1 == trig1_back))
    print('trig1:')
    print(tuple(trig1[:10]), '...')
    print('Number of True values: ', sum(trig1))
    print('trig1_back:')
    print(trig1_back[:10], '...')
    print('Number of True values: ', sum(trig1_back))
    print('')

    print('trig2 read back correctly? ', all(trig2 == trig2_back))
    print('trig2:')
    print(tuple(trig2[:10]), '...')
    print('Number of True values: ', sum(trig2))
    print('trig2_back:')
    print(trig2_back[:10], '...')
    print('Number of True values: ', sum(trig2_back))
    '''
    wfm read back correctly?:  True

    trig1 read back correctly?  False
    trig1:
    (False, True, True, False, True, True, False, True, True, False) ...
    Number of True values:  1365
    trig1_back:
    (False, False, True, True, True, True, False, False, True, True) ...
    Number of True values:  1364

    trig2 read back correctly?  False
    trig2:
    (True, False, True, True, False, True, True, False, True, True) ...
    Number of True values:  1365
    trig2_back:
    (True, True, True, True, False, False, True, True, True, True) ...
    Number of True values:  1366
    '''

# Maximum samples we can upload?
