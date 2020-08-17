from ivtools.instruments import *
from ivtools.analyze import *
from ivtools.plot import *

import numpy as np
from matplotlib import pyplot as plt

teo = instruments.TeoSystem()

# Don't use Value, use Steps
minHFgain = teo.HF_Measurement.GetMinValue() # -8
maxHFgain = teo.HF_Measurement.GetMaxValue() # 10
gainvals = range(minHFgain, maxHFgain)
for val in gainvals:
    teo.HF_Measurement.HF_Gain.SetValue(val)
    returnval = teo.HF_Measurement.HF_Gain.GetValue()
    print(f'{val}: {returnval}')

# test a variety of short waveforms
# using only teo supplied functions, so that we can send the results for support if needed

for n in range(10, 4000):
    # random walky thing
    #wfm = np.sin(np.linspace(0, 2*np.pi, n)) * 0.1 * np.cumsum(np.random.randn(n))
    wfm = np.sin(np.linspace(0, 2*np.pi, n))
    # actually, best to use a square pulse so we can detect the jitter
    wfm = wfm > .5
    plt.figure()
    plt.plot(wfm)
    data = []


# are there zeros at the end? how many?

# Maximum samples we can upload?
