'''
This file should be run using the %run -i magic in ipython.
Provides a command based user interface for IV measurements.
make sure %matplotlib is in your ipython startup, or just make sure to run it first.
'''
import numpy
np = numpy
import matplotlib
mpl = matplotlib
from matplotlib import pyplot
plt = pyplot
import os
import sys
import time
import ivtools
import ivtools.measure
import ivtools.plot
import ivtools.io
import ivtools.analyze

def reload():
    # Reload the ivtools modules
    # You will need to do this if you modify the code
    # There's probably a better way to do this
    # This is why I hate modules
    import importlib
    importlib.reload(ivtools)
    importlib.reload(ivtools.measure)
    importlib.reload(ivtools.plot)
    importlib.reload(ivtools.io)
    importlib.reload(ivtools.analyze)
    importlib.reload(ivtools)

# TODO: Don't overwrite certain variables

reload()
# Dump a bunch of stuff into the interactive namespace
# I feel this is worth doing, despite the downsides
from ivtools import *
from ivtools.measure import *
from pylab import *
from numpy import *

ivtools.measure.COUPLINGS = {'A': 'DC', 'B': 'DC', 'C': 'DC', 'D': 'DC'}
ivtools.measure.ATTENUATION = {'A': 1.0, 'B': 1, 'C': 1, 'D': 1.0}
ivtools.measure.OFFSET = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
ivtools.measure.RANGE = {'A': 2.0, 'B': 2.0, 'C': 1.0, 'D': 1.0}

# TODO: Can you get monitor size and pic figure size/location?
# For crappy small lab monitor
figsize = (6, 3.9)
fig1loc = (660, 0)
fig2loc = (660, 490)
# For remote desktop on a big monitor
# figsize = (6.4, 4.8)
# fig1loc = (1265, 0)
# fig2loc = (1265, 580)

try:
    # Close the figs if they already exist
    plt.close(fig1)
    plt.close(fig2)
except:
    pass
# Make some plot windows, put them in places
fig1, ax1 = plt.subplots(figsize=figsize)
ax1.set_title('IV Measurements')
ax1.set_xlabel('Voltage')
ax1.set_ylabel('Current')
fig1.canvas.manager.window.move(*fig1loc)
fig2, ax2 = plt.subplots(figsize=figsize)
ax2.set_title('Picoscope Traces')
# Would be really cool to show the signals in real time...
ax2.set_xlabel('Data point')
ax2.set_ylabel('Voltage [V]')
fig2.canvas.manager.window.move(*fig2loc)

# Try to connect the instruments
# TODO: ps opens every time.  is this a problem?  how can we reuse the last connection?
# TODO: don't override channel settings
ivtools.measure.connect()

# Your data gets stored in this variable
data = np.array([], dtype=object)
# Your metadata gets stored in this variable
meta = dotdict()
plt.show()

def iv(vmin, vmax, duration=None, rate=None, n=1, fs=1e7):
    '''
    Pulse a triangle waveform, plot pico channels, IV, and save to data variable
    '''
    global data

    # Channels that need to be sampled for measurement
    # channels = ['A', 'B', 'C']
    channels = ['A', 'B']

    # Need to know duration of pulse if only sweeprate is given
    # so that we know how long to capture
    sweeprate, pulsedur = ivtools.measure._rate_duration(vmin, vmax, rate, duration)

    # Set picoscope to capture
    ivtools.measure.pico_capture(ch=channels,
                                 freq=fs,
                                 duration=n*pulsedur)
    # Send a triangle pulse
    ivtools.measure.tripulse(n=n,
                            vmax=vmax,
                            vmin=vmin,
                            duration=duration,
                            rate=rate)
    # Get the picoscope data
    chdata = ivtools.measure.get_data(channels)
    # Convert to IV data
    ivdata = pico_to_iv(chdata)
    # Append that data to global
    #data = np.append(data, ivdata) -- Doesn't work because of stupid data structure
    # For now just overwrite data..
    data = ivdata

    # Plot the IV data
    ivtools.plot.plotiv(ivdata, ax=ax1)

    # Remove previous lines
    for l in ax2.lines[::-1]: l.remove()
    # Plot the channel data
    ivtools.plot.plot_channels(chdata, ax=ax2)

    plt.draw()
