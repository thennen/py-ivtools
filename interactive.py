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

reload()
# Dump a bunch of stuff into the interactive namespace
# I feel this is worth doing, despite the downsides
from ivtools import *
from ivtools.measure import *
from pylab import *
from numpy import *

# Make some plot windows, put them in places
fig1, ax1 = plt.subplots()
ax1.set_title('IV Measurements')
ax1.set_xlabel('Voltage')
ax1.set_ylabel('Current')
fig1.canvas.manager.window.move(1265, 0)
fig2, ax2 = plt.subplots()
ax2.set_title('Picoscope Traces')
# Would be really cool to show the signals in real time...
ax2.set_xlabel('Data point')
ax2.set_ylabel('Voltage [V]')
fig2.canvas.manager.window.move(1265, 580)

# Try to connect the instruments
ivtools.measure.connect()

# Your data gets stored in this variable
data = np.array()
# Your metadata gets stored in this variable
meta = dotdict()
plt.show()
