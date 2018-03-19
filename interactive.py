# Trying to combine picoscope and keithley measurement scripts
'''
This file should be run using the %run -i magic in ipython.
Provides a command based user interface for IV measurements.
Binds convenient names to functions contained in other modules

This script can be rerun, and all of the code will be updated, with
your settings not overwritten

Therefore you can modify any part of the code while making measurements
and without ever leaving the program.

TODO: push some of the code into the main library. here we should only handle interactive plotting, logging, metadata, and exposure of data to global variables
TODO: Maintain a database of all the metadata for all the data files created
TODO: GUI for displaying and changing channel settings, other status information
TODO: create a class for managing the automatic plotting.  make adding plots easy.
'''
import numpy
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import pyplot as plt
from functools import wraps, partial
import os
import sys
import time
import pandas as pd
import subprocess
import socket
from datetime import datetime
from collections import defaultdict, deque
# Stop a certain matplotlib warning from showing up
import warnings
warnings.filterwarnings("ignore",".*GUI is implemented.*")

import ivtools
# Reload all the modules in case they changed
import importlib
importlib.reload(ivtools)
importlib.reload(ivtools.measure)
importlib.reload(ivtools.analyze)
importlib.reload(ivtools.plot)
importlib.reload(ivtools.io)
importlib.reload(ivtools.instruments)
from ivtools import measure
from ivtools import analyze
from ivtools import plot as ivplot
from ivtools import io

# Define this on the first run only
try:
    firstrun
except:
    firstrun = True

if not firstrun:
    # Store old settings/connections before they get clobbered by module reload
    # There's probably a super slick way to do this ....
    old = {}
    old['ps'] = ps
    old['rigol'] = rigol
    old['k'] = k
    old['COMPLIANCE_CURRENT'] = measure.COMPLIANCE_CURRENT
    old['INPUT_OFFSET'] = measure.INPUT_OFFSET
    # Plotter settings -- maybe even the plots shouldn't be remade

# Dump everything into interactive namespace for convenience
# TODO: run test for overlapping names
from ivtools.measure import *
from ivtools.analyze import *
from ivtools.plot import *
from ivtools.io import *
from ivtools.instruments import *

magic = get_ipython().magic

hostname = socket.gethostname()
gitrev = io.getGitRevision()
datestr = time.strftime('%Y-%m-%d')

if hostname == 'pciwe46':
    datafolder = r'D:\t\ivdata'
else:
    datafolder = r'C:\t\data'

# Default data subfolder
subfolder = datestr
if len(sys.argv) > 1:
    # Can give a folder name with command line argument
    subfolder += '_' + sys.argv[1]
print('Data to be saved in {}'.format(os.path.join(datafolder, subfolder)))

def datadir():
    return os.path.join(datafolder, subfolder)

io.makefolder(datafolder, subfolder)
print('Overwrite \'datafolder\' and/or \'subfolder\' variables to change directory')

if firstrun:
    ### Runs only the first time
    magic('matplotlib')
    io.log_ipy(True, os.path.join(datadir(), datestr + '_IPython.log'))
    meta = io.MetaHandler()
    iplots = ivplot.interactive_figs(n=4)
    # Just try to connect normally
    measure.connect_picoscope()
    measure.connect_rigolawg()
    measure.connect_keithley()
    firstrun = False
else:
    # Transfer all the settings you want to keep into new instances/environment
    if old['ps'] is not None:
        measure.ps = instruments.Picoscope(previous_instance=old['ps'])
    # with visa I think it's best to just close the session and reconnect
    if old['k'] is not None:
        kresource = old['k'].conn.resource_name
        old['k'].close()
        measure.connect_keithley(kresource)
    if old['rigol'] is not None:
        rigolresource = old['rigol'].conn.resource_name
        old['rigol'].close()
        measure.connect_rigolawg(rigolresource)
    measure.COMPLIANCE_CURRENT = old['COMPLIANCE_CURRENT']
    measure.INPUT_OFFSET = old['INPUT_OFFSET']
    meta = io.MetaHandler(oldinstance=meta)
    iplots = ivtools.plot.interactive_figs(oldinstance=iplots)

if measure.ps is not None:
    measure.ps.print_settings()

class autocaller():
    '''
    Ugly hack to make a function call itself without the parenthesis.
    There's an ipython magic for this, but I only want it to apply to certain functions
    '''
    def __init__(self, function):
        self.function = function
    def __repr__(self):
        self.function()
        return 'autocalled ' + self.function.__name__

# Add items to this and they will be appended as metadata to all subsequent measurements
meta.static = {'gitrev':gitrev}


# Default plotter configuration
# For picoscope + rigol
pico_plotters = [[0, ivplot.ivplotter],
                 [1, ivplot.chplotter],
                 [2, ivplot.VoverIplotter]]
# For keithley
kargs = {'marker':'.'}
keithley_plotters = [[0, partial(ivplot.ivplotter, **kargs)],
                     [1, partial(ivplot.itplotter, **kargs)],
                     [2, partial(ivplot.VoverIplotter, **kargs)],
                     [3, partial(ivplot.vtplotter, **kargs)]]

# Need to specify what the plots should do
# There are a few different ways one could handle this
# For example, switch the plotters when certain measurement functions are used
if k is None:
    print('Setting up automatic plotting for picoscope')
    iplots.plotters = pico_plotters
else:
    print('Setting up automatic plotting for Keithley')
    iplots.plotters = keithley_plotters



################ Bindings #################

# Instruments
ps = measure.ps
k = measure.k
rigol = measure.rigol

# Metadata selector
pp = autocaller(meta.print)
n = autocaller(meta.next)
p = autocaller(meta.previous)

# Plotter
figs = [None] * 6
figs[:len(iplots.figs)] = iplots.figs
fig0, fig1, fig2, fig3, fig4, fig5 = figs
axs = [None] * 6
axs[:len(iplots.axs)] = iplots.axs
ax0, ax1, ax2, ax3, ax4, ax5 = axs
clearfigs = iplots.clear
c = autocaller(clearfigs)

def savedata(data=None, filepath=None, drop=('A', 'B', 'C', 'D')):
    '''
    Save data with metadata attached.
    if no data is passed, try to use the global variable d
    filepath automatic by default.
    can drop columns to save disk space.
    '''
    if data is None:
        global d
        if type(d) in (dict, list, pd.Series, pd.DataFrame):
            print('No data passed to savedata(). Using global variable d.')
            data = d
    if filepath is None:
        filepath = os.path.join(datadir(), meta.filename())
    io.write_pandas_pickle(meta.attach(data), filepath, drop=drop)
    # TODO: append metadata to a sql table

# just typing s will save the d variable
s = autocaller(savedata)

# TODO: Would I ever want to turn off autosaving? autoplotting?  Could call the iv functions from measure.py directly..
# TODO: what if I don't have a function for when data is done being collected?

# Wrap any fuctions that you want to automatically make plots with this
def interactive_wrapper(func, getdatafunc=None, donefunc=None, live=False):
    ''' Activates auto data plotting and saving for wrapped functions '''
    @wraps(func)
    def func_with_plotting(*args, **kwargs):
        # Call function as normal
        if getdatafunc is None:
            data = func(*args, **kwargs)
            savedata(data)
            # Plot the data
            iplots.newline(data)
        else:
            # Measurement function is different from getting data function
            # This gives the possibility for live plotting
            func(*args, **kwargs)
            if live:
                iplots.newline()
                while not donefunc():
                    if live:
                        data = getdatafunc()
                        iplots.updateline(data)
                    plt.pause(0.1)
                data = getdatafunc()
                iplots.updateline(data)
            else:
                while not donefunc():
                    plt.pause(0.1)
                data = getdatafunc()
                iplots.newline(data)
            savedata(data)
        return data
    return func_with_plotting

picoiv = interactive_wrapper(measure.picoiv)

# If keithley is connected
if k is not None:
    def donesweeping():
        # works with smua.trigger.initiate()
        return not bool(float(k.ask('print(status.operation.sweeping.condition)')))
    def donemeasuring():
        # works with smua.measure.overlappediv()
        return not bool(float(k.ask('print(status.operation.measuring.condition)')))

    live = True
    if '2636A' in k.idn():
        live = False
    kiv = interactive_wrapper(k.iv, k.get_data, donefunc=donesweeping, live=live)
    kvi = interactive_wrapper(k.vi, k.get_data, donefunc=donesweeping, live=live)
    # TODO: this one has a different ending condition.  will not work as written
    kit = interactive_wrapper(k.it, k.get_data, donefunc=donemeasuring, live=live)
