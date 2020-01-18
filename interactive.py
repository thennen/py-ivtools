'''
IF THE PLOT WINDOWS OPEN AND THEN CLOSE IMMEDIATELY, YOU HAVE TO RUN %matplotlib BEFORE THIS SCRIPT!

This file should be run using the %run -i magic in ipython.
Provides a command based user interface for IV measurements.
Binds convenient names to functions contained in other modules

This script is designed to be rerun, and all of the code will be updated,
with everything but your measurement settings not overwritten.

Therefore you can modify any part of the code/library while making measurements
and without ever leaving the running program or closing instrument connections.

TODO: fix the %matplotlib thing
TODO: Maintain a proper database of all the metadata for all the data files created
TODO: GUI for displaying and changing channel settings, other status information
TODO: Replace all print() with a logging system and display them with colors
IDEA: Patch the qtconsole itself to enable global hotkeys (for sample movement, etc)
IDEA: buy a wireless keypad
'''
import numpy
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import pyplot as plt
from functools import wraps, partial
import os
import getpass # to get user name
import sys
import time
import pandas as pd
# Because it does not autodetect in windows..
pd.set_option('display.width', 1000)
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
from ivtools import settings

# Dump everything into interactive namespace for convenience
# TODO: run test for overlapping names
from ivtools.measure import *
from ivtools.analyze import *
from ivtools.plot import *
from ivtools.io import *
from ivtools.instruments import *


magic = get_ipython().magic

# Define this on the first run only
try:
    # Will cause exception if firstrun not defined
    firstrun
    # If it didn't it should be false
    firstrun = False
except:
    firstrun = True

if firstrun:
    # Don't run this more than once, or all the existing plots will get de-registered from the
    # matplotlib state machine or whatever and nothing will update anymore
    # TODO find out whether it has been called already
    magic('matplotlib')

hostname = socket.gethostname()
username = getpass.getuser()
datestr = time.strftime('%Y-%m-%d')
#datestr = '2019-08-07'
gitstatus = io.getGitStatus()
if 'M' in gitstatus:
    print('The following files have uncommited changes:')
    print('\n'.join(gitstatus['M']))
    print('Automatically committing changes')
    gitCommit(message='AUTOCOMMIT')
if '??' in gitstatus:
    print('The following files are untracked by git:')
    print('\n'.join(gitstatus['??']))
# TODO: auto commit to some kind of auto commit branch
# problem is I don't want to pollute my commit history with a million autocommits
# and git is not really designed to commit to branches that are not checked out
# is this relevant?  https://github.com/bartman/git-wip
gitrev = io.getGitRevision()

# Helps you step through the metadata of your samples/devices
meta = io.MetaHandler()

# 2634B : 192.168.11.11
# 2636A : 192.168.11.12
# 2636B : 192.168.11.13

######### Plotter configurations

# Make sure %matplotlib has been called! Or else figures will appear and then disappear.
iplots = ivplot.interactive_figs(n=4)

# Determine the series resistance from meta data
def R_series():
    # Check static meta first
    R_series = meta.static.get('R_series')
    if R_series is not None:
        return R_series
    # Check normal meta
    R_series = meta.meta.get('R_series')
    if R_series is not None:
        # If it is a lassen coupon, then convert to the measured values of series resistors
        wafer_code = meta.meta.get('wafer_code')
        if wafer_code == 'Lassen':
            Rmap = {0:143, 1000:2164, 5000:8197, 9000:12857}
            if R_series in Rmap:
                R_series = Rmap[R_series]
        return R_series
    else:
        # Assumption for R_series if there's nothing in the meta data
        return 0

# For picoscope + rigol
pico_plotters = [[0, ivplot.ivplotter],
                 [1, ivplot.chplotter],
                 [2, ivplot.VoverIplotter],
                 [3, partial(ivplot.vcalcplotter, R=R_series)]]
# For keithley
kargs = {'marker':'.'}
keithley_plotters = [[0, partial(ivplot.vcalcplotter, R=R_series, **kargs)],
                     [1, partial(ivplot.itplotter, **kargs)],
                     [2, partial(ivplot.VoverIplotter, **kargs)],
                     [3, partial(ivplot.vtplotter, **kargs)]]

#########


# Default settings, that may get overwritten
datafolder = r'C:\data'
connections = {}
# Hostname specific settings
# TODO move this code to a different file (settings.py?), so that it's obvious where the pc specific settings are
if hostname == 'pciwe46':
    if username == 'hennen':
        datafolder = r'D:\t\ivdata'
    else:
        datafolder = r'D:\{}\ivdata'.format(username)
    # Variable name, Instrument class, arguments to pass to init
    connections = [('ps', instruments.Picoscope),
                   ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
                   ('daq', instruments.USB2708HS),
                   ('ts', instruments.EugenTempStage),
                   ('dp', instruments.WichmannDigipot),
                   #('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                   #('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                   ('k', instruments.Keithley2600)] # Keithley can be located automatically now
elif hostname == 'pciwe38':
    # Moritz computer
    datafolder = r'C:\Messdaten'
    connections = {}
elif hostname == 'pcluebben2':
    datafolder = r'C:\data'
    connections = [#('et', instruments.Eurotherm2408),
                   #('ps', instruments.Picoscope),
                   #('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
                   #('daq', instruments.USB2708HS),
                  #('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                  #('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                   ('k', instruments.Keithley2600, 'GPIB0::27::INSTR')]
elif hostname == 'pciwe34':
    # Mark II
    # This computer and whole set up is a massive irredeemable piece of shit
    # computer crashes when you try to access the data drive
    # Data drive gets mounted on different letters for some reason
    # Therefore I will use the operating system drive..
    #datafolder = r'G:\Messdaten\hennen'
    datafolder = r'C:\Messdaten\hennen'
    connections = [('et', instruments.Eurotherm2408),
                   #('ps', instruments.Picoscope),
                   #('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
                   #('daq', instruments.USB2708HS),
                  #('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                  #('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                   ('k', instruments.Keithley2600, 'GPIB0::27::INSTR')]
elif hostname == 'CHMP2':
    datafolder = r'C:\data'
    connections = [('ps', instruments.Picoscope),
                   ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T161750020::INSTR'),
                   #('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                   #('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                   #TEO
                   ('p', instruments.UF2000Prober, 'GPIB0::5::INSTR')]
else:
    print(f'No Hostname specific settings found for {hostname}')

globalvars = globals()
instrument_varnames = {instruments.Picoscope:'ps',
                       instruments.RigolDG5000:'rigol',
                       instruments.Keithley2600:'k',
                       instruments.PG5:'pg5',
                       instruments.Eurotherm2408:'et',
                       instruments.TektronixDPO73304D:'ttx',
                       instruments.USB2708HS:'daq'}
# Make varnames None until connected
for kk,v in instrument_varnames.items():
    globalvars[v] = None

visa_resources = settings.visa_rm.list_resources()
# Connect to all the instruments
# Instrument classes should all be Borg, because the instrument manager cannot be trusted
# to work properly and reuse existing connections
for varname, inst_class, *args in connections:
    if len(args) > 0:
        if args[0].startswith('USB') or args[0].startswith('GPIB'):
            # don't bother trying to connect to it if it's not in visa_resources
            if args[0] not in visa_resources:
                # TODO: I think there are multiple valid formats for visa addresses.
                # How to equate them?
                # https://pyvisa.readthedocs.io/en/stable/names.html
                continue
    globalvars[varname] = inst_class(*args)

# Default data subfolder -- will reflect the date of the last time this script ran
# Will NOT automatically rollover to the next date during a measurement that runs past 24:00
subfolder = datestr
if len(sys.argv) > 1:
    # Can give a folder name with command line argument
    subfolder += '_' + sys.argv[1]
print('Data to be saved in {}'.format(os.path.join(datafolder, subfolder)))
print('Overwrite \'datafolder\' and/or \'subfolder\' variables to change directory')
io.makefolder(datafolder, subfolder)
def datadir():
    return os.path.join(datafolder, subfolder)


# What the plots should do by default
if not iplots.plotters:
    if ps is not None:
        iplots.plotters = pico_plotters
    elif k is not None:
        iplots.plotters = keithley_plotters

### Runs only the first time ###
if firstrun:
    io.log_ipy(True, os.path.join(datadir(), datestr + '_IPython.log'))
    #iplots.plotters = keithley_plotters

if ps is not None:
    ps.print_settings()

class autocaller():
    '''
    Ugly hack to make a function call itself without the parenthesis.
    There's an ipython magic for this, but I only want it to apply to certain functions
    This is only for interactive convenience! Don't use it in a program or a script.
    '''
    def __init__(self, function, *args):
        self.function = function
        self.args = args
    def __repr__(self):
        self.function(*self.args)
        return 'autocalled ' + self.function.__name__

# Add items to this and they will be appended as metadata to all subsequent measurements
meta.static['gitrev'] = gitrev
meta.static['hostname'] = hostname
meta.static['username'] = username

################ Bindings for interactive convenience #################

# Metadata selector
pp = autocaller(meta.print)
n = autocaller(meta.next)
p = autocaller(meta.previous)

left = autocaller(meta.move_domeb, 'left')
right = autocaller(meta.move_domeb, 'right')
up = autocaller(meta.move_domeb, 'up')
down = autocaller(meta.move_domeb, 'down')

# Plotter
figs = [None] * 6
figs[:len(iplots.figs)] = iplots.figs
fig0, fig1, fig2, fig3, fig4, fig5 = figs
axs = [None] * 6
axs[:len(iplots.axs)] = iplots.axs
ax0, ax1, ax2, ax3, ax4, ax5 = axs
clearfigs = iplots.clear
showfigs = iplots.show
c = autocaller(clearfigs)
sf = autocaller(iplots.show)
plotters = iplots.plotters
add_plotter = iplots.add_plotter
del_plotters = iplots.del_plotters

def savedata(data=None, filepath=None, drop=None):
    '''
    Save data with metadata attached, as determined by the state of the global MetaHandler instance
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
    # TODO: metadata might have already been attached by the interactive_wrapper.  This doesn't cause any problems, right?
    io.write_pandas_pickle(meta.attach(data), filepath, drop=drop)
    # TODO: append metadata to a sql table
# just typing s will save the d variable
s = autocaller(savedata)


#############################################################

###### Interactive measurement functions #######

# Wrap any fuctions that you want to automatically make plots/write to disk with this:
# TODO how can we neatly combine data from multiple sources (e.g. temperature readings?)
#      could use the same wrapper and just compose a new getdatafunc..
#      or pass a list of functions as getdatafunc, then smash the results together somehow
def interactive_wrapper(measfunc, getdatafunc=None, donefunc=None, live=False, autosave=True):
    ''' Activates auto data plotting and saving for wrapped measurement functions '''
    @wraps(measfunc)
    def measfunc_interactive(*args, **kwargs):
        if autosave:
            # Protect the following code from keyboard interrupt until after the data is saved
            nointerrupt = measure.controlled_interrupt()
            nointerrupt.start()
        if getdatafunc is None:
            # There is no separate function to get data
            # Assume that the measurement function returns the data
            data = measfunc(*args, **kwargs)
            data = meta.attach(data)
            # Plot the data
            iplots.newline(data)
        else:
            # Measurement function is different from getting data function
            # This gives the possibility for live plotting
            measfunc(*args, **kwargs)
            if live:
                iplots.newline()
                while not donefunc():
                    if live:
                        data = getdatafunc()
                        iplots.updateline(data)
                    ivplot.mypause(0.1)
                data = getdatafunc()
                data = meta.attach(data)
                iplots.updateline(data)
            else:
                while not donefunc():
                    ivplot.mypause(0.1)
                data = getdatafunc()
                data = meta.attach(data)
                iplots.newline(data)
        if autosave:
            savedata(data)
            nointerrupt.breakpoint()
            nointerrupt.stop()
        measure.beep()
        return data
    return measfunc_interactive

picoiv = interactive_wrapper(measure.picoiv)

# If keithley is connected ..
# because I put keithley in a stupid class, I can't access the methods unless it was instantiated correctly
if k and hasattr(k, 'query'):
    live = True
    if '2636A' in k.idn():
        # This POS doesn't support live plotting
        live = False
    kiv = interactive_wrapper(k.iv, k.get_data, donefunc=k.done, live=live, autosave=True)
    kiv_4pt = interactive_wrapper(k.iv_4pt, k.get_data, donefunc=k.done, live=live, autosave=True)
    kvi = interactive_wrapper(k.vi, k.get_data, donefunc=k.done, live=live, autosave=True)
    kit = interactive_wrapper(k.it, k.get_data, donefunc=k.done, live=live, autosave=True)

# TODO def reload_settings, def reset_state
