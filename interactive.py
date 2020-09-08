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
TODO: GUI for displaying and changing channel settings, other status information
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
warnings.filterwarnings("ignore", ".*GUI is implemented.*")
import visa

import ivtools
import importlib
from importlib import reload
from ivtools import settings
from ivtools import analyze
from ivtools import plot as ivplot
from ivtools import instruments
from ivtools import io
from ivtools import measure
# Reload all the modules in case they changed
# every module EXCEPT settings
importlib.reload(ivtools.analyze)
importlib.reload(ivtools.plot)
importlib.reload(ivtools.instruments)
importlib.reload(ivtools.io)
importlib.reload(ivtools.measure)

# Dump everything into interactive namespace for convenience
# TODO: run test for overlapping names first
from ivtools.measure import *
from ivtools.analyze import *
from ivtools.plot import *
from ivtools.io import *
from ivtools.instruments import *
import logging

log = logging.getLogger('interactive')


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

    # Preview of the logging colors
    print('\nLogging color code:')
    for logger in ivtools.loggers.keys():
        print(f"\t{ivtools.loggers[logger].replace('%(message)s', logger)}")

hostname = settings.hostname
username = settings.username
db_path = settings.db_path  # Database path
datestr = time.strftime('%Y-%m-%d')
#datestr = '2019-08-07'
gitstatus = io.getGitStatus()
if 'M' in gitstatus:
    log.warning('The following files have uncommited changes:\n\t' + '\n\t'.join(gitstatus['M']))
    log.warning('Automatically committing changes!')
    gitCommit(message='AUTOCOMMIT')
if '??' in gitstatus:
    log.warning('The following files are untracked by git:\n\t' + '\n\t'.join(gitstatus['??']))
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
iplots = ivplot.InteractiveFigs(n=4)

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
# For Teo
teo_plotters = [[0, partial(ivplot.ivplotter, x='wfm')], # programmed waveform is less noisy
                [1, ivplot.itplotter],
                [2, ivplot.VoverIplotter],
                [3, ivplot.vtplotter]]

#########

datafolder = settings.datafolder
inst_connections = settings.inst_connections

globalvars = globals()
instrument_varnames = {instruments.Picoscope:'ps',
                       instruments.RigolDG5000:'rigol',
                       instruments.Keithley2600:'k',
                       instruments.TeoSystem:'teo',
                       instruments.PG5:'pg5',
                       instruments.Eurotherm2408:'et',
                       instruments.TektronixDPO73304D:'ttx',
                       instruments.USB2708HS:'daq',
                       instruments.WichmannDigipot: 'dp',
                       instruments.EugenTempStage: 'ts'}
# Make varnames None until connected
# but don't overwrite them if they exist already
for kk, v in instrument_varnames.items():
    if v not in globalvars:
        globalvars[v] = None

if visa.visa_rm is not None:
    visa_resources = visa.visa_rm.list_resources()
else:
    # you don't have visa installed, things probably won't end well.
    visa_resources = []

# Connect to all the instruments
# Instrument classes should all be Borg, because the instrument manager cannot be trusted
# to work properly and reuse existing inst_connections
for varname, inst_class, *args in inst_connections:
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
log.info('Data to be saved in {}'.format(os.path.join(datafolder, subfolder)))
log.info('Overwrite \'datafolder\' and/or \'subfolder\' variables to change directory')
io.makefolder(datafolder, subfolder)


def datadir():
    return os.path.join(datafolder, subfolder)
def open_datadir():
    os.system('explorer ' + datadir())
def cd_data():
    magic('cd ' + datadir())

# What the plots should do by default
if not iplots.plotters:
    if ps is not None:
        iplots.plotters = pico_plotters
        log.info('Setting up default plots for picoscope')
    elif k is not None:
        iplots.plotters = keithley_plotters
        log.info('Setting up default plots for keithley')

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


# noinspection SpellCheckingInspection
def savedata(data=None, folder_path=None, database_path=None, table_name='meta', drop=None):
    """
    Save data to disk and write a row of metadata to an sqlite3 database
    This is a "io.MetaHandler.savedata" wrapping but making use of "settings.py" parameters.

    :param data: Data to write to disk, non-array data is added to the database.
    :param folder_path: Folder where all data will be saved. If None, settings.datafolder/subfolder is used.
    :param database_path: Path of the database where data will be saved. If None, settings.dbpath is used.
    :param table_name: Name of the table in the database. If the table doesn't exist, create a new one.
    :param drop: drop columns to save disk space.
    """
    if data is None:
        global d
        if type(d) in (dict, list, pd.Series, pd.DataFrame):
            log.warning('No data passed to savedata(). Using global variable d.')
            data = d
    if folder_path is None:
        folder_path = datadir()
    if database_path is None:
        database_path = db_path
    meta.savedata(data, folder_path, database_path, table_name, drop)

def load_metadb(database_path=None, table_name='meta'):
    """
    Load the database into a data frame.
    :param database_path: Path of the database to load.
    :param table_name: Name of the database to load.
    :return: Table of the database as a pandas.DataFrame.
    """
    if database_path is None:
        database_path = db_path
    db = db_load(database_path, table_name)
    return db

s = autocaller(savedata)


#############################################################

###### Interactive measurement functions #######

# Wrap any fuctions that you want to automatically make plots/write to disk with this:
# TODO how can we neatly combine data from multiple sources (e.g. temperature readings?)
#      could use the same wrapper and just compose a new getdatafunc..
#      or pass a list of functions as getdatafunc, then smash the results together somehow
def interactive_wrapper(measfunc, getdatafunc=None, donefunc=None, live=False, autosave=True, shared_kws=None):
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
            if shared_kws:
                # Also pass specific keyword arguments to getdatafunc
                shared_kwargs = {k:v for k,v in kwargs.items() if k in shared_kws}
                newgetdatafunc = partial(getdatafunc, **shared_kwargs)
            else:
                newgetdatafunc = getdatafunc
            if live:
                iplots.newline()
                while not donefunc():
                    if live:
                        data = newgetdatafunc()
                        data = meta.attach(data)
                        iplots.updateline(data)
                    ivplot.mypause(0.1)
                data = newgetdatafunc()
                data = meta.attach(data)
                iplots.updateline(data)
            else:
                while not donefunc():
                    ivplot.mypause(0.1)
                data = newgetdatafunc()
                data = meta.attach(data)
                iplots.newline(data)
        if autosave:
            # print(data)
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
    kiv_lua = interactive_wrapper(k._iv_lua, k.get_data, donefunc=k.done, live=live, autosave=True, shared_kws=['ch'])
    kiv = interactive_wrapper(k.iv, k.get_data, donefunc=k.done, live=live, autosave=True, shared_kws=['ch'])
    kvi = interactive_wrapper(k.vi, k.get_data, donefunc=k.done, live=live, autosave=True)
    kit = interactive_wrapper(k.it, k.get_data, donefunc=k.done, live=live, autosave=True)

# define this if digipot is connected
if dp:
    def set_Rseries(val):
        Rs = dp.set_R(val)
        meta.static['R_series'] = Rs
        return Rs

# define this if temperature stage is connected
if ts:
    def set_temperature(T, delay=30):
        ts.set_temperature(T)
        ivplot.mybreakablepause(delay)
        meta.static['R_series'] = Rs

if teo:
    # HF mode
    teoiv = interactive_wrapper(teo.measure)

# TODO def reload_settings, def reset_state
