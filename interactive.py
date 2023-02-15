'''
This is a script that uses the ivtools library to provide a command-based user interface
for interactive IV measurements, for when you want a human in the feedback loop.

Here, we value brevity of commands that need to be typed into the console, and
the ability to modify arbitrary parts of the codebase without disrupting the
interactive measurement process.

This script is designed to be rerun, and all of the code will be updated, with
everything but the measurement settings and the program state overwritten.
Therefore you can modify any part of the code/library while making measurements
without ever leaving the running program or closing/opening instrument connections.
The need to restart the kernel should therefore be rare.

The file should be run (and rerun) using
%run -i interactive.py [folder name]
in ipython (Jupyter qtconsole).

Short version of what it does:
⋅ Puts all the functions from every ivtools module into the global namespace
⋅ Notifies user of the git status and can optionally auto-commit changes
⋅ Automatically connects to instruments as specified in the settings.py file
⋅ Uses a powerful (but minimalist) metadata management system (meta) that is connected to a local database
⋅ Creates a ISO8601 dated directory for storing data.
⋅ Logging of text input and output to the data directory, as well as other logging functionality
⋅ Opens a set of tiled figures (iplots) that know how to plot data and can be cleared etc from the console
⋅ Provides interactive versions of measurement functions that automatically plot and save data
⋅ Defines short bindings to certain function calls for interactive convenience (called without ())



IF THE PLOT WINDOWS OPEN AND THEN CLOSE IMMEDIATELY, YOU HAVE TO RUN %matplotlib BEFORE THIS SCRIPT!


TODO: In Spyder, ipython logging file isn't created, find out why
TODO: fix the %matplotlib thing
TODO: GUI for displaying and changing channel settings, other status information?
TODO define reload_settings, def reset_state
IDEA: Patch the qtconsole itself to enable global hotkeys (for sample movement, etc)
IDEA: buy a wireless keypad and make it index the metadata, start a measurement, etc

Author: Tyler Hennen (tyler@hennen.us)
'''
# Some of these imports are just to make sure the interactive user has access to them
# Not necessarily because they are used in this script!
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

# Because it does not autodetect in windows..
pd.set_option('display.width', 1000)
from datetime import datetime
from collections import defaultdict, deque
# Stop a certain matplotlib warning from showing up
import warnings
warnings.filterwarnings("ignore", ".*GUI is implemented.*")
import pyvisa as visa

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
# TODO: run test for overlapping names first (already written, in tests folder)
from ivtools.measure import *
from ivtools.analyze import *
from ivtools.plot import *
from ivtools.io import *
from ivtools.instruments import *
import logging

magic = get_ipython().run_line_magic

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
    magic('matplotlib', 'qt')
    # Preview of the logging colors
    print('\nLogging color code:')
    for logger in ivtools.loggers.keys():
        print(f"\t{ivtools.loggers[logger].replace('%(message)s', logger)}")
    print()
    sys.stdout.flush()

log = logging.getLogger('interactive')

# 𝗚𝗶𝘁
hostname = settings.hostname
username = settings.username
db_path = settings.db_path  # Database path
datestr = time.strftime('%Y-%m-%d')
#datestr = '2019-08-07'
gitstatus = io.getGitStatus()
if 'M' in gitstatus:
    log.warning('The following files have uncommited changes:\n\t' + '\n\t'.join(gitstatus['M']))
    if settings.autocommit:
        # TODO: auto commit to some kind of separate auto commit branch
        # problem is I don't want to pollute my commit history with a million autocommits
        # and git is not really designed to commit to branches that are not checked out
        # is this relevant?  https://github.com/bartman/git-wip
        log.info('Automatically committing changes!')
        gitCommit(message='AUTOCOMMIT')
if '??' in gitstatus:
    log.info('The following files are untracked by git:\n\t' + '\n\t'.join(gitstatus['??']))
gitrev = io.getGitRevision()

# 𝗠𝗲𝘁𝗮𝗱𝗮𝘁𝗮 𝗼𝗯𝗷𝗲𝗰𝘁
# Helps you step through the metadata of your samples/devices
meta = io.MetaHandler()

# Add items to meta.static and they will be appended as metadata to all subsequent measurements
# even if you step through the device index
meta.static['gitrev'] = gitrev
meta.static['hostname'] = hostname
meta.static['username'] = username


##########################################
# 𝗖𝗼𝗻𝗻𝗲𝗰𝘁 𝘁𝗼 𝗶𝗻𝘀𝘁𝗿𝘂𝗺𝗲𝗻𝘁𝘀
##########################################

inst_connections = settings.inst_connections

# Make varnames instances of this class until connected
# then referring to them doesn't simply raise an error
# (but don't overwrite them if they exist already)
class NotConnected():
    def __bool__(self):
        return False
    def __repr__(self):
        return 'Instrument not connected yet!'
instrument_varnames = ('ps','rigol','rigol2','keith','teo','sympuls','et','ttx','daq','dp','ts','cam', 'amb')
globalvars = globals()
for v in instrument_varnames:
    if v not in globalvars:
        globalvars[v] = NotConnected()

if visa.visa_rm is not None:
    visa_resources = visa.visa_rm.list_resources()
else:
    # you don't have visa installed, things probably won't end well.
    visa_resources = []

# Connect to all the instruments
# VISA instrument classes should all be Borg, because the instrument manager cannot be trusted
# to work properly and reuse existing inst_connections
if inst_connections: log.info('\nAutoconnecting to instruments...')
for varname, inst_class, *args in inst_connections:
    if len(args) > 0:
        if type(args[0])==str and (args[0].startswith('USB') or args[0].startswith('GPIB')):
            # don't bother trying to connect to it if it's not in visa_resources
            if args[0] not in visa_resources:
                # TODO: I think there are multiple valid formats for visa addresses.
                # How to equate them?
                # https://pyvisa.readthedocs.io/en/stable/names.html
                continue
    try:
        globalvars[varname] = inst_class(*args)
    except Exception as x:
        log.error(f'Autoconnection to {inst_class.__name__} failed: {x}')

#######################################
#𝗣𝗹𝗼𝘁𝘁𝗲𝗿 𝗰𝗼𝗻𝗳𝗶𝗴𝘂𝗿𝗮𝘁𝗶𝗼𝗻𝘀
#######################################

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
                 [3, partial(ivplot.vdeviceplotter, R=R_series)]]
# For keithley
kargs = {'marker':'.'}
keithley_plotters = [[0, partial(ivplot.vdeviceplotter, R=R_series, **kargs)],
                     [1, partial(ivplot.itplotter, **kargs)],
                     [2, partial(ivplot.VoverIplotter, **kargs)],
                     [3, partial(ivplot.vtplotter, **kargs)]]
# For Teo
teo_plotters = [[0, partial(ivplot.ivplotter, x='V')],  # programmed waveform is less noisy but I want to check V
                [1, ivplot.itplotter],
                [2, ivplot.VoverIplotter],
                [3, ivplot.vtplotter]]

teo_plotters_debug = [[0, partial(ivplot.plotiv, x='t', y='HFV')],
                      [1, partial(ivplot.plotiv, x='t', y='V')],
                      [2, partial(ivplot.plotiv, x='t', y='I')],
                      [3, partial(ivplot.plotiv, x='t', y='I2')]]

# What the plots should do by default
if not iplots.plotters:
    if ps:
        iplots.plotters = pico_plotters
        log.info('Setting up default plots for picoscope')
    elif keith:
        iplots.plotters = keithley_plotters
        log.info('Setting up default plots for keithley')
    elif teo:
        iplots.plotters = teo_plotters
        log.info('Setting up default plots for teo')


#################################################################################
# 𝗠𝗮𝗸𝗲 𝗱𝗮𝘁𝗮 𝗳𝗼𝗹𝗱𝗲𝗿 𝗮𝗻𝗱 𝗱𝗲𝗳𝗶𝗻𝗲 𝗵𝗼𝘄 𝗱𝗮𝘁𝗮 𝗶𝘀 𝘀𝗮𝘃𝗲𝗱
#################################################################################

datafolder = settings.datafolder
# Default data subfolder -- will reflect the date of the last time this script ran
# Will NOT automatically rollover to the next date during a measurement that runs past 24:00
subfolder = datestr
if len(sys.argv) > 1:
    # Can give a folder name with command line argument
    subfolder += '_' + sys.argv[1]
log.info('Data to be saved in {}'.format(os.path.join(datafolder, subfolder)))
# TODO:
log.info('Overwrite \'datafolder\' and/or \'subfolder\' variables to change directory')
io.makefolder(datafolder, subfolder)

def datadir():
    return os.path.join(datafolder, subfolder)

def open_datadir():
    os.system('explorer ' + datadir())

def cd_data():
    magic('cd ' + datadir())

# set up ipython log in the data directory
# TODO: make a copy and change location if datadir() changes
if firstrun:
    io.log_ipy(True, os.path.join(datadir(), datestr + '_IPython.log'))
    #iplots.plotters = keithley_plotters

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

    if drop is None:
        drop = settings.drop_cols

    meta.savedata(data, folder_path, database_path, table_name, drop)

def savefig(name=None, fig=None, **kwargs):
    '''
    Save a png of the figure in the data directory
    '''
    if fig is None:
        fig = plt.gcf()
    fn = meta.filename()
    if name:
        fn += '_' + name
    fp = os.path.join(datadir(), fn)
    fig.savefig(fp, **kwargs)
    log.info(f'Wrote {fp}')


################################################################
# 𝗕𝗶𝗻𝗱𝗶𝗻𝗴𝘀 𝗳𝗼𝗿 𝗶𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝘃𝗲 𝗰𝗼𝗻𝘃𝗲𝗻𝗶𝗲𝗻𝗰𝗲
################################################################

class autocaller():
    '''
    Ugly hack to make a function call itself without typing the parentheses.
    There's an ipython magic for this, but I only want it to apply to certain functions
    This is only for interactive convenience! Don't use it in a program or a script!
    '''
    def __init__(self, function, *args):
        self.function = function
        self.args = args
    def __repr__(self):
        self.function(*self.args)
        return 'autocalled ' + self.function.__name__


# Metadata selector
pp = autocaller(meta.print) # prettyprint
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

s = autocaller(savedata)


###########################################
# 𝗖𝗼𝗺𝗺𝗼𝗻 𝗰𝗼𝗻𝗳𝗶𝗴𝘂𝗿𝗮𝘁𝗶𝗼𝗻𝘀
###########################################

def setup_ccircuit(split=False):
    ps.coupling.a = 'DC'
    ps.coupling.b = 'DC50'
    ps.coupling.c = 'DC50'
    ps.coupling.d = 'DC50'
    ps.range.b = 2
    ps.range.c = 2
    ps.range.d = .5
    if split:
        settings.pico_to_iv = ccircuit_to_iv_split
    else:
        settings.pico_to_iv = ccircuit_to_iv
    iplots.plotters = pico_plotters

def setup_keithley():
    iplots.plotters = keithley_plotters
    iplots.preprocessing = []

def setup_digipot():
    ps.coupling.a = 'DC' # monitor
    ps.coupling.b = 'DC50' # device voltage
    ps.coupling.c = 'DC50' # current (unamplified)
    ps.range.b = 2
    ps.range.c = 0.05
    settings.pico_to_iv = digipot_to_iv
    iplots.plotters = pico_plotters

def setup_picoteo(HFV=None, V_MONITOR='B', HF_LIMITED_BW='C', HF_FULL_BW='D'):
    ps.coupling.a = 'DC'
    ps.coupling.b = 'DC'
    ps.coupling.c = 'DC50'
    ps.coupling.d = 'DC50'
    ps.range.a = 10
    ps.range.b = 1
    ps.range.c = 0.2
    ps.range.d = 0.2
    settings.pico_to_iv = partial(TEO_HFext_to_iv, HFV=HFV, V_MONITOR=V_MONITOR,
                                  HF_LIMITED_BW=HF_LIMITED_BW, HF_FULL_BW=HF_FULL_BW)
    iplots.plotters = teo_plotters

################################################################
# 𝗜𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝘃𝗲 𝗺𝗲𝗮𝘀𝘂𝗿𝗲𝗺𝗲𝗻𝘁 𝗳𝘂𝗻𝗰𝘁𝗶𝗼𝗻𝘀
################################################################

# Wrap any functions that you want to automatically make plots/write to disk with this:
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

        # Capture microscope camera image and store in the metadata after every measurement
        if settings.savePicWithMeas:
            if cam:
                frame = cam.getImg()
                frame = mat2jpg(frame,
                                scale = settings.camCompression["scale"],
                                quality = settings.camCompression["quality"])
                log.info('Updating camera image in metadata.')
                meta.meta.update({"cameraImage": frame})
            else:
                log.warning('No camera connected!')
        
        # Store ambient sensor data with measurement
        if settings.saveAmbient:
            if amb:
                ambient = amb.getAll()
                log.info('Updating ambient sensor data in metadata.')
                meta.meta.update({"ambientData": ambient})
            else:
                log.warning('No ambient sensor connected!')

        if autosave:
            # print(data)
            savedata(data)
            nointerrupt.breakpoint()
            nointerrupt.stop()

        measure.beep()
        return data

    measfunc_interactive.__signature__ = inspect.signature(measfunc)

    return measfunc_interactive

picoiv = interactive_wrapper(measure.picoiv)
digipotiv = interactive_wrapper(measure.digipotiv)
picoteoiv = interactive_wrapper(measure.picoteo)

def set_compliance(cc_value):
    # Just calls normal set_compliance and also puts the value in metadata
    meta.static['CC'] = cc_value
    measure.set_compliance(cc_value)

####  Stuff that gets defined only if a given instrument is present and connected

if ps:
    ps.print_settings()

if keith and keith.connected(): # Keithley is connected
    live = True
    if '2636A' in keith.idn():
        # This POS doesn't support live plotting
        live = False
    kiv_lua = interactive_wrapper(keith._iv_lua, keith.get_data, donefunc=keith.done, live=live, autosave=True, shared_kws=['ch'])
    kiv = interactive_wrapper(keith.iv, keith.get_data, donefunc=keith.done, live=live, autosave=True, shared_kws=['ch'])
    kvi = interactive_wrapper(keith.vi, keith.get_data, donefunc=keith.done, live=live, autosave=True)

if dp: # digipot is connected
    # TODO: monkeypatch dp.set_R instead?
    def set_Rseries(val):
        Rs = dp.set_R(val)
        meta.static['R_series'] = Rs
        return Rs

if ts: # temperature stage is connected
    # TODO: monkeypatch ts.set_temperature instead?
    def set_temperature(T, delay=30):
        ts.set_temperature(T)
        ivplot.mybreakablepause(delay)
        meta.static['T'] = ts.read_temperature()

if teo:
    # HF mode
    teoiv = interactive_wrapper(teo.measureHF)

# Microscope camera connected
if cam:
    def saveImg():
        path = os.path.join(datafolder, subfolder, meta.timestamp()+".png")
        cam.saveImg(path)
