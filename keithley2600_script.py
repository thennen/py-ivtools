'''
Simple script to do IV sweeps with keithley 2600 (Using 2634B) without hating your life very much
Optionally pass a dataframe that contains sample/device information, and "fill it in" with iv data
Data is stored in a list of dicts, so that it can be appended in both axes efficiently
The data is then converted to a pandas dataframe for storage and higher level analysis

This is designed for capturing relatively few IV loops, which is basically any number obtainable by keithley
But should be convenient to measure a few loops on many devices

The programming pattern I am using is usually considered harmful -- lots of global variables being accessed by
lots of functions. But this is meant to be used interactively, with quick convenient access to all of the globals
and functions. The goal is to be very productive without sacrificing the ability to use the entire power of the
Python language interactively.

Author: Tyler Hennen 2017
'''

#TODO: have liveplotter only receive chunks of data it hasn't already received.  Communication of arrays is ridiculously slow even over ethernet
#TODO: find a nice way to share common code with other measurement setups

import visa
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
import ctypes
from collections import deque
import warnings
# Stop a certain matplotlib warning from showing up
warnings.filterwarnings("ignore",".*GUI is implemented.*")
import time
from shutil import copyfile
import os
from collections import defaultdict
import sys
import socket
import subprocess

def makedatafolder():
    datasubfolder = os.path.join(datafolder, subfolder)
    if not os.path.isdir(datasubfolder):
        print('Making folder: {}'.format(datasubfolder))
        os.makedirs(datasubfolder)

def getGitRevision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        # Either there is no git or you are not in the py-ivtools directory.
        # Don't error because of this
        return 'Dunno'

gitrev = getGitRevision()
print("Git Revision:" + gitrev)

datestr = time.strftime('%Y-%m-%d')
timestr = time.strftime('%Y-%m-%d_%H%M%S')
hostname = socket.gethostname()

if hostname == 'pciwe46':
    datafolder = 'D:/t/ivdata/'
else:
    datafolder = r'C:\t\data'
subfolder = datestr
print('Data to be saved in {}'.format(os.path.join(datafolder, subfolder)))
makedatafolder()

print('Overwrite \'datafolder\' and/or \'subfolder\' variables to change directory')


############# Logging  ###########################
# Want IPython to save a log of input/output in the data directory
# This code is supposed to transparently mirror ipython input/output as well as standard output to a file
# However it's a bit of a hack and sometimes has problems when you run this script again (spyder crashes for example)

magic = get_ipython().magic
magic('logstop')
# Fancy logging of ipython in and out, as well as standard out
try:
    sys.stdout = stdstdout
except:
    # This should run first time only
    stdstdout = sys.stdout
class Logger(object):
    def __init__(self):
        self.terminal = stdstdout
        self.log = open(logfile, 'a')

    def write(self, message):
        self.terminal.write(message)
        # Comment the lines and append them to ipython log file
        self.log.writelines(['#[Stdout]# {}\n'.format(line) for line in message.split('\n') if line != ''])

    def flush(self):
        self.log.flush()
        self.terminal.flush()
try:
    # Close the previous file
    logger.log.close()
except:
    pass
logfile = os.path.join(datafolder, subfolder, datestr + '_IPython.log')
magic('logstart -o {} append'.format(logfile))
logger = Logger()
sys.stdout = logger

############# End of Logging ###########################


# Rather than importing the modules and dealing with reload shenanigans that never actually work, use ipython run magic
magic('matplotlib')
if hostname == 'pciwe46':
    ivtoolsdir = 'C:/t/py-ivtools/'
elif hostname == 'fenster': # Just guessed craptop hostname ..
    ivtoolsdir = 'c:/Users/t/Desktop/py-ivtools'
else:
    # Hope you are already running in the py-ivtools directory
    ivtoolsdir = '.'
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/measure.py')))
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/plot.py')))
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/io.py')))
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/analyze.py')))

############# Keithley 2600 functions ###############

# Connect to Keithley
# 2634B
#Keithley_ip = '192.168.11.11'
# 2636A
Keithley_ip = '192.168.11.12'
Keithley_id = 'TCPIP::' + Keithley_ip + '::inst0::INSTR'
rm = visa.ResourceManager()
try:
    k
except:
    k = rm.get_instrument(Keithley_id)
print('Keithley *IDN?')
idn = k.ask('*IDN?')
print(idn)

# Sadly, Keithley decided to embed a lua interpreter into its source meters instead of providing a proper programming interface.
# This means we have to communicate with Keithley via sending and receiving strings in the lua programming language.
# One could decide to wrap every useful lua command in a python function which writes the lua string, and parses the response, but this would be quite an undertaking
# Here we maintain a separate lua file "Keithley_2600.lua" which defines lua functions on the keithley, then we wrap those in python.

def run_lua_lines(lines):
    ''' Send some lines (list of strings) to Keithley lua interpreter '''
    k.write('loadandrunscript')
    for line in lines:
        k.write(line)
    k.write('endscript')


def run_lua_file(filepath):
    ''' Send the contents of a file to Keithley lua interpreter '''
    with open(filepath, 'r') as kfile:
        run_lua_lines(kfile.readlines())


# Run the separatue lua file on keithley
run_lua_file('lua/Keithley_2600.lua')


def send_list_to_keithley(list_in, varname='pythonlist'):
    '''
    In order to send a list of values to keithley, we need to compose a lua string to define it as a variable.
    Problem is that the input buffer of Keithley is very small, so the lua string needs to be separated into many lines.
    This function accomplishes that.
    '''
    chunksize = 50
    l = len(list_in)
    # List of commands to send to keithley
    cmdlist = []
    cmdlist.append('{} = {{'.format(varname))
    # Split array into chunks and write the string representation line-by-line
    for i in range(0, l, chunksize):
        chunk = ','.join(['{:.6e}'.format(v) for v in list_in[i:i+chunksize]])
        cmdlist.append(chunk)
        cmdlist.append(',')
    cmdlist.append('}')

    run_lua_lines(cmdlist)


def iv(vlist, Irange, Ilimit, nplc=1, delay='smua.DELAY_AUTO', plot=True, live=True):
    '''Wraps the SweepVList lua function defined on keithley''' 

    # Send list of voltage values to keithley
    send_list_to_keithley(vlist, varname='sweeplist')

    # TODO: make sure the inputs are valid
    k.write('SweepVList(sweeplist, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay))

    # Plotting has the side effect of assigning data to the global d variable
    # If you don't plot, function will just return and you will have to call getdata() manually
    if plot:
        if live:
            liveplotter()
        else:
            # In case keithley doesn't like to read from its buffers during a sweep..
            keithley_waitready()
            # will just make the plot once
            liveplotter()

def vi(ilist, Vrange, Vlimit, nplc=1, delay='smua.DELAY_AUTO', plot=True, live=True):
    '''Wraps the SweepIList lua function defined on keithley''' 

    # Send list of voltage values to keithley
    send_list_to_keithley(ilist, varname='sweeplist')

    # TODO: make sure the inputs are valid
    Irange = np.max(np.abs(ilist))
    k.write('SweepIList(sweeplist, {}, {}, {}, {}, {})'.format(Vrange, Vlimit, nplc, delay, Irange))

    # Plotting has the side effect of assigning data to the global d variable
    # If you don't plot, function will just return and you will have to call getdata() manually
    if plot:
        if live:
            liveplotter()
        else:
            # In case keithley doesn't like to read from its buffers during a sweep..
            keithley_waitready()
            # will just make the plot once
            liveplotter()


def ti(sourceVA, sourceVB, points, interval,rangeI, limitI, nplc):
    '''Wraps the constantVoltageMeasI lua function defined on keithley''' 
    # Call constantVoltageMeasI
    # TODO: make sure the inputs are valid
    print('constantVMeasI({}, {}, {}, {}, {}, {}, {})'.format(sourceV, sourceVB, points, interval, rangeI, limitI, nplc))
    k.write('constantVMeasI({}, {}, {}, {}, {}, {}, {})'.format(sourceV, sourceVB, points, interval, rangeI, limitI, nplc))
    liveplotter()
    #k.write('smua.source.levelv = 0')
    #k.write('smua.source.output = smub.OUTPUT_OFF')
    #k.write('smub.source.levelv = 0')
    #k.write('smub.source.output = smub.OUTPUT_OFF')


def keithley_waitready():
    ''' There's probably a better way to do this. '''
    k.write('waitcomplete()')
    k.write('print(\"Complete\")')
    answer = None
    while answer is None:
        try:
            # Keep trying to read until keithley says Complete
            answer = k.read()
        except:
            pass

    '''
    # Another way ...
    answer = 1
    while answer != 0.0:
        answer = float(k.ask('print(status.operation.sweeping.condition)'))
        time.sleep(.3)
    '''


def keithley_readbuffer(buffer='smua.nvbuffer1' , attr='readings', start=1, end=None):
    '''
    Read a data buffer and return an actual array.
    Keithley 2634B handles this just fine while still doing a sweep
    Keithley 2636A throws error 5042 - cannot perform requested action while overlapped operation is in progress.
    '''
    if end is None:
        # Read the whole length
        end = int(float(k.ask('print({}.n)'.format(buffer))))
    # makes keithley give numbers in ascii
    # k.write('format.data = format.ASCII')
    #readingstr = k.ask('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
    #return np.float64(readingstr.split(', '))

    # Makes keithley give numbers in binary float64
    # Should be much faster?
    k.write('format.data = format.REAL64')
    k.write('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
    # reply comes back with #0 or something in the beginning and a newline at the end
    raw = k.read_raw()[2:-1]
    return np.fromstring(raw, dtype=np.float64)

def getdata(start=1, end=None, history=True):
    '''
    Ask Keithley to print out the data arrays of interest (I, V, t, ...)
    Parse the strings into python arrays
    Return dict of arrays
    dict can also contain scalar values or other meta data

    Can pass start and end values if you want just a specific part of the arrays
    '''
    global dhistory
    numpts = int(float(k.ask('print(smua.nvbuffer1.n)')))
    if end is None:
        end = numpts
    if numpts > 0:
        # Output a dictionary with voltage/current arrays and other parameters
        out = {}
        out['units'] = {}
        out['longnames'] = {}

        # Keithley returns this special value when the measurement is out of range
        # replace it with a real nan so it doesn't mess up the plots
        nanvalue = 9.9100000000000005e+37

        ### Collect measurement conditions
        # TODO: What other information is available from Keithley registers?

        # Need to do something different if sourcing voltage vs sourcing current
        source = k.ask('print(smua.source.func)')
        source = float(source)
        if source:
            # Returns 1.0 for voltage source (smua.OUTPUT_DCVOLTS)
            out['source'] = 'V'
            out['V'] = keithley_readbuffer('smua.nvbuffer2', 'sourcevalues', start, end)
            Vmeasured = keithley_readbuffer('smua.nvbuffer2', 'readings', start, end)
            Vmeasured[Vmeasured == nanvalue] = np.nan
            out['Vmeasured'] = Vmeasured
            out['units']['Vmeasured'] = 'V'
            I = keithley_readbuffer('smua.nvbuffer1', 'readings', start, end)
            I[I == nanvalue] = np.nan
            out['I'] = I
            out['Icomp'] = float(k.ask('print(smua.source.limiti)'))
        else:
            # Current source
            out['source'] = 'I'
            out['Vrange'] =  float(k.ask('print(smua.nvbuffer2.measureranges[1])'))
            out['Vcomp'] = float(k.ask('print(smua.source.limitv)'))

            out['I'] = keithley_readbuffer('smua.nvbuffer1', 'sourcevalues', start, end)
            Imeasured = keithley_readbuffer('smua.nvbuffer1', 'readings', start, end)
            Imeasured[Imeasured == nanvalue] = np.nan
            out['Imeasured'] = Imeasured
            out['units']['Imeasured'] = 'A'
            V = keithley_readbuffer('smua.nvbuffer2', 'readings', start, end)
            V[V == nanvalue] = np.nan
            out['V'] = V

        out['t'] = keithley_readbuffer('smua.nvbuffer2', 'timestamps', start, end)
        out['Irange'] = keithley_readbuffer('smua.nvbuffer1', 'measureranges', start, end)
        out['Vrange'] = keithley_readbuffer('smua.nvbuffer2', 'measureranges', start, end)

        out['units']['I'] = 'A'
        out['units']['V'] = 'V'

    else:
        empty = np.array([])
        out = dict(t=empty, V=empty, I=empty, Vmeasured=empty)
    if history:
        dhistory.append(out)
    return out


def triangle(v1, v2, n=None, step=None):
    '''
    We like triangle sweeps a lot
    Very basic triangle pulse with some problems.
    Give either number of points or step size
    Pass either v1 or v2 = 0 to do a single sided sweep.
    '''
    if n is not None:
        dv = abs(v1) + abs(v2 - v1) + abs(v2)
        step = dv / n
    wfm = np.concatenate((np.arange(0, v1, np.sign(v1) * step),
                        np.arange(v1, v2, np.sign(v2 - v1) * step),
                        np.arange(v2, 0, -np.sign(v2) * step),
                        [0]))
    return wfm


def waitforkeithley():
    # I can't find out how to do this from the manual.  Should be possible
    # Shouldn't be necessary if you don't run any blocking scripts on keithley
    # Also I think this is dumb and should never be used
    k.write('print("lol")')
    done = None
    while done is None:
        try:
            done = k.read()
        except:
            plt.pause(.1)
            print('waiting for keithley...')


def keithley_error():
    ''' Get the next error from keithley queue'''
    return k.ask('print(errorqueue.next())')


#################### For interactive collection of IV loops ########################

# We construct a data set in a list of dictionaries
# The list of dictionaries can then be immediately converted to dataframe for storage and analysis

# We append metadata to the IV data -- devicemeta + staticmeta + ivarrays

# Data you care about
try:
    data
except:
    print('Defining data = []')
    data = []
else:
    if data != []:
        answer = input('\'data\' variable not empty.  Clobber it? ')
        if answer.lower() == 'y':
            print('Defining data = []')
            data = []

# Data you forgot to save (100 of them)
try:
    dhistory
except:
    print('Defining dhistory = deque(maxlen=100)')
    dhistory = deque(maxlen=10)
else:
    if len(dhistory) > 0:
        answer = input('\'dhistory\' variable not empty.  Clobber it? ')
        if answer.lower() == 'y':
            print('Defining dhistory = deque(maxlen=100)')
            dhistory = deque(maxlen=10)

# The data index you are currently on
meta_i = None

d = None

# Add keys to this and they will be appended as metadata to all subsequent measurements
staticmeta = {'keithley':idn, 'script':'keithley2600_script.py', 'gitrev':gitrev, 'scriptruntime':timestr}

# Metadata about the device currently being probed.  Controlled by a few of the following functions
devicemeta = {}

# Change this list of dicts (or dataframe) before starting
devicemetalist = {'device_number':n for n in range(100)}

# This controls which keys are printed when identifying a device
prettykeys = []
# The value of these keys will be written to the filename
filenamekeys = []

# Example of setting meta list
def load_lassen(**kwargs):
    ''' Load lassen, specify lists of keys to match on
    e.g. coupon=[23, 24], module=['001H', '014B']
    '''
    # Could of course specify devices by any other criteria (code name, deposition date, thickness ...)
    global lassen_df, meta_df, prettykeys, filenamekeys, devicemetalist, deposition_df
    # Load information from files on disk
    deposition_df = pd.read_excel('sampledata/CeRAM_Depositions.xlsx', header=8, skiprows=[9])
    # Only use Lassen devices
    deposition_df = deposition_df[deposition_df['wafer_code'] == 'Lassen']
    lassen_df = pd.read_pickle(r"sampledata/all_lassen_device_info.pkl")
    # Merge data
    merge_deposition_data_on = ['coupon']

    # If someone neglected to write the coupon number in the deposition sheet
    # Merge the non-coupon specific portion of lassen_df
    coupon_cols = ['coupon', 'die_x', 'die_y', 'die']
    non_coupon_cols = [c for c in lassen_df.columns if c not in coupon_cols]
    non_coupon_specific = lassen_df[lassen_df.coupon == 30][non_coupon_cols]
    lassen_df = pd.concat((lassen_df, non_coupon_specific))

    meta_df = pd.merge(lassen_df, deposition_df, how='left', on=merge_deposition_data_on)

    # Check that function got valid arguments
    for key, values in kwargs.items():
        if key not in meta_df.columns:
            raise Exception('Key must be in {}'.format(meta_df.columns))
        if isinstance(values, str) or not hasattr(values, '__iter__'):
            kwargs[key] = [values]

    #### Filter kwargs ####
    for key, values in kwargs.items():
        meta_df = meta_df[meta_df[key].isin(values)]
    #### Filter devices to be measured #####
    devices001 = [2,3,4,5,6,7,8]
    devices014 = [4,5,6,7,8,9]
    meta_df = meta_df[~((meta_df.module_num == 1) & ~meta_df.device.isin(devices001))]
    meta_df = meta_df[~((meta_df.module_num == 14) & ~meta_df.device.isin(devices014))]

    meta_df = meta_df.sort_values(by=['coupon', 'module', 'device'])

    # Try to convert data types
    typedict = dict(wafer_number=np.uint8,
                    coupon=np.uint8,
                    sample_number=np.uint16,
                    number_of_dies=np.uint8,
                    cr=np.uint8,
                    thickness_1=np.uint16,
                    thickness_2=np.uint16,
                    dep_temp=np.uint16,
                    etch_time=np.float32,
                    etch_depth=np.float32)
    for k,v in typedict.items():
        # int arrays don't support missing data, because python sucks
        if not any(meta_df[k].isnull()):
            meta_df[k] = meta_df[k].astype(v)

    devicemetalist = meta_df
    prettykeys = ['dep_code', 'sample_number', 'die', 'module', 'device', 'width_nm', 'R_series', 'layer_1', 'thickness_1']
    filenamekeys = ['dep_code', 'sample_number', 'module', 'device']
    print('Loaded {} devices into devicemetalist'.format(len(devicemetalist)))


# Dumb hack to call a function without using ()
# Because who wants to type?
class autocaller():
    def __init__(self, function):
        self.function = function
    def __repr__(self):
        self.function()
        return 'autocalled ' + self.function.__name__

def prettyprint_meta(hlkeys=None):
    # Print some information about the device
    global prettykeys
    if prettykeys is None:
        # Print all the information
        prettykeys = devicemeta.keys()
    for key in prettykeys:
        if key in devicemeta.keys():
            if hlkeys is not None and key in hlkeys:
                print('{:<18}\t{:<8} <----- Changed'.format(key[:18], devicemeta[key]))
            else:
                print('{:<18}\t{}'.format(key[:18], devicemeta[key]))

pp = autocaller(prettyprint_meta)

def nextsample():
    ''' Go to the next device '''
    global meta_i, devicemeta
    lastdevicemeta = devicemeta
    if meta_i is None:
        meta_i = 0
    else:
        meta_i += 1
    if len(devicemetalist) > meta_i:
        if type(devicemetalist) == pd.DataFrame:
            devicemeta = devicemetalist.iloc[meta_i]
        else:
            devicemeta = devicemetalist[meta_i]
    else:
        print('There is no more data in devicemetalist')
        return
    # Highlight keys that have changed
    hlkeys = []
    for key in devicemeta.keys():
        if key not in lastdevicemeta.keys() or devicemeta[key] != lastdevicemeta[key]:
            hlkeys.append(key)
    print('You are now measuring this device (index {} of devicemetalist):'.format(meta_i))
    # Print some information about the device
    prettyprint_meta(hlkeys)

n = autocaller(nextsample)

def previoussample():
    ''' Go to the previous device '''
    global meta_i, devicemeta
    lastdevicemeta = devicemeta
    meta_i -= 1
    if type(devicemetalist) == pd.DataFrame:
        devicemeta = devicemetalist.iloc[meta_i]
    else:
        devicemeta = devicemetalist[meta_i]
    # Highlight keys that have changed
    hlkeys = []
    for key in devicemeta.keys():
        if key not in lastdevicemeta.keys() or devicemeta[key] != lastdevicemeta[key]:
            hlkeys.append(key)
    print('You are now measuring this device (index {} of devicemetalist):'.format(meta_i))
    # Print some information about the device
    prettyprint_meta(hlkeys)

p = autocaller(previoussample)

### Functions that write to disk

def savedata(datadict=None, filename=None):
    ''' Write the current value of the d variable to places, attach metadata '''
    if datadict is None:
        global d
    else:
        d = datadict
    # Append all the data together
    # devicemeta might be a dict or a series
    print('Appending metadata to last iv loop measured:')
    prettyprint_meta()
    print('...')
    d.update(dict(devicemeta))
    d.update(staticmeta)
    # Write series to disk.  Append the path to metadata
    if filename is None:
        filename = time.strftime('%Y-%m-%d_%H%M%S')
        for fnkey in filenamekeys:
            if fnkey in d.keys():
                filename += '_{}'.format(d[fnkey])
        # s for series
        filename += '.s'
    filepath = os.path.join(datafolder, subfolder, filename)
    d['datafilepath'] = filepath
    makedatafolder()
    print('converting variable \'d\' to pd.Series and writing as pickle to {}'.format(filepath))
    pd.Series(d).to_pickle(filepath)
    # Append series to data list
    print('Appending variable \'d\' to list \'data\'')
    global data
    data.append(d)
    return filepath

s = autocaller(savedata)

def savedatalist(filename=None):
    if filename is None:
        # TODO: Figure out a filename that doesn't collide
        filename = '{}_keithley_loops.df'.format(datestr)
    filepath = os.path.join(datafolder, subfolder, filename)
    print('Writing {}'.format(filepath))
    df = pd.DataFrame(data)
    df.to_pickle(filepath)
    # Take out known arrays and export xls
    xlspath = os.path.splitext(filepath)[0] + '.xls'
    print('Writing {}'.format(xlspath))
    df.loc[:, ~df.columns.isin(['I', 'V', 't', 'Vmeasured'])].to_excel(xlspath)

### Functions that actually tell Keithley to do something

def testshort():
    ''' Do some measurement '''
    # Keep increasing voltage amplitude and feed back on the resistance
    iv(triangle(.01, -.01, 80), Irange=1e-2, Ilimit=1e-3)
    R = Rfitplotter()
    print('Resistance: {}'.format(R))
    #d['note'] = 'Short?'
    return R

def chooserange(Restimate):
    ''' Choose a range once you have an estimate of the resistance'''
    return Irange, Ilimit

def calculate_resistance():
    # This belongs in the analyze.py file
    mask = abs(d['V']) <= .1
    line = np.polyfit(d.V[mask], d['I'][mask], 1)
    R = 1 / line[0]
    return R

############# This stuff handles the plots that you see while running the script ###############
# Define as many plotter functions as you want.  They should make one line per function.
# A dict at the end determines which are used by the automatic plotting
# TODO: figure out how to handle more than one plotter per axis


def ivplotter(ax=None, **kwargs):
    ''' This defines what gets plotted on ax1'''
    if ax is None:
        ax = ax1
    ax.plot(d['V'], 1e6 * d['I'], '.-', **kwargs)
    # Fit a line and label the resistance?
    color = ax.lines[-1].get_color()
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Current [$\mu$A]', color=color)

def Rfitplotter(ax=None, **kwargs):
    ''' Plot a line of resistance fit'''
    if ax is None:
        ax = ax1
    mask = abs(d['V']) <= .1
    if sum(mask) > 1:
        line = np.polyfit(d['V'][mask], d['I'][mask], 1)
    else:
        line = [np.nan, np.nan]
    # Plot line only to max V or max I
    R = 1 / line[0]
    vmin = max(min(d['V']), min(d['I'] * R))
    vmax = min(max(d['V']), max(d['I'] * R))
    # Do some points in case you put it on a log scale later
    fitv = np.linspace(1.1 * vmin, 1.1 * vmax, 10)
    fiti = np.polyval(line, fitv)
    plotkwargs = dict(color='black', alpha=.3, linestyle='--')
    plotkwargs.update(kwargs)
    ax.plot(fitv, 1e6 * fiti, **plotkwargs)
    return R

def complianceplotter(ax=None, **kwargs):
    # Plot a dotted line indicating compliance current
    pass

def vtplotter(ax=None, **kwargs):
    if ax is None:
        ax = ax2
    ''' This defines what gets plotted on ax2'''
    ax.plot(d['t'], d['V'], '.-', **kwargs)
    color = ax.lines[-1].get_color()
    ax.set_ylabel('Voltage [V]', color=color)
    ax.set_xlabel('Time [S]')

def itplotter(ax=None, **kwargs):
    if ax is None:
        ax = ax3
    ax.plot(d['t'], 1e6 * d['I'], '.-', **kwargs)
    color = ax.lines[-1].get_color()
    ax.set_ylabel('Current [$\mu$A]', color=color)

def VoverIplotter(ax=None, **kwargs):
    ''' Plot V/I vs V, like GPIB control program'''
    if ax is None:
        ax = ax2
    # Mask small currents, since V/I will blow up
    # There's definitely a better way.
    if len(d['I'] > 0):
        max_current = np.max(np.abs(d['I']))
        mask = np.abs(d['I']) > .01 * max_current
    else:
        mask = []
    V = d['V'][mask]

    if 'Vmeasured' in d:
        VoverI = d['Vmeasured'] / d['I']
    elif 'Imeasured' in d:
        VoverI = d['V'] / d['Imeasured']
    else:
        VoverI = d['V'] / d['I']

    VoverI = VoverI[mask]

    ax.plot(V, VoverI, '.-', **kwargs)
    color = ax.lines[-1].get_color()

    ax.set_yscale('log')
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('V/I [$\Omega$]', color=color)

def vcalcplotter(ax, R=8197, **kwargs):
    d['Vcalc'] = d['V'] - R * d['I']
    plotiv(d, ax=ax, x='Vcalc', **kwargs)
    ax.set_xlabel('V device (calculated assuming Rseries={}$\Omega$) [V]'.format(R))
def updateplots(**kwargs):
    ''' Draw the standard plots for whatever data is in the d variable'''
    # dunno, it errored without this line
    global plotters
    for ax, plotrs in plotters.items():
        # In case axis has more than one plotter
        if not hasattr(plottrs, '__getitem__'):
            plottrs = [plottrs]
        for plotter in plotrs:
            nlines = len(ax.lines)
            ncolors = len(linecolors[ax])
            color = linecolors[ax][(nlines - 1) % ncolors]
            plotter(ax=ax, **kwargs)

def liveplotter():
    # Keep updating until keithley says it's done
    # Then update one more time
    # Only way I found so far to see if there is still a measurement going on
    # Returns '2.00000e+00\n' when it is sweeping
    # Not quite what I want, I want it to stop when it captured the total number of samples
    global d
    sweeping = True
    firstiter = True
    lastiter = False
    while sweeping or lastiter:
        sweeping = bool(float(k.ask('print(status.operation.sweeping.condition)')))
        if not sweeping and not lastiter:
            lastiter = True
        else:
            lastiter = False
        # d = getdata(history=False)
        # Try getting just the new chunk of data and appending it
        arrays = [k for k in d.keys() if type(d[k]) == np.ndarray]
        chunkstart = len(d['V']) + 1
        dchunk = getdata(history=False, start=chunkstart)
        for ar in arrays:
            d[ar] = np.append(d[ar], dchunk[ar])

        # Assuming that the plotters only make one line
        # Will probably break horribly if they make more than one
        for ax, plotter in plotters.items():
            if firstiter:
                # Only happens on first iteration
                # Dummy line that will just get deleted
                ax.plot([])
            #color = ax.lines[-1].get_color()
            if not firstiter:
                # Delete the previous line
                del ax.lines[-1]
            nlines = len(ax.lines)
            ncolors = len(linecolors[ax])
            color = linecolors[ax][(nlines - 1) % ncolors]
            plotter(ax, color=color)
            ax.get_figure().canvas.draw()
        plt.pause(.1)
        firstiter = False
    d = getdata(history=True)

def make_figs():
    # Get monitor information so we can put the plots in the right spot.
    # Only works in windows ...
    # I'm using globals because I only want to write make_figs() and have these variables accessible
    global fig1, fig2, fig3, ax1, ax2, ax22, ax3
    user32 = ctypes.windll.user32
    wpixels, hpixels = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    #aspect = hpixels / wpixels
    dc = user32.GetDC(0)
    LOGPIXELSX = 88
    LOGPIXELSY = 90
    hdpi = ctypes.windll.gdi32.GetDeviceCaps(dc, LOGPIXELSX)
    vdpi = ctypes.windll.gdi32.GetDeviceCaps(dc, LOGPIXELSY)
    ctypes.windll.user32.ReleaseDC(0, dc)
    bordertop = 79
    borderleft = 7
    borderbottom = 28
    taskbar = 40
    #figwidth = wpixels * .3
    #figwidth = 500
    figheight = (hpixels - bordertop*2 - borderbottom*2 - taskbar) / 2
    figwidth = figheight * 1.3
    figsize = (figwidth / hdpi, figheight / vdpi)
    fig1loc = (wpixels - figwidth - 2*borderleft, 0)
    fig2loc = (wpixels - figwidth - 2*borderleft, figheight + bordertop + borderbottom)
    fig3loc = (wpixels - 2*figwidth - 4*borderleft, 0)
    try:
        # Close the figs if they already exist
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
    except:
        pass
    # Make some plot windows, put them in places
    fig1, ax1 = plt.subplots(figsize=figsize, dpi=hdpi)
    fig1.canvas.manager.window.move(*fig1loc)
    fig2, ax2 = plt.subplots(figsize=figsize, dpi=hdpi)
    fig2.canvas.manager.window.move(*fig2loc)
    ax22 = ax2.twinx()
    fig3, ax3 = plt.subplots(figsize=figsize, dpi=hdpi)
    fig3.canvas.manager.window.move(*fig3loc)

    # Stop axis labels from getting cut off
    fig1.set_tight_layout(True)
    fig2.set_tight_layout(True)
    fig3.set_tight_layout(True)

    plt.show()

# Make the figures and define which plotters are responsible for which axis
# Might be able to get away with multiple plotters per axis
make_figs()
# Maybe could do it like this.  Too ugly.  only want to have to change one variable
#plotters = {'iv':ivplotter, 'channelsv':vtplotter, 'channelsi':itplotter, 'fitline':Rfitplotter}
#axes = {'iv':ax1, 'channelsv':ax2, 'channelsi':ax3, 'fitline':ax1}
plotters = {ax1:ivplotter, ax2:vtplotter, ax22:itplotter, ax3:VoverIplotter}
defaultcolors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
linecolors = defaultdict(lambda: defaultcolors)
linecolors[ax22] = defaultcolors[::-1]

def clear_plots():
    # Clear IV loop plots
    for ax in plotters.keys():
        ax.cla()
    # Some default labels so you know what the plots are all about
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Current [$\mu$A]')
    ax2.set_ylabel('Voltage [V]')
    ax2.set_xlabel('Data Point #')
    ax22.set_ylabel('Current [$\mu$A]')
    ax3.set_ylabel('V/I [$\Omega$]')
    ax3.set_xlabel('Voltage [V]')

c = autocaller(clear_plots)

clear_plots()
