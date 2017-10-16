'''
This file should be run using the %run -i magic in ipython.
Provides a command based user interface for IV measurements.
make sure %matplotlib is in your ipython startup, or just make sure to run it first.


There are some inefficiencies in the data storage, which is fine for moderately sized datasets.
For very large datasets I will need to do some further optimization:
store ADC values instead of floats
Do not copy metadata to every single loop when it's all the same.  Maybe use a separate file.

Storing metadata in a separate file could be a better idea in general  -- then you can just load the metadata to find which loops you need, instead of loading ALL the information including data arrays when you aren't sure what it is.  Could even use a csv for this. Arrays could then be stored very efficiently especially if they are about the same length.
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
import pandas as pd
import warnings
import subprocess
from datetime import datetime
# Stop a certain matplotlib warning from showing up
warnings.filterwarnings("ignore",".*GUI is implemented.*")
from collections import defaultdict, deque

def makedatafolder():
    datasubfolder = os.path.join(datafolder, subfolder)
    if not os.path.isdir(datasubfolder):
        print('Making folder: {}'.format(datasubfolder))
        os.makedirs(datasubfolder)

datestr = time.strftime('%Y-%m-%d')
timestr = time.strftime('%Y-%m-%d_%H%M%S')
datafolder = r'D:\t\ivdata'
subfolder = datestr
print('Data to be saved in {}'.format(os.path.join(datafolder, subfolder)))
makedatafolder()

print('Overwrite \'datafolder\' and/or \'subfolder\' variables to change directory')

############# Logging  ###########################
def getGitRevision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

gitrev = getGitRevision()

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
        # This needs to be here otherwise there's no line break somewhere.  Don't worry about it.
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

# Rather than importing the modules and dealing with reload shenanigans that never actually work, use ipython run magic
magic('matplotlib')
ivtoolsdir = 'C:/t/py-ivtools'
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/measure.py')))
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/plot.py')))
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/io.py')))
magic('run -i {}'.format(os.path.join(ivtoolsdir, 'ivtools/analyze.py')))

# Because who wants to type?
class autocaller():
    def __init__(self, function):
        self.function = function
    def __repr__(self):
        self.function()
        return 'autocalled ' + self.function.__name__

#####

# Try to connect the instruments
connect_instruments()

print('Channel settings:')
print(pd.DataFrame([COUPLINGS, ATTENUATION, OFFSET, RANGE],
                   index=['COUPLINGS', 'ATTENUATION', 'OFFSET', 'RANGE']))

def smart_range(v1, v2, R=None, ch=['A', 'B']):
        # Auto offset for current input
        global OFFSET, RANGE
        possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
        # Sadly, each range has a maximum possible offset
        max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))

        if 'A' in ch:
            # Assuming CHA is directly sampling the output waveform, we can easily optimize the range
            arange, aoffs = best_range((v1, v2))
            RANGE['A'] = arange
            OFFSET['A'] = aoffs

        if 'B' in ch:
            # Smart ranging channel B is harder, since we don't know what kind of device is being measured.
            # Center the measurement range on zero current
            #OFFSET['B'] = -COMPLIANCE_CURRENT * 2e3
            # channelb should never go below zero, except for potentially op amp overshoot
            # I have seen it reach -0.1V
            if R is None:
                # Hypothetical resistance method
                # Signal should never go below 0V (compliance)
                b_min = 0
                b_resistance = max(abs(v1), abs(v2)) / COMPLIANCE_CURRENT / 1.1
                # Compliance current sets the voltage offset at zero input.
                # Add 10% to be safe.
                b_max = (COMPLIANCE_CURRENT - min(v1, v2) / b_resistance) * 2e3 * 1.1
            else:
                # R was passed, assume device has constant resistance with this value
                b_min = (COMPLIANCE_CURRENT - max(v1, v2) / R) * 2e3
                b_max = (COMPLIANCE_CURRENT - min(v1, v2) / R) * 2e3
            brange, boffs = best_range((b_min, b_max))
            RANGE['B'] = brange
            OFFSET['B'] = boffs
            # Could do some other cool tricks here
            # Like look at previous measurements, use derivatives to predict appropriate range changes

# TODO: auto smoothimate
def iv(wfm, duration=1e-3, n=1, fs=None, nsamples=None, smartrange=False,
       autosave=True, autoplot=True, autosplit=True, into50ohm=False,
       channels=['A', 'B'], autosmoothimate=True, splitbylevel=None):
    '''
    Pulse a triangle waveform, plot pico channels, IV, and save to d variable
    Provide either fs or nsamples
    '''
    global d, chdata

    # Channels that need to be sampled for measurement
    # channels = ['A', 'B', 'C']


    if not (bool(fs) ^ bool(nsamples)):
        raise Exception('Must pass either fs or nsamples, and not both')
    if fs is None:
        fs = nsamples / duration

    if smartrange:
        smart_range(np.min(wfm), np.max(wfm), ch=['A', 'B'])
    else:
        # Always smart range channel A
        smart_range(np.min(wfm), np.max(wfm), ch=['A'])

    # Set picoscope to capture
    # Sample frequencies have fixed values, so it's likely the exact one requested will not be used
    actual_fs = pico_capture(ch=channels,
                             freq=fs,
                             duration=n*duration)
    if into50ohm:
        # Multiply voltages by 2 to account for 50 ohm input
        wfm = 2 * wfm

    # Send a pulse
    pulse(wfm, duration, n=n)

    trainduration = n * duration
    print('Applying pulse(s) ({:.2e} seconds).'.format(trainduration))
    time.sleep(n * duration * 1.05)
    #ps.waitReady()
    print('Getting data from picoscope.')
    # Get the picoscope data
    # This goes into a global strictly for the purpose of plotting the (unsplit) waveforms.
    chdata = get_data(channels, raw=True)
    print('Got data from picoscope.')
    # Convert to IV data (keeps channel data)
    ivdata = pico_to_iv(chdata)

    if autosmoothimate:
        nsamples_shot = ivdata['nsamples_capture'] / n
        # Smooth by 0.3% of a shot
        window = max(int(nsamples_shot * 0.003), 1)
        # End up with about 1000 data points per shot
        # This will be bad if you send in a single shot waveform with multiple cycles
        # In that case, you shouldn't be using autosmoothimate or autosplit
        # TODO: make a separate function for IV trains?
        factor = max(int(nsamples_shot / 1000), 1)
        print('Smoothimating data with window {}, factor {}'.format(window, factor))
        ivdata = smoothimate(ivdata, window=window, factor=factor, columns=None)

    if autosplit:
        print('Splitting data into individual pulses')
        if n > 1 and (splitbylevel is None):
            nsamples = duration * actual_fs
            if 'downsampling' in ivdata:
                # Not exactly correct but I hope it's close enough
                nsamples /= ivdata['downsampling']
            ivdata = splitiv(ivdata, nsamples=nsamples)
        elif splitbylevel is not None:
            # splitbylevel can split loops even if they are not the same length
            # Could take more time though?
            # This is not a genius way to determine to split at + or - dV/dt
            increasing = bool(sign(argmax(wfm) - argmin(wfm)) + 1)
            ivdata = split_by_crossing(ivdata, V=splitbylevel, increasing=increasing, smallest=20)

    d = ivdata
    dhistory.append(ivdata)

    if autosave:
        # Write to disk automatically
        # Can slow things down
        print('Writing data to disk')
        savedata()

    if autoplot:
        # Plot the IV data
        # Can slow things down
        print('Plotting data')
        plotupdate()



#################### For interactive collection of IV loops ########################

# We construct a data set in a list of dictionaries
# The list of dictionaries can then be immediately converted to dataframe for storage and analysis

# We append metadata to the IV data -- devicemeta + staticmeta + ivarrays

'''
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
'''

# Data you forgot to save (only 10 of them)
try:
    dhistory
except:
    print('Defining dhistory = deque(maxlen=10)')
    dhistory = deque(maxlen=10)
else:
    if len(dhistory) > 0:
        answer = input('\'dhistory\' variable not empty.  Clobber it? ')
        if answer.lower() == 'y':
            print('Defining dhistory = deque(maxlen=10)')
            dhistory = deque(maxlen=10)

# The data index you are currently on
meta_i = None
d = None

# Add items to this and they will be appended as metadata to all subsequent measurements
staticmeta = {'gitrev':gitrev, 'scriptruntime':timestr}

'''
If you want to measure more than one device, this code provides a nice way to step between the
device information without having to type it in each time.  Less error prone, many advantages.
'''

# Metadata about the device currently being probed.  Controlled by a few of the following functions
devicemeta = {}

# Change this list of dicts (or dataframe) before starting
devicemetalist = [{'device_number':n} for n in range(100)]

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
    global lassen_df, meta_df, prettykeys, filenamekeys, devicemetalist
    # Load information from files on disk
    deposition_df = pd.read_excel('CeRAM_Depositions.xlsx', header=8, skiprows=[9])
    # Only use Lassen devices
    deposition_df = deposition_df[deposition_df['wafer_code'] == 'Lassen']
    lassen_df = pd.read_pickle(r"all_lassen_device_info.pickle")
    # Merge data
    merge_deposition_data_on = ['coupon']
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
    prettykeys = ['dep_code', 'coupon', 'die', 'module', 'device', 'width_nm', 'R_series', 'layer_1', 'thickness_1']
    filenamekeys = ['dep_code', 'sample_number', 'module', 'device']
    print('Loaded {} devices into devicemetalist'.format(len(devicemetalist)))

def prettyprint_meta(hlkeys=None):
    # Print some information about the device
    global prettykeys
    if prettykeys is None or len(prettykeys) == 0:
        # Print all the information
        prettykeys = devicemeta.keys()
    for key in prettykeys:
        if key in devicemeta.keys():
            if hlkeys is not None and key in hlkeys:
                print('{:<18}\t{:<8} <----- Changed'.format(key[:18], devicemeta[key]))
            else:
                print('{:<18}\t{}'.format(key[:18], devicemeta[key]))

pp = autocaller(prettyprint_meta)

def nextdevice():
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
    print('You have selected this device (index {} of devicemetalist):'.format(meta_i))
    # Print some information about the device
    prettyprint_meta(hlkeys)

n = autocaller(nextdevice)

def previousdevice():
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

p = autocaller(previousdevice)

### Functions that write to disk


def savedata(data=None, filename=None, keepchannels=False):
    '''
    Attach sample information and write the current value of the d variable to disk.
    d variable can contain a single iv dict/series, or a list of them
    This could get ugly..
    '''
    if data is None:
        global d
        data = d
    islist = type(data) == list
    # Append all the data together
    # devicemeta might be a dict or a series
    print('Appending metadata to last iv measured:')
    prettyprint_meta()
    print('...')
    if islist:
        for l in data:
            l.update(dict(devicemeta))
            l.update(staticmeta)
    else:
        data.update(dict(devicemeta))
        data.update(staticmeta)
    # Write series/dataframe to disk.  Append the path to metadata
    if filename is None:
        #filename = time.strftime('%Y-%m-%d_%H%M%S')
        #Need milliseconds
        filename = datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]
        for fnkey in filenamekeys:
            if not islist and fnkey in data.keys():
                filename += '_{}'.format(data[fnkey])
            elif islist and fnkey in data[0].keys():
                # Use metadata from first dict.  Should all be same
                filename += '_{}'.format(data[0][fnkey])
        if islist: filename += '.df'
        # s for series
        else: filename += '.s'
    filepath = os.path.join(datafolder, subfolder, filename)
    makedatafolder()
    if islist:
        for l in data:
            l['datafilepath'] = filepath
        print('Converting data to pd.DataFrame for storage.')
        df = pd.DataFrame(data)
        if not keepchannels:
            print('Removing channel data to save space.'.format(filepath))
            todrop = set(('A', 'B', 'C', 'D')) & set(df.columns)
            df.drop(todrop, 1, inplace=True)
        print('Writing data as pickle to {}'.format(filepath))
        df.to_pickle(filepath)
        # Extend data list with d
        #print('Extending list \'data\' with variable \'d\'')
        #data.extend(d)
    else:
        data['datafilepath'] = filepath
        print('Converting variable \'d\' to pd.Series for storage.')
        s = pd.Series(data)
        if not keepchannels:
            todrop = set(('A', 'B', 'C', 'D')) & set(s.index)
            s.drop(todrop, inplace=True)
        print('Writing data as pickle to {}'.format(filepath))
        s.to_pickle(filepath)
        ### Decided this is a bad idea for the large datasets that tend to come out of this
        # Append series to data list
        #print('Appending variable \'d\' to list \'data\'')
        #data.append(d)
    print('File size: {:.3f} MB'.format(os.path.getsize(filepath) / 1048576))

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
    df.loc[:, ~df.columns.isin(['I', 'V', 't', 'Vmeasured', 'A', 'B'])].to_excel(xlspath)

##### Interactive Plotting

def plotupdate():
    # Plot the data in the data variable
    global d
    for ax, plotter in plotters.items():
        plotter(d, ax=ax)
        ax.get_figure().canvas.draw()
    plt.pause(.05)

# These functions define what gets plotted automatically on the interactive figures
# We are potentially dealing with a lot of data points such that plotting would limit the measurement speed
# Try to make some reasonable decisions about which subset of the data to display

def ax1plotter(data, ax=None, maxloops=100, smooth=True):
    # Smooth data a bit and give it to plotiv (from plot.py)
    # Would be better to smooth before splitting ...
    if ax is None:
        fig, ax = plt.subplots()
    if smooth:
        data = moving_avg(data, window=10)
    if type(data) is list:
        nloops = len(data)
    else:
        nloops = 1
    if nloops > maxloops:
        print('You captured {} loops.  Only plotting {} loops'.format(nloops, maxloops))
        loopstep = int(nloops / 99)
        data = data[::loopstep]
    plotiv(data, ax=ax, maxsamples=5000)

def ax1plotter_2(data, ax=None, maxloops=100, smooth=True):
    # Smooth data a bit and give it to plotiv (from plot.py)
    # Would be better to smooth before splitting ...
    if ax is None:
        fig, ax = plt.subplots()
    if smooth:
        data = moving_avg(data, window=10, columns=None)
    if type(data) is list:
        nloops = len(data)
    else:
        nloops = 1
    if nloops > maxloops:
        print('You captured {} loops.  Only plotting {} loops'.format(nloops, maxloops))
        loopstep = int(nloops / 99)
        data = data[::loopstep]
    plotiv(data, y='I2', ax=ax, maxsamples=5000)

def ax2plotter(data, ax=None):
    # data might contain multiple loops because of splitting, but we want the unsplit arrays
    # To avoid pasting them back together again, there is a global variable called chdata
    if ax is None:
        fig, ax = plt.subplots()
    # Remove previous lines
    for l in ax2.lines[::-1]: l.remove()
    # Plot at most 100000 datapoints of the waveform
    for ch in ['A', 'B', 'C', 'D']:
        if ch in chdata:
            lendata = len(chdata[ch])
            break
    if lendata > 100000:
        print('Captured waveform has {} pts.  Plotting channel data for only the first 100,000 pts.'.format(lendata))
        plotdata = sliceiv(chdata, stop=100000)
    else:
        plotdata = chdata
    plot_channels(plotdata, ax=ax)


def VoverIplotter(data, ax=None, **kwargs):
    ''' Plot V/I vs V, like GPIB control program'''
    if ax is None:
        fig, ax = plt.subplots()
    mask = np.abs(d['V']) > .01
    vmasked = d['V'][mask]
    imasked = d['I'][mask]
    ax.plot(vmasked, vmasked / imasked, '.-', **kwargs)
    ax.set_yscale('log')
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('V/I [$\Omega$]')
    ax.yaxis.set_major_formatter(metricprefixformatter)

def dVdIplotter(data, ax=None, **kwargs):
    ''' Plot dV/dI vs V'''
    if ax is None:
        fig, ax = plt.subplots()
    mask = np.abs(d['V']) > .01
    vmasked = d['V'][mask]
    imasked = d['I'][mask]
    dv = np.diff(vmasked)
    di = np.diff(imasked)
    ax.plot(vmasked[1:], dv/di, '.-', **kwargs)
    ax.set_yscale('log')
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('V/I [$\Omega$]')
    ax.yaxis.set_major_formatter(metricprefixformatter)


## Make some plot windows, put them in places
try:
    # Close the figs if they already exist
    plt.close(fig1)
    plt.close(fig2)
except:
    pass

def make_figs():
    global fig1, ax1, fig2, ax2
    (fig1, ax1) , (fig2, ax2) = interactive_figures()

make_figs()

plotters = {ax1:ax1plotter, ax2:ax2plotter}

def clear_plots():
    # Clear IV loop plots
    for ax in plotters:
        ax.cla()
    ax1.set_title('IV Measurements')
    ax1.set_xlabel('Voltage')
    ax1.set_ylabel('Current')
    ax2.set_title('Picoscope Traces')
    ax2.set_xlabel('Data point')
    ax2.set_ylabel('Voltage [V]')

clear_plots()
plt.show()
c = autocaller(clear_plots)
