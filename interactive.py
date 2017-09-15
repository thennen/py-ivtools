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

def smartrange(v1, v2, R=None):
        # Auto offset for current input
        global OFFSET, RANGE
        possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
        # Sadly, each range has a maximum possible offset
        max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))

        # Assuming CHA is directly sampling the output waveform, we can easily optimize the range
        amp = abs(v1 - v2) / 2
        selectedrange = possible_ranges[possible_ranges >= amp][0]
        RANGE['A'] = selectedrange
        middle = (v1 + v2) / 2
        OFFSET['A'] = -middle

        # Smart ranging channel B is harder, since we don't know what kind of device is being measured.
        # Center the measurement range on zero current
        #OFFSET['B'] = -COMPLIANCE_CURRENT * 2e3
        # channelb should never go below zero, except for potentially op amp overshoot
        # I have seen it reach -0.1V
        if R is None:
            # Hypothetical resistance method
            b_min = 0
            b_resistance = max(abs(v1), abs(v2)) / COMPLIANCE_CURRENT / 1.1
            b_max = (COMPLIANCE_CURRENT - min(v1, v2) / b_resistance) * 2e3
        else:
            # R was passed, assume device has constant resistance with this value
            b_min = (COMPLIANCE_CURRENT - max(v1, v2) / R) * 2e3
            b_max = (COMPLIANCE_CURRENT - min(v1, v2) / R) * 2e3
        b_amp = abs(b_max - b_min) / 2
        b_middle = (b_max + b_min) / 2
        mask_of_possibilities = possible_ranges >= b_amp
        for b_selectedrange, max_offset in zip(possible_ranges[mask_of_possibilities], max_offsets[mask_of_possibilities]):
            # Is b_middle an acceptable offset?
            if b_middle < max_offset:
                RANGE['B'] = b_selectedrange
                OFFSET['B'] = -b_middle
                break
            # Can we reduce the offset without the signal going out of range?
            elif (max_offset + b_selectedrange >= b_max) and (-max_offset - b_selectedrange <= b_min):
                RANGE['B'] = b_selectedrange
                OFFSET['B'] = np.clip(-b_middle, -max_offset, max_offset)
                break
            # Neither worked, try increasing the range ...

        # Could do some other cool tricks here
        # Like look at previous measurements, use derivatives to predict appropriate range changes

def iv(v1, v2, duration=None, rate=None, n=1, fs=1e7, dumbrange=True,
       autosave=True, autoplot=True, autosplit=True):
    '''
    Pulse a triangle waveform, plot pico channels, IV, and save to data variable
    '''
    global d, chdata

    # Channels that need to be sampled for measurement
    # channels = ['A', 'B', 'C']
    channels = ['A', 'B']

    # Need to know duration of pulse if only sweeprate is given
    # so that we know how long to capture
    sweeprate, pulsedur = _rate_duration(v1, v2, rate, duration)

    if not dumbrange:
        smartrange(v1, v2)

    # Set picoscope to capture
    actual_fs = pico_capture(ch=channels,
                             freq=fs,
                             duration=n*pulsedur)
    # Send a triangle pulse
    tripulse(n=n,
             v1=v1,
             v2=v2,
             duration=duration,
             rate=rate)

    pulseduration = n * duration
    print('Applying pulse ({:.2e} seconds).'.format(pulseduration))
    time.sleep(n * duration)
    print('Getting data from picoscope.')
    # Get the picoscope data
    # This goes into a global strictly for the purpose of plotting the (unsplit) waveforms.
    chdata = get_data(channels, raw=True)
    print('Got data from picoscope.')
    # Convert to IV data (keeps channel data)
    ivdata = pico_to_iv(chdata)

    if autosplit and n > 1:
        print('Splitting data into individual pulses')
        ivdata = splitiv(ivdata, nsamples=pulsedur*actual_fs)

    d = ivdata
    dhistory.append(ivdata)

    if autosave:
        # Write to disk automatically
        # Can slow things down
        print('Writing data to disk')
        savedata()

    if autoplot:
        # Plot the IV data
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
staticmeta = {'script':__file__, 'scriptruntime':timestr}

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
def load_lassen(coupons=[23], dies=[64], modules=['001H']):
    # Could of course specify devices by any other criteria (code name, deposition date, thickness ...)
    global wafer_df, meta_df, prettykeys, filenamekeys, devicemetalist
    wafer_df = pd.read_pickle(r"all_lassen_device_info.pickle")
    # Select the samples you want to measure
    meta_df = wafer_df
    #### Filter devices to be measured #####
    devices001 = [2,3,4,5,6,7,8]
    devices014 = [4,5,6,7,8,9]
    #########
    meta_df = meta_df[meta_df.coupon.isin(coupons)]
    meta_df = meta_df[meta_df.module.isin(modules)]
    meta_df = meta_df[~((meta_df.module_num == 1) & ~meta_df.device.isin(devices001))]
    meta_df = meta_df[~((meta_df.module_num == 14) & ~meta_df.device.isin(devices014))]
    meta_df = meta_df[meta_df.die.isin(dies)]
    # Merge with deposition data
    deposition_df = pd.read_excel('CeRAM_Depositions.xlsx', header=8, skiprows=[9])
    merge_deposition_data_on = ['coupon']
    meta_df = pd.merge(meta_df, deposition_df, how='left', on=merge_deposition_data_on)
    meta_df = meta_df.sort_values(by=['coupon', 'module', 'device'])
    devicemetalist = meta_df
    prettykeys = ['deposition_code', 'coupon', 'die', 'module', 'device', 'width_nm', 'R_series', 'layer_1', 'thickness_1']
    filenamekeys = ['deposition_code', 'sample_number', 'module', 'device']

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


def savedata(filename=None):
    '''
    Attach sample information and write the current value of the d variable to disk.
    d variable can contain a single iv dict/series, or a list of them
    This could get ugly..
    '''
    global d
    islist = type(d) == list
    # Append all the data together
    # devicemeta might be a dict or a series
    print('Appending metadata to last iv measured:')
    prettyprint_meta()
    print('...')
    if islist:
        for l in d:
            l.update(dict(devicemeta))
            l.update(staticmeta)
    else:
        d.update(dict(devicemeta))
        d.update(staticmeta)
    # Write series/dataframe to disk.  Append the path to metadata
    if filename is None:
        filename = time.strftime('%Y-%m-%d_%H%M%S')
        for fnkey in filenamekeys:
            if not islist and fnkey in d.keys():
                filename += '_{}'.format(d[fnkey])
            elif islist and fnkey in d[0].keys():
                # Use metadata from first dict.  Should all be same
                filename += '_{}'.format(d[0][fnkey])
        if islist: filename += '.df'
        # s for series
        else: filename += '.s'
    filepath = os.path.join(datafolder, subfolder, filename)
    makedatafolder()
    if islist:
        for l in d:
            l['datafilepath'] = filepath
        print('converting variable \'d\' to pd.DataFrame and writing as pickle to {}'.format(filepath))
        pd.DataFrame(d).to_pickle(filepath)
        # Extend data list with d
        #print('Extending list \'data\' with variable \'d\'')
        #data.extend(d)
    else:
        d['datafilepath'] = filepath
        print('converting variable \'d\' to pd.Series and writing as pickle to {}'.format(filepath))
        pd.Series(d).to_pickle(filepath)
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

def clear_plots():
    # Clear IV loop plots
    ax1.cla()
    ax2.cla()
    ax1.set_title('IV Measurements')
    ax1.set_xlabel('Voltage')
    ax1.set_ylabel('Current')
    ax2.set_title('Picoscope Traces')
    ax2.set_xlabel('Data point')
    ax2.set_ylabel('Voltage [V]')

clear_plots()
plt.show()
c = autocaller(clear_plots)

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

def ax1plotter(data, ax=ax1):
    # Smooth data a bit and give it to plotiv (from plot.py)
    # Would be better to smooth before splitting ...
    smoothdata = moving_avg(data, window=5)
    if type(data) is list:
        nloops = len(data)
    else:
        nloops = 1
    maxloops = 100
    if nloops > maxloops:
        print('You captured {} loops.  Only plotting {} loops'.format(nloops, maxloops))
        loopstep = int(nloops / 99)
        smoothdata = smoothdata[::loopstep]
    plotiv(smoothdata, ax=ax, maxsamples=5000)

def ax2plotter(data, ax=ax2):
    # Remove previous lines
    for l in ax2.lines[::-1]: l.remove()
    # Plot at most 100000 datapoints of the waveform
    # Hoping data has channel A, or you will have to fix this later
    lendata = len(chdata['A'])
    if lendata > 100000:
        print('Captured waveform has {} pts.  Plotting channel data for only the first 100,000 pts.'.format(lendata))
        plotdata = sliceiv(chdata, stop=100000)
    else:
        plotdata = chdata
    plot_channels(plotdata, ax=ax)

plotters = {ax1:ax1plotter, ax2:ax2plotter}
