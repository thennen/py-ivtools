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

import visa
import pandas as pd
from matplotlib import pyplot as plt
import ctypes
from collections import deque
import warnings
# Stop a certain warning from showing up
warnings.filterwarnings("ignore",".*GUI is implemented.*")
import time
from shutil import copyfile
import os

# TODO: Copy this script over to the data directory
datestr = time.strftime('%Y-%m-%d')
timestr = time.strftime('%Y-%m-%d_%H%M%S')

scriptpath = os.path.realpath(__file__)
scriptdir, scriptfile = os.path.split(scriptpath)
scriptcopydir = os.path.join(scriptdir, 'Script_copies')
scriptcopyfn = timestr + scriptfile
scriptcopyfp = os.path.join(scriptcopydir, scriptcopyfn)
if not os.path.isdir(scriptcopydir):
    os.makedirs(scriptcopydir)
copyfile(scriptpath, scriptcopyfp)

datafolder = r'C:\t\data'
subfolder = datestr
print('Data to be saved in {}'.format(os.path.join(datafolder, subfolder)))
print('Overwrite \'datafolder\' and/or \'subfolder\' variables to change directory')


############# Keithley 2600 functions ###############

# Connect to Keithley
Keithley_ip = '10.10.50.10'
Keithley_id = 'TCPIP::' + Keithley_ip + '::inst0::INSTR'
rm = visa.ResourceManager()
try:
    k
except:
    k = rm.get_instrument(Keithley_id)
print('Keithley *IDN?')
idn = k.ask('*IDN?')
print(idn)


# This is the code that runs on the keithley's lua interpreter.
Keithley_func = '''
                loadandrunscript
                function SweepVList(sweepList, rangeI, limitI, nplc, delay)
                    reset()

                    -- Configure the SMU
                    smua.reset()
                    smua.source.func			= smua.OUTPUT_DCVOLTS
                    smua.source.limiti			= limitI
                    smua.measure.nplc			= nplc
                    --smua.measure.delay		= smua.DELAY_AUTO
                    smua.measure.delay = delay
                    smua.measure.rangei = rangeI

                    -- Prepare the Reading Buffers
                    smua.nvbuffer1.clear()
                    smua.nvbuffer1.collecttimestamps	= 1
                    smua.nvbuffer2.clear()
                    smua.nvbuffer2.collecttimestamps	= 1

                    -- Configure SMU Trigger Model for Sweep
                    smua.trigger.source.listv(sweepList)
                    smua.trigger.source.limiti			= limitI
                    smua.trigger.measure.action			= smua.ENABLE
                    smua.trigger.measure.iv(smua.nvbuffer1, smua.nvbuffer2)
                    smua.trigger.endpulse.action		= smua.SOURCE_HOLD
                    -- By setting the endsweep action to SOURCE_IDLE, the output will return
                    -- to the bias level at the end of the sweep.
                    smua.trigger.endsweep.action		= smua.SOURCE_IDLE
                    numPoints = table.getn(sweepList)
                    smua.trigger.count					= numPoints
                    smua.trigger.source.action			= smua.ENABLE
                    -- Ready to begin the test

                    smua.source.output					= smua.OUTPUT_ON
                    -- Start the trigger model execution
                    smua.trigger.initiate()
                end
                endscript
                '''
# Load the script into keithley
for line in Keithley_func.split('\n'):
    k.write(line)

def iv(vlist, Irange, Ilimit, nplc=1, delay='smua.DELAY_AUTO'):
    '''Wraps the SweepVList lua function defined on keithley''' 

    # Hack to shove more data points into keithley despite its questionable design.
    # still sending a string, but need to break it into several lines due to small input buffer
    # Define an anonymous script that defines the array variable
    l = len(vlist)
    k.write('loadandrunscript')
    k.write('sweeplist = {')
    for i in range(0, l, 100):
        chunk = ','.join(['{:.3f}'.format(v) for v in vlist[i:i+100]])
        k.write(chunk)
        k.write(',')
    k.write('}')
    k.write('endscript')

    # Call SweepVList
    # TODO: make sure the inputs are valid
    # Built in version -- locks communication so you can't get the incomplete array
    #k.write('SweepVListMeasureI(smua, sweeplist, {}, {})'.format(.1, l))
    print('SweepVList(sweeplist, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay))
    k.write('SweepVList(sweeplist, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay))
    liveplotter()
    # liveplotter does this already
    #d = getdata()

def getdata(history=true):
    # get keithley data arrays as strings, and convert them to arrays..
    # return dataarrays, metadict
    global dhistory
    metadict = {}
    numpts = int(float(k.ask('print(smua.nvbuffer1.n)')))
    if numpts > 0:
        readingstr = k.ask('printbuffer(1, {}, smua.nvbuffer1.readings)'.format(numpts))
        sourcevalstr = k.ask('printbuffer(1, {}, smua.nvbuffer1.sourcevalues)'.format(numpts))
        timestampstr = k.ask('printbuffer(1, {}, smua.nvbuffer1.timestamps)'.format(numpts))
        # Dataframe version.  Let's keep it simple instead.
        #out = pd.DataFrame({'t':np.float16(readingstr.split(', ')),
        #                    'V':np.float32(sourcevalstr.split(', ')),
        #                    'I':np.float32(readingstr.split(', '))})
        #out = out[out['t'] != inf]
        # Collect the measurement conditions
        # TODO: see what other information is available
        #metadict['Irange'] =  float(k.ask('print(smua.nvbuffer1.measureranges[1])'))
        #return out, metadict

        # Dict version.  Downside is you can't use dot notation anymore..
        out = {'t':np.float16(readingstr.split(', ')),
               'V':np.float32(sourcevalstr.split(', ')),
               'I':np.float32(readingstr.split(', '))}
        out['Irange'] =  float(k.ask('print(smua.nvbuffer1.measureranges[1])'))
    else:
        #return pd.DataFrame({'t':[], 'V':[], 'I':[]}), {}
        empty = np.array([])
        out = dict(t=empty, V=empty, I=empty)

    if history:
        dhistory.append(out)
    return out

def triangle(v1, v2, n=None, step=None):
    # We like triangle sweeps a lot
    # Very basic triangle pulse with some problems.
    if n is not None:
        dv = abs(v1) + abs(v2 - v1) + abs(v2)
        step = dv / n
    wfm = np.concatenate((np.arange(0, v1, np.sign(v1) * step),
                        np.arange(v1, v2, np.sign(v2 - v1) * step),
                        np.arange(v2, 0, -np.sign(v2) * step),
                        [0]))
    #if len(wfm) > 100:
        #print('Too many steps for my idiot program.  Interpolating to 100 pts')
        #wfm = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(wfm)), wfm)
    return wfm

def waitforkeithley():
    # I can't find out how to do this from the manual.
    # Shouldn't be necessary if you don't run any blocking scripts on keithley
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
data = []

# Data you forgot to save (only 10 of them)
dhistory = deque(maxlen=10)

# The data index you are currently on
meta_i = -1

# Add keys to this and they will be appended as metadata to all subsequent measurements
staticmeta = {'keithley':idn, 'script':__file__, 'scriptruntime':timestr}

# Metadata about the device currently being probed.  Controlled by a few of the following functions
devicemeta = {}

# Change this list of dicts (or dataframe) before starting
devicemetalist = {'device_number':n for n in range(100)}

# This controls which keys are printed when identifying a device
prettykeys = None

# Example of setting meta list
wafer_df = pd.read_pickle(r"X:\emrl\Pool\Projekte\HGST-CERAM\Lassen_Test_Vehicle\all_lassen_device_info.pickle")
# Select the samples you want to measure
meta_df = wafer_df
coupons = [2, 3]
meta_df = meta_df[meta_df.coupon.isin(coupons)]
modules = ['001G', '001E']
meta_df = meta_df[meta_df.module.isin(modules)]
# Merge with deposition data
deposition_df = pd.read_excel('X:\emrl\Pool\Projekte\HGST-CERAM\CeRAM_Depositions.xlsx', header=8, skiprows=[9])
merge_deposition_data_on = ['coupon']
meta_df = pd.merge(meta_df, deposition_df, how='left', on=merge_deposition_data_on)
devicemetalist = meta_df
prettykeys = ['coupon', 'die', 'module', 'device', 'width_nm', 'R_series', 'layer_1', 'thickness_1']

# Because who wants to type?
class autocaller():
    def __init__(self, function):
        self.function = function
    def __repr__(self):
        self.function()
        return 'autocalled ' + self.function.__name__

def prettyprint_meta():
    # Print some information about the device
    if prettykeys is None:
        for k, v in devicemeta.items():
            print('{:<12}\t\t{}'.format(k[:12], v))
    else:
        for key in prettykeys:
            if key in devicemeta.keys():
                print('{:<12}\t\t{}'.format(key[:12], devicemeta[key]))

pp = autocaller(prettyprint_meta)

def nextsample():
    ''' Go to the next device '''
    global meta_i, devicemeta
    meta_i += 1
    if type(devicemetalist) == pd.DataFrame:
        devicemeta = devicemetalist.iloc[meta_i]
    else:
        devicemeta = devicemetalist[meta_i]
    print('You are now measuring this device (index {} of devicemetalist):'.format(meta_i))
    # Print some information about the device
    prettyprint_meta()

n = autocaller(nextsample)

def previoussample():
    ''' Go to the previous device '''
    global meta_i, devicemeta
    meta_i -= 1
    if type(devicemetalist) == pd.DataFrame:
        devicemeta = devicemetalist.iloc[meta_i]
    else:
        devicemeta = devicemetalist[meta_i]
    print('You are now measuring this device (index {} of devicemetalist):'.format(meta_i))
    # Print some information about the device
    prettyprint_meta()

p = autocaller(previoussample)

### Functions that write to disk

def savedata(filename=None):
    global d
    ''' Write the current value of the d variable to places, attach metadata '''
    # Append all the data together
    # devicemeta might be a dict or a series
    print('Appending metadata to last iv loop measured:')
    prettyprint_meta()
    print('...')
    d.update(dict(devicemeta))
    d.update(staticmeta)
    # Write series to disk.  Append the path to metadata
    if filename is None:
        filename = time.strftime('%Y-%m-%d_%H%M%S') + '_loop.s'
    datasubfolder = os.path.join(datafolder, subfolder)
    filepath = os.path.join(datafolder, subfolder, filename)
    if not os.path.isdir(datasubfolder):
        os.makedirs(datasubfolder)
    print('converting variable \'d\' to pd.Series and writing as pickle to {}'.format(filepath))
    pd.Series(d).to_pickle(filepath)
    # Append series to data list
    print('Appending variable \'d\' to list \'data\'')
    data.append(d)

s = autocaller(savedata)

def savedatalist(filename=None):
    if filename is None:
        # TODO: Figure out a filename that doesn't collide
        filename = '{}_keithley_loops.df'.format(datestr)
    filepath = os.path.join(datafolder, subfolder, filename)
    print('Writing {}'.format(filepath))
    pd.DataFrame(data).to_pickle(filepath)

### Functions that actually tell Keithley to do something

def autoresistance():
    ''' Do some measurement '''
    # Keep increasing voltage amplitude and feed back on the resistance
    set_compliance(ic)
    sweeplist(triangle(2, -1, 80))
    waitforkeithley()
    plot(d['V'], d['I'], '.-')
    # Do a live plot while doing this
    # Only keep the last one, or keep the others somewhere?

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


def plotter1(ax=None, **kwargs):
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
    global R
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
    fiti = polyval(line, fitv)
    plotkwargs = dict(color='black', alpha=.3, linestyle='--')
    plotkwargs.update(kwargs)
    ax.plot(fitv, 1e6 * fiti, **plotkwargs)
    return R

def complianceplotter(ax=ax1, **kwargs):
    # Plot a dotted line indicating compliance current
    pass

def plotter2(ax=None, **kwargs):
    if ax is None:
        ax = ax2
    ''' This defines what gets plotted on ax2'''
    ax.plot(d['V'], '.-', **kwargs)
    color = ax.lines[-1].get_color()
    ax.set_ylabel('Voltage [V]', color=color)
    ax.set_xlabel('Data Point #')

def plotter3(ax=None, **kwargs):
    if ax is None:
        ax = ax3
    ax.plot(d['I'], '.-', **kwargs)
    color = ax.lines[-1].get_color()
    ax.set_ylabel('Current [$\mu$A]', color=color)

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
        d = getdata(history=False)
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
    global fig1, fig2, fig3, ax1, ax2, ax3
    user32 = ctypes.windll.user32
    wpixels, hpixels = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
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
    figwidth = wpixels * .3
    figheight = (hpixels - bordertop*2 - borderbottom*2 - taskbar) / 2
    figsize = (figwidth / hdpi, figheight / vdpi)
    fig1loc = (wpixels - figwidth - 2*borderleft, 0)
    fig2loc = (wpixels - figwidth - 2*borderleft, figheight + bordertop + borderbottom)
    try:
        # Close the figs if they already exist
        plt.close(fig1)
        plt.close(fig2)
    except:
        pass
    # Make some plot windows, put them in places
    fig1, ax1 = plt.subplots(figsize=figsize, dpi=hdpi)
    fig1.canvas.manager.window.move(*fig1loc)
    fig2, ax2 = plt.subplots(figsize=figsize, dpi=hdpi)
    fig2.canvas.manager.window.move(*fig2loc)
    fig3 = fig2
    ax3 = ax2.twinx()

    # Some default labels so you know what the plots are all about

    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Current [$\mu$A]')
    ax2.set_ylabel('Voltage [V]')
    ax2.set_xlabel('Data Point #')
    ax3.set_ylabel('Current [$\mu$A]')
    # Stop axis labels from getting cut off
    fig1.set_tight_layout(True)
    fig2.set_tight_layout(True)

    plt.show()

    #return ((fig1, ax1), (fig2, ax2), (fig3, ax3))

def clear_plots():
    # Clear IV loop plots
    for ax in plotters.keys():
        ax.cla()

# Make the figures and define which plotters are responsible for which axis
# Might be able to get away with multiple plotters per axis
make_figs()
# Maybe could do it like this.  Too ugly.
#plotters = {'iv':plotter1, 'channelsv':plotter2, 'channelsi':plotter3, 'fitline':Rfitplotter}
#axes = {'iv':ax1, 'channelsv':ax2, 'channelsi':ax3, 'fitline':ax1}
plotters = {ax1:plotter1, ax2:plotter2, ax3:plotter3}
defaultcolors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
linecolors = {ax1:defaultcolors, ax2:defaultcolors, ax3:defaultcolors[::-1]}
