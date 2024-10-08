'''
Hans's NVMPyMeas program is not compatible with Sierra West layout.
But his measurement code is independent of the positioning
Looking into saving time programming TEO and Keysight B2985A by using what Hans already has done
This will end up being a frankenstein script that uses my data processing functions with some of Hans's measurement code
TH 2019-08-21
'''

import pandas as pd
import numpy as np
import sys
import copy
import os
import socket
import getpass
import time
sys.path.append("C:\\py-ivtools\\") 
from ivtools.instruments import UF2000Prober
import ivtools.io as io

datadir = r'D:\Data\Hennen'
hostname = socket.gethostname()
username = getpass.getuser()
datestr = time.strftime('%Y-%m-%d')
gitrev = io.getGitRevision()

#############################
# "Leakage" and "threshold" testing
#############################

# all the device locations with respect to the "home" device, which is mod 002, device 1, die_rel 3 (bottom left)
coupondf = pd.read_pickle('C:\\py-ivtools\\ivtools\\sampledata\\lassen_coupon_info.pkl').set_index(['die_rel', 'module', 'device'])

# Go to home position and get the position in microns
# This changes every time you load the wafer
print('Getting UF2000Prober connection')
p = UF2000Prober()
print('Moving to home position')
p.goHome()
x_home, y_home = p.getPosition_um()

def gotoDevice(die_rel=1, module='001', device=2):
    # find location of this device relative to home device
    wX, wY = coupondf.loc[(die_rel, module, device)][['wX', 'wY']]
    p.moveAbsolute_um(x_home + wX, y_home + wY)

#############################
# "Leakage" and "threshold" testing
#############################

'''
Some notes about Hans code:
Not structured like anything you have ever seen -- more like VB clumsily translated to python

There are Measurement modules in the directory NVMPyMeas/Measurements
These contain classes which need to be initialized using a nested dictionary that holds the measurement parameters, and lots of other information mostly contained in config files
The classes can also be initialized in a with block, in which case it will write to some log file somewhere
Initialization will make the instrument connections (?), then you can use a method that runs the measurement
name of the measurement method depends on class name -- seems to be 'Measure' + class.__name__
After you run the method (which returns nothing), the class instance attributes get updated with results.
Apparently the class tries to analyze the data as well, and returns some results of the analysis.  this is dumb.
You can run the measurement method again and the results get overwritten
If you try to find out details of how this code runs, you will be dragged through a massive web of unnecessary complexity
'''

sys.path.append("C:\\NVMPyMeas\\Utilities")
sys.path.append("C:\\NVMPyMeas\\Measurements")   
from MeasurementOTS import Leakage, VThreshold
import UtilsCollection

#### Ridiculous parameter/state dictionary that gets passed to every measurement class __init__
#### descriptions from excel file are transcribed here as comments
###  God knows what these do -- I think reads a bunch of config files, might still work if omitted?
paX = UtilsCollection.GetAuxInfoPlus()
paX["General"] = UtilsCollection.GetGeneralInfo()

def measureLeakage(RecID = '',        # Determines the filename of the scope traces.  if blank, D:/Test/test.bin
                   Pol = 1,           # Polarity
                   Stop_V = 0.75,
                   nPts = 10,         # limited to 4 <= nPts <= 50 for some reason
                   Bidirectional = 0, # Can sweep up and down
                   Mode = 1,          # Auto(scale): -1, Fix: 0, Limited: 1 (best) (Hans said Limited means autorange except it doesn't go below 1 pA)
                   Range = -9,        # in concert with Mode = -1: limits max current range. -9 is best
                   ACQDly = 0.2,      # Acquisition delay for Keysight. 0.1-0.2s is good
                   KSstairs = 1       # Keysight stairs are the fastest, so use that
                   ):
    '''
    Run leakage test with settings in the paX dictionary, return result dict
    # I think it just sweeps from zero to Stop_V and back if you want
    '''
    paX['Lk'] = locals()
    print(paX['Lk'])
    with Leakage(paX) as L:
        #print('measurement 1')
        L.MeasureLeakage()
    results = L.Rslts
    # For some reason we have lists instead of arrays.  probably Hans has converted between the two 10s of times already up to this point
    for k,v in results.items():
        if type(v) is list:
            results[k] = np.array(v)
    # Rename to my standard names
    results['I'] = results.pop('Imeas')
    results['V'] = results.pop('Vapp')
    # Stick the paX in for good measure
    results['paX'] = copy.deepcopy(paX)
    return results

def measureVThreshold(RecID = '',        # Determines the filename of the scope traces.  if blank, D:/Test/test.bin
                      Pol = 1,           # polarity (+/-1) will be overridden by VC-polarity (what?)
                      Algo = 0,          # Algorithms to detect threshold: 0 = matched filter, 1 = soft = 2-linefilt
                      Start_V = 0.25,    # threshold ramp: minimum voltages (>0)
                      End_V = 4,         # threshold ramp: end of voltage ramp (> Start_V)
                      Step_V= 0.05,      # threshold ramp: step size for volages > 0
                      Tsmax_ns = 2,      # threshold ramp: max sampling time of the scope in ns
                      Dly_ns = 10000,    # delay in ns to pulse ('pause')
                      Dur_ns = 100,      # duration of pulse in ns
                      Rise_ns = 2,       # rise time of pulse in ns
                      Fall_ns = 10,      # Fall time in ns of pulse
                      Deltat_ns = 2,     # Time interval for AWG programming in ns.  All times MUST be multiples of this time (including 1)
                      NCycl = 1,         # # of cycles to use
                      EvalStart_ns = 20, # evaluation of amplitudes of current/voltage is delayed after the pulse start
                      EvalEnd_ns = 4,    # evaluation of amplitudes of current/voltage is stopped before the pulse has ended
                      AMod = 0.0,        # amplitude modulation in V to be put over entire WF. 0 means off
                      fMod = 250000000,  # frequency in Hz of amplitude modulation
                      Analyze1 = 0,      # special mode. Can analyze only 1 WF and do more analysis later
                      MaxCycles = -1,    # special mode.  tells it to pack MaxCycles data in one waveform. if more required, record more
                      PreFire_V = 0,     # special mode.  was intended for superfast drift, not being used
                      TONCtime_ns = 0,   # TONC: time for which current is reversed. >0 triggers TONC
                      TONCstart_ns = 50, # time after pulse start in ns when TONC is initiated
                      TONCampl = -1,     # amplitude of 'TONC-ed' part: -1 fully reversed, 0 = switched off, +1 = not reversed
                      OSA = 0.,          # scaler for exp part of the overshoot amplitude for ALL pulses. 0 means no OS
                      OSAD = 0,          # scaler for constant part of overshoot for ALL pulses. 0 means no OS
                      OSAVar = 0,        # scaler for exp fart of overshoot amplitude for variable pulses. 0 means no OS
                      OSTau_ns = 0.1,    # time constant in ns for exponential part of OS for ALL pulses
                      OSDur_ns = 10,     # duration in ns of canstant overshoot for ALL pulses
                      OSVarTau_ns = 0.1, # time constant in ns for exponential part of OS for variable pulses
                      OSVarDur_ns = 50,  # duration in ns of constant overshoot for variable pulses
                      OSAVarD = 0,       # scaler for constant part of overshoot for variable pulses. 0 mean
                      RngA = -1,         # range of channel A for picoscope. -1 means that range is selected according to Auxfile
                      RngB = -1,         # range of channel B for picoscope. -1 means that range is selected according to Auxfile
                      RngC = -1,         # range of channel C for picoscope. -1 means that range is selected according to Auxfile
                      RngD = -1,         # range of channel D for picoscope. -1 means that range is selected according to Auxfile
                      TEOgain = 25       # gain of TEOs amplifier in dB. -99 means leave unchanged
                      ):
    '''
    Run VTh test with settings in the paX dictionary, return result dict
    Description of what this test does goes here
    Square Pulse sequence with increasing amplitude, and multiple cycles (repeats)
    Hans says: TEO AWG is has fixed 500 MS/s.  Starts breaking if waveform is a couple million samples.
    '''
    paX['VTh'] = locals()
    paX['VTh']['RecID'] = ''
    paXcopy = copy.deepcopy(paX)
    # Hans's comment: needed to simulate situation for many cycles, skip otherwise
    #paX["VTh"]["RecID"] = UtilsCollection.CreateRecID(paX["General"]["StageName"])
    with VThreshold(paX) as VT:
        VT.MeasureVThreshold()
    # Results is a list with length NCycl
    # Stupidly only returns some processed data -- waveforms are apparently written to disk, in a location that depends on the RecID
    # Check TEOModule.TEObox.StoreTraces
    results = VT.Rslts
    for r in results:
        r['paX'] = paXcopy
        # For some reason we have lists instead of arrays.
        for k,v in r.items():
            if type(v) is list:
                r[k] = np.array(v)
        # Rename to my standard names
        #r['I'] = r.pop('Imeas')
        r['V'] = r.pop('Vapp')
    if NCycl == 1:
        results = results[0]
    return results

def measure_resistances():
    # Use my dumb metadata cycler which wasn't really meant for auto probing.
    # TODO: modify MetaHandler for programmatic iteration
    m = io.MetaHandler()
    m.load_lassen(dep_code='Rentier', sample_number=1, module='001B', die_rel=1)
    m.filenamekeys.append('note')
    for i in range(len(m.df)):
        m.static['note'] = 'Leakage'
        gotoDevice(m['die_rel'], m['module'], m['device'])
        # Make contact
        p.zUp()
        # Measure
        d = measureLeakage(Stop_V=.01, nPts=20, Bidirectional=1, Mode=0, Range=-7)
        filepath = os.path.join(datadir, datestr + '_Leakage', m.filename())
        io.write_pandas_pickle(m.attach(d), filepath, drop=None)
        m.next()
    