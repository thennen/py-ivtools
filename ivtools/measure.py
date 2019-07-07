"""
Functions for measuring IV data
"""
# Local imports
from . import plot as ivplot
from . import analyze
from . import instruments
from . import persistent_state as settings

from matplotlib import pyplot as plt
from fractions import Fraction
from math import gcd
import numpy as np
import time
import pandas as pd
import os
import visa
from functools import partial
import pickle

########### Picoscope - Rigol AWG testing #############
def pulse_and_capture_builtin(ch=['A', 'B'], shape='SIN', amp=1, freq=None, offset=0, phase=0, duration=None,
                              ncycles=10, samplespercycle=None, fs=None, extrasample=0, **kwargs):
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    if not (bool(samplespercycle) ^ bool(fs)):
        raise Exception('Must give either samplespercycle, or sampling frequency (fs), and not both')
    if not (bool(freq) ^ bool(duration)):
        raise Exception('Must give either freq or duration, and not both')

    if fs is None:
        fs = freq * samplespercycle
    if freq is None:
        freq = 1. / duration

    ps.capture(ch=ch, freq=fs, duration=(ncycles+extrasample)/freq, **kwargs)

    rigol.pulse_builtin(freq=freq, amp=amp, offset=offset, phase=phase, shape=shape, n=ncycles)

    data = ps.get_data(ch)

    return data

def pulse_and_capture(waveform, ch=['A', 'B'], fs=1e6, duration=1e-3, n=1, interpwfm=True, extrasample=0,
                      **kwargs):
    '''
    Send n pulses of the input waveform and capture on specified channels of picoscope.
    Duration determines the length of one repetition of waveform.
    '''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    # Set up to capture for n times the duration of the pulse
    # TODO have separate arguments for pulse duration and frequency, sampling frequency, number of samples per pulse
    ps.capture(ch, freq=fs, duration=(n+extrasample)*duration, **kwargs)
    # Pulse the waveform n times, this will trigger the picoscope capture.
    rigol.pulse_arbitrary(waveform, duration, n=n, interp=interpwfm)

    data = ps.get_data(ch)

    return data


def picoiv(wfm, duration=1e-3, n=1, fs=None, nsamples=None, smartrange=1, autosplit=True,
           into50ohm=False, channels=['A', 'B'], autosmoothimate=True, splitbylevel=None,
           savewfm=False, pretrig=0, interpwfm=True, **kwargs):
    '''
    Pulse a waveform, plot pico channels, IV, and save to d variable
    Provide either fs or nsamples
    kwargs go nowhere
    TODO: Don't let smartrange change the global settings
    '''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    if not (bool(fs) ^ bool(nsamples)):
        raise Exception('Must pass either fs or nsamples, and not both')
    if fs is None:
        fs = nsamples / duration

    if smartrange == 2:
        # Smart range for the compliance circuit
        smart_range(np.min(wfm), np.max(wfm), ch=['A', 'B'])
    elif smartrange:
        # Smart range the monitor channel
        smart_range(np.min(wfm), np.max(wfm), ch=[settings.MONITOR_PICOCHANNEL])

    # Let pretrig refer to the fraction of a single pulse, not the whole pulsetrain
    pretrig /= n
    # Set picoscope to capture
    # Sample frequencies have fixed values, so it's likely the exact one requested will not be used
    actual_fs = ps.capture(ch=channels,
                             freq=fs,
                             duration=n*duration,
                             pretrig=pretrig)

    # This makes me feel good, but I don't think it's really necessary
    time.sleep(.05)
    if into50ohm:
        # Multiply voltages by 2 to account for 50 ohm input
        wfm = 2 * wfm

    # Send a pulse
    rigol.pulse_arbitrary(wfm, duration=duration, interp=interpwfm, n=n, ch=1)

    trainduration = n * duration
    print('Applying pulse(s) ({:.2e} seconds).'.format(trainduration))
    time.sleep(n * duration * 1.05)
    #ps.waitReady()
    print('Getting data from picoscope.')
    # Get the picoscope data
    # This goes into a global strictly for the purpose of plotting the (unsplit) waveforms.
    chdata = ps.get_data(channels, raw=True)
    print('Got data from picoscope.')
    # Convert to IV data (keeps channel data)
    ivdata = settings.pico_to_iv(chdata)

    ivdata['nshots'] = n

    if savewfm:
        # Measured voltage has noise sometimes it's nice to plot vs the programmed waveform.
        # You will need to interpolate it, however..
        # Or can we read it off the rigol??
        ivdata['Vwfm'] = wfm

    if autosmoothimate:
        nsamples_shot = ivdata['nsamples_capture'] / n
        # Smooth by 0.3% of a shot
        window = max(int(nsamples_shot * 0.003), 1)
        # End up with about 1000 data points per shot
        # This will be bad if you send in a single shot waveform with multiple cycles
        # In that case, you shouldn't be using autosmoothimate or autosplit
        # TODO: make a separate function for IV trains?
        if autosmoothimate is True:
            # yes I meant IS true..
            npts = 1000
        else:
            # Can pass the number of data points you would like to end up with
            npts = autosmoothimate
        factor = max(int(nsamples_shot / npts), 1)
        print('Smoothimating data with window {}, factor {}'.format(window, factor))
        # TODO: What if we want to retain a copy of the non-smoothed data?
        # It's just sometimes ugly to plot, doesn't always mean that I don't want to save it
        # Maybe only smoothimate I and V?
        ivdata = analyze.smoothimate(ivdata, window=window, factor=factor, columns=None)

    if autosplit:
        print('Splitting data into individual pulses')
        if n > 1 and (splitbylevel is None):
            nsamples = duration * actual_fs
            if 'downsampling' in ivdata:
                # Not exactly correct but I hope it's close enough
                nsamples /= ivdata['downsampling']
            ivdata = analyze.splitiv(ivdata, nsamples=nsamples)
        elif splitbylevel is not None:
            # splitbylevel can split loops even if they are not the same length
            # Could take more time though?
            # This is not a genius way to determine to split at + or - dV/dt
            increasing = bool(sign(argmax(wfm) - argmin(wfm)) + 1)
            ivdata = analyze.split_by_crossing(ivdata, V=splitbylevel, increasing=increasing, smallest=20)

    return ivdata


def freq_response(ch='A', fstart=10, fend=1e8, n=10, amp=.3, offset=0, trigsource='TriggerAux'):
    ''' Apply a series of sine waves with rigol, and sample the response on picoscope. Return data without analysis.'''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    if fend > 1e8:
        raise Exception('Rigol can only output up to 100MHz')

    freqs = np.logspace(np.log10(fstart), np.log10(fend), n)
    data = []
    for freq in freqs:
        # Figure out how many cycles to sample and at which sample rate.
        # In my tests with FFT:
        # Number of cycles did not matter much, as long as there was exactly an integer number of cycles
        # Higher sampling helps a lot, with diminishing returns after 10^5 total data points.

        # I don't know what the optimum sampling conditions for the sine curve fitting method.
        # Probably takes longer for big datasets.  And you need a good guess for the number of cycles contained in the dataset.
        # (Actually is much faster, surprisingly)

        # How many cycles you want to have per frequency
        target_cycles = 100
        # How many data points you want to have
        target_datapoints = 1e5
        # Max amount of time (s) you are willing to wait for a measurement of a single frequency
        max_time_per_freq = 10
        # Capture at least this many cycles
        minimum_cycles = 1

        # Can sample 5 MS/s, divided among the channels
        # ** To achieve 2.5 GS/s sampling rate in 2-channel mode, use channel A or B and channel C or D.
        if len(ch) == 1:
            maxrate = 5e9 / len(ch)
        elif len(ch) == 2:
            # 4 channel combinations allow 2.5 GS/s sampling rate
            if set(ch) in (set(('A', 'B')), set(('C', 'D'))):
                maxrate = 1.25e9
            else:
                maxrate = 2.5e9
        else:
            maxrate = 1.25e9

        cycles_per_maxtime = freq * max_time_per_freq
        time_for_target_cycles = target_cycles / freq

        # TODO: use hardware oversampling to increase resolution
        if cycles_per_maxtime < minimum_cycles:
            # We still need to capture at least certain number of whole cycles, so it will take longer.  Sorry.
            ncycles = minimum_cycles
            fs = target_datapoints * freq / ncycles
        elif cycles_per_maxtime < target_cycles:
            # Cycle a reduced number of (integer) times in order to keep measurement time down
            ncycles = int(cycles_per_maxtime)
            fs = target_datapoints * freq / ncycles
        elif target_datapoints / time_for_target_cycles < maxrate:
            # Excluding the possibility that someone set a really dumb max_time_per_freq,
            # this means that we acquire our target number of cycles, and our target number of samples.
            ncycles = target_cycles
            fs = target_datapoints * freq / ncycles
        else:
            # We are limited by the sampling rate of picoscope.
            # Capture the target number of cycles but with a reduced number of samples
            ncycles = target_cycles
            fs = maxrate
            # Or would it be better to capture an increased number of cycles?  To be determined..

        # Pico triggering appears to have about 6 ns of jitter.
        # To avoid capturing zeros at the end of the pulses, we will do an extra pulse at higher frequencies
        # Don't do it at low frequencies because it could lock up the AWG for an extra 1/freq
        if freq > 1e4:
            npulses = ncycles + 1
        else:
            npulses = ncycles

        duration = ncycles / freq


        # TODO: Should I apply the signal for a while before sampling?  Here I am sampling immediately from the first cycle.
        # The aux trigger has a delay and jitter for some reason.  Maybe triggering directly on the channel is better?
        # 
        if trigsource == 'TriggerAux':
            triglevel = 0.05
        else:
            triglevel = 0

        # This one pulses exactly npulses and tries to capture them from the beginning
        #ps.capture(ch, freq=fs, duration=duration, pretrig=0, triglevel=triglevel, trigsource=trigsource)
        #rigol.pulse_builtin(freq=freq, amp=amp, offset=offset, shape='SIN', n=npulses, ch=1)
        rigol.continuous_builtin(freq=freq, amp=amp, offset=offset, shape='SIN', ch=1)
        #time.sleep(.1)
        # I have seen this take a while on the first shot
        time.sleep(.5)
        ps.capture(ch, freq=fs, duration=duration, pretrig=0, triglevel=triglevel, trigsource=trigsource)
        d = ps.get_data(ch)
        d['ncycles'] = ncycles
        data.append(d)

        # TODO: make some plots that show when debug=True is passed
        rigol.output(False, ch=1)

    return data


def tripulse(n=1, v1=1.0, v2=-1.0, duration=None, rate=None):
    '''
    Generate n bipolar triangle pulses.
    Voltage sweep rate will  be constant.
    Trigger immediately
    '''
    rigol = instruments.RigolDG5000()

    rate, duration = _rate_duration(v1, v2, rate, duration)

    wfm = tri(v1, v2)

    rigol.pulse_arbitrary(wfm, duration, n=n)


def sinpulse(n=1, vmax=1.0, vmin=-1.0, duration=None):
    '''
    Generate n sine pulses.
    Trigger immediately
    If you pass vmin != -vmax, will not start at zero!
    '''
    rigol = instruments.RigolDG5000()

    wfm = (vmax - vmin) / 2 * np.sin(np.linspace(0, 2*pi, ps.AWGMaxSamples)) + ((vmax + vmin) / 2)

    rigol.pulse_arbitrary(wfm, duration, n=n)


def smart_range(v1, v2, R=None, ch=['A', 'B']):
    # TODO: don't let this function change the pico state.  Just return the calculated ranges.
    ps = instruments.Picoscope()
    # Auto offset for current input
    possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
    # Each range has a maximum possible offset
    max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))

    monitor_channel = settings.MONITOR_PICOCHANNEL
    if monitor_channel in ch:
        # Assuming CHA is directly sampling the output waveform, we can easily optimize the range
        arange, aoffs = ps.best_range((v1, v2), atten=ps.atten[monitor_channel])
        ps.range[monitor_channel] = arange
        ps.offset[monitor_channel] = aoffs

    if 'B' in ch:
        # Smart ranging channel B is harder, since we don't know what kind of device is being measured.
        # Center the measurement range on zero current
        #OFFSET['B'] = -COMPLIANCE_CURRENT * 2e3
        # channelb should never go below zero, except for potentially op amp overshoot
        # I have seen it reach -0.1V
        CC = settings.COMPLIANCE_CURRENT
        if R is None:
            # Hypothetical resistance method
            # Signal should never go below 0V (compliance)
            b_min = 0
            b_resistance = max(abs(v1), abs(v2)) / CC / 1.1
            # Compliance current sets the voltage offset at zero input.
            # Add 10% to be safe.
            b_max = (CC - min(v1, v2) / b_resistance) * 2e3 * 1.1
        else:
            # R was passed, assume device has constant resistance with this value
            b_min = (CC - max(v1, v2) / R) * 2e3
            b_max = (CC- min(v1, v2) / R) * 2e3
        brange, boffs = ps.best_range((b_min, b_max))
        ps.range['B'] = brange
        ps.offset['B'] = boffs

def raw_to_V(datain, dtype=np.float32):
    '''
    Convert 8 bit values to voltage values.  datain should be a dict with the 8 bit channel
    arrays and the RANGE and OFFSET values.
    return a new dict with updated channel arrays
    '''
    channels = ['A', 'B', 'C', 'D']
    dataout = {}
    for c in channels:
        if (c in datain.keys()) and (datain[c].dtype == np.int8):
            dataout[c] = datain[c] / dtype(2**8) * dtype(datain['RANGE'][c] * 2) - dtype(datain['OFFSET'][c])
    for k in datain.keys():
        if k not in dataout.keys():
            dataout[k] = datain[k]
    return dataout

def _rate_duration(v1, v2, rate=None, duration=None):
    '''
    Determines the duration or sweep rate of a triangle type pulse with constant sweep rate.
    Pass rate XOR duration, return (rate, duration).
    '''
    if not (bool(duration) ^ bool(rate)):
        raise Exception('Must give either duration or rate, and not both')
    if duration is not None:
        duration = float(duration)
        rate = 2 * (v1 - v2) / duration
    elif rate is not None:
        rate = float(rate)
        duration = 2 * (v1 - v2) / rate

    return rate, duration


########### Compliance circuit ###################

# Voltage dividers
# (Compliance control voltage)      DAC0 - 12kohm - 12kohm
# (Input offset corrcetion voltage) DAC1 - 12kohm - 1.2kohm


def set_compliance(cc_value):
    '''
    Use two analog outputs to set the compliance current and compensate input offset.
    Right now we use static lookup tables for compliance and compensation values.
    '''
    daq = instruments.USB2708HS()
    if cc_value > 1e-3:
        raise Exception('Compliance value out of range! Max 1 mA.')
    fn = settings.COMPLIANCE_CALIBRATION_FILE
    print('Reading calibration from file {}'.format(os.path.abspath(fn)))
    with open(fn, 'rb') as f:
        cc = pickle.load(f)
    DAC0 = round(np.interp(cc_value, cc['ccurrent'], cc['dacvals']))
    DAC1 = np.interp(DAC0, cc['dacvals'], cc['compensationV'])
    print('Setting compliance to {} A'.format(cc_value))
    daq.analog_out(0, dacval=DAC0)
    daq.analog_out(1, volts=DAC1)
    settings.COMPLIANCE_CURRENT = cc_value
    settings.INPUT_OFFSET = 0

def calibrate_compliance(iterations=3, startfromfile=True, ndacvals=40):
    '''
    Set and measure some compliance values throughout the range, and save a calibration look up table
    Need picoscope channel B connected to circuit output
    and picoscope channel A connected to circuit input (through needles or smallish resistor is fine)
    This takes a minute..
    '''

    ps = instruments.Picoscope()
    daq = instruments.USB2708HS()
    # Measure compliance currents and input offsets with static Vb

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ccurrent_list = []
    offsets_list = []
    dacvals = np.int16(np.linspace(0, 2**11, ndacvals))

    for it in range(iterations):
        ccurrent = []
        offsets = []
        if len(offsets_list) == 0:
            if startfromfile:
                fn = 'compliance_calibration.pkl'
                print('Reading calibration from file {}'.format(os.path.abspath(fn)))
                with open(fn, 'rb') as f:
                    cc = pickle.load(f)
                compensations = np.interp(dacvals, cc['dacvals'], cc['compensationV'])
            else:
                # Start with constant compensation
                compensations = .55 /0.088 * np.ones(len(dacvals))
        else:
            compensations -= np.array(offsets_list[-1]) / .085
        for v,cv in zip(dacvals, compensations):
            daq.analog_out(1, volts=cv)
            daq.analog_out(0, v)
            ivplot.mypause(.1)
            #plt.pause(.1)
            cc, offs = measure_compliance()
            ccurrent.append(cc)
            offsets.append(offs)
        ccurrent_list.append(ccurrent)
        offsets_list.append(offsets)
        ax1.plot(dacvals, np.array(ccurrent) * 1e6, '.-')
        ax1.set_xlabel('DAC0 value')
        ax1.set_ylabel('Compliance Current [$\mu$A]')
        ax2.plot(dacvals, offsets, '.-', label='Iteration {}'.format(it))
        ax2.set_xlabel('DAC0 value')
        ax2.set_ylabel('Input offset')
        ax2.legend()
        ivplot.mypause(.1)
        #plt.pause(.1)
    output = {'dacvals':dacvals, 'ccurrent':ccurrent, 'compensationV':compensations,
              'date':time.strftime('%Y-%m-%d'), 'time':time.strftime('%H:%M:%S'), 'iterations':iterations}
    calibrationfile = 'compliance_calibration.pkl'
    with open(calibrationfile, 'wb') as f:
        pickle.dump(output, f)
    print('Wrote calibration to ' + calibrationfile)

    return compensations

def plot_compliance_calibration():
    fn = 'compliance_calibration.pkl'
    print('Reading calibration from file {}'.format(os.path.abspath(fn)))
    with open(fn, 'rb') as f:
        cc = pickle.load(f)
    ccurrent = 1e6 * np.array(cc['ccurrent'])
    dacvals = cc['dacvals']
    compensationV = cc['compensationV']
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.plot(dacvals, ccurrent, '.-')
    ax.set_xlabel('DAC0 value')
    ax.set_ylabel('Compliance Current [$\mu$A]')
    ax2.plot(dacvals, compensationV, '.-')
    ax2.set_xlabel('DAC0 value')
    ax2.set_ylabel('Compensation V (DAC1)')
    plt.tight_layout()
    return cc

def measure_compliance():
    '''
    Our circuit does not yet compensate the output for different current compliance levels
    Right now current compliance is set by a physical knob, not by the computer.  This will change.
    The current way to measure compliance approximately is by measuring the output at zero volts input,
    because in this case, the entire compliance current flows across the output resistor.

    There is a second complication because the input is not always at zero volts, because it is not compensated fully.
    This can be measured as long is there is some connection between the AWG output and the compliance circuit input (say < 1Mohm).
    '''
    gain = settings.CCIRCUIT_GAIN
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()
    # Put AWG in hi-Z mode (output channel off)
    # Then current at compliance circuit input has to be ~zero
    # (except for CHA scope input, this assumes it is set to 1Mohm, not 50ohm)
    ps.ps.setChannel('A', 'DC', 50e-3, 1, 0)
    rigol.outputstate(False)
    ivplot.mypause(.1)
    #plt.pause(.1)
    # Immediately capture some samples on channels A and B
    # Use these channel settings for the capture -- does not modify global settings
    # TODO pick the channel settings better or make it complain when the signal is out of range
    picosettings = {'chrange': {'A':.2, 'B':2},
                    'choffset': {'A':0, 'B':np.sign(gain)*-2},
                    'chatten': {'A':1, 'B':1},
                    'chcoupling': {'A':'DC', 'B':'DC'}}
    ps.capture(['A', 'B'], freq=1e5, duration=1e-1, timeout_ms=1, **picosettings)
    picodata = ps.get_data(['A', 'B'])
    #plot_channels(picodata)
    Amean = np.mean(picodata['A'])
    Bmean = np.mean(picodata['B'])

    # Channel A should be connected to the rigol output and to the compliance circuit input, perhaps through a resistance.
    settings.INPUT_OFFSET = Amean
    print('Measured voltage offset of compliance circuit input: {}'.format(Amean))

    # Channel B should be measuring the circuit output with the entire compliance current across the output resistance.

    # Seems rigol doesn't like to pulse zero volts. It makes a beep but then apparently does it anyway.
    #Vout = pulse_and_capture(waveform=np.zeros(100), ch='B', fs=1e6, duration=1e-3)
    ccurrent =  Bmean / (gain)
    settings.COMPLIANCE_CURRENT = ccurrent
    print('Measured compliance current: {} A'.format(ccurrent))

    return (ccurrent, Amean)

########### Conversion from picoscope channel data to IV data ###################

def ccircuit_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    For the early version of compliance circuit, which needs manual compensation
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    CC = settings.COMPLIANCE_CURRENT
    IO = settings.INPUT_OFFSET
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    B = datain['B']
    #C = datain['C']
    gain = settings.CCIRCUIT_GAIN
    dataout['V'] = dtype(A - IO)
    #dataout['V_formula'] = 'CHA - IO'
    dataout['INPUT_OFFSET'] = IO
    #dataout['I'] = 1e3 * (B - C) / R
    # Current circuit has 0V output in compliance, and positive output under compliance
    # Unless you know the compliance value, you can't get to current, because you don't know the offset
    # TODO: Figure out if/why this works
    dataout['I'] = dtype(-B / gain + CC)
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    dataout['units'] = {'V':'V', 'I':'A'}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
    dataout['CC'] = CC
    dataout['gain'] = gain
    return dataout

def rehan_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    for Rehan amplifier
    Assumes constant gain vs frequency.  This isn't right.
    Careful! Scope input couplings will affect the gain!
    if x10 channel has 50 ohm termination, then gain of x200 channel reduced by 2!
    everything should be terminated with 50 ohms obviously
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    # Volts per amp
    gainC = 10.6876 * 50
    gainD = 113.32 * 50
    # 1 Meg, 33,000

    # I think these depend a lot on the scope input range/offset..
    offsD = 0.073988
    offsC = 0.006025

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    C = datain['C']

    dataout['V'] = A
    dataout['I'] = (C - offsC) / gainC
    dataout['units'] = {'V':'V', 'I':'A'}
    dataout['Cgain'] = gainC
    dataout['Coffs'] = offsC

    if 'D' in datain:
        D = datain['D']
        dataout['I2'] = (D - offsD) / gainD
        dataout['Dgain'] = gainD
        dataout['Doffs'] = offsD
        dataout['units'].update({'I2':'A'})

    return dataout

def femto_log_to_iv(datain, dtype=np.float32):
    # Adjust output offset so that 0.1V in --> 1V out on the 2V input setting
    # Then 0.1V in --> 1.25V out on the 200mV setting
    # Input offset also important. should minimize the output signal when input is zero
    dataout = datain

    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    B = datain['B']

    dataout['V'] = A
    # 2V setting
    #dataout['I'] = 10**((B - 1) / 0.25) * 0.1 / 50
    dataout['I'] = 10**((B - 1) / 0.25) * 0.01 / 50
    dataout['units'] = {'V':'V', 'I':'A'}

    return dataout

def TEO_HFext_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    for TEO HF output channels
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    # Volts per amp
    gainA = 1
    gainB = 1
    gainC = 1
    gainD = -2

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    B = datain['B']
    C = datain['C']
    D = datain['D']

    dataout['V'] = C / gainC
    dataout['I'] = A / gainA
    dataout['I2'] = B / gainB
    dataout['I3'] = D / gainD
    dataout['units'] = {'V':'V', 'I':'A', 'I2':'A', 'I3':'A'}
    dataout['gain'] = {'A':gainA, 'B':gainB, 'C':gainC, 'D':gainD}

    return dataout


def Rext_to_iv(datain, R=50, Ichannel='C', dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    This is for the configuration where you are using a series resistor
    and probing the center junction
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)

    # Use channel A and C as input, because simultaneous sampling is faster than using A and B
    A = datain['A']
    C = datain[Ichannel]

    # V device
    dataout['V'] = A - C
    dataout['I'] = C / R
    dataout['units'] = {'V':'V', 'I':'A'}
    dataout['Rs_ext'] = R

    return dataout

def measure_dc_gain(Vin=1, ch='C', R=10e3):
    # Measure dc gain of rehan amplifier
    # Apply voltage
    rigol = instruments.RigolDG5000()

    print('Outputting {} volts on Rigol CH1'.format(Vin))
    rigol.pulse_arbitrary(np.repeat(Vin, 100), 1e-3)
    time.sleep(1)
    # Measure output
    measurechannels = ['A', ch]
    ps.capture(measurechannels, freq=1e6, duration=1, timeout_ms=1)
    time.sleep(.1)
    chdata = ps.get_data(measurechannels)
    ivplot.plot_channels(chdata)
    chvalue = np.mean(chdata[ch])
    print('Measured {} volts on picoscope channel {}'.format(chvalue, ch))

    gain = R * chvalue / Vin
    # Set output back to zero
    rigol.pulse_arbitrary([Vin, 0,0,0,0], 1e-3)
    return gain

def measure_ac_gain(R=1000, freq=1e4, ch='C', outamp=1):
    # Probably an obsolete function
    # Send a test pulse to determine better range to use
    arange, aoffset = best_range([outamp, -outamp])
    RANGE = {}
    OFFSET = {}
    RANGE['A'] = arange
    OFFSET['A'] = aoffset
    # Power supply is 5V, so this should cover the whole range
    RANGE[ch] = 5
    OFFSET[ch] = 0
    sinwave = outamp * sin(np.linspace(0, 1, 2**12)*2*pi)
    chs = ['A', ch]
    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1, chrange=RANGE, choffset=OFFSET)
    data = ps.get_data(chs)
    ivplot.plot_channels(data)

    # will change the range and offset after all
    squeeze_range(data, [ch])

    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1000)
    data = ps.get_data(chs)

    ivplot.plot_channels(data)

    return max(abs(fft.fft(data[ch]))[1:-1]) / max(abs(fft.fft(data['A']))[1:-1]) * R



def tri(v1, v2, n=None, step=None, repeat=1):
    '''
    Create a triangle pulse waveform with a constant sweep rate.  Starts and ends at zero.

    Can optionally pass number of data points you want, or the voltage step between points.

    If neither n or step is passed, return the shortest waveform which reaches v1 and v2.

    Building these waveforms has some unexpected nuances.  Check the code if unsure.
    '''
    # How this waveform is constructed depends on our value system.
    # For example, when we pass n, do we care most that the waveform:
    # 1. actually has n points?
    # 2. has constant datapoint spacing?
    # 3. contains the extrema?

    # For IV sweeps I would say containing the extrema is important,
    # In order to make sure the waveform actually contains 0, v1 and v2, we need to allow
    # uneven spacing for the extreme points (v1, v2, last 0).
    # I think this is better than strictly enforcing equal sweep rate, because you can then get
    # deformed undersampled triangle waveforms

    # Within this scheme, there does not necessarily exist a step size that leads to n data points.
    # (e.g. going from zero to one extreme and back to zero can't be done in an even number of data points)
    # When a step size does exist, it is not unique.
    # What to do when a step size doesn't exist, and how to find the optimum step size when it does exist?
    # It's a fun problem to think about, but hard to solve.

    #nulls = sum(np.array((v1, v2-v1, v2)) == 0)
    #endpts = 3 - nulls
    if n is not None:
        dv = abs(v1) + abs(v2 - v1) + abs(v2)
        step = dv / (n - 1)
        # If I take the above step, I will end up between 1 and "endpts" extra points
        # Dumb stuff below
        #sstart = dv / (n-1)
        #sstop = dv / (n-endpts-1)
        #stest = np.linspace(sstart, sstop, 1024)
        # number of points you will end up with
        #ntest = abs(v1)//stest + (abs(v2-v1))//stest + abs(v2)//stest + 1 + endpts
        #step = stest[np.where(ntest == n)[0][0]]
    if step is not None:
        def sign(num):
            npsign = np.sign(num)
            return npsign if npsign !=0 else 1
        wfm = np.concatenate((np.arange(0, v1, sign(v1) * step),
                             np.arange(v1, v2, sign(v2 - v1) * step),
                             np.arange(v2, 0, -sign(v2) * step),
                             [0]))
        return wfm
    else:
        # Find the shortest waveform that truly reaches v1 and v2 with constant spacing
        # I don't think we need better than 1 mV resolution
        v1 = round(v1, 3)
        v2 = round(v2, 3)
        f1 = Fraction(str(v1))
        f2 = Fraction(str(v2))
        # This is depreciated for some reason
        #vstep = float(abs(fractions.gcd(fmax, fmin)))
        # Doing it this other way.. Seems faster by a large factor.
        a, b = f1.numerator, f1.denominator
        c, d = f2.numerator, f2.denominator
        # not a typo
        commond = float(b * d)
        vstep = gcd(a*d, b*c) / commond
        dv = v1 - v2
        # Using round because floating point errors suck
        # e.g. int(0.3 / 0.1) = int(2.9999999999999996) = 2
        n1 = round(abs(v1) / vstep + 1)
        n2 = round(abs(dv) / vstep + 1)
        n3 = round(abs(v2) / vstep + 1)
        wfm = np.concatenate((np.linspace(0 , v1, n1),
                            np.linspace(v1, v2, n2)[1:],
                            np.linspace(v2, 0 , n3)[1:]))

        # Filling the AWG record length with probably take more time than it's worth.
        # Interpolate to a "Large enough" waveform size
        #enough = 2**16
        #x = np.linspace(0, 1, enough)
        #xp = np.linspace(0, 1, len(wfm))
        #wfm = np.interp(x, xp, wfm)

        # Let AWG do the interpolation

        if repeat > 1:
            def lol():
                for i in range(repeat-1):
                    yield wfm[:-1]
                yield wfm
            wfm = np.concatenate([*lol()])
        return wfm

def square(vpulse, duty=.5, length=2**14, startval=0, endval=0, startendratio=1):
    '''
    Calculate a square pulse waveform.
    '''
    ontime = int(duty * length)
    remainingtime = length - ontime
    pretime = int(startendratio * remainingtime / (startendratio + 1))
    pretime = max(1, pretime)
    posttime = remainingtime - pretime
    posttime = max(1, posttime)
    prearray = np.ones(pretime) * startval
    pulsearray = np.ones(ontime) * vpulse
    postarray = np.ones(posttime) * endval
    return np.concatenate((prearray, pulsearray, postarray))


