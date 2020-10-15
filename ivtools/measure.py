"""
Functions for measuring IV data
"""
import ivtools.analyze
import ivtools.instruments as instruments
import ivtools.settings

from matplotlib import pyplot as plt
from fractions import Fraction
from math import gcd
import numpy as np
import time
import pandas as pd
import os
from functools import partial
import pickle
import signal
import logging

log = logging.getLogger('measure')


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

def pulse_and_capture(waveform, ch=['A', 'B'], fs=1e6, duration=1e-3, n=1, interpwfm=True, pretrig=0, posttrig=0,
                      **kwargs):
    '''
    Send n pulses of the input waveform and capture on specified channels of picoscope.
    Duration determines the length of one repetition of waveform.
    '''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    # Set up to capture for n times the duration of the pulse
    # TODO have separate arguments for pulse duration and frequency, sampling frequency, number of samples per pulse
    # TODO make pulse and capture for builtin waveforms
    sampling_factor = (n + pretrig + posttrig)

    ps.capture(ch, freq=fs,
                   duration=duration * sampling_factor,
                   pretrig=pretrig / sampling_factor,
                   **kwargs)
    # Pulse the waveform n times, this will trigger the picoscope capture.
    rigol.pulse_arbitrary(waveform, duration, n=n, interp=interpwfm)

    data = ps.get_data(ch)

    return data

def picoiv(wfm, duration=1e-3, n=1, fs=None, nsamples=None, smartrange=1, autosplit=True,
           into50ohm=False, channels=['A', 'B'], autosmoothimate=True, splitbylevel=None,
           savewfm=False, pretrig=0, posttrig=0, interpwfm=True, **kwargs):
    '''
    Pulse a waveform, measure on picoscope channels, and return data
    Provide either fs or nsamples

    smartrange 1 autoranges the monitor channel
    smartrange 2 tries some other fancy shit to autorange the current measurement channel

    autosplit will split the waveforms into n chunks

    into50ohm will double the waveform amplitude to cancel resistive losses when using terminator

    by default we sample for exactly the length of the waveform,
    use "pretrig" and "posttrig" to sample before and after the waveform
    units are fraction of one pulse duration

    kwargs go nowhere
    '''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    if not type(wfm) == np.ndarray:
        wfm = np.array(wfm)

    if not (bool(fs) ^ bool(nsamples)):
        raise Exception('Must pass either fs or nsamples, and not both')
    if fs is None:
        fs = nsamples / duration

    if smartrange == 2:
        # Smart range for the compliance circuit
        smart_range(np.min(wfm), np.max(wfm), ch=['A', 'B'])
    elif smartrange:
        # Smart range the monitor channel
        smart_range(np.min(wfm), np.max(wfm), ch=[ivtools.settings.MONITOR_PICOCHANNEL])

    # Let pretrig and posttrig refer to the fraction of a single pulse, not the whole pulsetrain

    sampling_factor = (n + pretrig + posttrig)

    # Set picoscope to capture
    # Sample frequencies have fixed values, so it's likely the exact one requested will not be used
    actual_fs = ps.capture(ch=channels,
                           freq=fs,
                           duration=duration * sampling_factor,
                           pretrig=pretrig / sampling_factor)

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
    ivdata = ivtools.settings.pico_to_iv(chdata)

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
        ivdata = ivtools.analyze.smoothimate(ivdata, window=window, factor=factor, columns=None)

    if autosplit and (n > 1):
        print('Splitting data into individual pulses')
        if splitbylevel is None:
            nsamples = duration * actual_fs
            if 'downsampling' in ivdata:
                # Not exactly correct but I hope it's close enough
                nsamples /= ivdata['downsampling']
            ivdata = ivtools.analyze.splitiv(ivdata, nsamples=nsamples)
        elif splitbylevel is not None:
            # splitbylevel can split loops even if they are not the same length
            # Could take more time though?
            # This is not a genius way to determine to split at + or - dV/dt
            increasing = bool(sign(argmax(wfm) - argmin(wfm)) + 1)
            ivdata = ivtools.analyze.split_by_crossing(ivdata, V=splitbylevel, increasing=increasing, smallest=20)

    return ivdata

def freq_response(ch='A', fstart=10, fend=1e8, n=10, amp=.3, offset=0, trigsource='TriggerAux'):
    ''' Apply a series of sine waves with rigol, and sample the response on picoscope. Return data without analysis.'''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    if fend > 1e8:
        raise Exception('Rigol can only output up to 100MHz')

    freqs = np.geomspace(fstart, fend, n)
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
            maxrate = 5e9
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
    from instruments import RigolDG5000
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
    '''
    Tries to choose the best range, offset for input channels, given the minimum and maximum values of the applied waveform
    This is easy to do for the monitor channel, but hard to do for channels that depend on the load (DUT)
    TODO: Somehow have reconfigurable auto-ranging functions for different setups.  Function attributes on pico_to_iv?
    TODO: don't let this function change the pico state.  Just return the calculated ranges.
    '''
    ps = instruments.Picoscope()
    # Auto offset for current input
    possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
    # Each range has a maximum possible offset
    max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))

    # TODO allow multiple monitors?
    monitor_channel = ivtools.settings.MONITOR_PICOCHANNEL
    if monitor_channel in ch:
        # Assuming CHA is directly sampling the output waveform, we can easily optimize the range
        arange, aoffs = ps.best_range((v1, v2), atten=ps.atten[monitor_channel], coupling=ps.coupling[monitor_channel])
        ps.range[monitor_channel] = arange
        ps.offset[monitor_channel] = aoffs

    if 'B' in ch:
        # Smart ranging channel B is harder, since we don't know what kind of device is being measured.
        # Center the measurement range on zero current
        #OFFSET['B'] = -COMPLIANCE_CURRENT * 2e3
        # channelb should never go below zero, except for potentially op amp overshoot
        # I have seen it reach -0.1V
        CC = ivtools.settings.COMPLIANCE_CURRENT
        polarity = np.sign(ivtools.settings.CCIRCUIT_GAIN)
        if R is None:
            # Hypothetical resistance method
            # Signal should never go below 0V (compliance)
            b_min = 0
            b_resistance = max(abs(v1), abs(v2)) / CC / 1.1
            # Compliance current sets the voltage offset at zero input.
            # Add 10% to be safe.
            b_max = polarity * (CC - min(v1, v2) / b_resistance) * 2e3 * 1.1
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

def measure_dc_gain(Vin=1, ch='C', R=10e3):
    import ivtools.plot as plot
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
    plot.plot_channels(chdata)
    chvalue = np.mean(chdata[ch])
    print('Measured {} volts on picoscope channel {}'.format(chvalue, ch))

    gain = R * chvalue / Vin
    # Set output back to zero
    rigol.pulse_arbitrary([Vin, 0,0,0,0], 1e-3)
    return gain

def measure_ac_gain(R=1000, freq=1e4, ch='C', outamp=1):
    # Probably an obsolete function
    # Send a test pulse to determine better range to use
    import ivtools.plot as plot
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
    plot.plot_channels(data)

    # will change the range and offset after all
    squeeze_range(data, [ch])

    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1000)
    data = ps.get_data(chs)

    plot.plot_channels(data)

    return max(abs(fft.fft(data[ch]))[1:-1]) / max(abs(fft.fft(data['A']))[1:-1]) * R


########### Old compliance circuit ###################
# this version had no auto-offset-compensation
# used a look up table and usb DAQ

# Voltage dividers
# (Compliance control voltage)      DAC0 - 12kohm - 12kohm
# (Input offset corrcetion voltage) DAC1 - 12kohm - 1.2kohm

def set_compliance_old(cc_value):
    '''
    Use two analog outputs to set the compliance current and compensate input offset.
    Right now we use static lookup tables for compliance and compensation values.
    '''
    daq = instruments.USB2708HS()
    if cc_value > 1e-3:
        raise Exception('Compliance value out of range! Max 1 mA.')
    fn = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    abspath = os.path.abspath(fn)
    if os.path.isfile(abspath):
        print('Reading calibration from file {abspath}'.format())
    else:
        raise Exception('No compliance calibration.  Run calibrate_compliance().')
    with open(fn, 'rb') as f:
        cc = pickle.load(f)
    DAC0 = round(np.interp(cc_value, cc['ccurrent'], cc['dacvals']))
    DAC1 = np.interp(DAC0, cc['dacvals'], cc['compensationV'])
    print('Setting compliance to {} A'.format(cc_value))
    daq.analog_out(0, dacval=DAC0)
    daq.analog_out(1, volts=DAC1)
    ivtools.settings.COMPLIANCE_CURRENT = cc_value
    ivtools.settings.INPUT_OFFSET = 0

def calibrate_compliance_old(iterations=3, startfromfile=True, ndacvals=40):
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

    fn = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    abspath = os.path.abspath(fn)
    fdir = os.path.split(abspath)[0]
    if not os.path.isdir(fdir):
        os.makedirs(fdir)
    if startfromfile and not os.path.isfile(fn):
        print(f'No calibration file exists at {abspath}')
        startfromfile = False

    for it in range(iterations):
        ccurrent = []
        offsets = []
        if len(offsets_list) == 0:
            if startfromfile:
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
            plot.mypause(.1)
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
        plot.mypause(.1)
        #plt.pause(.1)
    output = {'dacvals':dacvals, 'ccurrent':ccurrent, 'compensationV':compensations,
              'date':time.strftime('%Y-%m-%d'), 'time':time.strftime('%H:%M:%S'), 'iterations':iterations}

    with open(fn, 'wb') as f:
        pickle.dump(output, f)
    print('Wrote calibration to ' + fn)

    return compensations

def plot_compliance_calibration_old():
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

def measure_compliance_old():
    '''
    Our circuit does not yet compensate the output for different current compliance levels
    Right now current compliance is set by a physical knob, not by the computer.  This will change.
    The current way to measure compliance approximately is by measuring the output at zero volts input,
    because in this case, the entire compliance current flows across the output resistor.

    There is a second complication because the input is not always at zero volts, because it is not compensated fully.
    This can be measured as long is there is some connection between the AWG output and the compliance circuit input (say < 1Mohm).
    '''
    import ivtools.plot as plot
    gain = ivtools.settings.CCIRCUIT_GAIN
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()
    # Put AWG in hi-Z mode (output channel off)
    # Then current at compliance circuit input has to be ~zero
    # (except for CHA scope input, this assumes it is set to 1Mohm, not 50ohm)
    ps.ps.setChannel('A', 'DC', 50e-3, 1, 0)
    rigol.output(False)
    plot.mypause(.1)
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
    ivtools.settings.INPUT_OFFSET = Amean
    print('Measured voltage offset of compliance circuit input: {}'.format(Amean))

    # Channel B should be measuring the circuit output with the entire compliance current across the output resistance.

    # Seems rigol doesn't like to pulse zero volts. It makes a beep but then apparently does it anyway.
    #Vout = pulse_and_capture(waveform=np.zeros(100), ch='B', fs=1e6, duration=1e-3)
    ccurrent =  Bmean / (gain)
    ivtools.settings.COMPLIANCE_CURRENT = ccurrent
    print('Measured compliance current: {} A'.format(ccurrent))

    return (ccurrent, Amean)


########### New compliance circuit ###################
# self-biasing compliance circuit version that uses AWG for current control

def calibrate_compliance_with_keithley(Rload=1000):
    '''
    Use keithley to calibrate current compliance levels
    Attach keithley channel A to input of compliance circuit through a resistor (Rload)
    Use rigol channel 2 for the compliance control voltage
    this sets different compliance levels and measures the whole I-V curve for that setting
    takes a while, but will be accurate
    '''
    k = instruments.Keithley2600()
    rigol = instruments.RigolDG5000()

    Remitter = 2050
    Vneg = 9.6 # negative power rail
    def approxVc(Ic):
        # first approximation for calibration of current source
        return Ic*Remitter - Vneg + 0.344

    def approxIc(Vc):
        return (Vc + Vneg - 0.344) / Remitter

    # Measure IV sweeps with different voltages applied to the current control
    approxImax = 2e-3
    n = 30
    approxVcmax = approxVc(approxImax)
    vlist = np.linspace(-9.6, approxVcmax, n)
    # what V is necessary to reach the highest I? apply 1.25 times that
    Vmax = approxIc(max(vlist)) * Rload * 1.25
    Vmax = min(Vmax, 5)
    Vmin = Rload * 500e-6
    data = []
    for v in vlist:
        rigol.DC(v, 2)
        d = k.iv(tri(Vmin, Vmax, n=100))
        k.waitready()
        d = k.get_data()
        data.append(d)
    data = pd.DataFrame(data)

    # calculate the current saturation level and slope for every control voltage
    # early effect slope (hundreds of kohm)
    data['R'] = ivtools.analyze.resistance(data, Vmax/1.25, Vmax)
    data['Ic'] = ivtools.analyze.slicebyvalue(data, 'V', Vmax/1.25, Vmax).I.apply(np.mean)
    data['Vc'] = vlist
    #plt.figure()
    #plt.plot(vc, np.polyval(p, vc))

    # fit diode equation
    from scipy.optimize import curve_fit
    # params Is=1e-10, Ioff=-2.075e-6, Re=2050, Vneg=-9.6
    guess = (1e-10, -2e-6, 2000)
    try:
        p = curve_fit(compliance_voltage, data.Ic, data.Vc, guess)[0]
        Is, Ioff, Re = p
    except RuntimeError as e:
        log.error('compliance current calibration fit failed!')
        Is = 2.5825e-10
        Ioff = -2.063e-6
        Re = 1997.3

    #guess = (1e-10, -2e-6)
    #p = curve_fit(compliance_voltage, data.Ic, data.Vc, guess)[0]
    #Is, Ioff = p
    #Re = 2000

    # There's not a great way to store these scalars in the dataframe
    data._metadata = ['Is', 'Ioff', 'Re']
    data.Is = Is
    data.Ioff = Ioff
    data.Re = Re

    calfp = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    os.makedirs(os.path.split(calfp)[0], exist_ok=True)
    data.to_pickle(calfp)

    plot_compliance_calibration()

    return data

def plot_compliance_calibration(calfile=None):
    cal = read_compliance_calibration_file(calfile)

    plt.figure()
    plt.plot(cal.Vc, cal.Ic, marker='.')
    I = np.linspace(-4e-6, np.max(cal.Ic), 1000)
    V = compliance_voltage(I, cal.Is, cal.Ioff, cal.Re)
    plt.plot(V, I)
    plt.xlabel('V_base [V]')
    plt.ylabel('I_limit [A]')

    plt.figure()
    for i,r in cal.iterrows():
        plt.plot(r.V, r.I, label=r.Vc)
    plt.xlabel('V_input')
    plt.ylabel('I_input')

def read_compliance_calibration_file(calfile=None):
    if calfile is None:
        calfile = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    if os.path.isfile(calfile):
        return pd.read_pickle(calfile)
    else:
        log.error('Calibration file not found!')

def set_compliance(cc_value):
    '''
    Setting a DC value for the current limit
    '''
    rigol = instruments.RigolDG5000()
    Vc = compliance_voltage_lookup(cc_value)
    log.debug(f'Setting compliance control voltage to {Vc}')
    rigol.DC(Vc, ch=2)
    rigol.output(True, ch=2)

def compliance_voltage_lookup(I):
    '''
    Interpolates a calibration table to convert between Icompliance and Vcontrol
    if you ask for a current outside of the calibration range, the output will be clipped
    # TODO Use a spline
    '''
    calfile = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    if os.path.isfile(calfile):
        cal = pd.read_pickle(calfile)
        # interpolate for best value for vc
        Vc = np.interp(I, cal['Ic'], cal['Vc'])
    else:
        print('No calibration file! Using diode equation.')
        #Remitter = 2050
        #Vneg = 9.6
        #Vc = Ic*Remitter - Vneg + 0.344
        Vc = compliance_voltage(I)
    return Vc

def compliance_voltage(I, Is=2.5825e-10, Ioff=-2.063e-6, Re=1997.3, Vneg=-9.6):
    '''
    Use continuous diode equation to calculate the control voltage needed for a target compliance current (Icc)
    default values are from calibration
    depends on a good fit! plot_compliance_calibration() to check it!

    it's quite linear above 100 uA or so.  This is more important for lesser values

    Inputs are calibration constants
    Is -- diode reverse saturation current
    Re -- emitter resistance
    Ioff -- input offset current
    Vneg -- negative rail of power supply
    '''
    # System that needs to be solved for I(V):
    # I = Is(exp(Vd/VT) - 1) + Ioff
    # Vd = V - IRe
    # where V is the voltage level above the negative rail

    # Thermal Voltage
    V_T = 25.69e-3
    V = V_T * np.log((I - Ioff + Is) / Is) + I*Re + Vneg
    if type(V) is np.ndarray:
        V[np.isnan(V)] = Vneg
    return V

def hybrid_IV(Imax=500e-6, Vmin=-3, dur=1e-3):
    '''
    this does a current sweep in the positive direction, followed by a voltage sweep in the
    negative direction using compliance circuit.  All I,V points are measured

    will work better with compliance circuit that sits on the sourcing side

    tested a bit but it's a work in progress
    TODO: make a pulse2ch_and_capture for these synchronized signals, and use that
    '''

    # To synchronize rigol channels, you need an external trigger split into the two sync ports on the back.
    # This is provided by a second rigol here
    rigol = instruments.RigolDG5000('USB0::0x1AB1::0x0640::DG5T155000186::INSTR')
    rigol2 = instruments.RigolDG5000('USB0::0x1AB1::0x0640::DG5T182500117::INSTR')
    ps = instruments.Picoscope()

    I_to_Vc = compliance_voltage_lookup

    # TODO we should transition more smoothly at the crossover
    #      might need to carefully decide how to do it
    wfm_len = 2**12
    mid = wfm_len // 2
    t = np.linspace(0, 2*np.pi, wfm_len)
    # current source during voltage sweep (should be high enough to bring the collector
    # current up and therefore emitter resistance down)
    Iidle = I_to_Vc(Imax*1.3)
    # voltage source during current sweep (needs to be enough to supply all the currents)
    # but Vneedle can only go up to 4v or so!
    # meaning that we can't get an accurate Vdevice unless Vneedle < 4
    Vidle = 4
    iwfm = I_to_Vc(Imax*np.sin(t)**2)
    iwfm[mid:] = Iidle
    vwfm = -np.abs(Vmin)*np.sin(t)**2
    vwfm[:mid] = Vidle
    vwfm[0] = vwfm[-1] = 0
    iwfm[0] = iwfm[-1] = Iidle
    # just add some smoother linear transitions
    # sorry this is ugly af
    # when converting to current source, drop the current before increasing the voltage limit
    # when converting to voltage source, drop the voltage before increasing the current limit
    trans_len = 2**6
    iwfm = np.concatenate(([iwfm[0]], np.linspace(iwfm[0], iwfm[1], trans_len), np.repeat(iwfm[1], trans_len),
                           iwfm[1:mid], np.repeat(iwfm[mid-1], trans_len),
                           np.linspace(iwfm[mid-1], [iwfm[mid]], trans_len), iwfm[mid:]))
    vwfm = np.concatenate(([vwfm[0]], np.repeat(vwfm[0], trans_len), np.linspace(vwfm[0], vwfm[1], trans_len), vwfm[1:mid],
                           np.linspace(vwfm[mid-1], [vwfm[mid]], trans_len), np.repeat(vwfm[mid], trans_len), vwfm[mid:]))
    '''
    plt.figure()
    plt.plot(iwfm)
    plt.plot(vwfm)
    '''

    rigol.load_volatile_wfm(vwfm, dur, ch=1)
    time.sleep(.1)
    # Use rigol offset, or else the zero idle level will destroy everything
    iwfm -= Iidle
    rigol.load_volatile_wfm(iwfm, dur, ch=2, offset=Iidle)
    rigol.setup_burstmode(n=1, trigsource='EXT', ch=1)
    rigol.setup_burstmode(n=1, trigsource='EXT', ch=2)
    # This delay seems to be necessary the first time you set up burst mode?
    time.sleep(1)

    ps.capture(['A', 'B', 'C'], 1e8, dur, timeout_ms=3000, pretrig=0)
    # trigger everything
    rigol2.pulse_builtin('SQU', duration=1e-6, amp=5, offset=5)
    d = ps.get_data(['A', 'B', 'C'])
    d = ccircuit_to_iv(d)
    #plot_channels(d)
    #savedata(d)

    d['t'] = ivtools.analyze.maketimearray(d)
    '''
    sd = smoothimate(d, 300, 10)
    # TODO cut out the crossover distortion
    #plotiv(sd, 'Vd', 'I')

    # if no linear transitions
    #plotiv(slicebyvalue(sd, 't', dur*.0015, dur*(0.5-0.001)), 'Vd', 'I')
    #plotiv(slicebyvalue(sd, 't', dur*(0.5 + 0.001)), 'Vd', 'I', ax=plt.gca())

    dplus = slicebyvalue(sd, 't', dur*.017, dur*(0.5-0.001))
    dminus = slicebyvalue(sd, 't', dur*(0.5 + 0.017))
    plotiv(dplus, 'Vd', 'I')
    hplotiv(dminus, 'Vd', 'I')

    plotiv(dplus, 'Vd', 'I', plotfunc=arrowpath, maxsamples=100)
    hplotiv(dminus, 'Vd', 'I', plotfunc=arrowpath, maxsamples=100, color='C1')
    '''

    return d

########### Digipot ####################

def digipot_test(plot=True):
    '''
    This rapidly makes sure everything is working properly
    Short the needles, use channel A as the input waveform monitor, chC as the current and chB as node voltage
    '''
    # Use these settings but don't change the state of picoscope
    coupling = dict(A='DC', B='DC50', C='DC50')
    ranges = dict(A=5, B=5, C=1)
    ps = instruments.Picoscope()
    dp = instruments.WichmannDigipot()
    rigol = instruments.RigolDG5000()
    dur = 1e-2
    if plot:
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        plt.show()
    data = []
    channels = ['A', 'B', 'C']

    # Check Bypass first
    dp.set_bypass(1)
    ps.capture(channels, freq=10000/dur, duration=dur, chrange=ranges, chcoupling=coupling)
    rigol.pulse_builtin('SIN', duration=dur, amp=1)
    d = ps.get_data(channels)
    d = digipot_to_iv(d)
    d = ivtools.analyze.moving_avg(d,1000)
    data.append(d)
    if plot:
        #ax.plot(d['V'], d['I'])
        ax.plot(d['V'], d['V']/d['I'], color='black')

    dp.set_bypass(0)

    for w,Rnom in dp.Rmap.items():
        dp.set_wiper(w) # Should have the necessary delay built in
        # Put 10 milliwatt through each resistor
        #A = np.sqrt(10e-3 * Rnom)
        #A = min(A, 5)
        A = 3
        Iexpected = A/Rnom
        ranges['C'] = ps.best_range([-Iexpected*50, Iexpected*50])[0]
        ps.capture(channels, freq=10000/dur, duration=dur, chrange=ranges, chcoupling=coupling)
        rigol.pulse_builtin('SIN', duration=dur, amp=A)
        d = ps.get_data(channels)
        d = digipot_to_iv(d)
        d = ivtools.analyze.moving_avg(d,100)
        data.append(d)
        if plot:
            #ax.plot(d['V'], d['I'], label=w)
            #color = ax.lines[-1].get_color()
            #ax.plot(d['V'], d['V']/Rnom, label=w, linestyle='--', alpha=.2, color=color)
            # Or
            ax.plot(d['V'], d['V']/d['I'], label=w)
            color = ax.lines[-1].get_color()
            ax.plot([-5,5], [Rnom, Rnom], label=w, linestyle='--', alpha=.2, color=color)
            plt.pause(.1)
            plt.xlim(-3, 3)
            plt.ylim(40, 60000)

    return data

def digipot_calibrate(plot=True):
    '''
    Connect keithley to digipot to measure the resistance values
    Can be done by contacting something conductive or smashing the needles together in the air

    configured for the single pot mode, not parallel or series
    '''
    dp = instruments.WichmannDigipot()
    k = instruments.Keithley2600()
    if plot:
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(np.arange(34), dp.Rlist, marker='.', label='current calibration')
        ax.set_xlabel('Pot setting')
        ax.set_ylabel('Resistance [Ohm]')
        plt.show()
        plt.pause(.1)

    dp.set_bypass(0)
    data = []
    for w,Rnom in dp.Rmap.items():
        dp.set_wiper(w) # Should have the necessary delay built in
        # Apply a volt, measure current
        k.iv([1], Irange=0, Ilimit=10e-3, nplc=1)
        while not k.done():
            plt.pause(.1)
        d = k.get_data()
        d['R'] = d['V']/d['I']
        data.append(d)
        if plot:
            plt.scatter(w, d['R'])
            plt.pause(.1)

    print([d['R'][0].round(2) for d in data])
    return data


########### Conversion from picoscope channel data to IV data ###################

def ccircuit_to_iv_old(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    For the early version of compliance circuit, which needs manual compensation
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    CC = ivtools.settings.COMPLIANCE_CURRENT
    IO = ivtools.settings.INPUT_OFFSET
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    B = datain['B']
    #C = datain['C']
    gain = ivtools.settings.CCIRCUIT_GAIN
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

def ccircuit_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    For the newer version of compliance circuit, which compensates itself and amplifies the needle voltage
    chA should monitor the applied voltage
    chB should be the needle voltage
    chC should be the amplified current
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    B = datain['B']
    C = datain['C']
    #C = datain['C']
    gain = ivtools.settings.CCIRCUIT_GAIN
    dataout['V'] = dtype(A)
    #dataout['V_formula'] = 'CHA - IO'
    #dataout['I'] = 1e3 * (B - C) / R
    dataout['Vneedle'] = dtype(B)
    dataout['Vd'] = dataout['V'] - dataout['Vneedle'] # Does not take phase shift into account!
    dataout['I'] = dtype(C / gain)
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    dataout['units'] = {'V':'V', 'Vd':'V', 'Vneedle':'V', 'I':'A'}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
    dataout['gain'] = gain
    return dataout


def ccircuit_yellow_to_iv(datain, dtype=np.float32):
    '''
    For the redesigned circuit that sources voltage
    needle voltage is the device voltage
    no need to sample AWG waveform -- don't have a cow if there's no CHA

    Convert picoscope channel data to IV dict
    For the newer version of compliance circuit, which compensates itself and amplifies the needle voltage

    chA should monitor the applied voltage
    chB should be the needle voltage
    chC should be the amplified current
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    if datain['B'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    B = datain['B']
    C = datain['C']
    if 'A' in datain:
        A = datain['A']
        dataout['V'] = dtype(A)
    else:
        dataout['V'] = dtype(B)
    gain = ivtools.settings.CCIRCUIT_GAIN
    #dataout['V_formula'] = 'CHA - IO'
    #dataout['I'] = 1e3 * (B - C) / R
    dataout['I'] = dtype(C / gain)
    dataout['Vd'] = dtype(B) # actually should subtract 50*I
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    dataout['units'] = {'V':'V', 'Vd':'V', 'I':'A'}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
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

def digipot_to_iv(datain, gain=1/50, Vd_channel='B', I_channel='C', dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    for digipot circuit with device voltage probe
    gain is in A/V, in case you put an amplifier on the output

    Simultaneous sampling is faster when not using adjacent channels (i.e. A&B)
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    chs_sampled = [ch for ch in ['A', 'B', 'C', 'D'] if ch in datain]
    if not chs_sampled:
        print('No picoscope data detected')
        return datain

    dataout = datain
    # If data is raw, convert it here
    if datain[chs_sampled[0]].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)

    if 'units' not in dataout:
        dataout['units'] = {}

    monitor_channel = ivtools.settings.MONITOR_PICOCHANNEL
    if monitor_channel in datain:
        V = datain[monitor_channel]
        dataout['V'] = V # Subtract voltage on output?  Don't know what it is necessarily.
        dataout['units']['V'] = 'V'
    if Vd_channel in datain:
        Vd = datain[Vd_channel]
        dataout['Vd'] = Vd
        dataout['units']['Vd'] = 'V'
    if I_channel in datain:
        I = datain[I_channel] * gain
        dataout['I'] = I
        dataout['units']['I'] = 'A'

    dataout['Igain'] = gain

    return dataout


############# Waveforms ###########################
def tri(v1, v2=0, n=None, step=None, repeat=1):
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


############# Misc ##############################

def beep(freq=500, ms=300):
    import winsound
    winsound.Beep(freq, ms)

def tts(text=''):
    # immediately speak the text, does not return until finished
    import pyttsx3
    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty('voices')       #getting details of current voice
    tts_engine.setProperty('voice', voices[1].id)
    tts_engine.say(text)
    tts_engine.runAndWait()

def tts_queue(text):
    # Put some utterances in a queue, then you can call tts_thread() to start it
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.say(text)

def tts_thread(text=None):
    '''
    runs in a separate thread, but you can't call it again until the previous one is complete

    sometimes it gives an error and some unknown condition can get it to work again..

    Seems to be that if you run it outside of a thread first, then it works inside a thread forever.
    therefore you can initialize with text_to_speech('')

    You can't run it again until it finishes -- check the done flag
    '''
    from threading import Thread
    import pyttsx3
    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty('voices')       #getting details of current voice
    tts_engine.setProperty('voice', voices[1].id)
    #tts_engine.say('')
    class Threader(Thread):
        def __init__(self, *args, **kwargs):
            Thread.__init__(self, *args, **kwargs)
            self.daemon = True
            self.start()
        def run(self):
            # Doesn't work
            #while tts_engine.isBusy():
                #time.sleep(.1)
            if text:
                tts_engine.say(text)
            tts_engine.runAndWait()
            # Is this an abuse of a function attribute?
            tts_thread.done = True
    tts_thread.done = False
    ttsthread = Threader()

class controlled_interrupt():
    '''
    Allows you to protect code from keyboard interrupt using a context manager.
    Potential safe break points can be individually specified by breakpoint().
    If a ctrl-c is detected during protected execution, the code will be interrupted at the next break point.
    You can always press ctrl-c TWICE to bypass this protection and interrupt immediately.
    ctrl-c behavior will be returned to normal at the end of the with block
    '''
    def __init__(self):
        self.interruptable = None
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, *args):
        self.stop()
    def start(self):
        signal.signal(signal.SIGINT, self.int_handler)
        self.interruptable = False
    def stop(self):
        # Now you can use "ctrl+c" as usual.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    def int_handler(self, signalNumber, frame):
        if self.interruptable:
            signal.default_int_handler()
        else:
            print('Not safe to interrupt! Will interrupt at next opportunity.')
            print('Press ctrl-c again to override')
            self.interruptable = True
    def breakpoint(self):
        if self.interruptable:
            raise KeyboardInterrupt


########### Teo calibration #####################

def teo_calibration_keithley(V=5, pts=100, plot=True, check=True, nplc=1):
    '''
    Calibration of the output voltage and the physical voltage monitor (SMA)
    '''

    teo = instruments.TeoSystem()
    k = instruments.Keithley2600()

    teo.LF_voltage(0)
    k.source_func('i', ch=1)
    k.source_func('i', ch=2)
    k.source_level('i', 0, ch=1)
    k.source_level('i', 0, ch=2)

    input("Connect Keithley channel A to HFV and channel B to Vmonitor")

    k.source_output(True, ch=1)
    k.source_output(True, ch=2)
    k.nplc(nplc)
    k.measure_range('v', 'auto', ch=1)
    k.measure_range('v', 'auto', ch=2)

    def cal(c):
        '''
        Start calibration of HFV and Vmonitor_SMA.

        c stands for check, so if should be False to perform an actual calibration, and True to check
        '''

        data = {}
        data['desired'] = list(np.linspace(-V, V, pts))
        data['HFV'] = []
        data['monitor'] = []

        for v in data['desired']:
            teo.LF_voltage(v)
            m1 = k.measure('v', ch=1)
            m2 = k.measure('v', ch=2)
            # We will use picoscope to measure Vmonitor, and this will measure half of the voltage at Vmonitor,
            # so we fake that bellow.
            m2 = m2/2
            if c:# Application of calibration
                m1 = (m1 - calibration_data['fit_HFV'][1]) / calibration_data['fit_HFV'][0]
                m2 = (m2 - calibration_data['fit_monitor'][1]) / calibration_data['fit_monitor'][0]
            data['HFV'].append(m1)
            data['monitor'].append(m2)

        k.source_level('i', 0, ch=1)
        k.source_level('i', 0, ch=2)
        teo.LF_voltage(0)

        fit = np.polyfit(data['desired'], data['HFV'], 1)
        data['fit_HFV'] = fit
        fit = np.polyfit(data['HFV'], data['monitor'], 1)
        data['fit_monitor'] = fit

        return data

    calibration_data = cal(False)

    log.info(f"""
Calibration results of HFV:
    Slope: {calibration_data['fit_HFV'][0]}
    Interception: {calibration_data['fit_HFV'][1]}V

Calibration results of Vmonitor:
    Slope: {calibration_data['fit_monitor'][0]}
    Interception: {calibration_data['fit_monitor'][1]}V
        """)

    if check:
        check_data = cal(True)

        log.info(f"""
Check results of HFV:
    Slope: {check_data['fit_HFV'][0]}
    Interception: {check_data['fit_HFV'][1]}V

Check results of Vmonitor:
    Slope: {check_data['fit_monitor'][0]}
    Interception: {check_data['fit_monitor'][1]}V
            """)

        data = {'calibration': calibration_data, 'check': check_data}

    else:
        data = calibration_data

    if plot:
        x = np.array(data['calibration']['desired'])

        plt.figure('HFV', clear=True)
        plt.title(f'HFV Calibration\nV={V}, pts={pts}, nplc={nplc}')
        plt.xlabel('Desired voltage (V)')
        plt.ylabel('Measured voltage (V)')
        plt.plot(x, x, label='Target', color='red')

        plt.figure('Vmoitor', clear=True)
        plt.title(f'Vmonitor Calibration\nV={V}, pts={pts}, nplc={nplc}')
        plt.xlabel('Measured voltage at HFV (V)')
        plt.ylabel('Measured voltage at Vmonitor (V)')
        plt.plot(x, x, label='Target', color='red')

        plt.figure('HFV')

        y = data['calibration']['HFV']
        plt.plot(x, y, label='Calibration data', color='lightgreen', alpha=0.8, marker='.')

        y = data['calibration']['fit_HFV'][0] * x + data['calibration']['fit_HFV'][1]
        plt.plot(x, y, label='Calibration fit', color='green', alpha=0.8)

        plt.figure('Vmoitor')

        y = data['calibration']['monitor']
        plt.plot(x, y, label='Calibration data', color='lightgreen', alpha=0.8, marker='.')

        y = data['calibration']['fit_monitor'][0] * x + data['calibration']['fit_monitor'][1]
        plt.plot(x, y, label='Calibration fit', color='green', alpha=0.8)

        if check:

            plt.figure('HFV')

            y = data['check']['HFV']
            plt.plot(x, y, label='Check data', color='lightblue', alpha=0.8, marker='.')

            y = data['check']['fit_HFV'][0] * x + data['check']['fit_HFV'][1]
            plt.plot(x, y, label='Check fit', color='blue', alpha=0.8)

            plt.figure('Vmoitor')

            y = data['check']['monitor']
            plt.plot(x, y, label='Check data', color='lightblue', alpha=0.8, marker='.')

            y = data['check']['fit_monitor'][0] * x + data['check']['fit_monitor'][1]
            plt.plot(x, y, label='Check fit', color='blue', alpha=0.8)

        plt.figure('HFV')
        plt.legend()
        plt.show()
        plt.figure('Vmoitor')
        plt.legend()
        plt.show()

    k.source_output(False, ch=1)
    k.source_output(False, ch=2)

    return data

def teo_calibration_picoscope(plot=True):
    '''
    Calibration of the output voltage and the physical voltage monitor (SMA)
    '''

    teo = instruments.TeoSystem()
    ps = instruments.Picoscope()

    teo.LF_voltage(0)

    input("Connect Picoscope channel A to HFV and channel B to Vmonitor")

    big_samples = 100
    small_samples = 100
    small_sample_duration = 10e-5

    stimated_time = 2* big_samples * small_samples * small_sample_duration

    stimated_mins = int(stimated_time/60)
    stimated_secs = int(stimated_time - stimated_mins*60)
    log.info(f"Stimated time: {stimated_mins} minutes {stimated_secs} seconds.")

    def cal(check):
        '''
        Start calibration of HFV and Vmonitor_SMA.

        c stands for check, so if should be False to perform an actual calibration, and True to check
        '''

        data = {}
        data['desired'] = list(np.linspace(-4, 4, big_samples))
        data['HFV'] = []
        data['Vmonitor_SMA'] = []
        data['Vmonitor_sw'] = []
        data['Vmonitor_sw'] = []

        m1 = ps.measure(ch=['A', 'B'], freq=None, duration=small_sample_duration, nsamples=small_samples,
                        trigsource='TriggerAux', triglevel=0.1, timeout_ms=1000, direction='Rising', pretrig=0.0,
                        chrange={'A': 5, 'B': 1}, choffset={'A': 0, 'B': 0}, chcoupling=None, chatten=None,
                        raw=False, dtype=np.float32, plot=False, ax=None)

        logging.getLogger('instruments').setLevel(50) # To avoid logging 100 of "Actual picoscope sample freq..."
        for v in data['desired']:
            teo.LF_voltage(v)
            m1 = ps.measure(ch=['A', 'B'], freq=None, duration=small_sample_duration, nsamples=small_samples,
                           trigsource='TriggerAux', triglevel=0.1, timeout_ms=1000, direction='Rising', pretrig=0.0,
                           chrange={'A': 5, 'B': 1}, choffset={'A': 0, 'B': 0}, chcoupling=None, chatten=None,
                           raw=False, dtype=np.float32, plot=False, ax=None)
            m2 = teo.LF_voltage()

            if check:# Application of calibration
                m1['A'] = (m1['A'] - calibration_data['fit_HFV'][1]) / calibration_data['fit_HFV'][0]
                m1['B'] = (m1['B'] - calibration_data['fit_Vmonitor_SMA'][1]) / calibration_data['fit_Vmonitor_SMA'][0]
                m2 = (m2 - calibration_data['fit_Vmonitor_sw'][1]) / calibration_data['fit_Vmonitor_sw'][0]

            data['HFV'].append(np.mean(m1['A']))
            data['Vmonitor_SMA'].append(np.mean(m1['B']))
            data['Vmonitor_sw'].append(m2)

        logging.getLogger('instruments').setLevel(1)  # Back to normal

        teo.LF_voltage(0)

        fit = np.polyfit(data['desired'], data['HFV'], 1)
        data['fit_HFV'] = fit
        fit = np.polyfit(data['HFV'], data['Vmonitor_SMA'], 1)
        data['fit_Vmonitor_SMA'] = fit
        fit = np.polyfit(data['HFV'], data['Vmonitor_sw'], 1)
        data['fit_Vmonitor_sw'] = fit

        return data

    calibration_data = cal(False)

    log.info(f"""
Calibration results of HFV:
    Slope: {calibration_data['fit_HFV'][0]}
    Interception: {calibration_data['fit_HFV'][1]}V

Calibration results of Vmonitor_SMA:
    Slope: {calibration_data['fit_Vmonitor_SMA'][0]}
    Interception: {calibration_data['fit_Vmonitor_SMA'][1]}V

Calibration results of Vmonitor_software:
    Slope: {calibration_data['fit_Vmonitor_sw'][0]}
    Interception: {calibration_data['fit_Vmonitor_sw'][1]}V
        """)

    check_data = cal(True)

    log.info(f"""
Check results of HFV:
    Slope: {check_data['fit_HFV'][0]}
    Interception: {check_data['fit_HFV'][1]}V

Check results of Vmonitor_SMA:
    Slope: {check_data['fit_Vmonitor_SMA'][0]}
    Interception: {check_data['fit_Vmonitor_SMA'][1]}V

Check results of Vmonitor_software:
    Slope: {check_data['fit_Vmonitor_sw'][0]}
    Interception: {check_data['fit_Vmonitor_sw'][1]}V
        """)

    data = {'calibration': calibration_data, 'check': check_data}

    if plot:
        teo_calibration_plot(data)

    return data

def teo_calibration_plot(data):
    x = np.array(data['calibration']['desired'])

    plt.figure()
    plt.title(f'HFV Calibration')
    plt.xlabel('Desired voltage (V)')
    plt.ylabel('Measured voltage (V)')
    plt.plot(x, x, label='Target', color='red')

    y = data['calibration']['HFV']
    plt.plot(x, y, label='Before calibration', color='lightgreen', alpha=0.8, marker='.')

    y = data['calibration']['fit_HFV'][0] * x + data['calibration']['fit_HFV'][1]
    #plt.plot(x, y, label='Calibration fit', color='green', alpha=0.8)

    y = data['check']['HFV']
    plt.plot(x, y, label='After calibration', color='lightblue', alpha=0.8, marker='.')

    y = data['check']['fit_HFV'][0] * x + data['check']['fit_HFV'][1]
    #plt.plot(x, y, label='Check fit', color='blue', alpha=0.8)

    plt.legend()
    plt.show()

    plt.figure()
    plt.title(f'Vmonitor_SMA Calibration')
    plt.xlabel('Measured voltage at HFV (V)')
    plt.ylabel('Measured voltage at Vmonitor_SMA (V)')
    plt.plot(x, x, label='Target', color='red')

    y = data['calibration']['Vmonitor_SMA']
    plt.plot(x, y, label='Before calibration', color='lightgreen', alpha=0.8, marker='.')

    y = data['calibration']['fit_Vmonitor_SMA'][0] * x + data['calibration']['fit_Vmonitor_SMA'][1]
    #plt.plot(x, y, label='Calibration fit', color='green', alpha=0.8)

    y = data['check']['Vmonitor_SMA']
    plt.plot(x, y, label='After calibration', color='lightblue', alpha=0.8, marker='.')

    y = data['check']['fit_Vmonitor_SMA'][0] * x + data['check']['fit_Vmonitor_SMA'][1]
    #plt.plot(x, y, label='Check fit', color='blue', alpha=0.8)

    plt.legend()
    plt.show()

    plt.figure()
    plt.title(f'Vmonitor_software Calibration')
    plt.xlabel('Measured voltage at HFV (V)')
    plt.ylabel('Measured voltage at Vmonitor_sw (V)')
    plt.plot(x, x, label='Target', color='red')

    y = data['calibration']['Vmonitor_sw']
    plt.plot(x, y, label='Before calibration', color='lightgreen', alpha=0.8, marker='.')

    y = data['calibration']['fit_Vmonitor_sw'][0] * x + data['calibration']['fit_Vmonitor_sw'][1]
    #plt.plot(x, y, label='Calibration fit', color='green', alpha=0.8)

    y = data['check']['Vmonitor_sw']
    plt.plot(x, y, label='After calibration', color='lightblue', alpha=0.8, marker='.')

    y = data['check']['fit_Vmonitor_sw'][0] * x + data['check']['fit_Vmonitor_sw'][1]
    #plt.plot(x, y, label='Check fit', color='blue', alpha=0.8)

    plt.legend()
    plt.show()


























