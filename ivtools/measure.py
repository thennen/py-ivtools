""" Functions for measuring IV data with picoscope 6403C and Rigol AWG """

# Local imports
from . import plot

from fractions import Fraction
from math import gcd
import numpy as np
import time
import pandas as pd
import os
import visa

visa_rm = visa.ResourceManager()
'USB0::0x1AB1::0x0640::DG5T155000186::INSTR',
'TCPIP0::192.168.11.12::inst0::INSTR'

# TODO: try to connect to all known instruments

# These are None until the instruments are connected
# Don't clobber them though, in case this script is used with run -i
try:
    ps
except:
    ps = None
try:
    rigol
except:
    rigol = None
try:
    keithley
except:
    keithley = None


# Settings for picoscope channels.
# Also don't clobber them
try:
    COMPLIANCE_CURRENT
    INPUT_OFFSET
except:
    COMPLIANCE_CURRENT = 0
    INPUT_OFFSET = 0


def connect_picoscope():
    global ps
    if ps is None:
        try:
            ps = ps6000.PS6000()
            model = ps.getUnitInfo('VariantInfo')
            print('Picoscope {} connection succeeded.'.format(model))
            #print(ps.getAllUnitInfo())
        except:
            print('Connection to picoscope failed.  Could be an unclosed session.')
            ps = None
    else:
        try:
            model = ps.getUnitInfo('VariantInfo')
            print('Picoscope {} already connected.'.format(model))
            #info = ps.getAllUnitInfo()
            #print(info)
        except:
            print('ps variable is not None, and not an active picoscope connection.')


def connect_rigolawg():
    global rigol
    rigolstr = 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'
    if rigol is None:
        try:
            rigol = visa_rm.open_resource(rigolstr)
            idn = rigol.query('*IDN?')
            print('Rigol connection succeeded.')
            print('*IDN?  {}'.format(idn))
        except:
            print('Connection to Rigol AWG failed.')
            rigol = None
    else:
        try:
            # Check if rigol is already defined and connected
            idn = rigol.query('*IDN?')
            print('Rigol AWG already connected')
            print(idn)
        except:
            print('rigol variable is not None.  Doing nothing.')

def connect_keithley(ip='192.168.11.11'):
    global keithley
    # 2634B
    #Keithley_ip = '192.168.11.11'
    # 2636A
    #Keithley_ip = '192.168.11.12'
    Keithley_ip = ip
    Keithley_id = 'TCPIP::' + Keithley_ip + '::inst0::INSTR'
    if keithley is None:
        try:
            # Impatiently try to connect to keithley
            # Because it runs even if keithley is not connected and I have no intention to use it
            keithley = visa_rm.get_instrument(Keithley_id, open_timeout=250)
            idn = keithley.ask('*IDN?')
            print('Keithley *IDN?: {}'.format(idn))
        except:
            print('Connection to Keithley failed.')
            keithley = None
    else:
        try:
            # Is keithley already connected?
            idn = keithley.ask('*IDN?')
            print('Keithley already connected')
            print('Keithley *IDN?: {}'.format(idn))
        except:
            print('Keithley not responding, and keithley variable is not None.')


def connect_instruments():
    ''' Connect all the necessary equipment '''
    print('Attempting to connect all instruments.')
    connect_picoscope()
    connect_rigolawg()
    connect_keithley()


def close():
    global ps
    global rigol
    # Close connection to pico
    ps.close()
    ps = None
    # Close connection to rigol
    rigol.close()
    rigol = None


def pulse_and_capture_builtin(ch=['A', 'B'], shape='SIN', amp=1, freq=1e3, duration=None, ncycles=10, samplespercycle=1000, fs=None):

    if not (bool(samplespercycle) ^ bool(fs)):
        raise Exception('Must give either samplespercycle, or sampling frequency (fs), and not both')
    if not (bool(freq) ^ bool(duration)):
        raise Exception('Must give either freq or duration, and not both')

    if fs is None:
        fs = freq * samplespercycle
    if freq is None:
        freq = 1. / duration

    pico_capture(ch=ch, freq=fs, duration=ncycles/freq)

    pulse_builtin(freq=freq, amp=amp, shape=shape, n=ncycles)

    data = get_data(ch)

    return data

def pulse_and_capture(waveform, ch=['A', 'B'], fs=1e6, duration=1e-3, n=1, interpwfm=True, **kwargs):
    '''
    Send n pulses of the input waveform and capture on specified channels of picoscope.
    Duration determines the length of one repetition of waveform.
    '''

    # Set up to capture for n times the duration of the pulse
    # TODO have separate arguments for pulse duration and frequency, sampling frequency, number of samples per pulse
    pico_capture(ch, freq=fs, duration=n*duration, **kwargs)
    # Pulse the waveform n times, this will trigger the picoscope capture.
    pulse(waveform, duration, n=n, interp=interpwfm)

    data = get_data(ch)

    return data

def freq_response(ch='A', fstart=10, fend=1e8, n=10, amp=.3, offset=0):
    ''' Apply a series of sine waves with rigol, and sample the response on picoscope. Return data without analysis.'''
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
        pico_capture(ch, freq=fs, duration=duration, pretrig=0, triglevel=.05)
        pulse_builtin(freq=freq, amp=amp, offset=offset, shape='SIN', n=npulses, ch=1)
        d = get_data(ch)
        d['ncycles'] = ncycles
        data.append(d)
        # Probably not necessary but makes me feel good
        time.sleep(.1)

        # TODO: make some plots that show when debug=True is passed

    return data


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


def tripulse(n=1, v1=1.0, v2=-1.0, duration=None, rate=None):
    '''
    Generate n bipolar triangle pulses.
    Voltage sweep rate will  be constant.
    Trigger immediately
    '''

    rate, duration = _rate_duration(v1, v2, rate, duration)

    wfm = tri(v1, v2)

    pulse(wfm, duration, n=n)


def sinpulse(n=1, vmax=1.0, vmin=-1.0, duration=None):
    '''
    Generate n sine pulses.
    Trigger immediately
    If you pass vmin != -vmax, will not start at zero!
    '''

    wfm = (vmax - vmin) / 2 * np.sin(np.linspace(0, 2*pi, ps.AWGMaxSamples)) + ((vmax + vmin) / 2)

    pulse(wfm, duration, n=n)


def tri(v1, v2, n=None, step=None):
    '''
    Create a triangle pulse waveform with a constant sweep rate.  Starts and ends at zero.

    Can optionally pass number of data points you want, or the voltage step between points.

    If neither n or step is passed, return the shortest waveform which reaches v1 and v2.
    '''
    if n is not None:
        dv = abs(v1) + abs(v2 - v1) + abs(v2)
        step = dv / n
    if step is not None:
        # I could choose here to either make sure the waveform surely contains v1 and v2
        # or to make sure the waveform really has a constant sweep rate
        # I choose the former..
        def sign(num):
            npsign = np.sign(num)
            return npsign if npsign !=0 else 1
        wfm = np.concatenate((np.arange(0, v1, sign(v1) * step),
                             np.arange(v1, v2, sign(v2 - v1) * step),
                             np.arange(v2, 0, -sign(v2) * step),
                             [0]))
        return wfm
    else:
        # Find the shortest waveform that truly reaches v1 and v2 with constant sweep rate
        # Don't think we need better than 1 mV resolution
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


def set_compliance(cc_value):
    '''
    Use two analog outputs to set the compliance current and compensate input offset.
    Right now we use static lookup tables for compliance and compensation values.
    '''
    global COMPLIANCE_CURRENT, INPUT_OFFSET
    if cc_value > 1e-3:
        raise Exception('Compliance value out of range! Max 1 mA.')
    fn = 'c:/t/py-ivtools/compliance_calibration.pkl'
    print('Reading calibration from file {}'.format(os.path.abspath(fn)))
    with open(fn, 'rb') as f:
        cc = pickle.load(f)
    DAC0 = round(np.interp(cc_value, cc['ccurrent'], cc['dacvals']))
    DAC1 = np.interp(DAC0, cc['dacvals'], cc['compensationV'])
    print('Setting compliance to {} A'.format(cc_value))
    analog_out(0, dacval=DAC0)
    analog_out(1, volts=DAC1)
    COMPLIANCE_CURRENT = cc_value
    INPUT_OFFSET = 0

def calibrate_compliance(iterations=3, startfromfile=True, ndacvals=40):
    '''
    Set and measure some compliance values throughout the range, and save a calibration look up table
    Need picoscope channel B connected to circuit output
    and picoscope channel A connected to circuit input (through needles or smallish resistor is fine)
    This takes some time..
    '''
    # Measure compliance currents and input offsets with static Vb

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ccurrent_list = []
    offsets_list = []
    dacvals = np.int16(linspace(0, 2**11, ndacvals))

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
            analog_out(1, volts=cv)
            analog_out(0, v)
            time.sleep(.1)
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
        plt.pause(.1)
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
    global COMPLIANCE_CURRENT
    global INPUT_OFFSET

    # Put AWG in hi-Z mode (output channel off)
    # Then current at compliance circuit input has to be ~zero
    # (except for CHA scope input, this assumes it is set to 1Mohm, not 50ohm)
    ps.setChannel('A', 'DC', 50e-3, 1, 0)
    rigol_outputstate(False)
    time.sleep(.1)
    # Immediately capture some samples on channels A and B
    # Use these channel settings for the capture -- does not modify global settings
    picosettings = {'chrange': {'A':.2, 'B':2},
                    'choffset': {'A':0, 'B':-2},
                    'chatten': {'A':.2, 'B':1},
                    'chcoupling': {'A':'DC', 'B':'DC'}}
    pico_capture(['A', 'B'], freq=1e5, duration=1e-1, timeout_ms=1, **picosettings)
    picodata = get_data(['A', 'B'])
    #plot_channels(picodata)
    Amean = np.mean(picodata['A'])
    Bmean = np.mean(picodata['B'])

    # Channel A should be connected to the rigol output and to the compliance circuit input, perhaps through a resistance.
    INPUT_OFFSET = Amean
    print('Measured voltage offset of compliance circuit input: {}'.format(Amean))

    # Channel B should be measuring the circuit output with the entire compliance current across the output resistance.

    # Circuit parameters
    gain = 1
    R = 2e3
    # Seems rigol doesn't like to pulse zero volts. It makes a beep but then apparently does it anyway.
    #Vout = pulse_and_capture(waveform=np.zeros(100), ch='B', fs=1e6, duration=1e-3)
    ccurrent = Bmean / (R * gain)
    COMPLIANCE_CURRENT = ccurrent
    print('Measured compliance current: {} A'.format(ccurrent))

    return (ccurrent, Amean)

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

# For compliance amp
def ccircuit_to_iv(datain, dtype=np.float32):
    ''' Convert picoscope channel data to IV dict'''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    global COMPLIANCE_CURRENT
    global INPUT_OFFSET

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    B = datain['B']
    #C = datain['C']
    gain = 1
    # Common base resistor
    R = 2e3
    dataout['V'] = A - dtype(INPUT_OFFSET)
    #dataout['V_formula'] = 'CHA - INPUT_OFFSET'
    dataout['INPUT_OFFSET'] = INPUT_OFFSET
    #dataout['I'] = 1e3 * (B - C) / R
    # Current circuit has 0V output in compliance, and positive output under compliance
    # Unless you know the compliance value, you can't get to current, because you don't know the offset
    dataout['I'] = -1 * B / dtype(R * gain) + dtype(COMPLIANCE_CURRENT)
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    dataout['units'] = {'V':'V', 'I':'A'}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
    dataout['CC'] = COMPLIANCE_CURRENT
    dataout['gain'] = gain * R
    return dataout

# For Rehan current amp
def rehan_to_iv(datain, dtype=np.float32):
    ''' Convert picoscope channel data to IV dict'''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    # Volts per amp
    gainC = 524
    # I have no idea why this one is off by a factor of two.  I did exactly the same thing to measure it ..
    gainD = 11151 / 2
    # 1 Meg, 33,000

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    C = datain['C']

    dataout['V'] = A
    dataout['I'] = C / gainC
    dataout['units'] = {'V':'V', 'I':'A'}
    dataout['Cgain'] = gainC

    if 'D' in datain:
        D = datain['D']
        dataout['I2'] = D / gainD
        dataout['Dgain'] = gainD
        dataout['units'].update({'I2':'A'})

    return dataout

# Change this when you change probing circuits
#pico_to_iv = rehan_to_iv
pico_to_iv = ccircuit_to_iv

def measure_dc_gain(Vin=1, ch='C', R=10e3):
    # Measure dc gain of rehan amplifier
    # Apply voltage
    print('Outputting {} volts on Rigol CH1'.format(Vin))
    pulse(np.repeat(Vin, 100), 1e-3)
    time.sleep(1)
    # Measure output
    measurechannels = ['A', ch]
    pico_capture(measurechannels, freq=1e6, duration=1, timeout_ms=1)
    time.sleep(.1)
    chdata = get_data(measurechannels)
    plot.plot_channels(chdata)
    chvalue = np.mean(chdata[ch])
    print('Measured {} volts on picoscope channel {}'.format(chvalue, ch))

    gain = R * chvalue / Vin
    # Set output back to zero
    pulse([Vin, 0,0,0,0], 1e-3)
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
    sinwave = outamp * sin(linspace(0, 1, 2**12)*2*pi)
    chs = ['A', ch]
    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1, chrange=RANGE, choffset=OFFSET)
    data = get_data(chs)
    plot.plot_channels(data)

    # will change the range and offset after all
    squeeze_range(data, [ch])

    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1000)
    data = get_data(chs)

    plot.plot_channels(data)

    return max(abs(fft.fft(data[ch]))[1:-1]) / max(abs(fft.fft(data['A']))[1:-1]) * R

