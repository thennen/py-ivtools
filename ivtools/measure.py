""" Functions for measuring IV data with picoscope 6403C and Rigol AWG """

from picoscope import ps6000
import visa
from fractions import Fraction
from math import gcd
import numpy as np
import time
from dotdict import dotdict
import pandas as pd

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

# TODO make a container class for all the pico settings, so that they are aware of each other and enforce valid values.  Can build in some fancy methods.  Could split it to another file.  Could submit PR to pico-python.
#class picosettings(dict):

class picorange(dict):
    # Holds the values for picoscope channel ranges.  Enforces valid values.
    # TODO: add increment and decrement
    def __init__(self):
        self.possible_ranges = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
        self.setall(1.0)

    def set(self, channel, value):
        if value in self.possible_ranges:
            self[channel] = value
        else:
            argclosest = argmin([abs(p - value) for p in self.possible_ranges])
            closest = self.possible_ranges[argclosest]
            print('{} is an impossible range setting. Using closest valid setting {}.'.format(value, closest))
            self[channel] = closest

    def setall(self, value=1.0):
        # Set all the channel ranges to this value
        self.set('A', value)
        self.set('B', value)
        self.set('C', value)
        self.set('D', value)

    @property
    def a(self):
        print(self['A'])
    @a.setter
    def a(self, value):
        self.set('A', value)
    @property
    def b(self):
        print(self['B'])
    @b.setter
    def b(self, value):
        self.set('B', value)
    @property
    def c(self):
        print(self['C'])
    @c.setter
    def c(self, value):
        self.set('C', value)
    @property
    def d(self):
        print(self['D'])
    @d.setter
    def d(self, value):
        self.set('D', value)

class picooffset(dict):
    # picooffset needs to be aware of the range setting in order to determine valid values
    def __init__(self, picorange):
        self._picorange = picorange
        self.possible_ranges = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
        self.max_offsets = (.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20)
        self.setall(0.0)

    def set(self, channel, value):
        # Assuming that _picorange has a valid range value for this channel
        channelrange = self._picorange[channel]
        maxoffset = self.max_offsets[self.possible_ranges.index(channelrange)]
        if abs(value) < maxoffset:
            self[channel] = value
        else:
            clippedvalue = np.sign(value) * maxoffset
            print(('{} is above the maximum offset for channel {} with range {} V. '
                   'Setting offset to {}.').format(value, channel, channelrange, clippedvalue))
            self[channel] = clippedvalue

    def setall(self, value=0.0):
        # Set all the channel ranges to this value
        self.set('A', value)
        self.set('B', value)
        self.set('C', value)
        self.set('D', value)

    @property
    def a(self):
        print(self['A'])
    @a.setter
    def a(self, value):
        self.set('A', value)
    @property
    def b(self):
        print(self['B'])
    @b.setter
    def b(self, value):
        self.set('B', value)
    @property
    def c(self):
        print(self['C'])
    @c.setter
    def c(self, value):
        self.set('C', value)
    @property
    def d(self):
        print(self['D'])
    @d.setter
    def d(self, value):
        self.set('D', value)

def best_range(data):
    '''
    Return the best RANGE and OFFSET values to use for a particular input signal (array)
    Just uses minimim and maximum values of the signal, therefore you could just pass (min, max), too
    Don't pass int8 signals, would then need channel information to convert to V
    '''
    possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
    # Sadly, each range has a different maximum possible offset
    max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))
    minimum = np.min(data)
    maximum = np.max(data)
    amplitude = abs(maximum - minimum) / 2
    middle = round((maximum + minimum) / 2, 3)
    # Mask of possible ranges that fit the signal
    mask = possible_ranges >= amplitude
    for selectedrange, max_offset in zip(possible_ranges[mask], max_offsets[mask]):
        # Is middle an acceptable offset?
        if middle < max_offset:
            return (selectedrange, -middle)
            break
        # Can we reduce the offset without the signal going out of range?
        elif (max_offset + selectedrange >= maximum) and (-max_offset - selectedrange <= minimum):
            return(selectedrange, np.clip(-middle, -max_offset, max_offset))
            break
        # Neither worked, try increasing the range ...
    # If no range was enough to fit the signal
    print('Signal out of pico range!')
    return (max(possible_ranges), 0)

def squeeze_range(data, ch=['A', 'B', 'C', 'D']):
    '''
    Find the best range for given input data (can be any number of channels)
    Set the range and offset to the lowest required to fit the data
    '''
    global RANGE, OFFSET
    for c in ch:
        if c in data:
            if type(data[c][0]) is np.int8:
                # Need to convert to float
                usedrange = data['RANGE'][c]
                usedoffset = data['OFFSET'][c]
                maximum = np.max(data[c])  / 2**8 * usedrange * 2 - usedoffset
                minimum = np.min(data[c]) / 2**8 * usedrange * 2 - usedoffset
                rang, offs = best_range((minimum, maximum))
            else:
                rang, offs = best_range(data[c])
            RANGE[c] = rang
            OFFSET[c] = offs

# Settings for picoscope channels.
# Also don't clobber them
try:
    COUPLINGS
    ATTENUATION
    OFFSET
    RANGE
    COMPLIANCE_CURRENT
    INPUT_OFFSET
except:
    COUPLINGS = {'A': 'DC', 'B': 'DC', 'C': 'DC', 'D': 'DC'}
    ATTENUATION = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
    #RANGE = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
    #OFFSET = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    RANGE = picorange()
    OFFSET = picooffset(RANGE)
    COMPLIANCE_CURRENT = 0
    INPUT_OFFSET = 0


# For reference
possible_ranges = pd.DataFrame(dict(Range=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                                    Max_offset=(.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20)))

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
            rigol = visa.ResourceManager().open_resource(rigolstr)
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


def connect_instruments():
    ''' Connect all the necessary equipment '''
    print('Attempting to connect all instruments.')
    connect_picoscope()
    connect_rigolawg()

def pico_capture(ch='A', freq=1e6, duration=0.04, nsamples=None,
                 trigsource='TriggerAux', triglevel=0.5, timeout_ms=30000, pretrig=0.0):
    '''
    Set up picoscope to capture from specified channel(s).
    Won't actually do it until it receives the specified trigger event.
    Datasheet says it will trigger automatically after a timeout.
    ch can be a list of characters, i.e. ch=['A','B'].
    # TODO: provide a way to override the global variable channel settings
    '''

    # If ch not iterable, just put it in a list by itself
    if not hasattr(ch, '__iter__'):
        ch = [ch]

    # Maximum sample rate is different depending on the number of channels that are enabled.
    # Therefore, if you want the highest possible rate, you should keep unused channels disabled.
    # Enable only the channels being used, disable the rest
    existing_channels = COUPLINGS.keys()
    for c in existing_channels:
        if c in ch:
            ps.setChannel(c, enabled=True)
        else:
            ps.setChannel(c, enabled=False)

    # This will return actual sample frequency, then we can determine
    # the number of samples needed.
    freq, _ = ps.setSamplingFrequency(freq, 0)
    if nsamples is None:
        nsamples = duration * freq
    freq, max_samples = ps.setSamplingFrequency(freq, nsamples)
    print('Actual picoscope sampling frequency: {:,}'.format(freq))
    if nsamples > max_samples:
        raise(Exception('Trying to sample more than picoscope memory capacity'))
    # Set up the channels
    for c in ch:
        ps.setChannel(c,
                      COUPLINGS[c],
                      RANGE[c],
                      probeAttenuation=ATTENUATION[c],
                      VOffset=OFFSET[c])
    # Set up the trigger.  Will timeout in 30s
    ps.setSimpleTrigger(trigsource, triglevel, timeout_ms=timeout_ms)
    ps.runBlock(pretrig)
    return freq


def load_volatile_wfm(waveform, duration, n=1, ch=1, interp=True):
    '''
    Load waveform into volatile memory, but don't trigger
    '''
    if len(waveform) > 512e3:
        raise Exception('Too many samples requested for rigol AWG (probably?)')

    # Rigol uses stupid SCPI commands.
    # Construct a string out of the waveform
    waveform = np.array(waveform, dtype=np.float32)
    maxamp = np.max(np.abs(waveform))
    if maxamp != 0:
        normwaveform = waveform/maxamp
    else:
        # Not a valid waveform anyway .. rigol will beep
        normwaveform = waveform
    wfm_str = ','.join([str(w) for w in normwaveform])
    freq = 1 / duration

    # toggling output state is slow, clunky, annoying, and should not be necessary.
    # it might also cause some spikes that could damage the device.
    # Also goes into high impedance output which could have some undesirable consequences.
    # Problem is that the command which loads in a volatile waveform switches rigol
    # out of burst mode automatically.  If the output is still enabled, you will get a
    # continuous pulse train until you can get back into burst mode.
    # contacted RIGOL about the problem but they did not help.  Seems there is no way around it.
    rigol.write(':OUTPUT:STATE OFF')
    #
    # Turn off screen saver.  It sends a premature pulse on SYNC if on..  Really dumb.
    rigol.write(':DISP:SAV OFF')
    time.sleep(.01)
    # Manual says you can change output resistance from 1 to 10k
    #rigol.write('OUTPUT{}:IMPEDANCE 50'.format(ch))
    # Can turn on/off the sync output (on rear)
    #rigol.write('OUTPUT{}:SYNC ON')
    if interp==True:
        # Turn on interpolation for IVs
        rigol.write(':DATA:POIN:INT LIN')
    elif interp==False:
        # Turn off interpolation for steps
        rigol.write(':DATA:POIN:INT OFF')
    # This command switches out of burst mode for some stupid reason
    rigol.write(':TRAC:DATA VOLATILE,{}'.format(wfm_str))
    rigol.write(':SOURCE{}:FREQ:FIX {}'.format(ch, freq))
    rigol.write(':SOURCE{}:VOLTAGE:AMPL {}'.format(ch, 2*maxamp))
    rigol.write(':SOURCE{}:BURST:MODE TRIG'.format(ch))
    rigol.write(':SOURCE{}:BURST:NCYCLES {}'.format(ch, n))
    rigol.write(':SOURCE{}:BURST:TRIG:SOURCE MAN'.format(ch))
    rigol.write(':SOURCE{}:BURST:STATE ON'.format(ch))
    rigol.write(':OUTPUT{}:STATE ON'.format(ch))
    # Enable screensaver again because it makes me feel good
    #rigol.write(':DISP:SAV ON')

def trigger_rigol(ch=1):
    ''' Send signal to rigol to trigger immediately'''
    rigol.write(':SOURCE{}:BURST:TRIG IMM'.format(ch))


def pulse(waveform, duration, n=1, ch=1, interp=True):
    '''
    Generate n pulses of the input waveform on Rigol AWG.
    Trigger immediately.
    Manual says you can use up to 128 Mpts, ~2^27, but for some reason you can't.
    Another part of the manual says it is limited to 512 kpts, but can't seem to do that either.
    '''
    # Load waveform
    load_volatile_wfm(waveform, duration, n, ch, interp)
    # Trigger rigol
    trigger_rigol(ch=1)


def get_data(ch='A', raw=False, dtype=np.float32):
    '''
    Wait for data and transfer it from pico memory.
    ch can be a list of channels
    This function returns a simple dict of the arrays and metadata.
    Use pico_to_iv to convert to current, voltage, different data structure.

    if raw is True, do not convert from ADC value - this saves a lot of memory
    return dict of arrays and metadata (sample rate, channel settings, time...)

    '''
    data = dict()
    # Wait for data
    while(not ps.isReady()):
        time.sleep(0.01)

    if not hasattr(ch, '__iter__'):
        ch = [ch]
    for c in ch:
        if raw:
            # For some reason pico-python gives the values as int16
            # Probably because some scopes have 16 bit resolution
            # The 6403c is only 8 bit, and I'm looking to save memory here
            rawint16, _, _ = ps.getDataRaw(c)
            data[c] = np.int8(rawint16 / 2**8)
        else:
            # I added dtype argument to pico-python
            data[c] = ps.getDataV(c, dtype=dtype)

    Channels = ['A', 'B', 'C', 'D']
    data['RANGE'] = {ch:chr for ch, chr in zip(Channels, ps.CHRange)}
    data['OFFSET'] = {ch:cho for ch, cho in zip(Channels, ps.CHOffset)}
    data['sample_rate'] = ps.sampleRate
    # Specify captured, because this field will persist even after splitting for example
    # Then if you split 100,000 samples into 10 x 10,000 having nsamples = 100,000 will be confusing
    data['nsamples_capture'] = len(data[ch[0]])
    # Using the current state of the global variables to record what settings were used
    # I don't know a way to get couplings and attenuations from the picoscope instance
    data['COUPLINGS'] = COUPLINGS
    data['ATTENUATION'] = ATTENUATION
    # Sample frequency?
    return data


def close():
    global ps
    global rigol
    # Close connection to pico
    ps.close()
    ps = None
    # Close connection to rigol
    rigol.close()
    rigol = None


def pulse_and_capture(waveform, ch=['A', 'B'], fs=1e6, duration=1e-3, n=1):
    '''
    Send n pulses of the input waveform and capture on specified channels of picoscope.
    Duration determines the length of one repetition of waveform.
    '''

    # Set up to capture for n times the duration of the pulse
    pico_capture(ch, freq=fs, duration=n*duration)
    # Pulse the waveform n times, this will trigger the picoscope capture.
    pulse(waveform, duration, n=n)

    data = get_data(ch)

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


def tri(v1, v2):
    '''
    Calculate a triangle pulse waveform.

    This is a slightly tricky because the AWG takes equally spaced samples,
    so finding the shortest waveform that truly reaches v1 and v2 with
    constant sweep rate involves finding the greatest common divisor of
    v1 and v2.
    '''
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
    fn = 'compliance_calibration.pkl'
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

def calibrate_compliance(iterations=3, startfromfile=True):
    '''
    Set and measure some compliance values throughout the range, and save a calibration look up table
    Need picoscope channel B connected to circuit output
    and picoscope channel A connected to circuit input (through needles or smallish resistor is fine)
    This takes some time..
    '''
    # Measure compliance currents and input offsets with static Vb
    # Have to change the range.  I'll change it back ...
    global RANGE
    global OFFSET
    oldrange = RANGE.copy()
    oldoffset = OFFSET.copy()
    RANGE['A'] = 1
    OFFSET['A'] = 0
    #RANGE['B'] = 5
    RANGE['B'] = 2
    OFFSET['B'] = -2

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ccurrent_list = []
    offsets_list = []
    dacvals = range(0, 2**11, 40)
    for it in range(iterations):
        ccurrent = []
        offsets = []
        if len(offsets_list) == 0:
            if startfromfile:
                fn = 'compliance_calibration.pkl'
                print('Reading calibration from file {}'.format(os.path.abspath(fn)))
                with open(fn, 'rb') as f:
                    cc = pickle.load(f)
                compensations = cc['compensationV']
            else:
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
        ax1.plot(dacvals, ccurrent, '.-')
        ax1.set_xlabel('DAC0 value')
        ax1.set_ylabel('Compliance Current')
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

    # Set scope settings back to old values
    RANGE['A'] = oldrange['A']
    OFFSET['A'] = oldoffset['A']
    RANGE['B'] = oldrange['B']
    OFFSET['B'] = oldoffset['B']

    return compensations

def plot_compliance_calibration():
    fn = 'compliance_calibration.pkl'
    print('Reading calibration from file {}'.format(os.path.abspath(fn)))
    with open(fn, 'rb') as f:
        cc = pickle.load(f)
    ccurrent = cc['ccurrent']
    dacvals = cc['dacvals']
    compensationV = cc['compensationV']
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.plot(dacvals, ccurrent, '.-')
    ax.set_xlabel('DAC0 value')
    ax.set_ylabel('Compliance Current')
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
    rigol.write(':OUTPUT:STATE OFF')
    time.sleep(.1)
    # Immediately capture some samples on channels A and B
    pico_capture(['A', 'B'], freq=1e5, duration=1e-1, timeout_ms=1)
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

    # dotdict was nice, but caused too many problems ...
    #dataout = dotdict(datain)
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
    gainC = 1050
    gainD = 12867
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
    plot_channels(chdata)
    chvalue = np.mean(chdata[ch])
    print('Measured {} volts on picoscope channel {}'.format(chvalue, ch))

    gain = R * chvalue / Vin
    # Set output back to zero
    pulse([Vin, 0,0,0,0], 1e-3)
    return gain

def analog_out(ch, dacval=None, volts=None):
    '''
    I found a USB-1208HS so this is how you use it I guess.
    Pass a digital value between 0 and 2**12 - 1
    0 is -10V, 2**12 - 1 is 10V
    Voltage values that don't make sense for my current set up are disallowed.
    '''
    # Import here because I don't want the entire module to error if you don't have mcculw installed
    from mcculw import ul
    from mcculw.enums import ULRange
    from mcculw.ul import ULError
    board_num = 0
    ao_range = ULRange.BIP10VOLTS

    # Can pass dacval or volts.  Prefer dacval.
    if dacval is None:
        # Better have passed volts...
        dacval = ul.from_eng_units(board_num, ao_range, volts)
    else:
        volts = ul.to_eng_units(board_num, ao_range, dacval)

    # Just protect against doing something that doesn't make sense
    if ch == 0 and volts > 0:
        print('I disallow voltage value {} for analog output {}'.format(volts, ch))
        return
    elif ch == 1 and volts < 0:
        print('I disallow voltage value {} for analog output {}'.format(volts, ch))
        return
    else:
        print('Setting analog out {} to {} ({} V)'.format(ch, dacval, volts))

    try:
        ul.a_out(board_num, ch, ao_range, dacval)
    except ULError as e:
        # Display the error
        print("A UL error occurred. Code: " + str(e.errorcode)
            + " Message: " + e.message)


def digital_out(ch, val):
    # Import here because I don't want the entire module to error if you don't have mcculw installed
    from mcculw import ul
    from mcculw.enums import DigitalPortType, DigitalIODirection
    from mcculw.ul import ULError
    #ul.d_config_port(0, DigitalPortType.AUXPORT, DigitalIODirection.OUT)
    ul.d_config_bit(0, DigitalPortType.AUXPORT, 8, DigitalIODirection.OUT)
    ul.d_bit_out(0, DigitalPortType.AUXPORT, ch, val)

