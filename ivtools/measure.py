""" Functions for measuring IV data with picoscope 6403C and Rigol AWG """

from picoscope import ps6000
import visa
from fractions import Fraction
from math import gcd
import numpy as np
import time
from dotdict import dotdict
import pandas as pd
import os

visa_rm = visa.ResourceManager()

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

# TODO make a container class for all the pico settings, so that they are aware of each other and enforce valid values.  Can build in some fancy methods.  Could split it to another file.  Could submit PR to pico-python.
#class picosettings(dict):

# NOT DONE
class picosettings(ps6000.PS6000):
    '''
    Class for managing all the channel settings for picoscope
    range, offset, couplings, attenuations
    Makes sure the settings are valid, and provides some methods for convenience
    don't know if this is a good idea

    Why use a class?
    don't like all the globals
    want a better syntax for changing channel settings, since I have to do it a lot
    code won't need to change much
    want functions for doing things to the settings, without dumping them all over the interactive namespace

    disadvantages:
    loss of simplicity
    wanted a banana but got a gorilla holding the banana and the whole jungle

    '''
    def __init__(self):
        self.possible_ranges = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


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
    try:
        # Is keithley already connected?
        idn = keithley.ask('*IDN?')
    except:
        # impatiently try to connect to keithley
        keithley = visa_rm.get_instrument(Keithley_id, open_timeout=250)
        idn = keithley.ask('*IDN?')
    print('Keithley *IDN?: {}'.format(idn))


def connect_instruments():
    ''' Connect all the necessary equipment '''
    print('Attempting to connect all instruments.')
    connect_picoscope()
    connect_rigolawg()
    connect_keithley()

def pico_capture(ch='A', freq=None, duration=None, nsamples=None,
                 trigsource='TriggerAux', triglevel=0.1, timeout_ms=30000, pretrig=0.0,
                 chrange=None, choffset=None, chcoupling=None, chatten=None):
    '''
    Set up picoscope to capture from specified channel(s).

    pass exactly two of: freq(sampling frequency), duration, nsamples
    sampling frequency has limited possible values, so actual number of samples will vary
    will try to sample for the intended duration, either the value of the duration argument or nsamples/freq

    Won't actually start capture until picoscope receives the specified trigger event.

    It will trigger automatically after a timeout.

    ch can be a list of characters, i.e. ch=['A','B'].

    # TODO: provide a way to override the global variable channel settings
    if any of chrange, choffset, chcouplings, chattenuation (dicts) are not passed,
    the settings will be taken from the global variables
    '''

    # Check that two of freq, duration, nsamples was passed
    if not sum([x is None for x in (freq, duration, nsamples)]) == 1:
        raise Exception('Must give exactly two of the arguments freq, duration, and nsamples.  These are needed to determine sampling conditions.')

    # If ch not iterable, just put it in a list by itself
    if not hasattr(ch, '__iter__'):
        ch = [ch]

    # Maximum sample rate is different depending on the number of channels that are enabled.
    # Therefore, if you want the highest possible rate, you should keep unused channels disabled.
    # Enable only the channels being used, disable the rest
    for c in ['A', 'B', 'C', 'D']:
        if c not in ch:
            ps.setChannel(c, enabled=False)

    # If freq and duration are passed, take as many samples as it takes to actually sample for duration
    # If duration and nsamples are passed, sample with frequency as near as possible to nsamples/duration (nsamples will vary)
    # If freq and nsamples are passed, sample at closest possible frequency for nsamples (duration will vary)
    if freq is None:
        freq = nsamples / duration
    # This will return actual sample frequency, then we can determine
    # the number of samples needed.
    actualfreq, _ = ps.setSamplingFrequency(freq, 0)

    if duration is not None:
        nsamples = duration * actualfreq

    def global_replace(kwarg, globalarg):
        if kwarg is None:
            # No values passed, use the global values
            return globalarg
        else:
            # Fill missing values with global values
            kwargcopy = kwarg.copy()
            for c in ch:
                if c not in kwargcopy:
                    kwargcopy[c] = globalarg[c]
            return kwargcopy

    chrange = global_replace(chrange, RANGE)
    choffset = global_replace(choffset, OFFSET)
    chcoupling = global_replace(chcoupling, COUPLINGS)
    chatten = global_replace(chatten, ATTENUATION)

    actualfreq, max_samples = ps.setSamplingFrequency(actualfreq, nsamples)
    print('Actual picoscope sampling frequency: {:,}'.format(actualfreq))
    if nsamples > max_samples:
        raise(Exception('Trying to sample more than picoscope memory capacity'))
    # Set up the channels
    for c in ch:
        ps.setChannel(channel=c,
                      coupling=chcoupling[c],
                      VRange=chrange[c],
                      probeAttenuation=chatten[c],
                      VOffset=choffset[c],
                      enabled=True)
    # Set up the trigger.  Will timeout in 30s
    ps.setSimpleTrigger(trigsource, triglevel, timeout_ms=timeout_ms)
    ps.runBlock(pretrig)
    return actualfreq

### These directly wrap SCPI commands that can be sent to the rigol AWG
# TODO: turn these into a class, since the code will not change very often
# There is at least one python library for DG5000, but I could not get it to run.

def rigol_shape(shape='SIN', ch=1):
    '''
    Change the waveform shape to a built-in value. Possible values are:
    SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|GAUSsian |HAVersine|LORentz|ARBPULSE|DUAltone
    '''
    rigol.write('SOURCE{}:FUNC:SHAPE {}'.format(ch, shape))

def rigol_outputstate(state=True, ch=1):
    ''' Turn output state on or off '''
    statestr = 'ON' if state else 'OFF'
    rigol.write(':OUTPUT{}:STATE {}'.format(ch, statestr))

def rigol_frequency(freq, ch=1):
    ''' Set frequency of AWG waveform.  Not the sample rate! '''
    rigol.write(':SOURCE{}:FREQ:FIX {}'.format(ch, freq))

def rigol_amplitude(amp, ch=1):
    ''' Set amplitude of AWG waveform '''
    rigol.write(':SOURCE{}:VOLTAGE:AMPL {}'.format(ch, amp))

def rigol_offset(offset, ch=1):
    ''' Set offset of AWG waveform '''
    rigol.write(':SOURCE{}:VOLT:OFFS {}'.format(ch, offset))

def rigol_output_resistance(r=50, ch=1):
    ''' Manual says you can change output resistance from 1 to 10k ''' 
    rigol.write('OUTPUT{}:IMPEDANCE {}'.format(ch, r))

def rigol_sync(state=True):
    ''' Can turn on/off the sync output (on rear) '''
    statestr = 'ON' if state else 'OFF'
    rigol.write('OUTPUT{}:SYNC ' + statestr)

def rigol_screensaver(state=False):
    ''' Turn the screensaver on or off.  Screensaver causes problems with triggering because DG5000 is a piece of junk. '''
    statestr = 'ON' if state else 'OFF'
    rigol.write(':DISP:SAV ' + statestr)

def rigol_ramp_symmetry(percent=50, ch=1):
    ''' Change the symmetry of a ramp output. Refers to the sweep rates of increasing/decreasing ramps. '''
    rigol.write('SOURCE{}:FUNC:RAMP:SYMM {}'.format(ch, percent))

def rigol_dutycycle(percent=50, ch=1):
    ''' Change the duty cycle of a square output. '''
    rigol.write('SOURCE{}:FUNC:SQUare:DCYCle {}'.format(ch, percent))

def rigol_error():
    ''' Get error message from rigol '''
    return rigol.ask(':SYSTem:ERRor?')

# <<<<< For burst mode
def rigol_ncycles(n, ch=1):
    ''' Set number of cycles that will be output in burst mode '''
    rigol.write(':SOURCE{}:BURST:NCYCLES {}'.format(ch, n))

def rigol_trigsource(source='MAN', ch=1):
    ''' Change trigger source for burst mode. INTernal|EXTernal|MANual '''
    rigol.write(':SOURCE{}:BURST:TRIG:SOURCE {}'.format(ch, source))

def rigol_trigger(ch=1):
    '''
    Send signal to rigol to trigger immediately.  Make sure that trigsource is set to MAN:
    rigol_trigsource('MAN')
    '''
    rigol.write(':SOURCE{}:BURST:TRIG IMM'.format(ch))

def rigol_burstmode(mode='TRIG', ch=1):
    '''Set the burst mode.  I don't know what it means. 'TRIGgered|GATed|INFinity'''
    rigol.write(':SOURCE{}:BURST:MODE {}'.format(ch, mode))

def rigol_burst(state=True, ch=1):
    ''' Turn the burst mode on or off '''
    statestr = 'ON' if state else 'OFF'
    rigol.write(':SOURCE{}:BURST:STATE {}'.format(ch, statestr))

# End for burst mode >>>>>
def rigol_load_wfm(waveform):
    '''
    Load some data as an arbitrary waveform to be output.
    Data will be normalized.  Use rigol_amplitude to set the amplitude.
    Make sure that the output is off, because the command switches out of burst mode and will start outputting immediately.
    '''
    # It seems to be possible to send bytes to the rigol instead of strings.  This would be much better.
    # But I haven't been able to figure out how to convert the data to the required format.  It's complicated.
    # Construct a string out of the waveform
    waveform = np.array(waveform, dtype=np.float32)
    maxamp = np.max(np.abs(waveform))
    if maxamp != 0:
        normwaveform = waveform/maxamp
    else:
        # Not a valid waveform anyway .. rigol will beep
        normwaveform = waveform
    wfm_str = ','.join([str(w) for w in normwaveform])
    # This command switches out of burst mode for some stupid reason
    rigol.write(':TRAC:DATA VOLATILE,{}'.format(wfm_str))

def rigol_interp(interp=True):
    ''' Set AWG datapoint interpolation mode '''
    modestr = 'LIN' if interp else 'OFF'
    rigol.write('TRACe:DATA:POINts:INTerpolate {}'.format(modestr))

def rigol_color(c='RED'):
    '''
    Change the highlighting color on rigol screen for some reason
    'RED', 'DEEPRED', 'YELLOW', 'GREEN', 'AZURE', 'NAVYBLUE', 'BLUE', 'LILAC', 'PURPLE', 'ARGENT'
    '''
    rigol.write(':DISP:WIND:HLIG:COL {}'.format(c))


def load_volatile_wfm(waveform, duration, n=1, ch=1, interp=True):
    '''
    Load waveform into volatile memory, but don't trigger
    '''
    if len(waveform) > 512e3:
        raise Exception('Too many samples requested for rigol AWG (probably?)')

    # toggling output state is slow, clunky, annoying, and should not be necessary.
    # it might also cause some spikes that could damage the device.
    # Also goes into high impedance output which could have some undesirable consequences.
    # Problem is that the command which loads in a volatile waveform switches rigol
    # out of burst mode automatically.  If the output is still enabled, you will get a
    # continuous pulse train until you can get back into burst mode.
    # contacted RIGOL about the problem but they did not help.  Seems there is no way around it.
    rigol_outputstate(False, ch=ch)
    #
    # Turn off screen saver.  It sends a premature pulse on SYNC output if on.
    # This will make the scope trigger early and miss part or all of the pulse.  Really dumb.
    rigol_screensaver(False)
    #time.sleep(.01)
    # Turn on interpolation for IVs, off for steps
    rigol_interp(interp)
    # This command switches out of burst mode for some stupid reason
    rigol_load_wfm(waveform)
    freq = 1. / duration
    rigol_frequency(freq, ch=ch)
    maxamp = np.max(np.abs(waveform))
    rigol_amplitude(2*maxamp, ch=ch)
    rigol_burstmode('TRIG', ch=ch)
    rigol_ncycles(n, ch=ch)
    rigol_trigsource('MAN', ch=ch)
    rigol_burst(True, ch=ch)
    rigol_outputstate(True, ch=ch)


def load_builtin_wfm(shape='SIN', duration=None, freq=None, amp=1, offset=0, n=1, ch=1):
    '''
    Set up a built-in waveform to pulse n times
    SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|GAUSsian |HAVersine|LORentz|ARBPULSE|DUAltone
    '''

    if not (bool(duration) ^ bool(freq)):
        raise Exception('Must give either duration or frequency, and not both')

    if freq is None:
        freq = 1. / duration

    # Set up waveform
    rigol_burst(True, ch=ch)
    rigol_burstmode('TRIG', ch=ch)
    rigol_shape(shape, ch=ch)
    # Rigol's definition of amp is peak-to-peak, which is unusual.
    rigol_amplitude(2*amp, ch=ch)
    rigol_offset(offset, ch=ch)
    rigol_burstmode('TRIG', ch=ch)
    rigol_ncycles(n, ch=ch)
    rigol_trigsource('MAN', ch=ch)
    rigol_frequency(freq, ch=ch)

    return locals()


def pulse_builtin(shape='SIN', duration=None, freq=None, amp=1, offset=0, n=1, ch=1):
    '''
    Pulse a built-in waveform n times
    SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|GAUSsian |HAVersine|LORentz|ARBPULSE|DUAltone
    '''
    load_builtin_wfm(**locals())

    rigol_outputstate(True)
    # Trigger rigol
    rigol_trigger(ch=ch)


def pulse(waveform, duration, n=1, ch=1, interp=True):
    '''
    Generate n pulses of the input waveform on Rigol AWG.
    Trigger immediately.
    Manual says you can use up to 128 Mpts, ~2^27, but for some reason you can't.
    Another part of the manual says it is limited to 512 kpts, but can't seem to do that either.
    '''
    # Load waveform
    load_volatile_wfm(**locals())
    # Trigger rigol
    rigol_trigger(ch=1)


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
    data['ATTENUATION'] = {ch:cha for ch, cha in zip(Channels, ps.ProbeAttenuation)}
    data['sample_rate'] = ps.sampleRate
    # Specify samples captured, because this field will persist even after splitting for example
    # Then if you split 100,000 samples into 10 x 10,000 having nsamples = 100,000 will be confusing
    data['nsamples_capture'] = len(data[ch[0]])
    # Using the current state of the global variables to record what settings were used
    # I don't know a way to get couplings and attenuations from the picoscope instance
    # TODO: pull request a change to setChannel
    data['COUPLINGS'] = COUPLINGS
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

def pulse_and_capture(waveform, ch=['A', 'B'], fs=1e6, duration=1e-3, n=1, interpwfm=True):
    '''
    Send n pulses of the input waveform and capture on specified channels of picoscope.
    Duration determines the length of one repetition of waveform.
    '''

    # Set up to capture for n times the duration of the pulse
    # TODO have separate arguments for pulse duration and frequency, sampling frequency, number of samples per pulse
    pico_capture(ch, freq=fs, duration=n*duration)
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
    plot_channels(chdata)
    chvalue = np.mean(chdata[ch])
    print('Measured {} volts on picoscope channel {}'.format(chvalue, ch))

    gain = R * chvalue / Vin
    # Set output back to zero
    pulse([Vin, 0,0,0,0], 1e-3)
    return gain

def measure_ac_gain(R=1000, freq=1e4, ch='C', outamp=1):
    global RANGE
    global OFFSET
    oldrange = RANGE.copy()
    oldoffset = OFFSET.copy()

    # Send a test pulse to determine better range to use
    arange, aoffset = best_range([outamp, -outamp])
    RANGE['A'] = arange
    OFFSET['A'] = aoffset
    # Power supply is 5V, so this should cover the whole range
    RANGE[ch] = 5
    OFFSET[ch] = 0
    sinwave = outamp * sin(linspace(0, 1, 2**12)*2*pi)
    chs = ['A', ch]
    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1)
    data = get_data(chs)
    plot_channels(data)

    squeeze_range(data, [ch])

    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1000)
    data = get_data(chs)

    # Set scope settings back to old values
    RANGE['A'] = oldrange['A']
    OFFSET['A'] = oldoffset['A']
    RANGE[ch] = oldrange[ch]
    OFFSET[ch] = oldoffset[ch]

    plot_channels(data)

    return max(abs(fft.fft(data[ch]))[1:-1]) / max(abs(fft.fft(data['A']))[1:-1]) * R


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
