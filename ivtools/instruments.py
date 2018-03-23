'''
These classes contain functionality specific to only one instrument.
Don't put code in an instrument class that has anything to do with a different instrument,
or any particular application!

They are grouped into classes because there may be some overlapping functions/names which should be contained
Should only put instruments here that have an actual data connection to the computer
'''

import numpy as np
import visa
import time
import os
import pandas as pd
import serial
from collections import deque


visa_rm = visa.ResourceManager()

#########################################################
# Picoscope 6000 ########################################
#########################################################
class Picoscope(object):
    '''
    This class will basically extend the colinoflynn picoscope module
    Has some higher level functionality, and it stores/manipulates the channel settings.

    If you pass a previous connected instance to init, it will take over the existing connection
    '''
    def __init__(self, previous_instance=None):
        from picoscope import ps6000
        self.ps6000 = ps6000
        # I might have subclassed PS6000, but then I would have to import it before the class definition...
        # Not sure at the moment how to take over an old connection when reinstantiating
        # self.get_data will return data as well as
        self.data = None
        # Store channel settings in this class
        # TODO find a way to make these persist when this module gets reloaded!
        # This might be really bad programming ..
        self.range = self._PicoRange(self)
        self.offset = self._PicoOffset(self)
        self.atten = self._PicoAttenuation(self)
        self.coupling = self._PicoCouping(self)
        if previous_instance is None:
            self.ps = ps6000.PS6000(connect=True)
        else:
            self.ps = ps6000.PS6000(connect=False)
            self.ps.handle = previous_instance.handle
            self.range.update(previous_instance.range)
            self.offset.update(previous_instance.offset)
            self.atten.update(previous_instance.atten)
            self.coupling.update(previous_instance.coupling)
            # if you don't clear the handle on the old instance, the garbage collector
            # can close the picoscope connection
            previous_instance.ps.handle = None
        self.close = self.ps.close
        self.handle = self.ps.handle
        # TODO: methods of PS6000 to expose?
        self.getAllUnitInfo = self.ps.getAllUnitInfo
        self.getUnitInfo = self.ps.getUnitInfo

    def print_settings(self):
        print('Picoscope channel settings:')
        print(pd.DataFrame([self.coupling, self.atten, self.offset, self.range],
                           index=['Couplings', 'Attenuations', 'Offsets', 'Ranges']))

    class _PicoSetting(dict):
        def __init__(self, parent):
            self._parent = parent

        def set(self, channel, value):
            self[channel] = value

        def setall(self, value):
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

    class _PicoRange(_PicoSetting):
        # Holds the values for picoscope channel ranges.  Enforces valid values.
        # TODO: add increment and decrement
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self.possible = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
            self.max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))
            self['A'] = 1.0
            self['B'] = 1.0
            self['C'] = 1.0
            self['D'] = 1.0

        def set(self, channel, value):
            offset = self._parent.offset[channel]
            atten = self._parent.atten[channel]
            newvalue = self[channel]

            if value in self.possible * atten:
                newvalue = value
            else:
                argclosest = argmin(np.abs(self.possible * atten - value))
                closest = self.possible[argclosest] * atten
                print('{} is an impossible range setting. Using closest valid setting {}.'.format(value, closest))
                newvalue = closest

            # Forgive me
            diffs = self.max_offsets - offset/atten
            leastabove = self.max_offsets[diffs > 0][0]
            firstleastabove = np.where(self.max_offsets == leastabove)[0][0]
            min_range = self.possible[firstleastabove] * atten
            if newvalue < min_range:
                print('Range {} is too low for current offset {}. Using closest valid range setting {}.'.format(newvalue, offset, min_range))
                newvalue = min_range

            self[channel] = newvalue

    class _PicoOffset(_PicoSetting):
        # _PicoOffset needs to be aware of the range setting in order to determine valid values
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self.possible_ranges = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
            self.max_offsets = (.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20)
            self['A'] = 0.0
            self['B'] = 0.0
            self['C'] = 0.0
            self['D'] = 0.0

        def set(self, channel, value):
            channelrange = self._parent.range[channel]
            channelatten = self._parent.atten[channel]
            maxoffset = self.max_offsets[self.possible_ranges.index(channelrange / channelatten)] * channelatten
            if abs(value) < maxoffset:
                self[channel] = value
            else:
                clippedvalue = np.sign(value) * maxoffset
                print(('{} is above the maximum offset for channel {} with range {} V. '
                    'Setting offset to {}.').format(value, channel, channelrange, clippedvalue))
                self[channel] = clippedvalue

    class _PicoAttenuation(_PicoSetting):
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self['A'] = 1.0
            self['B'] = 1.0
            self['C'] = 1.0
            self['D'] = 1.0
            # I am not sure what the possible values of this setting are ..
            #self.possible =

    class _PicoCouping(_PicoSetting):
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self['A'] = 'DC'
            self['B'] = 'DC'
            self['C'] = 'DC'
            self['D'] = 'DC'
            self.possible = ('DC', 'AC', 'DC50')  # I think?

        def set(self, channel, value):
            if value in self.possible:
                self[channel] = value
            else:
                print('{} is not a valid coupling setting.'.format(value))


    def squeeze_range(self, data, ch=['A', 'B', 'C', 'D']):
        '''
        Find the best range for given input data (can be any number of channels)
        Set the range and offset to the lowest required to fit the data
        '''
        for c in ch:
            if c in data:
                if type(data[c][0]) is np.int8:
                    # Need to convert to float
                    usedrange = data['RANGE'][c]
                    usedoffset = data['OFFSET'][c]
                    maximum = np.max(data[c]) / 2**8 * usedrange * 2 - usedoffset
                    minimum = np.min(data[c]) / 2**8 * usedrange * 2 - usedoffset
                    rang, offs = self.best_range((minimum, maximum))
                else:
                    rang, offs = self.best_range(data[c])
                print('Setting picoscope channel {} range {}, offset {}'.format(c, rang, offs))
                self.range[c] = rang
                self.offset[c] = offs

    def best_range(self, data):
        '''
        Return the best RANGE and OFFSET values to use for a particular input signal (array)
        Just uses minimim and maximum values of the signal, therefore you could just pass (min, max), too
        Don't pass int8 signals, would then need channel information to convert to V
        '''
        # TODO: consider the attenuation!
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

    def capture(self, ch='A', freq=None, duration=None, nsamples=None,
                trigsource='TriggerAux', triglevel=0.1, timeout_ms=30000, pretrig=0.0,
                chrange=None, choffset=None, chcoupling=None, chatten=None):
        '''
        Set up picoscope to capture from specified channel(s).

        pass exactly two of: freq(sampling frequency), duration, nsamples
        sampling frequency has limited possible values, so actual number of samples will vary
        will try to sample for the intended duration, either the value of the duration argument
        or nsamples/freq

        Won't actually start capture until picoscope receives the specified trigger event.

        It will trigger automatically after a timeout.

        ch can be a list of characters, i.e. ch=['A','B'].

        if any of chrange, choffset, chcouplings, chattenuation (dicts) are not passed,
        the settings will be taken from the global variables
        '''

        # Check that two of freq, duration, nsamples was passed
        if not sum([x is None for x in (freq, duration, nsamples)]) == 1:
            raise Exception('Must give exactly two of the arguments freq, duration, and nsamples.')

        # If ch not iterable, just put it in a list by itself
        if not hasattr(ch, '__iter__'):
            ch = [ch]

        # Maximum sample rate is different depending on the number of channels that are enabled.
        # Therefore, if you want the highest possible rate, you should keep unused channels disabled.
        # Enable only the channels being used, disable the rest
        for c in ['A', 'B', 'C', 'D']:
            if c not in ch:
                self.ps.setChannel(c, enabled=False)

        # If freq and duration are passed, take as many samples as it takes to sample for duration
        # If duration and nsamples are passed, sample with frequency as near as possible to nsamples/duration (nsamples will vary)
        # If freq and nsamples are passed, sample at closest possible frequency for nsamples (duration will vary)
        if freq is None:
            freq = nsamples / duration
        # This will return actual sample frequency, then we can determine
        # the number of samples needed.
        actualfreq, _ = self.ps.setSamplingFrequency(freq, 0)

        if duration is not None:
            nsamples = duration * actualfreq

        def global_replace(kwarg, instancearg):
            if kwarg is None:
                # No values passed, use the instance values
                return instancearg
            else:
                # Fill missing values with instance values
                kwargcopy = kwarg.copy()
                for c in ch:
                    if c not in kwargcopy:
                        kwargcopy[c] = instancearg[c]
                return kwargcopy

        chrange = global_replace(chrange, self.range)
        choffset = global_replace(choffset, self.offset)
        chcoupling = global_replace(chcoupling, self.coupling)
        chatten = global_replace(chatten, self.atten)

        actualfreq, max_samples = self.ps.setSamplingFrequency(actualfreq, nsamples)
        print('Actual picoscope sampling frequency: {:,}'.format(actualfreq))
        if nsamples > max_samples:
            raise(Exception('Trying to sample more than picoscope memory capacity'))
        # Set up the channels
        for c in ch:
            self.ps.setChannel(channel=c,
                               coupling=chcoupling[c],
                               VRange=chrange[c],
                               probeAttenuation=chatten[c],
                               VOffset=choffset[c],
                               enabled=True)
        # Set up the trigger.  Will timeout in 30s
        self.ps.setSimpleTrigger(trigsource, triglevel, timeout_ms=timeout_ms)
        self.ps.runBlock(pretrig)
        return actualfreq

    def get_data(self, ch='A', raw=False, dtype=np.float32):
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
        while(not self.ps.isReady()):
            time.sleep(0.01)

        if not hasattr(ch, '__iter__'):
            ch = [ch]
        for c in ch:
            if raw:
                # For some reason pico-python gives the values as int16
                # Probably because some scopes have 16 bit resolution
                # The 6403c is only 8 bit, and I'm looking to save memory here
                rawint16, _, _ = self.ps.getDataRaw(c)
                data[c] = np.int8(rawint16 / 2**8)
            else:
                # I added dtype argument to pico-python
                data[c] = self.ps.getDataV(c, dtype=dtype)

        Channels = ['A', 'B', 'C', 'D']
        data['RANGE'] = {ch:chr for ch, chr in zip(Channels, self.ps.CHRange)}
        data['OFFSET'] = {ch:cho for ch, cho in zip(Channels, self.ps.CHOffset)}
        data['ATTENUATION'] = {ch:cha for ch, cha in zip(Channels, self.ps.ProbeAttenuation)}
        data['sample_rate'] = self.ps.sampleRate
        # Specify samples captured, because this field will persist even after splitting for example
        # Then if you split 100,000 samples into 10 x 10,000 having nsamples = 100,000 will be confusing
        data['nsamples_capture'] = len(data[ch[0]])
        # Using the current state of the global variables to record what settings were used
        # I don't know a way to get couplings from the picoscope instance
        # TODO: pull request a change to setChannel to fix this
        data['COUPLINGS'] = dict(self.coupling)
        # Sample frequency?
        self.data = data
        return data


#########################################################
# Rigol DG5000 AWG ######################################
#########################################################
class RigolDG5000(object):
    # There is at least one python library for DG5000, but I could not get it to run.
    def __init__(self, addr='USB0::0x1AB1::0x0640::DG5T155000186::INSTR'):
        try:
            self.connect(addr)
        except:
            print('Rigol connection failed.')

    def connect(self, addr):
        self.conn = visa_rm.open_resource(addr)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.ask = self.conn.ask
        self.close = self.conn.close

    ### These directly wrap SCPI commands that can be sent to the rigol AWG

    def shape(self, shape='SIN', ch=1):
        '''
        Change the waveform shape to a built-in value. Possible values are:
        SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|
        GAUSsian |HAVersine|LORentz|ARBPULSE|DUAltone
        '''
        self.write('SOURCE{}:FUNC:SHAPE {}'.format(ch, shape))

    def outputstate(self, state=True, ch=1):
        ''' Turn output state on or off '''
        statestr = 'ON' if state else 'OFF'
        self.write(':OUTPUT{}:STATE {}'.format(ch, statestr))

    def frequency(self, freq, ch=1):
        ''' Set frequency of AWG waveform.  Not the sample rate! '''
        self.write(':SOURCE{}:FREQ:FIX {}'.format(ch, freq))

    def amplitude(self, amp, ch=1):
        ''' Set amplitude of AWG waveform '''
        self.write(':SOURCE{}:VOLTAGE:AMPL {}'.format(ch, amp))

    def offset(self, offset, ch=1):
        ''' Set offset of AWG waveform '''
        self.write(':SOURCE{}:VOLT:OFFS {}'.format(ch, offset))

    def output_resistance(self, r=50, ch=1):
        ''' Manual says you can change output resistance from 1 to 10k'''
        self.write('OUTPUT{}:IMPEDANCE {}'.format(ch, r))

    def sync(self, state=True):
        ''' Can turn on/off the sync output (on rear) '''
        statestr = 'ON' if state else 'OFF'
        self.write('OUTPUT{}:SYNC ' + statestr)

    def screensaver(self, state=False):
        ''' Turn the screensaver on or off.
        Screensaver causes problems with triggering because DG5000 is a piece of junk. '''
        statestr = 'ON' if state else 'OFF'
        self.write(':DISP:SAV ' + statestr)

    def ramp_symmetry(self, percent=50, ch=1):
        ''' Change the symmetry of a ramp output.
        Refers to the sweep rates of increasing/decreasing ramps. '''
        self.write('SOURCE{}:FUNC:RAMP:SYMM {}'.format(ch, percent))

    def dutycycle(self, percent=50, ch=1):
        ''' Change the duty cycle of a square output. '''
        self.write('SOURCE{}:FUNC:SQUare:DCYCle {}'.format(ch, percent))

    def error(self, ):
        ''' Get error message from rigol '''
        return self.ask(':SYSTem:ERRor?')

    # <<<<< For burst mode
    def ncycles(self, n, ch=1):
        ''' Set number of cycles that will be output in burst mode '''
        self.write(':SOURCE{}:BURST:NCYCLES {}'.format(ch, n))

    def trigsource(self, source='MAN', ch=1):
        ''' Change trigger source for burst mode. INTernal|EXTernal|MANual '''
        self.write(':SOURCE{}:BURST:TRIG:SOURCE {}'.format(ch, source))

    def trigger(self, ch=1):
        '''
        Send signal to rigol to trigger immediately.  Make sure that trigsource is set to MAN:
        trigsource('MAN')
        '''
        self.write(':SOURCE{}:BURST:TRIG IMM'.format(ch))

    def burstmode(self, mode='TRIG', ch=1):
        '''Set the burst mode.  I don't know what it means. 'TRIGgered|GATed|INFinity'''
        self.write(':SOURCE{}:BURST:MODE {}'.format(ch, mode))

    def burst(self, state=True, ch=1):
        ''' Turn the burst mode on or off '''
        statestr = 'ON' if state else 'OFF'
        self.write(':SOURCE{}:BURST:STATE {}'.format(ch, statestr))

    # End for burst mode >>>>>
    def load_wfm(self, waveform):
        '''
        Load some data as an arbitrary waveform to be output.
        Data will be normalized.  Use amplitude to set the amplitude.
        Make sure that the output is off, because the command switches out of burst mode
        and will start outputting immediately.
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
        self.write(':TRAC:DATA VOLATILE,{}'.format(wfm_str))

    def interp(self, interp=True):
        ''' Set AWG datapoint interpolation mode '''
        modestr = 'LIN' if interp else 'OFF'
        self.write('TRACe:DATA:POINts:INTerpolate {}'.format(modestr))

    def color(self, c='RED'):
        '''
        Change the highlighting color on rigol screen for some reason
        'RED', 'DEEPRED', 'YELLOW', 'GREEN', 'AZURE', 'NAVYBLUE', 'BLUE', 'LILAC', 'PURPLE', 'ARGENT'
        '''
        self.write(':DISP:WIND:HLIG:COL {}'.format(c))

    def idn(self):
        return self.ask('*IDN?').replace('\n', '')

    ### These use the wrapped SCPI commands to accomplish something useful

    def load_volatile_wfm(self, waveform, duration, n=1, ch=1, interp=True):
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
        self.outputstate(False, ch=ch)
        #
        # Turn off screen saver.  It sends a premature pulse on SYNC output if on.
        # This will make the scope trigger early and miss part or all of the pulse.  Really dumb.
        self.screensaver(False)
        #time.sleep(.01)
        # Turn on interpolation for IVs, off for steps
        self.interp(interp)
        # This command switches out of burst mode for some stupid reason
        self.load_wfm(waveform)
        freq = 1. / duration
        self.frequency(freq, ch=ch)
        maxamp = np.max(np.abs(waveform))
        self.amplitude(2*maxamp, ch=ch)
        self.burstmode('TRIG', ch=ch)
        self.ncycles(n, ch=ch)
        self.trigsource('MAN', ch=ch)
        self.burst(True, ch=ch)
        self.outputstate(True, ch=ch)


    def load_builtin_wfm(self, shape='SIN', duration=None, freq=None, amp=1, offset=0, n=1, ch=1):
        '''
        Set up a built-in waveform to pulse n times
        SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|GAUSsian|
        HAVersine|LORentz|ARBPULSE|DUAltone
        '''

        if not (bool(duration) ^ bool(freq)):
            raise Exception('Must give either duration or frequency, and not both')

        if freq is None:
            freq = 1. / duration

        # Set up waveform
        self.burst(True, ch=ch)
        self.burstmode('TRIG', ch=ch)
        self.shape(shape, ch=ch)
        # Rigol's definition of amp is peak-to-peak, which is unusual.
        self.amplitude(2*amp, ch=ch)
        self.offset(offset, ch=ch)
        self.burstmode('TRIG', ch=ch)
        self.ncycles(n, ch=ch)
        self.trigsource('MAN', ch=ch)
        self.frequency(freq, ch=ch)


    def pulse_builtin(self, shape='SIN', duration=None, freq=None, amp=1, offset=0, n=1, ch=1):
        '''
        Pulse a built-in waveform n times
        SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|GAUSsian|
        HAVersine|LORentz|ARBPULSE|DUAltone
        '''
        passthrough = {k:v for k,v in locals().items() if k != 'self'}
        self.load_builtin_wfm(**passthrough)

        self.outputstate(True)
        # Trigger rigol
        self.trigger(ch=ch)


    def pulse_arbitrary(self, waveform, duration, n=1, ch=1, interp=True):
        '''
        Generate n pulses of the input waveform on Rigol AWG.
        Trigger immediately.
        Manual says you can use up to 128 Mpts, ~2^27, but for some reason you can't.
        Another part of the manual says it is limited to 512 kpts, but can't seem to do that either.
        '''
        # Load waveform
        passthrough = {k:v for k, v in locals().items() if k != 'self'}
        self.load_volatile_wfm(**passthrough)
        # Trigger rigol
        self.trigger(ch=1)


#########################################################
# Keithley 2600 #########################################
#########################################################
class Keithley2600(object):
    '''
    Sadly, Keithley decided to embed a lua interpreter into its source meters
    instead of providing a proper programming interface.

    This means we have to communicate with Keithley via sending and receiving
    strings in the lua programming language.

    One could wrap every useful lua command in a python function which writes
    the lua string, and parses the response, but this would be quite an
    undertaking.

    Here we maintain a separate lua file "Keithley_2600.lua" which defines lua
    functions on the keithley, then we wrap those in python.
    '''
    def __init__(self, addr='TCPIP::192.168.11.11::inst0::INSTR'):
        try:
            self.connect(addr)
        except:
            print('Keithley connection failed at {}'.format(addr))

    def connect(self, addr='TCPIP::192.168.11.11::inst0::INSTR'):
        self.conn = visa_rm.get_instrument(addr, open_timeout=0)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.ask = self.conn.ask
        self.read = self.conn.read
        self.read_raw = self.conn.read_raw
        self.close = self.conn.close
        moduledir = os.path.split(__file__)[0]
        self.run_lua_file(os.path.join(moduledir, 'Keithley_2600.lua'))
        # Store up to 100 loops in memory in case you forget to save them to disk
        self.data= deque(maxlen=100)

    def idn(self):
        return self.ask('*IDN?').replace('\n', '')

    def run_lua_lines(self, lines):
        ''' Send some lines (list of strings) to Keithley lua interpreter '''
        self.write('loadandrunscript')
        for line in lines:
            self.write(line)
        self.write('endscript')

    def run_lua_file(self, filepath):
        ''' Send the contents of a file to Keithley lua interpreter '''
        with open(filepath, 'r') as kfile:
            self.run_lua_lines(kfile.readlines())

    def send_list(self, list_in, varname='pythonlist'):
        '''
        In order to send a list of values to keithley, we need to compose a lua
        string to define it as a variable.

        Problem is that the input buffer of Keithley is very small, so the lua string
        needs to be separated into many lines. This function accomplishes that.
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

        self.run_lua_lines(cmdlist)

    def iv(self, vlist, Irange, Ilimit, nplc=1, delay='smua.DELAY_AUTO'):
        '''Wraps the SweepVList lua function defined on keithley'''
        # Send list of voltage values to keithley
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write('SweepVList(sweeplist, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay))

    def vi(self, ilist, Vrange, Vlimit, nplc=1, delay='smua.DELAY_AUTO'):
        '''Wraps the SweepIList lua function defined on keithley'''

        # Send list of voltage values to keithley
        self.send_list(ilist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        Irange = np.max(np.abs(ilist))
        self.write('SweepIList(sweeplist, {}, {}, {}, {}, {})'.format(Vrange, Vlimit, nplc, delay, Irange))

    def it(self, sourceVA, sourceVB, points, interval, rangeI, limitI, nplc=1):
        '''Wraps the constantVoltageMeasI lua function defined on keithley'''
        # Call constantVoltageMeasI
        # TODO: make sure the inputs are valid
        self.write('constantVMeasI({}, {}, {}, {}, {}, {}, {})'.format(sourceVA, sourceVB, points, interval, rangeI, limitI, nplc))
        #self.write('smua.source.levelv = 0')
        #self.write('smua.source.output = smub.OUTPUT_OFF')
        #self.write('smub.source.levelv = 0')
        #self.write('smub.source.output = smub.OUTPUT_OFF')

    def done(self):
        # works with smua.trigger.initiate()
        donesweeping = not bool(float(self.ask('print(status.operation.sweeping.condition)')))
        # works with smua.measure.overlappediv()
        donemeasuring = not bool(float(self.ask('print(status.operation.measuring.condition)')))
        # works with both

        return donesweeping & donemeasuring

    def waitready(self):
        ''' There's probably a better way to do this. '''

        self.write('waitcomplete()')
        self.write('print(\"Complete\")')
        answer = None
        while answer is None:
            try:
                # Keep trying to read until keithley says Complete
                answer = self.read()
            except:
                pass

        '''
        # Another bad way ...
        answer = 1
        while answer != 0.0:
            answer = float(self.ask('print(status.operation.sweeping.condition)'))
            plt.pause(.3)
        '''


    def read_buffer(self, buffer='smua.nvbuffer1', attr='readings', start=1, end=None):
        '''
        Read a data buffer and return an actual array.
        Keithley 2634B handles this just fine while still doing a sweep
        Keithley 2636A throws error 5042 - cannot perform requested action while overlapped operation is in progress.
        '''
        if end is None:
            # Read the whole length
            end = int(float(self.ask('print({}.n)'.format(buffer))))
        # makes keithley give numbers in ascii
        # self.write('format.data = format.ASCII')
        #readingstr = self.ask('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
        #return np.float64(readingstr.split(', '))

        # Makes keithley give numbers in binary float64
        # Should be much faster?
        self.write('format.data = format.REAL64')
        self.write('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
        # reply comes back with #0 or something in the beginning and a newline at the end
        raw = self.read_raw()[2:-1]
        return np.fromstring(raw, dtype=np.float64)

    def get_data(self, start=1, end=None, history=True):
        '''
        Ask Keithley to print out the data arrays of interest (I, V, t, ...)
        Parse the strings into python arrays
        Return dict of arrays
        dict can also contain scalar values or other meta data

        Can pass start and end values if you want just a specific part of the arrays
        '''
        numpts = int(float(self.ask('print(smua.nvbuffer1.n)')))
        if end is None:
            end = numpts
        if numpts > 0:
            # Output a dictionary with voltage/current arrays and other parameters
            out = {}
            out['units'] = {}
            out['longnames'] = {}

            ### Collect measurement conditions
            # TODO: What other information is available from Keithley registers?

            # Need to do something different if sourcing voltage vs sourcing current
            source = self.ask('print(smua.source.func)')
            source = float(source)
            if source:
                # Returns 1.0 for voltage source (smua.OUTPUT_DCVOLTS)
                out['source'] = 'V'
                out['V'] = self.read_buffer('smua.nvbuffer2', 'sourcevalues', start, end)
                Vmeasured = self.read_buffer('smua.nvbuffer2', 'readings', start, end)
                Vmeasured = Keithley2600.replace_nanvals(Vmeasured)
                out['Vmeasured'] = Vmeasured
                out['units']['Vmeasured'] = 'V'
                I = self.read_buffer('smua.nvbuffer1', 'readings', start, end)
                I = Keithley2600.replace_nanvals(I)
                out['I'] = I
                out['Icomp'] = float(self.ask('print(smua.source.limiti)'))
            else:
                # Current source
                out['source'] = 'I'
                out['Vrange'] = float(self.ask('print(smua.nvbuffer2.measureranges[1])'))
                out['Vcomp'] = float(self.ask('print(smua.source.limitv)'))

                out['I'] = self.read_buffer('smua.nvbuffer1', 'sourcevalues', start, end)
                Imeasured = self.read_buffer('smua.nvbuffer1', 'readings', start, end)
                Imeasured = Keithley2600.replace_nanvals(Imeasured)
                out['Imeasured'] = Imeasured
                out['units']['Imeasured'] = 'A'
                V = self.read_buffer('smua.nvbuffer2', 'readings', start, end)
                V = Keithley2600.replace_nanvals(V)
                out['V'] = V

            out['t'] = self.read_buffer('smua.nvbuffer2', 'timestamps', start, end)
            out['Irange'] = self.read_buffer('smua.nvbuffer1', 'measureranges', start, end)
            out['Vrange'] = self.read_buffer('smua.nvbuffer2', 'measureranges', start, end)

            out['units']['I'] = 'A'
            out['units']['V'] = 'V'

        else:
            empty = np.array([])
            out = dict(t=empty, V=empty, I=empty, Vmeasured=empty)
        if history:
            self.data.append(out)
        return out

    @staticmethod
    def replace_nanvals(array):
        # Keithley returns this special value when the measurement is out of range
        # replace it with a nan so it doesn't mess up the plots
        # They aren't that smart at Keithley, so different models return different special values.
        nanvalues = (9.9100000000000005e+37, 9.9099995300309287e+37)
        for nv in nanvalues:
            array[array == nv] = np.nan
        return array

    def set_channel_state(self, channel='A', state=True):
        ch = channel.lower()
        if state:
            self.write('smu{0}.source.output = smu{0}.OUTPUT_ON'.format(ch))
        else:
            self.write('smu{0}.source.output = smu{0}.OUTPUT_OFF'.format(ch))

#########################################################
# Eurotherm 2408 -- #################
#########################################################
class Eurotherm2408():
    '''
    This uses some dumb proprietary EI-BISYNCH protocol
    You can also use modbus
    '''
    def __init__(self, addr='COM32', gid=0, uid=1):
        self.conn = serial.Serial(addr, timeout=1, bytesize=7, parity=serial.PARITY_EVEN)
        self.gid = gid
        self.uid = uid
    def write_data(self, mnemonic, data):
        # Select
        # C1 C2 are the two characters of the mnemonic
        # [EOT] (GID) (GID) (UID) (UID) [STX] (CHAN) (C1) (C2) <DATA> [ETX] (BCC)
        from functools import reduce
        from operator import xor
        STX = '\x02'
        ETX = '\x03'
        EOT = '\x04'
        ENQ = '\x05'
        CHAN = '1'
        gid = str(self.gid)
        uid = str(self.uid)
        data = format(data, '.2f')
        bcc = str(reduce(xor, (mnemonic + data + ETX).encode()))
        msg = EOT + gid + gid + uid + uid + STX + mnemonic + data + ETX + bcc
        self.conn.write(msg.encode())

        # Wait?
        time.sleep(.05)

        # Should reply
        # [NAK] - failed to write
        # [ACK] - successful write
        # (nothing) - oh shit
        ACK = '\x06'
        NAK = '\x15'
        reply = self.conn.read_all()
        return reply

    def read_data(self, mnemonic):
        EOT = '\x04'
        ENQ = '\x05'
        gid = str(self.gid)
        uid = str(self.uid)
        # Poll
        # [EOT] (GID) (GID) (UID) (UID) (CHAN) (C1) (C2) [ENQ]
        poll = EOT + gid + gid + uid + uid + mnemonic + ENQ
        self.conn.write(poll.encode())

        # Wait?
        time.sleep(.05)

        # Reply
        # [STX] (CHAN) (C1) (C2) <DATA> [ETX] (BCC)
        reply = self.conn.read_all()
        return float(reply[3:-2])

    def read_temp(self):
        return self.read_data('PV')

    def set_temp(self, value):
        return self.write_data('SL', value)


#########################################################
# Measurement Computing USB1208HS DAQ ###################
#########################################################
class USB2708HS():
    def __init__(self):
        # Import here because I don't want the entire module to error if you don't have mcculw installed
        from mcculw import ul
        from mcculw import enums
        self.ul = ul
        self.enums = enums

    def analog_out(self, ch, dacval=None, volts=None):
        '''
        I found a USB-1208HS so this is how you use it I guess.
        Pass a digital value between 0 and 2**12 - 1
        0 is -10V, 2**12 - 1 is 10V
        Voltage values that don't make sense for my current set up are disallowed.
        '''
        board_num = 0
        ao_range = self.enums.ULRange.BIP10VOLTS

        # Can pass dacval or volts.  Prefer dacval.
        if dacval is None:
            # Better have passed volts...
            dacval = self.ul.from_eng_units(board_num, ao_range, volts)
        else:
            volts = self.ul.to_eng_units(board_num, ao_range, dacval)

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
            self.ul.a_out(board_num, ch, ao_range, dacval)
        except ULError as e:
            # Display the error
            print("A UL error occurred. Code: " + str(e.errorcode)
                  + " Message: " + e.message)


    def digital_out(self, ch, val):
        #ul.d_config_port(0, DigitalPortType.AUXPORT, DigitalIODirection.OUT)
        self.ul.d_config_bit(0, self.enums.DigitalPortType.AUXPORT, 8, self.enums.DigitalIODirection.OUT)
        self.ul.d_bit_out(0, self.enums.DigitalPortType.AUXPORT, ch, val)


#########################################################
# TektronixDPO73304D ####################################
#########################################################

class TektronixDPO73304D(object):
    def __init__(self, addr='GPIB0::1::INSTR'):
        try:
            self.connect(addr)
        except:
            print('TektronixDPO73304D connection failed at {}'.format(addr))
    def connect(self, addr='GPIB0::1::INSTR'):
        self.conn = visa_rm.get_instrument(addr)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.ask = self.conn.ask
        self.read = self.conn.read
        self.read_raw = self.conn.read_raw
        self.close = self.conn.close
        moduledir = os.path.split(__file__)[0]

    def idn(self):
        return self.ask('*IDN?').replace('\n', '')

    def bandwidth(self, channel=1, bandwidth=33e9):
        self.write('CH' + str(channel) + ':BAN ' + str(bandwidth))

    def scale(self, channel=1, scale=0.0625):
        self.write('CH' + str(channel) + ':SCAle ' + str(scale))
        self.write('*WAI')

    def position(self, channel=1, position=0):
        self.write('CH'+str(channel)+':POS '+str(position))

    def inputstate(self, channel=1, mode=True):
        if mode:
            self.write('SELECT:CH' + str(channel) + ' ON')
        else:
            self.write('SELECT:CH' + str(channel) + ' OFF')

    def offset(self, channel=1, offset=0):
        self.write('CH' + str(channel) + ':OFFSet ' + str(offset))

    # TODO: Should be two separate functions
    def change_div_and_samplerate(self, division, samplerate):
        self.write('HORIZONTAL:MODE AUTO')
        self.write('HORIZONTAL:MODE:SAMPLERATE ' + str(samplerate))
        self.write('HOR:MODE:SCA ' + str(division))
        self.write('HORIZONTAL:MODE:AUTO:LIMIT 10000')

    def recordlength(self, recordlength=1e5):
        self.write('HORIZONTAL:MODE MANUAL')
        self.write('HORIZONTAL:MODE:RECORDLENGTH ' + str(recordlength))
        self.write('HORIZONTAL:MODE:AUTO:LIMIT ' + str(recordlength))

    # TODO: Should be two separate functions
    def change_samplerate_and_recordlength(self, samplerate=100e9, recordlength=1e5):
        self.write('HORIZONTAL:MODE MANUAL')
        self.write('HORIZONTAL:MODE:SAMPLERATE ' + str(samplerate))
        self.write('HORIZONTAL:MODE:RECORDLENGTH ' + str(recordlength))
        self.write('HORIZONTAL:MODE:AUTO:LIMIT ' + str(recordlength))
        self.write('DATA:STOP ' + str(recordlength))

    def ext_db_attenuation(self, channel=1, attenuation=0):
        self.write('CH' + str(channel) + ':PROBEFUNC:EXTDBATTEN ' + str(attenuation))

    def trigger(self):
        self.write('TRIGger FORCe')

    def arm(self, source=1, level=-0.1, edge='e'):
        if source == 0:
            self.write('TRIG:A:EDGE:SOUrce AUX')
        else:
            self.write('TRIG:A:EDGE:SOUrce CH' + str(source))
        self.write('TRIG:A:LEVEL ' + str(level))
        self.write('ACQ:STOPA SEQUENCE')
        self.write('ACQ:STATE 1')
        if edge == 'r':
            self.write('TRIG:A:EDGE:SLO RIS')
        elif edge == 'f':
            self.write('TRIG:A:EDGE:SLO FALL')
        else:
            self.write('TRIG:A:EDGE:SLO EIT')
        triggerstate = self.ask('TRIG:STATE?')
        while 'REA' not in triggerstate or 'SAVE' in triggerstate:
            self.write('ACQ:STATE 1')
            triggerstate = self.ask('TRIG:STATE?')

    def get_curve(self, channel=1):
        self.write('HEAD 0')
        self.write('WFMOUTPRE:BYT_NR 1')
        self.write('WFMOUTPRE:BIT_NR 8')
        self.write('DATA:ENC RPB')
        self.write('DATA:SOURCE CH' + str(channel))
        rl = int(self.ask('HOR:RECO?'))

        pre = self.ask('WFMOutpre?')
        pre_split = pre.split(';')
        if len(pre_split) == 5:
            print('Channel ' + str(channel) + ' is not used.')
            return None

        x_incr = float(pre_split[9])
        x_offset = int(pre_split[11])
        y_mult = float(pre_split[13])
        y_off = float(pre_split[14])

        self.write('DATA:STOP ' + str(rl))
        self.write('CURVE?')
        data_str = self.read_raw()
        data = np.fromstring(data_str[6:-1], np.uint8)

        time = x_incr * (np.arange(len(data)) - x_offset)
        voltage = y_mult * (data - y_off)

        return_dict = {}
        return_dict['t_ttx'] = time
        return_dict['V_ttx'] = voltage
        return return_dict

    def disarm(self):
        self.write('ACQ:STATE 0')

    def triggerstate(self):
        trigger_str = self.ask('TRIG:STATE?')
        return trigger_str == 'READY\n'
