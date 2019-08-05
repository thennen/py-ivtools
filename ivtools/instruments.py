'''
These classes contain functionality specific to only one instrument.
Don't put code in an instrument class that has anything to do with a different instrument,
or any particular application!

They are grouped into classes because there may be some overlapping function names which
should be contained.  Also there might be a situation where we would like multiple instances
(e.g. when using two Keithley's).

Should only put instruments here that have an actual data connection to the computer

Right now we use the Borg pattern to maintain instrument state (and reuse existing connections),
and we keep the state in a separate module so that it even survives reload of this module.
I don't know if this is a horrible idea or not, but it seems to suit our purposes very nicely.

You can create an instance of these classes anywhere in your code, and they will automatically
reuse a connection if it exists, EVEN IF THE CLASS DEFINITION ITSELF HAS CHANGED.
One downside is that if you screw up the state somehow, you have to manually delete it to start over.
But one could add some kind of reset_state flag to __init__ to handle this.

If, in the future, we need multiple instances of the same instrument class, we can implement
something that detects the appropriate state dict to use.

#TODO make parent class or decorator to implement the borg stuff

Another approach could be to have the module maintain weak references to all instrument instances,
and have a function that decides whether to instantiate a new instance or return an existing one.
I tried this for a while and I think it's a worse solution.
'''

# TODO: Maybe split this up into one file per instrument

import numpy as np
import visa
import time
import os
import pandas as pd
import serial
from serial.tools.list_ports import grep as comgrep
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import deque
from . import persistent_state
visa_rm = persistent_state.visa_rm
# Could also store the visa_rm in visa itself
#visa.visa_rm = visa.ResourceManager()
#visa_rm = visa.visa_rm



# Picoscope is borg:
# https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/
# This means if you ever want several instances of the same instrument class to control different
# instruments (e.g. two picoscopes), you will need to find another solution


#########################################################
# Picoscope 6000 ########################################
#########################################################
class Picoscope(object):
    '''
    This class will basically extend the colinoflynn picoscope module
    Has some higher level functionality, and it stores/manipulates the channel settings.
    '''
    def __init__(self, connect=True):
        self.__dict__ = persistent_state.pico_state
        from picoscope import ps6000
        self.ps6000 = ps6000
        # I could have subclassed PS6000, but then I would have to import it before the class definition...
        # Then this whole package would have picoscope module as a dependency
        # self.get_data will return data as well as save it here
        self.data = None
        # Store channel settings in this class
        if not hasattr(self, 'range'):
            self.offset = self._PicoOffset(self)
            self.atten = self._PicoAttenuation(self)
            self.coupling = self._PicoCoupling(self)
            self.range = self._PicoRange(self)
        if connect:
            self.connect()

    def connect(self):
        # We are borg, so might already be connected!
        if self.connected():
            #info = self.ps.getUnitInfo('VariantInfo')
            #print('Picoscope {} already connected!'.format(info))
            pass
        else:
            try:
                self.ps = self.ps6000.PS6000(connect=True)
                model = self.ps.getUnitInfo('VariantInfo')
                print('Picoscope {} connection succeeded.'.format(model))
                self.close = self.ps.close
                self.handle = self.ps.handle
                # TODO: methods of PS6000 to expose?
                self.getAllUnitInfo = self.ps.getAllUnitInfo
                self.getUnitInfo = self.ps.getUnitInfo
            except Exception as e:
                self.ps = None
                print('Connection to picoscope failed. There could be an unclosed session.')
                print(e)

    def connected(self):
        if hasattr(self, 'ps'):
            try:
                self.ps.getUnitInfo('VariantInfo')
                return True
            except:
                return False

    def print_settings(self):
        print('Picoscope channel settings:')
        print(pd.DataFrame([self.coupling, self.atten, self.offset, self.range],
                           index=['Couplings', 'Attenuations', 'Offsets', 'Ranges']))

    # Settings are a class mainly because I wanted a convenient syntax for typing in repeatedly
    class _PicoSetting(dict):
        possible_ranges_1M = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
        max_offsets_1M = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))
        possible_ranges_50 = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0))
        max_offsets_50 = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 2.5))
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
            return self['A']
        @a.setter
        def a(self, value):
            self.set('A', value)
        @property
        def b(self):
            return self['B']
        @b.setter
        def b(self, value):
            self.set('B', value)
        @property
        def c(self):
            return self['C']
        @c.setter
        def c(self, value):
            self.set('C', value)
        @property
        def d(self):
            return self['D']
        @d.setter
        def d(self, value):
            self.set('D', value)

    class _PicoRange(_PicoSetting):
        # Holds the values for picoscope channel ranges.
        # Tries to enforce valid values, though I have not extensively tested that it is correct.
        # Trying to comprehensively determine which settings are allowed and
        # figuring out how to handle all the setting changes is logically NON-TRIVIAL
        # Should changing one setting ever affect another setting?
        # You would think not,
        # But what if e.g. you change the attenuation but want to keep looking at the same range?
        # The set of possible ranges changes, so it may only be possible to make the change simultaneously
        # If a non-valid value is passed, should the closest valid one be used? or always the lower or higher setting?
        #
        # TODO: add increment and decrement?
        # TODO: add gui?
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self['A'] = 1.0
            self['B'] = 1.0
            self['C'] = 1.0
            self['D'] = 1.0

        def set(self, channel, value):
            offset = self._parent.offset[channel]
            atten = self._parent.atten[channel]
            coupling = self._parent.coupling[channel]
            # Default value for new setting (no change)
            oldvalue = self[channel]
            newvalue = oldvalue

            if coupling == 'DC50':
                possible_ranges = self.possible_ranges_50 * atten
                max_offsets = self.max_offsets_50 * atten
            else:
                possible_ranges = self.possible_ranges_1M * atten
                max_offsets = self.max_offsets_1M * atten

            # Check if the offset is too high for any of these settings
            possible_ranges = np.array([pr for pr, mo in zip(possible_ranges, max_offsets) if mo >= np.abs(offset)])

            # Choose the next higher possible value, unless value is higher than all possible values
            if value > possible_ranges[-1]:
                newvalue = possible_ranges[-1]
            else:
                newvalue = possible_ranges[np.where(possible_ranges - value >= 0)][0]

            if value != newvalue:
                print(f'Range {value}V is not possible for offset: {offset}, atten: {atten}, coupling: {coupling}. Using range {newvalue}V.')

            self[channel] = newvalue

    class _PicoOffset(_PicoSetting):
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self['A'] = 0.0
            self['B'] = 0.0
            self['C'] = 0.0
            self['D'] = 0.0

        def set(self, channel, value):
            vrange = self._parent.range[channel]
            atten = self._parent.atten[channel]
            coupling = self._parent.coupling[channel]
            # Default value for new setting (no change)
            oldvalue = self[channel]
            newvalue = oldvalue

            if coupling == 'DC50':
                possible_ranges = self.possible_ranges_50 * atten
                max_offsets = self.max_offsets_50 * atten
            else:
                possible_ranges = self.possible_ranges_1M * atten
                max_offsets = self.max_offsets_1M * atten

            # I assume we are currently set to a possible range
            max_offset = max_offsets[np.where(vrange == possible_ranges)][0]

            if np.abs(value) <= max_offset:
                newvalue = value
            else:
                newvalue = np.sign(value) * max_offset

            if value != newvalue:
                print(f'Offset {value}V is not possible for range: {vrange}, atten: {atten}, coupling: {coupling}. Using offset {newvalue}V.')

            self[channel] = newvalue

    class _PicoAttenuation(_PicoSetting):
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self['A'] = 1.0
            self['B'] = 1.0
            self['C'] = 1.0
            self['D'] = 1.0

        def set(self, channel, value):
            # It's a little different to set this setting, because it changes the possible values
            # of other settings, and therefore must modify them at the same time
            # You could either try to keep the same ranges/offsets as before the attenuation change
            # Or you could just scale the ranges/offsets with the attenuation
            # The latter is simpler, so we'll do that.
            # ALL attenuation settings are possible.  I don't know whether this is ok.
            vrange = self._parent.range[channel]
            offset = self._parent.offset[channel]
            coupling = self._parent.coupling[channel]
            oldvalue = self[channel]
            self[channel] = value

            print('Scaling the range and offset settings by the new attenuation setting')
            # Don't use the set methods, because they can assume that other settings are already valid
            self._parent.range[channel] *= value/oldvalue
            self._parent.offset[channel] *= value/oldvalue

    class _PicoCoupling(_PicoSetting):
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self['A'] = 'DC'
            self['B'] = 'DC'
            self['C'] = 'DC'
            self['D'] = 'DC'

        def set(self, channel, value):
            # Don't allow it if the range and offset are not compatible
            # Could also coerce to valid values, but I am tired of writing all this stuff
            vrange = self._parent.range[channel]
            offset = self._parent.offset[channel]
            atten = self._parent.atten[channel]

            if value == 'DC50':
                possible_ranges = self.possible_ranges_50 * atten
                max_offsets = self.max_offsets_50 * atten
            else:
                possible_ranges = self.possible_ranges_1M * atten
                max_offsets = self.max_offsets_1M * atten

            if vrange in possible_ranges:
                where = np.where(vrange == possible_ranges)[0][0]
                if np.abs(offset) <= max_offsets[where]:
                    self[channel] = value

            # Fukken logic structures
            if self[channel] != value:
                # If we didn't make it to the bottom of the last set of conditions
                print(f'Coupling {value} is not possible for range: {vrange}, offset: {offset}, atten: {atten}.')


    def squeeze_range(self, data, padpercent=0, ch=['A', 'B', 'C', 'D']):
        '''
        Find the best range for given input data (can be any number of channels)
        Set the range and offset to the lowest required to fit the data
        '''
        for c in ch:
            if c in data:
                usedatten = data['ATTENUATION'][c]
                usedcoupling = data['COUPLINGS'][c]
                usedrange = data['RANGE'][c]
                usedoffset = data['OFFSET'][c]
                if type(data[c][0]) is np.int8:
                    # Need to convert to float
                    # TODO: consider attenuation
                    maximum = np.max(data[c]) / 2**8 * usedrange * 2 - usedoffset
                    minimum = np.min(data[c]) / 2**8 * usedrange * 2 - usedoffset
                    rang, offs = self.best_range((minimum, maximum), padpercent=padpercent, atten=usedatten, coupling=usedcoupling)
                else:
                    rang, offs = self.best_range(data[c], padpercent=padpercent, atten=usedatten, coupling=usedcoupling)
                print('Setting picoscope channel {} range {}, offset {}'.format(c, rang, offs))
                self.range[c] = rang
                self.offset[c] = offs

    def best_range(self, data, padpercent=0, atten=1, coupling='DC'):
        '''
        Return the best RANGE and OFFSET values to use for a particular input signal (array)
        Just uses minimim and maximum values of the signal, therefore you could just pass (min, max), too
        Don't pass int8 signals, would then need channel information to convert to V
        TODO: Use an offset that includes zero if it doesn't require increasing the range
        '''
        # Consider coupling!
        # Consider the attenuation!
        if coupling in ['DC', 'AC']:
            possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)) * atten
            max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20)) * atten
        elif coupling == 'DC50':
            possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)) * atten
            max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 2.5)) * atten

        # Sadly, each range has a different maximum possible offset
        minimum = np.min(data)
        maximum = np.max(data)
        amplitude = abs(maximum - minimum) / 2
        padamp = amplitude * (1 + padpercent / 100)
        middle = round((maximum + minimum) / 2, 3)
        padmin = minimum - amplitude * padpercent / 2 / 100
        padmax = maximum + amplitude * padpercent / 2 / 100
        # Mask of possible ranges that fit the signal
        mask = possible_ranges >= padamp
        for selectedrange, max_offset in zip(possible_ranges[mask], max_offsets[mask]):
            # Is middle an acceptable offset?
            if middle < max_offset:
                return (selectedrange, -middle)
                break
            # Can we reduce the offset without the signal going out of range?
            elif (max_offset + selectedrange >= padmax) and (-max_offset - selectedrange <= padmin):
                return(selectedrange, np.clip(-middle, -max_offset, max_offset))
                break
            # Neither worked, try increasing the range ...
        # If no range was enough to fit the signal
        print('Signal out of pico range!')
        return (max(possible_ranges), 0)

    def capture(self, ch='A', freq=None, duration=None, nsamples=None,
                trigsource='TriggerAux', triglevel=0.1, timeout_ms=30000, direction='Rising',
                pretrig=0.0, chrange=None, choffset=None, chcoupling=None, chatten=None):
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
            self.ps.setChannel(c, enabled=c in ch)

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
        self.ps.setSimpleTrigger(trigsource, triglevel, direction=direction, timeout_ms=timeout_ms)
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
            rawint16, _, overflow = self.ps.getDataRaw(c)
            if overflow:
                print(f'!! Picoscope overflow on Ch {c} !!')
            if raw:
                # For some reason pico-python gives the values as int16
                # Probably because some scopes have 16 bit resolution
                # The 6403c is only 8 bit, and I'm looking to save memory here
                data[c] = np.int8(rawint16 / 2**8)
            else:
                # I added dtype argument to pico-python
                data[c] = self.ps.rawToV(c, rawint16, dtype=dtype)
                #data[c] = self.ps.getDataV(c, dtype=dtype)
        Channels = ['A', 'B', 'C', 'D']
        # Unfortunately, picopython updates these when you use ps.setChannel
        # So don't setChannel again before get_data, or the following metadata could be wrong!
        # picopython SHOULD return the metadata for each capture with ps.getDataV
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

    def measure(self, ch='A', freq=None, duration=None, nsamples=None, trigsource='TriggerAux', triglevel=0.1,
                timeout_ms=1000, direction='Rising', pretrig=0.0, chrange=None, choffset=None, chcoupling=None,
                chatten=None, raw=False, dtype=np.float32, plot=True, ax=None):
        '''
        Just capture and get_data in one step
        good for interactive use
        '''
        self.capture(ch, freq=freq, duration=duration, nsamples=nsamples, trigsource=trigsource,
                     triglevel=triglevel, timeout_ms=timeout_ms, direction=direction, pretrig=pretrig,
                     chrange=chrange, choffset=choffset, chcoupling=chcoupling, chatten=chatten)
        # Hopefully this doesn't timeout or something
        data = self.get_data(ch, raw=raw, dtype=dtype)
        if plot:
            self.plot(data, ax=ax)
        return data

    def plot(self, chdata=None, ax=None, alpha=.9):
        '''
        Plot the channel data of picoscope
        Includes an indication of the measurement range used
        uses self.data (most recent data) if chdata is not passed
        '''
        if ax is None:
            fig, ax = plt.subplots()
        if chdata is None:
            if self.data is None:
                print('No data to plot')
            else:
                chdata = self.data
        # Colors match the code on the picoscope
        # Yellow is too hard to see
        colors = dict(A='Blue', B='Red', C='Green', D='Gold')
        channels = ['A', 'B', 'C', 'D']
        # Remove any previous range indicators that might exist on the plot
        ax.collections = []
        for c in channels:
            if c in chdata.keys():
                if chdata[c].dtype == np.int8:
                # Convert to voltage for plot
                    chplotdata = chdata[c] / 2**8 * chdata['RANGE'][c] * 2 - chdata['OFFSET'][c]
                else:
                    chplotdata = chdata[c]
                if 'sample_rate' in chdata:
                    # If sample rate is available, plot vs time
                    x = np.arange(len(chdata[c])) * 1/chdata['sample_rate']
                    if 'downsampling' in chdata:
                        x *= chdata['downsampling']
                    ax.set_xlabel('Time [s]')
                    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
                else:
                    x = range(len(chdata[c]))
                    ax.set_xlabel('Data Point')
                ax.plot(x, chplotdata, color=colors[c], label=c, alpha=alpha)
                # lightly indicate the channel range
                choffset = chdata['OFFSET'][c]
                chrange = chdata['RANGE'][c]
                ax.fill_between((0, np.max(x)), -choffset - chrange, -choffset + chrange, alpha=0.05, color=colors[c])
        ax.legend(title='Channel')
        ax.set_ylabel('Voltage [V]')


#########################################################
# Rigol DG5000 AWG ######################################
#########################################################
class RigolDG5000(object):
    '''
    This instrument is really a pain in the ass.  Good example of a job not well done by Rigol.
    But we spent a lot of time learning its quirks and are kind of stuck with it.

    Do not send anything to the Rigol that differs in any way from what it expects,
    or it will just hang forever and need to be manually restarted along with the entire python kernel.

    Certain commands just randomly cause the machine to crash in the same way.  Such as when you try
    to query the number of points stored in the NV memory
    '''
    def __init__(self, addr=None):
        self.verbose = False
        try:
            if addr is None:
                addr = self.get_visa_addr()
            self.connect(addr)
        except:
            print('Rigol connection failed.')
            return
        # Turn off screen saver.  It sends a premature pulse on SYNC output if on.
        # This will make the scope trigger early and miss part or all of the pulse.  Really dumb.
        self.screensaver(False)
        # Store here the last waveform that was programmed, so that we can skip uploading it if it
        # hasn't changed
        self.volatilewfm = []

    def get_visa_addr(self):
        # Look for the address of the DG5000 using visa resource manager
        for resource in visa_rm.list_resources():
            if 'DG5' in resource:
                return resource
        return 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'


    def connect(self, addr):
        try:
            self.conn = visa_rm.open_resource(addr)
            # Expose a few methods directly to self
            self.write = self.conn.write
            self.query = self.conn.query
            self.ask = self.query
            self.close = self.conn.close
            idn = self.conn.ask('*IDN?').replace('\n', '')
            #print('Rigol connection succeeded. *IDN?: {}'.format(idn))
        except:
            print('Connection to Rigol AWG failed.')

    def set_or_query(self, cmd, setting=None):
        # Sets or returns the current setting
        if setting is None:
            if self.verbose: print(cmd + '?')
            reply = self.query(cmd + '?').strip()
            # Convert to numeric?
            replymap = {'ON': 1, 'OFF': 0}

            def will_it_float(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            if reply in replymap.keys():
                return replymap[reply]
            elif reply.isnumeric():
                return int(reply)
            elif will_it_float(reply):
                return float(reply)
            else:
                return reply
        else:
            if self.verbose: print(f'{cmd} {setting}')
            self.write(f'{cmd} {setting}')
            return None

    @staticmethod
    def write_wfm_file(wfm, filepath='wfm.RAF'):
        '''
        The only way to get anywhere near the advertised number of samples
        theoretically works up to 16 MPts = 2**24 samples
        wfm should be between -1 and 1, this will convert it to uint16
        '''
        # Needs extension RAF!
        filepath = os.path.splitext(filepath)[0] + '.RAF'

        if np.any(np.abs(wfm) > 1):
            print('Waveform must be in [-1, 1].  Clipping it!')
            A = np.clip(A, -1, 1)
        wfm = ((wfm + 1)/2 * (2**14 - 1))
        wfm = np.round(wfm).astype(np.dtype('H'))
        print(f'Writing binary waveform to {filepath}')
        with open(filepath, 'wb') as f:
            f.write(wfm.tobytes())


    ### These directly wrap SCPI commands that can be sent to the rigol AWG

    def shape(self, shape=None, ch=1):
        '''
        Change the waveform shape to a built-in value. Possible values are:
        SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|
        GAUSsian |HAVersine|LORentz|ARBPULSE|DUAltone
        '''
        return self.set_or_query(f'SOURCE{ch}:FUNC:SHAPE', shape)

    def output(self, state=None, ch=1):
        ''' Turn output state on or off '''
        if state is not None:
            state = 'ON' if state else 'OFF'
        return self.set_or_query(f':OUTPUT{ch}:STATE', state)

    def frequency(self, freq=None, ch=1):
        ''' Set frequency of AWG waveform.  Not the sample rate! '''
        return self.set_or_query(f':SOURCE{ch}:FREQ:FIX', freq)

        self.write(":SOURC{}:PER {}".format(ch, period))

    def period(self, period=None, ch=1):
        ''' Set period of AWG waveform.  Not the sample period! '''
        return self.set_or_query(f':SOURCE{ch}:PERiod:FIX', period)

    def phase(self, phase=None, ch=1):
        ''' Set phase offset of AWG waveform '''
        if phase is not None:
            phase = phase % 360
        return self.set_or_query(f':SOURCE{ch}:PHASe:ADJust', phase)

    def amplitude(self, amp=None, ch=1):
        ''' Set amplitude of AWG waveform.  Rigol defines this as peak-to-peak. '''
        return self.set_or_query(f':SOURCE{ch}:VOLTAGE:AMPL', amp)

    def offset(self, offset, ch=1):
        ''' Set offset of AWG waveform '''
        return self.set_or_query(f':SOURCE{ch}:VOLT:OFFS', offset)

    def output_resistance(self, r=None, ch=1):
        '''
        Manual says you can change output resistance from 1ohm to 10kohm
        I think this is just mistranslated chinese meaning the resistance of the load
        '''
        # Default is infinity??
        return self.set_or_query(f'OUTPUT{ch}:IMPEDANCE', r)

    def sync(self, state=None):
        ''' Can turn on/off the sync output (on rear) '''
        if state is not None:
            state = 'ON' if state else 'OFF'
        return self.set_or_query(f'OUTPUT{ch}:SYNC', state)

    def screensaver(self, state=None):
        ''' Turn the screensaver on or off.
        Screensaver causes problems with triggering because DG5000 is a piece of junk. '''
        if state is not None:
            state = 'ON' if state else 'OFF'
        return self.set_or_query(':DISP:SAV', state)

    def ramp_symmetry(self, percent=None, ch=1):
        ''' The symmetry of a ramp output.
        Refers to the sweep rates of increasing/decreasing ramps. '''
        return self.set_or_query(f'SOURCE{ch}:FUNC:RAMP:SYMM', percent)

    def dutycycle(self, percent=None, ch=1):
        ''' The duty cycle of a square output. '''
        return self.set_or_query(f'SOURCE{ch}:FUNC:SQUare:DCYCle', percent)

    def interp(self, mode=None):
        ''' Interpolation mode of volatile waveform.  LINear, SINC, OFF '''
        if mode is not None:
            if not isinstance(mode, str):
                # Use the boolean value of whatever the heck you passed
                mode = 'LIN' if mode else 'OFF'
        return self.set_or_query('TRACe:DATA:POINts:INTerpolate', mode)

    def error(self):
        ''' Get error message from rigol '''
        return self.query(':SYSTem:ERRor?').strip()

    # <<<<< For burst mode
    def ncycles(self, n=None, ch=1):
        ''' Set number of cycles that will be output in burst mode '''
        if n > 1000000:
            # Rigol does not give error, leaving you to waste a bunch of time discovering this
            raise Exception('Rigol can only pulse maximum 1,000,000 cycles')
        else:
            return self.set_or_query(f':SOURCE{ch}:BURST:NCYCLES', n)

    def trigsource(self, source=None, ch=1):
        ''' Change trigger source for burst mode. INTernal|EXTernal|MANual '''
        return self.set_or_query(f':SOURCE{ch}:BURST:TRIG:SOURCE', source)

    def trigger(self, ch=1):
        '''
        Send signal to rigol to trigger immediately.  Make sure that trigsource is set to MAN:
        trigsource('MAN')
        '''
        if self.trigsource() != 'MAN':
            raise Exception('You must first set trigsource to MANual')
        else:
            self.write(':SOURCE{}:BURST:TRIG IMM'.format(ch))

    def burstmode(self, mode=None, ch=1):
        '''Set the mode of burst mode.  I don't know what it means. 'TRIGgered|GATed|INFinity'''
        return self.set_or_query(f':SOURCE{ch}:BURST:MODE', mode)

    def burst(self, state=None, ch=1):
        ''' Turn the burst mode on or off '''
        # I think rigol is retarded, so it doesn't always turn off the burst mode on the first command
        # It switches something else off instead, but only if you set up a waveform after entering burstmode
        # The fix is to just issue the command twice..
        if state is not None:
            state = 'ON' if state else 'OFF'
            self.set_or_query(f':SOURCE{ch}:BURST:STATE', state)
        return self.set_or_query(f':SOURCE{ch}:BURST:STATE', state)

    # End for burst mode >>>>>

    def load_wfm_usbdrive(self, filename='wfm.RAF', wait=True):
        '''
        Load waveform from usb drive.  Should be a binary sequence of unsigned shorts.
        File needs to have extension .RAF
        This is the only way to reach the advertised number of waveform samples, or anywhere near it
        Should be able to go to 16 MPts on normal mode, 512 MPts on play mode, but this was not tested
        '''
        self.write(':MMEMory:CDIR "D:\"')
        self.write(f':MMEMory:LOAD "{filename}"')
        if wait:
            oldtimeout = self.conn.timeout
            # set timeout to a minute!
            self.conn.timeout = 60000
            # Rigol won't reply to this until it is done loading the waveform
            self.idn()
            self.conn.timeout = oldtimeout

    def writebinary(self, message, values):
        self.conn.write_binary_values(message, values, datatype='H', is_big_endian=False)

    def load_wfm_binary(self, wfm, ch=1):
        """
        TODO: write about all the bullshit involved with this
        I have seen these waveforms simply fail to trigger unless you wait a second after enabling the channel output
        You need to wait after loading this waveform and after turning the output on, sometimes an obscene amount of time?
        the "idle level" in burst mode will be the first value of the waveform ??
        """
        CHUNK = 2**14
        # vertical resolution
        VRES = 2**14
        # pad with zero value
        PADVAL = 2**13

        nA = len(wfm)
        A = np.array(wfm)

        print(f'You are trying to program {nA} pts')

        if np.any(np.abs(A) > 1):
            print('Waveform must be in [-1, 1].  Clipping it!')
            A = np.clip(A, -1, 1)

        # change from float interval [-1, 1] to int interval [0, 2^14-1]
        # Better to round than to floor
        A = np.int32(np.round(((A + 1) / 2 * (VRES - 1))))

        # Pad A to a magic length
        MAGICLENGTHS = np.array([2**14, 2**15, 2**16, 2**17, 2**18, 2**19])
        Nptsprog = MAGICLENGTHS[np.where(MAGICLENGTHS >= nA)[0][0]]
        A = np.append(A, PADVAL * np.ones(Nptsprog - nA, dtype='int32'))

        NptsProg = len(A)
        Nchunks = int(NptsProg / CHUNK)
        print(f'I am sending {NptsProg} points in {Nchunks} chunks')

        nptsProg = len(A)

        A2send = [A[i:i + CHUNK].tolist() for i in range(0, nptsProg, CHUNK)]

        # This command doesn't seem to be necessary?
        # Does it hurt?
        self.write(":DATA:POIN VOLATILE, " + str(nptsProg))

        for chunk in A2send[:-1]:
            self.writebinary(":TRAC:DATA:DAC16 VOLATILE,CON,", chunk)
            # What the manual says to do:
            #self.writebinary(":TRAC:DATA:DAC16 VOLATILE,CON,#532768", chunk)

            # Apparently need for USB (trial and error)
            #time.sleep(0.02)
            time.sleep(0.1)

        self.writebinary(":TRAC:DATA:DAC16 VOLATILE,END,", A2send[-1])


    def load_wfm_strings(self, waveform):
        '''
        Load some data as an arbitrary waveform to be output.
        Data will be normalized.  Use self.amplitude() to set the amplitude.
        Make sure that the output is off, because the command switches out of burst mode
        and otherwise will start outputting immediately.
        very limited number of samples can be written ~20,000
        Rigol will just stop responding and need to be restarted if you send too many points..
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

        #wfm_str = ','.join([str(w) for w in normwaveform])
        # I think rigol has a very small limit for input buffer, so can't send a massive string
        # So I am truncating the string to only show mV level.  This might piss me off in the future when I want better than mV accuracy.
        wfm_str = ','.join([str(round(w, 3)) for w in normwaveform])
        # This command switches out of burst mode for some stupid reason
        self.write(':TRAC:DATA VOLATILE,{}'.format(wfm_str))


    def load_wfm_ints(self, waveform):
        '''
        Load some data as an arbitrary waveform to be output.
        Data will be normalized.  Use self.amplitude() to set the amplitude.
        Make sure that the output is off, because the command switches out of burst mode
        and otherwise will start outputting immediately.
        convert to integers so that we can send more data points!
        Supposedly gets to about 40,000 samples
        I have seen it interpolate to only ~10,000 points, which is very unexpected!
        it should interpolate to the entire size of the waveform memory
        or at the very least, 2**14 = 16k samples!
        Maybe we need to issue a :TRACe:DATA:POINTs VOLATILE, <value> command to "set the number of initial points"
        '''
        # It seems to be possible to send bytes to the rigol instead of strings.  This would be much better.
        # But I haven't been able to figure out how to convert the data to the required format.  It's complicated.
        # Construct a string out of the waveform
        # TODO: Maybe also detect an offset to use?  Then we can make full use of the 12 bit resolution
        waveform = np.array(waveform, dtype=np.float32)
        maxamp = np.max(np.abs(waveform))
        if maxamp != 0:
            normwaveform = waveform/maxamp
        else:
            # Not a valid waveform anyway .. rigol will beep
            normwaveform = waveform
        normwaveform = ((normwaveform + 1) / 2 * 16383).astype(int).tolist()
        wfm_str = str(normwaveform).strip('[]').replace(' ', '')
        if len(wfm_str) > 261863:
            raise Exception('There is no way to know for sure, but I think Rigol will have a problem with the length of waveform you want to use.  Therefore I refuse to send it.')
        # This command switches out of burst mode for some stupid reason
        self.write(':TRAC:DATA:DAC VOLATILE,{}'.format(wfm_str))

    def color(self, c='RED'):
        '''
        Change the highlighting color on rigol screen for some reason
        'RED', 'DEEPRED', 'YELLOW', 'GREEN', 'AZURE', 'NAVYBLUE', 'BLUE', 'LILAC', 'PURPLE', 'ARGENT'
        '''
        self.write(':DISP:WIND:HLIG:COL {}'.format(c))

    def idn(self):
        return self.query('*IDN?').replace('\n', '')

    def read_volatile_wfm(self):
        '''
        Sometimes rigol outputs bizarre unaccountable waveforms.
        Use this to see what is in the volatile memory

        Takes a really long time
        Might fail outright
        Rigol is quite happy to randomly not respond to these kinds of commands
        '''
        numpackets = int(self.query(':TRACE:DATA:LOAD? VOLATILE'))
        numpoints = 2**14 * numpackets

        # from the programming guide:
        # This command is only available when the current output waveform is arbitrary waveform
        # and the type of the arbitrary waveform is volatile.

        # Otherwise it just gives a parameter error and doesn't reply..
        # In fact it can do that even when you ARE outputting a volatile arb. waveform..
        # Seems it only works when the packet size is 1

        values = []
        for i in range(1, numpoints + 1):
            # Takes about 5 ms, but I think much longer for different parts of the memory..
            val = int(self.query(f':TRACE:DATA:VAL? VOLATILE,{i}'))
            print(val)
            values.append(val)
            #time.sleep(.05)

    ### These use the wrapped SCPI commands to accomplish something useful

    def load_volatile_wfm(self, waveform, duration, offset=0, ch=1, interp=True):
        '''
        Load waveform into volatile memory, but don't trigger
        NOTICE: This will momentarily leave burst mode as a side-effect!  Thank RIGOL.
        The output will be toggled off to prevent output of free-running waveform before
        we turn burst mode back on.
        '''
        # toggling output state is slow, clunky, annoying, and should not be necessary.
        # it might also cause some spikes that could damage the device.
        # Also goes into high impedance output which could have some undesirable consequences.
        # Problem is that the command which loads in a volatile waveform switches rigol
        # out of burst mode automatically.  If the output is still enabled, you will get a
        # continuous pulse train until you can get back into burst mode.
        # contacted RIGOL about the problem but they did not help.  Seems there is no way around it.

        if len(waveform) > 512e3:
            raise Exception('Too many samples requested for rigol AWG (probably?)')

        burst_state = self.burst(ch=ch)
        # Only update waveform if necessary
        if np.any(waveform != self.volatilewfm):
            if burst_state:
                output_state = self.output(None, ch=ch)
                if output_state:
                    self.output(False, ch=ch)
                # This command switches out of burst mode for some stupid reason
                self.load_wfm_ints(waveform)
                self.burst(True, ch=ch)
                if output_state:
                    self.output(True, ch=ch)
            else:
                self.load_wfm_ints(waveform)
            self.volatilewfm = waveform
        else:
            # Just switch to the arbitrary waveform that is already in memory
            self.shape('USER', ch)
        freq = 1. / duration
        self.frequency(freq, ch=ch)
        maxamp = np.max(np.abs(waveform))
        self.amplitude(2*maxamp, ch=ch)
        # Apparently offset affects the arbitrary waveforms, too
        self.offset(offset, ch)
        # Turn on interpolation for IVs, off for steps
        self.interp(interp)

    def setup_burstmode(self, n=1, burstmode='TRIG', trigsource='MAN', ch=1):
        # Set up bursting
        self.burstmode(burstmode, ch=ch)
        self.trigsource(trigsource, ch=ch)
        self.ncycles(n, ch=ch)
        self.burst(True, ch=ch)

    def load_builtin_wfm(self, shape='SIN', duration=None, freq=None, amp=1, offset=0, phase=0, ch=1):
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
        self.shape(shape, ch=ch)
        # Rigol's definition of amplitude is peak-to-peak, which is unusual.
        self.amplitude(2*amp, ch=ch)
        self.offset(offset, ch=ch)
        self.frequency(freq, ch=ch)
        # Necessary because Rigol is terrible?
        self.phase(0, ch=ch)
        self.phase(phase, ch=ch)


    def continuous_builtin(self, shape='SIN', duration=None, freq=None, amp=1, offset=0, ch=1):
        self.load_builtin_wfm(shape=shape, duration=duration, freq=freq, amp=amp, offset=offset, ch=ch)
        # Get out of burst mode
        self.burst(False, ch=ch)
        self.output(True)

    def pulse_builtin(self, shape='SIN', duration=None, freq=None, amp=1, offset=0, phase=0, n=1, ch=1):
        '''
        Pulse a built-in waveform n times
        SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|GAUSsian|
        HAVersine|LORentz|ARBPULSE|DUAltone
        TODO: I think some of these waveforms have additional options.  Add them
        !! Will idle at the offset level in between pulses !!
        '''
        self.setup_burstmode(n=n)
        self.load_builtin_wfm(shape=shape, duration=duration, freq=freq, amp=amp, offset=offset, phase=phase, ch=ch)
        self.output(True, ch=ch)
        # Trigger rigol
        self.trigger(ch=ch)

    def continuous_arbitrary(self, waveform, duration=None, offset=0, ch=1):
        self.load_volatile_wfm(waveform, duration=duration, offset=offset, ch=ch)
        # Get out of burst mode
        self.burst(False, ch=ch)
        self.output(True)

    def pulse_arbitrary(self, waveform, duration, n=1, ch=1, offset=0, interp=True):
        '''
        Generate n pulses of the input waveform on Rigol AWG.
        Trigger immediately.
        Manual says you can use up to 128 Mpts, ~2^27, but for some reason you can't.
        Another part of the manual says it is limited to 512 kpts, but can't seem to do that either.
        !! will idle at the FIRST VALUE of waveform after the pulse is over !!
        '''
        # Load waveform
        self.load_volatile_wfm(waveform=waveform, duration=duration, offset=offset, ch=ch, interp=interp)
        self.setup_burstmode(n=n, ch=ch)
        self.output(True, ch=ch)
        # Trigger rigol
        self.trigger(ch=ch)

    def DC(self, value, ch=1):
        '''
        Do not rely heavily on this working.  It can be unpredictable.
        Don't know if you are even supposed to be able to set a DC level
        '''
        # One way that I know to make the rigol do DC..
        # Doesn't go straight to the DC level from where it was, because it has to turn off the output to load
        # a waveform.  this makes the annoying relay click
        # also beeps when you use value=0, but seems to work anyway
        #self.pulse_arbitrary([value, value], 1e-3, ch=ch)
        # This might be a better way
        # Goes directily to the next voltage
        # UNLESS you transition from abs(value) <= 2 t abs(value) > 2
        # then it will click and briefly output zero volts
        self.setup_burstmode(ch=ch)
        self.amplitude(.01, ch=ch)
        # Limited to +- 9.995
        self.offset(value, ch=ch)


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

    TODO: ResourceManager does not register TCP connections properly, and there
    does not seem to be an obvious way to tell quickly whether they are connected,
    because .list_resources() does not show them.
    This is the only reason Keithley2600 is Borg
    '''
    def __init__(self, addr='TCPIP::192.168.11.11::inst0::INSTR'):
        try:
            self.__dict__ = persistent_state.keithley_state
            self.connect(addr)
        except:
            print('Keithley connection failed at {}'.format(addr))

    def connect(self, addr='TCPIP::192.168.11.11::inst0::INSTR'):
        if not self.connected():
            self.conn = visa_rm.get_instrument(addr, open_timeout=0)
            # Expose a few methods directly to self
            self.write = self.conn.write
            self.query = self.conn.query
            self.ask = self.query
            self.read = self.conn.read
            self.read_raw = self.conn.read_raw
            self.close = self.conn.close
            # Store up to 100 loops in memory in case you forget to save them to disk
            self.data= deque(maxlen=100)
        # Always re-run lua file
        moduledir = os.path.split(__file__)[0]
        self.run_lua_file(os.path.join(moduledir, 'Keithley_2600.lua'))

    def connected(self):
        if hasattr(self, 'conn'):
            try:
                self.idn()
                return True
            except:
                pass
        return False

    def idn(self):
        return self.query('*IDN?').replace('\n', '')

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

    def iv(self, vlist, Irange=0, Ilimit=0, nplc=1, delay='smua.DELAY_AUTO', Vrange=0):
        '''
        range = 0 enables autoranging
        Wraps the SweepVList lua function defined on keithley
        '''
        # Send list of voltage values to keithley
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write('SweepVList(sweeplist, {}, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay, Vrange))

    def iv_4pt(self, vlist, Irange=0, Ilimit=0, nplc=1, delay='smua.DELAY_AUTO', Vrange=0):
        '''
        range = 0 enables autoranging
        Wraps the SweepVList lua function defined on keithley
        '''
        # Send list of voltage values to keithley
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write('SweepVList_4pt(sweeplist, {}, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay, Vrange))

    def vi(self, ilist, Vrange=0, Vlimit=0, nplc=1, delay='smua.DELAY_AUTO', Irange=None):
        '''
        range = 0 enables autoranging
        if Irange not passed, it will be max(abs(ilist))
        Wraps the SweepIList lua function defined on keithley
        '''

        # Send list of voltage values to keithley
        self.send_list(ilist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        if Irange is None:
            # Fix the current source range, as I have had instability problems that are different
            # for different ranges
            Irange = np.max(np.abs(ilist))
        self.write('SweepIList(sweeplist, {}, {}, {}, {}, {})'.format(Vrange, Vlimit, nplc, delay, Irange))

    def it(self, sourceVA=0, sourceVB=0, points=10, interval=.1, rangeI=0, limitI=0, nplc=1):
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
        donesweeping = not bool(float(self.query('print(status.operation.sweeping.condition)')))
        # works with smua.measure.overlappediv()
        donemeasuring = not bool(float(self.query('print(status.operation.measuring.condition)')))
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
            answer = float(self.query('print(status.operation.sweeping.condition)'))
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
            end = int(float(self.query('print({}.n)'.format(buffer))))
        # makes keithley give numbers in ascii
        # self.write('format.data = format.ASCII')
        #readingstr = self.query('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
        #return np.float64(readingstr.split(', '))

        # Makes keithley give numbers in binary float64
        # Should be much faster?
        self.write('format.data = format.REAL64')
        self.write('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
        # reply comes back with #0 or something in the beginning and a newline at the end
        raw = self.read_raw()[2:-1]
        # TODO: replace nanvals here, not in get_data
        data_array = np.fromstring(raw, dtype=np.float64)
        data_array = self.replace_nanvals(data_array)
        return data_array

    def get_data(self, start=1, end=None, history=True):
        '''
        Ask Keithley to print out the data arrays of interest (I, V, t, ...)
        Parse the strings into python arrays
        Return dict of arrays
        dict can also contain scalar values or other meta data

        Can pass start and end values if you want just a specific part of the arrays
        '''
        numpts = int(float(self.query('print(smua.nvbuffer1.n)')))
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
            source = self.query('print(smua.source.func)')
            source = float(source)
            if source:
                # Returns 1.0 for voltage source (smua.OUTPUT_DCVOLTS)
                out['source'] = 'V'
                out['V'] = self.read_buffer('smua.nvbuffer2', 'sourcevalues', start, end)
                Vmeasured = self.read_buffer('smua.nvbuffer2', 'readings', start, end)
                out['Vmeasured'] = Vmeasured
                out['units']['Vmeasured'] = 'V'
                I = self.read_buffer('smua.nvbuffer1', 'readings', start, end)
                out['I'] = I
                out['Icomp'] = float(self.query('print(smua.source.limiti)'))
            else:
                # Current source
                out['source'] = 'I'
                out['Vrange'] = float(self.query('print(smua.nvbuffer2.measureranges[1])'))
                out['Vcomp'] = float(self.query('print(smua.source.limitv)'))

                out['I'] = self.read_buffer('smua.nvbuffer1', 'sourcevalues', start, end)
                Imeasured = self.read_buffer('smua.nvbuffer1', 'readings', start, end)
                out['Imeasured'] = Imeasured
                out['units']['Imeasured'] = 'A'
                V = self.read_buffer('smua.nvbuffer2', 'readings', start, end)
                out['V'] = V

            out['t'] = self.read_buffer('smua.nvbuffer2', 'timestamps', start, end)
            out['Irange'] = self.read_buffer('smua.nvbuffer1', 'measureranges', start, end)
            out['Vrange'] = self.read_buffer('smua.nvbuffer2', 'measureranges', start, end)

            out['units']['I'] = 'A'
            out['units']['V'] = 'V'

            out['idn'] = self.idn()

        else:
            empty = np.array([])
            out = dict(t=empty, V=empty, I=empty, Vmeasured=empty)
            out['units'] = {}
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
# UF2000 Prober #########################################
#########################################################
class UF2000Prober(object):
    '''
    T Hennen modified 2018-12-11

    !!! Important !!!
    There are two ways to move the prober: By index, and by micron
    These two reference frames are centered on different locations
    indexing system is centered on a home device, so depends on how the wafer/coupon is loaded
    micron system is centered somewhere far away from the chuck

    The coordinate systems sound easy, but will confuse you for days
    in part because the coordinate system and units used for setting
    and getting the position are sometimes different and sometimes the same!!
    e.g. when reading the position in microns, the x and y axes are reflected!!

    I attempted to hide all of this nonsense from the user of this class
    !!!!!!

    UF2000 has its own device indexing system which requires some probably horrible setup that you need to do for each wafer.
    But we also have the option to specify directly position in micrometers, then we can handle the positioning here in the python universe.

    The indexing system is referenced to some home device, but the micron coordinate system is referenced to the chuck and centered god knows where.

    There are a few ways we can deal with this.  Right now I choose to deal with it outside of the class, so that it does not have any hidden state.

    UF2000Prober has no concept of what a device is or where they are located, except the home position is on the home device.

    Prober coordinate system is not intuitive, because the prober moves the chuck/wafer, not the probe, and has inverted Y axis
    But I like to think about moving the probe, and +X should be right, +Y should be up, in other words, the lab frame
    I will attempt to shield the user completely from the probers coordinate system, and always use the lab frame
    '''

    def __init__(self, idstring = 'GPIB0::5::INSTR'):
        self.inst = visa_rm.open_resource(idstring)
        self.inst.timeout = 3000
        # UF2000 seems to call this position home, could depend on the setup!
        self.home_indices = (128, 128)
        # Very roughly the center of the chuck...
        self.center_position_um = (160_126.3, 388_264.5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.zDn()
        #print('Closing UF2000')
        return False

    ####### Communication ########

    def write_8min(self, message, stbList):
        #will wait 8 min for stblist, with no breaking for stalls
        self.inst.write(message)
        self.errorCheck()
        return self.waitforStatusByte_8min(stbList)

    def write(self, message, stbList):

        self.inst.write(message)
        #self.errorCheck()
        return self.waitForStatusByte(stbList)

    def waitForSRQ_readStatus(self, timeout):
        self.inst.timeout = timeout
        self.inst.wait_for_srq()
        del self.inst.timeout
        return self.inst.read_stb()

    def waitforStatusByte_8min(self, stb ):
        #this function has no provision for 'stalls' it will wait for the stb until the for loop completes
        # this is necessary so that the 'stall' doesn't terminate the loop before the next wafer is loaded
        if type(stb) == int:
            stb = [stb]
        #
        old_a = 0
        #
        for _ in range(int(1E6)):
            a = self.inst.read_stb()


            """
            if a == 76:
                #Error!
                self.errorCheck()
                self.errorClearanceReq()

            """
            if a != old_a:
                #print('STB '+str(a)+': '+self.getSTBMessage(a))
                old_a = a
            if a in stb:
                return a

            time.sleep(.0005)
            pass
        else:
            #this will execute if the above for loop executes sucessfully--that is, 1e6 iterations were completed, and STB
            raise UF2000ProberError

    def waitForStatusByte(self, stb):
        if type(stb) == int:
            stb = [stb]
        #
        old_a = 0
        #
        for _ in range(int(1E6)):
            a = self.inst.read_stb()


            """
            if a == 76:
                #Error!
                self.errorCheck()
                self.errorClearanceReq()

            """
            if a != old_a:
                #print('STB '+str(a)+': '+self.getSTBMessage(a))
                old_a = a
            if (a in stb) :
                return a
            if a==3 and _ > 6000:
                print('**********\n**********\nSTB 3 Stall\n**********\n**********\n')
                return(a)

            if a==4 and _ >6000:  #ONLY FOR 'makeContact = No --comment out otherwise
                print('**********\n**********\nSTB 4 Stall\n**********\n**********\n')
                return(a)

            time.sleep(.0005)
            pass
        else:
            #this will execute if the above for loop executes sucessfully--that is, 1e6 iterations were completed, and STB was never 3 or 4
            raise UF2000ProberError

    def query(self, queryStr):
        for retry_nr in range(4):
            try:
                result = self.inst.query(queryStr)
                if result[0] != queryStr and queryStr != 'ms':
                    raise UF2000ProberError('Return String not well-formed:%s %s' %(queryStr, result))
                self.errorCheck()
                return result
            except pyvisa.errors.VisaIOError as e:
                #emailData("alexander.elias@gmail.com")
                traceback.print_exc()
                time.sleep(10)
                continue

    def waitForSTB(self):
        stb1 = str(self.inst.read_stb())
        #self.inst.wait_for_srq()

        stb2 = str(self.inst.read_stb())
        return int(stb1)

    def errorCheck(self):
        errorTypeDict = {'S': 'System Error: ',
                        'E': 'Error State: ',
                        'O': 'Operator Call: ',
                        'W': 'Warning Condition: ',
                        'I': 'Information: '}
        errorCodeDict = {'0650' : 'GPIB Receive Error ',
                         '0651': 'GPIB Transmit Error ',
                         '0660': 'GPIB Command Format Invalid ',
                         '0661': 'GPIB Command Execution Error ',
                         '0665': 'GPIB Stop Command Received ',
                         '0667': 'GPIB Communication Timeout Error ',
                         '0669': 'GPIB Timeout Error '}
        for retry_nr in range(1):
            try:
                rawError = self.inst.query('E')
                break
            except pyvisa.errors.VisaIOError as e:
                #emailData("alexander.elias@gmail.com")
                traceback.print_exc()
                time.sleep(10)
                continue

        if rawError[0] != 'E':
            raise UF2000ProberError('Return String not well-formed:%s %s' %(rawError, 'E'))

        errorTypeCode = rawError[1]
        errorCode = rawError[2:6]
        errorString = errorTypeDict.get(errorTypeCode , 'Unknown Type') + errorCodeDict.get(errorCode, 'Unknown Code: ') + errorCode
        #self.errorClearanceReq()
        return errorString

    def getID(self):
        '''returns prober ID string'''
        return self.query('B')
    def getSTBMessage(self, stbIntStr):
        stbDict = {'64': 'GPIB inital setting done',
                   '65': 'Absolute Value Travel Done',
                   '66': 'Coordinate Travel Done',
                   '67': 'Z-Up (Test Start)',
                   '68': 'Z-Down',
                   '69': 'Marking Done',
                   '70': 'Wafer Loading Done',
                   '71': 'Wafer Unloading Done',
                   '72': 'Lot End',
                   '74': 'Out of Probing Area',
                   '75': 'Prober Initial Setting Done',
                   '76': 'Error: Lock/Unlock cassette',
                   '77': 'Index Setting Done',
                   '78': 'Pass Counting Up/Execution Error',
                   '79': 'Fail Counting Up Done',
                   '80': 'Wafer Unloaded',
                   '81': 'Wafer End',
                   '82': 'Cassette End',
                   '84': 'Alignment Rejection Error',
                   '85': 'Stop Command Received',
                   '86': 'Print Data Receiving Done',
                   '87': 'Warning Error',
                   '88': 'Test Start (Count Not Needed)',
                   '89': 'Needle Cleaning Done',
                   '90': 'Probing Stop',
                   '91': 'Probing Start',
                   '92': 'Z-Up/Down Done',
                   '93': 'Hot Chuck Cont. Command Received',
                   '94': 'Lot Done',
                   '98': 'Command Normally Done',
                   '99': 'Command Abnormally Done',
                   '100': 'Test Done Received',
                   '101': '(em command correct end)',
                   '103': 'Map Data Downloading Normally Done',
                   '104': 'Map Data Downloading Abormally Done',
                   '105': 'Able To Adjust Needle Height',
                   '107': 'Binary Data Uploading',
                   '108': 'Binary Data Uploading Finish',
                   '110': 'Needle Mark OK',
                   '111': 'Needle Mark NG',
                   '112': 'Cassette Sensing Done',
                   '113': 'Re-Alignment Done',
                   '114': 'Auto Needle Alignment Normally Done',
                   '115': 'Auto Needle Alignment Abnormally Done',
                   '116': 'Chuck Height Setting Done',
                   '117': 'Continuous Fail Error',
                   '118': 'Wafer Loading Done',
                   '119': 'Error Recovery Done (Wafer Centering Complete)',
                   '120': 'Start Normally Done',
                   '121': 'Start Abnormally Done',
                   '122': 'Probe Mark Insapection Finish',
                   '123': 'Fail Mark Inspection Finish',
                   '124': 'Preload Done',
                   '125': 'Probing Stop by GEM Host',
                   '127': 'Travel Done',
                   '6': '6: Probing...',
                   '16': '16: Wafer Loading...',
                   '17': '17: Wafer Unloading...',
                   '30': '30: Waiting For New Cassette...',
                   '34': '34: Cassette Ready...'}
        return stbDict.get(str(stbIntStr), 'Unknown Status Byte: '+str(stbIntStr))

    def proberStatusReq(self):
        rawStatus =  self.query('ms')
        status = rawStatus[2]
        statusDict = {'W': 'Cassette Process Going on',
                      ' ':'Status process Not going on',
                      'I': 'Waiting for Lot Start',
                      'C': 'Card Replacement Going On',
                      'R': 'Lot Process Going On',
                      'E': 'Waiting For Operator\'s Help With an Error '}

        return statusDict[status], status

    def probeClean(self):
        '''intiate tip clean'''
        self.write('W', [89])

    def pushStart(self):
        self.write('st', [120,121])

    def stopTesting(self):
        self.write('K', [90,85,26])

    def errorClearanceReq(self):
        self.inst.write('es')

    def lotEndReq(self):
        self.write('le', [98,99])

    def getWaferID(self):
        '''returns wafer ID as scanned by prober OCR, removes leading/trailing character'''
        return self.query('b')[1:-1]

    def pollStatusByte(self, doneResponse):
        done = False
        while done == False:
            response = self.waitForSTB()
            time.sleep(1)
            print(response)
            ent = raw_input
            if response == doneResponse:
                done = True
        return


    ######## Wafer loading ########
    def loadWafer(self):
        return self.write_8min('L', [70,94])
        """
        if self.waitForStatusByte([2,17]) == 2:
            #print ('Needle Cleaning on Unit...')
        elif self.waitForStatusByte([2,17]) == 17:
            #print("Wafer Unloading...")
        else:
            #print('Loading Next Wafer...')
        """
    def unLoadWafer(self):
        #TODO increase timeout -- wafer takes a while to unload
        self.write('U', [71])

    ######## Temperature control ########
    def getChuckTemp(self):
        temp = self.inst.query('f1')
        return float(temp)

    def setChuckTemp(self, temp):
        if temp < 15 or temp > 150:
            print('Temperature out of range, must be 15->150C')
            return

        temp = temp*10 #conver degree C to 0.1 C
        temp = str(temp).zfill(4)
        self.write('h{}'.format(temp), 93)

    def waitForTemp(self):
        self.inst.timeout = None
        tempStr = self.inst.query('f')
        if len(tempStr)!=11:
            print('Hot Chuck not enabled!')
            raise UF2000ProberError


        currTemp =  float(tempStr[1:5])/10.
        setTemp =   float(tempStr[5:9])/10.
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        #sys.stdout.write('Waiting on STB ')
        #print(('Waiting for Temperature = {}...'.format(setTemp)))
        while abs(currTemp-setTemp)>0.2:

            time.sleep(10)
            # sys.stdout.write(spinner.next())  # write the next character
            # sys.stdout.flush()                # flush stdout buffer (actual character display)
            # time.sleep(0.33)
            # sys.stdout.write('\b')            # erase the last written char
            tempStr = self.inst.query('f')
            #print(tempStr)
            currTemp =  float(tempStr[1:5])/10.
            setTemp =   float(tempStr[5:9])/10.
            print(('Set Temp: {}, CurrentTemp: {}, Diff: {}'.format(setTemp, currTemp, abs(currTemp-setTemp))))
        self.inst.timeout = 3000
        print('Set Temperature Achieved!')


    ######## Movement ########
    def zDn(self):
        '''moves chuck to NO_CONTACT position'''
        #hp.shortAll()
        self.write('D', [68])
        #self.pollStatusByte(68)
        #self.waitForSTB()
        pass

    def zUp(self):
        '''moves chuck unto CONTACT position'''
        #hp.shortAll()
        #print('Zupping...')
        self.write('Z', [67])
        #self.pollStatusByte(67)
        #self.waitForSTB()
        pass

    def goHome(self):
        # Prober seems to call home position 128, 128.  Could be wrong!
        # I think it depends on the set up
        self.moveAbsolute(0, 0)

    # Reference frame conversion
    def prober_to_lab_indices(self, xprober, yprober):
        xlab = -xprober + self.home_indices[0]
        ylab = yprober - self.home_indices[1]
        return xlab, ylab

    def lab_to_prober_indices(self, xlab, ylab):
        xprober = -xlab + self.home_indices[0]
        yprober = ylab + self.home_indices[1]
        return xprober, yprober

    def prober_to_lab_um(self, xprober, yprober):
        xlab = xprober - self.center_position_um[0]
        ylab = -yprober + self.center_position_um[1]
        return xlab, ylab

    def lab_to_prober_um(self, xlab, ylab):
        xprober = xlab + self.center_position_um[0]
        yprober = -ylab + self.center_position_um[1]
        return xprober, yprober

    # Index based -- moves by unit cell and has only integer values
    def getPosition(self):
        '''
        Get position indices
        Home position is subtracted
        converted to +Y up, +X right in the lab frame
        '''
        rawPosString = self.query('Q')
        y = int(rawPosString[2:5])
        x = int(rawPosString[6:9])
        return self.prober_to_lab_indices(x, y)

    def moveAbsolute(self, absX, absY):
        '''
        Move to a given index in the lab frame
        '''
        X0, Y0 = self.getPosition()
        if (X0, Y0) == (absX, absY):
            return (X0, Y0)
        Xrel = absX - X0
        Yrel = absY - Y0
        self.moveRelative(Xrel, Yrel)
        newPos = self.getPosition()
        return newPos

    def moveRelative(self, x_rel, y_rel):
        '''
        Moves by whatever the prober thinks is the unit cell distance
        with respect to a view of the top of the wafer from the front of the machine:
        +X moves probe right
        +Y moves probe up
        '''
        if((x_rel, y_rel) == (0,0)):
            return self.getPosition()
        x_rel_prober = -x_rel
        y_rel_prober = y_rel
        strX = '%+04d' % x_rel_prober
        strY = '%+04d' % y_rel_prober
        moveString = 'SY'+ strY + 'X' + strX
        self.write(moveString, [66,67,74])
        return self.getPosition()


    # Micron based
    def getHomePosition_um(self):
        '''
        This is to center the micron coordinate system on the home device
        Does not set self.center_position_um, but you could do that
        '''
        self.goHome()
        home = self.getPosition_um()
        return home

    def getPosition_um(self):
        '''
        The position UF2000 thinks it is in, in um
        '''
        pos_str = self.query('R')
        # Manual says the unit is 1e-7 meter
        y, x = int(pos_str[2:9])/10, int(pos_str[10:-2])/10
        return self.prober_to_lab_um(x, y)

    def moveRelative_um(self, xum_rel, yum_rel):
        '''
        with respect to a view of the top of the wafer from the front of the machine:
        +X moves probe right
        +Y moves probe up
        '''
        xum_rel_prober = -xum_rel
        yum_rel_prober = yum_rel
        str_xum = '{:+07d}'.format(int(round(xum_rel_prober)))
        str_yum = '{:+07d}'.format(int(round(yum_rel_prober)))
        # manual says the unit is 1e-6 meter
        moveString = 'AY{}X{}'.format(str_yum, str_xum)
        self.write(moveString, [65, 67, 74])

    def moveAbsolute_um(self, xum_abs, yum_abs):
        xum_curr, yum_curr = self.getPosition_um()
        print(('Current position:     {}, {}'.format(xum_curr, yum_curr)))
        print(('Destination position: {}, {}'.format(xum_abs, yum_abs)))
        xum_rel = int(xum_abs - xum_curr)
        yum_rel = int(yum_abs - yum_curr)
        self.moveRelative_um(xum_rel, yum_rel)


#########################################################
# Eurotherm 2408 -- #################
#########################################################
class Eurotherm2408(object):
    '''
    This uses some dumb proprietary EI-BISYNCH protocol over serial.
    Make the connections DB2 -> HF, DB3 -> HE, DB5 -> HD.
    You can also use modbus.
    '''
    def __init__(self, addr='COM32', gid=0, uid=1):
        # BORG
        self.__dict__ = persistent_state.eurotherm_state
        self.connect(addr, gid, uid)

    def connect(self, addr='COM32', gid=0, uid=1):
        if not self.connected():
            self.conn = serial.Serial(addr, timeout=1, bytesize=7, parity=serial.PARITY_EVEN)
            self.gid = gid
            self.uid = uid

    def connected(self):
        if hasattr(self,'conn'):
            return self.conn.isOpen()
        else:
            return False

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
        data = format(data, '.1f')
        bcc = chr(reduce(xor, (mnemonic + data + ETX).encode()))
        msg = EOT + gid + gid + uid + uid + STX + mnemonic + data + ETX + bcc
        # print(msg)
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
        if reply == ACK.encode():
            return True
        elif reply == NAK.encode():
            return False
        else:
            raise Exception('Eurotherm not connected properly')

    def read_data(self, mnemonic):
        EOT = '\x04'
        ENQ = '\x05'
        gid = str(self.gid)
        uid = str(self.uid)
        # Poll
        # [EOT] (GID) (GID) (UID) (UID) (CHAN) (C1) (C2) [ENQ]
        # CHAN optional, will be returned only if sent
        poll = EOT + gid + gid + uid + uid + mnemonic + ENQ
        self.conn.write(poll.encode())

        # Wait?
        time.sleep(.05)

        # Reply
        # [STX] (CHAN) (C1) (C2) <DATA> [ETX] (BCC)
        reply = self.conn.read_all()
        try:
            return float(reply[3:-2])
        except:
            print('Failed to read Eurotherm 2408 temperature.')
            return np.nan

    def read_temp(self):
        return float(self.read_data('PV'))

    def set_temp(self, value):
        return self.write_data('SL', value)

    def output_level(self):
        return self.read_data('OP')

    def status(self):
        statusdict = {1: 'Reset',
                      2: 'Run',
                      3: 'Hold',
                      4: 'Holdback',
                      5: 'Complete'}
        return self.read_data('PC')


#########################################################
# Measurement Computing USB1208HS DAQ ###################
#########################################################
class USB2708HS(object):
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
        Can also pass volts instead of dacval
        Voltage values that don't make sense for my current set up are disallowed.
        '''
        board_num = 0
        ao_range = self.enums.ULRange.BIP10VOLTS

        # Can pass dacval or volts.  Prefer dacval.
        if dacval is None:
            # You better have passed volts...
            dacval = self.ul.from_eng_units(board_num, ao_range, volts)
        else:
            dacval = int(dacval)
            volts = self.ul.to_eng_units(board_num, ao_range, dacval)

        # Just protect against doing something that doesn't make sense
        # TODO: remove this restriction from this part of the code, should go in the application part
        if ch == 0 and volts > 0:
            print('I disallow voltage value {} for analog output {}'.format(volts, ch))
            return
        elif ch == 1 and volts < 0:
            print('I disallow voltage value {} for analog output {}'.format(volts, ch))
            return
        else:
            print('Setting analog out {} to {} ({} V)'.format(ch, dacval, volts))

        try:
            self.ul.a_out(board_num, ch, ao_range, int(dacval))
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

    def connect(self, addr):
        self.conn = visa_rm.get_instrument(addr)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.query = self.conn.query
        self.ask = self.query
        self.read = self.conn.read
        self.read_raw = self.conn.read_raw
        self.close = self.conn.close

    def idn(self):
        return self.query('*IDN?').replace('\n', '')

    def bandwidth(self, channel=1, bandwidth=33e9):
        self.write('CH' + str(channel) + ':BAN ' + str(bandwidth))

    def scale(self, channel=1, scale=0.0625):
        self.write('CH' + str(channel) + ':SCAle ' + str(scale))

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
            self.write('TRIG:A:EDGE:SOUrce CH ' + str(source))
        self.write('TRIG:A:LEVEL ' + str(level))
        self.write('ACQ:STOPA SEQUENCE')
        self.write('ACQ:STATE 1')
        if edge == 'r':
            self.write('TRIG:A:EDGE:SLO RIS')
        elif edge == 'f':
            self.write('TRIG:A:EDGE:SLO FALL')
        else:
            self.write('TRIG:A:EDGE:SLO EIT')
        triggerstate = self.query('TRIG:STATE?')
        while 'REA' not in triggerstate or 'SAVE' in triggerstate:
            self.write('ACQ:STATE 1')
            triggerstate = self.query('TRIG:STATE?')

    def get_curve(self, channel=1):
        self.write('HEAD 0')
        self.write('WFMOUTPRE:BYT_NR 1')
        self.write('WFMOUTPRE:BIT_NR 8')
        self.write('DATA:ENC RPB')
        self.write('DATA:SOURCE CH' + str(channel))
        rl = int(self.query('HOR:RECO?'))

        pre = self.query('WFMOutpre?')
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
        trigger_str = self.query('TRIG:STATE?')
        return trigger_str == 'READY\n'

    def trigger_position(self, position):
        self.write('HORIZONTAL:POSITION ' + str(position))


#########################################################
# Erik Wichmann's Digipot circuit ###################
#########################################################
class WichmannDigipot_new(object):
    '''
    Probing circuit developed by Erik Wichmann to provide remote series resistance switching
    There are two digipots on board.  You can use a single digipot or connect the two in series or in parallel
    There are 31 ~log spaced resistance values per digipot

    TODO: Change arduino command system to not need entire state in one chunk
    should be three commands, for setting wiper1, wiper2, and relay

    TODO: Is there a way to poll the current state from the microcontroller?
    The class instance might not be aware of it when we first connect.

    TODO: make a test routine that takes a few seconds to measure that everything is working properly.  belongs in measure.py
    TODO: In addition to LCDs that display that the communication is working, we need a programmatic way to verify the connections as well
    '''
    def __init__(self, addr=None):
        # BORG
        self.__dict__ = persistent_state.digipot_state
        self.connect(addr)
        # map from setting to resistance -- needs to be measured by source meter
        # TODO: does the second digipot have a different calibration?
        #self.Rlist = [43080, 38366, 34242, 30547, 27261, 24315, 21719, 19385, 17313,
        #              15441, 13801, 12324, 11022, 8805, 7061, 5670, 4539, 3667, 2964,
        #              2416, 1965, 1596, 1313, 1089, 906, 716, 576, 478, 432, 384, 349,
        #              324, 306, 306]
        # Keithley calibration at 1V applied 2019-07-17
        self.Rlist = [43158.62, 38438.63, 34301.27, 30596.64, 27306.63, 24354.61, 21752.65,
                      19413.07, 17336.84, 15461.77, 13818.91, 12338.34, 11033.65, 8812.41,
                      7064.97, 5672.71, 4539.82, 3666.53, 2961.41, 2412.55, 1960.89, 1591.29,
                      1307.28, 1083.48, 902.42, 711.69, 570.92, 472.24, 426.55, 377.22, 342.16,
                      316.79, 299.09, 299.06]
        self.Rmap = {n:v for n,v in enumerate(self.Rlist)}

    def connect(self, addr=None):
        if not self.connected():
            if addr is None:
                # Connect to the first thing you see that has Leonardo in the description
                matches = list(comgrep('Leonardo'))
                if any(matches):
                    addr = matches[0].device
                else:
                    print('WichmannDigipot could not find Leonardo')
                    return
            self.conn = serial.Serial(addr, timeout=1)
            self.write = self.conn.write
            self.open = self.conn.open
            self.close = self.conn.close
            if self.connected():
                print(f'Connected to digipot on {addr}')

    @property
    def Rstate(self):
        # Returns the current set resistance state
        # TODO: depends on the configuration (single, series, parallel)
        return self.Rmap[self.wiper1state]

    @Rstate.setter
    def Rstate(self, val):
        self.set_R(val)

    @property
    def wiper0state(self):
        self.write(f'get_wiper 0 \n'.encode())
        time.sleep(5e-3)
        return int(self.conn.read_all().decode())

    @property
    def wiper1state(self):
        self.write(f'get_wiper 1 \n'.encode())
        time.sleep(5e-3)
        return int(self.conn.read_all().decode())

    @property
    def read(self):
         return self.conn.read_all().decode()

    def connected(self):
        if hasattr(self,'conn'):
            return self.conn.isOpen()
        else:
            return False

    def writeCMD(self,textstr):
        ''' 
        Debugging tool. 
        Send serial Command and print returned answer like a Serial monitor
        '''
        self.write((textstr+' \n').encode())
        time.sleep(5e-3)
        print(self.conn.read_all().decode())

    def set_bypass(self, state):
        '''
        State:
        True = connected
        False = not connected
        '''
        self.write(f'bypass {int(state)} \n'.encode())
        self.bypassstate = state
        #Wait until the AVR has sent a message Back
        time.sleep(5e-3)
        self.conn.read_all().decode()

    def set_wiper(self, state, n=1):
        '''
        Change the digipot wiper setting 
        n=1 is main potentiometer on chip 
        0 ist only used in parallel/series Mode
        '''
        self.write(f'wiper {n} {state}'.encode())
        #Wait until the AVR has sent a message Back
        time.sleep(5e-3)
        self.conn.read_all().decode()

    def set_R(self, R, n=1):
        if R == 0:
            self.set_bypass(1)
            #Set wiper to highest value
            self.set_wiper(0)
            return 0
        else:
            # Find closest resistance value
            # I hope the dictionary returns values and keys in the same order
            Rmap = list(self.Rmap.values())
            wvals= list(self.Rmap.keys())
            i_closest = np.argmin(np.abs(R - np.array(Rmap)))
            R_closest = Rmap[i_closest]
            w_closest = wvals[i_closest]
            print(i_closest)
            print(self.Rmap[i_closest])
            self.set_wiper(w_closest, n)
            # Could have sent one command, but I'm sending two
            self.set_bypass(0)
            time.sleep(1e-3)
            return R_closest

    def set_series_R(self, R):
        # TODO calculate nearest series value
        pass

    def set_parallel_R(self, R):
        # TODO calculate nearest parallel value
        pass

class WichmannDigipot(object):
    '''
    Probing circuit developed by Erik Wichmann to provide remote series resistance switching
    There are two digipots on board.  You can use a single digipot or connect the two in series or in parallel
    There are 31 ~log spaced resistance values per digipot

    TODO: Change arduino command system to not need entire state in one chunk
    should be three commands, for setting wiper1, wiper2, and relay

    TODO: Is there a way to poll the current state from the microcontroller?
    The class instance might not be aware of it when we first connect.

    TODO: Shouldn't relay = 1 mean that the input is connected to the output?

    TODO: make a test routine that takes a few seconds to measure that everything is working properly.  belongs in measure.py
    TODO: In addition to LCDs that display that the communication is working, we need a programmatic way to verify the connections as well
    '''
    def __init__(self, addr=None):
        # BORG
        self.__dict__ = persistent_state.digipot_state
        self.connect(addr)
        # map from setting to resistance -- needs to be measured by source meter
        # TODO: does the second digipot have a different calibration?
        #self.Rlist = [43080, 38366, 34242, 30547, 27261, 24315, 21719, 19385, 17313,
        #              15441, 13801, 12324, 11022, 8805, 7061, 5670, 4539, 3667, 2964,
        #              2416, 1965, 1596, 1313, 1089, 906, 716, 576, 478, 432, 384, 349,
        #              324, 306, 306]
        # Keithley calibration at 1V applied 2019-07-17
        self.Rlist = [43158.62, 38438.63, 34301.27, 30596.64, 27306.63, 24354.61, 21752.65,
                      19413.07, 17336.84, 15461.77, 13818.91, 12338.34, 11033.65, 8812.41,
                      7064.97, 5672.71, 4539.82, 3666.53, 2961.41, 2412.55, 1960.89, 1591.29,
                      1307.28, 1083.48, 902.42, 711.69, 570.92, 472.24, 426.55, 377.22, 342.16,
                      316.79, 299.09, 299.06]
        self.Rmap = {n:v for n,v in enumerate(self.Rlist)}

    def connect(self, addr=None):
        if not self.connected():
            if addr is None:
                # Connect to the first thing you see that has Leonardo in the description
                matches = list(comgrep('Leonardo'))
                if any(matches):
                    addr = matches[0].device
                else:
                    print('WichmannDigipot could not find Leonardo')
                    return
            self.conn = serial.Serial(addr, timeout=1)
            self.write = self.conn.write
            self.close = self.conn.close
            # We don't know what state we are in initially
            # For now we will just set them all to 1 when we connect
            self.bypassstate = 1
            self.wiper1state = 0
            self.wiper2state = 0
            self.write('0 0 1'.encode())
            if self.connected():
                print(f'Connected to digipot on {addr}')

    @property
    def Rstate(self):
        # Returns the current set resistance state
        # TODO: depends on the configuration (single, series, parallel)
        return self.Rmap[self.wiper2state]

    def connected(self):
        if hasattr(self,'conn'):
            return self.conn.isOpen()
        else:
            return False

    def writeRead(self,textstr):
        # Simple send serial Command and print returned answer
        time.sleep(5e-3)
        print(self.conn.read_all())
        self.write(textstr)
        time.sleep(5e-3)
        print(self.conn.read_all())
        time.sleep(5e-3)
        print(self.conn.read_all())

    def set_bypass(self, state):
        '''
        State:
        True = connected
        False = not connected
        '''
        # Keep current wiper state, set the bypass relay state
        w1 = self.wiper1state
        w2 = self.wiper2state
        self.write(f'{w1} {w2} {state}'.encode())
        self.bypassstate = state
        #Wait until the AVR has sent a message Back
        time.sleep(5e-3)
        return self.conn.read_all().decode()

    def set_wiper(self, state, n=2):
        '''
        Change the digipot wiper setting
        n=2 is main potentiometer on chip
        '''
        bypass = self.bypassstate

        if n==1:
            w2 = self.wiper2state
            self.write(f'{state} {w2} {bypass}'.encode())
            self.wiper1state = state
        elif n == 2:
            w1 = self.wiper1state
            self.write(f'{w1} {state} {bypass}'.encode())
            self.wiper2state = state
        #Wait until the AVR has sent a message Back
        time.sleep(5e-3)
        return self.conn.read_all().decode()

    def set_R(self, R, n=2):
        if R == 0:
            self.set_bypass(1)
            #Set wiper to highest value
            self.set_wiper(0)
            return 0
        else:
            # Find closest resistance value
            # I hope the dictionary returns values and keys in the same order
            Rmap = list(self.Rmap.values())
            wvals= list(self.Rmap.keys())
            i_closest = np.argmin(np.abs(R - np.array(Rmap)))
            R_closest = Rmap[i_closest]
            w_closest = wvals[i_closest]
            print(i_closest)
            self.set_wiper(w_closest, n)
            # Could have sent one command, but I'm sending two
            self.set_bypass(0)
            time.sleep(1e-3)
            return R_closest

    def set_series_R(self, R):
        # TODO calculate nearest series value
        pass

    def set_parallel_R(self, R):
        # TODO calculate nearest parallel value
        pass

#########################################################
# PG5 (Picosecond Pulse generator) ######################
#########################################################
class PG5(object):
    def __init__(self, addr='ASRL3::INSTR'):
        try:
            self.connect(addr)
        except:
            print('PG5 connection failed at {}'.format(addr))

    def connect(self, addr):
        self.conn = visa_rm.get_instrument(addr)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.query = self.conn.query
        self.ask = self.query
        self.read = self.conn.read
        self.read_raw = self.conn.read_raw
        self.close = self.conn.close

    def idn(self):
        idn = self.ask('*IDN?')
        self.read()   # read necessary to avoid empty line issue
        return idn.replace('\n', '')

    def error(self):
        '''prints the last error'''
        error_msg = self.query(':SYST:ERR:NEXT?')
        print(error_msg)
    # TODO: fix set_trigger_type (doesnt do anything right now, because commands dont do anything => ask company)

    # def set_trigger_type(self, type):
    #     '''sets the trigger type:
    #     type = \'IMM\' for internal clock
    #     type = \'TTL\' for external triggering
    #     tpye = \'MANUAL\' for manual triggering'''
    #     if type is 'IMM':
    #         self.write(':TRIG:SOUR IMM')
    #     elif type is 'TTL':
    #         self.write(':TRIG:SOUR TTL')
    #     elif type is 'MANUAL':
    #         self.write(':TRIG:SOUR MANUAL')
    #     else:
    #         print('Unknown trigger type. Make sure it is \'IMM\', \'TTL\' or \'MANUAl\'')

    def set_pulse_width(self, pulse_width):
        '''sets the pulse width (between 50 and 250 ps)'''
        self.write(':PULS:WIDT ' + str(pulse_width))

    def set_period(self, period):
        '''sets the period  (between  1 and 1e6 µs)'''
        self.write(':PULS:PER ' + str(period))

    def trigger(self):
        '''Executes a pulse'''
        self.write(':INIT')


#########################################################
# Temperature PID-Control ###############################
#########################################################
class EugenTempStage(object):
    # Global Variables
    # Resistor-Values bridge
    r_1 = 9975
    r_3 = 9976
    r_4 = 1001
    # Gain from instrumental-opamp
    opamp_gain = 12.55
    # Voltage Bridge
    # TODO: do a sample of this voltage to make sure the voltage supply is on, otherwise return an error that says to turn it on!
    volt_now = 10

    def __init__(self, addr=None, baudrate=9600):
        # BORG
        self.__dict__ = persistent_state.tempstage_state
        try:
            self.connect(addr, baudrate)
        except:
            print('Arduino connection failed at {}'.format(addr))

    def connect(self, addr=None, baudrate=9600):
        if not self.connected():
            if addr is None:
                # Connect to the first thing you see that has Arduino Micro in the description
                matches = list(comgrep('Arduino Micro'))
                if any(matches):
                    addr = matches[0].device
                else:
                    print('EugenTempStage could not find Arduino Micro')
                    return
            self.conn = serial.Serial(addr, baudrate, timeout=1)
            self.write = self.conn.write
            self.close = self.conn.close
            if self.connected():
                print(f'Connected to PID controller on {addr}')

    def connected(self):
        if hasattr(self, 'conn'):
            return self.conn.isOpen()
        else:
            return False

    def analogOut(self, voltage):
        ''' Tell arduino to output a voltage on pin 9 '''
        # Arduino will take the set voltage in bits.
        vmax = 5
        numbits = 12
        # Find the closest value that can be output.
        vstep = vmax / (2**numbits - 1)  # 5 /4095
        value = voltage / vstep  # exact value for analogWrite()-function
        cmd_str = '0,{};'.format(value).encode()
        self.write(cmd_str)
        actualvoltage = vstep * value
        return actualvoltage

    def analogIn(self, channel):
        # Function to get Voltage from Bridge
        # Arduino will return the voltage in bits
        vmax = 5
        numbits = 10
        vstep = round(vmax / (2**numbits - 1), 5)# 5 /1023
        cmd_str = '1,{};'.format(channel).encode()
        self.write(cmd_str)

        reply = self.conn.readline().decode()
        adc_value = float(reply.split(',')[-1].strip().strip(';'))
        voltage = adc_value * vstep
        # print('Sent command to read analog input on pin {}'.format(channel))
        return voltage

    def set_temperature(self, temp):
        #Temperature Setpoint Function, should be between 0-100Celsius

        if temp > 100:
            print('Its too HOT! DANGERZONE!')

        if temp <= 100 and temp >= 0:
            pt_res = round((1000 * (1.00385**temp)), 1)
            volt_zaehler = self.volt_now * (pt_res * (self.r_3 + self.r_4) - self.r_4 * (self.r_1 + pt_res))
            volt_nenner = (self.r_4 + self.r_3) * self.r_1 + (self.r_3 + self.r_4) * pt_res
            volt_bruch = volt_zaehler / volt_nenner
            volt_set = volt_bruch * self.opamp_gain
            temp_set = self.analogOut(volt_set)
            #print('Sent command to output {0:.3f} Volt'.format(temp_set))
            print('Temperature set to {0:.2f} \u00b0C'.format(temp))
        else:
            print('Its too COLD! Can not do that :-/')

    def read_temperature(self):
        r_1 = self.r_1
        r_3 = self.r_3
        r_4 = self.r_4
        volt_now = self.volt_now
        opamp_gain = self.opamp_gain

        volt_bridge = self.analogIn(1) / opamp_gain
        pt_zaehler = (((r_3 + r_4) * volt_bridge) + (volt_now * r_4)) * r_1
        pt_nenner = ((r_3 + r_4) * volt_now) - (volt_bridge * (r_3 + r_4) + (r_4 * volt_now))
        pt_res = round((pt_zaehler / pt_nenner), 1)
        temp_read = np.log(pt_res / 1000) / np.log(1.00385)
        return temp_read



def com_port_info():
    comports = serial.tools.list_ports.comports()
