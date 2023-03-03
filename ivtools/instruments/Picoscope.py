import numpy as np
import time
import pandas as pd
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
import ivtools
log = logging.getLogger('instruments')

class Picoscope(object):
    '''
    This class will basically extend the colinoflynn picoscope module
    Has some higher level functionality, and it stores/manipulates the channel settings.
    Picoscope is borg:
    https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/
    '''
    def __init__(self, SerialNumber=None, connect=True):
        if SerialNumber is None:
            statename = self.__class__.__name__
        else:
            statename = SerialNumber
        if statename not in ivtools.instrument_states:
            ivtools.instrument_states[statename] = {}
        self.__dict__ = ivtools.instrument_states[statename]
        from picoscope import ps6000
        self.ps6000 = ps6000
        # I could have subclassed PS6000, but then I would have to import it before the class definition...
        # Then this whole package would have picoscope module as a dependency
        # self.get_data will return data as well as save it here
        self.data = None
        # Store channel settings in this class
        if not hasattr(self, 'range'):
            # TODO somehow contain these in a settings attribute,
            #      so you can easily assign all settings to one variable to switch between them
            #      should be compatible with dict
            #      But also expose these attributes for easy syntax i.e. ps.offset.a = 1
            self.offset = self._PicoOffset(self)
            self.atten = self._PicoAttenuation(self)
            self.coupling = self._PicoCoupling(self)
            self.range = self._PicoRange(self)
            self.BWlimit = self._PicoBWLimit(self)
        if connect:
            self.connect(SerialNumber)

    def connect(self, SerialNumber=None):
        # We are borg, so might already be connected!
        if self.connected():
            #info = self.ps.getUnitInfo('VariantInfo')
            #log.info('Picoscope {} already connected!'.format(info))
            pass
        else:
            try:
                self.ps = self.ps6000.PS6000(SerialNumber, connect=True)
                model = self.ps.getUnitInfo('VariantInfo')
                log.info('Picoscope {} connection succeeded.'.format(model))
                self.close = self.ps.close
                self.handle = self.ps.handle
                # TODO: methods of PS6000 to expose?
                self.getAllUnitInfo = self.ps.getAllUnitInfo
                self.getUnitInfo = self.ps.getUnitInfo
            except Exception as e:
                self.ps = None
                log.error('Connection to picoscope failed. There could be an unclosed session.')
                log.error(e)

    def connected(self):
        if hasattr(self, 'ps'):
            try:
                self.ps.getUnitInfo('VariantInfo')
                return True
            except:
                return False

    def print_settings(self):
        log.info('Picoscope channel settings:')
        log.info(pd.DataFrame([self.coupling, self.atten, self.offset, self.range, self.BWlimit],
                           index=['Couplings', 'Attenuations', 'Offsets', 'Ranges', 'BWlimit']))

    # Settings are a class because I wanted a slightly more convenient syntax for typing in repeatedly
    # Namely, ps.range.b = 2 vs ps.range['B'] = 2
    # and we can also enforce that the settings are valid
    class _PicoSetting(dict):
        possible_ranges_1M = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
        max_offsets_1M = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))
        possible_ranges_50 = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0))
        max_offsets_50 = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 0.5))
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

            if coupling == '':
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
                log.warning(f'Range {value}V is not possible for offset: {offset}, atten: {atten}, coupling: {coupling}. Using range {newvalue}V.')

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
                log.warning(f'Offset {value}V is not possible for range: {vrange}, atten: {atten}, coupling: {coupling}. Using offset {newvalue}V.')

            self[channel] = newvalue

        def invert(self):
            # multiplies all offsets by -1
            self['A'] *= -1
            self['B'] *= -1
            self['C'] *= -1
            self['D'] *= -1

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

            log.info('Scaling the range and offset settings by the new attenuation setting')
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
                log.warning(f'Coupling {value} is not possible for range: {vrange}, offset: {offset}, atten: {atten}.')

    class _PicoBWLimit(_PicoSetting):
        # Used to limit the bandwidth of a particular channel to 20 MHz using an internal ANALOG filter (I hope)
        def __init__(self, parent):
            parent._PicoSetting.__init__(self, parent)
            self['A'] = 0
            self['B'] = 0
            self['C'] = 0
            self['D'] = 0


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
                log.info('Setting picoscope channel {} range {}, offset {}'.format(c, rang, offs))
                self.range[c] = rang
                self.offset[c] = offs

    def best_range(self, data, padpercent=0, atten=1, coupling='DC'):
        '''
        Return the best RANGE and OFFSET values to use for a particular input signal (array)
        Just uses minimim and maximum values of the signal, therefore you could just pass (min, max), too
        Don't pass int8 signals, would then need channel information to convert to V
        TODO: Use an offset that includes zero if it doesn't require increasing the range
        TODO: fix the error in logic leading to incompatible range-offset pairs
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
            if np.abs(middle) < max_offset:
                return (selectedrange, -middle)
                break
            # Can we reduce the offset without the signal going out of range?
            elif (max_offset + selectedrange >= padmax) and (-max_offset - selectedrange <= padmin):
                return(selectedrange, np.clip(-middle, -max_offset, max_offset))
                break
            # Neither worked, try increasing the range ...
        # If no range was enough to fit the signal
        log.error('Signal out of pico range!')
        return (max(possible_ranges), 0)

    def capture(self, ch='A', freq=None, duration=None, nsamples=None,
                trigsource='TriggerAux', triglevel=0.1, timeout_ms=30000, direction='Rising',
                pretrig=0.0, delay=0,
                chrange=None, choffset=None, chcoupling=None, chatten=None, chbwlimit=None):
        '''
        Set up picoscope to capture from specified channel(s).

        pass exactly two of: freq(sampling frequency), duration, nsamples
        sampling frequency has limited possible values, so actual number of samples will vary
        will try to sample for the intended duration, either the value of the duration argument
        or nsamples/freq

        Won't actually start capture until picoscope receives the specified trigger event.

        It will trigger automatically after a timeout.
        I think if you set timeout_ms to zero this means an infinite timeout

        ch can be a list of characters, i.e. ch=['A','B'].

        pretrig is in fraction of the whole sampling time, delay is in samples..

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
        chbwlimit = global_replace(chbwlimit, self.BWlimit)

        actualfreq, max_samples = self.ps.setSamplingFrequency(actualfreq, nsamples)
        log.info('Actual picoscope sampling frequency: {:,}'.format(actualfreq))
        if nsamples > max_samples:
            raise(Exception('Trying to sample more than picoscope memory capacity'))
        # Set up the channels
        for c in ch:
            self.ps.setChannel(channel=c,
                               coupling=chcoupling[c],
                               VRange=chrange[c],
                               probeAttenuation=chatten[c],
                               VOffset=choffset[c],
                               BWLimited=chbwlimit[c],
                               enabled=True)
        # Set up trigger.  Will timeout in 30s
        self.ps.setSimpleTrigger(trigsource, triglevel, direction=direction, delay=delay, timeout_ms=timeout_ms)
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
                log.warning(f'!! Picoscope overflow on Ch {c} !!')
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
        CHCOUPLINGS = dict(map(reversed, self.ps.CHANNEL_COUPLINGS.items()))
        data['COUPLINGS'] = {ch:CHCOUPLINGS[chc] for ch, chc in zip(Channels, self.ps.CHCoupling)}
        data['sample_rate'] = self.ps.sampleRate
        # Specify samples captured, because this field will persist even after splitting for example
        # Then if you split 100,000 samples into 10 x 10,000 having nsamples = 100,000 will be confusing
        nsamples = len(data[ch[0]])
        data['nsamples_capture'] = nsamples
        data['t'] = np.linspace(0, nsamples/data['sample_rate'], nsamples)
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
                log.error('No data to plot')
            else:
                chdata = self.data
        # Colors match the code on the picoscope
        # Yellow is too hard to see
        colors = dict(A='Blue', B='Red', C='Green', D='Gold')
        channels = ['A', 'B', 'C', 'D']
        # Remove any previous range indicators that might exist on the plot
        # ax.collections = [] # can't do this anymore since matplotlib update
        for coll in ax.collections:
            coll.remove()
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
