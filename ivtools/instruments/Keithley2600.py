import numpy as np
import pandas as pd
from ..instruments import ping
import os
import re
from collections import deque
import ivtools
import logging
log = logging.getLogger('instruments')
import visa
visa_rm = visa.visa_rm # stored here by __init__

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
    def __init__(self, addr=None):
        valid_ip_re = "^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
        if addr is None:
            # I don't trust the resource manager at all, but you didn't pass an address so..
            # I assume you are using ethernet
            ipresources = [r for r in visa_rm.list_resources() if r.startswith('TCPIP')]
            log.debug('Looking for ip address for Keithley...')
            for ipr in ipresources:
                # Sorry..
                ip = re.search(valid_ip_re[1:-1] +':', ipr)[0][:-1]
                # I'm not sure how to check if it is a keithley or not
                # for now, if it is in resource_manager and replies to a ping, it's a keithley
                up = ping(ip)
                if up:
                    log.debug(f'{ip} is up. Is it keithley?')
                    addr = ipr
                    break
        elif re.match(valid_ip_re, addr):
            # You passed an ip alone and we will turn it into a gpib string
            addr = f'TCPIP::{addr}::inst0::INSTR'

        try:
            statename = '_'.join((self.__class__.__name__, addr))
            if statename not in ivtools.instrument_states:
                ivtools.instrument_states[statename] = {}
                say_if_successful = True
            else:
                say_if_successful = False
            self.__dict__ = ivtools.instrument_states[statename]
            self.connect(addr)
            if say_if_successful:
                log.info('Keithley connection successful at {}'.format(addr))
        except Exception as E:
            log.error('Keithley connection failed at {}'.format(addr))
            log.error(E)

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
            self.data = deque(maxlen=100)
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

    def _iv_lua(self, vlist, Irange=0, Ilimit=0, Plimit=0, nplc=1, delay='smua.DELAY_AUTO', Vrange=0):
        '''
        range = 0 enables autoranging
        Wraps the SweepVList lua function defined on keithley
        '''
        # Send list of voltage values to keithley
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write(f'SweepVList(sweeplist, {Irange}, {Ilimit}, {Plimit}, {nplc}, {delay}, {Vrange})')

    def _iv_4pt_lua(self, vlist, Irange=0, Ilimit=0, nplc=1, delay='smua.DELAY_AUTO', Vrange=0):
        '''
        range = 0 enables autoranging
        Wraps the SweepVList lua function defined on keithley
        '''
        # Send list of voltage values to keithley
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write('SweepVList_4pt(sweeplist, {}, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay, Vrange))

    def _vi_lua(self, ilist, Vrange=0, Vlimit=0, nplc=1, delay='smua.DELAY_AUTO', Irange=None):
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

    def iv(self, source_list, source_func='v', source_range=None, measure_range=None,
           v_limit=None, i_limit=None, p_limit=None,
           nplc=1, delay=None, point4=False, ch='a'):
        """
        Measure IV curve.

        :param source_list: During the sweep, the source will output the sequence of source values given in
            the source_list array.
        :param source_func: Configure the specified SMU channel as either a voltage source or a current source.
        :param source_range: Set a range large enough to source the assigned value.
        If None the maximum of source_list will passed as range.
        :param measure_range: Set a range large enough to measure the assigned value.
        :param v_limit: Limit the voltage output of the current source.
        :param i_limit: Limit the current output of the voltage source.
        :param p_limit: Limit the output power of the source.
        :param nplc: Set the lowest measurement range that is used when the instrument is autoranging.
        :param delay: Set an additional delay (settling time) between measurements.
        :param point4: Select the sense mode to 4-wire remote if True or 2-wire local if False
        :param ch: Select the Source-measure unit (SMU) channel to A or B
        :return: None
        """

        source_func = source_func.lower()
        ch = ch.lower()
        self.reset()
        # Configure the SMU
        self.reset_ch(ch)
        if point4 is True:
            self.sense('remote', ch=ch)
        self.source_func(source_func, ch=ch)
        self.nplc(nplc, ch=ch)
        if delay is None:
            delay = 'auto'
        self.measure_delay(delay, ch=ch)
        # Set the limits of voltage, current and power
        for s, l in [('v', v_limit), ('i', i_limit), ('p', p_limit)]:
            if l is not None:
                self.source_limit(s, l, ch)
                self.trigger_source_limit(s, l, ch)
        # Set the source range
        if source_range is None:
            source_range = np.max(np.abs(source_list))
        self.source_range(source_func, source_range, ch=ch)
        # Set the measure range
        if measure_range is None:
            measure_range = 'auto'
        self.measure_range(source_func, measure_range, ch=ch)
        # Prepare the Reading Buffers
        self.prepare_buffers(source_func, ch=ch)
        # Configure SMU Trigger Model for Sweep
        self.trigger_source_list(source_func, source_list, ch=ch)
        self.prepare_trigger('iv', source_list, ch=ch)
        self.trigger_initiate(ch=ch)
        self.waitcomplete()
        self.source_output(False, ch=ch)

    def vi(self, source_list, source_range=None, measure_range=None,
           v_limit=None, i_limit=None, p_limit=None,
           nplc=1, delay=None, point4=False, ch='a'):
        """
        Measure IV curve sourcing current.
        This is just self.iv with source_func='i'.

        :param source_list: During the sweep, the source will output the sequence of source values given in
            the source_list array.
        :param source_range: Set a range large enough to source the assigned value.
        If None the maximum of source_list will passed as range.
        :param measure_range: Set a range large enough to measure the assigned value.
        :param v_limit: Limit the voltage output of the current source.
        :param i_limit: Limit the current output of the voltage source.
        :param p_limit: Limit the output power of the source.
        :param nplc: Set the lowest measurement range that is used when the instrument is autoranging.
        :param delay: Set an additional delay (settling time) between measurements.
        :param point4: Select the sense mode to 4-wire remote if True or 2-wire local if False
        :param ch: Select the Source-measure unit (SMU) channel to A or B
        :return: None
        """

        self.iv(source_list, 'i', source_range, measure_range, v_limit, i_limit, p_limit,
                nplc, delay, point4, ch)

    def iv_2ch(self,
               a_source_list, b_source_list,
               a_source_func='v', b_source_func='v',
               a_source_range=None, b_source_range=None,
               a_measure_range=None, b_measure_range=None,
               a_v_limit=None, b_v_limit=None,
               a_i_limit=None, b_i_limit=None,
               a_p_limit=None, b_p_limit=None,
               a_nplc=1, b_nplc=1,
               a_delay=None, b_delay=None,
               a_point4=False, b_point4=False):

        if not isinstance(a_source_list, (list, np.ndarray, pd.Series)):
            a_source_list = a_source_list * np.ones(len(b_source_list))
        if not isinstance(b_source_list, (list, np.ndarray, pd.Series)):
            b_source_list = b_source_list * np.ones(len(a_source_list))

        if len(a_source_list) != len(b_source_list):
            raise Exception('Source values lists must have the same length')

        self.reset()

        def configure_channel(ch, source_list, source_func, source_range, measure_range,
                              v_limit, i_limit, p_limit, nplc, delay, point4):
            source_func = source_func.lower()
            ch = ch.lower()
            # Configure the SMU
            self.reset_ch(ch)
            if point4 is True:
                self.sense('remote', ch=ch)
            self.source_func(source_func, ch=ch)
            self.nplc(nplc, ch=ch)
            if delay is None:
                delay = 'auto'
            self.measure_delay(delay, ch=ch)
            # Set the limits of voltage, current and power
            for s, l in [('v', v_limit), ('i', i_limit), ('p', p_limit)]:
                if l is not None:
                    self.source_limit(s, l, ch)
                    self.trigger_source_limit(s, l, ch)
            # Set the source range
            self.source_range(source_func, source_range, ch=ch)
            # Set the measure range
            self.measure_range(source_func, measure_range, ch=ch)
            # Prepare the Reading Buffers
            self.prepare_buffers(source_func, ch=ch)
            # Configure SMU Trigger Model for Sweep
            self.trigger_source_list(source_func, source_list, ch=ch)
            self.prepare_trigger('iv', source_list, ch=ch)

        configure_channel('a', a_source_list, a_source_func, a_source_range, a_measure_range,
                          a_v_limit, a_i_limit, a_p_limit, a_nplc, a_delay, a_point4)
        configure_channel('b', b_source_list, b_source_func, b_source_range, b_measure_range,
                          b_v_limit, b_i_limit, b_p_limit, b_nplc, b_delay, b_point4)

        self.trigger_initiate('both')
        self.waitcomplete()
        self.source_output(False, 'both')


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
            time.sleep(.3)
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

    def get_data_2ch(self, start=1, end=None, history=True):
        dataA = self.get_data(start, end, history, ch='A')
        dataB  = self.get_data(start, end, history, ch='B')
        data = {}
        data['A'] = dataA
        data['B'] = dataB
        data_flat = {f'{kk}_{k}': vv for k, v in data.items() for kk, vv in v.items()}
        return data_flat

    @staticmethod
    def replace_nanvals(array):
        # Keithley returns this special value when the measurement is out of range
        # replace it with a nan so it doesn't mess up the plots
        # They aren't that smart at Keithley, so different models return different special values.
        nanvalues = (9.9100000000000005e+37, 9.9099995300309287e+37)
        for nv in nanvalues:
            array[array == nv] = np.nan
        return array

    ### Wrap some of the lua commands directly
    ### There are a million commands so this is not a complete wrapper..
    ### See the 900 page pdf reference manual..

    def _set_or_query(self, prop, val=None, bool=False):
        # Sets or returns the current val
        if val is None:
            reply = self.query(f'print({prop})').strip()
            return self._string_parser(reply)
        else:
            if bool:
                val = 1 if val else 0
            self.write(f'{prop} = {val}')
            return None

    def _string_parser(self, string):
        # Since we have to communicate via strings and these string might just be numeric..
        # Convert to numeric?
        def will_it_float(value):
            try:
                float(value)
                return True
            except ValueError:
                return False
        if string.isnumeric():
            return int(string)
        elif will_it_float(string):
            return float(string)
        else:
            # dunno
            return string

    def reset(self):
        self.write('reset()')

    def reset_ch(self, ch='A'):
        ch = ch.lower()
        self.write(f'smu{ch}.reset()')

    def waitcomplete(self):
        self.write('waitcomplete()')

    def prepare_buffers(self, source_func, buffers=None, ch='A'):
        """
        Configure the typical buffer settings used for triggering.

        :param source_func: Type of measure: current (i), or voltage (v).
        :param buffers: List of the two buffers names.
        :param ch: Channel to be configured.
        :return: None
        """

        if buffers is None:
            buffers = ['nvbuffer1', 'nvbuffer2']
        ch = ch.lower()
        source_func = source_func.lower()
        if source_func == 'v':
            csv1 = False
            csv2 = True
        elif source_func == 'i':
            csv1 = True
            csv2 = False
        else:
            raise Exception("'source_value can only set as 'v' or 'i'.")
        self.clear_buffer(buffers[0], ch=ch)
        self.collect_timestamps(True, buffers[0], ch=ch)
        self.collect_sourcevalues(csv1, buffers[0], ch=ch)
        self.clear_buffer(buffers[1], ch=ch)
        self.collect_timestamps(True, buffers[1], ch=ch)
        self.collect_sourcevalues(csv2, buffers[1], ch=ch)

    def prepare_trigger(self, measurement, source_list, ch='A'):
        """
        Configure the typical trigger settings.

        :param measurement: Current(i), voltage(v), resistance(r), power(p), or current and voltage (iv).
        :param source_list: List of values to be sourced.
        :param ch: Channel to be configured.
        :return: None
        """

        self.trigger_measure_action(True, ch=ch)
        self.trigger_measure(measurement, ch=ch)
        self.trigger_endpulse_action('hold', ch=ch)
        self.trigger_endsweep_action('idle', ch=ch)
        num_points = len(source_list)
        self.trigger_count(num_points, ch=ch)
        self.trigger_source_action(True, ch=ch)
        self.source_output(True, ch=ch)

    def measure(self, measurement, ch='A'):
        """
        This function makes one or more measurements.

        :param measurement: Parameter to measure. I t can be current (i), voltage (v), resistance (r),
        power (p) or (iv). In the last case it returns the last actual current measurement and voltage
        measurement.
        :param ch: Channel where measure.
        :return: Measurement value.
        """
        ch = ch.lower()
        if measurement in ['i', 'v', 'r', 'p']:
            reply = self.query(f'print(smu{ch}.measure.{measurement}())')
            reply = float(reply)
        elif measurement == 'iv':
            reply = self.query(f'print(smu{ch}.measure.iv())')
            reply = reply.split('\t')
            reply = [float(i) for i in reply]
        else:
            raise Exception("Parameter 'measurement' can only be 'i', 'v', 'r', 'p' or 'iv'.")
        return reply

    def measure_range(self, source_func, m_range='auto', ch='A'):
        """
        Set the SMU to a fixed range large enough to measure the assigned value.

        :param source_func: Type of measure: current (i), or voltage (v).
        :param m_range: Range to be set. In amperes or volts.
        :param ch: Channel to be configured.
        :return: If m_range is None, configured ranged is returned.
        """
        source_func = source_func.lower()
        ch = ch.lower()
        if source_func == 'v':
            range_func = 'i'
        elif source_func == 'i':
            range_func = 'v'
        else:
            raise Exception("'source_value can only set as 'v' or 'i'.")
        if type(m_range) == str:
            if m_range.lower() == 'auto':
                return self._set_or_query(f'smu{ch}.measure.autorange{range_func}', True, bool=True)
            else:
                raise Exception("'m_range' can only be a string if it's value is 'auto'")
        else:
            return self._set_or_query(f'smu{ch}.measure.range{range_func}', m_range)

    def source_level(self, source_func, source_val=None, ch='A'):
        """
        Sets the source level, or ask for it.

        :param source_func: Parameter to source. 'i' if current and 'v' if voltage.
        :param source_val: Value to set the source. If None, this function returns the previous level.
        :param ch: Channel to which set the source level.
        :return: Source level.
        """
        ch = ch.lower()
        if source_func in ['i', 'v']:
            reply = self._set_or_query(f'smu{ch}.source.level{source_func}', source_val)
        else:
            raise Exception("Parameter 'source_func' can only be 'i' or 'v'.")
        return reply

    def source_func(self, func=None, ch='A'):
        """
        This function set the source as voltage (v) or current (i).

        :param func: Voltage (v) or current (i).
        :param ch: Channel to which apply changes.
        :return: If func is None it returns the previous value.
        """

        ch = ch.lower()
        if func is not None:
            if func.lower() == 'i':
                func = f'smu{ch}.OUTPUT_DCAMPS'
            elif func.lower() == 'v':
                func = f'smu{ch}.OUTPUT_DCVOLTS'
        elif func is None:
            pass
        else:
            raise Exception("Error in source_func: 'func' can only be I, V or None")
        return self._set_or_query(f'smu{ch}.source.func', func)

    def source_limit(self, source_param, limit=None, ch='A'):
        """
        Set compliance limits for current (i), voltage(v) or power(p).
        This attribute should be set in the test sequence before the turning the source on.

        :param source_param: Source parameter to limit: current (i), voltage(v) or power(p).
        :param limit: Limit to be set. In amperes, volts or watts.
        :param ch: Channel to be configured
        :return: If limit is None, configured limit is returned.
        """
        ch = ch.lower()
        source_param = source_param.lower()
        return self._set_or_query(f'smu{ch}.source.limit{source_param}', limit)

    def source_range(self, source_func, s_range='auto', ch='A'):
        """
        Set the SMU to a fixed range large enough to source the assigned value.
        Autoranging could be very slow.

        :param source_func: Type of source: current (i), or voltage (v).
        :param s_range: Range to be set. In amperes or volts.
        :param ch: Channel to be configured
        :return: If s_range is None, configured ranged is returned.
        """
        source_func = source_func.lower()
        ch = ch.lower()
        if type(s_range) == str:
            if s_range.lower() == 'auto':
                return self._set_or_query(f'smu{ch}.source.autorange{source_func}', True, bool=True)
            else:
                raise Exception("'range' can only be a string if it's value is 'auto'")
        else:
            return self._set_or_query(f'smu{ch}.source.range{source_func}', s_range)

    def source_output(self, state=None, ch='A'):
        # Set output state
        ch = ch.lower()
        if ch == 'both':
            self._set_or_query(f'smua.source.output', state, bool=True)
            self._set_or_query(f'smub.source.output', state, bool=True)
        else:
            self._set_or_query(f'smu{ch}.source.output', state, bool=True)

    def trigger_source_limit(self, source_func, limit=None, ch='A'):
        """
        Set the sweep source limit for current.

        If this attribute is set to any other numeric value, the SMU will switch
        in this limit at the start of the source action and will return to the normal
        limit setting at the end of the end pulse action.

        :param source_func: Type of measure: current (i), or voltage (v).
        :param limit: Limit to be set. In amperes, volts or watts.
        :param ch: Channel to be configured.
        :return: None
        """

        ch = ch.lower()
        source_func = source_func.lower()
        return self._set_or_query(f'smu{ch}.trigger.source.limit{source_func}', limit)

    def trigger_source_list(self, source_func, source_list, ch='A'):
        """
        Configure source list sweep for SMU channel.

        :param source_func: Type of measure: current (i), or voltage (v).
        :param source_list: List to be set. In amperes or volts.
        :param ch: Channel to be configured.
        :return: None
        """

        ch = ch.lower()
        source_func = source_func.lower()
        self.send_list(source_list, 'listvarname')
        return self.write(f'smu{ch}.trigger.source.list{source_func}(listvarname)')

    def trigger_measure(self, measurement, buffer=None, ch='A'):
        """
        This function configures the measurements that are to be made in a subsequent sweep.

        :param measurement: Current(i), voltage(v), resistance(r), power(p), or current and voltage (iv).
        :param buffer: Name of the the buffer to save the data. If 'iv' it must be a list of two names.
        :param ch: Channel to be configured.
        :return: None
        """

        ch = ch.lower()
        measurement = measurement.lower()
        if buffer is None:
            if measurement == 'iv':
                i_buffer = 'nvbuffer1'
                v_buffer = 'nvbuffer2'
                buffer = [i_buffer, v_buffer]
            elif measurement in ['i', 'v', 'r', 'p']:
                buffer = 'nvbuffer1'
            else:
                raise Exception("'measurement' can only be 'i', 'v', 'r', 'p' or 'iv'.")
        if measurement == 'iv':
            return self.write(f'smu{ch}.trigger.measure.iv(smu{ch}.{buffer[0]},smu{ch}.{buffer[1]})')
        else:
            return self.write(f'smu{ch}.trigger.measure.{measurement}({buffer})')

    def trigger_measure_action(self, action=None, ch='A'):
        """
        Possible action values:
            True: Make measurements during the sweep
            False: Do not make measurements during the sweep
            ASYNC: Make measurements during the sweep, but asynchronously with the source area of the trigger model
        """
        ch = ch.lower()
        if action is True or False:
            return self._set_or_query(f'smu{ch}.trigger.measure.action', action, bool=True)
        else: action = action.lower()
        if action == 'async':
            return self._set_or_query(f'smu{ch}.trigger.measure.action', 2)
        else:
            log.error('Error in trigger_measure_action: action can only be True, False or ASYNC')

    def trigger_endpulse_action(self, action=None, ch='A'):
        """
        This attribute enables or disables pulse mode sweeps.

        Possible action values:
            IDLE: Enables pulse mode sweeps, setting the source level to the programmed (idle) level at the end of the pulse.
            HOLD: Disables pulse mode sweeps, holding the source level for the remainder of the step.
            (where X is the chanel selcted, such as "a" )
        """
        ch = ch.lower()
        if action.lower() == 'idle':
            action = 0
        elif action.lower() == 'hold':
            action = 1
        elif action is None:
            pass
        else:
            log.error("Error with endpulse_action: action can only be IDLE or HOLD.")
        return self._set_or_query(f'smu{ch}.trigger.endpulse.action', action)

    def trigger_endsweep_action(self, action=None, ch='A'):
        """
        This attribute sets the action of the source at the end of a sweep

        Possible action values:
            IDLE: Sets the source level to the programmed (idle) level at the end of the sweep
            HOLD: Sets the source level to stay at the level of the last step.
            (where X is the chanel selcted, such as "a" )
        """
        ch = ch.lower()
        if action.lower() == 'idle':
            action = 0
        elif action.lower() == 'hold':
            action = 1
        elif action is None:
            pass
        else:
            log.error("Error with endpulse_action: action can only be IDLE or HOLD.")
        return self._set_or_query(f'smu{ch}.trigger.endsweep.action', action)

    def trigger_count(self, count=None, ch='A'):
        """
        This attribute sets the trigger count in the trigger model.

        The trigger count is the number of times the source-measure unit (SMU) will iterate in the trigger layer for any given sweep.

        If this count is set to zero (0), the SMU stays in the trigger model indefinitely until aborted.
        """
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.trigger.count', count)

    def trigger_source_action(self, action=None, ch='A'):
        """
        This attribute enables or disables sweeping the source (on or off).

        Possible action values:
            False: Do not sweep the source
            True: Sweep the source
        """
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.trigger.source.action', action, bool=True)

    def trigger_initiate(self, ch='A'):
        """
        This function initiates a sweep operation.

        This function causes the SMU to clear the four trigger model event detectors and enter its trigger model (moves the SMU from the idle state into the arm layer).
        """
        ch = ch.lower()
        if ch == 'both':
            self.write(f'smua.trigger.initiate()')
            self.write(f'smub.trigger.initiate()')
        else:
            self.write(f'smu{ch}.trigger.initiate()')

    def trigger_blender_reset(self, N):
        """
        This function resets some of the trigger blender settings to their factory defaults.
        """

        self.write(f'trigger.blender[{N}].reset()')

    def trigger_blender_clear(self, N):
        """
        This function clears the command interface trigger event detector
        """

        self.write(f'trigger.blender[{N}].clear()')

    def trigger_blender_stimulus(self, N, stimulus, or_mode=False):
        """
        This attribute specifies which events trigger the blender.

        'stimulus' must be a list of strings.

        Possible values of 'stimulus':
            'ch_sweeping': Occurs when the source-measure unit (SMU)
                transitions from the idle state to the arm layer of the
                trigger model.
                'ch' must be 'a' or 'b'.
            'ch_armed': Occurs when the SMU moves from the arm layer to
                the trigger layer of the trigger model.
                'ch' must be 'a' or 'b'.
            'ch_sourcecomplete': Occurs when the SMU completes a source action.
                'ch' must be 'a' or 'b'.
            'ch_measurecomplete': Occurs when the SMU completes a measurement
                action.
                'ch' must be 'a' or 'b'.
            'ch_pulsecomplete': Occurs when the SMU completes a pulse.
                'ch' must be 'a' or 'b'.
            'ch_sweepcomplete': Occurs when the SMU completes a sweep.
                'ch' must be 'a' or 'b'.
            'ch_idle': Occurs when the SMU returns to the idle state.
                'ch' must be 'a' or 'b'.
            'blender_N': Occurs after a collection of events is detected.
                'N' must be an integer representing the trigger event blender (1 to 6).

        If 'or_mode' is set to 'True' the blender will operate in OR mode.
        If 'or_mode' is set to 'False' the blender will operate in AND mode.

        More stimulus are possible in LUA.
        One blender can not have more than 4 stimulus.
        There can not exist more than 6 blenders.
        """

        if or_mode is True:
            enable = 'true'
        elif or_mode is False:
            enable = 'false'
        self.write(f'trigger.blender[{N}].orenable = {enable}')

        NumStimulus = len(stimulus)
        if NumStimulus > 4:
            raise Exception('One blender can not have more than 4 stimulus.')

        for M in range(0, NumStimulus):
            stM = stimulus[M]
            stM = stM.lower().split("_")
            if stM[0] == 'a' or stM[0] == 'b':
                ch = stM[0]
                if stM[1] == 'sweeping':
                    self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = smu{ch}.trigger.SWEEPING_EVENT_ID')
                elif stM[1] == 'armed':
                    self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = smu{ch}.trigger.ARMED_EVENT_ID')
                elif stM[1] == 'sourcecomplete':
                    self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = smu{ch}.trigger.SOURCE_COMPLETE_EVENT_ID')
                elif stM[1] == 'measurecomplete':
                    self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = smu{ch}.trigger.MEASURE_COMPLETE_EVENT_ID')
                elif stM[1] == 'pulsecomplete':
                    self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = smu{ch}.trigger.PULSE_COMPLETE_EVENT_ID')
                elif stM[1] == 'sweepcomplete':
                    self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = smu{ch}.trigger.SWEEP_COMPLETE_EVENT_ID')
                elif stM[1] == 'idle':
                    self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = smu{ch}.trigger.IDLE_EVENT_ID')
                else:
                    raise Exception('Wrong stimulus name')
            elif stM[0] == 'blender':
                sN = int(stM[1])
                if sN > 6: raise Exception('There can not exist more than 6 blenders')
                self.write(f'trigger.blender[{N}].stimulus[{M + 1}] = trigger.blender[{sN}].EVENT_ID')
            else:
                raise Exception('Wrong stimulus name')

    def trigger_measure_stimulus(self, stimulus, ch='a'):
        """
        This attribute selects which event will cause the measure event detector to enter the detected state.

        Possible values of 'stimulus':
            'ch_sweeping': Occurs when the source-measure unit (SMU)
                transitions from the idle state to the arm layer of the
                trigger model.
                'ch' must be 'a' or 'b'.
            'ch_armed': Occurs when the SMU moves from the arm layer to
                the trigger layer of the trigger model.
                'ch' must be 'a' or 'b'.
            'ch_sourcecomplete': Occurs when the SMU completes a source action.
                'ch' must be 'a' or 'b'.
            'ch_measurecomplete': Occurs when the SMU completes a measurement
                action.
                'ch' must be 'a' or 'b'.
            'ch_pulsecomplete': Occurs when the SMU completes a pulse.
                'ch' must be 'a' or 'b'.
            'ch_sweepcomplete': Occurs when the SMU completes a sweep.
                'ch' must be 'a' or 'b'.
            'ch_idle': Occurs when the SMU returns to the idle state.
                'ch' must be 'a' or 'b'.
            'blender_N': Occurs after a collection of events is detected.
                'N' must be an integer representing the trigger event blender (1 to 6).

        More stimulus are possible in LUA.
        """
        ch = ch.lower()
        stimulus = stimulus.lower()
        stimulus = stimulus.split("_")
        if stimulus[0] == 'a' or stimulus[0] == 'b':
            ch_st = stimulus[0]
            if stimulus[1] == 'sweeping':
                return self.write(f'smu{ch}.trigger.measure.stimulus = smu{ch_st}.trigger.SWEEPING_EVENT_ID')
            elif stimulus[1] == 'armed':
                return self.write(f'smu{ch}.trigger.measure.stimulus = smu{ch_st}.trigger.ARMED_EVENT_ID')
            elif stimulus[1] == 'sourcecomplete':
                return self.write(f'smu{ch}.trigger.measure.stimulus = smu{ch_st}.trigger.SOURCE_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'measurecomplete':
                return self.write(f'smu{ch}.trigger.measure.stimulus = smu{ch_st}.trigger.MEASURE_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'pulsecomplete':
                return self.write(f'smu{ch}.trigger.measure.stimulus = smu{ch_st}.trigger.PULSE_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'sweepcomplete':
                return self.write(f'smu{ch}.trigger.measure.stimulus = smu{ch_st}.trigger.SWEEP_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'idle':
                return self.write(f'smu{ch}.trigger.measure.stimulus = smu{ch_st}.trigger.IDLE_EVENT_ID')
            else:
                raise Exception('Wrong stimulus name')
        elif stimulus[0] == 'blender':
            sN = int(stimulus[1])
            if sN > 6: raise Exception('There can not exist more than 6 blenders')
            self.write(f'smu{ch}.trigger.measure.stimulus = trigger.blender[{sN}].EVENT_ID')
        else:
            raise Exception('Wrong stimulus name')

    def trigger_endpulse_stimulus(self, stimulus, ch='a'):
        """
        This attribute defines which event will cause the end pulse event detector to enter the detected state.

        Possible values of 'stimulus':
            'ch_sweeping': Occurs when the source-measure unit (SMU)
                transitions from the idle state to the arm layer of the
                trigger model.
                'ch' must be 'a' or 'b'.
            'ch_armed': Occurs when the SMU moves from the arm layer to
                the trigger layer of the trigger model.
                'ch' must be 'a' or 'b'.
            'ch_sourcecomplete': Occurs when the SMU completes a source action.
                'ch' must be 'a' or 'b'.
            'ch_measurecomplete': Occurs when the SMU completes a measurement
                action.
                'ch' must be 'a' or 'b'.
            'ch_pulsecomplete': Occurs when the SMU completes a pulse.
                'ch' must be 'a' or 'b'.
            'ch_sweepcomplete': Occurs when the SMU completes a sweep.
                'ch' must be 'a' or 'b'.
            'ch_idle': Occurs when the SMU returns to the idle state.
                'ch' must be 'a' or 'b'.
            'blender_N': Occurs after a collection of events is detected.
                'N' must be an integer representing the trigger event blender (1 to 6).

        More stimulus are possible in LUA.
        """
        ch = ch.lower()
        stimulus = stimulus.lower()
        stimulus = stimulus.split("_")
        if stimulus[0] == 'a' or stimulus[0] == 'b':
            ch_st = stimulus[0]
            if stimulus[1] == 'sweeping':
                return self.write(f'smu{ch}.trigger.endpulse.stimulus = smu{ch_st}.trigger.SWEEPING_EVENT_ID')
            elif stimulus[1] == 'armed':
                return self.write(f'smu{ch}.trigger.endpulse.stimulus = smu{ch_st}.trigger.ARMED_EVENT_ID')
            elif stimulus[1] == 'sourcecomplete':
                return self.write(f'smu{ch}.trigger.endpulse.stimulus = smu{ch_st}.trigger.SOURCE_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'measurecomplete':
                return self.write(f'smu{ch}.trigger.endpulse.stimulus = smu{ch_st}.trigger.MEASURE_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'pulsecomplete':
                return self.write(f'smu{ch}.trigger.endpulse.stimulus = smu{ch_st}.trigger.PULSE_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'sweepcomplete':
                return self.write(f'smu{ch}.trigger.endpulse.stimulus = smu{ch_st}.trigger.SWEEP_COMPLETE_EVENT_ID')
            elif stimulus[1] == 'idle':
                return self.write(f'smu{ch}.trigger.endpulse.stimulus = smu{ch_st}.trigger.IDLE_EVENT_ID')
            else:
                raise Exception('Wrong stimulus name')
        elif stimulus[0] == 'blender':
            sN = int(stimulus[1])
            if sN > 6: raise Exception('There can not exist more than 6 blenders')
            self.write(f'smu{ch}.trigger.endpulse.stimulus = trigger.blender[{sN}].EVENT_ID')
        else:
            raise Exception('Wrong stimulus name')

    def sense(self, state='LOCAL', ch='A'):
        """
        Possible values for state:
            LOCAL: Selects local sense (2-wire)
            REMOTE: Selects local sense (4-wire)
            CALA: Selects calibration sense mode
        """
        ch = ch.lower()
        state = state.upper()

        return self.write(f'smu{ch}.sense = smu{ch}.SENSE_{state}')

    def nplc(self, nplc=None, ch='A'):
        """
        This command sets the integration aperture for measurements
        This attribute controls the integration aperture for the analog-to-digital converter (ADC).
        The integration aperture is based on the number of power line cycles (NPLC), where 1 PLC for 60 Hzis 16.67 ms (1/60) and 1 PLC for 50 Hz is 20 ms (1/50)
        """
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.measure.nplc', nplc)

    def measure_delay(self, delay=None, ch='A'):
        """
        This attribute controls the measurement delay
        Specify delay in second.
        To use autodelay set delay='AUTO'
        """
        ch = ch.lower()
        try: delay=delay.lower()
        except AttributeError: pass
        if delay == 'auto': delay = f'smu{ch}.DELAY_AUTO'
        return self._set_or_query(f'smu{ch}.measure.delay', delay)

    def clear_buffer(self, buffer='nvbuffer1', ch='A'):
        """
        This function empties the buffer

        This function clears all readings and related recall attributes from the buffer (for example,
        bufferVar.timestamps and bufferVar.statuses) from the specified buffer.
        """
        ch = ch.lower()
        return self.write(f'smu{ch}.{buffer}.clear()')

    def collect_timestamps(self, state=None, buffer='nvbuffer1', ch='A'):
        """
        This attribute sets whether or not timestamp values are stored with the readings in the buffer

        Possible state values:
             False: Timestamp value collection disabled (off)
             True: Timestamp value collection enabled (on)
        """
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.{buffer}.collecttimestamps', state, bool=True)

    def collect_timevalues(self, state=None, buffer='nvbuffer1', ch='A'):
        """
        This attribute sets whether or not source values are stored with the readings in the buffer

        Possible source values:
            False: Source value collection disabled (off)
            True: Source value collection enabled (on)
        """
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.{buffer}.collecttimevalues', state, bool=True)

    def collect_sourcevalues(self, state=None, buffer='nvbuffer1', ch='A'):
        """
        This attribute sets whether or not source values will be stored with the readings in the buffer
        """
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.{buffer}.collectsourcevalues', state, bool=True)
