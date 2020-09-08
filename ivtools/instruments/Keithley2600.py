import numpy as np
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

    We have wrapped a large number of the lua commands individually in python
    functions which construct and send lua strings, and parse the reply into
    python datatypes.

    It is not a complete wrapping, however, as that would be quite an undertaking.

    We also maintain a separate lua file "Keithley_2600.lua" which defines lua
    functions on the Keithley, that can be then be called through python.  So one
    can write directly in lua there if desired.

    visa.ResourceManager does not register TCP connections properly, and there
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
                # I'm not sure how to check if it is a Keithley or not
                # for now, if it is in resource_manager and replies to a ping, it's a Keithley
                up = ping(ip)
                if up:
                    log.debug(f'{ip} is up. Is it Keithley?')
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
            self.display('ALL YOUR BASE', 'ARE BELONG TO US')
            self.write('delay(2)')
            self.display_SMU()
            if say_if_successful:
                log.info('Keithley connection successful at {}'.format(addr))
        except Exception as E:
            log.error('Keithley connection failed at {}'.format(addr))
            log.error(E)

    def connect(self, addr='TCPIP::192.168.11.11::inst0::INSTR'):
        if not self.connected():
            self.debug = False
            # Store up to 100 loops in memory in case you forget to save them to disk
            self.data = deque(maxlen=100)
            self.conn = visa_rm.get_instrument(addr, open_timeout=0)
            self.conn.timeout = 4000
            # Expose a few methods directly to self
            self.ask = self.query
            self.read = self.conn.read
            self.read_raw = self.conn.read_raw
            self.close = self.conn.close
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

    def write(self, msg):
        if self.debug:
            print(msg)
        self.conn.write(msg)

    def query(self, msg):
        if self.debug:
            print(msg)
        reply = self.conn.query(msg)
        if self.debug:
            print(reply)
        return reply

    def _set_or_query(self, prop, val=None):
        '''
        Sets prop = val in lua
        if val is None, return the current value of prop

        the instrument returns bools as floats, which is annoying
        '''
        if val is None:
            reply = self.query(f'print({prop})').strip()
            return self._string_parser(reply)
        else:
            if type(val) is bool:
                val = 1 if val else 0
            self.write(f'{prop} = {val}')
            return None

    def _string_parser(self, string):
        '''
        Since we have to communicate via strings and these strings might just be numeric
        Convert to numeric
        '''
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

    def idn(self):
        return self.conn.query('*IDN?').replace('\n', '')

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
        In order to send a list of values to Keithley, we need to compose a lua
        string to define it as a variable.

        Problem is that the input buffer of Keithley is very small, so the lua string
        needs to be separated into many lines. This function accomplishes that.
        '''
        chunksize = 50
        l = len(list_in)
        # List of commands to send to Keithley
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
        Wraps the SweepVList lua function defined on Keithley
        '''
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write(f'SweepVList(sweeplist, {Irange}, {Ilimit}, {Plimit}, {nplc}, {delay}, {Vrange})')

    def _iv_4pt_lua(self, vlist, Irange=0, Ilimit=0, nplc=1, delay='smua.DELAY_AUTO', Vrange=0):
        '''
        range = 0 enables autoranging
        Wraps the SweepVList lua function defined on Keithley
        '''
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write('SweepVList_4pt(sweeplist, {}, {}, {}, {}, {})'.format(Irange, Ilimit, nplc, delay, Vrange))

    def _vi_lua(self, ilist, Vrange=0, Vlimit=0, nplc=1, delay='smua.DELAY_AUTO', Irange=None):
        '''
        range = 0 enables autoranging
        if Irange not passed, it will be max(abs(ilist))
        Wraps the SweepIList lua function defined on Keithley
        '''
        self.send_list(ilist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        if Irange is None:
            # Fix the current source range, as I have had instability problems that are different
            # for different ranges
            Irange = np.max(np.abs(ilist))
        self.write(f'SweepIList(sweeplist, {Vrange}, {Vlimit}, {nplc}, {delay}, {Irange})')

    def _it_lua(self, sourceVA=0, sourceVB=0, points=10, interval=.1, rangeI=0, limitI=0, nplc=1):
        '''Wraps the constantVoltageMeasI lua function defined on Keithley'''
        # Call constantVoltageMeasI
        # TODO: make sure the inputs are valid
        self.write(f'constantVMeasI({sourceVA}, {sourceVB}, {points}, {interval}, {rangeI}, {limitI}, {nplc})')

    def iv(self, source_list, source_func='v', source_range=None, measure_range=None,
           v_limit=None, i_limit=None, p_limit=None,
           nplc=1, delay=None, point4=False, ch='a'):
        '''
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
        '''

        source_func = source_func.lower()
        ch = self._convert_to_ch(ch)
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
        if source_func == 'v':
            if v_limit is not None:
                source_list = np.clip(source_list, -v_limit, v_limit)
            if i_limit is not None:
                self.source_limit('i', i_limit, ch)
                self.trigger_source_limit('i', i_limit, ch)
        elif source_func == 'i':
            if v_limit is not None:
                self.source_limit('v', v_limit, ch)
                self.trigger_source_limit('v', v_limit, ch)
            if i_limit is not None:
                source_list = np.clip(source_list, -i_limit, i_limit)
        if p_limit is not None:
            self.source_limit('p', p_limit, ch)
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

    def vi(self, source_list, source_range=None, measure_range=None,
           v_limit=None, i_limit=None, p_limit=None,
           nplc=1, delay=None, point4=False, ch='a'):
        '''
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
        '''

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
               a_point4=False, b_point4=False,
               sync=True):

        if not isinstance(a_source_list, (list, np.ndarray)):
            a_source_list = a_source_list * np.ones(len(b_source_list))
        if not isinstance(b_source_list, (list, np.ndarray)):
            b_source_list = b_source_list * np.ones(len(a_source_list))

        if len(a_source_list) != len(b_source_list):
            raise Exception('Source values lists must have the same length')

        self.reset()

        def configure_channel(ch, source_list, source_func, source_range, measure_range,
                              v_limit, i_limit, p_limit, nplc, delay, point4):
            source_func = source_func.lower()
            ch = self._convert_to_ch(ch)
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

        if sync:
            self.trigger_blender_clear(1)
            self.trigger_blender_reset(1)
            self.trigger_blender_stimulus(1, ['a_sourcecomplete', 'b_sourcecomplete'])
            self.trigger_blender_clear(2)
            self.trigger_blender_reset(2)
            self.trigger_blender_stimulus(2, ['a_measurecomplete', 'b_measurecomplete'])
            for ch in ['a', 'b']:
                self.trigger_measure_stimulus('blender_1', ch=ch)
                self.trigger_endpulse_stimulus('blender_2', ch=ch)

        self.trigger_initiate('a')
        self.trigger_initiate('b')

    def done(self):
        ''' Ask Keithley if the measurement sequence is finished or not '''
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
                # Keep trying to read until Keithley says Complete
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
        # makes Keithley give numbers in ascii
        # self.write('format.data = format.ASCII')
        #readingstr = self.query('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
        #return np.float64(readingstr.split(', '))

        # Makes Keithley give numbers in binary float64
        # Should be much faster?
        self.write('format.data = format.REAL64')
        self.write('printbuffer({}, {}, {}.{})'.format(start, end, buffer, attr))
        # reply comes back with #0 or something in the beginning and a newline at the end
        raw = self.read_raw()[2:-1]
        # TODO: replace nanvals here, not in get_data
        data_array = np.fromstring(raw, dtype=np.float64)
        data_array = self.replace_nanvals(data_array)
        return data_array

    def get_data(self, start=1, end=None, history=True, ch='a'):
        '''
        Ask Keithley to print out the data arrays of interest (I, V, t, ...)
        Parse the strings into python arrays
        Return dict of arrays
        dict can also contain scalar values or other meta data

        Can pass start and end values if you want just a specific part of the arrays
        '''

        ch = self._convert_to_ch(ch)
        numpts = int(float(self.query(f'print(smu{ch}.nvbuffer1.n)')))
        if end is None:
            end = numpts
        if numpts > 0:
            # Output a dictionary with voltage/current arrays and other parameters
            out = {}
            out['units'] = {}
            out['longnames'] = {}
            out['channel'] = ch

            ### Collect measurement conditions
            # TODO: What other information is available from Keithley registers?
            #       nplc would be good to save..

            # Need to do something different if sourcing voltage vs sourcing current
            source = self.source_func(ch=ch)
            if source == 'v':
                out['source'] = 'V'
                out['V'] = self.read_buffer(f'smu{ch}.nvbuffer2', 'sourcevalues', start, end)
                Vmeasured = self.read_buffer(f'smu{ch}.nvbuffer2', 'readings', start, end)
                out['Vmeasured'] = Vmeasured
                out['units']['Vmeasured'] = 'V'
                I = self.read_buffer(f'smu{ch}.nvbuffer1', 'readings', start, end)
                out['I'] = I
                out['Icomp'] = float(self.query(f'print(smu{ch}.source.limiti)'))
            elif source == 'i':
                # Current source
                out['source'] = 'I'
                out['Vrange'] = float(self.query(f'print(smu{ch}.nvbuffer2.measureranges[1])'))
                out['Vcomp'] = float(self.query(f'print(smu{ch}.source.limitv)'))

                out['I'] = self.read_buffer(f'smu{ch}.nvbuffer1', 'sourcevalues', start, end)
                Imeasured = self.read_buffer(f'smu{ch}.nvbuffer1', 'readings', start, end)
                out['Imeasured'] = Imeasured
                out['units']['Imeasured'] = 'A'
                V = self.read_buffer(f'smu{ch}.nvbuffer2', 'readings', start, end)
                out['V'] = V

            out['t'] = self.read_buffer(f'smu{ch}.nvbuffer2', 'timestamps', start, end)
            out['Irange'] = self.read_buffer(f'smu{ch}.nvbuffer1', 'measureranges', start, end)
            out['Vrange'] = self.read_buffer(f'smu{ch}.nvbuffer2', 'measureranges', start, end)

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

    @staticmethod
    def _convert_to_ch(ch):
        '''
        :param ch: Name of the channel: 1, 2, 'a', 'b', 'A', 'B'
        :return: Name of the channel as a lower case: 'a', 'b'
        '''
        if ch in (1, 'a', 'A'):
            ch = 'a'
        elif ch in (2, 'b', 'B'):
            ch = 'b'
        else:
            raise Exception(f"Channel '{ch}' doesn't exist")
        return ch

    def reset(self):
        self.write('reset()')

    def reset_ch(self, ch='A'):
        ch = self._convert_to_ch(ch)
        self.write(f'smu{ch}.reset()')

    def waitcomplete(self):
        self.write('waitcomplete()')

    def prepare_buffers(self, source_func, buffers=None, ch='A'):
        '''
        Configure the typical buffer settings used for triggering.

        :param source_func: Type of measure: current (i), or voltage (v).
        :param buffers: List of the two buffers names.
        :param ch: Channel to be configured.
        :return: None
        '''

        if buffers is None:
            buffers = ['nvbuffer1', 'nvbuffer2']
        ch = self._convert_to_ch(ch)
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
        '''
        Configure the typical trigger settings.

        :param measurement: Current(i), voltage(v), resistance(r), power(p), or current and voltage (iv).
        :param source_list: List of values to be sourced.
        :param ch: Channel to be configured.
        :return: None
        '''

        self.trigger_measure_action(True, ch=ch)
        self.trigger_measure(measurement, ch=ch)
        self.trigger_endpulse_action('hold', ch=ch)
        self.trigger_endsweep_action('idle', ch=ch)
        num_points = len(source_list)
        self.trigger_count(num_points, ch=ch)
        self.trigger_source_action(True, ch=ch)
        self.source_output(True, ch=ch)

    def measure(self, measurement='i', ch='A'):
        '''
        This function makes one or more measurements.

        :param measurement: Parameter to measure. I t can be current (i), voltage (v), resistance (r),
        power (p) or (iv). In the last case it returns the last actual current measurement and voltage
        measurement.
        :param ch: Channel where measure.
        :return: Measurement value.
        '''
        ch = self._convert_to_ch(ch)
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

    def measure_range(self, meas_func='i', m_range=None, ch='A'):
        '''
        Set the SMU to a fixed range large enough to measure the assigned value.

        :param meas_func: Type of measure: current (i), or voltage (v).
        :param m_range: Range to be set. In amperes or volts.
        :param ch: Channel to be configured.
        :return: If m_range is None, configured ranged is returned.
        '''
        meas_func = meas_func.lower()
        ch = self._convert_to_ch(ch)
        if meas_func == 'v':
            range_func = 'i'
        elif meas_func == 'i':
            range_func = 'v'
        else:
            raise Exception("meas_func can only be 'v' or 'i'.")
        if type(m_range) == str:
            if m_range.lower() == 'auto':
                return self._set_or_query(f'smu{ch}.measure.autorange{range_func}', True)
            else:
                raise Exception("'m_range' can only be 'auto' if it's a string")
        else:
            return self._set_or_query(f'smu{ch}.measure.range{range_func}', m_range)

    def source_level(self, source_func='v', source_val=None, ch='A'):
        '''
        Set the source level, or ask for it.

        :param source_func: Parameter to source. 'i' if current and 'v' if voltage.
        :param source_val: Value to set the source. If None, this function returns the previous level.
        :param ch: Channel to which set the source level.
        :return: Source level.
        '''
        ch = self._convert_to_ch(ch)
        if source_func in ['i', 'v']:
            reply = self._set_or_query(f'smu{ch}.source.level{source_func}', source_val)
        else:
            raise Exception("Parameter 'source_func' can only be 'i' or 'v'.")
        return reply

    def source_func(self, func=None, ch='A'):
        '''
        This function set the source as voltage (v) or current (i).

        :param func: Voltage (v) or current (i).
        :param ch: Channel to which apply changes.
        :return: If func is None it returns the current value.
        '''

        ch = self._convert_to_ch(ch)
        if func is not None:
            if type(func) is str:
                if func.lower() == 'i':
                    func = f'smu{ch}.OUTPUT_DCAMPS'
                elif func.lower() == 'v':
                    func = f'smu{ch}.OUTPUT_DCVOLTS'
                else:
                    raise Exception("func can only be 'i', 'v' or None")

        a = self._set_or_query(f'smu{ch}.source.func', func)

        if a == 0:
            return 'i'
        elif a == 1:
            return 'v'

    def source_limit(self, source_param='i', limit=None, ch='A'):
        '''
        Set compliance limits for current (i), voltage(v) or power(p).
        This attribute should be set in the test sequence before the turning the source on.

        Voltage can only be limited when sourcing current.
        Current can only be limited when sourcing voltage.
        Power limits look to behave like trigger_source_limit, so the limit must be higher than
        10% of the measurement range.

        :param source_param: Source parameter to limit: current (i), voltage(v) or power(p).
        :param limit: Limit to be set. In amperes, volts or watts.
        :param ch: Channel to be configured.
        :return: If limit is None, configured limit is returned.
        '''
        ch = self._convert_to_ch(ch)
        source_param = source_param.lower()
        source_func = self.source_func(ch=ch)
        if source_param == 'v' and source_func == 'v':
            log.warning("Voltage can not be limited when sourcing voltage")
        elif source_param == 'i' and source_func == 'i':
            log.warning("Current can not be limited when sourcing current")

        return self._set_or_query(f'smu{ch}.source.limit{source_param}', limit)

    def source_range(self, source_func='v', s_range=None, ch='A'):
        '''
        Set the SMU to a fixed range large enough to source the assigned value.
        Autoranging could be very slow.

        :param source_func: Type of source: current (i), or voltage (v).
        :param s_range: Range to be set. In amperes or volts.
        :param ch: Channel to be configured
        :return: If s_range is None, configured ranged is returned.
        '''
        source_func = source_func.lower()
        ch = self._convert_to_ch(ch)
        if type(s_range) == str:
            if s_range.lower() == 'auto':
                return self._set_or_query(f'smu{ch}.source.autorange{source_func}', True)
            else:
                raise Exception("'s_range' can only be a string if it's value is 'auto'")
        else:
            return self._set_or_query(f'smu{ch}.source.range{source_func}', s_range)

    def source_output(self, state=None, ch='A'):
        '''
        Set output state
        toggles output relays and turns front LEDs on or off
        '''
        ch = self._convert_to_ch(ch)
        return self._set_or_query(f'smu{ch}.source.output', state)

    def trigger_source_limit(self, source_param, limit=None, ch='A'):
        '''
        Set the sweep source limit for current.

        If this attribute is set to any other numeric value, the SMU will switch
        in this limit at the start of the source action and will return to the normal
        limit setting at the end of the end pulse action.

        :param source_param: Source parameter to be limited: current (i), or voltage (v).
        :param limit: Limit to be set. In amperes or volts.
        :param ch: Channel to be configured.
        :return: Previous limit, if 'limit'=None
        '''

        ch = self._convert_to_ch(ch)
        source_param = source_param.lower()
        sf = self.source_func(ch=ch)
        if sf is 'v':
            if source_param == 'i':
                mr = self.measure_range('i')
                log.debug(f"Measure range: {mr}A")
                lim_min = 0.1 * mr
                log.debug(f"Minimum limit: {lim_min}A\n"
                          f"Your limit: {limit}A")
                if limit < lim_min:
                    log.warning(f"Your current limit is lower than 10% of the measure range.\n"
                                f"Current limit will be set to {lim_min}A")
            elif source_param == 'v':
                log.warning("You can not limit the voltage when sourcing voltage.")
        elif sf is 'i':
            if source_param == 'v':
                mr = self.measure_range('v')
                log.debug(f"Measure range: {mr}V")
                lim_min = 0.1 * mr
                log.debug(f"Minimum limit: {lim_min}V\n"
                          f"Your limit: {limit}V")
                if limit < lim_min:
                    log.warning(f"Your voltage limit is lower than 10% of the measure range.\n"
                                f"Voltage limit will be set to {lim_min}V")
            elif source_param == 'i':
                log.warning("You can not limit the current when sourcing current.")

        return self._set_or_query(f'smu{ch}.trigger.source.limit{source_param}', limit)

    def trigger_source_list(self, source_func, source_list, ch='A'):
        '''
        Configure source list sweep for SMU channel.

        :param source_func: Type of measure: current (i), or voltage (v).
        :param source_list: List to be set. In amperes or volts.
        :param ch: Channel to be configured.
        :return: None
        '''

        ch = self._convert_to_ch(ch)
        source_func = source_func.lower()
        self.send_list(source_list, 'listvarname')
        return self.write(f'smu{ch}.trigger.source.list{source_func}(listvarname)')

    def trigger_measure(self, measurement, buffer=None, ch='A'):
        '''
        This function configures the measurements that are to be made in a subsequent sweep.

        :param measurement: Current(i), voltage(v), resistance(r), power(p), or current and voltage (iv).
        :param buffer: Name of the the buffer to save the data. If 'iv' it must be a list of two names.
        :param ch: Channel to be configured.
        :return: None
        '''

        ch = self._convert_to_ch(ch)
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
        '''
        Possible action values:
            True: Make measurements during the sweep
            False: Do not make measurements during the sweep
            'ASYNC': Make measurements during the sweep, but asynchronously with the source area of the trigger model
        '''
        ch = self._convert_to_ch(ch)

        if (type(action) is str) and (action.lower() == 'async'):
            action = 2
        elif (action is not None) and (type(action) is not bool):
            raise Exception('action can only be True, False or ASYNC')

        return self._set_or_query(f'smu{ch}.trigger.measure.action', action)

    def trigger_endpulse_action(self, action=None, ch='A'):
        '''
        This attribute enables or disables pulse mode sweeps.

        Possible action values:
            IDLE: Enables pulse mode sweeps, setting the source level to the programmed (idle) level at the end of the pulse.
            HOLD: Disables pulse mode sweeps, holding the source level for the remainder of the step.
            (where X is the chanel selcted, such as "a" )
        '''
        ch = self._convert_to_ch(ch)
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
        '''
        This attribute sets the action of the source at the end of a sweep

        Possible action values:
            IDLE: Sets the source level to the programmed (idle) level at the end of the sweep
            HOLD: Sets the source level to stay at the level of the last step.
            (where X is the chanel selcted, such as "a" )
        '''
        ch = self._convert_to_ch(ch)
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
        '''
        This attribute sets the trigger count in the trigger model.

        The trigger count is the number of times the source-measure unit (SMU)
        will iterate in the trigger layer for any given sweep.

        If this count is set to zero (0), the SMU stays in the trigger model
        indefinitely until aborted.
        '''
        ch = self._convert_to_ch(ch)
        return self._set_or_query(f'smu{ch}.trigger.count', count)

    def trigger_source_action(self, action=None, ch='A'):
        '''
        This attribute enables or disables sweeping the source (on or off).

        Possible action values:
            False: Do not sweep the source
            True: Sweep the source
        '''
        ch = self._convert_to_ch(ch)
        return self._set_or_query(f'smu{ch}.trigger.source.action', action)

    def trigger_initiate(self, ch='A'):
        '''
        This function initiates a sweep operation.

        This function causes the SMU to clear the four trigger model event
        detectors and enter its trigger model (moves the SMU from the idle
        state into the arm layer).
        '''
        ch = self._convert_to_ch(ch)
        self.write(f'smu{ch}.trigger.initiate()')

    def trigger_blender_reset(self, N):
        '''
        This function resets some of the trigger blender settings to their factory defaults.
        '''
        self.write(f'trigger.blender[{N}].reset()')

    def trigger_blender_clear(self, N):
        '''
        This function clears the command interface trigger event detector
        '''
        self.write(f'trigger.blender[{N}].clear()')

    def trigger_blender_stimulus(self, N, stimulus, or_mode=False):
        '''
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
        '''
        if or_mode is True:
            enable = 'true'
        elif or_mode is False:
            enable = 'false'
        else:
            log.error("Error at trigger_blender_stimulus: 'or_mode' can only be set as True or False")
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
        '''
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
        '''
        ch = self._convert_to_ch(ch)
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
        '''
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
        '''
        ch = self._convert_to_ch(ch)
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
        '''
        Possible values for state:
            LOCAL: Selects local sense (2-wire)
            REMOTE: Selects local sense (4-wire)
            CALA: Selects calibration sense mode
        '''
        ch = self._convert_to_ch(ch)
        state = state.upper()

        return self.write(f'smu{ch}.sense = smu{ch}.SENSE_{state}')

    def nplc(self, nplc=None, ch='A'):
        '''
        This command sets the integration aperture for measurements
        This attribute controls the integration aperture for the analog-to-digital converter (ADC).
        The integration aperture is based on the number of power line cycles (NPLC), where 1 PLC for 60 Hzis 16.67 ms (1/60) and 1 PLC for 50 Hz is 20 ms (1/50)
        '''
        ch = self._convert_to_ch(ch)
        return self._set_or_query(f'smu{ch}.measure.nplc', nplc)

    def measure_delay(self, delay=None, ch='A'):
        '''
        This attribute controls the measurement delay
        Specify delay in second.
        To use autodelay set delay='AUTO'
        '''
        ch = self._convert_to_ch(ch)
        try: delay=delay.lower()
        except AttributeError: pass
        if delay == 'auto': delay = f'smu{ch}.DELAY_AUTO'
        return self._set_or_query(f'smu{ch}.measure.delay', delay)

    def clear_buffer(self, buffer='nvbuffer1', ch='A'):
        '''
        This function empties the buffer

        This function clears all readings and related recall attributes from the buffer (for example,
        bufferVar.timestamps and bufferVar.statuses) from the specified buffer.
        '''
        ch = self._convert_to_ch(ch)
        return self.write(f'smu{ch}.{buffer}.clear()')

    def collect_timestamps(self, state=None, buffer='nvbuffer1', ch='A'):
        '''
        This attribute sets whether or not timestamp values are stored with the readings in the buffer

        Possible state values:
             False: Timestamp value collection disabled (off)
             True: Timestamp value collection enabled (on)
        '''
        ch = self._convert_to_ch(ch)
        return self._set_or_query(f'smu{ch}.{buffer}.collecttimestamps', state)

    def collect_timevalues(self, state=None, buffer='nvbuffer1', ch='A'):
        '''
        This attribute sets whether or not source values are stored with the readings in the buffer

        Possible source values:
            False: Source value collection disabled (off)
            True: Source value collection enabled (on)
        '''
        ch = self._convert_to_ch(ch)
        return self._set_or_query(f'smu{ch}.{buffer}.collecttimevalues', state)

    def collect_sourcevalues(self, state=None, buffer='nvbuffer1', ch='A'):
        '''
        This attribute sets whether or not source values will be stored with the readings in the buffer
        '''
        ch = self._convert_to_ch(ch)
        return self._set_or_query(f'smu{ch}.{buffer}.collectsourcevalues', state)

    def display(self, line1='', line2=''):
        '''
        Very important function for writing silly things on the screen
        there can be 20 characters per line

        $N Starts text on the next line (newline)
        $R Sets text to Normal.
        $B Sets text to Blink.
        $D Sets text to Dim intensity.
        $F Set text to background blink.
        $$ Escape sequence to display a single “$”.
        '''
        self.write('display.clear()')
        self.write('display.setcursor(1, 1)')
        lines = '$R$N'.join((line1, line2))
        self.write(f'display.settext("{lines}")')

    def display_SMU(self, a=True, b=True):
        '''
        Displays source-measure for SMU A and/or SMU B
        '''
        if a & b:
            self.write('display.screen = display.SMUA_SMUB')
        elif a:
            self.write('display.screen = display.SMUA')
        elif b:
            self.write('display.screen = display.SMUB')
        else:
            self.write('display.screen = display.USER')


