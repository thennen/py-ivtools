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

    def iv(self, vlist, Irange=0, Ilimit=0, Plimit=0, nplc=1, delay='smua.DELAY_AUTO', Vrange=0):
        '''
        range = 0 enables autoranging
        Wraps the SweepVList lua function defined on keithley
        '''
        # Send list of voltage values to keithley
        self.send_list(vlist, varname='sweeplist')
        # TODO: make sure the inputs are valid
        self.write('SweepVList(sweeplist, {}, {}, {}, {}, {}, {})'.format(Irange, Ilimit, Plimit, nplc, delay, Vrange))

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

    def output(self, state=None, ch='A'):
        # Set output state
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.output', state, bool=True)

    def measure_autorangei(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.measure.autorangei', state, bool=True)

    def measure_autorangev(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.measure.autorangev', state, bool=True)

    def measure_rangei(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.measure.rangei', state)

    def measure_rangev(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.measure.rangev', state)

    def measurei(self, ch='A'):
        # Request a current reading
        ch = ch.lower()
        reply = self.query(f'print(smu{ch}.measure.i())')
        return float(reply)

    def measurev(self, ch='A'):
        # Request a voltage reading
        ch = ch.lower()
        reply = self.query(f'print(smu{ch}.measure.v())')
        return float(reply)

    def measurer(self, ch='A'):
        # Request a resistance reading
        ch = ch.lower()
        reply = self.query(f'print(smu{ch}.measure.r())')
        return float(reply)

    def measurep(self, ch='A'):
        # Request a power reading
        ch = ch.lower()
        reply = self.query(f'print(smu{ch}.measure.p())')
        return float(reply)

    def measureiv(self, ch='A'):
        # Request a current and voltage reading
        ch = ch.lower()
        reply = self.query(f'print(smu{ch}.measure.iv())')
        i, v = reply.split('\t')
        return float(i), float(v)

    def source_autorangev(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.autorangev', state, bool=True)

    def source_autorangei(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.autorangei', state, bool=True)

    def source_func(self, state=None, ch='A'):
        # 'i' or 'v'
        # 1 for volts, 0 for current
        ch = ch.lower()
        if state is not None:
            if state.lower() == 'i':
                state = 0
            elif state.lower() == 'v':
                state = 1
        reply = self._set_or_query(f'smu{ch}.source.func', state)
        if reply is None: return None
        return 'v' if int(reply) else 'i'

    def source_leveli(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.leveli', state)

    def source_levelv(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.levelv', state)

    def source_limiti(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.limiti', state)

    def source_limitv(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.limitv', state)

    def source_limitp(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.limitp', state)

    def source_rangei(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.rangei', state)

    def source_rangev(self, state=None, ch='A'):
        ch = ch.lower()
        return self._set_or_query(f'smu{ch}.source.rangev', state)

    def sense(self, state=None, ch='A'):
        # local (2-wire), remote (4-wire)
        # 0 for local, 1 for remote
        ch = ch.lower()
        if state is not None:
            if state.lower() == 'local':
                state = 0
            elif state.lower() == 'remote':
                state = 1
        reply = self._set_or_query(f'smu{ch}.source.func', state)
        if reply is None: return None
        return 'remote' if int(reply) else 'local'
