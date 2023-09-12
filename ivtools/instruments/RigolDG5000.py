import numpy as np
import time
import os
import hashlib
import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__

class RigolDG5000(object):
    '''
    This instrument is really a pain in the ass.  Good example of a job not well done by Rigol.
    In addition to the terrible RF design (output resistance is not 50 Ω), the programming is extremely flaky and many of the datasheet specs are false.

    But we spent a lot of time learning its quirks and are kind of stuck with it for now.

    Do not send anything to the Rigol that differs in any way from what it expects,
    or it will just hang forever and need to be manually restarted along with the entire python kernel.

    Certain commands just randomly cause the machine to crash in the same way.  Such as when you try
    to query the number of points stored in the NV memory
    '''
    def __init__(self, addr=None):
        self.verbose = False
        try:
            if not self.connected():
                if addr is None:
                    addr = self.get_visa_addr()
                self.connect(addr)
        except:
            log.error('Rigol connection failed.')
            return
        # Turn off screen saver.  It sends a premature pulse on SYNC output if on.
        # This will make the scope trigger early and miss part or all of the pulse.  Really dumb.
        self.screensaver(False)
        # Store the last waveform that was programmed, so that we can skip uploading if it
        # hasn't changed
        # One waveform per channel
        self.volatilewfm = {1:[], 2:[]}

    @staticmethod
    def get_visa_addr():
        # Look for the address of the DG5000 using visa resource manager
        # Prefer DG5102
        resources = visa_rm.list_resources()
        for resource in resources:
            if 'DG5T155000186' in resource:
                return resource
        for resource in resources:
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
            idn = self.query('*IDN?').replace('\n', '')
            log.debug('Rigol connection succeeded. *IDN?: {}'.format(idn))
        except:
            log.error('Connection to Rigol AWG failed.')

    def connected(self):
        if hasattr(self, 'conn'):
            try:
                self.idn()
                return True
            except:
                pass
        return False

    def set_or_query(self, cmd, setting=None):
        # Sets or returns the current setting
        if setting is None:
            if self.verbose: log.info(cmd + '?')

            try:
                reply = self.query(cmd + '?').strip()
            except Exception as E:
                # This error might be related to having a different __pycache__ directory unexpectedly added to your path?
                # I think you can safely eat the first error and just try again
                # VI_ERROR_INP_PROT_VIOL (-1073807305): Device reported an input protocol error during transfer.
                # ...\Anaconda3\Lib\site-packages\__pycache__
                # Cannot directly catch the exception because it does not inherit from BaseException
                if hasattr(E, 'error_code') and E.error_code == visa.errors.VI_ERROR_INP_PROT_VIOL:
                    log.warning('VI_ERROR_INP_PROT_VIOL encountered. Simply trying again as the error seems to resolve itself.')
                    reply = self.query(cmd + '?').strip()
                else:
                    raise E


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
            if self.verbose: log.info(f'{cmd} {setting}')
            self.write(f'{cmd} {setting}')
            return None

    @staticmethod
    def write_wfm_file(wfm, filepath=None, drive='F'):
        '''
        The only way to get anywhere near the advertised number of samples
        theoretically works up to 16 MPts = 2**24 samples
        wfm should be between -1 and 1, this will convert it to uint16
        Can load up to 512 MPts in "play mode", which reduces the sample rate
        There are magic values of waveform lengths that can be used, there is no obvious logic to this
        safe values are anything < 2^19 = 524,288 samples, and any whole power of 2 after that
        if > 2^14 = 16383 points, the bursts are delayed by ~910 ns after the trigger..
        '''
        if filepath is None:
            filepath = f'{drive}:\\' + hashlib.md5(wfm).hexdigest()[:8] + '.RAF'
        else:
            # Needs extension RAF!
            filepath = os.path.splitext(filepath)[0] + '.RAF'

        if np.any(np.abs(wfm) > 1):
            log.warning('Waveform must be in [-1, 1].  Clipping it!')
            A = np.clip(A, -1, 1)
        wfm = ((wfm + 1)/2 * (2**14 - 1))
        n = len(wfm)
        if (n > 2**19) and (np.log(n)/np.log(2)%1 != 0):
            log.info('write_wfm_file: If waveform has more than 2^19 points, it should have a whole power of 2 points!')
        wfm = np.round(wfm).astype(np.dtype('H'))
        log.info(f'Writing binary waveform to {filepath}')
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
        # Default is infinity
        return self.set_or_query(f'OUTPUT{ch}:IMPEDANCE', r)

    def sync(self, state=None):
        ''' Can turn on/off the sync output (on rear) '''
        if state is not None:
            state = 'ON' if state else 'OFF'
        return self.set_or_query(f'OUTPUT{ch}:SYNC', state)

    def screensaver(self, state=None):
        ''' Turn the screensaver on or off.
        Screensaver causes serious problems with triggering because DG5000 is a piece of junk. '''
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
        err = self.query(':SYSTem:ERRor?').strip()
        if err == '0,"No error"':
            # So you can do "if rigol.error()"
            return False
        return err

    # <<<<< For burst mode
    def ncycles(self, n=None, ch=1):
        ''' Set number of cycles that will be output in burst mode '''
        if (n is not None) and (n > 1000000):
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
        '''
        Set the mode of burst mode.  I don't know what it means. 'TRIGgered|GATed|INFinity
        Resets your idle level to zero for some reason!
        therefore only sets the mode if you are not already in the mode
        '''
        currentmode = self.set_or_query(f':SOURCE{ch}:BURST:MODE', None)
        if mode is None:
            return currentmode
        # If already in the requested state, don't send this command because it has side effects
        elif currentmode != mode:
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

    def cd(self, dir='D:\\'):
        # Change directory.  Can crash rigol.
        self.write(f':MMEM:CDIR \"{dir}\"')

    def listdir(self):
        '''
        List the files in the current directory
        Highly unreliable.  Rigol can crashes on whatever command you send it after this!
        Errors not consistently repeatable
        File sizes have come back different -- they are probably wrong
        '''
        horrible_string = self.query('MMEM:CAT?')
        quote = horrible_string.find('\"')
        first_number,second_number = horrible_string[:quote-1].split(',')
        rest = horrible_string[quote:].strip().strip('\"').split('\",\"')
        splitrest = [r.split(',') for r in rest]
        size,wtf,fn = zip(*splitrest)
        # Idiot rigol writes .RAF.RAF when it is just .RAF
        fn = [n.replace('.RAF.RAF', '.RAF') for n in fn]
        #out = {f:s for f,s in zip(fn,size)}
        return fn

    def writebinary(self, message, values):
        self.conn.write_binary_values(message, values, datatype='H', is_big_endian=False)

    ### Waveform loading by many different methods, all of which are terrible for their own set of reasons

    # CAREFUL: for some undocumented reason, Rigol turns its sync ports off (:OUTP1:SYNC OFF) while uploading waveforms
    # this causes a 10 mV edge on the sync port which can cause unwanted triggering!
    # problem goes away if the line is terminated, but trigger ports may be high impedance (i.e. picoscope 6000e series)

    def load_wfm_usbdrive(self, filename='wfm.RAF', wait=True):
        '''
        Load waveform from usb drive.  Should be a binary sequence of unsigned shorts.
        File needs to have extension .RAF
        This is the only way to reach the advertised number of waveform samples, or anywhere near it
        Should be able to go to 16 MPts on normal mode, 512 MPts on play mode, but this was not tested

        wait=True can cause problems, because it uses another command to query whether rigol is responding
        again, but this command itself can make rigol puke..
        Ideally you just have a good idea how long it takes to load the waveform, and you wait manually..
        This seems to only be an issue if you have a lot of waveforms to load sequentially

        Like everything on rigol, this can be flaky. Can not have 100% confidence that the waveform loaded properly
        Usually if it didn't load it, it will take an abnormally short time to respond to the next command
        It does not "like" fractional powers of 2 if the waveform is longer than 2^19 = 524kSamples
        but sometimes, unaccountably, it can load them anyway.  I wouldn't trust it.

        It can also just crash and cause the python program to hang indefinitely
        The only solution seems to be to cycle the power on rigol and usually restart the python kernel..
        '''
        self.write(':MMEMory:CDIR "D:\"')
        self.write(f':MMEMory:LOAD "{filename}"')
        if wait:
            oldtimeout = self.conn.timeout
            # set timeout to a minute!
            self.conn.timeout = 60000
            time.sleep(1) # Stupid thing
            # Rigol won't reply to this until it is done loading the waveform
            err = self.error()
            #self.idn()
            self.conn.timeout = oldtimeout
            # This shit causes an error every time now.  Used to work.
            if err:
            #    raise Exception(err)
                log.error(err)

    def load_wfm_binary(self, wfm, ch=1):
        """
        TODO: write about all the bullshit involved with this
        I have seen these waveforms simply fail to trigger unless you wait a second after enabling the channel output
        You need to wait after loading this waveform and after turning the output on, sometimes an obscene amount of time?
        the "idle level" in burst mode will be the first value of the waveform ??
        No, the idle level is unpredictable, which is the killer for this upload mode
        """
        CHUNK = 2**14
        # vertical resolution
        VRES = 2**14
        # pad with zero value
        PADVAL = 2**13

        nA = len(wfm)
        log.warning(f'You are trying to program {nA} pts')
        if nA > 2**16:
            log.warning('Programming over 2^16 = 65,536 points by this method usually leads to problems!')

        A = np.array(wfm)
        if np.any(np.abs(A) > 1):
            log.warning('Waveform must be in [-1, 1].  Clipping it!')
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
        log.info(f'I am sending {NptsProg} points in {Nchunks} chunks')

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
            log.info(val)
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
        if np.any(waveform != self.volatilewfm[ch]):
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
            self.volatilewfm[ch] = waveform
        else:
            # Just switch to the arbitrary waveform that is already in memory
            log.info('Volatile wfm already uploaded, skipping re-upload.')
            if self.shape(ch=ch) != 'USER':
                self.shape('USER', ch) # makes a small sound, might take time?
        freq = 1. / duration
        self.frequency(freq, ch=ch)
        maxamp = np.max(np.abs(waveform))
        self.amplitude(2*maxamp, ch=ch)
        # Apparently offset affects the arbitrary waveforms, too
        self.offset(offset, ch)
        # Turn on interpolation for IVs, off for steps
        self.interp(interp)

    def setup_burstmode(self, n=1, burstmode='TRIG', trigsource='MAN', ch=1):
        '''
        Several commands grouped together to set up bursting
        MIGHT temporarily mess with your idle level until you send the first pulse
        this is because of the burstmode command
        '''
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
        '''
        SINusoid|SQUare|RAMP|PULSe|NOISe|USER|DC|SINC|EXPRise|EXPFall|CARDiac|
        GAUSsian |HAVersine|LORentz|ARBPULSE|DUAltone
        '''
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
