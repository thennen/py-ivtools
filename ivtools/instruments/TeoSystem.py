import numpy as np
import itertools
import time
import win32com.client
from win32com.client import CastTo, WithEvents, Dispatch
from pythoncom import com_error
import hashlib
import logging
from numbers import Number
log = logging.getLogger('instruments')

class TeoSystem(object):
    '''
    Class for control of Teo Systems Memory Tester
    TEO: Name=TS_MemoryTester SN=201428 Rev=2.4

    TODO: put specs and basic explanation of what the system does here
    fixed 500 MSamples/s sample rate for internal AWG and ADCs
    +-10V output voltage amplitude, with 14 bit resolution (~5 mV), 50 ohm output
    rise/fall time < 1 ns
    ADC has 12 bit resolution
    ADC memory is 256 MSamples per channel
    ADC bandwidth is about 200 MHz

    COM objects directly exposed:
    DriverID
    DeviceID
    DeviceControl
    LF_Measurement
    HF_Measurement
    HF_Gain
    AWG_WaveformManager

    Brings some of the most used methods that are inconveniently deep
    in the object tree to the top level and gives them shorter names

    Most of the functions I add start with a lowercase letter.
    All of TEOs functions are CapitalCamelCased, sometimes with underscores.

    Adds content-addressability of waveforms for effortless replay

    Seems to handle re-initialization just fine.
    You can make two instances and they will both work


    # TODO: does the class need to store any internal state?  should we make it BORG?

    # TODO: write high level methods (e.g. pulse_and_capture..)

    # TODO: only voltage amplitude is used for the hash, maybe we should hash the triggers as well

    # TODO: how can we be aware of the TEO memory state?  do we care?

    # TODO: TEO remembers waveforms that you upload by name
            this is to minimize unecessary data transfer, which takes time
            we should also have this class remember the waveforms in a similar way (done)
            would be useful then to also have a method that synchronizes the memories

    # TODO: methods to read the waveforms back that are already on TEO memory? (done)
            but is there a way to read the triggers back?

    # TODO: figure out what the gain really does and document it
            at what voltage does the ADC saturate vs gain?

    # TODO: do all the commands work regardless of which mode we are in? e.g. waveform upload, gain setting

    # TODO: what happens when we send commands while the board is busy?

    # TODO: my understanding is that there is an idle voltage level set by
            LF_Measurement.LF_Voltage.SetValue(DClevel)
            is this always applied when a waveform is not playing? (yes)
            does that mean the instrument switches into LF mode when a waveform is not playing? (no)

    # TODO: there are apparently restrictions on what size samples you should send to the AWG
            there's a chunk size or something. e.g. 1024 plays fine but anything below that seems broken
            we need code that pads arrays that don't conform to the right chunk size
            Teo said that they need to be padded to a multiple of 2048
            but that his software/firmware is supposed to do it for you
            but there are bugs (try very short waveforms)

    # TODO: could wrap some of the functions only for the purpose of giving them signatures, docstrings,
            default arguments.

    # TODO: should we hide the entire COM interface in a container?  like
            self.com.DeviceControl, self.com.LF_Measurement etc
            then on the top level we have mostly stuff that we have defined that has docstrings and
            so on, but we can still access the com interface if needed.

            downside is that when we ask for support our code will be unrecognizable to Teo.
            therefore:
    # TODO: should we have some kind of a debug mode that prints out all the COM calls?


    # TODO: since there seem to be a lot of situations that cause the output to go to the negative rail
            and blow up your device, document them here
            seems to be whenever TSX_DM.exe connects to the system
            1. on first initialization (Dispatch('TSX_HMan'))
            2. if you disconnect USB and plug it back in

    # TODO: should we store a calibration to remove the offsets? monitor has a big offset and for
            the other channel it depends a bit on the gain setting.

    '''

    def __init__(self):
        '''
        This will do software/hardware initialization and set HFV output voltage to zero
        requires TEO software package and drivers to be installed on the PC

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Make sure there is no DUT connected when you initialize!
        HFV output goes to the negative rail!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        try:
            # Launches programs for software interface to TEO board
            # TSX_DM.exe is the process we communicate with to send commands to the board
            # TSX_HardwareManager.exe is a gui that sits in the tray area that displays whether
            # a board is connected.  it communicates with TSX_DM.exe and does not seem to be needed
            # for the python code to function.
            # First time it runs:
            #   Takes a few seconds to start
            #   round board gets power and HFV output goes to the negative rail!!!
            # subsequent runs also work and seem not to produce anything bad on the output
            HMan = Dispatch('TSX_HMan')
        except com_error as e:
            # TODO make sure this is necessarily the meaning of this error
            raise type(e)(str(e) +
                          ' TEO software not installed?').with_traceback(sys.exc_info()[2])

        # Asks the program for a device called MEMORY_TESTER
        MemTester = HMan.GetSystem('MEMORY_TESTER')
        if MemTester is None:
            raise Exception('Teo software cannot locate a connected memory tester. Check USB connection.')

        # Access a bunch of classes used to control the TEO board.
        # The contained methods/attributes appear in tab completion, but the contained classes do not
        DriverID =            TeoSystem.CastTo('ITS_DriverIdentity'     , MemTester)
        DeviceID =            TeoSystem.CastTo('ITS_DeviceIdentity'     , DriverID)
        DeviceControl =       TeoSystem.CastTo('ITS_DeviceControl'      , DriverID)
        LF_Measurement =      TeoSystem.CastTo('ITS_LF_Measurement'     , DriverID)
        #LF_Voltage =          LF_Measurement.LF_Voltage # ?
        HF_Measurement =      TeoSystem.CastTo('ITS_HF_Measurement'     , DriverID)
        # Is this different from HF_Gain = HF_Measurement.HF_Gain?
        # TODO: why can't we see e.g. HF_Measurement.HF_Gain in tab completion?
        HF_Gain =             TeoSystem.CastTo('ITS_DAC_Control'        , HF_Measurement.HF_Gain)
        AWG_WaveformManager = TeoSystem.CastTo('ITS_AWG_WaveformManager', HF_Measurement.WaveformManager)

        # Assign com methods/attributes to the instance
        self.HMan = HMan
        self.MemTester = MemTester
        self.DriverID = DriverID
        self.DeviceID = DeviceID
        self.DeviceControl = DeviceControl
        self.LF_Measurement = LF_Measurement
        self.HF_Measurement = HF_Measurement
        self.HF_Gain = HF_Gain
        self.AWG_WaveformManager = AWG_WaveformManager

        # TODO: Break the hierarchy for some functions?
        self.GetFreeMemory = AWG_WaveformManager.GetFreeMemory
        self.StopDevice = DeviceControl.StopDevice


        # TODO: assign properties that do not change, like max/min values
        #       so that we don't keep polling the instrument for fixed values

        # Store them in this dumb container so they don't clutter everything
        class dotdict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        self.constats = dotdict()
        self.constants = type('constants', (), {})
        self.constants.idn = self.idn()
        self.constants.max_LF_Voltage = self.LF_Measurement.LF_Voltage.GetMaxValue()
        self.constants.min_LF_Voltage = self.LF_Measurement.LF_Voltage.GetMaxValue()
        self.constants.max_HFgain = self.HF_Gain.GetMaxValue() # 10
        self.constants.min_HFgain = self.HF_Gain.GetMinValue() # -8
        self.constants.max_LFgain = self.LF_Measurement.LF_Gain.GetMaxValue() # 0?
        self.constants.min_LFgain = self.LF_Measurement.LF_Gain.GetMinValue() # also 0?
        self.constants.AWG_memory = self.AWG_WaveformManager.GetTotalMemory()

        # if you have the jumper, HFI impedance is 50 ohm, otherwise 100 ohm
        self.J29 = True

        # set the power line frequency for averaging over integer cycles
        self.PLF = 50

        # Store the same waveform/trigger data that gets uploaded to the board
        # TODO: somehow prevent this from taking too much memory
        #       should always reflect the state of the teo board
        self.waveforms = {}
        # Store the name of the last waveform output
        self.last_waveform = None
        # and gain used
        self.last_gain = None


        # Teo says this powers up round board, but the LEDS are already on by the time we call it.
        # it appears to be fine to call it multiple times:
        # everything still works, and I didn't see any disturbances on the HFV output

        # TODO: what state do we exactly start up in the first time this is called?
        # it seems to start up in HF mode, but I don't see the internal pulses on HFV,
        # so maybe it starts up in external mode (probably not!)
        # subsequent calls seem to stay in whatever mode it was in before,
        # even if we lost the python-TSX_DM connection for some reason
        DeviceControl.StartDevice()
        # This is output even on HF mode, when the waveform isn't playing!
        LF_Measurement.LF_Voltage.SetValue(0)
        #self.HF_mode(external=False)


    ###### Direct wrappers for adding python function signatures and docstrings ####

    def StopDevice(self):
        '''
        Lights should turn off on the round board and HFV output probably floats.
        Controller board remains on.
        '''
        self.DeviceControl.StopDevice()

    # TODO add more


    ################################################################################

    @staticmethod
    def CastTo(name, to):
        # CastTo that clearly lets you know something isn't working right with the software setup
        try:
            result = win32com.client.CastTo(to, name)
        except Exception as E:
            raise Exception(f'Teo software connection failed! CastTo({name}, {to})')
            #print(f'CastTo({name}, {to}) has failed')
            #print(E)
        if result is None:
            raise Exception(f'Teo software connection failed! CastTo({name}, {to})')
        return result


    def idn(self):
        # Get and print some information from the board
        DevName = self.DeviceID.GetDeviceName()
        DevRevMajor = self.DeviceID.GetDeviceMajorRevision()
        DevRevMinor = self.DeviceID.GetDeviceMinorRevision()
        DevSN = self.DeviceID.GetDeviceSerialNumber()
        return f'TEO: Name={DevName} SN={DevSN} Rev={DevRevMajor}.{DevRevMinor}'


    def print_function_names(self):
        '''
        Because there's no manual yet
        TODO: find out if we can discover the class names
        '''
        top_level_classes = ['DeviceID', 'DeviceControl', 'LF_Measurement', 'HF_Measurement']

        for tlc in top_level_classes:
            print(tlc)
            c = getattr(self, tlc)
            for node in dir(c):
                if not node.startswith('_'):
                    print(f'\t{node}')


    ##################################### HF mode #############################################

    def HF_mode(self, external=False):
        '''
        Call to turn on HF mode
        Teo said that currently nothing changes between internal and external mode!
        external:
        True for external mode (use SMA ports to external equipment)
        False to use the internal ADC
        '''
        # First argument (0) does nothing?
        self.HF_Measurement.SetHF_Mode(0, external)

    @staticmethod
    def hash_array(array):
        '''
        If you don't want to explicitly name the arrays, you can hash them and use that as the name
        then everything is content-addressable
        sha1 is ~fast and there's no security concern obviously
        time for a random float64 array of length 2^28 = 256M is about 2 seconds
        you should just name the arrays if you have really long ones.
        '''
        if len(array) > 2**26:
            log.warning('Consider manually defining names for long waveforms, '
                        'as hashing them can take a long time.')
        return hashlib.sha1(array).hexdigest()
        #return hashlib.md5(array).hexdigest()
        # There is also this?
        # hash(array.tostring())


    @staticmethod
    def interp_wfm(wfm, t):
        '''
        Interpolate waveform for the fixed 500 MHz sample rate
        return wfm compatible with Teo AWG

        wfm can be an array or a function of time
        e.g. lambda t: np.sin(ωt)

        if t is a number, assume it is the desired duration and assume equally spaced samples
        t may also be an increasing, arbitrarily spaced time array
        '''

        max_t = np.max(t)
        if max_t > 0.5: # <-- might not be precisely the limit!
            raise Exception('Waveform duration is too long for TEO memory.')

        # Teo compatible time array
        nsamples = int(round(500e6 * max_t))
        new_t = np.linspace(0, nsamples/500e6, nsamples)

        if isinstance(t, Number):
            if hasattr(wfm, '__call__'):
                # assume wfm is a function of t
                return wfm(new_t)
            else:
                t = np.linspace(0, t, len(wfm))

        return np.interp(new_t, t, wfm)


    # TODO: define some standard waveforms that use 500 Msample/second
    # e.g. pulse trains
    @staticmethod
    def sine(freq=1e5, amp=1):
        ''' One cycle of a sine wave '''
        nsamples = int(round(500e6/freq))
        x = np.linspace(0, 2*np.pi, nsamples)
        return amp * np.sin(x)

    @staticmethod
    def tri():
        pass

    @staticmethod
    def pulse_train(amps, pulsedur, timebetween):
        pass


    def gain(self, step=None):
        '''
        Unit is "steps" and each step corresponds to 6dB (2x)
        step: int 0-3
        if step is None, return the current step setting
        TODO: find out which current saturates the input for each gain step
              and document it here!
        '''
        # Note: Do not use HF_gain.Get/SetValue
        #       somehow this is a register that is split into two "step"
        #       values for HF and LF gain.
        #       these seem like details that should have been hidden from the API

        if step is None:
            return self.HF_Gain.GetStep()

        if step > 3:
            log.warning('Requested TEO gain is too high')
            step = 3
        if step < 0:
            log.warning('Requested TEO gain is too low')
            step = 0

        self.HF_Gain.SetStep(step)

    def _LFgain(self, LFgain=None):
        '''
        Apparently there is a gain setting for LF as well
        Need to hear from TEO how this is supposed to be used

        GetMinValue and GetMaxValue both return 0
        so I have a feeling this is not used
        '''
        return self.LF_Measurement.LF_Gain.GetValue()


    def _pad_wfms(self, varray, trig1, trig2):
        '''
        Make sure the number of samples in the waveform is compatible with the system
        pad with the standby offset value (usually zero volts)
        TODO: find out what the real limitation on sample size is
              TEO said it has to be a multiple of 2048
              and that his software should take care of it correctly
              but as of 2020-08-13, this is false
        '''
        lenv = len(varray)
        chunksize = 2**11
        npad = chunksize - len(varray) % chunksize
        if npad != 0:
            Vstandby = self.LF_Measurement.LF_Voltage.GetValue()
            # resolution is not below 1 mV, and LF_Voltage returns some strange numbers
            Vstandby = np.round(Vstandby, 3)
            varray = np.concatenate((varray, np.repeat(Vstandby, npad)))
            # Maybe the trigs should be padded with False istead?
            trig1 = np.concatenate((trig1, np.repeat(trig1[-1], npad)))
            trig2 = np.concatenate((trig2, np.repeat(trig2[-2], npad)))

        return varray, trig1, trig2


    def upload_wfm(self, varray, name=None, trig1=None, trig2=None):
        '''
        Add waveform and associated trigger arrays to TEO memory

        Fixed 500 MHz sample rate
        Min # of samples is 2¹¹ = 2048, max is supposed to be 2²⁸ = 268,435,456
        so durations between 4.096 μs and 536.8 ms
        # TODO test maximum size

        trig1 defines where we will get internal ADC readings on both channels (Vmonitor, current)

        Both triggers are synchronous digital signals accessible on controller board
        scope points TRIG1 and TRIG2.  Use for synchronizing external instruments.

        Waveforms remain in memory even if round board is turned off, but they are lost if
        controller board loses power

        if you reuse a waveform name, the previous waveform with that name gets overwritten

        TODO: what datatype are the triggers supposed to be?  bool?
              does it work if we try to upload ints?
              or do we need to be sure to cast to bool

        TODO: is there a limit to the NUMBER of waveforms that can be stored?
        TODO: is there a limit on the length of a waveforms name?
        '''
        if trig1 is None:
            trig1 = np.ones(len(varray), dtype=bool)

        if trig2 is None:
            trig2 = np.ones(len(varray), dtype=bool)

        if name is None:
            name = self.hash_array(varray)
            # TODO: could ask if this hash is already in memory, then don't bother to upload again
            #       but make sure the triggers are also hashed in that case
            #name = self.hash_array(np.concatenate((varray, trig1, trig2))


        wf = self.AWG_WaveformManager.CreateWaveform(name)

        varray, trig1, trig2 = self._pad_wfms(varray, trig1, trig2)

        wf.AddSamples(varray, trig1, trig2)

        # this is where you would also write all the information to the class instance
        # and hope that you don't ever get a memory overflow...
        # TODO: prevent memory overflow.
        self.waveforms[name] = (varray, trig1, trig2)


    def download_wfm(self, name):
        '''
        Read the waveform back from TEO memory
        TODO: determine if trigger arrays can also be read back
        '''
        wfm = self.AWG_WaveformManager.GetWaveform(name)
        # I don't know of a way to read back the trigger arrays
        #wfm.All_ADC_Gates() # tested and this is not trigger1
        #wfm.All_BER_Gates() # and this is not trigger2
        return np.array(wfm.AllSamples())


    def output_wfm(self, wfm, n=1, trig1=None, trig2=None):
        '''
        Output waveform by name or by values
        in internal mode, this automatically captures on both channels (where trig1 = True)

        wfm -- name of wfm or np.array of waveform voltages
        n   -- number of consecutive shots of the waveform to output

        careful if using automatic names, right now we only hash the wfm, not the triggers
        so if you try to change the triggers but not the wfm, they will not update
        '''
        if type(wfm) is str:
            name = wfm
        else:
            name = self.hash_array(wfm)

        # Returns False if there is no waveform with that name
        success = self.AWG_WaveformManager.Run(name, n)
        if not success:
            if type(wfm) in (np.ndarray, list, tuple):
                # Upload the waveform and try again
                self.upload_wfm(wfm, name=name, trig1=trig1, trig2=trig2)
                success = self.AWG_WaveformManager.Run(name, n)

        if success:
            self.last_waveform = name
            self.last_gain = self.gain()

        return success


    def get_data(self):
        '''
        Get the data for both ADC channels for the last capture
        returns a dict of information

        Gain setting is factored in already -- gain gets divided out

        # TODO: align data?  we want arrays of the same length, even if the triggers are not always on
        # TODO: return the programmed array?  but don't want to read it from TEO memory every time
        # TODO: what units does Vreturn come back with? something strange.
        # TODO: should we convert to current or not? needs the input impedance, which can change for some reason (self.J29)
        '''

        # We only get waveform samples where trigger is True, so these could be shorter than wfm
        # Vmonitor waveform
        wf00 = self.AWG_WaveformManager.GetLastResult(0)
        # Iout waveform
        wf01 = self.AWG_WaveformManager.GetLastResult(1)

        if wf00.IsSaturated():
            # I don't think this will ever happen
            print('TEO channel 0 is saturated!')
        if wf01.IsSaturated():
            print('TEO channel 1 is saturated!')

        Vmonitor = wf00.GetWaveformDataArray()
        Vreturn = wf01.GetWaveformDataArray()

        sample_rate = wf00.GetWaveformSamplingRate()

        # TODO somehow add the programmed waveform name/values and gain value that was used
        #      if the board has no provision for this, we will use values stored in the class instance

        # should always be there, but maybe you reset the instance state but wfm was still in teo memory..
        if self.last_waveform in self.waveforms:
            prog_wfm, trig1, trig2 = self.waveforms[self.last_waveform]
        else:
            prog_wfm = trig1 = trig2 = None

        gain = self.last_gain

        # TODO: time? sample rate is fixed, but not all samples are necessarily captured
        #       requires knowledge of the trigger waveforms, which we don't want to read from TEO memory
        #       because that would take a lot of time (and there might not be a way to do it)
        #       it would technically be enough to just store the inital value of trigger and the
        #       locations of the transitions, which would be a compression in many cases
        #
        #       we could also think about splitting the arrays where there are gaps in the capturing

        t = np.arange(len(Vmonitor))/sample_rate

        # TODO: for some reason we can get one or two 0s at the end of the measured waveforms
        #       this makes them longer than the programmed waveform, or longer than the number
        #       of Trues in trigger1

        # TODO: should we compress the trigger signals? they could be up to 64 MB each. do we need to output the triggers?

        # Very approximate conversion to current
        I = Vreturn * 1.988e-5

        return dict(V0=np.array(Vmonitor), V1=np.array(Vreturn), idn=self.constants.idn, sample_rate=sample_rate, t=t,
                    wfm=prog_wfm, gain=gain, I=I)


    def align_data(self, wfm, trig1, trig2, ch1, ch2):
        # Align measured waveforms with programmed waveform array
        # for the case where the triggers are not all True
        pass


    @staticmethod
    def _compress_trigger(trigarray):
        '''
        The trigger arrays are equal in length to the main waveform
        but probably have far fewer transitions (maybe just one or two!)
        convert the data into the transitions?

        this won't always compress
        "compressed" takes at least 32 bits per index
        but only 1 bit per raw waveform sample
        so a notable case of non-compression would be if the trigger is used to
        downsample with a factor less than 32
        in this case you would could use a different compression..

        we would also we need to store the first value (0 or 1)
        to know the sign of the transitions
        so this can't return a simple uint32 array
        could use a int32 and store the positive transitions (0 -> 1)
        as positive ints and negative transitions (1 -> 0) as negative ints.

        maybe we don't need the negative transitions
        since sampling should always start on the positive transitions,
        and we know the duration by the number of samples that come back
        '''
        # all transitions
        diff = np.diff(trigarray, prepend=False)
        # positive transitions
        p = diff & trigarray
        # negative transitions
        # n = diff & ~trigarray
        return np.where(p)[0]
        #return np.int32(np.where(p)[0])
        # -np.int32(np.where(n)[0])


    def get_wfm_names(self):
        wfm_names = []
        for i in itertools.count():
            name = self.AWG_WaveformManager.GetWaveformName(i)
            if name == '':
                break
            else:
                wfm_names.append(name)
        return wfm_names


    ##################################### LF mode #############################################

    # Source meter mode
    # I believe you just set voltages and read currents in a software defined sequence
    # TODO: is there a sequence mode and is there any advantage to using it?
    # TODO: what is the output impedance in LF mode?  hope it's still 50 so we don't risk blowing up 50 ohm inputs
    # TODO: will we blow up the input if we have more than 4 uA? how careful should we be?
    # TODO: is the current range bipolar?

    def LF_mode(self, external=False):
        '''
        Switch to LF mode
        (HF LED should turn off)
        '''
        self.LF_Measurement.SetLF_Mode(0, external)

    def LF_voltage(self, value=None):
        '''
        Gets or sets the LF source voltage value

        Also seems to be the idle level for HF mode..
        also seems to NOT change the voltage level in LF mode.. ?

        TODO: check that we are in the possible output voltage range
        TODO: rename?  LF_Voltage is the name of a class within LF_Measurement
        '''
        if value is None:
            return self.LF_Measurement.LF_Voltage.GetValue()
        else:
            self.LF_Measurement.LF_Voltage.SetValue(value)

    def LF_current(self, NPLC=10):
        '''
        Reads the LF current

        specs say ADC is 24 bits, 4 uA range, 1pA resolution
        but in practice the noise is much higher than 1pA

        Not sure if this is the right way to do it!

        # TODO Verify below
        I THINK:
        the ADC has a sample rate of something like 31,248 Hz, but the buffer size is 8,000
        LF_MeasureCurrent(duration) returns all the samples it could store for that duration
        you can then average them.
        but it means we can't average longer than 256 ms in a single shot
        which is 12.8 PLCS for 50 Hz line freq

        '''
        if NPLC > .256 * self.PLF:
            print(f'I THINK the LFOutput buffer is too small for NPLC={NPLC}')

        duration = NPLC / self.PLF
        Iwfm = self.LF_Measurement.LF_MeasureCurrent(duration)
        if Iwfm.IsSaturated():
            print('TEO LFOutput is saturated!')
        I = Iwfm.GetWaveformDataArray()
        return np.mean(I)



    ##################################### Tests #############################################

    def pulse_and_capture(self, wfm):
        self.HF_mode(external=False)
        self.output_wfm(wfm)
        return self.get_data()

    def measure_leakage(self, Vvalues, avgtime):
        '''
        doesn't work, because LF_voltage doesn't set the voltage in LF mode..
        '''
        self.LF_mode(external=False)

        Vidle = self.LF_voltage()

        I = []
        for V in Vvalues:
            self.LF_voltage(V)
            time.sleep(.1)
            i = self.LF_current(avgtime)
            I.append(i)

        I = np.array(I)

        # Go back to the idle level
        self.LF_voltage(Vidle)

        return dict(I=I, V=Vvalues)
