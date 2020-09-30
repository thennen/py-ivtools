import numpy as np
import itertools
import sys
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

    The main idea of this system is to combine a variety of measurement devices used for
    modern memory or sensor routine test processes, yet leave the ability to connect
    external equipment if needed.

    There are two "modes", HF and LF

    HF mode:
        There is one DAC channel (AWG), whose output appears on the HFV port
        256 MSamples can be stored in AWG memory
        ±10V output voltage amplitude, with 14 bit resolution (~5 mV), 50 ohm output
        rise/fall time < 1 ns
        software can recall uploaded waveforms by name, to save data transfer time
        There are two ADCs, one monitors the AWG voltage, and the other measures the voltage on the HFI port
        ADCs have 12 bit resolution
        ADC memory is 256 MSamples per channel
        Fixed 500 MSamples/s sample rate for internal AWG and ADCs
        There are several amplifiers on board
        HFV monitor has jumper selectable gain, normally set to read full scale ±10V
        HFI has two amplifiers whose gain can be controlled by software
        1. 4 GHz bandwidth amplifier which goes to the HF FULL BW port
        2. 200 MHz bandwidth, higher gain, output is digitized by ADC channel and also goes to HF LIMITED BW port

    LF mode:
        TODO: elaborate this section
        There is one DAC channel, one ADC channel

    Designed for minimum DUT resistance 1kΩ, but shouldn't break easily if there is a short circuit

    We communicate with Teo software using the windows COM interface.
    The provided COM classes are named like this:
    DriverID
    DeviceID
    DeviceControl
    LF_Measurement
    HF_Measurement
    HF_Gain
    AWG_WaveformManager

    In this class, we define some higher level functions that appeal to our "pythonic" sensibilities.

    Bring some of the most used methods that are inconveniently deep
    in the object hierarchy to the top level and gives them shorter names

    All of TEOs functions are CapitalCamelCased, sometimes with underscores.
    Most of the functions I added start with a lowercase letter.

    Adds content-addressability of waveforms for effortless replay, without manual naming

    Seems to handle re-initialization just fine.
    You can make multiple instances and they will all work

    # TODO: does the class need to store any internal state?  should we make it BORG?

    # TODO: write more high level methods (e.g. pulse_and_capture..), but nothing application specific

    # TODO: do all the commands work regardless of which mode we are in? e.g. waveform upload, gain setting
            how do we avoid issuing commands and expecting it to do something but we are in the wrong mode?

    # TODO: what happens when we send commands while the board is busy?

    # TODO: could wrap all of the functions only for the purpose of giving them signatures, docstrings,
            default arguments.

    # TODO: should we hide the entire COM interface in a container?  like
            self.com.DeviceControl, self.com.LF_Measurement etc
            then on the top level we have mostly stuff that we have defined that has docstrings and
            so on, but we can still access the com interface if needed.
            downside is that when we ask for support our code will be unrecognizable to Teo.
            therefore:

    # TODO: should we have some kind of a debug mode that prints out all the COM calls?
            then we can send those to Teo for help

    # TODO: since there seem to be a several of situations that cause the output to go to the negative rail
            and blow up your device, make sure to document them here
            seems to be whenever TSX_DM.exe connects to the system
            1. on first initialization (Dispatch('TSX_HMan'))
            2. if you disconnect USB and plug it back in

    # TODO: store a calibration to remove the offsets and scale voltages.
            monitor has a big offset and for the other channel it depends a bit on the gain setting.
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
            ## Launch programs for software interface to TEO board
            # Starts two processes:
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
            # TODO is this necessarily the meaning of this error?
            #raise type(e)(str(e) +
            #              ' TEO software not installed?').with_traceback(sys.exc_info()[2])
            log.error('Teo software not installed?')
            return

        # Asks the program for a device called MEMORY_TESTER
        MemTester = HMan.GetSystem('MEMORY_TESTER')
        if MemTester is None:
            log.error('Teo software cannot locate a connected memory tester. Check USB connection.')
            return

        # Access a bunch of COM classes used to control the TEO board.
        # The contained methods/attributes appear in tab completion, but the contained classes do not
        DriverID =            TeoSystem._CastTo('ITS_DriverIdentity'     , MemTester)
        DeviceID =            TeoSystem._CastTo('ITS_DeviceIdentity'     , DriverID)
        DeviceControl =       TeoSystem._CastTo('ITS_DeviceControl'      , DriverID)
        LF_Measurement =      TeoSystem._CastTo('ITS_LF_Measurement'     , DriverID)
        #LF_Voltage =          LF_Measurement.LF_Voltage # ?
        HF_Measurement =      TeoSystem._CastTo('ITS_HF_Measurement'     , DriverID)
        # Is this different from HF_Gain = HF_Measurement.HF_Gain?
        # TODO: why can't we see e.g. HF_Measurement.HF_Gain in tab completion?
        HF_Gain =             TeoSystem._CastTo('ITS_DAC_Control'        , HF_Measurement.HF_Gain)
        AWG_WaveformManager = TeoSystem._CastTo('ITS_AWG_WaveformManager', HF_Measurement.WaveformManager)

        # Assign com methods/attributes to the instance
        # TODO: store these in a container (dotdict), since we are writing a higher level wrapper?
        #       that would tidy things up a bit
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
        #       We aren't used to these really long nested function calls in python
        self.GetFreeMemory = AWG_WaveformManager.GetFreeMemory
        self.StopDevice = DeviceControl.StopDevice

        # TODO: assign properties that do not change, like max/min values
        #       so that we don't keep polling the instrument for fixed values

        # Store them in this dumb container so they don't clutter everything
        class dotdict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        self.constants = dotdict()
        self.constants.idn = self.idn()
        self.constants.max_LF_Voltage = self.LF_Measurement.LF_Voltage.GetMaxValue()
        self.constants.min_LF_Voltage = self.LF_Measurement.LF_Voltage.GetMinValue()
        #self.constants.max_HFgain = self.HF_Gain.GetMaxValue() # 10
        #self.constants.min_HFgain = self.HF_Gain.GetMinValue() # -8
        #self.constants.max_LFgain = self.LF_Measurement.LF_Gain.GetMaxValue() # 0?
        #self.constants.min_LFgain = self.LF_Measurement.LF_Gain.GetMinValue() # also 0?
        self.constants.AWG_memory = self.AWG_WaveformManager.GetTotalMemory()

        # if you have the jumper, HFI impedance is 50 ohm, otherwise 100 ohm
        self.J29 = True

        # TODO: Do we need a setting for the LF internal/external jumpers? Probably not.

        # set the power line frequency for averaging over integer cycles
        self.PLF = 50

        ## Teo says this powers up round board, but the LEDS are already on by the time we call it.
        # it appears to be fine to call it multiple times;
        # everything still works, and I didn't see any disturbances on the HFV output
        # TODO: what state do we exactly start up in the first time this is called?
        # seems the HF mode LED is on, but I don't see the internal pulses on HFV,
        # So I think it starts in some undefined mode
        # subsequent calls seem to stay in whatever mode it was in before,
        # even if we lost the python-TSX_DM connection for some reason
        DeviceControl.StartDevice()
        # This command sets the idle level for HF mode
        LF_Measurement.LF_Voltage.SetValue(0)
        #self.HF_mode()

        # Store the same waveform/trigger data that gets uploaded to the board/TSX_DM process
        # TODO: somehow prevent this from taking too much memory
        #       should always reflect the state of the teo board
        #self.waveforms = {}
        self.waveforms = self.download_all_wfms()
        # Store the name of the last played waveform
        self.last_waveform = None
        self.last_gain = None
        self.last_nshots = None

        log.info('TEO connection successful: ' + self.constants.idn)

    ###### Direct wrappers for adding python function signatures and docstrings ####

    def StopDevice(self):
        '''
        Lights should turn off on the round board and HFV output probably floats.
        Controller board remains on.
        '''
        self.DeviceControl.StopDevice()

    # TODO add more wrappers with docstrings
    #      I understand that win32com actually generates the python wrapper code.
    #      might be interesting to look at it, maybe just modify that


    ################################################################################

    @staticmethod
    def _CastTo(name, to):
        # CastTo that clearly lets you know something isn't working right with the software setup
        try:
            result = win32com.client.CastTo(to, name)
        except Exception as E:
            log.error(f'Teo software connection failed! CastTo({name}, {to})')
        if result is None:
            log.error(f'Teo software connection failed! CastTo({name}, {to})')
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

    def HF_mode(self):
        '''
        Call to turn on HF mode

        Previous revision had internal/external mode for switching between internal and external AWG
        this revision has HF AWG input SMA (J8), but it is controlled manually by switch SW1!
        following revisions remove AWG input entirely!

        External scope outputs are always active (no external mode needed)
        '''
        # First argument (0) does nothing?
        # So does second argument apparently
        external = False
        self.HF_Measurement.SetHF_Mode(0, external)

    @staticmethod
    def _hash_arrays(wfm, trig1, trig2):
        '''
        If you don't want to explicitly name the arrays, you can hash them and use that as the name
        then everything is content-addressable
        sha1 is ~fast and there's no security concern obviously
        time for a random float64 array of length 2^28 = 256M is about 2 seconds
        you should just name the arrays if you have really long ones.

        TODO: do a more efficient kind of hash, since 2/3 of the data is just boolean
        '''
        array = np.concatenate((wfm, trig1, trig2))
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
        if max_t > 0.512: # <-- might not be precisely the limit!
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
        This simultaneously sets the gain for two current amplifiers
        Amp1: High BW output
        Amp2: Low BW output, also what you read on second ADC channel

        The system shares a 5 bit register to set both gains
        the two amplifiers both read and interpret this register in different ways.

        Amp1 uses all of the bits, for possible step settings 0-31
        each step corresponds to 1dB (1.122×)
        step = 0 is the highest gain setting

        Amp2 uses the two LSB, for possible step settings 0,1,2,3
        each step corresponds to 6dB (2×)
        step = 0 is the highest gain setting

        This would mean that if you want to set the gain of Amp2, the gain
        of Amp1 changes with it and in a not monotonic way (whatever the 2 LSBs say)
        important to consider this if you want to use both outputs at the same time
        then you would have to set MSB and LSB independently

        if step is None, return the current step setting

        TODO: find out which current saturates the input for each gain step
              and document it here!

        TODO: abstract away this "steps" stuff

        TODO: make two separate gain functions for the two amps, which only modify MSB or LSB
        '''
        # Note: Do not use HF_gain.Get/SetValue
        #       somehow this is a shared register that is split into two gain settings
        #       these seem like details that should have been hidden from the API

        if step is None:
            return self.HF_Gain.GetStep()

        if step > 31:
            log.warning('Requested TEO gain step is too high')
            step = 31
        if step < 0:
            log.warning('Requested TEO gain step is too low')
            step = 0

        self.HF_Gain.SetStep(step)


    def _pad_wfms(self, varray, trig1, trig2):
        '''
        Make sure the number of samples in the waveform is compatible with the system
        pad with the standby offset value (usually zero volts)

        Teo said it has to be a multiple of 2048
        and that his software should take care of it correctly
        but as of 2020-08, this appears to be false

        if you don't play a correct chunk size, expect very wrong readback if n < 1024
        and a random number of extra/missing samples for other lengths
        '''
        lenv = len(varray)
        chunksize = 2**11
        remainder = lenv % chunksize
        if remainder != 0:
            npad = chunksize - remainder
            Vstandby = self.LF_Measurement.LF_Voltage.GetValue()
            # resolution is not below 1 mV, and LF_Voltage returns some strange numbers
            Vstandby = np.round(Vstandby, 3)
            varray = np.concatenate((varray, np.repeat(Vstandby, npad)))
            # Maybe the trigs should be padded with False instead?
            trig1 = np.concatenate((trig1, np.repeat(trig1[-1], npad)))
            trig2 = np.concatenate((trig2, np.repeat(trig2[-1], npad)))

        return varray, trig1, trig2


    def upload_wfm(self, varray, name=None, trig1=None, trig2=None):
        '''
        Add waveform and associated trigger arrays to TEO memory

        I think it just transfers the arrays to the TSX_DM process
        and the hardware only gets it when we try to play the waveform.
        it still takes some time to transfer large arrays to TSX_DM

        Fixed 500 MHz sample rate
        Min # of samples is 2¹¹ = 2048, max is supposed to be 2²⁸ = 268,435,456
        so sample durations are possible between 4.096 μs and 536.8 ms

        trig1 defines where we will get internal ADC readings on both channels (Vmonitor, current)

        Both triggers are synchronous digital signals accessible on controller board
        MCX ports TRIG1 and TRIG2.  Use for synchronizing external instruments.

        Waveforms remain in memory even if round board is turned off, but they are lost if
        controller board loses power

        if you reuse a waveform name, the previous waveform with that name gets overwritten

        Because the triggers are generated by FPGA which operates at only 250 MHz, there is a
        trigger jitter of one sample. The triggers get downsampled by TSX_DM to its 250 MHz equivalent.
        We will handle the conversion here so everything stays consistent.

        Note that lots of transitions in the trigger was not really designed for, so be careful if you try it

        TODO: for long waveforms, are there big delays when we use
              AWG_WaveformManager.Run() on a newly define waveform?

        TODO test maximum size
             seem to get System.OutOfMemoryException if we use more than 2^23

        TODO Benchmark how long it takes to transfer these to and from TSX_DM
             is it slow enough to warrant keeping a copy here in python?  (self.waveforms)

        TODO: is there a limit to the NUMBER of waveforms that can be stored?

        TODO: is there a limit on the length of a waveforms name?
        '''
        n = len(varray)

        if not type(varray) == np.ndarray:
            varray = np.array(varray)

        if trig1 is None:
            trig1 = np.ones(n, dtype=bool)
        else:
            # Convert for the 250 MHz FPGA
            trig1 = np.repeat(np.array(trig1[::2], bool), 2)[:n]

        if trig2 is None:
            trig2 = np.ones(n, dtype=bool)
        else:
            # Convert for the 250 MHz FPGA
            trig2 = np.repeat(np.array(trig2[::2], bool), 2)[:n]

        loaded_names = self.get_wfm_names()

        if name is None:
            name = self._hash_arrays(varray, trig1, trig2)
            # Ask if this hash is already in memory, if so then don't bother to upload again
            # then we not only remember waveforms by name but also by value
            if name in loaded_names:
                log.debug('Waveform already in memory (hash match)')
                return name

        if name in loaded_names:
            log.debug(f'Overwriting waveform named {name}')

        wf = self.AWG_WaveformManager.CreateWaveform(name)
        varray, trig1, trig2 = self._pad_wfms(varray, trig1, trig2)
        wf.AddSamples(varray, trig1, trig2)

        # also write all the waveform data to the class instance
        # hope that you don't ever get a memory overflow...
        # TODO: prevent memory overflow, or transfer from TSX_DM every time if it's not too slow
        self.waveforms[name] = (varray, trig1, trig2)

        return name


    def download_wfm(self, name):
        '''
        Read the waveform back from TEO memory
        return (varray, trig1, trig2)
        seems not to come from hardware memory, just the TSX_DM process working set
        '''
        wfm = self.AWG_WaveformManager.GetWaveform(name)
        v = np.array(wfm.AllSamples())
        trig1 = np.array(wfm.All_ADC_Gates())
        trig2 = np.array(wfm.All_BER_Gates())
        return (v, trig1, trig2)


    def download_all_wfms(self):
        '''
        With this we can resynchronize our instance with the TSX_DM memory
        e.g. self.waveforms = self.download_all_wfms()
        '''
        return {name:self.download_wfm(name) for name in self.get_wfm_names()}


    def output_wfm(self, wfm, n=1, trig1=None, trig2=None):
        '''
        Output waveform by name or by values
        This automatically captures on both ADC channels (where trig1 = True)
        you need to call get_data() afterward for the result

        wfm -- name of wfm or np.array of waveform voltages
        n   -- number of consecutive shots of the waveform to output
        '''
        if type(wfm) is str:
            name = wfm
            success = self.AWG_WaveformManager.Run(name, n)
            if not success:
                log.error('No waveform with that name has been uploaded')
        elif type(wfm) in (np.ndarray, list, tuple):
            # this will hash the data to make a name
            # won't upload again if the hash matches
            name = self.upload_wfm(wfm, trig1=trig1, trig2=trig2)
            success = self.AWG_WaveformManager.Run(name, n)

        if success:
            self.last_waveform = name
            self.last_gain = self.gain()
            self.last_nshots = n
        else:
            log.error('TEO waveform output failed')

        return success


    def get_data(self, raw=False):
        '''
        Get the data for both ADC channels for the last capture.
        Returns a dict of information.

        We return the programmed waveform data as well,
        which is useful because the monitor signal can have a lot of noise

        if raw is True, then keys 'HFV' and 'HFI' will be in the returned dict
        which are the ADC values before calibration/conversion/trimming

        TODO: Calibrate voltages and currents
              Store the calibrations here in the class definition
        TODO: How should we align data with trigger?  we want to return arrays of the same length, even if the
              triggers are not always on. We have two options:
              1. delete programmed voltage waveform and time waveform where trig1 is False
              2. pad measured waveform with np.nan where trig1 is False
              we could also think about slicing the arrays where there are gaps in the capturing
              then this would return a list of dicts
              currently we do 2. but there are some potential problems with extra samples
        '''
        # We only get waveform samples where trigger is True, so these could be shorter than wfm
        # V monitor waveform (HFV)
        wf00 = self.AWG_WaveformManager.GetLastResult(0)
        # Current waveform (HFI)
        wf01 = self.AWG_WaveformManager.GetLastResult(1)

        if wf00.IsSaturated():
            # I don't think this will ever happen.
            # but there is some manual way to increase the gain on this channel such that it could happen
            log.warning('TEO ADC channel 0 (HFV monitor) is saturated!')
        if wf01.IsSaturated():
            log.warning('TEO ADC channel 1 (HFI) is saturated!')

        # Signals on the ports
        # Gain is divided out already before it is read in here
        # they are not in volts and they need calibration
        HFV = np.array(wf00.GetWaveformDataArray())
        HFI = np.array(wf01.GetWaveformDataArray())

        R_HFI = 50 if self.J29 else 100

        # Very approximate conversion to current
        # TODO: calibrate this better
        I = HFI * 99.4e-5 / R_HFI
        # TODO: also calibrate this, this is just a guess
        V = (HFV + 320e-3) / 2.047

        sample_rate = wf00.GetWaveformSamplingRate() # Always 500 MHz

        # last_waveform should always be in self.waveforms, but maybe not if you reset the
        # instance state but wfm was still loaded in teo memory..
        # if we go BORG this will be less likely to happen
        if self.last_waveform in self.waveforms:
            # Refer to the waveform dict stored in the instance
            prog_wfm, trig1, trig2 = self.waveforms[self.last_waveform]
        else:
            log.debug('Waveform data was missing from TeoSystem instance')
            # Try to read it from TSX_DM.exe
            prog_wfm, trig1, trig2 = self.download_wfm(self.last_waveform)

        nshots = self.last_nshots

        if all(trig1):
            # According to testing, these waveforms always come back with two zeros appended
            # (as long as we pad the waveforms so the length is a multiple of 2048, which we do)
            if not all(HFV[-2:] == 0) or HFV[-3] == 0:
                log.warning('The TEO ADC values did not have exactly two zeros appended! Fix the code!')
            I = I[:-2]
            V = V[:-2]
        else:
            # Here the situation is more complicated and buggy
            # waveforms are read back with a random number in [0,3] zeros at the end
            # and the length differs from number of True values in trig1 by a random number in [-1, 2]
            # these two random numbers are uncorrelated..
            # We just have to make the waveform fit and hope these few samples never matter..
            extra_samples = len(V) - sum(trig1) * nshots
            log.debug(f'{extra_samples} extra samples')
            if extra_samples > 0:
                V = V[:-extra_samples]
                I = I[:-extra_samples]
            elif extra_samples < 0:
                # Just put in extra zeros since that seems to be this instruments style
                V = np.append(V, [0]*-extra_samples)
                I = np.append(I, [0]*-extra_samples)

        # TODO: return time array? sample rate is fixed, but not all samples are necessarily captured
        #       it would technically be enough to just store the inital value of trigger and the
        #       locations of the transitions, which would be a compression in many cases
        t = np.arange(len(prog_wfm) * nshots)/sample_rate

        # This may become a memory problem
        # e.g. if we use a large number of shots with large waveforms and sparse triggers
        wfm = np.tile(prog_wfm, nshots)
        trig1 = np.tile(trig1, nshots)
        # trig2 = np.tile(trig2, nshots) # we do nothing with trig2 at the moment

        # Align measurement waveforms with the programmed waveform (for the case that not all(trig1))
        # TODO: the extra samples might come at the end of every chunk, every shot, random locations,
        #    we don't know yet.  the following assumes that they are all at the end!
        I, V = self._align_with_trigger(trig1, I, V)
        # Alternatively we could cut the programmed wfm to match trig1
        # This would use less memory, but the time array will reflect the gap in data acquisition
        # prog_wfm = prog_wfm[trig1]
        # t = t[trig1]

        # TODO: should we compress the trigger signals and return them?
        #       otherwise they could be up to 64 MB per shot.
        #       if we use _align_with_trigger, the information for trig1 is already there

        # TODO: This could return a really large data structure.
        #       we might need options to return something more minimal if we want to use long waveforms
        #       could also convert to float32, then I would use a dtype argument: get_data(..., dtype=float32)
        out = dict(V=V, I=I, t=t, wfm=wfm,
                   idn=self.constants.idn,
                   sample_rate=sample_rate,
                   gain_step=self.last_gain,
                   nshots=self.last_nshots)
        if raw:
            out['HFV'] = HFV
            out['HFI'] = HFI
        return out


    @staticmethod
    def _align_with_trigger(trig1, *arrays):
        '''
        Align measured waveforms with the programmed waveform array
        for the case where the triggers are not all True
        sum(trig1) must be equal to len(array) for array in arrays

        all output arrays will have the same length as trigger.

        '''
        arrays_out = []
        for arr in arrays:
            arr_out = np.empty(len(trig1), dtype=type(arr[0]))
            arr_out[trig1] = arr
            arr_out[~trig1] = np.nan
            arrays_out.append(arr_out)
        return arrays_out


    @staticmethod
    def _compress_trigger(trigarray):
        '''
        Not used at the moment

        The trigger arrays are equal in length to the main waveform
        but probably have far fewer transitions (maybe just one or two!)
        convert the data into the transitions?

        this won't always compress
        the "compressed" array takes at least 32 bits per index
        but uncompressed takes only 1 bit per raw waveform sample
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

    def delete_all_wfms(self):
        for name in self.get_wfm_names():
            self.AWG_WaveformManager.DeleteWaveform(name)


    ##################################### LF mode #############################################

    # Source meter mode
    # I believe you just set voltages and read currents in a software defined sequence
    # TODO: is there a sequence mode and is there any advantage to using it?
    # TODO: what is the output impedance in LF mode?  hope it's still 50 so we don't risk blowing up 50 ohm inputs
    # TODO: will we blow up the input if we have more than 4 uA? how careful should we be?
    # TODO: is the current range bipolar?

    def LF_mode(self):
        '''
        Switch to LF mode (HF LED should turn off)

        The voltage output goes to port HFI, not HFV for some reason
        HFV goes to zero. So the polarity is reversed wrt HF mode.

        There's no software controllable internal/external mode anymore.
        for internal mode, put jumpers J4 and J5 to position 2-3
        for external mode, put jumpers J4 and J5 to position 1-2

        J4 and J5 are located underneath the square metal RF shield,
        you can pry off the top cover to access them.
        '''
        external = False
        self.LF_Measurement.SetLF_Mode(0, external)

    def LF_voltage(self, value=None):
        '''
        Gets or sets the LF source voltage value

        In LF mode, the voltage appears on the HFI port, not HFV, which is the reverse of HF mode.

        This also sets the idle level for HF mode..  but doesn't switch to LF_mode or anything

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
        but in practice the noise may be much higher than 1pA

        the ADC has a sample rate of something like 31,248 Hz, but the buffer size is 8,000

        LF_MeasureCurrent(duration) returns all the samples it could store for that duration
        you can then average them.

        but it means we can't average longer than 256 ms in a single shot
        which is 12.8 PLCS for 50 Hz line freq

        TODO: calibrate this with external source meter and store the calibration values in the class
        '''
        if NPLC > .256 * self.PLF:
            log.warning(f'I THINK the LFOutput buffer is too small for NPLC={NPLC}')

        duration = NPLC / self.PLF
        Iwfm = self.LF_Measurement.LF_MeasureCurrent(duration)
        if Iwfm.IsSaturated():
            log.warning('TEO LFOutput is saturated!')
        I = Iwfm.GetWaveformDataArray()
        return np.mean(I)


    ##################################### Tests #############################################

    def measure(self, wfm):
        '''
        Pulse wfm and return I,V,... data
        '''
        self.HF_mode()
        self.output_wfm(wfm)
        return self.get_data()


    def measure_leakage(self, Vvalues, NPLC=10):
        '''
        Use internal LF mode to make a low-current measurement
        not tested
        TODO: test this!
        '''
        self.LF_mode()

        Vidle = self.LF_voltage()

        I = []
        for V in Vvalues:
            self.LF_voltage(V)
            time.sleep(.1)
            i = self.LF_current(NPLC)
            I.append(i)

        I = np.array(I)

        # Go back to the idle level
        self.LF_voltage(Vidle)

        return dict(I=I, V=Vvalues)
