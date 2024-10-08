import hashlib
import itertools
import logging
import os
import time
from inspect import signature
from numbers import Number
import shutil

import numpy as np
import pandas as pd

import ivtools  # for calibration file path

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
        This is a high resolution, low speed mode
        can be "internal" or "external" as controlled by jumpers J4 and J5
        If external, turning on LF just functions as a switch for an external SMU connected to LFV and LFI ports
        If internal, an on-board ADC is used instead, 24 bits, 4 uA range, 1pA resolution, but too much noise
        Sample rate 31,248 Hz, buffer size 8,000

    Designed for minimum DUT resistance 1kΩ, but shouldn't break easily if there is a short circuit

    We communicate with Teo software using the windows COM interface.
    The provided COM classes are named like this:
    DriverID
    DeviceID
    DeviceControl
    LF_Measurement
    LF_Voltage
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

    Saturation values in volts:
        V MONITOR: -1 ; 1

    TODO: We can think about a way to correct the time delay between voltage and current signals. Right now if you
     try to plot an I,V loop for a <1 μs signal you get a circle. This is because it takes time for the signal to go
     through the sample and into the HFI port, depending on how long the cables are.

    TODO: Gain goes back to 0 everytime we do teo = instruments.TeoSystem(), this is a problem if you are using that
     line inside your code.

    TODO: do all the commands work regardless of which mode we are in? e.g. waveform upload, gain setting how do we
     avoid issuing commands and expecting it to do something but we are in the wrong mode? need to check something
     like isin(HF_mode) ?

    TODO: since there seem to be several situations that cause the HFV output to go to the negative rail and blow up
     your device, make sure to document them here seems to be whenever TSX_DM.exe connects to the system
          1. on first initialization (Dispatch('TSX_HMan'))
          2. if you disconnect USB and plug it back in

    '''

    def __init__(self):


        '''
        This will do software/hardware initialization and set HFV output voltage to zero
        requires TEO software package and drivers to be installed on the PC

        Reasons for TeoSystem to be BORG
        - Download waveforms from TeoSystem on can take too long, so it' s annoying to download them at every
        connection.
        - Gain step is set to 0 at every connection.
        - Some variables like 'last_wfm' are lost.

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Make sure there is no DUT connected when you initialize!
        HFV output goes to the negative rail!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''

        statename = self.__class__.__name__
        if statename not in ivtools.instrument_states:
            ivtools.instrument_states[statename] = {}
        self.__dict__ = ivtools.instrument_states[statename]

        if not self.connected():
            self.connect()

    ################################################################################

    def connect(self):
        class dotdict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        self.constants = dotdict()

        self.J29 = True  # if you have the J29 jumper, HFI impedance is 50 ohm, otherwise 100 ohm
        self.PLF = 50  # set the power line frequency for averaging over integer cycles
        self.freq = 500e6  # Teo frequency
        self.lbw_Vsat0 = 3.6    # V  # Saturation level for HF_LIMITED_BW
        self.hbw_Vsat0 = 1.55   # V  # Saturation level for HF_FULL_BW
        self.int_Vsat0 = 0.016  # V  # Saturation level for internal scope

        # Store the same waveform/trigger data that gets uploaded to the board/TSX_DM process
        # TODO: somehow prevent this from taking too much memory
        #       should always reflect the state of the teo board
        self.waveforms = {}  # Store the waveforms used so we they are always in memory.
        # could happen that there are waveforms in Teo memory and not here, but is better to get them when you really
        # need them, synchronizing the whole memory can take minutes.
        self.last_waveform = None  # Store the name of the last played waveform
        self.last_gain = None
        self.last_nshots = None

        if os.path.isfile(ivtools.settings.teo_calibration_file):
            self.calibration = pd.read_pickle(ivtools.settings.teo_calibration_file)
        else:
            log.warning('Calibration file not found!')
            self.calibration = None

        self.base = TeoBase()

        if self.base.conn:
            self.memoryleft = self.base.AWG_WaveformManager.GetFreeMemory()

            # Assign properties that do not change, like max/min values
            # so that we don't keep polling the instrument for fixed values
            self.constants.idn = self.idn()
            self.constants.maxLFVoltage = self.base.LF_Voltage.GetMaxValue()
            self.constants.minLFVoltage = self.base.LF_Voltage.GetMinValue()
            self.constants.AWG_memory = self.base.AWG_WaveformManager.GetTotalMemory()

            if os.path.isfile(ivtools.settings.teo_calibration_file):
                self.calibration = pd.read_pickle(ivtools.settings.teo_calibration_file)
            else:
                log.warning('Calibration file not found!')
                self.calibration = None

            ## Teo says this powers up round board, but the LEDS are already on by the time we call it.
            # it appears to be fine to call it multiple times;
            # everything still works, and I didn't see any disturbances on the HFV output
            # TODO: what state do we exactly start up in the first time this is called?
            # seems the HF mode LED is on, but I don't see the internal pulses on HFV,
            # So I think it starts in some undefined mode
            # subsequent calls seem to stay in whatever mode it was in before,
            # even if we lost the python-TSX_DM connection for some reason
            self.base.DeviceControl.StartDevice()
            # This command sets the idle level for HF mode
            self.base.LF_Voltage.SetValue(0)

            self.conn = True
            log.info('TEO connection successful: ' + self.constants.idn)
        else:
            self.memoryleft = None
            self.constants.idn = None
            self.constants.maxLFVoltage = None
            self.constants.minLFVoltage = None
            self.constants.AWG_memory = None
            self.conn = False

    def connected(self):
        if not hasattr(self, 'conn'):
            self.conn = False
        elif self.conn is True:
            MemTester = self.base.HMan.GetSystem('MEMORY_TESTER')
            if MemTester is None:
                self.conn = False
            else:
                self.conn = True

        return self.conn

    def idn(self):
        # Get and print some information from the board
        DevName = self.base.DeviceID.GetDeviceName()
        DevRevMajor = self.base.DeviceID.GetDeviceMajorRevision()
        DevRevMinor = self.base.DeviceID.GetDeviceMinorRevision()
        DevSN = self.base.DeviceID.GetDeviceSerialNumber()
        return f'TEO: Name={DevName} SN={DevSN} Rev={DevRevMajor}.{DevRevMinor}'

    def kill_TSX(self):
        # /F tells taskkill we aren't Fing around here
        os.system("taskkill /F /im TSX_HardwareManager.exe")
        os.system("taskkill /F /im TSX_DM.exe")

    def restart_TSX(self):
        self.kill_TSX()
        self.__init__()

    def close(self):
        self.base.DeviceControl.StopDevice()
        self.kill_TSX()


    ##################################### HF mode #############################################

    def HF_mode(self, external=True):
        '''
        Call to turn on HF mode

        Previous revision had internal/external mode for switching between internal and external AWG
        this revision has HF AWG input SMA (J8), but it is controlled manually by switch SW1!
        following revisions remove AWG input entirely!

        External scope outputs are always active (no external mode needed)

        If external is True, setting voltages is disabled
        '''
        # First argument (0) does nothing?
        # So does second argument apparently
        self.base.HF_Measurement.SetHF_Mode(0, external)

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

    # Some standard waveforms
    @staticmethod
    def sine(freq=1e5, amp=1, offs=0, cycles=1):
        '''
        Basic sine wave
        can modulate amplitude if len(amp) == cycles
        '''
        nsamples = int(round(500e6*cycles/freq))
        x = np.linspace(0, 2*cycles*np.pi, nsamples)
        if hasattr(amp, '__iter__') and (len(amp) == cycles):
            icycle = x // (2*np.pi + 1e-15)
            amp = np.repeat(amp, [sum(icycle == i) for i in range(cycles)])
        return amp * np.sin(x) + offs

    @staticmethod
    def tri(V, sweep_rate=1e6):
        '''
        take a list of voltages, and sweep to all of them at a fixed sweep rate

        # do we want it to start and end at zero?
        '''

        teo_freq = 500e6
        sweep_samples = int(teo_freq / sweep_rate)
        if len(V) <= 1:
            raise Exception("'V' must be a list with at least two values")
        sweeps = [[V[0]]]

        for v in V[1:]:
            last = sweeps[-1][-1]
            sweep_samples = int(abs(v-last)*teo_freq/sweep_rate)
            sweep = np.linspace(last, v, sweep_samples+1)[1:]
            sweeps.append(sweep)

        wfm = np.concatenate(sweeps)

        return wfm

    @staticmethod
    def pulse_train(amps, durs=1e-6, delays=1e-6, first_delay=None, zero_val=0, n=1):
        '''
        Create a rectangular pulse train

        delays come after each pulse

        amps, durs, delays may be vectors or scalars

        if vector lengths are not the same, they must at least be divisors of the maximum length
        then they will be tiled to match the maximum length.
        '''
        teo_freq = 500e6

        # Make everything the same length
        length = lambda x: 1 if isinstance(x, Number) else len(x)
        lengths = map(length, (amps, durs, delays))
        npulses = max(lengths)
        amps = np.tile(amps, int(npulses/length(amps)))
        durs = np.tile(durs, int(npulses/length(durs)))
        delays = np.tile(delays, int(npulses/length(delays)))

        if first_delay is None:
            first_delay = delays[0]

        wfm = [np.repeat(zero_val, int(round(teo_freq*first_delay)))]
        for amp, dur, delay in zip(amps, durs, delays):
            amp_samples = int(round(teo_freq*dur))
            delay_samples = int(round(teo_freq*delay))
            wfm.append(np.repeat(amp, amp_samples))
            wfm.append(np.repeat(zero_val, delay_samples))

        wfm = np.concatenate(wfm)
        if n > 1:
            wfm = np.tile(wfm, n)

        return wfm


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
              and document it!
        '''
        # Note: Do not use HF_gain.Get/SetValue

        if step is None:
            return self.base.HF_Gain.GetStep()

        if step > 31:
            log.warning('Requested TEO gain step is too high')
            step = 31
        if step < 0:
            log.warning('Requested TEO gain step is too low')
            step = 0

        self.base.HF_Gain.SetStep(step)


    def gain_LBW(self, gain=None):
        '''
        Set the low BW gain with minimal effect on the high BW gain.
        This is the gain used for the internal HFI channel

        Possible gain settings are 0,1,2,3
        0 is the LOWEST gain, 3 is the HIGHEST.
        each step corresponds to 6dB (2×)

        Side effect is 1dB extra gain on HBW per step added to LBW.

        TODO: WRITE THE GAIN VALUES HERE IN V/A OR V/V
              GIVE SATURATION LEVEL in V!
        '''
        step = self.base.HF_Gain.GetStep()

        if gain is None:
            return 3 - (step & 0b00011)
        elif gain in (0, 1, 2, 3):
            newstep = step & 0b11100
            newstep |= 3 - gain
            self.base.HF_Gain.SetStep(newstep)
        else:
            raise Exception(f'{gain} is not a possible LBW gain.')


    def gain_HBW(self, gain=None):
        '''
        Set the high BW gain without affecting the low BW gain.
        This is the gain for the HF FULL BW SMA port.
        Note that the HBW gain still depends up to 4 dB on the LBW gain ...
        Fewer settings are available as we can only change the MSB as explained in self.gain()

        Possible gain settings are 0-7
        0 is the LOWEST gain, 7 is the HIGHEST.
        each step corresponds to 4dB (1.585×)

        TODO: WRITE THE GAIN VALUES HERE IN V/A OR V/V
              GIVE SATURATION LEVEL IN V!
        '''
        step = self.base.HF_Gain.GetStep()

        if gain is None:
            return 7 - (step >> 2)
        elif gain in (0, 1, 2, 3, 4, 5, 6, 7):
            newstep = step & 0b00011
            newstep |= (7 - gain) << 2
            self.base.HF_Gain.SetStep(newstep)
        else:
            raise Exception(f'{gain} is not a possible LBW gain.')


    def Irange(self, I=None, LBW=True, HBW=True, INT=False, ext_coupling=50):
        '''
        Selects the appropriate gain step depending on what channels are being used:
        LBW (Low Band Width): HF_LIMITED_BW
        HBW (High Band Width): HF_FULL_BW
        INT: Internal Scope

        If I is None, the current gain step is returned.
        '''

        # Current saturation levels
        lbw_Isat0, cal = self.apply_calibration(self.lbw_Vsat0, 'HF_LIMITED_BW', 0)  # A
        hbw_Isat0, cal = self.apply_calibration(self.hbw_Vsat0, 'HF_FULL_BW', 0)  # A
        int_Isat0 = self.int_Vsat0/50 if self.J29 else self.int_Vsat0/100  # A

        # Range increase per step
        hbw_multiplier = 1.122
        lbw_multiplier = 2

        # Number of steps
        hbw_nsteps = 32
        lbw_nsteps = 4

        if I is not None:
            # Checking if currents are too high for each channel. If they are, the channel will be ignored for the step
            # calculation.
            if HBW and I > hbw_Isat0 * hbw_multiplier**(hbw_nsteps-1):
                log.warning("Current is too high for HF_FULL_BW. Ignoring this channel.")
                HBW = False
            if LBW and I > lbw_Isat0 * lbw_multiplier**(lbw_nsteps-1):
                log.warning("Current is too high for HF_LIMITED_BW. Ignoring this channel.")
                LBW = False
            if INT and I > int_Isat0 * lbw_multiplier**(lbw_nsteps-1):
                log.warning("Current is too high for Internal Scope. Ignoring this channel.")
                INT = False

            # Since Internal Scope and HF_LIMITED_BW use the same amp, and Internal Scope has lower ranges, this one has
            # the preference.
            lbw_Isat00 = int_Isat0 if INT else lbw_Isat0

            # If LBW is used, this range is calculated first, letting HBW with (32/4-1) = 7 possibilities.
            if LBW:
                # Looking for the LBW range
                for ls in range(lbw_nsteps):
                    if I < (lbw_Isat00 * lbw_multiplier ** ls):
                        break

                # Looking for the HBW range maintaining the LBW range
                HBW_possible_steps = [ls + lbw_nsteps * i for i in range(hbw_nsteps // lbw_nsteps - 1)]
                for gain_step in HBW_possible_steps:
                    if I < (hbw_Isat0*hbw_multiplier**gain_step):
                        break
            elif HBW:
                # Looking for the HBW range ignoring LBW range
                for gain_step in range(hbw_nsteps):
                    if I < (hbw_Isat0 * hbw_multiplier ** gain_step):
                        break
            else:
                gain_step = 31

            self.gain(gain_step)
        else:
            gain_step = self.gain()

        lbw_range = lbw_Isat0 * lbw_multiplier**(gain_step%4)
        int_range = int_Isat0 * lbw_multiplier**(gain_step%4)
        hbw_range = hbw_Isat0 * hbw_multiplier**gain_step

        log.info(f"HF_LIMITED_BW range: {lbw_range}A\n"
                 f"HF_FULL_BW range: {hbw_range}A\n"
                 f"Internal Scope range: {int_range}A")

        return gain_step


    def apply_calibration(self, datain, channel, gain_step, reverse=False):

        channel = channel.upper()
        if channel not in ['HFI_INT', 'HFV', 'HFV_INT', 'HF_FULL_BW', 'HF_LIMITED_BW', 'V_MONITOR', 'LFV', 'LFI']:
            raise Exception(f"There is no channel '{channel}'. Possible channels are 'HFI_INT', 'HFV', 'HFV_INT',"
                            f"'HF_FULL_BW', 'HF_LIMITED_BW', 'V_MONITOR', 'LFV', 'LFI'.")

        if gain_step < 0 or gain_step > 31:
            raise Exception(f"There is no gain step {gain_step}, they only go from 0 to 31.")

        if self.calibration is not None:
            a, b = self.calibration.loc[gain_step][channel]
            a, b = (a/2,b/2) if not self.J29 and channel in ['HFI_INT', 'HF_FULL_BW', 'HF_LIMITED_BW', 'LFI'] else (a,b)
            a, b = (1/a, -b/a) if reverse else (a, b)
            return np.polyval((a, b), datain), (a, b)
        else:
            log.warning("There is not a calibration loaded.")
            return datain, (None, None)


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
            Vstandby = self.base.LF_Voltage.GetValue()
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

        varray, trig1, trig2 = self._pad_wfms(varray, trig1, trig2)

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

        wf = self.base.AWG_WaveformManager.CreateWaveform(name)
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
        wfm = self.base.AWG_WaveformManager.GetWaveform(name)
        v = np.array(wfm.AllSamples())
        trig1 = np.array(wfm.All_ADC_Gates())
        trig2 = np.array(wfm.All_BER_Gates())
        return (v, trig1, trig2)


    def download_all_wfms(self):
        '''
        With this we can resynchronize our instance with the TSX_DM memory
        e.g. self.waveforms = self.download_all_wfms()
        '''
        return {name: self.download_wfm(name) for name in self.get_wfm_names()}


    def delete_all_wfms(self):
        name = self.base.AWG_WaveformManager.GetWaveformName(0)
        if name != '':
            self.base.AWG_WaveformManager.DeleteWaveform(name)
            self.delete_all_wfms()
        # there is also
        # teo.HF_WaveformManager_Reset() which might do something similar


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
            success = self.base.AWG_WaveformManager.Run(name, n)
            if not success:
                log.error('No waveform with that name has been uploaded')
        elif type(wfm) in (np.ndarray, list, tuple):
            # this will hash the data to make a name
            # won't upload again if the hash matches
            name = self.upload_wfm(wfm, trig1=trig1, trig2=trig2)
            success = self.base.AWG_WaveformManager.Run(name, n)

        if success:
            self.last_waveform = name
            self.last_gain = self.gain()
            self.last_nshots = n
        else:
            log.error('TEO waveform output failed')

        return success


    def get_data(self, raw=False, nanpad=True):
        '''
        Get the data for both ADC channels for the last capture.
        Returns a dict of information.

        if raw is True, then keys 'HFV' and 'HFI' will also be in the returned dict
        which are the ADC values before calibration/conversion/trimming

        We return the programmed waveform data as well,
        which is useful because the monitor signal can have a lot of noise

        We want all the returned arrays to be the same length, even if the triggers are not always on.
        We have two options:
        nanpad=False: delete programmed voltage waveform and time waveform where trig1 is False
        nanpad=True: pad measured waveform with np.nan where trig1 is False
        '''
        # We only get waveform samples where trigger is True, so these could be shorter than wfm
        # V monitor waveform (HFV)
        wf00 = self.base.AWG_WaveformManager.GetLastResult(0)
        # Current waveform (HFI)
        wf01 = self.base.AWG_WaveformManager.GetLastResult(1)

        if wf00.IsSaturated():
            # I don't think this will ever happen.
            # but there is some manual way to increase the gain on this channel such that it could happen
            log.warning('TEO ADC channel 0 (HFV monitor) is saturated!')
        if wf01.IsSaturated():
            log.warning('TEO ADC channel 1 (HFI) is saturated!')

        # Signals on the HFV, HFI ports
        # Gain is divided out already before it is read in here
        # but they values are still not in volts and they require calibration
        HFV = np.array(wf00.GetWaveformDataArray())
        HFI = np.array(wf01.GetWaveformDataArray())

        R_HFI = 50 if self.J29 else 100
        if self.last_gain is not None:
            gain_step = self.last_gain
        else:
            # could be wrong if you changed the gain between output_wfm and get_data
            gain_step = self.gain()

        # Conversion to current if calibration is available
        I, I_cal_line = self.apply_calibration(HFI, 'HFI_INT', gain_step)
        V, V_cal_line = self.apply_calibration(HFV, 'HFV_INT', gain_step)

        sample_rate = wf00.GetWaveformSamplingRate() # Always 500 MHz

        # last_waveform should always be in self.waveforms, but maybe not if you reset the
        # instance state but wfm was still loaded in teo memory..
        # if we go BORG this will be less likely to happen
        if self.last_waveform in self.waveforms:
            # Refer to the waveform dict stored in the instance
            prog_wfm, trig1, trig2 = self.waveforms[self.last_waveform]
        elif self.last_waveform in self.get_wfm_names():
            log.debug('Waveform data was missing from TeoSystem instance. It will be downloaded from device memory.')
            prog_wfm, trig1, trig2 = self.download_wfm(self.last_waveform)
            self.waveforms[self.last_waveform] = (prog_wfm, trig1, trig2)
        else:
            log.debug('Waveform data was missing from TeoSystem instance and from device memory.')

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

        if nanpad:
            # Align measurement waveforms with the programmed waveform (for the case that not all(trig1))
            # TODO: the extra samples might come at the end of every chunk, every shot, random locations,
            #    we don't know yet.  the following assumes that they are all at the end!
            I, V = self._align_with_trigger(trig1, I, V)
        else:
            # Alternatively we can cut the programmed wfm to match trig1, dropping some information
            # This uses less memory, but the time array will still reflect the gap in data acquisition
            prog_wfm = prog_wfm[trig1]
            t = t[trig1]

        # TODO: should we compress the trigger signals and return them?
        #       otherwise they could be up to 64 MB per shot.
        #       if we use _align_with_trigger, the information for trig1 is already there

        # TODO: This could return a really large data structure.
        #       we might need options to return something more minimal if we want to use long waveforms
        #       could also convert to float32, then I would use a dtype argument: get_data(..., dtype=float32)
        out = dict(V=V, I=I, t=t, Vwfm=wfm,
                   idn=self.constants.idn,
                   sample_rate=sample_rate,
                   gain_step=gain_step,
                   nshots=self.last_nshots,
                   units=dict(V='V', I='A', t='s', Vwfm='V'),
                   calibration=dict(V=V_cal_line, I=I_cal_line))
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
            name = self.base.AWG_WaveformManager.GetWaveformName(i)
            if name == '':
                break
            else:
                wfm_names.append(name)
        return wfm_names


    def delete_all_wfms(self):
        for name in self.get_wfm_names():
            self.base.AWG_WaveformManager.DeleteWaveform(name)


    ##################################### LF mode #############################################

    # Source meter mode
    # I believe you just set voltages and read currents in a software defined sequence
    # Output impedance is 50Ω
    # TODO: will we blow up the input if we have more than 4 uA? how careful should we be?
    # TODO: is the current range bipolar?

    def LF_mode(self, external=True):
        '''
        Switch to LF mode (HF LED should turn off)

        There's no software controllable internal/external mode anymore.
        for internal mode, put jumpers J4 and J5 to position 2-3
        for external mode, put jumpers J4 and J5 to position 1-2

        J4 and J5 are located underneath the square metal RF shield,
        you can pry off the top cover to access them.

        If external is True, setting voltages is disabled
        '''

        self.base.LF_Measurement.SetLF_Mode(0, external)


    def LF_voltage(self, value=None):
        '''
        Gets or sets the LF source voltage value
        This also sets the idle level for HF mode..  but doesn't switch to LF_mode or anything
        Minimum voltage step is 0.002V.

        Teo indicated that in LF mode, the voltage appears on the HFI port, not HFV, which is the reverse of HF mode.
        Alejandro said this is not actually the case..

        TODO: check that we are in the possible output voltage range
        '''
        if value is None:
            value = self.base.LF_Voltage.GetValue()
            return value
        else:
            self.base.LF_Voltage.SetValue(value)
            time.sleep(0.002)  # It takes around 2ms to stabilize the voltage


    def LF_current(self, NPLC=10):
        '''
        Reads the LF current

        specs say ADC is 24 bits, 4 uA range, 1pA resolution
        but in practice the noise may be much higher than 1pA
        For 1Mohm and 1V the standard deviation is around 300 pA before averaging
        After averaging with NPLC=1 standard deviation is 150 pA

        the ADC has a sample rate of 31,248 Hz, but the buffer size is 8,000

        LF_MeasureCurrent(duration) returns all the samples it could store for that duration
        you can then average them.

        but it means we can't average longer than 256 ms in a single shot
        which is 12.8 PLCS for 50 Hz line freq

        TODO: calibrate this with external source meter and store the calibration values in the class
        '''
        if NPLC > .256 * self.PLF:
            log.warning(f'I THINK the LF Output buffer is too small for NPLC={NPLC}')

        duration = NPLC / self.PLF
        Iwfm = self.base.LF_Measurement.LF_MeasureCurrent(duration)
        if Iwfm.IsSaturated():
            log.warning('TEO LF Output is saturated!')
        I = Iwfm.GetWaveformDataArray()
        return np.mean(I)


    ##################################### Highest level commands #############################################

    def measureHF(self, wfm, n=1, trig1=None, trig2=None, raw=False, nanpad=True):
        '''
        Pulse wfm and return I,V,... data
        '''
        self.HF_mode()
        self.output_wfm(wfm, n=n, trig1=trig1, trig2=trig2)
        return self.get_data(raw=raw, nanpad=nanpad)


    def measureLF(self, V_list, NPLC=10):
        '''
        Use internal LF mode to make a low-current measurement
        not tested
        '''

        self.LF_mode(external=True)

        Vidle = self.LF_voltage()

        I = []
        V = []
        t = []
        for v in V_list:
            self.LF_voltage(v)
            vv = self.LF_voltage()
            tt1 = time.time()
            ii = self.LF_current(NPLC)
            tt2 = time.time()
            tt = tt1 + (tt2 - tt1) / 2
            I.append(ii)
            V.append(vv)
            t.append(tt)
        t = np.array(t) - t[0]

        # Go back to the idle level
        self.LF_voltage(Vidle)

        I = np.array(I)
        V = np.array(V)
        t = np.array(t)
        V_list = np.array(V_list)
        if self.calibration is not None:
            I = np.polyval(self.calibration.loc[0, 'LFI'], I)

        data = dict(V_list=V_list, I=I, V=V, t=t,
                                 idn=self.idn(), gain_step=self.gain(),
                                 units=dict(V_list='V', I='A', V='V', t='s'))

        return data

class TeoBase(object):
    '''
    This class is basically a direct wrapper for the Teo API commands.
    It launches the TSX processes and communicates with them using COM.

    Each command prints itself to the debug log so that if we have problems we can provide code
    that is understood by the instrument designer.
    '''
    def __init__(self):

        from win32com.client import Dispatch
        from pythoncom import com_error

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
        except AttributeError as e:
            print(e)
            # probably dumb error that requires you to delete this directory
            gen_py = os.path.expandvars('%userprofile%\\AppData\\Local\\Temp\\gen_py')
            if os.path.isdir(gen_py):
                log.error('Deleting gen_py directory. Try again.')
                shutil.rmtree(gen_py)
                # Try again
                HMan = Dispatch('TSX_HMan')
            else:
                raise e
            return

        # Asks the program for a device called MEMORY_TESTER
        MemTester = HMan.GetSystem('MEMORY_TESTER')
        if MemTester is None:
            log.error('Teo software cannot locate a connected memory tester. Check USB connection.')
            self.conn = False
            return

        self.conn = True

        # Access a bunch of COM classes used to control the TEO board.
        # The contained methods/attributes appear in tab completion, but the contained classes do not
        DriverID =            self._CastTo('ITS_DriverIdentity'     , MemTester)
        DeviceID =            self._CastTo('ITS_DeviceIdentity'     , DriverID)
        DeviceControl =       self._CastTo('ITS_DeviceControl'      , DriverID)
        LF_Measurement =      self._CastTo('ITS_LF_Measurement'     , DriverID)
        LF_Voltage =          LF_Measurement.LF_Voltage
        HF_Measurement =      self._CastTo('ITS_HF_Measurement'     , DriverID)
        #TODO: Is this different from HF_Gain = HF_Measurement.HF_Gain?
        HF_Gain =             self._CastTo('ITS_DAC_Control'        , HF_Measurement.HF_Gain)
        AWG_WaveformManager = self._CastTo('ITS_AWG_WaveformManager', HF_Measurement.WaveformManager)

        # These are all the High Level Classes to be wrapped
        classes = dict(HMan=HMan, DriverID=DriverID, DeviceControl=DeviceControl, DeviceID=DeviceID,
                       LF_Measurement=LF_Measurement, LF_Voltage=LF_Voltage,
                       HF_Measurement=HF_Measurement, HF_Gain=HF_Gain, AWG_WaveformManager=AWG_WaveformManager)

        # Every High Level Class will be an instance of hlc() with all its methods, so one can call them like:
        # teo.base.HF_Gain.Set_Value(5) and the autocompletion will be shown.
        class hlc(object):
            pass

        #log.debug('-'*30 + '\nBASE FUNCTIONS:')
        for cls_name, cls_obj in classes.items():
            # Create the instance of the High level class
            setattr(self, f"{cls_name}", hlc())
            cls_obj = classes[cls_name]
            #log.debug(f'{cls_name}')
            # Define all its methods
            for mtd in dir(cls_obj):
                if not mtd.startswith('_') and mtd not in ['CLSID', 'coclass_clsid']:
                    setattr(getattr(self, f"{cls_name}"), f"{mtd}", self._wrapper(cls_name, cls_obj, mtd))
                    #log.debug(f'\t{mtd}')
        #log.debug('-' * 30)


    @staticmethod
    def _CastTo(name, to):
        from win32com.client import CastTo
        # CastTo that clearly lets you know something isn't working right with the software setup
        try:
            result = CastTo(to, name)
        except Exception as E:
            log.error(E)
            log.error(f'Teo software connection failed! CastTo({name}, {to})')
            result = None

        return result

    @staticmethod
    def _wrapper(cls_name, cls_obj, mtd):
        '''
        Create a function with logging.

        Parameters
        ----------
        cls_name: Class name
        mtd: Method

        Returns
        Function called 'cls_mtd'
        -------
        '''

        func = getattr(cls_obj, mtd)
        sig = signature(func)


        def wfunc(*args, **kwargs):
            '''

            '''
            # 'par' is a string that contains all the arguments passed to the function, so the log can look
            # exactly like the command used
            par = f'('
            if len(args) > 0:
                par += f'{args[0]}'
                for arg in args[1:]:
                    par += f', {arg}'
                for k, i in kwargs.items():
                    par += f', {k}={i}'
            elif len(kwargs) > 0:
                for k, i in kwargs.items():
                    par += f'{k}={i}, '
                par = par[:-2]
            par += ')'
            log.debug(f"{cls_name}.{mtd}{par}")

            v = func(*args, **kwargs)

            log.debug(f"\t{v}")
            return v

        wfunc.__signature__ = sig

        return wfunc
