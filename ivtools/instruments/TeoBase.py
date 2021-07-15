import logging

log = logging.getLogger('instruments')

class TeoBase(object):
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

    # TODO: do all the commands work regardless of which mode we are in? e.g. waveform upload, gain setting
            how do we avoid issuing commands and expecting it to do something but we are in the wrong mode?
            need to check something like isin(HF_mode) ?

    # TODO: could wrap all of the Teo COM functions only for the purpose of giving them signatures, docstrings,
            default arguments.

    # TODO: Debug mode that always prints out all the COM calls, so we have code that is recognizable to Teo
            for support purposes

    # TODO: since there seem to be several situations that cause the HFV output to go to the negative rail
            and blow up your device, make sure to document them here
            seems to be whenever TSX_DM.exe connects to the system
            1. on first initialization (Dispatch('TSX_HMan'))
            2. if you disconnect USB and plug it back in

    '''

    def __init__(self):
        # This imports are here os a macOS user can import ivtools and use
        from win32com.client import Dispatch
        from pythoncom import com_error
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
        DriverID =            self._CastTo('ITS_DriverIdentity'     , MemTester)
        DeviceID =            self._CastTo('ITS_DeviceIdentity'     , DriverID)
        DeviceControl =       self._CastTo('ITS_DeviceControl'      , DriverID)
        LF_Measurement =      self._CastTo('ITS_LF_Measurement'     , DriverID)
        LF_Voltage =          LF_Measurement.LF_Voltage
        HF_Measurement =      self._CastTo('ITS_HF_Measurement'     , DriverID)
        # Is this different from HF_Gain = HF_Measurement.HF_Gain?
        # TODO: why can't we see e.g. HF_Measurement.HF_Gain in tab completion?
        HF_Gain =             self._CastTo('ITS_DAC_Control'        , HF_Measurement.HF_Gain)
        AWG_WaveformManager = self._CastTo('ITS_AWG_WaveformManager', HF_Measurement.WaveformManager)


        self.classes = ['HMan', 'MemTester', 'DriverID', 'DeviceControl', 'DeviceID',
                        'LF_Measurement', 'LF_Voltage',
                        'HF_Measurement', 'HF_Gain', 'AWG_WaveformManager']

        self._wrap_all()


    @staticmethod
    def _CastTo(name, to):
        from win32com.client import CastTo
        # CastTo that clearly lets you know something isn't working right with the software setup
        try:
            result = CastTo(to, name)
        except Exception as E:
            log.error(f'Teo software connection failed! CastTo({name}, {to})')
        if result is None:
            log.error(f'Teo software connection failed! CastTo({name}, {to})')
        return result


    def print_function_names(self):
        '''
        Because there's no manual yet
        TODO: find out if we can discover the class names
        '''

        for tlc in self.classes:
            print(tlc)
            c = getattr(self.com, tlc)
            for node in dir(c):
                if not node.startswith('_'):
                    print(f'\t{node}')


    def _wrapper(self, cls, mtd):
        '''
        Create a function with logging.

        Parameters
        ----------
        cls: Class
        mtd: Method

        Returns
        Function called 'cls_mtd'
        -------

        '''
        def func(*args, **kwargs):
            '''

            '''

            # 'par' is a string than contains all the arguments passed to the function, so the log can look
            # exactly like the command used
            par = f'('
            if len(args) > 0:
                par += f'{args[0]}'
            for arg in args[1:]:
                par += f', {arg}'
            for k, i in kwargs.items():
                par += f', {k}={i}'
            par += ')'
            log.debug(f"{cls}.{mtd}{par}")

            v = getattr(getattr(self.com, cls), mtd)(*args, **kwargs)

            log.debug(f"\t{v}")
            return v

        return func

    def _wrap_all(self):
        '''

        '''

        for cls in self.classes:
            c = getattr(self.com, cls)
            for mtd in dir(c):
                if not mtd.startswith('_') and mtd not in ['CLSID', 'coclass_clsid']:
                    setattr(self, f"{cls}_{mtd}", self._wrapper(cls, mtd))









"""
    ###### Direct wrappers for adding python function signatures and docstrings ####
    ## Could it be better with classes and subclasses?

    ## DeviceID
    
    def GetDeviceDescription(self, *args):
        '''

        '''
        v = self.com.DeviceID.GetDeviceDescription(*args)
        log.debug(f"DeviceID.GetDeviceDescription{args}  -->  {v}")
        return v

    def GetDeviceMajorRevision(self, *args):
        '''

        '''
        v = self.com.DeviceID.GetDeviceMajorRevision(*args)
        log.debug(f"DeviceID.GetDeviceMajorRevision{args}  -->  {v}")
        return v

    def GetDeviceMinorRevision(self, *args):
        '''

        '''
        v = self.com.DeviceID.GetDeviceMinorRevision(*args)
        log.debug(f"DeviceID.GetDeviceMinorRevision{args}  -->  {v}")
        return v

    def GetDeviceName(self, *args):
        '''

        '''
        v = self.com.DeviceID.GetDeviceName(*args)
        log.debug(f"DeviceID.GetDeviceName{args}  -->  {v}")
        return v

    def GetDeviceSerialNumber(self, *args):
        '''

        '''
        v = self.com.DeviceID.GetDeviceSerialNumber(*args)
        log.debug(f"DeviceID.GetDeviceSerialNumber{args}  -->  {v}")
        return v


    ## DeviceControl

    def InitDevice(self, *args):
        '''

        '''

        v = self.com.DeviceControl.InitDevice(*args)
        log.debug(f"DeviceControl.InitDevice{args}  -->  {v}")
        return v

    def IsCompatible(self, *args):
        '''

        '''

        v = self.com.DeviceControl.IsCompatible(*args)
        log.debug(f"DeviceControl.IsCompatible{args}  -->  {v}")
        return v

    def IsStarted(self, *args):
        '''

        '''

        v = self.com.DeviceControl.IsStarted(*args)
        log.debug(f"DeviceControl.IsStarted{args}  -->  {v}")
        return v

    def ReinitDevice(self, *args):
        '''

        '''

        v = self.com.DeviceControl.ReinitDevice(*args)
        log.debug(f"DeviceControl.ReinitDevice{args}  -->  {v}")
        return v

    def ResetDevice(self, *args):
        '''

        '''

        v = self.com.DeviceControl.ResetDevice(*args)
        log.debug(f"DeviceControl.ResetDevice{args}  -->  {v}")
        return v

    def StartDevice(self, *args):
        '''

        '''

        v = self.com.DeviceControl.StartDevice(*args)
        log.debug(f"DeviceControl.StartDevice{args}  -->  {v}")
        return v

    def StopDevice(self, *args):
        '''
        Lights should turn off on the round board and HFV output probably floats.*args
        Controller board remains on.
        '''

        v = self.com.DeviceControl.StopDevice(*args)
        log.debug(f"DeviceControl.StopDevice{args}  -->  {v}")
        return v


    ## LF_Measurement

    def GetLF_Mode(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.GetLF_Mode(*args)
        log.debug(f"LF_Measurement.GetLF_Mode{args}  -->  {v}")
        return v

    def GetLF_ModeDescription(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.GetLF_ModeDescription(*args)
        log.debug(f"LF_Measurement.GetLF_ModeDescription{args}  -->  {v}")
        return v

    def GetLF_ModeMax(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.GetLF_ModeMax(*args)
        log.debug(f"LF_Measurement.GetLF_ModeMax{args}  -->  {v}")
        return v

    def GetLF_Supported(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.GetLF_Supported(*args)
        log.debug(f"LF_Measurement.GetLF_Supported{args}  -->  {v}")
        return v

    def LF_MeasureCurrent(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_MeasureCurrent(*args)
        log.debug(f"LF_Measurement.LF_MeasureCurrent{args}  -->  {v}")
        return v

    def RunLF_Calibration(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.RunLF_Calibration(*args)
        log.debug(f"LF_Measurement.RunLF_Calibration{args}  -->  {v}")
        return v

    def SetLF_Mode(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.SetLF_Mode(*args)
        log.debug(f"LF_Measurement.SetLF_Mode{args}  -->  {v}")
        return v

    def LF_Voltage_GetMaxValue(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetMaxValue(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetMaxValue{args}  -->  {v}")
        return v

    def LF_Voltage_GetMinValue(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetMinValue(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetMinValue{args}  -->  {v}")
        return v

    def LF_Voltage_GetPrefix(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetPrefix(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetPrefix{args}  -->  {v}")
        return v

    def LF_Voltage_GetStep(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetStep(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetStep{args}  -->  {v}")
        return v

    def LF_Voltage_GetStepForValue(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetStepForValue(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetStepForValue{args}  -->  {v}")
        return v

    def LF_Voltage_GetStepNumber(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetStepNumber(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetStepNumber{args}  -->  {v}")
        return v

    def LF_Voltage_GetUnit(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetUnit(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetUnit{args}  -->  {v}")
        return v

    def LF_Voltage_GetValue(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetValue(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetValue{args}  -->  {v}")
        return v

    def LF_Voltage_GetValueForStep(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.GetValueForStep(*args)
        log.debug(f"LF_Measurement.LF_Voltage.GetValueForStep{args}  -->  {v}")
        return v

    def LF_Voltage_SetStep(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.SetStep(*args)
        log.debug(f"LF_Measurement.LF_Voltage.SetStep{args}  -->  {v}")
        return v

    def LF_Voltage_SetValue(self, *args):
        '''

        '''
        v = self.com.LF_Measurement.LF_Voltage.SetValue(*args)
        log.debug(f"LF_Measurement.LF_Voltage.SetValue{args}  -->  {v}")
        return v


    ## HF_Measurement

    def GetHF_Mode(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.GetHF_Mode(*args)
        log.debug(f"HF_Measurement.GetHF_Mode{args}  -->  {v}")
        return v

    def GetHF_ModeDescription(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.GetHF_ModeDescription(*args)
        log.debug(f"HF_Measurement.GetHF_ModeDescription{args}  -->  {v}")
        return v

    def GetHF_ModeMax(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.GetHF_ModeMax(*args)
        log.debug(f"HF_Measurement.GetHF_ModeMax{args}  -->  {v}")
        return v

    def GetHF_Supported(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.GetHF_Supported(*args)
        log.debug(f"HF_Measurement.GetHF_Supported{args}  -->  {v}")
        return v

    def SetHF_Mode(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.SetHF_Mode(*args)
        log.debug(f"HF_Measurement.SetHF_Mode{args}  -->  {v}")
        return v

    def HF_Gain_GetMaxValue(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetMaxValue(*args)
        log.debug(f"HF_Gain.GetMaxValue{args}  -->  {v}")
        return v

    def HF_Gain_GetMinValue(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetMinValue(*args)
        log.debug(f"HF_Gain.GetMinValue{args}  -->  {v}")
        return v

    def HF_Gain_GetPrefix(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetPrefix(*args)
        log.debug(f"HF_Gain.GetPrefix{args}  -->  {v}")
        return v

    def HF_Gain_GetStep(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetStep(*args)
        log.debug(f"HF_Gain.GetStep{args}  -->  {v}")
        return v

    def HF_Gain_GetStepForValue(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetStepForValue(*args)
        log.debug(f"HF_Gain.GetStepForValue{args}  -->  {v}")
        return v

    def HF_Gain_GetStepNumber(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetStepNumber(*args)
        log.debug(f"HF_Gain.GetStepNumber{args}  -->  {v}")
        return v

    def HF_Gain_GetUnit(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetUnit(*args)
        log.debug(f"HF_Gain.GetUnit{args}  -->  {v}")
        return v

    def HF_Gain_GetValue(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetValue(*args)
        log.debug(f"HF_Gain.GetValue{args}  -->  {v}")
        return v

    def HF_Gain_GetValueForStep(self, *args):
        '''

        '''
        v = self.com.HF_Gain.GetValueForStep(*args)
        log.debug(f"HF_Gain.GetValueForStep{args}  -->  {v}")
        return v

    def HF_Gain_SetStep(self, *args):
        '''

        '''
        v = self.com.HF_Gain.SetStep(*args)
        log.debug(f"HF_Gain.SetStep{args}  -->  {v}")
        return v

    def HF_Gain_SetValue(self, *args):
        '''

        '''
        v = self.com.HF_Gain.SetValue(*args)
        log.debug(f"HF_Gain.SetValue{args}  -->  {v}")
        return v

    def HF_WaveformManager_CreateWaveform(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.CreateWaveform(*args)
        log.debug(f"HF_Measurement.WaveformManager.CreateWaveform{args}  -->  {v}")
        return v

    def HF_WaveformManager_DeleteWaveform(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.DeleteWaveform(*args)
        log.debug(f"HF_Measurement.WaveformManager.DeleteWaveform{args}  -->  {v}")
        return v

    def HF_WaveformManager_GetFreeMemory(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.GetFreeMemory(*args)
        log.debug(f"HF_Measurement.WaveformManager.GetFreeMemory{args}  -->  {v}")
        return v

    def HF_WaveformManager_GetLastResult(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.GetLastResult(*args)
        log.debug(f"HF_Measurement.WaveformManager.GetLastResult{args}  -->  {v}")
        return v

    def HF_WaveformManager_GetTotalMemory(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.GetTotalMemory(*args)
        log.debug(f"HF_Measurement.WaveformManager.GetTotalMemory{args}  -->  {v}")
        return v

    def HF_WaveformManager_GetWaveform(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.GetWaveform(*args)
        log.debug(f"HF_Measurement.WaveformManager.GetWaveform{args}  -->  {v}")
        return v

    def HF_WaveformManager_GetWaveformName(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.GetWaveformName(*args)
        log.debug(f"HF_Measurement.WaveformManager.GetWaveformName{args}  -->  {v}")
        return v

    def HF_WaveformManager_GetWaveformNumber(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.GetWaveformNumber(*args)
        log.debug(f"HF_Measurement.WaveformManager.GetWaveformNumber{args}  -->  {v}")
        return v

    def HF_WaveformManager_LoadWaveform(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.LoadWaveform(*args)
        log.debug(f"HF_Measurement.WaveformManager.LoadWaveform{args}  -->  {v}")
        return v

    def HF_WaveformManager_Reset(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.Reset(*args)
        log.debug(f"HF_Measurement.WaveformManager.Reset{args}  -->  {v}")
        return v

    def HF_WaveformManager_Run(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.Run(*args)
        log.debug(f"HF_Measurement.WaveformManager.Run{args}  -->  {v}")
        return v

    def HF_WaveformManager_StartFromTrigger(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.StartFromTrigger(*args)
        log.debug(f"HF_Measurement.WaveformManager.StartFromTrigger{args}  -->  {v}")
        return v

    def HF_WaveformManager_UnloadWaveform(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.UnloadWaveform(*args)
        log.debug(f"HF_Measurement.WaveformManager.UnloadWaveform{args}  -->  {v}")
        return v

    def HF_WaveformManager_WaveformAddress(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.WaveformAddress(*args)
        log.debug(f"HF_Measurement.WaveformManager.WaveformAddress{args}  -->  {v}")
        return v

    def HF_WaveformManager_WaveformIsLoaded(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.WaveformIsLoaded(*args)
        log.debug(f"HF_Measurement.WaveformManager.WaveformIsLoaded{args}  -->  {v}")
        return v

    def HF_WaveformManager_WaveformMemoryUsed(self, *args):
        '''

        '''
        v = self.com.HF_Measurement.WaveformManager.WaveformMemoryUsed(*args)
        log.debug(f"HF_Measurement.WaveformManager.WaveformMemoryUsed{args}  -->  {v}")
        return v
        
"""


