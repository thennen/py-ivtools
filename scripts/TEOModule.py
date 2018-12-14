#Trying to clean this up and take the banana out of the gorilla's hand - TH 2018-12-10
import win32com.client
from win32com.client import CastTo, WithEvents, Dispatch
import os
from matplotlib import pyplot as plt

## TEO not working? look here and erase. C:\Users\User\AppData\Local\Temp\gen_py\3.6

class TEObox(object):
    def __init__(self, mode='intHF', HFgain=10)
        """
        TODO: Explain the different modes of TEO here
        Q: Can we enable more than one mode at a time??  I hope not. What happens if we try?

        gain is in 1 dB "steps"

        :param      InstList:   list of external devices that need to be instantiated. Caller to provide
        1. determine how we use TEO
        2. initialze com objects and establish communications
        The clue of this is to use the TEObox in a somewhat similar way as the old Sc. Inside the TEOBox, we keep the TEO instance open to save
        time. Also, inside this box, we make instantiations to all other devices (Scope, AWG, etc) as needed. the caller must provide a list of
        POTENTIAL devices. the Teobox applies its logic ti figure out whether they are actually needed. These are kept alive as well to save time.
        """
        self.pa = paX

        self.TEOintHF = mode == 'intHF'                 # do we use TEO's AWG and scope?
        self.TEOextHF = mode == 'extHF'                 # do we use external HF things?
        self.TEOintLF = mode == 'intLF'                 # do we use TEO's internal LF mode?
        #self.TEOextLF = mode == 'extLF'   # Hans left this out for some reason?  is this not a valid mode?
        self.HFgain = HFgain

        # wtf is this stuff?  So that I don't lose my mind because of all the conditional assignments?
        self.HMan = None
        self.DriverID = None
        self.DeviceID = None
        self.DeviceControl = None
        self.LF_Measurement = None
        self.HF_Measurement = None
        self.LF_Measure = None
        self.HF_Gain = None
        self.LF_Voltage = None
        self.AWG_WaveformManager = None
        self.wf = None
        self.DevName = None
        self.DevDesc = None
        self.DevSN = None
        self.DevRev = None
        self.LFVolt = None
        self.TrcFolder = None
        self.RecIDwPathJSON = None
        self.RecIDwPathBIN = None
        self.RecID = None = None
        self.Sw = None

        self.TEOattndB = 14
        self.TEOAttn = 10**(self.TEOattndB/20)

        # 2 establish communication to TEO
        self.InitializeTEO()

    def AdjustMainGain(self, NewGaindB):
        """
        :param          NewGaindB:
        :return:
        """
        #if self.TS_System_HF_Gain.GetValue() == NewGaindB: return
        self.TS_System_HF_Gain.SetValue(NewGaindB)
        #print("TEO reports " + str(self.TS_System_HF_Gain.GetValue()))


    def InitializeTEO(self):
        """
        :return:        all in self
        Talk to TEO and get the com objects
        """
        print("initializing TEO")
        self.HMan = Dispatch("TSX_HMan")
        if self.HMan is None:
            print("TSX_HMan has failed")
        try:
            self.DriverID = CastTo(self.HMan.GetSystem("MEMORY_TESTER"), "ITS_DriverIdentity")
        except:
            print("TEO's .GetSystem crashes - this may be a license error - 408-332-4449")

        if self.DriverID is None:
            print("DriverID has failed")

        def tryCastTo(name, to=self.DriverID):
            result = CastTo(to, name)
            if result is None: print(f'{name} has failed')
            return result

        self.DeviceID = tryCastTo('ITS_DeviceIdentity')
        self.DeviceControl = tryCastTo('ITS_DeviceControl')
        self.LF_Measurement = tryCastTo('ITS_LF_Measurement')
        self.HF_Measurement = tryCastTo('ITS_HF_Measurement')
        self.DAC_Control = tryCastTo('ITS_DAC_Control', self.TS_System_HF_Measurement.HF_Gain)
        self.AWG_WaveformManager = tryCastTo('ITS_AWG_WaveformManager', self.TS_System_HF_Measurement.WaveformManager)

        # dd1 = self.TS_System_HF_Measurement.HF_Gain
        # dd2 = self.TS_System_HF_Measurement.GetHF_Result(0, 1000, 0)
        # dd3 = self.TS_System_HF_Measurement.CreateWaveform()
        # dd4 = self.TS_System_HF_Measurement.WaveformManager
        self.DeviceControl.StartDevice()

        #  === find out current state, DO FOR DEVICE AND SWITCHING = LFMode ========================================================================
        self.DevName = self.DeviceID.GetDeviceName()
        #self.DevDesc = self.DeviceID.GetDeviceDescription() # not very useful
        self.DevRev = [self.DeviceID.GetDeviceMajorRevision(), self.DeviceID.GetDeviceMinorRevision()]
        self.DevSN = self.DeviceID.GetDeviceSerialNumber()

        ##  ==== HF section. Let it default to full gain. ==========================================================================================
        if float(self.HFgain) > self.HF_Gain.GetMaxValue():
            print("Input error: check Auxfile, requested TEO gain too high")
        self.TS_System_HF_Gain.SetValue(self.HFgain)
        #self.TS_System_HF_Gain.SetValue(0)
        #print("FAKING GAIN CHANGE")
        if not self.HF_Measurement.GetHF_Supported():
            Msg = "Error in TEObox: HF is not reported as supported"
            print(Msg)
        self.HF_Measurement.SetHF_Mode(0, not self.TEOintHF)      # boolean is whether or not it is external

        # collect items we use for internal HFmode
        if self.TEOintHF:
            pass

        # collect items that are needed for switching and internal LFmode
        if not self.LF_Measurement.GetLF_Supported():
            print("TS_System_LF_Measurement: LF mode not supported")

        if self.TEOintLF:
            self.LF_Voltage = self.LF_Measurement.LF_Voltage
            self.LFVolt = self.LF_Voltage.GetValue()
            self.LF_Measure = self.LF_Measurement.LF_Measure

        print ("TEO: Name/SN/Rev=" + self.DevName + " SN=" + str(self.DevSN) + " Rev=" + str(self.DevRev[0]) + "." + str(self.DevRev[0]))

    def HF_GainGetValue(self):
        """
        :return: temporary workaround. just put . in calling when obsolete
        """
        self.HFgain = 31 - self.HF_Gain.GetStep()

    def Test(self):
        A = np.sin(np.linspace(0, 100*pi, 1000))

        self.wf = self.AWG_WaveformManager.CreateWaveform("test")

        for a in A:
            # wfm value, trigger1, trigger2, ?
            self.wf.AddSample(a, True, True, 1)

        wf_read_length = self.wf.Length()
        self.HF_Measurement.SetHF_Mode(0, False)
        self.AWG_WaveformManager.Run("test", 1)
        # We only get waveform samples where trigger1 is True, so these could be shorter than A
        wf00 = self.AWG_WaveformManager.GetLastResult(0)
        wf01 = self.AWG_WaveformManager.GetLastResult(1)
