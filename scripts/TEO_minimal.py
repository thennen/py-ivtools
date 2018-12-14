#Trying to clean this up and take the banana out of the gorilla's hand - TH 2018-12-10
import win32com.client
from win32com.client import CastTo, WithEvents, Dispatch
import os


TEOintHF = False                # do we use TEO's AWG and scope?
TEOextHF = True                 # do we use external HF things?
TEOintLF = False                 # do we use TEO's internal LF mode?
HFgain = 10

TEOattndB = 14
TEOAttn = 10**(TEOattndB/20)

HMan = Dispatch("TSX_HMan")
DriverID = CastTo(HMan.GetSystem("MEMORY_TESTER"), "ITS_DriverIdentity")

def tryCastTo(name, to=DriverID):
    result = CastTo(to, name)
    if result is None: print(f'{name} has failed')
    return result

DeviceID = tryCastTo('ITS_DeviceIdentity')
DeviceControl = tryCastTo('ITS_DeviceControl')
LF_Measurement = tryCastTo('ITS_LF_Measurement')
HF_Measurement = tryCastTo('ITS_HF_Measurement')
DAC_Control = tryCastTo('ITS_DAC_Control', TS_System_HF_Measurement.HF_Gain)
AWG_WaveformManager = tryCastTo('ITS_AWG_WaveformManager', TS_System_HF_Measurement.WaveformManager)

DeviceControl.StartDevice()

#  === find out current state, DO FOR DEVICE AND SWITCHING = LFMode ===================
DevName = DeviceID.GetDeviceName()
DevRev = [DeviceID.GetDeviceMajorRevision(), DeviceID.GetDeviceMinorRevision()]
DevSN = DeviceID.GetDeviceSerialNumber()

##  ==== HF section. Let it default to full gain. =====================================
if float(HFgain) > HF_Gain.GetMaxValue():
    print("Input error: Requested TEO gain too high")

TS_System_HF_Gain.SetValue(HFgain)

if not HF_Measurement.GetHF_Supported():
    print("Error in TEObox: HF is not reported as supported")

HF_Measurement.SetHF_Mode(0, TEOextHF)      # boolean is whether or not it is external

# collect items that are needed for switching and internal LFmode
if not LF_Measurement.GetLF_Supported():
    print("TS_System_LF_Measurement: LF mode not supported")

if TEOintLF:
    LF_Voltage = LF_Measurement.LF_Voltage
    LFVolt = LF_Voltage.GetValue()
    LF_Measure = LF_Measurement.LF_Measure

print ("TEO: Name/SN/Rev=" + DevName + " SN=" + str(DevSN) + " Rev=" + str(DevRev[0]) + "." + str(DevRev[0]))

def Test(self):
    A = np.sin(np.linspace(0, 100*pi, 1000))

    wf = AWG_WaveformManager.CreateWaveform("test")

    for a in A:
        # wfm value, trigger1, trigger2, ?
        wf.AddSample(a, True, True, 1)

    wf_read_length = wf.Length()
    HF_Measurement.SetHF_Mode(0, False)
    AWG_WaveformManager.Run("test", 1)
    # We only get waveform samples where trigger1 is True, so these could be shorter than A
    wf00 = AWG_WaveformManager.GetLastResult(0)
    wf01 = AWG_WaveformManager.GetLastResult(1)
