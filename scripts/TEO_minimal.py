# TH 2018-12-10
# HJR 2018
import win32com.client
from win32com.client import CastTo, WithEvents, Dispatch
import numpy as np
from matplotlib import pyplot as plt

# Launches program that knows about which TEO boards are connected via usb
HMan = Dispatch("TSX_HMan")

# Asks the program for a device called MEMORY_TESTER
MemTester = HMan.GetSystem('MEMORY_TESTER')

def CastTo(name, to):
    # CastTo that prints a warning if it doesn't work for some reason
    result = CastTo(to, name)
    if result is None: print(f'{name} has failed')
    return result

# Access a bunch of classes used to control the TEO board
DriverID =            CastTo('ITS_DriverIdentity'     , MemTester)
DeviceID =            CastTo('ITS_DeviceIdentity'     , DriverID)
DeviceControl =       CastTo('ITS_DeviceControl'      , DriverID)
LF_Measurement =      CastTo('ITS_LF_Measurement'     , DriverID)
HF_Measurement =      CastTo('ITS_HF_Measurement'     , DriverID)
HF_Gain =             CastTo('ITS_DAC_Control'        , HF_Measurement.HF_Gain)
AWG_WaveformManager = CastTo('ITS_AWG_WaveformManager', HF_Measurement.WaveformManager)

DeviceControl.StartDevice()

# Get and print some information from the board
DevName = DeviceID.GetDeviceName()
DevRev = [DeviceID.GetDeviceMajorRevision(), DeviceID.GetDeviceMinorRevision()]
DevSN = DeviceID.GetDeviceSerialNumber()
print ("TEO: Name/SN/Rev=" + DevName + " SN=" + str(DevSN) + " Rev=" + str(DevRev[0]) + "." + str(DevRev[0]))

# Gain settings.  Need to see a schematic to know what this gain actually refers to
# unit is "steps" and each steps correspond to 1dB
HFgain = 20
if float(HFgain) > HF_Gain.GetMaxValue():
    print("Input error: Requested TEO gain too high")
HF_Gain.SetValue(HFgain)

HF_Measurement.SetHF_Mode(0, True)      # Call to turn on HF mode, 0 is useless, True for external mode

# How to use LF mode?
#LF_Voltage = LF_Measurement.LF_Voltage
#LFVolt = LF_Voltage.GetValue()
#LF_Measure = LF_Measurement.LF_Measure

def Test():
    A = np.sin(np.linspace(0, 100*3.141592653589793238462, 100000))
    trig1 = np.sin(np.linspace(0, 3.1415, 100000)) > .3

    wf = AWG_WaveformManager.CreateWaveform("test")

    # Have to upload the waveform and triggers in parallel, one point at a time..
    for a, trig in zip(A, trig1):
        # wfm value, trigger1, trigger2, ?
        wf.AddSample(a, trig, True, 1)

    wf_read_length = wf.Length()
    HF_Measurement.SetHF_Mode(0, False)
    HF_Gain.SetValue(20)
    AWG_WaveformManager.Run("test", 1)
    # We only get waveform samples where trigger1 is True, so these could be shorter than A
    # Vmonitor waveform
    wf00 = AWG_WaveformManager.GetLastResult(0)
    # Iout waveform
    wf01 = AWG_WaveformManager.GetLastResult(1)
    print('Getting data...')
    # Have to get the data one point at a time ....  takes forever obviously.
    # Allocate
    wflen = wf00.GetWaveformLength()
    t = np.where(trig1)[0]
    Vmonitor = np.array([wf00.GetWaveformData(i) for i in range(wflen)])
    Iout =     np.array([wf01.GetWaveformData(i) for i in range(wflen)])
    print('Plotting Data...')
    plt.figure()
    plt.plot(A, label='Input Waveform')
    plt.plot(trig1, label='Sampling Trigger')
    plt.plot(t, Vmonitor, label='V monitor [?]')
    plt.plot(t, Iout, label='Current [?]')
    plt.legend()

    fig, ax = plt.subplots()
    ax.plot(Vmonitor, Iout)
