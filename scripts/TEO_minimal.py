# Trying to clean this up and take the metaphoricaL banana out of the gorilla's hand
# TH 2018-12-10
# HJR 2018
import sys

sys.path.append(r'C:\NVMPyMeas\Analysis')
sys.path.append(r'C:\NVMPyMeas\Measurements')
sys.path.append(r'C:\NVMPyMeas\Utilities')
sys.path.append(r'C:\NVMPyMeas\Instruments')
sys.path.append(r'C:\NVMPyMeas\Projects\DataAnalysis')
sys.path.append(r'C:\NVMPyMeas\Projects\DataAnalysis')
sys.path.append(r'C:\NVMPyMeas\Projects\ProdTest')


from TEOModule import TEObox
import MeasurementOTS
import win32com.client
from win32com.client import CastTo, WithEvents, Dispatch
import numpy as np
from matplotlib import pyplot as plt
import UtilsCollection

# Launches program that knows about which TEO boards are connected via usb
HMan = Dispatch("TSX_HMan")

# Asks the program for a device called MEMORY_TESTER
MemTester = HMan.GetSystem('MEMORY_TESTER')

def whiningCastTo(name, to):
    # CastTo that prints a warning if it doesn't work for some reason
    result = CastTo(to, name)
    if result is None: print(f'{name} has failed')
    return result

# Access a bunch of classes used to control the TEO board
DriverID =            whiningCastTo('ITS_DriverIdentity'     , MemTester)
DeviceID =            whiningCastTo('ITS_DeviceIdentity'     , DriverID)
DeviceControl =       whiningCastTo('ITS_DeviceControl'      , DriverID)
LF_Measurement =      whiningCastTo('ITS_LF_Measurement'     , DriverID)
HF_Measurement =      whiningCastTo('ITS_HF_Measurement'     , DriverID)
HF_Gain =             whiningCastTo('ITS_DAC_Control'        , HF_Measurement.HF_Gain)
AWG_WaveformManager = whiningCastTo('ITS_AWG_WaveformManager', HF_Measurement.WaveformManager)

DeviceControl.StartDevice()

# Get and print some information from the board
DevName = DeviceID.GetDeviceName()
DevRev = [DeviceID.GetDeviceMajorRevision(), DeviceID.GetDeviceMinorRevision()]
DevSN = DeviceID.GetDeviceSerialNumber()
print ("TEO: Name/SN/Rev=" + DevName + " SN=" + str(DevSN) + " Rev=" + str(DevRev[0]) + "." + str(DevRev[0]))

# Gain settings.  Need to see a schematic to know what this gain actually refers to
# unit is "steps" and each steps correspond to 1dB
HFgain = 14
if float(HFgain) > HF_Gain.GetMaxValue():
    print("Input error: Requested TEO gain too high")
HF_Gain.SetValue(HFgain)

HF_Measurement.SetHF_Mode(0, True)      # Call to turn on HF mode, 0 is useless, True for external mode

# How to use LF mode?
#LF_Voltage = LF_Measurement.LF_Voltage
#LFVolt = LF_Voltage.GetValue()
#LF_Measure = LF_Measurement.LF_Measure

def Test():
    #A = .75*np.sin(np.linspace(0, 100*3.141592653589793238462, 100000))
    #trig1 = np.sin(np.linspace(0, 3.1415, 100000)) > .3

    A, trig1 = DCSweep(v_start=0, v_stop=.05, t_pulse = 1000, n_pulse=100)


    wf = AWG_WaveformManager.CreateWaveform("test")

    # Have to upload the waveform and triggers in parallel, one point at a time..
    for a, trig in zip(A, trig1):
        # wfm value, trigger1, trigger2, ?
        wf.AddSample(a, trig, True, 1)

    wf_read_length = wf.Length()
    HF_Measurement.SetHF_Mode(0, False)
    HF_Gain.SetValue(23)
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

    plt.plot(trig1, label='Sampling Trigger')
    plt.plot(t, Vmonitor, label='V monitor [?]')
    plt.plot(t, Iout, label='Current [?]')
    plt.plot(A, label='Input Waveform')
    plt.legend()

    fig, ax = plt.subplots()
    ax.plot(Vmonitor, Iout)

def trianglePulse(t_start=0, t_pulse = 10, v_pulse = .250, t_sep = 2):
    """Create a single triangle pulse
    t_pulse: width of pulse in nanoseconds
    v_pulse: height of pulse in volts
    """
    n_triggers = t_pulse/t_sep #Each trigger is 2 nS apart by default for TEO

    #A = np.concatenate(np.linspace(0,v_pulse,n_triggers),np.linspace(v_pulse, 0 , n_triggers))
    A1 = np.linspace(0,v_pulse, n_triggers)
    A2 = np.linspace(v_pulse, 0, n_triggers)[1:]

    A = np.concatenate([A1,A2])
    t = np.linspace(t_start,t_pulse, len(A))

    return A, t

def DCSweep(v_start=0, v_stop=1, t_pulse = 10, n_pulse=100, t_sep=2):
    """Create train of sawtooth pulses"""
    v_step = (v_stop-v_start)/n_pulse
    A = []
    t = []
    t_pulseStart = 0
    for n in range(n_pulse):
        A_p, t_p,  = trianglePulse(t_start=t_pulseStart, t_pulse = t_pulse,  v_pulse=v_start+n*v_step, t_sep=t_sep)
        t_pulseStart = t_pulseStart + n*t_pulse
        A = np.concatenate([A,A_p[1:]])
    t = np.linspace(0, t_pulse*n_pulse, len(A)) > 0

    return A, t

def DCSweepBipolar(v_start=0, v_stop=1, t_pulse = 10, n_pulse=100 ):
    """Create train of bipolar sawtooth pulses"""
    v_step = (v_stop-v_start)/n_pulse
    A = []
    t = []
    t_pulseStart = 0
    for n in range(n_pulse):
        A_p, t_p,  = trianglePulse(t_start=t_pulseStart, t_pulse = t_pulse,  v_pulse=v_start+n*v_step)
        A_p = np.concatenate([A_p,A_p[1:]*-1])
        t_pulseStart = t_pulseStart + n*t_pulse
        A = np.concatenate([A,A_p[1:]])
    t = np.linspace(0, t_pulse*n_pulse, len(A)) > 0

    return A, t


def setPulse(v_read=0.010, t_read1= .001, t_read2 = 0.001, v_set=0.020, t_set=0.001, t_v0 = 0.001, t_sep=1e-6, t_start=0, v_baseline=0):
    '''
    Create single read/set/read pulse
    :param v_read: Voltage to read, unit Volt
    :param t_read1: time of first read, unit Second
    :param t_read2: time of second read, unit Second
    :param v_set: Voltage of Set pulse, unit Volt
    :param t_set: time of Set pulse, unit Second
    :param t_v0: time of zero voltage between pulses
    :param t_sep: time between points, default to 2nS b/c TEO
    :return: Amplitude, trigger array
    '''
    t_total = t_v0+t_read1+t_set+t_read2
    n_triggers = t_total/t_sep

    A1 = np.linspace(v_baseline,v_baseline,int(t_v0/t_sep))
    A2 = np.linspace( v_read, v_read,int(t_read1/t_sep))
    A3 = np.linspace(v_set,v_set, int(t_set/t_sep))
    A4 = np.linspace(v_read, v_read,int(t_read2/t_sep))
    A = np.concatenate([A1,A2,A3,A4])

    t = np.linspace(t_start,t_start+t_total, len(A))

    return A, t

def setPulseSweep(v_read= 0.010, t_read1=0.001, t_read2=0.001,
                  v_setStart=0.010, v_setStop=1.0, t_pulse=10, t_v0= 0.001,
                  n_pulse=100, t_set=100e-6, v_baseline=0):
    '''
    Chain n set Pulses
    :param v_read: read voltage
    :param t_read: length of first read pulse
    :param t_read2: length of second readpulse
    :param v_setStart: set voltage to start
    :param v_setStop: set voltage to end
    :param t_pulse: length of set pulse
    :param n_pulse: total number of pulses
    :return: waveform
    '''
    #v_setStart = v_setStart * 1.082
    #v_setStop = v_setStop * 1.082
    v_step = (v_setStop - v_setStart) / n_pulse

    A = []
    t = []
    t_pulseStart = 0

    for n in range(n_pulse):
        if len(t)==0:
            t_start=0
        else:
            t_start = max(t)
        A_p, t_p, = setPulse(v_read = v_read, t_read1=t_read1, t_read2=t_read2,v_set = v_setStart+n*v_step, t_v0=t_v0, t_set=t_set, v_baseline=v_baseline, t_start=t_start)
        # A_p = np.concatenate([A_p, A_p[1:] * -1])
        t_pulseStart = t_pulseStart + n * t_pulse
        A = np.concatenate([A, A_p[1:]])
        t = np.concatenate([t, t_p[1:]])

    A = np.concatenate([A, [v_baseline]]) #make sure wavefrom ends on baseline
    dt = t[-1]-t[-2]
    t = np.concatenate([t, [t[-1] + dt]])  #this adds one additional time point, so A and t are equal length)
    return A, t

def KS_Resistance():
    v, c = MeasurementOTS.ZTest('Lk')
    return max(v)/max(c)

def Pulses_wDCRead(die, mod, dev):
    """measure R w/ keysight at 50mV,
        do pulses sweep 0.2 to 1V, in steps of 0.02
        remeasure R,
        do pulse sweep
        remeasure R
    """
    wfm = setPulseSweep(v_read=0.10, t_read1=0.10, t_read2=0.10, v_setStart=.2, v_setStop=1, t_pulse=20, t_set=0.10,
                        n_pulse=40, t_v0=0.10, v_baseline=0.1)[0]
    wfm2 = wfm[0::4]

    paX, _ = UtilsCollection.GetAuxInfoPlus()
    paX["General"] = UtilsCollection.GetGeneralInfo()
    tb = TEObox(paX, [])

    v1, c1 = MeasurementOTS.ZTest('Lk')
    DC1_arr = v1, c1
    R1 = max(v1) / max(c1)

    tb.SwitchTo('Pulsing')
    iv1 = picoiv(wfm2, die=die, mod=mod, dev=dev, nsamples=10000000, channels=['A', 'B', 'C', 'D'],
                autosmoothimate=100000, duration=1.26e-2, into50ohm=False)


    v2, c2 = MeasurementOTS.ZTest('Lk')
    DC2_arr = v2, c2
    R2 = max(v2)/max(c2)

    tb.SwitchTo('Pulsing')

    iv2 = picoiv(wfm2, die=die, mod=mod, dev=dev, nsamples=10000000, channels=['A', 'B', 'C', 'D'],
                autosmoothimate=100000, duration=1.26e-2, into50ohm=False)


    v3, c3 = MeasurementOTS.ZTest('Lk')
    DC3_arr = v3, c3
    R3 = max(v3)/max(c3)

    Rdict = {'R1':R1, 'R2': R2,'R3':R3}

    return iv1 , iv2, DC1_arr, DC2_arr, DC3_arr, Rdict
    #return iv1, DC1_arr, DC2_arr, DC3_arr, Rdict
