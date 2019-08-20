from scipy import signal
import time
import inspect

paX, _ = UtilsCollection.GetAuxInfoPlus()
paX["General"] = UtilsCollection.GetGeneralInfo()
tb = TEObox(paX, [])

print(1, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
from TEOModule import TEObox
import MeasurementOTS
import win32com.client
from win32com.client import CastTo, WithEvents, Dispatch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
formatter = mpl.ticker.EngFormatter()
import pandas as pd

print(2, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
plt.ion()
import UtilsCollection

# Triangular pulse to find volatile transition

# Parameters:
# Overall pulse duration T=300us
Ttotal = 300e-6
Npts = 1e3+1

print(Npts)
# Start voltage: Ustart = 10mV
Ustart = 10e-3
# Default increment voltage = 10mV
Uinc = 10e-3



# Measurement loop

currentVoltage = Ustart
currentCycle = 0



# q=quit, enter=increment and do measurement, spc = redo measurement, new value = new increment
while True:
    command = input('Current voltage = %s Increment = %s; RET runs %sV> ' % (currentVoltage, Uinc, currentVoltage + Uinc))
    if command == 'q':
        print('quitting')
        break
    elif command == '':
        currentVoltage += Uinc
        print('New voltage is %s' % currentVoltage)
    elif command[0] == '*':
        currentVoltage = float(command[1:]) * 1e-3
        print('New voltage is %s' % currentVoltage)
    elif command == "'":
        currentVoltage = 0.5
        print('New voltage is %s' % currentVoltage)
    elif command == ".":
        currentVoltage = 0.1
        print('New voltage is %s' % currentVoltage)
    elif command == ' ':
        print('Redoing last measurement')
        currentVoltage = currentVoltage  # spc = redo last voltage
    else: # try to see if we can convert to a number, in that case, use value as new increment (positive or negative)
        try:
            newIncrement = float(command)
            Uinc = newIncrement * 1e-3 # value entered in mV
            print('New increment is %s mV' % newIncrement)
            continue
        except:
            print('Could not convert value to float, try again %s ' % command)


    currentCycle += 1
    print(2.1, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    v2, c2 = MeasurementOTS.ZTest('Lk')
    print(2.2, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    DC2_arr = v2, c2
    #print(DC2_arr)
    R0 = max(v2)/max(c2) # why max / min? average?. should work either way: instead of fit

    tb.SwitchTo('Pulsing')


    # TODO: this has to be replaced with a triangular waveform


    A = signal.triang(Npts) * currentVoltage
    #A, t = trianglePulse(t_start=0, t_pulse=1e-3, v_pulse=.250, t_sep=2)


    print(3, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    iv1 = picoiv(A, die=die, mod=mod, dev=dev, nsamples=100000, channels=['A', 'B', 'C', 'D'],
                 autosmoothimate=100000, duration=Ttotal, into50ohm=False) # False gives correct voltage at the sample into 1M
    print(4, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)

    Umeas = np.array(iv1['V'])
    Imeas = np.array(iv1['I2']) # low range scale
    Imeas -= (Imeas[0] + Imeas[-1]) /2
    t = np.linspace(0, Ttotal, len(Umeas))

    print(5, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    v2, c2 = MeasurementOTS.ZTest('Lk')
    DC2_arr = v2, c2

    R1 = max(v2)/max(c2)
    print(6, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)

    #tb.SwitchTo('Pulsing')


    # TODO: create new figure
    # TODO: Title of figure
    # sampleID, currentCycle, currentVoltage, R0, R1

    #fig, ax = plt.subplots()
    print(7, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    fig = plt.figure()
    ax = plt.subplot(2,1,1)
    title = 'Die: {}, Mod: {}, Device: {}, Cycle: {}, Voltage: {}mV \nR0: {}, R1: {}, dR {}'.format(iv1['die'], iv1['mod'], iv1['dev'],
                                                    currentCycle, int(currentVoltage*1000), formatter(int(R0)), formatter(int(R1)), formatter(int(R1 - R0)))
    iv1['R0'] = R0
    iv1['R1'] = R1

    ax.set_title(title, fontsize=10)
    ax.annotate(iv1['filepath'].split('\\')[-1], (0.1,0.9), xycoords = 'axes fraction', fontsize = 5, color = 'y')
    print(8, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    # TODO: Plot vs time
    # 1. Applied Voltage U
    # 2. Measured Current I
    # 3. Resistance U/I
    # 4. mirror current values for first half of pulse from T=0 to T/2
    ax.plot(t, Umeas, label = 'U', c='k')
    ax2 = ax.twinx()
    ax2.plot(t, Imeas, label = 'I', c='b')
    print(9, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    R = np.array(Umeas) / np.array(Imeas)
    #fig, ax = subplots()

    print('Rmin, Rmax %s %s' % (min(R), max(R)))
    Rscaled = R / max(R) * max(Umeas)  # scale to U scale



    Iflipped = np.flipud(Imeas)

    posFlip = len(Umeas) // 2

    # TODO: normalize by time, print
    DIint = sum(Imeas[posFlip:][: -1] - Imeas[posFlip - 1::-1])

    print(10, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)
    print(len(t), len(t[posFlip:]), len(Iflipped), len(Iflipped[:posFlip ]))
    #ax2.plot(t[posFlip:], Iflipped[:posFlip ], c='g', linestyle = '--')
    print(len(t[posFlip:]), len(Imeas[posFlip -1 ::-1]))

    ax2.plot(t[posFlip:][:-1], Imeas[posFlip -1 ::-1], c='g', linestyle='--')

    ax2.axvline(t[posFlip], linestyle = '--')
    #ax.legend(frameon=False)
    #ax2.legend(frameon=False, loc=0)
    #fig.legend(loc=1)
    ax_R = plt.subplot(2, 1, 2, sharex=ax)
    ax_R.plot(t, R, label='R', c='r')
    Rwinsize = 5
    Rcenters = R[len(R) // 2 - Rwinsize: len(R) // 2 + Rwinsize]
    Rcenter = sum(Rcenters) / len(Rcenters)
    R1_4s = R[len(R) // 4 - Rwinsize: len(R) // 4 + Rwinsize]
    R3_4s = R[3*len(R) // 4 - Rwinsize: 3*len(R) // 4 + Rwinsize]
    R1_4 = sum(R1_4s) / len(R1_4s)
    R3_4 = sum(R3_4s) / len(R3_4s)
    #ax_R.set_ylim((min(R0, R1)/3), max(R0, R1)*3)

    ax_R.set_yscale("log", nonposy='mask')#, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
    #ax_R.set_ylim(0.5 * min(R0, R1, Rcenter), max(R0, R1, Rcenter) * 1.1)
    ax_R.set_ylim(0.5 * min(R1_4, R3_4, Rcenter), max(R1_4, R3_4, Rcenter) * 1.1)

    fig.savefig('\\'.join(iv1['filepath'].split('\\')[:-1]) + '\\' + title.replace(':', '_').replace(',', ' ').replace('\n', ' ') + '.png')
    s = pd.Series(iv1)
    s.to_pickle(path ='\\'.join(iv1['filepath'].split('\\')[:-1]) + '\\' + title.replace(':', '_').replace(',', ' ').replace('\n', ' ') + '.s')
    # files automatically saved? yes
    plt.pause(0.001)
    print(11, round(time.clock(), 2), inspect.getframeinfo(inspect.currentframe()).filename,
          inspect.currentframe().f_back.f_lineno)