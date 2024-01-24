# Copyright (c) Jari Klinkmann 2023
import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__
import numpy as np
import matplotlib.pyplot as plt
import os
import atexit
from time import localtime, strftime, sleep
from typing import Final
from decimal import Decimal

"""
connect SaturnAWG
"""

from saturnpy.saturn_system import Saturn_System
from saturnpy.saturn_system_enum import Globaltrigger, Trigger_sources, System_state, Rhea_state
from saturnpy.helpers import send_ascii_file
from saturnpy import rhea

class SaturnAWG(Saturn_System):

    # Forward declaration of channel objects
    # (not strictly necessary, but makes it easier to work with, 
    #  especially editors which support function autocompletion etc.)
    S1M1: rhea.DA_module
    S1M1R1: rhea.DA_channel
    S1M1R2: rhea.DA_channel
    S1M1R3: rhea.DA_channel
    S1M1R4: rhea.DA_channel
    # TCP settings of Saturn Studio II
    # (Use 'localhost' if Saturn Studio II is running on the PC on which this script is executed.
    #  Use Saturn System IP address instead, if Saturn Studio II is running on the Saturn System.)
    ss2_host_ip ='192.168.10.5'
    ss2_host_port = 8081

    def __init__(self, verbose=False):

        # register in ivtools
        #statename = self.__class__.__name__
        #if statename not in ivtools.instrument_states:
        #    ivtools.instrument_states[statename] = {}
        #self.__dict__ = ivtools.instrument_states[statename]

        # init the saturn
        super().__init__(verbose = verbose)
        self.connect_S2(ip=self.ss2_host_ip, port=self.ss2_host_port )
        atexit.register(self.disconnect_S2)

        # Add modules and/or channels to system object
        # Rhea module
        self.S1M1: Final[rhea.DA_module] = self.add_DA_module('S1M1', samplerate=Decimal('1e9'))

        # Initialize RHEA DA-module
        # Initialization is done for all RHEA channels/modules at once
        self.S1M1.init()

        # Reset all channels to default config (off, term, on, 1V sine at 1MHz)
        self.S1M1R1.reset()
        self.S1M1R2.reset()
        self.S1M1R3.reset()
        self.S1M1R4.reset()
        self.S1M1.confirm(diff=False)

        # Read RHEA state
        print("RHEA state: ", self.S1M1.state)


# connect the necessary instruments

k = keith
awg = SaturnAWG()
rf_switches = RFswitches()

# define helper functions
"""
returns an array of that first ascend/descend from 0 to peak 
and then descend/ascend from preak to 0 with step as step size.
"""
def _tri (peak, step):
    flank = np.arange(0, peak, step)
    return np.concatenate([flank, [peak], flank[::-1]])
"""
this is the code to read device under test resistance on the
RF switches, AWG, PPG, PG5 setup. It works as follows:
    - RF switch for Keithley (output A) is turned ON
    - Keithley performs iv-sweeps
    - RF switch for Keithley (output A) is turned OFF
    - Keithley data is read and saved to dictionary
"""
def _iv_sweeps (
    no_sweeps,
    V_SET,
    V_RESET,
    V_step,
    nplc=1
):
    data = {}
    # turn on RF switch for keithley
    rf_switches.a_on()
    plt.pause(0.1)
    # perform i-v sweep to
    for i in range(no_sweeps):
        k._iv_lua(
            _tri(V_SET, V_step), Irange=1e-3, Ilimit=3e-4, 
            Plimit=V_SET*1e-3, nplc=nplc, Vrange=V_SET
        )
        while not k.done():
            sleep(0.01)
        data[f'SET_{i+1}'] = k.get_data()
        k._iv_lua(
            _tri(V_RESET, V_step), Irange=1e-2, Ilimit=1e-2, 
            Plimit=1, nplc=nplc, Vrange=V_RESET
        )
        while not k.done():
            sleep(0.01)
        data[f'RESET_{i+1}'] = k.get_data()
    plt.pause(0.1)
    # turn off RF switch for keithley
    rf_switches.a_off()
    
    return data
"""
this is the code to read device under test resistance on the
RF switches, AWG, PPG, PG5 setup. It works as follows:
    - RF switch for Keithley (output A) is turned ON
    - Keithley reads resistance
    - RF switch for Keithley (output A) is turned OFF
    - Keithley data is read and saved to dictionary
"""
def _read_resistance (
    V_read,
    V_step,
    V_range,
    I_range,
    I_limit,
    P_limit=0,
    nplc=1
):
    
    # turn on RF switch for keithley
    rf_switches.a_on()
    plt.pause(0.1)
    # perform i-v sweep to read resistance
    k._iv_lua(
        _tri(V_read, V_step), Irange=I_range, Ilimit=I_limit, 
        Plimit=P_limit, nplc=nplc, Vrange=V_range
    )
    while not k.done():
        sleep(0.01)
    data = k.get_data()
    plt.pause(0.1)
    # turn off RF switch for keithley
    rf_switches.a_off()
    
    return data
"""
returns two arrays time, voltage
the time is in seconds, voltage in volts
the signal is a square wave followed by a linear decrease
parameters:
- amplitude is the voltage of the square signal in volts
- max_time is the width of the square signal in seconds
- flank_time is the time in which the voltage linearly decreases
    from amplitude to 0 in seconds
- fade_out is extra time at the end 
    where the signal is 0 in seconds
"""
def _SET_signal (amplitude, max_time, flank_time, fade_out):
    if max_time!=0 and flank_time!=0:
        time = [
            0,
            max_time, 
            max_time+flank_time, 
            max_time+flank_time+fade_out
        ]
        voltage = [
            amplitude,
            amplitude, 
            0,
            0
        ]
    elif max_time==0 and flank_time!=0:
        time = [
            0,
            flank_time, 
            flank_time+fade_out
        ]
        voltage = [
            amplitude,
            0,
            0
        ]
    elif flank_time==0 and max_time!=0:
        time = [
            0, max_time, 
            max_time+1e-9,
            max_time+1e-9+fade_out
        ]
        voltage = [
            amplitude, amplitude, 
            0, 
            0
        ]
    else:
        raise Exception('Signal Form is currently not possible')
    return time, voltage
"""
returns two arrays time, voltage
the time is in seconds, voltage in volts
the signal is a square wave followed by a linear decrease
parameters:
- amplitude is the voltage of the square signal in volts
- length is the length of the square signal in seconds
- delay is extra time at the end 
    where the signal is 0 in seconds
"""
def _RESET_signal (amplitude, length, fade_out):
    time = [
        0, length, length+1e-9, length+1e-9+fade_out
    ]
    voltage = [
        amplitude, amplitude, 0, 0
    ]
    return time, voltage
"""
extract measurement data from the keithley
get resistance
"""
def _extract_keithley_measurement(k_data):
    return k_data["t"], k_data["Vmeasured"], k_data["I"]
def _get_resistance(V,I):
    idx = np.where(
        (np.abs(V)>=0.15)
    )[0]
    return np.mean(V[idx] / I[idx])

"""
this is the code for jari's pcm measurements.

The following wiring of inputs to the RF switches is assumed:
    - input A: Keithley
    - input B: AWG channel 1 (SET)
    - input C: AWG channel 2 (RESET)
Furthermore it is assumed that ground goes to Tektronix channel 3

The measurement works as follows:

1) INITIALIZATION STAGE
    - AWG channel 1 (input B) is initialized to give SET signal (rectangular followed by sawtooth on channel 1)
    - AWG channel 2 (input C) is initialized to give RESET signal (rectangular)
    - Keithley (input A) is initialized
    - Tektronix is initialized
    - ALL RF switches turned OFF

2) MEASUREMENT STAGE
    - a dictionary with all measurement settings and for all measurement data is begun

    - RF switch for Keithley (input A) is turned ON
    - perform defined number of iv-sweeps
    - RF switch for Keithley (input A) is turned OFF
    - Keithley data is read and saved to dictionary

    - RF switch for Keithley (input A) is turned ON
    - Keithley reads resistance
    - RF switch for Keithley (input A) is turned OFF
    - Keithley data is read and saved to dictionary

    - Tektronix is armed
    - RF switch for AWG channel 1 (input B) is turned ON
    - AWG SET is triggered
    - RF switch for AWG channel 1 (input B) is turned OFF
    - Tektronix data is read and saved to dictionary

    - RF switch for Keithley (input A) is turned ON
    - Keithley reads resistance
    - RF switch for Keithley (input A) is turned OFF
    - Keithley data is read and saved to dictionary

    - Tektronix is armed
    - RF switch for AWG channel 2 (input C) is turned ON
    - AWG RESET is triggered
    - RF switch for AWG (input C) is turned OFF
    - Tektronix data is read and saved to dictionary

    - RF switch for Keithley (input A) is turned ON
    - Keithley reads resistance
    - RF switch for Keithley (input A) is turned OFF
    - Keithley data is read and saved to dictionary

    - dictionary is written to file and saved
REPEAT

3) SHUTDOWN phase
    - all Tektronix channels are turned OFF and disarmed
    - all Keithley channels are turned OFF
    - all AWG channels are turned OFF
    - all RF switches are turned OFF
"""
def jari_pcm_measurement_2 (
    # total number of measurements
    total_measurements=1,

    # parameters for the documentation
    samplename='test',
    padname='jari_pcm_measurements_test',
    comments='',

    # parameters for AWG
    V_SET=1,
    t_SET_max=10e-9, 
    t_SET_flank=10e-9,
    t_RESET=10e-9,
    V_RESET=1,
    fade_out=10e-9, 
    
    # parameters for keithley read and sweep
    no_iv_sweeps=0,
    V_SET_sweep=1,
    V_RESET_sweep=-1.1,
    V_step_sweep=0.05,
    V_read=0.5,
    V_step=0.05,
    V_range=1,
    I_range=1e-3,
    I_limit=1e-3,
    P_limit=1e-3,
    nplc=1,

    # parameters for Tektronix
    ttx_attenuation = 0,
    channel = 1,
    trigger_level = 0.1,
    polarity = 1,
    recordlength = 5000,
    position = -2.5,
    scale = 0.5,
):
    # INITIALIZATION STAGE

    # set up Tektronix
    for i in range(1,5):
        if i==channel:
            ttx.inputstate(channel, True)
        else:
            ttx.inputstate(i, False)
    ttx.scale(channel, scale)
    ttx.position(channel, position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, recordlength)
    trigger_level = trigger_level*polarity
    # record any disturbances during setup
    ttx.arm(source = channel, level = trigger_level, edge = 'r') 
    plt.pause(0.1)

    # set up Keithley
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = True)
    k.source_level(source_val=V_read, source_func='v', ch='A')
    plt.pause(1)

    # set up AWG
    # turn off all channels and turn 1,2 to default settings
    awg.S1M1R1.reset()
    awg.S1M1R2.reset()
    awg.S1M1R1.on(False)
    awg.S1M1R2.on(False)
    awg.S1M1R3.on(False)
    awg.S1M1R4.on(False)
    # configure channel 1
    # Name and path for signal definition
    name_SET = "SET_signal"
    name_RESET = "RESET_signal"
    path = "c:\\rhea_demo\\"
    # Create signal definition
    signal_time_SET, signal_voltage_SET = _SET_signal(
        amplitude=V_SET, max_time=t_SET_max, flank_time=t_SET_flank, fade_out=fade_out
    )
    signal_time_RESET, signal_voltage_RESET = _RESET_signal(
        amplitude=V_RESET, length=t_RESET, fade_out=fade_out
    )
    signal_SET = rhea.DA_Signal(
        name_SET, path, signal_time_SET, 
        signal_voltage_SET, stepsize = '1n'
    )
    signal_RESET = rhea.DA_Signal(
        name_RESET, path, signal_time_RESET, 
        signal_voltage_RESET, stepsize = '1n'
    )
    # Create definition file
    rhea_ini_SET = signal_SET.create_file()
    rhea_ini_RESET = signal_RESET.create_file()
    # Send signal definitions file to remote Saturn System
    # This is only necessary if Saturn Studio II is running on the Saturn System.
    if (awg.ss2_host_ip != 'localhost'):
        send_ascii_file(awg.ss2_host_ip, source = rhea_ini_SET, target = rhea_ini_SET)
        send_ascii_file(awg.ss2_host_ip, source = rhea_ini_RESET, target = rhea_ini_RESET)
    # Initialize RHEA DA-module
    # Initialization is done for all RHEA channels/modules at once
    awg.S1M1.init()
    # Load waveform data to selected channel, do not combine with predefined waveforms
    awg.S1M1R1.load(rhea_ini_SET)
    awg.S1M1R2.load(rhea_ini_RESET)
    # set 50 ohm termination to true
    awg.S1M1R1.term(True)
    awg.S1M1R2.term(True)
    # set channel to trigger from GT1 trigger
    awg.S1M1R1.trigger([Trigger_sources.GT1], True)
    awg.S1M1R2.trigger([Trigger_sources.GT2], True)
    # Switch channel output on
    awg.S1M1R1.on(True)
    awg.S1M1R2.on(True)
    # confirm all changes
    awg.S1M1.confirm(diff=False)
    awg.wait_for_rhea(Rhea_state.READY, timeout_seconds=20)
    print("RHEA state after init: ", awg.S1M1.state)
    print("System state after init: ", awg.system_state)
    plt.pause(5)
    # an initial triggering is needed for the next trigger to work
    # somehow the device is ignoring the first time it is triggered
    print("Test trigger ALL: ", awg.manual_trigger([Globaltrigger.ALL]))
    
    # record any initial disturbances
    plt.pause(1)
    if ttx.triggerstate():
        plt.pause(0.1)
        ttx.disarm()
        print('setup finished.')
    else:
        data_scope = ttx.get_curve(channel)
        print('setup finished. an unexpected signal was recorded during setup.')
        plt.plot(data_scope['t_ttx'], data_scope['V_ttx'])
        plt.show()      
    ttx.disarm()
    plt.pause(10)

    # MEASUREMENT STAGE
    # perform measurements
    for measurement_no in range(1, total_measurements+1):

        # save measurement data and settings
        data = {}
        timestamp = strftime("%Y.%m.%d-%H.%M.%S", localtime())
        data['timestamp'] = timestamp
        data['samplename'] = samplename
        data['padname'] = padname
        data['comments'] = comments
        data['total_measurements'] = total_measurements
        data['measurement_no'] = measurement_no
        # values for Keithley
        data['no_iv_sweeps'] = no_iv_sweeps
        data['V_SET_sweep'] = V_SET_sweep
        data['V_RESET_sweep'] = V_RESET_sweep
        data['V_step_sweep'] = V_step_sweep
        data['V_read'] = V_read
        data['V_step'] = V_step
        data['V_range'] = V_range
        data['I_range'] = I_range
        data['I_limit'] = I_limit
        data['P_limit'] = P_limit
        data['nplc'] = nplc
        # values for Tektronix
        data['channel'] = channel
        data['trigger_level'] = trigger_level
        data['polarity'] = polarity
        data['position'] = position
        data['scale'] = scale
        data['recordlength'] = recordlength
        data['ttx_attenuation'] = ttx_attenuation
        # values for AWG
        data['V_SET'] = V_SET
        data['t_SET_max'] = t_SET_max
        data['t_SET_flank'] = t_SET_flank
        data['V_RESET'] = V_RESET
        data['t_RESET'] = t_RESET
        data['fade_out'] = fade_out 
        data['stepsize'] = 1e-9
        data['SET_signal'] = signal_time_SET, signal_voltage_SET
        data['RESET_signal'] = signal_time_RESET, signal_voltage_RESET

        #0: perform preliminary sweeps
        data['iv_sweeps'] = _iv_sweeps(
            no_sweeps=no_iv_sweeps, V_SET=V_SET_sweep,
            V_RESET=V_RESET_sweep, V_step=V_step_sweep, nplc=nplc
        )

        #1: read resistance before SET with Keithley
        data['initial_resistance_measurement'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #2: perform SET with AWG channel 1 and measure with Tektronix
        rf_switches.b_on()
        ttx.arm(source = channel, level = trigger_level, edge = 'r') 
        plt.pause(0.5)
        data['SET_trigger'] = awg.manual_trigger([Globaltrigger.GT1])
        plt.pause(0.3)
        if ttx.triggerstate():
            plt.pause(0.1)
            ttx.disarm()
            data['t_set'] = None
            data['v_set'] = None
            print('no signal was found for SET')
        else:
            data_scope = ttx.get_curve(channel)
            data['t_set'] = data_scope['t_ttx']
            data['v_set'] = data_scope['V_ttx']
            plt.plot(data['t_set'], data['v_set'], label='SET')
        ttx.disarm()
        rf_switches.b_off()

        #3: read resistance after SET and before RESET with Keithley
        data['middle_resistance_measurement'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #4: perform RESET with PG5 and measure with Tektronix
        rf_switches.c_on()
        ttx.arm(source = channel, level = trigger_level, edge = 'r') 
        plt.pause(0.5)
        data['RESET_trigger'] = awg.manual_trigger([Globaltrigger.GT2])
        plt.pause(0.3)
        if ttx.triggerstate():
            plt.pause(0.1)
            ttx.disarm()
            data['t_reset'] = None
            data['v_reset'] = None
            print('no signal was found for RESET')
        else:
            data_scope = ttx.get_curve(channel)
            data['t_reset'] = data_scope['t_ttx']
            data['v_reset'] = data_scope['V_ttx']
            plt.plot(data['t_reset'], data['v_reset'], label='RESET')
        ttx.disarm()
        rf_switches.c_off()

        #5: read resistance after RESET with Keithley
        data['end_resistance_measurement'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        # print some summary infos about measurement
        t_ini, V_ini, I_ini = _extract_keithley_measurement(data["initial_resistance_measurement"])
        t_mid, V_mid, I_mid = _extract_keithley_measurement(data["middle_resistance_measurement"])
        t_end, V_end, I_end = _extract_keithley_measurement(data["end_resistance_measurement"])
        R_ini = _get_resistance(V=V_ini, I=I_ini)
        R_mid = _get_resistance(V=V_mid, I=I_mid)
        R_end = _get_resistance(V=V_end, I=I_end)
        print(f"round {measurement_no}: R_ini={R_ini/1e3:.1f}kΩ, R_mid={R_mid/1e3:.1f}kΩ, R_end={R_end/1e3:.1f}kΩ")

        # write measurement data to file
        date = strftime("%Y.%m.%d", localtime())
        datafolder = os.path.join('C:\\Messdaten', padname, date)
        i=1
        filepath = os.path.join(datafolder, f"{timestamp}_{samplename}.s")
        while os.path.isfile(filepath):
            filepath = os.path.join(datafolder, f"{timestamp}_{samplename}_{i}.s")
            i+=1
        io.write_pandas_pickle(meta.attach(data), filepath)

    # SHUTDOWN PHASE

    # shut down Keithley
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = False)

    # shut down Tektronix
    ttx.disarm()
    ttx.inputstate(channel, False)  

    # shut down AWG
    # awg.S1M1R1.reset()
    # awg.S1M1.confirm(diff=False)

"""
this is the code for jari's pcm measurements.

The following wiring of inputs to the RF switches is assumed:
    - input A: Keithley
    - input B: AWG
    - input C: Sympuls PG5
Furthermore it is assumed that ground goes to Tektronix channel 3

The measurement works as follows:

1) INITIALIZATION STAGE
    - AWG (input B) is initialized to give a SET signal (rectangular followed by sawtooth on channel 1)
    - Keithley (input A) is initialized
    - Sympuls PG5 (input C) is initiaized
    - Tektronix is initialized
    - ALL RF switches turned OFF

2) MEASUREMENT STAGE
    - a dictionary with all measurement settings and for all measurement data is begun

    - RF switch for Keithley (input A) is turned ON
    - Keithley reads resistance
    - RF switch for Keithley (input A) is turned OFF
    - Keithley data is read and saved to dictionary

    - Tektronix is armed
    - RF switch for AWG (input B) is turned ON
    - AWG SET is triggered
    - RF switch for AWG (input B) is turned OFF
    - Tektronix data is read and saved to dictionary

    - RF switch for Keithley (input A) is turned ON
    - Keithley reads resistance
    - RF switch for Keithley (input B) is turned OFF
    - Keithley data is read and saved to dictionary

    - Tektronix is armed
    - RF switch for PG5 (input C) is turned ON
    - PG5 RESET is triggered
    - RF switch for PG5 (input C) is turned OFF
    - Tektronix data is read and saved to dictionary

    - RF switch for Keithley (input A) is turned ON
    - Keithley reads resistance
    - RF switch for Keithley (input A) is turned OFF
    - Keithley data is read and saved to dictionary

    - dictionary is written to file and saved
REPEAT

3) SHUTDOWN phase
    - all Tektronix channels are turned OFF and disarmed
    - all Keithley channels are turned OFF
    - all AWG channels are turned OFF
    - all RF switches are turned OFF
"""
def jari_pcm_measurement (
    # total number of measurements
    total_measurements=1,

    # parameters for the documentation
    samplename='test',
    padname='jari_pcm_measurements_test',
    comments='',

    # parameters for pg5
    pg5_attenuation=0,
    pg5_pulse_width=10e-9,

    # parameters for AWG
    V_SET=1,
    t_SET_max=10e-9, 
    t_SET_flank=10e-9, 
    fade_out=10e-9, 
    
    # parameters for keithley
    V_read=0.5,
    V_step=0.05,
    V_range=1,
    I_range=1e-3,
    I_limit=1e-3,
    P_limit=1e-3,
    nplc=1,

    # parameters for Tektronix
    ttx_attenuation = 0,
    channel = 1,
    trigger_level = 0.1,
    polarity = 1,
    recordlength = 5000,
    position = -2.5,
    scale = 0.5,
):

    # INITIALIZATION STAGE

    # set up Tektronix
    for i in range(1,5):
        if i==channel:
            ttx.inputstate(channel, True)
        else:
            ttx.inputstate(i, False)
    ttx.scale(channel, scale)
    ttx.position(channel, position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, recordlength)
    trigger_level = trigger_level*polarity
    # record any disturbances during setup
    ttx.arm(source = channel, level = trigger_level, edge = 'r') 
    plt.pause(0.1)

    # set up Keithley
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = True)
    k.source_level(source_val=V_read, source_func='v', ch='A')
    plt.pause(1)

    # set up sympuls
    sympuls.set_pulse_width(pg5_pulse_width)

    # set up AWG
    # turn off all channels and turn 1 to default settings
    awg.S1M1R1.reset()
    awg.S1M1R1.on(False)
    awg.S1M1R2.on(False)
    awg.S1M1R3.on(False)
    awg.S1M1R4.on(False)
    # configure channel 1
    # Name and path for signal definition
    name = "rhea_demo"
    path = "c:\\rhea_demo\\"
    # Create signal definition
    signal_time, signal_voltage = _SET_signal(
        amplitude=V_SET, max_time=t_SET_max, flank_time=t_SET_flank, fade_out=fade_out
    )
    signal = rhea.DA_Signal(
        name, path, signal_time, 
        signal_voltage, stepsize = '1n'
    )
    # Create definition file
    rhea_ini = signal.create_file()
    # Send signal definitions file to remote Saturn System
    # This is only necessary if Saturn Studio II is running on the Saturn System.
    if (awg.ss2_host_ip != 'localhost'):
        send_ascii_file(awg.ss2_host_ip, source = rhea_ini, target = rhea_ini)
    # Initialize RHEA DA-module
    # Initialization is done for all RHEA channels/modules at once
    awg.S1M1.init()
    # Load waveform data to selected channel, do not combine with predefined waveforms
    awg.S1M1R1.load(rhea_ini)
    # set 50 ohm termination to true
    awg.S1M1R1.term(True)
    # set channel to trigger from GT1 trigger
    awg.S1M1R1.trigger([Trigger_sources.GT1], True)
    # Switch channel output on
    awg.S1M1R1.on(True)
    # confirm all changes
    awg.S1M1.confirm(diff=False)
    awg.wait_for_rhea(Rhea_state.READY, timeout_seconds=20)
    print("RHEA state after init: ", awg.S1M1.state)
    print("System state after init: ", awg.system_state)
    plt.pause(5)
    # an initial triggering is needed for the next trigger to work
    # somehow the device is ignoring the first time it is triggered
    print("Test trigger ALL: ", awg.manual_trigger([Globaltrigger.ALL]))
    
    # record any initial disturbances
    plt.pause(1)
    if ttx.triggerstate():
        plt.pause(0.1)
        ttx.disarm()
        print('setup finished.')
    else:
        data_scope = ttx.get_curve(channel)
        print('setup finished. an unexpected signal was recorded during setup.')
        plt.plot(data_scope['t_ttx'], data_scope['V_ttx'])
        plt.show()      
    ttx.disarm()
    plt.pause(10)

    # MEASUREMENT STAGE
    # perform measurements
    for measurement_no in range(1, total_measurements+1):

        # save measurement data and settings
        data = {}
        timestamp = strftime("%Y.%m.%d-%H.%M.%S", localtime())
        data['timestamp'] = timestamp
        data['samplename'] = samplename
        data['padname'] = padname
        data['comments'] = comments
        data['total_measurements'] = total_measurements
        data['measurement_no'] = measurement_no
        # values for Sympuls
        data['pg5_attenuation'] = pg5_attenuation
        data['pg5_pulse_width'] = pg5_pulse_width
        # values for Keithley
        data['V_read'] = V_read
        data['V_step'] = V_step
        data['V_range'] = V_range
        data['I_range'] = I_range
        data['I_limit'] = I_limit
        data['P_limit'] = P_limit
        data['nplc'] = nplc
        # values for Tektronix
        data['channel'] = channel
        data['trigger_level'] = trigger_level
        data['polarity'] = polarity
        data['position'] = position
        data['scale'] = scale
        data['recordlength'] = recordlength
        data['ttx_attenuation'] = ttx_attenuation
        # values for AWG
        data['V_SET'] = V_SET
        data['t_SET_max'] = t_SET_max
        data['t_SET_flank'] = t_SET_flank
        data['fade_out'] = fade_out 
        data['stepsize'] = 1e-9
        data['awg_signal'] = signal_time, signal_voltage

        #1: read resistance before SET with Keithley
        data['initial_resistance_measurement'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #2: perform SET with AWG and measure with Tektronix
        rf_switches.b_on()
        ttx.arm(source = channel, level = trigger_level, edge = 'r') 
        plt.pause(0.5)
        data['awg_trigger'] = awg.manual_trigger([Globaltrigger.GT1])
        plt.pause(0.3)
        if ttx.triggerstate():
            plt.pause(0.1)
            ttx.disarm()
            data['t_set'] = None
            data['v_set'] = None
            print('no signal was found for awg')
        else:
            data_scope = ttx.get_curve(channel)
            data['t_set'] = data_scope['t_ttx']
            data['v_set'] = data_scope['V_ttx']
            plt.plot(data['t_set'], data['v_set'], label='awg')
        ttx.disarm()

        rf_switches.b_off()

        #3: read resistance after SET and before RESET with Keithley
        data['middle_resistance_measurement'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #4: perform RESET with PG5 and measure with Tektronix
        rf_switches.c_on()
        ttx.arm(source = channel, level = trigger_level, edge = 'r') 
        plt.pause(0.5)
        sympuls.trigger()
        plt.pause(0.3)
        if ttx.triggerstate():
            plt.pause(0.1)
            ttx.disarm()
            data['t_reset'] = None
            data['v_reset'] = None
            print('no signal was found for pg5')
        else:
            data_scope = ttx.get_curve(channel)
            data['t_reset'] = data_scope['t_ttx']
            data['v_reset'] = data_scope['V_ttx']
            plt.plot(data['t_reset'], data['v_reset'], label='pg5')
        ttx.disarm()
        rf_switches.c_off()

        #5: read resistance after RESET with Keithley
        data['end_resistance_measurement'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        # print some summary infos about measurement
        t_ini, V_ini, I_ini = _extract_keithley_measurement(data["initial_resistance_measurement"])
        t_mid, V_mid, I_mid = _extract_keithley_measurement(data["middle_resistance_measurement"])
        t_end, V_end, I_end = _extract_keithley_measurement(data["end_resistance_measurement"])
        R_ini = _get_resistance(V=V_ini, I=I_ini)
        R_mid = _get_resistance(V=V_mid, I=I_mid)
        R_end = _get_resistance(V=V_end, I=I_end)
        print(f"round {measurement_no}: R_ini={R_ini/1e3:.1f}kΩ, R_mid={R_mid/1e3:.1f}kΩ, R_end={R_end/1e3:.1f}kΩ")

        # write measurement data to file
        date = strftime("%Y.%m.%d", localtime())
        datafolder = os.path.join('C:\\Messdaten', padname, date)
        i=1
        filepath = os.path.join(datafolder, f"{timestamp}_{samplename}.s")
        while os.path.isfile(filepath):
            filepath = os.path.join(datafolder, f"{timestamp}_{samplename}_{i}.s")
            i+=1
        io.write_pandas_pickle(meta.attach(data), filepath)

    # SHUTDOWN PHASE

    # shut down Keithley
    k.source_output(ch = 'A', state = False)
    k.source_output(ch = 'B', state = False)

    # shut down Tektronix
    ttx.disarm()
    ttx.inputstate(channel, False)  

    # shut down AWG
    # awg.S1M1R1.reset()
    # awg.S1M1.confirm(diff=False)