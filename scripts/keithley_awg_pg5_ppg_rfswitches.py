# Copyright (c) Jari Klinkmann 2023
import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__
import numpy as np
import matplotlib.pyplot as plt
import os
from time import localtime, strftime, sleep
from saturnpy.saturn_system import Saturn_System
from saturnpy.saturn_system_enum import Globaltrigger, Trigger_sources, System_state, Rhea_state
from saturnpy import rhea
from saturnpy.helpers import send_ascii_file
from typing import Final
from decimal import Decimal

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
    total_measurements,

    # parameters for the documentation
    samplename,
    padname,

    # parameters for pg5
    pg5_attenuation,
    pg5_pulse_width,

    # parameters for AWG
    V_SET,
    t_SET_max, 
    t_SET_flank, 
    delay, 
    
    # parameters for keithley
    V_read,
    V_step,
    V_range,
    I_range,
    I_limit,
    P_limit=0,
    nplc=1,

    # parameters for Tektronix
    trigger_level = 0.025,
    polarity = 1,
    recordlength = 5000,
    position = -2.5,
    scale = 0.04,

):
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

        # perform i-v sweep to read resistance
        k._iv_lua(
            _tri(V_read, V_step), Irange=I_range, Ilimit=I_limit, 
            Plimit=P_limit, nplc=nplc, Vrange=V_range
        )
        while not k.done():
            sleep(0.01)
        data = k.get_data()

        # turn off RF switch for keithley
        rf_switches.a_off()
        
        return data
    """
    returns two arrays time, voltage
    the time is in seconds, voltage in volts
    the signal is a square wave followed by a linear decrease
    parameters:
    - amplitude is the voltage of the square signal
    - max_time is the width of the square signal
    - flank_time is the time in which the voltage linearly decreases
        from amplitude to 0
    - delay is extra time at the beginning and at the end 
        where the signal is 0
    - stepsize in seconds is the width of the square signals that the
        signal is composed of
    """
    def _SET_signal (amplitude, max_time, flank_time, delay, stepsize):
        time = [
            0, delay, 
            delay+stepsize, delay+stepsize+max_time,
            delay+stepsize+max_time+flank_time,
            2*delay+stepsize+max_time+flank_time
        ]
        voltage = [
            0, 0,
            amplitude, amplitude,
            0,
            0
        ]
        return time, voltage


    # INITIALIZATION STAGE 

    # set up Keithley
    k.source_output(ch = 'A', state = True)
    k.source_level(source_val=V_read, source_func='v', ch='A')
    plt.pause(1)

    # set up Tektronix
    ttx.inputstate(1, False)
    ttx.inputstate(2, False)
    ttx.inputstate(3, True)    
    ttx.inputstate(4, False)
    ttx.scale(3, scale)
    ttx.position(3, position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, recordlength)
    trigger_level = trigger_level*polarity

    # set up sympuls
    sympuls.set_pulse_width(pg5_pulse_width)

    # set up AWG
    # turn off all channels and turn 1 to default settings
    awg = saturnAWG()
    awg.S1M1R1.on(False)
    awg.S1M1R2.on(False)
    awg.S1M1R3.on(False)
    awg.S1M1R4.on(False)
    awg.S1M1R1.reset()
    # configure channel 1
    # set 50 ohm termination to true
    awg.S1M1R1.term(True)
    # set channel to trigger from GT1 trigger
    awg.S1M1R1.trigger([Trigger_sources.GT1], True)
    # Name and path for signal definition
    name = "rhea_demo"
    path = "c:\\rhea_demo\\"
    # Create signal definition
    signal_time, signal_voltage = _SET_signal(
        V_SET, t_SET_max, t_SET_flank,
        delay, 1e-9
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
    # Read RHEA state
    print("RHEA state before init: ", awg.S1M1.state)
    # Load waveform data to selected channel, do not combine with predefined waveforms
    awg.S1M1R1.load(rhea_ini)
    # Switch channel output on
    awg.S1M1R1.on(True)
    # confirm all changes
    awg.S1M1.confirm(diff=False)
    awg.wait_for_rhea(Rhea_state.READY, timeout_seconds=20)
    print("RHEA state after init: ", awg.S1M1.state)
    print("System state after init: ", awg.system_state)
    
    # MEASUREMENT STAGE
    # perform measurements
    for measurement_no in range(1, total_measurements+1):

        # save measurement data and settings
        data = {}
        timestamp = strftime("%Y.%m.%d-%H.%M.%S", localtime())
        data['timestamp'] = timestamp
        data['samplename'] = samplename
        data['padname'] = padname
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
        data['trigger_level'] = trigger_level
        data['polarity'] = polarity
        data['position'] = position
        data['scale'] = scale
        data['recordlength'] = recordlength
        # values for AWG
        data['V_SET'] = V_SET
        data['t_SET_max'] = t_SET_max
        data['t_SET_flank'] = t_SET_flank
        data['delay'] = delay 
        data['stepsize'] = 1e-9

        #1: read resistance before SET with Keithley
        data['initial_resistance'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #2: perform SET with AWG and measure with Tektronix
        rf_switches.b_on()

        ttx.arm(source = 3, level = trigger_level, edge = 'r') 
        plt.pause(0.1)
        
        # AWG code
        awg.manual_trigger([Globaltrigger.GT1])

        plt.pause(0.2)
        if ttx.triggerstate():
            plt.pause(0.1)
            ttx.disarm()
            data['t_set'] = None
            data['v_set'] = None
        else:
            data_scope = ttx.get_curve(3)
            data['t_set'] = data_scope['t_ttx']
            data['v_set'] = data_scope['V_ttx']
        ttx.disarm()

        rf_switches.b_off()

        #3: read resistance after SET and before RESET with Keithley
        data['middle_resistance'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #4: perform RESET with PG5 and measure with Tektronix
        rf_switches.c_on()
        ttx.arm(source = 3, level = trigger_level, edge = 'r') 
        plt.pause(0.1)
        sympuls.trigger()
        plt.pause(0.2)
        if ttx.triggerstate():
            plt.pause(0.1)
            ttx.disarm()
            data['t_reset'] = None
            data['v_reset'] = None
        else:
            data_scope = ttx.get_curve(3)
            data['t_reset'] = data_scope['t_ttx']
            data['v_reset'] = data_scope['V_ttx']
        ttx.disarm()
        rf_switches.c_off()

        #5: read resistance after RESET with Keithley
        data['end_resistance'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

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
    ttx.inputstate(3, False)  

    # shut down AWG
    awg.S1M1R1.reset()
    awg.S1M1.confirm(diff=False)