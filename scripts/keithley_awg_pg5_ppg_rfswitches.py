# Copyright (c) Jari Klinkmann 2023
import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__
import numpy as np
import os
from time import localtime, strftime, sleep


"""
this is the code for jari's pcm measurements.
The measurement works as follows:

1) INITIALIZATION STAGE
    - AWG is initialized to give a SET signal (rectangular followed by sawtooth on channel 1)
    - Keithley is initialized
    - Tektronix is initialized
    - ALL RF switches turned OFF

2) MEASUREMENT STAGE
    - a dictionary with all measurement settings and for all measurement data is begun

    - RF switch for Keithley is turned ON
    - Keithley reads resistance
    - RF switch for Keithley is turned OFF
    - Keithley data is read and saved to dictionary

    - Tektronix is armed
    - RF switch for AWG is turned ON
    - AWG SET is triggered
    - RF switch for AWG is turned OFF
    - Tektronix data is read and saved to dictionary

    - RF switch for Keithley is turned ON
    - Keithley reads resistance
    - RF switch for Keithley is turned OFF
    - Keithley data is read and saved to dictionary

    - Tektronix is armed
    - RF switch for PG5 is turned ON
    - PG5 RESET is triggered
    - RF switch for PG5 is turned OFF
    - Tektronix data is read and saved to dictionary

    - RF switch for Keithley is turned ON
    - Keithley reads resistance
    - RF switch for Keithley is turned OFF
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
    
    # parameters for keithley
    V_read,
    V_step,
    V_range,
    I_range,
    I_limit,
    P_limit=0,
    nplc=1,

    # parameters for awg

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
        - RF switch for Keithley is turned ON
        - Keithley reads resistance
        - RF switch for Keithley is turned OFF
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
        
        # TODO: turn on RF switch for keithley
        ...

        # perform i-v sweep to read resistance
        k._iv_lua(
            _tri(V_read, V_step), Irange=I_range, Ilimit=I_limit, 
            Plimit=P_limit, nplc=nplc, Vrange=V_range
        )
        while not k.done():
            sleep(0.01)
        data = k.get_data()

        # TODO: turn off RF switch for keithley
        ...
        
        return data
    
    # define 
    
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
        data['pg5_attenuation'] = pg5_attenuation
        data['pg5_pulse_width'] = pg5_pulse_width
        data['V_read'] = V_read
        data['V_step'] = V_step
        data['V_range'] = V_range
        data['I_range'] = I_range
        data['I_limit'] = I_limit
        data['P_limit'] = P_limit
        data['nplc'] = nplc

        #1: read resistance before SET with Keithley
        data['initial_resistance'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #2: TODO: perform SET with AWG and measure with Tektronix

        #3: read resistance after SET and before RESET with Keithley
        data['middle_resistance'] = _read_resistance(
            V_read=V_read, V_step=V_step, V_range=V_range,
            I_range=I_range, I_limit=I_limit, P_limit=P_limit, nplc=nplc
        )

        #4: TODO: perform RESET with PG5 and measure with Tektronix

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

    # finish up measurement