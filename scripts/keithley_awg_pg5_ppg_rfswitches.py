# Copyright (c) Jari Klinkmann 2023
import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__

"""
this is the code to read device under test resistance on the
RF switches, AWG, PPG, PG5 setup. It works as follows:
    - RF switch for Keithley is turned ON
    - Keithley reads resistance
    - RF switch for Keithley is turned OFF
    - Keithley data is read and saved to dictionary
"""
def _read_resistance (
    
):
    ...


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
    # parameters for the documentation
    samplename,
    padname,

    # parameters for pg5
    pg5_attenuation,
    pg5_pulse_width,
    
    # parameters for keithley



):
    ...
    

    return 