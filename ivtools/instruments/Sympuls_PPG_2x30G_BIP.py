# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:31:22 2023

@author: klinkmann
"""

import math
import time
import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__

class Sympuls_PPG_2x30G_BIP(object):    
    def __init__(self, addr='ASRL5::INSTR', debug=False):
        
        self.name = "Sympuls_PPG_2x30G_BIP"
        self.past_errors = ""
        self.debug = debug
        
        try:
            self.connect(addr)
        except:
            log.error('Sympuls connection failed at {}'.format(addr))

    def connect(self, addr):
        self.conn = visa_rm.get_instrument(addr)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.query = self.conn.query
        self.ask = self.query
        self.read = self.conn.read
        self.read_raw = self.conn.read_raw
        self.close = self.conn.close
        self.conn.write_termination ='\n'

    def idn(self):
        idn = self.query('*IDN?')
        self.read()   # read necessary to avoid empty line issue
        return idn.replace('\n', '')
    
    def get_error_log(self):
        print(self.past_errors)

    def error(self):
        '''prints the last error'''
        error_msg = self.query(':SYST:ERR:NEXT?')
        log.info(error_msg)
        self.write('*CLS')
        
        new_error = time.asctime() + ": " + error_msg
        self.past_errors += new_error
        return new_error
        
        
    # TODO: fix set_trigger_type (doesnt do anything right now, because commands dont do anything => ask company)

    # def set_trigger_type(self, type):
    #     '''sets the trigger type:
    #     type = \'IMM\' for internal clock
    #     type = \'TTL\' for external triggering
    #     tpye = \'MANUAL\' for manual triggering'''
    #     if type is 'IMM':
    #         self.write(':TRIG:SOUR IMM')
    #     elif type is 'TTL':
    #         self.write(':TRIG:SOUR TTL')
    #     elif type is 'MANUAL':
    #         self.write(':TRIG:SOUR MANUAL')
    #     else:
    #         log.info('Unknown trigger type. Make sure it is \'IMM\', \'TTL\' or \'MANUAl\'')
    
    def set_pattern(self, databytes):
        
        if not isinstance(databytes, bytes):
            # get the number of digits of our binary number
            # we subtract two because of the leading "0b"
            length = len(str(bin(databytes))) - 2 
            
            # obtain number of bytes necessary to store pattern
            num_of_bytes = math.ceil(length / 8)
            
            databytes = databytes.to_bytes(num_of_bytes, byteorder='big')
        
        back = databytes
        
        for i in range (-16, 256):
            
            databytes = back
            
            length = len(databytes)+i
            if length < 0: continue
            no_digits_length = len(str(length))
            databytes += bytes('\n', 'utf-8')
            
            cmd = f":SOURce1:LUPattern #{no_digits_length}{length}"
            
            self.conn.write_binary_values(cmd, databytes)
            time.sleep(0.02)
            if self.debug:
                if not "-160" in self.error():
                    print(f"length = {length}", f"i = {i}", databytes, no_digits_length, length, flush=True)
        
    def sendBlockDataCmd (self, cmd: str, databytes):
        
        l = len(databytes)
        ba = bytearray(cmd, 'utf-8')
        ba += databytes
        ba += bytes('\n', 'utf-8')
        
        self.conn.write

    def set_pulse_width(self, pulse_width):
        """
        sets the pulse width
        
        :pulse_width: expects pulse width in seconds
        betweeen 50 ps and 250 ps
        """
        if pulse_width < 50e-12 or pulse_width > 1000e-12:
            raise Exception("pulse width should be between 50ps and 250ps, "
                            f"but got {pulse_width*1e12:e} ps")
        
        self.write(f':SOUR:FREQ {1 / (2*pulse_width)}Hz')
        if self.debug:
            self.error()

    def set_period(self, period):
        """
        sets the period
        
        :period: expects period width in seconds
        between 100ps and 500 ps
        """
        if period < 100e-12 or period > 500e-12:
            raise Exception("pulse width should be between 100ps and 500ps, "
                            f"but got {period*1e12:e} ps")
            
        self.write(f':SOUR:FREQ {1/period}Hz')
        if self.debug:
            self.error() 
        
    def set_trigger_source_internal(self):
        '''sets the trigger source to internal'''
        self.write(':TRIG:SOUR INT')
        if self.debug:
            self.error()

    def trigger_single(self, repetitions=1):
        '''Executes a pulse'''
        self.write(":TRIGger2:SOURce IMMediate")
        self.write(f":SOURce:REPetition {repetitions}")
        self.write(":INIT:IMMediate")
        if self.debug: 
            self.error()
            
    def trigger_auto(self):
        '''Executes continuous pulse'''
        self.write(":TRIGger2:SOURce AUTO")
        if self.debug: 
            self.error()
            
    
    
    def reset(self):
        '''resets the device to factory default'''
        self.write('*RST')
        if self.debug: 
            self.error()
            
    def execute(self, command):
        self.write(command)
        self.error()
