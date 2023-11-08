# Jari Klinkmann (c) 2023

import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__
import time
import scipy

class RFswitches (object):

    def __init__(self, addr='COM7'):
        try:
            self.connect(addr)
        except:
            log.error(f'RFswitches connection failed at {addr}')

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

    # turn all outputs ON/OFF
    def all_off (self):
        '''turn off all inputs'''
        self.write(':A0')
        self.write(':B0')
        self.write(':C0')

    # turn output A ON/OFF
    def a_off (self):
        '''turn off input A'''
        self.write(':A0')
    def a_on (self):
        '''turn on inputs A
	    and turn off B and C'''
	    self.write(':B0')
	    self.write(':C0')
        self.write(':A1')

    # turn output B ON/OFF
    def b_off (self):
        '''turn off input B'''
        self.write(':B0')
    def b_on (self):
        '''turn on input B
	    and turn off A and C'''
	    self.write(':A0')
	    self.write(':C0')
        self.write(':B1')

    # turn output C ON/OFF
    def c_off (self):
        '''turn off input C'''
        self.write(':C0')
    def c_on (self):
        '''turn on input C
	    and turn off A and B'''
	    self.write(':A0')
	    self.write(':B0')
        self.write(':C1')