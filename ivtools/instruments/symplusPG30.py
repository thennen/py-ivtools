import logging
import math
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__
import time
import scipy
class Sympuls(object):
    def __init__(self, addr='ASRL5::INSTR'):
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

    def error(self):
        '''prints the last error'''
        error_msg = self.query(':SYST:ERR:NEXT?')
        log.info(error_msg)

    def pattern(self, freq):
        '''Executes a pulse'''
        self.write(':SOURce:FREQ' + str(freq))
        #self.Trig_Man =  self.write(':TRIG:SOUR MANUAL')
