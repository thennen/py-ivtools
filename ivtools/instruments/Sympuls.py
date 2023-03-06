import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__
import time
import scipy
class Sympuls(object):
    def __init__(self, addr='ASRL3::INSTR'):
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

    def set_pulse_width(self, pulse_width):
        '''sets the pulse width (between 50 and 250 ps)'''
        self.write(':PULS:WIDT ' + str(pulse_width))

    def set_period(self, period):
        '''sets the period  (between  1 and 1e6 µs)'''
        self.write(':PULS:PER ' + str(period))

    def trigger(self):
        '''Executes a pulse'''
        self.write(':INIT')
        #self.Trig_Man =  self.write(':TRIG:SOUR MANUAL')
    def trigger_manual(self, Trig_Type):
        '''Type of Trigger, Manual or IMM (Intern)'''
        if Trig_Type == 'IMM':
             IMM = self.write(':TRIG:SOUR IMM')
             Trig_Type = IMM
             return Trig_Type
        else:
             Manual = self.write(':TRIG:SOUR MANUAL')
             Trig_Type = Manual
             return Trig_Type
             

    def Apply_Burst(self, pulse_width, period, Number_of_pulses):
        '''Apply Burst of Pulses for specific Pulse width, Period and Number of Pulses'''
        self.set_pulse_width(pulse_width)
        self.set_period(period)
        time_executed = (Number_of_pulses ) *period
        print('Excecutaion TIme',time_executed)
        self.write(':TRIG:SOUR IMM')
        time.sleep(time_executed)
        self.write(':TRIG:SOUR MANUAL')

    def Apply_Burst_time(self, pulse_width, period, Number_of_pulses, wait_for_sequence = False):
        '''Apply Burst of Pulses for specific Pulse width, Period and Number of Pulses'''
        self.set_pulse_width(pulse_width)
        self.set_period(period)
        time_executed = (Number_of_pulses ) *period
    #    if wait_for_sequence:
        print('Excecutaion TIme',time_executed)
        self.write(':TRIG:SOUR IMM')
        if wait_for_sequence:
            time.sleep(time_executed)
            return time_executed
