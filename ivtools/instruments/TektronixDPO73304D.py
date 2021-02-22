import numpy as np
import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__

class TektronixDPO73304D(object):
    def __init__(self, addr='GPIB0::1::INSTR'):
        try:
            self.connect(addr)
        except:
            log.error('TektronixDPO73304D connection failed at {}'.format(addr))

    def connect(self, addr):
        self.conn = visa_rm.get_instrument(addr)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.query = self.conn.query
        self.ask = self.query
        self.read = self.conn.read
        self.read_raw = self.conn.read_raw
        self.close = self.conn.close

    def idn(self):
        return self.query('*IDN?').replace('\n', '')

    def bandwidth(self, channel=1, bandwidth=33e9):
        self.write('CH' + str(channel) + ':BAN ' + str(bandwidth))

    def scale(self, channel=1, scale=0.0625):
        self.write('CH' + str(channel) + ':SCAle ' + str(scale))

    def position(self, channel=1, position=0):
        self.write('CH'+str(channel)+':POS '+str(position))

    def inputstate(self, channel=1, mode=True):
        if mode:
            self.write('SELECT:CH' + str(channel) + ' ON')
        else:
            self.write('SELECT:CH' + str(channel) + ' OFF')

    def offset(self, channel=1, offset=0):
        self.write('CH' + str(channel) + ':OFFSet ' + str(offset))

    # TODO: Should be two separate functions
    def change_div_and_samplerate(self, division, samplerate):
        self.write('HORIZONTAL:MODE AUTO')
        self.write('HORIZONTAL:MODE:SAMPLERATE ' + str(samplerate))
        self.write('HOR:MODE:SCA ' + str(division))
        self.write('HORIZONTAL:MODE:AUTO:LIMIT 10000')

    def recordlength(self, recordlength=1e5):
        self.write('HORIZONTAL:MODE MANUAL')
        self.write('HORIZONTAL:MODE:RECORDLENGTH ' + str(recordlength))
        self.write('HORIZONTAL:MODE:AUTO:LIMIT ' + str(recordlength))

    # TODO: Should be two separate functions
    def change_samplerate_and_recordlength(self, samplerate=100e9, recordlength=1e5):
        self.write('HORIZONTAL:MODE MANUAL')
        self.write('HORIZONTAL:MODE:SAMPLERATE ' + str(samplerate))
        self.write('HORIZONTAL:MODE:RECORDLENGTH ' + str(recordlength))
        self.write('HORIZONTAL:MODE:AUTO:LIMIT ' + str(recordlength))
        self.write('DATA:STOP ' + str(recordlength))

    def ext_db_attenuation(self, channel=1, attenuation=0):
        self.write('CH' + str(channel) + ':PROBEFUNC:EXTDBATTEN ' + str(attenuation))

    def trigger(self):
        self.write('TRIGger FORCe')

    def arm(self, source=1, level=-0.1, edge='e'):
        if source == 0:
            self.write('TRIG:A:EDGE:SOUrce AUX')
        else:
            self.write('TRIG:A:EDGE:SOUrce CH ' + str(source))
        self.write('TRIG:A:LEVEL ' + str(level))
        self.write('ACQ:STOPA SEQUENCE')
        self.write('ACQ:STATE 1')
        if edge == 'r':
            self.write('TRIG:A:EDGE:SLO RIS')
        elif edge == 'f':
            self.write('TRIG:A:EDGE:SLO FALL')
        else:
            self.write('TRIG:A:EDGE:SLO EIT')
        triggerstate = self.query('TRIG:STATE?')
        while 'REA' not in triggerstate or 'SAVE' in triggerstate:
            self.write('ACQ:STATE 1')
            triggerstate = self.query('TRIG:STATE?')

    def get_curve(self, channel=1):
        self.write('HEAD 0')
        self.write('WFMOUTPRE:BYT_NR 1')
        self.write('WFMOUTPRE:BIT_NR 8')
        self.write('DATA:ENC RPB')
        self.write('DATA:SOURCE CH' + str(channel))
        rl = int(self.query('HOR:RECO?'))

        pre = self.query('WFMOutpre?')
        pre_split = pre.split(';')
        if len(pre_split) == 5:
            log.warning('Channel ' + str(channel) + ' is not used.')
            return None

        x_incr = float(pre_split[9])
        x_offset = int(pre_split[11])
        y_mult = float(pre_split[13])
        y_off = float(pre_split[14])

        self.write('DATA:STOP ' + str(rl))
        self.write('CURVE?')
        data_str = self.read_raw()
        data = np.fromstring(data_str[6:-1], np.uint8)

        time = x_incr * (np.arange(len(data)) - x_offset)
        voltage = y_mult * (data - y_off)

        return_dict = {}
        return_dict['t_ttx'] = time
        return_dict['V_ttx'] = voltage
        return return_dict

    def disarm(self):
        self.write('ACQ:STATE 0')

    def triggerstate(self):
        trigger_str = self.query('TRIG:STATE?')
        return trigger_str == 'READY\n'

    def trigger_position(self, position):
        self.write('HORIZONTAL:POSITION ' + str(position))
