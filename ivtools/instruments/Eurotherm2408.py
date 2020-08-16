import numpy as np
import serial
import time
import logging
import ivtools
log = logging.getLogger('instruments')

class Eurotherm2408(object):
    '''
    This uses some dumb proprietary EI-BISYNCH protocol over serial.
    Make the connections DB2 -> HF, DB3 -> HE, DB5 -> HD.
    You can also use modbus.
    '''
    def __init__(self, addr='COM32', gid=0, uid=1):
        # BORG
        statename = self.__class__.__name__
        if statename not in ivtools.instrument_states:
            ivtools.instrument_states[statename] = {}
        self.__dict__ = ivtools.instrument_states[statename]
        self.connect(addr, gid, uid)

    def connect(self, addr='COM32', gid=0, uid=1):
        if not self.connected():
            self.conn = serial.Serial(addr, timeout=1, bytesize=7, parity=serial.PARITY_EVEN)
            self.gid = gid
            self.uid = uid

    def connected(self):
        return hasattr(self, 'conn')

    def write_data(self, mnemonic, data):
        # Select
        # C1 C2 are the two characters of the mnemonic
        # [EOT] (GID) (GID) (UID) (UID) [STX] (CHAN) (C1) (C2) <DATA> [ETX] (BCC)
        from functools import reduce
        from operator import xor
        STX = '\x02'
        ETX = '\x03'
        EOT = '\x04'
        ENQ = '\x05'
        CHAN = '1'
        gid = str(self.gid)
        uid = str(self.uid)
        data = format(data, '.1f')
        bcc = chr(reduce(xor, (mnemonic + data + ETX).encode()))
        msg = EOT + gid + gid + uid + uid + STX + mnemonic + data + ETX + bcc
        log.debug(msg)
        # Clear the buffer in case there is some garbage in there for some reason
        # have recieved this reply before: b'\x18\x06'
        self.conn.read_all()
        self.conn.write(msg.encode())

        # Wait?
        time.sleep(.1)

        # Should reply
        # [NAK] - failed to write
        # [ACK] - successful write
        # (nothing) - oh shit
        ACK = '\x06'
        NAK = '\x15'
        reply = self.conn.read_all()
        log.debug(reply)
        if reply == ACK.encode():
            return True
        elif reply == NAK.encode():
            return False
        else:
            #raise Exception('Eurotherm not connected properly')
            # Sometimes the eurotherm actually got the message, but we failed to read the acknowledgement
            log.error('Trouble with Eurotherm communication (wrong/no acknowledgement)')

    def read_data(self, mnemonic, attempt=0):
        EOT = '\x04'
        ENQ = '\x05'
        gid = str(self.gid)
        uid = str(self.uid)
        # Poll
        # [EOT] (GID) (GID) (UID) (UID) (CHAN) (C1) (C2) [ENQ]
        # CHAN optional, will be returned only if sent
        poll = EOT + gid + gid + uid + uid + mnemonic + ENQ
        self.conn.write(poll.encode())

        # Wait?
        time.sleep(.1)

        # Reply
        # [STX] (CHAN) (C1) (C2) <DATA> [ETX] (BCC)
        reply = self.conn.read_all()
        log.debug(reply)
        try:
            return float(reply[3:-2])
        except:
            log.error('Failed to read Eurotherm 2408')
            # Just try again?
            # Sometimes there is a lot of noise on the serial line ???
            if attempt < 10:
                time.sleep(.1)
                return self.read_data(mnemonic, attempt+1)
            else:
                return np.nan

    def read_temp(self):
        return float(self.read_data('PV'))

    def set_temp(self, value):
        return self.write_data('SL', value)

    def output_level(self):
        return self.read_data('OP')

    def status(self):
        statusdict = {1: 'Reset',
                      2: 'Run',
                      3: 'Hold',
                      4: 'Holdback',
                      5: 'Complete'}
        return self.read_data('PC')
