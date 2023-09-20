# FF means Flip Flop not Flash Field
# apparently all signals are latched, so the chip has state
# resetFF() puts them all in a known state

import time
import socket

# bit position of each pin
# NOT THE ARDUINO PIN!
block1 = 17  # FIO0
block2 = 16  # FIO1
block3 = 15  # FIO2
slset1 = 14  # FIO3
slset2 = 13  # FIO4
slreset = 12 # FIO5
wlset = 11   # FIO6
wlreset = 10 # FIO7
s0 = 9       # EIO0
s1 = 8       # EIO1
s2 = 7       # EIO2
s3 = 6       # EIO3
s4 = 5       # EIO4
s5 = 4       # EIO5
s6 = 3       # EIO6
s7 = 2       # EIO7
s8 = 1       # CIO0
s9 = 0       # CIO1

sPort = [s1, s2, s3, s4, s5, s6, s7, s8, s9]

class MAD200_Wifi():
    """
    First power up the wifi board and connect to the wifi network "Tasty Corn" (WEP password in arduino_secrets.h)
    Then, this will communicate with the MAD200 chip over a TCP socket on port 1337
    It's not particularly fast, but should get the job done.
    """
    def __init__(self):
        self.connect()
        self.debug = False
        self.nbytes = 3
        self.data = 0b00000000_00000000_00000000

    def send_data(self, databinary):
        """ databinary can also be int"""
        databytes = databinary.to_bytes(self.nbytes, 'big') # big or little?
        self.conn.send(databytes)
        # except TimeoutError, ConnectionResetError, WinError
        self.conn.send(b'\n')

    def setDIOs(self, bits):
        for bit in bits:
            self.data |= 1 << bit
        self.send_data(self.data)

    def resetDIOs(self, bits):
        for bit in bits:
            self.data &= 2**(8*self.nbytes) - 1 - (1 << bit)
        return self.send_data(self.data)

    def resetFF(self):
        self._resetAllDIOs()
        for dev in range(512):
            bits = [block1, block2, block3, slreset, wlreset, s0] + [sPort[k] for k in range(9) if (dev >> k) % 2]
            if self.debug: print(bits)
            self.setDIOs(bits)
            self.resetDIOs(bits)
        self._resetAllDIOs()
        return True

    def debugFF(self):
        self._resetAllDIOs()
        for dev in range(512):
            bits = [block1, s0] + [sPort[k] for k in range(9) if (dev >> k) % 2]
            if self.debug: print(bits)
            self.setDIOs(bits)
            #time.sleep(0.001)
            self.resetDIOs(bits)
        self._resetAllDIOs()
        return True

    def setFF(self, dev):
        bits = [block1, slset1, wlset] + [sPort[k] for k in range(9) if (dev >> k) % 2]
        self.setDIOs(bits)
        time.sleep(0.1)
        self.setDIOs([s0])
        time.sleep(0.1)
        print('what')
        self.resetDIOs([s0]+bits)
        # TODO: return block1 to its prior state?

    def setFF2(self, dev, bl):
        bits = [block2, slset2] + [sPort[k] for k in range(9) if (dev >> k) % 2]
        self.setDIOs(bits)
        self.setDIOs([s0])
        self.resetDIOs([s0] + bits)
        bits = [block2, wlset] + [sPort[k] for k in range(9) if ((31-bl) >> k) % 2]
        self.setDIOs(bits)
        self.setDIOs([s0])
        self.resetDIOs([s0] + bits)

    def _status(self):
        return 0

    def connect(self, host='192.168.4.1', port=1337):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        self.conn = s
        return s

    def close(self):
        if hasattr(self.conn, 'close'):
            self.conn.close()

    def _resetAllDIOs(self):
        self.data = 0
        self.send_data(self.data)