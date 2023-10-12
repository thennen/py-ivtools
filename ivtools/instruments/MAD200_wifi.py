# FF means Flip Flop not Flash Field
# apparently all signals are latched, so the chip has state
# resetFF() puts them all in a known state

import time
import socket

# bit position of each pin
# NOT THE ARDUINO PIN! Those are defined in the firmware, and commented here
block1 = 17  # Arduino D3 , formerly Labjack FIO0
block2 = 16  # Arduino D4 , formerly Labjack FIO1
block3 = 15  # Arduino D5 , formerly Labjack FIO2
slset1 = 14  # Arduino D6 , formerly Labjack FIO3
slset2 = 13  # Arduino D7 , formerly Labjack FIO4
slreset = 12 # Arduino D8 , formerly Labjack FIO5
wlset = 11   # Arduino D9 , formerly Labjack FIO6
wlreset = 10 # Arduino D10, formerly Labjack FIO7
s0 = 9       # Arduino D0 , formerly Labjack EIO0
s1 = 8       # Arduino D21, formerly Labjack EIO1
s2 = 7       # Arduino D20, formerly Labjack EIO2
s3 = 6       # Arduino D19, formerly Labjack EIO3
s4 = 5       # Arduino D18, formerly Labjack EIO4
s5 = 4       # Arduino D17, formerly Labjack EIO5
s6 = 3       # Arduino D16, formerly Labjack EIO6
s7 = 2       # Arduino D15, formerly Labjack EIO7
s8 = 1       # Arduino D2 , formerly Labjack CIO0
s9 = 0       # Arduino D1 , formerly Labjack CIO1

sPort = [s1, s2, s3, s4, s5, s6, s7, s8, s9]

class MAD200_Wifi():
    """
    First power up the wifi board and connect to the wifi network "Tasty Corn" (WEP password in arduino_secrets.h)
    Then, this will communicate with the MAD200 chip over a TCP socket on port 1337
    It's not particularly fast, but should get the job done.
    """
    def __init__(self, host='192.168.4.1', port=1337):
        self.connect(host, port)
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
        if (self.data >> block1) % 2: # If block1 already set, don't touch it (not sure if this is right)
            bits = [slset1, wlset] + [sPort[k] for k in range(9) if (dev >> k) % 2]
        else:
            bits = [block1, slset1, wlset] + [sPort[k] for k in range(9) if (dev >> k) % 2]
        self.setDIOs(bits)
        #time.sleep(0.1)
        self.setDIOs([s0])
        #time.sleep(0.1)
        #print('what')
        self.resetDIOs([s0]+bits)

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

# store pins in the class in case we don't have the module ...
MAD200_Wifi.block1 = block1
MAD200_Wifi.block2 = block2
MAD200_Wifi.block3 = block3
MAD200_Wifi.slset1 = slset1
MAD200_Wifi.slset2 = slset2
MAD200_Wifi.slreset = slreset
MAD200_Wifi.wlset = wlset
MAD200_Wifi.wlreset = wlreset
MAD200_Wifi.s0 = s0
MAD200_Wifi.s1 = s1
MAD200_Wifi.s2 = s2
MAD200_Wifi.s3 = s3
MAD200_Wifi.s4 = s4
MAD200_Wifi.s5 = s5
MAD200_Wifi.s6 = s6
MAD200_Wifi.s7 = s7
MAD200_Wifi.s8 = s8
MAD200_Wifi.s9 = s9