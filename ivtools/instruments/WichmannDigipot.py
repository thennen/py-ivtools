import serial
import numpy as np
import time
import logging
import ivtools
from itertools import combinations_with_replacement
from serial.tools.list_ports import grep as comgrep
log = logging.getLogger('instruments')

class WichmannDigipot(object):
    '''
    Probing circuit developed by Erik Wichmann to provide remote series resistance switching
    There are two digipots on board.  You can use a single digipot or connect the two in series or in parallel
    There are 31 ~log spaced resistance values per digipot

    The firmware to use on the arduino is ReadASCIIString.ino

    TODO: Is there a way to poll the current state from the microcontroller?
    The class instance might not be aware of it when we first connect.
    '''
    def __init__(self, addr=None):
        # BORG
        statename = self.__class__.__name__
        if statename not in ivtools.instrument_states:
            ivtools.instrument_states[statename] = {}
        self.__dict__ = ivtools.instrument_states[statename]
        self.connect(addr)
        # map from setting to resistance -- needs to be measured by source meter
        # TODO: does the second digipot need a different calibration?
        #self.Rlist = [43080, 38366, 34242, 30547, 27261, 24315, 21719, 19385, 17313,
        #              15441, 13801, 12324, 11022, 8805, 7061, 5670, 4539, 3667, 2964,
        #              2416, 1965, 1596, 1313, 1089, 906, 716, 576, 478, 432, 384, 349,
        #              324, 306, 306]
        # Keithley calibration at 1V applied 2019-07-17 (prototype board)
        #self.Rlist = [43158.62, 38438.63, 34301.27, 30596.64, 27306.63, 24354.61, 21752.65,
        #              19413.07, 17336.84, 15461.77, 13818.91, 12338.34, 11033.65, 8812.41,
        #              7064.97, 5672.71, 4539.82, 3666.53, 2961.41, 2412.55, 1960.89, 1591.29,
        #              1307.28, 1083.48, 902.42, 711.69, 570.92, 472.24, 426.55, 377.22, 342.16,
        #              316.79, 299.09, 299.06]
        # Keithley calibration at 1V applied 2020-01-17 (red pcb rev3)

        # The wiper settings corresponding to these values are just range(34)
        self.Rlist = np.array([43157.6, 38446.63, 34301.13, 30599.28, 27314.15, 24380.41, 21780.81,
                      19442.55, 17365.22, 15492.7, 13840.67, 12353.05, 11048.16, 8837.83,
                      7072.3, 5662.19, 4526.77, 3654.42, 2951.77, 2407.68, 1953.7, 1583.89,
                      1308.5, 1086.92, 906.51, 715.82, 574.08, 476.34, 431.47, 380.62, 345.39,
                      320.04, 302.77, 302.83])

        self.Rcombinations = list(combinations_with_replacement(self.Rlist, 2))
        self.Rparlist = np.array([R1*R2/(R1+R2) for R1, R2 in self.Rcombinations])
        self.Rserlist = np.array([R1+R2 for R1, R2 in self.Rcombinations])


    def connect(self, addr=None):
        if not self.connected():
            if addr is None:
                # Connect to the first thing you see that has Leonardo in the description
                matches = list(comgrep('Leonardo'))
                if any(matches):
                    addr = matches[0].device
                else:
                    log.error('WichmannDigipot could not find Leonardo')
                    return
            self.conn = serial.Serial(addr, timeout=1)
            self.write = self.conn.write
            self.close = self.conn.close
            # We don't know what state we are in initially
            # For now we will just set them all to 1 when we connect
            self.bypass_state = 1
            self.wiper1state = 0
            self.wiper2state = 0
            self.write('0 0 1'.encode())
            if self.connected():
                log.info(f'Connected to digipot on {addr}')

    @property
    def Rstate(self, n=2):
        '''
        Return the current resistance state of the nth potentiometer
        '''
        # TODO: depends on the configuration (single, series, parallel)
        wiperstate = getattr(self, f'wiper{n}state')
        return self.Rlist[self.wiperstate]

    def Rstate_parallel(self):
        '''
        Return total resistance assuming parallel connection of the two pots
        '''
        R1 = self.Rstate(n=1)
        R2 = self.Rstate(n=2)
        return R1*R2/(R1+R2)

    def Rstate_series(self):
        '''
        Return total resistance assuming series connection of the two pots
        '''
        R1 = self.Rstate(n=1)
        R2 = self.Rstate(n=2)
        return R1+R2

    def connected(self):
        if hasattr(self,'conn'):
            return self.conn.isOpen()
        else:
            return False

    def writeRead(self,textstr):
        # Simple send serial Command and print returned answer
        time.sleep(5e-3)
        log.info(self.conn.read_all())
        self.write(textstr)
        time.sleep(5e-3)
        log.info(self.conn.read_all())
        time.sleep(5e-3)
        log.info(self.conn.read_all())

    def set_bypass(self, state):
        '''
        State:
        True = potentiometer path is bypassed by a relay
        False = potentiometer is not bypassed
        '''
        # Keep current wiper state, set the bypass relay state
        w1 = self.wiper1state
        w2 = self.wiper2state
        self.write(f'{w1} {w2} {state}'.encode())
        self.bypass_state = state
        #Wait until the AVR has sent a message Back
        time.sleep(5e-3)
        return self.conn.read_all().decode()

    def set_wiper(self, state, n=2):
        '''
        Change the digipot wiper setting
        n=2 is main potentiometer on chip
        '''
        bypass = self.bypass_state

        if n==1:
            w2 = self.wiper2state
            self.write(f'{state} {w2} {bypass}'.encode())
            self.wiper1state = state
        elif n == 2:
            w1 = self.wiper1state
            self.write(f'{w1} {state} {bypass}'.encode())
            self.wiper2state = state
        #Wait until the AVR has sent a message back
        time.sleep(5e-3)
        return self.conn.read_all().decode()

    def set_state(self, state1=None, state2=None, bypass=None):
        '''
        Change all state settings at the same time
        if anything is None, use the previous setting
        '''
        if bypass is None:
            bypass = self.bypass_state
        if state1 is None:
            state1 = self.wiper1state
        if state2 is None:
            state2 = self.wiper2state
        self.write(f'{state1} {state2} {bypass}'.encode())
        self.wiper1state = state1
        self.wiper2state = state2
        self.bypass_state = bypass
        #Wait until the AVR has sent a message back
        time.sleep(5e-3)
        return self.conn.read_all().decode()

    def set_R(self, R, n=2):
        '''
        Sets individual resistance level of the nth digipot
        n=2 is main potentiometer on chip
        '''
        if R == 0:
            self.set_bypass(1)
            #Set wiper to highest value
            self.set_wiper(0)
            return 0
        else:
            # Find closest resistance value
            # I hope the dictionary returns values and keys in the same order
            i_closest = np.argmin(np.abs(R - self.Rlist))
            R_closest = self.Rlist[i_closest]
            #log.info(i_closest)
            self.set_wiper(i_closest, n)
            # Could have sent one command, but I'm sending two
            self.set_bypass(0)
            time.sleep(1e-3)
            return R_closest

    def set_R_series(self, R):
        '''
        Sets resistance level of a series combination
        '''
        if R == 0:
            self.set_bypass(1)
            return 0
        # Find closest series resistance value
        i_closest = np.argmin(np.abs(R - self.Rserlist))
        R_closest = self.Rserlist[i_closest]
        R1, R2 = self.Rcombinations[i_closest]
        # Better to set both wipers in one shot
        # need to calculate the wiper values
        i1 = np.where(R1 == self.Rlist)[0][0]
        i2 = np.where(R2 == self.Rlist)[0][0]
        self.set_state(i1, i2, bypass=0)
        self.set_bypass(0)
        time.sleep(1e-3)
        return R_closest

    def set_R_parallel(self, R):
        '''
        Sets resistance level of a series combination
        '''
        if R == 0:
            self.set_bypass(1)
            return 0
        # Find closest series resistance value
        i_closest = np.argmin(np.abs(R - self.Rparlist))
        R_closest = self.Rparlist[i_closest]
        R1, R2 = self.Rcombinations[i_closest]
        # Better to set both wipers in one shot
        # need to calculate the wiper values
        i1 = np.where(R1 == self.Rlist)[0][0]
        i2 = np.where(R2 == self.Rlist)[0][0]
        self.set_state(i1, i2, bypass=0)
        time.sleep(1e-3)
        return R_closest

class WichmannDigipot_newfirmware(object):
    '''
    Class for the new digipot firmware (DigiPotSerialInterpreter.ino)
    But this had some problems and was not completed!!
    Not a big deal -- it was mainly just to improve the command syntax

    Probing circuit developed by Erik Wichmann to provide remote series resistance switching
    There are two digipots on board.  You can use a single digipot or connect the two in series or in parallel
    There are 31 ~log spaced resistance values per digipot

    TODO: Change arduino command system to not need entire state in one chunk
    should be three commands, for setting wiper1, wiper2, and relay

    TODO: Is there a way to poll the current state from the microcontroller?
    The class instance might not be aware of it when we first connect.

    TODO: make a test routine that takes a few seconds to measure that everything is working properly.  belongs in measure.py
    TODO: In addition to LCDs that display that the communication is working, we need a programmatic way to verify the connections as well
    '''
    def __init__(self, addr=None):
        # BORG
        statename = self.__class__.__name__
        if statename not in ivtools.instrument_states:
            ivtools.instrument_states[statename] = {}
        self.__dict__ = ivtools.instrument_states[statename]
        self.connect(addr)
        # map from setting to resistance -- needs to be measured by source meter
        # TODO: does the second digipot have a different calibration?
        #self.Rlist = [43080, 38366, 34242, 30547, 27261, 24315, 21719, 19385, 17313,
        #              15441, 13801, 12324, 11022, 8805, 7061, 5670, 4539, 3667, 2964,
        #              2416, 1965, 1596, 1313, 1089, 906, 716, 576, 478, 432, 384, 349,
        #              324, 306, 306]
        # Keithley calibration at 1V applied 2019-07-17
        self.Rlist = [43158.62, 38438.63, 34301.27, 30596.64, 27306.63, 24354.61, 21752.65,
                      19413.07, 17336.84, 15461.77, 13818.91, 12338.34, 11033.65, 8812.41,
                      7064.97, 5672.71, 4539.82, 3666.53, 2961.41, 2412.55, 1960.89, 1591.29,
                      1307.28, 1083.48, 902.42, 711.69, 570.92, 472.24, 426.55, 377.22, 342.16,
                      316.79, 299.09, 299.06]
        self.Rmap = {n:v for n,v in enumerate(self.Rlist)}

    def connect(self, addr=None):
        if not self.connected():
            if addr is None:
                # Connect to the first thing you see that has Leonardo in the description
                # This assumes you programmed the microcontroller as Leonardo.
                # TODO: Figure out how to rename the com device
                # https://github.com/MHeironimus/ArduinoJoystickLibrary/issues/14
                matches = list(comgrep('Leonardo'))
                if any(matches):
                    addr = matches[0].device
                else:
                    log.error('WichmannDigipot could not find Leonardo')
                    return
            self.conn = serial.Serial(addr, timeout=1)
            self.write = self.conn.write
            self.open = self.conn.open
            self.close = self.conn.close
            if self.connected():
                log.info(f'Connected to digipot on {addr}')

    @property
    def Rstate(self):
        # Returns the current set resistance state
        # TODO: depends on the configuration (single, series, parallel)
        return self.Rmap[self.wiper1state]

    @Rstate.setter
    def Rstate(self, val):
        self.set_R(val)

    @property
    def wiper0state(self):
        self.write(f'get_wiper 0 \n'.encode())
        time.sleep(5e-3)
        return int(self.conn.read_all().decode())

    @property
    def wiper1state(self):
        self.write(f'get_wiper 1 \n'.encode())
        time.sleep(5e-3)
        return int(self.conn.read_all().decode())

    @property
    def read(self):
         return self.conn.read_all().decode()

    def connected(self):
        if hasattr(self,'conn'):
            return self.conn.isOpen()
        else:
            return False

    def writeCMD(self,textstr):
        '''
        Debugging tool.
        Send serial Command and print returned answer like a Serial monitor
        '''
        self.write((textstr+' \n').encode())
        time.sleep(5e-3)
        log.info(self.conn.read_all().decode())

    def set_bypass(self, state):
        '''
        State:
        True = connected
        False = not connected
        '''
        self.write(f'bypass {int(state)} \n'.encode())
        self.bypassstate = state
        #Wait until the AVR has sent a message Back
        time.sleep(5e-3)
        self.conn.read_all().decode()

    def set_wiper(self, state, n=1):
        '''
        Change the digipot wiper setting
        n=1 is main potentiometer on chip
        0 ist only used in parallel/series Mode
        '''
        self.write(f'wiper {n} {state}'.encode())
        #Wait until the AVR has sent a message Back
        time.sleep(5e-3)
        self.conn.read_all().decode()

    def set_R(self, R, n=1):
        if R == 0:
            self.set_bypass(1)
            #Set wiper to highest value
            self.set_wiper(0)
            return 0
        else:
            # Find closest resistance value
            # I hope the dictionary returns values and keys in the same order
            Rmap = list(self.Rmap.values())
            wvals= list(self.Rmap.keys())
            i_closest = np.argmin(np.abs(R - np.array(Rmap)))
            R_closest = Rmap[i_closest]
            w_closest = wvals[i_closest]
            log.info(i_closest)
            log.info(self.Rmap[i_closest])
            self.set_wiper(w_closest, n)
            # Could have sent one command, but I'm sending two
            self.set_bypass(0)
            time.sleep(1e-3)
            return R_closest

    def set_series_R(self, R):
        # TODO calculate nearest series value
        pass

    def set_parallel_R(self, R):
        # TODO calculate nearest parallel value
        pass
