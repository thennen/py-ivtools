import numpy as np
import serial
import logging
import ivtools
from serial.tools.list_ports import grep as comgrep
log = logging.getLogger('instruments')

class EugenTempStage(object):
    # Global Variables
    # Resistor-Values bridge
    r_1 = 9975
    r_3 = 9976
    r_4 = 1001
    # Gain from instrumental-opamp
    opamp_gain = 12.55
    # Voltage Bridge
    # TODO: do a sample of this voltage to make sure the voltage supply is on, otherwise return an error that says to turn it on!
    volt_now = 10

    def __init__(self, addr=None, baudrate=9600):
        # BORG
        statename = self.__class__.__name__
        if statename not in ivtools.instrument_states:
            ivtools.instrument_states[statename] = {}
        self.__dict__ = ivtools.instrument_states[statename]
        try:
            self.connect(addr, baudrate)
        except:
            log.error('Arduino connection failed at {}'.format(addr))

    def connect(self, addr=None, baudrate=9600):
        if not self.connected():
            if addr is None:
                # Connect to the first thing you see that has Arduino Micro in the description
                matches = list(comgrep('Arduino Micro'))
                if any(matches):
                    addr = matches[0].device
                else:
                    log.error('EugenTempStage could not find Arduino Micro')
                    return
            self.conn = serial.Serial(addr, baudrate, timeout=1)
            self.write = self.conn.write
            self.close = self.conn.close
            if self.connected():
                log.info(f'Connected to PID controller on {addr}')

    def connected(self):
        if hasattr(self, 'conn'):
            return self.conn.isOpen()
        else:
            return False

    def analogOut(self, voltage):
        ''' Tell arduino to output a voltage for the DAC '''
        vmax = 5
        numbits = 12
        # Find the closest value that can be output.
        vstep = vmax / (2**numbits - 1)  # 5 /4095
        value = voltage / vstep  # exact value for analogWrite()-function
        cmd_str = '0,{};'.format(value).encode()
        self.write(cmd_str)
        actualvoltage = vstep * value
        return actualvoltage

    def analogIn(self, channel):
        ''' Function to get Voltage from Bridge, Arduino reads Voltage on PIN A1'''
        vmax = 5
        numbits = 10
        vstep = round(vmax / (2**numbits - 1), 5)# 5 /1023
        cmd_str = '1,{};'.format(channel).encode()
        self.write(cmd_str)

        reply = self.conn.readline().decode()
        adc_value = float(reply.split(',')[-1].strip().strip(';'))
        voltage = adc_value * vstep
        return voltage

    def set_temperature(self, temp):
        '''Temperature Setpoint Function, should be between 0-100Celsius'''

        if temp > 100:
            log.warning('Its too HOT! DANGERZONE!')

        if temp <= 100 and temp >= 0:
            pt_res = round((1000 * (1.00385**temp)), 1)
            volt_zaehler = self.volt_now * (pt_res * (self.r_3 + self.r_4) - self.r_4 * (self.r_1 + pt_res))
            volt_nenner = (self.r_4 + self.r_3) * self.r_1 + (self.r_3 + self.r_4) * pt_res
            volt_bruch = volt_zaehler / volt_nenner
            volt_set = volt_bruch * self.opamp_gain
            temp_set = self.analogOut(volt_set)
            log.info('Temperature set to {0:.2f} \u00b0C'.format(temp))
        else:
            log.warning('Its too COLD! Can not do that :-/')

    def read_temperature(self):
        '''Function which reads temperature'''
        r_1 = self.r_1
        r_3 = self.r_3
        r_4 = self.r_4
        volt_now = self.volt_now
        opamp_gain = self.opamp_gain

        volt_bridge = self.analogIn(1) / opamp_gain
        pt_zaehler = (((r_3 + r_4) * volt_bridge) + (volt_now * r_4)) * r_1
        pt_nenner = ((r_3 + r_4) * volt_now) - (volt_bridge * (r_3 + r_4) + (r_4 * volt_now))
        pt_res = round((pt_zaehler / pt_nenner), 1)
        temp_read = np.log(pt_res / 1000) / np.log(1.00385)
        return temp_read
