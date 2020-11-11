import numpy as np
import serial
import logging
import ivtools
from serial.tools.list_ports import grep as comgrep
log = logging.getLogger('instruments')

class EugenTempStage(object):
    '''
    Controls the temperature of a thermoelectric device (peltier cooler), which is used as a sample stage
    the temperature range is about 5-100°C

    Temperature is measured using a Pt1000 sensor.

    We built this from scratch because all of the cheap digital PID controllers use PWM, which cause
    interference if you try to measure on the heating surface/near the thermometer.
    Our design is totally DC and uses an op-amp based PID controller

    computer interface is using Arduino micro, firmwire in py-ivtools/firmware

    schematics, firmware, and other documentation at
    https://git.rwth-aachen.de/ujeane/pt-element-pid-controller
    '''
    # Resistor-Values used in wheatstone bridge
    r_1 = 9975
    r_3 = 9976
    r_4 = 1001
    # Gain from instrumentation amp
    opamp_gain = 12.55
    # Temperature coeff of pt1000 (given in ppm/K in datasheet)
    tempco = 0.00385
    # Voltage supplied to the wheatstone bridge
    Vsupply = 10

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
        ''' Output a voltage using the external 12 bit DAC chip '''
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
        ''' Read voltage from one of arduino pins '''
        vmax = 5
        numbits = 10
        vstep = round(vmax / (2**numbits - 1), 5)# 5 /1023
        cmd_str = '1,{};'.format(channel).encode()
        self.write(cmd_str)

        reply = self.conn.readline().decode()
        adc_value = float(reply.split(',')[-1].strip().strip(';'))
        voltage = adc_value * vstep
        return voltage

    def _V_to_T(self, Vread):
        '''
        The Pt1000 is in an amplified Wheatstone bridge, so we need to do a
        little calculation to convert to temperature
        '''
        r_1 = self.r_1
        r_3 = self.r_3
        r_4 = self.r_4
        Vsupply = self.Vsupply
        opamp_gain = self.opamp_gain
        tempco = self.tempco
        Vbridge = Vread / opamp_gain
        pt_zaehler = (((r_3 + r_4) * Vbridge) + (Vsupply * r_4)) * r_1
        pt_nenner = ((r_3 + r_4) * Vsupply) - (Vbridge * (r_3 + r_4) + (r_4 * Vsupply))
        pt_res = pt_zaehler / pt_nenner
        temp_read = np.log(pt_res / 1000) / tempco
        return temp_read

    def _T_to_V(self, temp):
        '''
        The Pt1000 is in an amplified Wheatstone bridge, so we need to do a
        little calculation to convert to temperature
        '''
        r_1 = self.r_1
        r_3 = self.r_3
        r_4 = self.r_4
        Vsupply = self.Vsupply
        opamp_gain = self.opamp_gain
        tempco = self.tempco

        pt_res = 1000 * np.exp(temp * tempco)
        volt_zaehler = Vsupply * (pt_res * (r_3 + r_4) - r_4 * (r_1 + pt_res))
        volt_nenner = (r_4 + r_3) * r_1 + (r_3 + r_4) * pt_res
        volt_bruch = volt_zaehler / volt_nenner
        volt_set = volt_bruch * opamp_gain
        # Prevent negative Voltages DAC can't handle signed Integer and goes to the highest range
        if(volt_set<0):
            volt_set = 0
        return volt_set

    def set_temperature(self, temp):
        '''
        Temperature Setpoint Function, should be between 0-100° C

        It's considered ill-advised to try to go outside of these ranges
        but further testing is needed to determine the real limitations
        two peltier stacked peltier elements connected in parallel is an
        option if an increased range is required.
        '''
        if temp > 100:
            log.warning(f'{temp} °C is too high! Trying 100°C.')
            temp = 100
        elif temp < 0:
            log.warning(f'{temp} °C is too low! Trying 0°C.')
            temp = 0
        volt_set = self._T_to_V(temp)
        temp_set = self.analogOut(volt_set)
        log.info('Temperature set to {0:.2f} °C'.format(temp))

        if self.analogIn(2) < 4:
            log.error('Temperature stage has no power?')

    def read_temperature(self):
        '''Function which reads temperature from Pt1000 '''
        Vread = self.analogIn(1)
        temp_read = self._V_to_T(Vread)

        if self.analogIn(2) < 4:
            log.error('Temperature stage has no power?')

        return np.round(temp_read, 2)
