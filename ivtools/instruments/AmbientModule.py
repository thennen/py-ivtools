import serial
import json

class AmbientModule:
    """
    Handles the connection to the ambient sensor module. Most of the work
    is done in the firmware, this just makes a JSON-RPC call via serial to it.
    Available values and units are:
        Temperature: C, F
        Humidity: percent
        Pressure: mbar, psi, kPa
        Illuminance: lux, footcandle, metercandle
    """
    
    def __init__(self, com = "COM15"):
        """
        Connects to the ambient module.

        Parameters
        ----------
        com : str, optional
            Com port where the module is connected. The default is "COM15".

        Returns
        -------
        None.

        """
        self.conn = serial.Serial()
        self.conn.port = com;
        self.conn.baudrate = 9600;
        self.conn.timeout = 2;
        
        try:
            self.conn.open()
        except:
            raise Exception("Could not connect to ambient module, " +
                            "is a connection already open?")
        
    def close(self):
        """
        Closes the connection to the module.

        Returns
        -------
        None.

        """
        self.conn.close()
        
    def request(self, method, units = None):
        """
        Makes a JSON-RPC call to the module to run the desired method,
        with the given parameters/units.

        Parameters
        ----------
        method : str
            Method to call.
        units : str/list, optional
            Either a string specifing the desired unit, or a list of strings,
            if all sensors are requested. The default is None.

        Returns
        -------
        data : dict
            With property and unit. If multiple sensors are requested, dict
            of such dicts.

        """
        req = {"jsonrpc": "2.0",
               "method": method,
               "id": 0}
        
        if not units == None:
            if type(units) is list:
                req["params"] = {"units": units}
            else:
                req["params"] = {"unit": units}
            
        
        self.conn.write((json.dumps(req) + "\n").encode("utf-8"))
        
        res = self.conn.read_until()
        res = json.loads(res[:-1])
        
        if "error" in res.keys():
            # TODO this could be more descriptive?
            raise Exception(res["error"]["message"])
        else:
            data = res["result"]
            
        return data
    
    def getTemp(self, unit = "C"):
        """
        Return current temperature reading.

        Parameters
        ----------
        unit : str, optional
            Desired unit, available are 'C' and 'F'. The default is "C".

        Returns
        -------
        dict
            Contains value and unit.

        """
        return self.request(method = "getTemperature", units = unit)
    
    def getHumid(self):
        """
        Return current humidity reading.

        Returns
        -------
        dict
            Contains value and unit.

        """
        return self.request(method = "getHumidity")
    
    def getPress(self, unit = "mbar"):
        """
        Return current pressure reading.
        
        Parameters
        ----------
        unit : str, optional
            Desired unit, available are 'mbar', 'psi' and 'kpa'.
            The default is "mbar".

        Returns
        -------
        dict
            Contains value and unit.

        """
        return self.request(method = "getPressure", units = unit)
        
    def getIllum(self, unit = "lux"):
        """
        Return current illuminance reading.
        
        Parameters
        ----------
        unit : str, optional
            Desired unit, available are 'lux', 'footcandle' and 'metercandle'.
            The default is "lux".

        Returns
        -------
        dict
            Contains value and unit.

        """
        return self.request(method = "getIlluminance", units = unit)
    
    def getAll(self, units = ["C", "mbar", "lux"]):
        """
        Gets a dict with all available sensor data.

        Parameters
        ----------
        units : list, optional
            Desired units for temperature, pressure and illuminance. Humidity
            is always in percent. The default is ["C", "mbar", "lux"].

        Returns
        -------
        dict
            Dict with sensor data and unit.

        """
        return self.request(method = "getAll", units = units)
    