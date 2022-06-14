import serial

class RelayBoard:
    """ Controls the relay board. """
    
    def __init__(self, comPort = "COM12"):
        """
        Open serial connection to relay board.

        Parameters
        ----------
        comPort : str, optional
            The serial port where the board is connected.
            The default is "COM12".

        Raises
        ------
        Exception
            If the connection to the board was unsuccessfull.

        Returns
        -------
        None.

        """
        self.ser = serial.Serial()
        self.ser.port = comPort
        self.ser.baudrate = 9600
        self.ser.timeout = 2
        
        try:
            self.ser.open()
        except:
            raise Exception("Could not connect to board, " +
                            "is a connection already open?")
        
    def switch(self, ch1 = "off", ch2 = "off"):
        """
        Switches the relays.
        
        They are dual throw, one input is always
        connected to one of two outputs. 'off' refers to no current in the
        coil.

        Parameters
        ----------
        ch1 : str, optional
            What output to switch the relay to. Can be 'off' or 'on'.
            The default is "off".
        ch2 : str, optional
            What output to switch the relay to. Can be 'off' or 'on'.
            The default is "off".

        Raises
        ------
        Exception
            The board should respond to an attempt to switch, otherwise there
            is a problem.
            Wrong parameters also yield an exception.

        Returns
        -------
        None.

        """
        if ch1 == "off" and ch2 == "off":
            self.ser.write(b"0")
            if self.ser.read(1) != b"0":
                raise Exception("Something went wrong setting the relays!")
        elif ch1 == "on" and ch2 == "off":
            self.ser.write(b"1")
            if self.ser.read(1) != b"1":
                raise Exception("Something went wrong setting the relays!")
        elif ch1 == "off" and ch2 == "on":
            self.ser.write(b"2")
            if self.ser.read(1) != b"2":
                raise Exception("Something went wrong setting the relays!")
        elif ch1 == "on" and ch2 == "on":
            self.ser.write(b"3")
            if self.ser.read(1) != b"3":
                raise Exception("Something went wrong setting the relays!")
        else:
            raise Exception("ch1/ch2 must be 'on' or 'off'.")
    
    def state(self):
        """
        Get the current state of the relays.

        Returns
        -------
        dict
            One key for each channel. Values can be 'on' or 'off'.

        """
        
        self.ser.write(b"?")
        data = self.ser.read(2)
        
        return {"ch1": "off" if data[0] == 48 else "on",
                "ch2": "off" if data[1] == 48 else "on"}
        
    def close(self):
        """
        Close the serial port.

        Returns
        -------
        None.

        """
        self.ser.close()