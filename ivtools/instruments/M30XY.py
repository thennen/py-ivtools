"""
This is some basic code to make the Thorlabs XY stage M30XY available in python.

Thorlabs provides a C#/.net API, which this more or less wraps. Only a very limited
subset of the API is implemented, most notably the 'jog mode' is missing.
The implementation is mostly guided by the examples in the API reference,
this might not be the best way to do things.

The stage has two separate channels for the X and Y direction. These can move
simulateously if 'blocking = false' is used.

The moves use a trapezoidal velocity profile with a given acceleration to a
steady state speed. Slowing down is also gradual. The API allows to use a
non-zero initial speed. This seems to be of little use, as the stage should be
at rest before a move starts. The Thorlabs GUI doesn't use this either.

The stage can move 15 mm in either direction.

Basic use:
    stage = M30XY()
    stage.connect()
    stage.homeStage()
    stage.moveTo(...)
    ...
    stage.close()
"""

# To call Thorlabs .net API
import clr

api_path : str = "C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.Benchtop.DCServoCLI.dll"

clr.AddReference(api_path)
# The MX30 is apparently a benchtop DC servo stage
import Thorlabs.MotionControl.Benchtop.DCServoCLI as bt
import Thorlabs.MotionControl.DeviceManagerCLI as dev
# Move functions come from this I think
import Thorlabs.MotionControl.GenericMotorCLI as gmc

from System import Decimal

# Timeout used in the move and home commands. Homing takes more time normally.
defaultTimeout = 60000

class M30XY:
    """
    Controls the Thorlabs M30XY stage.
    """
    
    connections = {}
    
    def __init__(self, sn : str = "101334404"):
        """
        Constructor for a M30XY stage.

        Parameters
        ----------
        sn : str, optional
            The serial number of the stage to connect to.
            The default is "101334404".

        Returns
        -------
        None.

        """
        self.serialNumber = sn
        self.connected = False
        
    def connect(self):
        """
        Tries to connect to the stage.

        If a connection to the stage with this serial number is already open,
        that connection is recycled.
        
        Returns
        -------
        None.

        """
        dev.DeviceManagerCLI.BuildDeviceList()
        devicesList = dev.DeviceManagerCLI.GetDeviceList()
        
        
        try:
            if(not devicesList.Contains(self.serialNumber)):
                raise Exception("Device with serial number "
                                + self.serialNumber + " not found!")
        except TypeError:
            raise TypeError("Serial number must be a string!")
        
        if(self.serialNumber in self.connections.keys()):
            self.stage = self.connections[self.serialNumber]["stage"]
            self.CH1 = self.connections[self.serialNumber]["CH1"]
            self.CH2 = self.connections[self.serialNumber]["CH2"]
            self.connected = True
        else:
            self.stage = bt.BenchtopDCServo.CreateBenchtopDCServo(self.serialNumber)
            self.stage.Connect(self.serialNumber)
            
            self.CH1 = self.stage.GetChannel(1)
            self.CH1.StartPolling(250)
            self.CH1.EnableDevice()
            
            self.CH1.LoadMotorConfiguration(self.CH1.DeviceID)
            
            self.CH2 = self.stage.GetChannel(2)
            self.CH2.StartPolling(250)
            self.CH2.EnableDevice()
            
            self.CH2.LoadMotorConfiguration(self.CH2.DeviceID)
            
            self.connections.update({ self.serialNumber:
                                     {"stage": self.stage,
                                      "CH1": self.CH1,
                                      "CH2": self.CH2}
                                     })
            self.connected = True
        
    def homeStage(self):
        """
        Homes the stage. Recommended to do before using the stage.
        
        Apparently not strictly necessary.

        Returns
        -------
        None.

        """
        if(self.serialNumber in self.connections.keys()):
            self.CH1.Home(defaultTimeout)
            self.CH2.Home(defaultTimeout)
        else: raise Exception("Stage not connected!")
        
        
    def getPosition(self) -> dict:
        """
        Returns the current position of both axes in millimeters.

        Returns
        -------
        dict
            Current position of CH1 and CH2.

        """
        return {"CH1": Decimal.ToDouble(self.CH1.Position),
                "CH2": Decimal.ToDouble(self.CH2.Position)}
    
    def moveTo(self, ch : str, millimeter : float, blocking : bool = True):
        """
        Moves the given axis to the specified position in millimeters.

        The stage moves with the acceleration and velocity last set with
        setVelocity().
        
        Parameters
        ----------
        ch : str
            Selects the axis to move, can be 'CH1' or 'CH2'.
        millimeter : float
            Coordinate to move to. Valid values are -15 ... 15.
        blocking : bool, optional
            If false, returns immediately,
            e.g. so a move of the other axis can be initiated.
            The default is True.

        Returns
        -------
        None.

        """
        if(blocking):
            timeout = defaultTimeout
        else:
            timeout = 0
        
        if(self.serialNumber in self.connections.keys()):
            if(ch == "CH1"):
                self.CH1.MoveTo(Decimal(millimeter), timeout)
            elif(ch == "CH2"):
                self.CH2.MoveTo(Decimal(millimeter), timeout)
            else:
                raise ValueError("Channel must be 'CH1'/'CH2'!")
        else: raise Exception("Stage not connected!")
        
    def moveRelative(self, ch : str, millimeter : float, direction : str = "+"):
        """
        Performes a move relative to the current position.
        
        The actual direction is controlled by both the sign of <millimeters>
        and the setting of <direction>.

        Parameters
        ----------
        ch : str
            Selects the axis to move, can be 'CH1' or 'CH2'..
        millimeter : float
            Distance to move. Can be positive or negative.
        direction : str, optional
            Direction to move stage in. The default is "+".

        Returns
        -------
        None.

        """
       
        if(direction == "+"):
            direction = 2
        elif(direction == "-"):
            direction = 1
        else:
            raise ValueError("Direction must be '+' or '-'!")
        
        if(self.serialNumber in self.connections.keys()):
            if(ch == "CH1"):
                self.CH1.MoveRelative(gmc.MotorDirection(direction), Decimal(millimeter), defaultTimeout)
            elif(ch == "CH2"):
                self.CH2.MoveRelative(gmc.MotorDirection(direction), Decimal(millimeter), defaultTimeout)
            else:
                raise ValueError("Channel must be 'CH1'/'CH2'!")
        else: raise Exception("Stage not connected!")
        
    def getVelocity(self) -> dict:
        """
        Returns the current velocity and acceleration settings.

        Returns
        -------
        dict
            Current settings for both channels. Velocities are in mm/s,
            accelerations in mm/s^2

        """
        if(self.serialNumber in self.connections.keys()):
            velA = self.CH1.GetVelocityParams()
            velB = self.CH2.GetVelocityParams()
        else: raise Exception("Stage not connected!")
        
        return {"CH1 min. vel.": Decimal.ToDouble(velA.MinVelocity),
                "CH1 max. vel.": Decimal.ToDouble(velA.MaxVelocity),
                "CH1 acc.": Decimal.ToDouble(velA.Acceleration),
                "CH2 min. vel.": Decimal.ToDouble(velB.MinVelocity),
                "CH2 max. vel.": Decimal.ToDouble(velB.MaxVelocity),
                "CH2 acc.": Decimal.ToDouble(velB.Acceleration)}
    
    def setVelocity(self, ch : str, maxVel : float = 2.3, acc : float = 5.0, minVel : float = 0.0):
        """
        Sets the stage channels acceleration and velocity.

        Parameters
        ----------
        ch : str
            Selects an axis, can be 'CH1' or 'CH2'.
        maxVel : float, optional
            Steady state speed of movement in mm/s. The default is 2.3.
        acc : float, optional
            Acceleration to maxVel in mm/s^2. The default is 5.0.
        minVel : float, optional
            Initial speed. The default is 0.0.

        Returns
        -------
        None.

        """
        if(maxVel > 2.4): # mm/s ?
            raise ValueError("Maximum velocity must be < 2.4!")
        if(acc > 5): # mm/s^2 ?
            raise ValueError("Maximum velocity must be < 5.0!")
        
        if(self.serialNumber in self.connections.keys()):
            if(ch == "CH1"):
                self.CH1.SetVelocityParams(Decimal(maxVel), Decimal(acc))
            elif(ch == "CH2"):
                self.CH2.SetVelocityParams(Decimal(maxVel), Decimal(acc))
            raise ValueError("Channel must be 'CH1'/'CH2'!")
        else: raise Exception("Stage not connected!")
        
        
    def close(self):
        """
        Disables and disconnects from the stage.

        Returns
        -------
        None.

        """
        if(self.serialNumber in self.connections.keys()):
            self.CH1.StopPolling()
            self.CH2.StopPolling()
            self.CH1.DisableDevice()
            self.CH2.DisableDevice()
            self.stage.Disconnect(True)
            
            self.connections.pop(self.serialNumber)
            self.connected = False
        else: raise Exception("Stage not connected!")