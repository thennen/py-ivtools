"""
This is some basic code to make the Thorlabs vertical stage KVS30 available in python.

Thorlabs provides a C#/.net API, which this more or less wraps. Only a very limited
subset of the API is implemented, most notably the 'jog mode' is missing.
The implementation is mostly guided by the examples in the API reference,
this might not be the best way to do things.

The moves use a trapezoidal velocity profile with a given acceleration to a
steady state speed. Slowing down is also gradual. The API allows to use a
non-zero initial speed. This seems to be of little use, as the stage should be
at rest before a move starts. The Thorlabs GUI doesn't use this either.

The stage can move between 0 and 30 mm. Zero is lowest position, 30 mm is
highest.

Basic use:
    stage = KVS30()
    stage.connect()
    stage.homeStage()
    stage.moveTo(...)
    ...
    stage.close()
"""

import time

# To call Thorlabs .net API
import clr

api_path : str = "C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.Benchtop.DCServoCLI.dll"
stage_path : str ="C:/Program Files/Thorlabs/Kinesis/ThorLabs.MotionControl.VerticalStageCLI.dll"

clr.AddReference(api_path)
clr.AddReference(stage_path)
# The MX30 is apparently a benchtop DC servo stage
import Thorlabs.MotionControl.Benchtop.DCServoCLI as bt
import Thorlabs.MotionControl.DeviceManagerCLI as dev
# Move functions come from this I think
import Thorlabs.MotionControl.GenericMotorCLI as gmc
import Thorlabs.MotionControl.VerticalStageCLI as vsc

from System import Decimal

# Timeout used in the move and home commands. Homing takes more time normally.
defaultTimeout = 60000

class KVS30:
    """
    Controls the Thorlabs KVS30 vertical stage.
    """
    
    connections = {}
    
    def __init__(self, sn : str = "24347834"):
        """
        Constructor for a KVS30 stage.

        Parameters
        ----------
        sn : str, optional
            The serial number of the stage to connect to.
            The default is "24347834".

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
        
        Note that in contrast to the M30XY stage, this one can (and should)
        be switched off at its back.
        
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
            self.connected = True
        else:
            self.stage = vsc.VerticalStage.CreateVerticalStage(self.serialNumber)
            self.stage.Connect(self.serialNumber)
            
            self.stage.StartPolling(250)
            self.stage.EnableDevice()
            
            time.sleep(0.5)           
            
            self.stage.LoadMotorConfiguration(self.serialNumber)
            
            self.connections.update({ self.serialNumber:
                                     {"stage": self.stage
                                     }})
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
            self.stage.Home(defaultTimeout)
        else: raise Exception("Stage not connected!")
        
        
    def getPosition(self) -> float:
        """
        Returns the current position of in millimeters.

        Returns
        -------
        float
            Position in millimeters. Higher values are up direction.

        """
        return Decimal.ToDouble(self.stage.Position)
    
    def moveTo(self, millimeter : float, blocking : bool = True):
        """
        Movesto the specified position in millimeters.

        The stage moves with the acceleration and velocity last set with
        setVelocity().

        Parameters
        ----------
        millimeter : float
            Coordinate to move to. Valid values are 0 ... 30.
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
            self.stage.MoveTo(Decimal(millimeter), timeout)
        else: raise Exception("Stage not connected!")
    
    def moveRelative(self, millimeter : float, direction : str = "+"):
        """
        Performes a move relative to the current position.
        
        The actual direction is controlled by both the sign of <millimeters>
        and the setting of <direction>.

        Parameters
        ----------
        millimeter : float
            Distance to move. Can be positive or negative.
        direction : str, optional
            Direction to move stage in. "+" is up direction.
            The default is "+".

        Returns
        -------
        None.

        """
        
        if(direction == "+"):
            direction = 1
        elif(direction == "-"):
            direction = 2
        else:
            raise ValueError("Direction must be '+' or '-'!")
        
        if(self.serialNumber in self.connections.keys()):
            self.stage.MoveRelative(gmc.MotorDirection(direction), Decimal(millimeter), defaultTimeout)
        else: raise Exception("Stage not connected!")
    
    def getVelocity(self) -> dict:
        """
        Returns the current velocity and acceleration settings.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        dict
            Velocity is in mm/s, acceleration in mm/s^2.

        """
        if(self.serialNumber in self.connections.keys()):
            velA = self.stage.GetVelocityParams()
        else: raise Exception("Stage not connected!")
        
        return {"min. vel.": Decimal.ToDouble(velA.MinVelocity),
                "max. vel.": Decimal.ToDouble(velA.MaxVelocity),
                "acc.": Decimal.ToDouble(velA.Acceleration)}
    
    def setVelocity(self, maxVel : float = 2.0, acc : float = 1.0,
                    minVel : float = 0.0):
        """
        Sets the stages acceleration and velocity.

        Parameters
        ----------
        maxVel : float, optional
            Steady state speed of movement in mm/s. The default is 2.0.
        acc : float, optional
            Acceleration to maxVel in mm/s^2. The default is 1.0.
        minVel : float, optional
            Initial speed. The default is 0.0.

        Returns
        -------
        None.

        """
        if(maxVel > 8.0): # mm/s ?
            raise ValueError("Maximum velocity must be < 8.0!")
        if(acc > 5): # mm/s^2 ?
            raise ValueError("Maximum velocity must be < 5.0!")
        
        if(self.serialNumber in self.connections.keys()):
            self.stage.SetVelocityParams(Decimal(maxVel), Decimal(acc))
        else: raise Exception("Stage not connected!")
    
    def close(self):
        """
        Disables and disconnects from the stage.

        Returns
        -------
        None.

        """
        if(self.serialNumber in self.connections.keys()):
            self.stage.StopPolling()
            self.stage.DisableDevice()
            self.stage.Disconnect(True)
            
            self.connections.pop(self.serialNumber)
            self.connected = False
        else: raise Exception("Stage not connected!")