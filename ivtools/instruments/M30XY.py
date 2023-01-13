import clr

clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.Benchtop.DCServoCLI.dll')
import Thorlabs.MotionControl.Benchtop.DCServoCLI as bt
import Thorlabs.MotionControl.DeviceManagerCLI as dev
import Thorlabs.MotionControl.GenericMotorCLI as gmc

from System import Decimal

# TODO: Jog mode

defaultTimeout = 60000

class M30XY:
    connections = {}
    
    def __init__(self, sn = "101334404"):
        self.serialNumber = sn
        self.connected = False
        
    def connect(self):
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
        if(self.serialNumber in self.connections.keys()):
            self.CH1.Home(defaultTimeout)
            self.CH2.Home(defaultTimeout)
        else: raise Exception("Stage not connected!")
        
        
    def getPosition(self):
        print(self.CH1.IsPositionCalibrated)
        return {"CH1": Decimal.ToDouble(self.CH1.Position),
                "CH2": Decimal.ToDouble(self.CH2.Position)}
    
    def moveTo(self, ch, millimeter, blocking = True):
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
        
    def moveRelative(self, ch, millimeter, direction = "+"):
        # Sign for millimeter also changes direction
        
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
            raise ValueError("Channel must be 'CH1'/'CH2'!")
        else: raise Exception("Stage not connected!")
        
    def getVelocity(self):
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
    
    def setVelocity(self, ch, maxVel = 2.3, acc = 5.0, minVel = 0.0):
        
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
        if(self.serialNumber in self.connections.keys()):
            self.CH1.StopPolling()
            self.CH2.StopPolling()
            self.CH1.DisableDevice()
            self.CH2.DisableDevice()
            self.stage.Disconnect(True)
            
            self.connections.pop(self.serialNumber)
            self.connected = False
        else: raise Exception("Stage not connected!")