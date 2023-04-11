import M30XY
import KVS30

import keyboard

import numpy as np
from scipy.optimize import curve_fit 

class AutoStage:
    
    def __init__(self, xy_sn : str = "101334404", z_sn : str = "24347834"):
        self.xyStage = M30XY.M30XY(sn = xy_sn)
        self.zStage = KVS30.KVS30(sn = z_sn)
        
    def setup(self):
        self.xyStage.connect()
        self.zStage.connect()
        
        self.xyStage.homeStage()
        self.zStage.homeStage()
        
    def moveXY(self, x : float = 0, y : float = 0, blocking = False):
        self.xyStage.moveTo("CH1", x, blocking = blocking)
        self.xyStage.moveTo("CH2", y, blocking = blocking)
        
    def moveZ(self, z : float = 0, blocking = False):
        self.zStage.moveTo(z, blocking = blocking)
    
    def moveXYZ(self, x : float = 0, y : float = 0, z : float = 0, blocking = False):
        self.xyStage.moveTo("CH1", x, blocking = blocking)
        self.xyStage.moveTo("CH2", y, blocking = blocking)
        self.zStage.moveTo(z, blocking = blocking)
    
    def interactiveMove(self):
        positions = list()
        letter = ""
        modifier = ""
        while(letter != "q"):
            key = keyboard.read_hotkey(suppress = False)
            keys = key.split("+")
            
            if(len(keys) == 2):
                modifier = keys[0].casefold()
                letter = keys[1].casefold()
            elif(len(keys) == 1):
                modifier = ""
                letter = keys[0].casefold()
                
            if(modifier == "shift"):
                step = 5
                vel = 2.4
                acc = 5
            else:
                step = 1
                vel = 2.0
                acc = 3
            
            try:
                if(letter == "w"):
                    self.xyStage.moveRelative("CH2", step)
                elif(letter == "s"):
                    self.xyStage.moveRelative("CH2", -step)
                elif(letter == "d"):
                    self.xyStage.moveRelative("CH1", step)
                elif(letter == "a"):
                    self.xyStage.moveRelative("CH1", -step)
                elif(letter == "["):
                    self.zStage.moveRelative(-step)
                elif(letter == "]"):
                    self.zStage.moveRelative(step)
                elif(letter == "p"):
                    xy = self.xyStage.getPosition()
                    z = self.zStage.getPosition()
                    positions.append((xy["CH1"], xy["CH2"], z))
            except:
                pass
                
        return positions
    
    def gotoAndDo(self, positions, separation = -1, func = None):
        for p in positions:
            self.moveXY(p[0], p[1], blocking = True)
            self.moveZ(p[2], blocking = True)
            
            if(func != None):
                func()
            
            self.zStage.moveRelative(separation)

    def transformPositions(pos, angle, shift):
        sh = np.array(shift, ndmin = 2)
        rot = np.array([(np.cos(angle), -np.sin(angle)),
                        (np.sin(angle), np.cos(angle))])
        
        new_pos = np.dot(rot, pos.T)  + sh.T
        return new_pos.T
    
    def flat(pos, angle, x, y):
        return AutoStage.transformPositions(pos, angle, (x, y)).flatten()

    def estimateSample(pos, ref):
        xy = np.array([(p[0], p[1]) for p in pos])
        xy_ref = np.array(ref)
        
        popt, pcov = curve_fit(AutoStage.flat, xy_ref, xy.flatten())
        return popt[0], (popt[1], popt[2])
    
    def nom2actual(nom_pos, angle, shift, z_mean):
        corr_pos = AutoStage.transformPositions(np.array(nom_pos), angle, shift)
        
        return [(p[0], p[1], z_mean) for p in corr_pos]
            
    def close(self):
        self.moveXYZ(blocking = True)
        self.xyStage.close()
        self.zStage.close()