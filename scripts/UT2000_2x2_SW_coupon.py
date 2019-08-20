import pandas as pd
import sys
sys.path.append(r'C:\py-ivtools\ivtools')
from instruments import UF2000Prober
# Want a function that takes die_rel, module, devicenum, and moves to that location

# For that, I need all the device locations with respect to the "home" device, which is mod 002, device 1
coupondf = pd.read_pickle(r'C:\py-ivtools\ivtools\sampledata\lassen_coupon_info.pkl').set_index(['die_rel', 'module', 'device'])

# Go to home position and get the position in microns
p = UF2000Prober()
p.goHome()

current_die = 1
current_mod = '002'
current_dev = 1

x_home, y_home = p.getPosition_um()


def gotoDevice(die_rel=1, module='001', device=2):
    # find location of this device relative to home device
    wX, wY = coupondf.loc[(die_rel, module, device)][['wX', 'wY']]
    p.moveAbsolute_um(x_home + wX, y_home + wY)
    print('WARNING: you must use location return values!')
    return die_rel, module, device


def get_currentPosition():
    print('Die: {}\nModule: {}\nDevice: {}\n'.format(self.current_die, self.current_mod, self.current_dev))


def __enter__(self):
    pass