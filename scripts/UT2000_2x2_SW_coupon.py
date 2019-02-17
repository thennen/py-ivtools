import pandas as pd

# Want a function that takes die_rel, module, devicenum, and moves to that location

# For that, I need all the device locations with respect to the "home" device, which is mod 002, device 1
coupondf = pd.read_pickle('lassen_coupon_info.pkl').set_index(['die_rel', 'module', 'device'])

# Go to home position and get the position in microns
p = instruments.UF2000Prober()
p.goHome()
x_home, y_home = p.getPosition_um()

def gotoDevice(die_rel=1, module='001', device=2):
    # find location of this device relative to home device
    wX, wY = coupondf.loc[(die_rel, module, device)][['wX', 'wY']]
    p.moveAbsolute_um(x_home + wX, y_home + wY)

