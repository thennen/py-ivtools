'''
This is a module for containing program global state
which persists on reload of other modules

Also storing some settings here that one might want to change
TODO: separate settings and persistent state, because we might want to reload settings separately

I don't know if it's a dumb idea, but it seems to do what I want
'''
import visa
import os
from functools import partial
visa_rm = visa.ResourceManager()

ivtools_dir = os.path.split(os.path.abspath(__file__))[0]
pyivtools_dir = os.path.split(ivtools_dir)[0]

### Settings for compliance circuit
COMPLIANCE_CURRENT = 0
INPUT_OFFSET = 0
MONITOR_PICOCHANNEL = 'A'
COMPLIANCE_CALIBRATION_FILE = os.path.join(pyivtools_dir, 'compliance_calibration.pkl')
CCIRCUIT_GAIN = -2000 # common base resistance * differential amp gain

### Change this when you change probing circuits - defines how to get from pico channels to I, V
# circular import?
from . import measure
#pico_to_iv = measure.rehan_to_iv
#pico_to_iv = measure.ccircuit_to_iv
pico_to_iv = partial(measure.Rext_to_iv, R=50)
#pico_to_iv = measure.TEO_HFext_to_iv

#TODO: figure out how to handle settings all in one dedicated file, such that git doesn't mess it up
# maybe do a "switch statement" on hostname here, to override the defaults above

# More settings?
# For interactive mode?
# data folder?
# instrument connections


# These hold the BORG instance states, to protect them from reload
pico_state = {}
plotter_state = {}
metahandler_state = {}
eurotherm_state = {}
keithley_state = {}
digipot_state = {}
