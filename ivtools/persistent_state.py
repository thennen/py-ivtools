'''
This is a module for containing program global state
which persists on reload of other modules

I don't know if it's a dumb idea, but I 
'''
import visa

COMPLIANCE_CURRENT = 0
INPUT_OFFSET = 0
MONITOR_PICOCHANNEL = 'A'

visa_rm = visa.ResourceManager()
pico_state = {}
plotter_state = {}
metahandler_state = {}
eurotherm_state = {}
keithley_state = {}
