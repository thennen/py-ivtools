'''
We have a lot of instruments, so they are split into one .py file per instrument
each should contain at least one instrument class (usually with the same name as the python file)

Each of these classes contain functionality specific to only one instrument.
Don't put code in an instrument class that has anything to do with a different instrument,
or anything specific to a particular application!

They are classes, not modules, because we might have more than one of the same instrument type
(e.g. keithley, picoscope, or rigol), requiring multiple class instances.

Should only put instruments here that have an actual data connection to the computer

We use the Borg pattern to maintain instrument state (and reuse existing connections),
and we keep the state in a separate module (ivtools) so that it even survives reload of this module.

You can create an instance of these classes anywhere in your code, and they will automatically
reuse a connection if it exists, EVEN IF THE CLASS DEFINITION ITSELF HAS CHANGED.
One downside is that if you screw up the state somehow, you have to manually delete it to start over.
But one could add some kind of reset_state argument to __init__ to handle this.
'''
import os
import serial
import pyvisa as visa
from importlib import reload, import_module
import logging
import glob

log = logging.getLogger('instruments')

def ping(host):
    ping_param = "-n 1"
    # -c 1 on linux
    reply = os.popen("ping " + ping_param + " " + host).read()
    return "TTL=" in reply

def com_port_info():
    comports = serial.tools.list_ports.comports()

# Store visa resource manager in the visa module, so it doesn't get clobbered on reload
if not hasattr(visa, 'visa_rm'):
    try:
        visa.visa_rm = visa.ResourceManager()
    except ValueError as e:
        # don't raise exception if you didn't install visa
        log.error(e)
        visa.visa_rm = None
visa_rm = visa.visa_rm


'''
Import and reload all of the modules (instrument files)
and put their contents into the top level of instruments package
so we don't have to write instrumentname.instrumentname()
this is all ugly and offensive but I can't accept the normal behavior of
"from instruments import *" in this case

I don't want to write this:
    import instruments
    ps = instruments.Picoscope.Picoscope()
or
    from instruments import Picoscope
    ps = Picoscope.Picoscope()
I want this:
    import instruments
    ps = instruments.Picoscope()

    from instruments import *
    ps = Picoscope()
'''
files = os.listdir(os.path.dirname(__file__))
module_names = [m[:-3] for m in files if m.endswith('.py') and m[0] not in ('_', '#')]
modules = [import_module('.'+mn, 'ivtools.instruments') for mn in module_names]
imported_names = []
module_globals = set(globals())
for module in modules:
    reload(module) # so that reload(instruments) reloads all the files
    for thing_name in dir(module):
        if not thing_name.startswith('_'):
            thing = getattr(module, thing_name)
            if type(thing) is type: # only import the classes
                # allow collision only with the module names
                #if thing_name in module_globals - set(module_names) ^ set(imported_names):
                    #log.error(f'Name collision on import: {thing_name}')
                #else:
                    globals()[thing_name] = thing
                    imported_names.append(thing_name)
# These names get imported when you do "from instruments import *"
__all__ = imported_names

# TODO make parent class or decorator to implement the borg stuff.
# Then one could simply copy-paste the instrument classes and use them without the decorator
# def Borg():

# clean up namespace?
del reload, import_module, files, modules, module_globals
