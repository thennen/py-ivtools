'''
This is a module for containing program global state
which persists on reload of other modules
the point is that you can reload everything else except this, so anything stored here can persist

You can store whatever settings you want here

There are hostname/username specific blocks that can set machine and user specific settings
without messing everyone else up

to avoid circular imports, you shouldn't import anything here that uses the settings module on the top level

TODO: some way to export and load settings
'''
import getpass  # to get user name
import socket
import os
from functools import partial
from importlib import reload
# circular import?
import ivtools.measure
import ivtools.instruments as instruments

# untested..
def reload():
    import ivtools.settings
    reload(ivtools.settings)

ivtools_dir = os.path.split(os.path.abspath(__file__))[0]
pyivtools_dir = os.path.split(ivtools_dir)[0]

#####################################################################################
######## Default settings that may get overwritten by hostname/user settings ########
#####################################################################################

# TODO: why did I put these in all caps?
### Settings for compliance circuit
COMPLIANCE_CURRENT = 0
INPUT_OFFSET = 0
COMPLIANCE_CALIBRATION_FILE = os.path.join(ivtools_dir, 'instruments', 'calibration', 'compliance_calibration.pkl')
CCIRCUIT_GAIN = 1930  # common base resistance * differential amp gain

# This is the channel where you are sampling the input waveform
MONITOR_PICOCHANNEL = 'A'

teo_calibration_file = os.path.join(ivtools_dir, 'instruments', 'calibration', 'teo_calibration.json')

# Drop these data columns before writing to disk
# usually if you need to save space and the columns can be recomputed
drop_cols = [] # ['I', 'V', 't']

### Change this when you change probing circuits - defines how to get from pico channels to I, V
# pico_to_iv = ivtools.measure.rehan_to_iv
# pico_to_iv = ivtools.measure.ccircuit_to_iv
pico_to_iv = ivtools.measure.Rext_to_iv # 50 ohm channel C
# pico_to_iv = ivtools.measure.TEO_HFext_to_iv

# More settings?
# Like picoscope defaults?
# For interactive mode?

hostname = socket.gethostname()
username = getpass.getuser()

datafolder = r'C:\data\{}'.format(username)

# Should interactive script automatically commit changes?
autocommit = False

'''
logging_print allows you to chose how verbose the logging module is. You can custom every level 
by setting 'all' to 'None', or you can set it to 'True' or 'False' to change all levels.
'''
logging_prints = {
    'instruments': {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
    'io':          {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
    'plots':       {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
    'analyze':     {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
    'measure':     {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
    'interactive': {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True}
}

# Specifies which instruments to connect to and what variable names to give them (for interactive script)
# Could also use it to specify different addresses needed on different PCs to connect to the same kind of instrument
# list of (Variable name, Instrument class name, *arguments to pass to class init)
inst_connections = []

db_path = os.path.join(pyivtools_dir, 'metadata.db')

logging_file = os.path.join(pyivtools_dir, 'logging.log')


#################################################
######## Hostname/user specific settings ########
#################################################

if hostname == 'pciwe46':
    db_path = 'D:\metadata.db'
    if username == 'hennen':
        autocommit = True
        datafolder = r'D:\t\ivdata'
        for di in logging_prints.values():
            di['all'] = True
    elif username == 'munoz':
        munoz = 'D:/munoz/'
        datafolder = os.path.join(munoz, 'ivdata')
        db_path = os.path.join(munoz, 'Thesis/Metadata/munoz_database.db')
        logging_prints = {
            'instruments': {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'io':          {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'plots':       {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'analyze':     {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'measure':     {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'interactive': {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True}
        }
    elif username == 'wtf':
        datafolder = r'D:\{}\ivdata'.format(username)
        logging_prints = {
            'instruments': {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'io':          {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'plots':       {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'analyze':     {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'measure':     {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
            'interactive': {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True}
        }
    else:
        datafolder = r'D:\{}\ivdata'.format(username)

    inst_connections = [('ps', instruments.Picoscope),
                        ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
                        ('rigol2', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T182500117::INSTR'),
                        #('teo', instruments.TeoSystem),
                        ('daq', instruments.USB2708HS),
                        ('ts', instruments.EugenTempStage),
                        ('dp', instruments.WichmannDigipot),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                        ('k', instruments.Keithley2600)]  # Keithley can be located automatically now
elif hostname == 'pciwe38':
    # Moritz computer
    datafolder = r'C:\Messdaten'
    inst_connections = {}
elif hostname == 'pcluebben2':
    datafolder = r'C:\data'
    inst_connections = [# ('et', instruments.Eurotherm2408),
                        # ('ps', instruments.Picoscope),
                        # ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
                        # ('daq', instruments.USB2708HS),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                        ('k', instruments.Keithley2600, 'GPIB0::27::INSTR')]
elif hostname == 'pciwe34':
    # Mark II
    # This computer and whole set up is a massive irredeemable piece of shit
    # computer crashes when you try to access the data drive
    # Data drive gets mounted on different letters for some reason
    # Therefore I will use the operating system drive..
    # datafolder = r'G:\Messdaten\hennen'
    datafolder = r'C:\Messdaten\hennen'
    inst_connections = [('et', instruments.Eurotherm2408),
                        # ('ps', instruments.Picoscope),
                        # ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
                        # ('daq', instruments.USB2708HS),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                        ('k', instruments.Keithley2600, 'GPIB0::27::INSTR')]
elif hostname == 'CHMP2':
    datafolder = r'C:\data'
    inst_connections = [('ps', instruments.Picoscope),
                        ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T161750020::INSTR'),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                        # ('k', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                        # TEO
                        ('p', instruments.UF2000Prober, 'GPIB0::5::INSTR')]
elif username == 'alexgar':
    munoz = '/Users/alexgar/sciebo/munoz/'
    datafolder = os.path.join(munoz, 'ivdata')
    db_path = os.path.join(munoz, 'Thesis/Metadata/munoz_database.db')
    logging_prints = {
        'instruments': {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'io':          {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'plots':       {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'analyze':     {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'measure':     {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'interactive': {'all': None, 'DEBUG': False, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True}
    }
else:
    print(f'No Hostname specific settings found for {hostname}.  Using defaults.')
