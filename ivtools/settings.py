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
import logging.config
from colorama import Fore, Back, Style
# circular import?
import ivtools.measure
import ivtools.instruments as instruments


ivtools_dir = os.path.split(os.path.abspath(__file__))[0]
pyivtools_dir = os.path.split(ivtools_dir)[0]

#####################################################################################
######## Default settings that may get overwritten by hostname/user settings ########
#####################################################################################

### Settings for compliance circuit
COMPLIANCE_CURRENT = 0
INPUT_OFFSET = 0
COMPLIANCE_CALIBRATION_FILE = os.path.join(ivtools_dir, 'calibration', 'compliance_calibration.pkl')
CCIRCUIT_GAIN = 1930  # common base resistance * differential amp gain

# This is the channel where you are sampling the input waveform
MONITOR_PICOCHANNEL = 'A'

### Change this when you change probing circuits - defines how to get from pico channels to I, V
# pico_to_iv = ivtools.measure.rehan_to_iv
# pico_to_iv = ivtools.measure.ccircuit_to_iv
pico_to_iv = partial(ivtools.measure.Rext_to_iv, R=50)
# pico_to_iv = ivtools.measure.TEO_HFext_to_iv

# More settings?
# For interactive mode?

hostname = socket.gethostname()
username = getpass.getuser()

datafolder = r'C:\data\{}'.format(username)
# Specifies which instruments to connect to and what variable names to give them (for interactive script)
# Could also use it to specify different addresses needed on different PCs to connect to the same kind of instrument
# list of (Variable name, Instrument class name, *arguments to pass to class init)
inst_connections = []

db_path = os.path.join(ivtools_dir, 'metadata.db')


### Settings for the logging module
class LevelFilter(logging.Filter):
    def __init__(self, level=None, show=True):
        self.level = level
        self.show = show  # Define if the log will be printed or not

    def filter(self, record):
        if record.levelno == self.level and self.show:
            allow = True
        else:
            allow = False
        return allow


log_format = '%(levelname)s : %(asctime)s : %(message)s'

log_config = {
    'version': 1,
    'filters': {
        'debug_filter': {
            '()': LevelFilter,
            'level': 10,
            'show': True
        },
        'info_filter': {
            '()': LevelFilter,
            'level': 20,
            'show': True
        },
        'warning_filter': {
            '()': LevelFilter,
            'level': 30,
            'show': True
        },
        'error_filter': {
            '()': LevelFilter,
            'level': 40,
            'show': True
        },
        'critical_filter': {
            '()': LevelFilter,
            'level': 50,
            'show': True
        }
    },
    'formatters': {
        'standard': {
            'format': log_format
        },
        'black': {
            'format': Fore.BLACK + log_format + Style.RESET_ALL
        },
        'blue': {
            'format': Fore.CYAN + log_format + Style.RESET_ALL
        },
        'yellow': {
            'format': Fore.YELLOW + log_format + Style.RESET_ALL
        },
        'red': {
            'format': Fore.RED + log_format + Style.RESET_ALL
        },
        'magenta': {
            'format': Fore.MAGENTA + log_format + Style.RESET_ALL
        },
        'back_magenta': {
            'format': Back.MAGENTA + Fore.BLACK + log_format + Style.RESET_ALL
        },
    },
    'handlers': {
        'debug_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['debug_filter'],
            'formatter': 'black'
        },
        'info_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['info_filter'],
            'formatter': 'blue'
        },
        'warning_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['warning_filter'],
            'formatter': 'yellow'
        },
        'error_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['error_filter'],
            'formatter': 'red'
        },
        'critical_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['critical_filter'],
            'formatter': 'magenta'
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'LogFile.log'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['debug_stream', 'info_stream', 'warning_stream', 'error_stream', 'critical_stream', 'file']
    },
}

logging.config.dictConfig(log_config)

#################################################
######## Hostname/user specific settings ########
#################################################

if hostname == 'pciwe46':
    db_path = 'D:\metadata.db'
    if username == 'hennen':
        datafolder = r'D:\t\ivdata'
    else:
        datafolder = r'D:\{}\ivdata'.format(username)
    inst_connections = [('ps', instruments.Picoscope),
                        ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
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
else:
    print(f'No Hostname specific settings found for {hostname}.  Using defaults.')
