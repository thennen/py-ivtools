'''
This is a module for containing settings and state information of the program.
the point is so that we can reload every other module except this one, and anything stored here will persist

Note that this is not a typical config file -- it is programatically defined and shared with all users,
at least while the number of users is manageable.

We first define default settings and afterward there are hostname/username specific blocks
that can set hostname and user specific settings that don't affect anyone else.
'''
# TODO: some way to export and load settings
# to avoid circular imports, you shouldn't import anything here that uses the settings module on the top level
import getpass  # to get user name
import os
import socket
from importlib import reload

import ivtools.instruments as instruments
# circular import?
import ivtools.measure


# untested..
def reload():
    import ivtools.settings
    reload(ivtools.settings)

ivtools_dir = os.path.split(os.path.abspath(__file__))[0]
pyivtools_dir = os.path.split(ivtools_dir)[0]

#########################################################################################################################
#𝗗𝗲𝗳𝗮𝘂𝗹𝘁 𝘀𝗲𝘁𝘁𝗶𝗻𝗴𝘀 𝘁𝗵𝗮𝘁 𝗺𝗮𝘆 𝗴𝗲𝘁 𝗼𝘃𝗲𝗿𝘄𝗿𝗶𝘁𝘁𝗲𝗻 𝗯𝘆 𝗵𝗼𝘀𝘁𝗻𝗮𝗺𝗲 𝗼𝗿 𝘂𝘀𝗲𝗿 𝘀𝗲𝘁𝘁𝗶𝗻𝗴𝘀
#########################################################################################################################

# TODO: why did I put these in all caps?
### Settings for compliance circuit
COMPLIANCE_CALIBRATION_FILE = os.path.join(ivtools_dir, 'instruments', 'calibration', 'compliance_calibration.pkl')
CCIRCUIT_GAIN = 1930  # common base resistance * differential amp gain

# This is the channel where you are sampling the input waveform
MONITOR_PICOCHANNEL = 'A'

teo_calibration_file = os.path.join(ivtools_dir, 'instruments', 'calibration', 'teo_calibration.df')

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
# list of (Variable name, Instrument class name, *arguments to pass to class init) tuples
# TODO probably a dict of tuples would be better
inst_connections = []

# Shared metadatabase
db_path = os.path.join(pyivtools_dir, 'metadata.db')

# Shared logging file
logging_file = os.path.join(pyivtools_dir, 'logging.log')

# Settings for MikrOkular camera
savePicWithMeas = False
camSettings = {'brightness': 0.70,
               'contrast': 0.5,
               'hue': 0.5,
               'saturation': 0.50,
               'gamma': 0.5,
               'sharpness': 1.0,
               'exposure': 1.0}
camCompression = {"scale" : 0.5,
                  "quality" : 50}

saveAmbient = False

######################################################################################
# 𝗛𝗼𝘀𝘁𝗻𝗮𝗺𝗲 𝗮𝗻𝗱 𝘂𝘀𝗲𝗿 𝘀𝗽𝗲𝗰𝗶𝗳𝗶𝗰 𝘀𝗲𝘁𝘁𝗶𝗻𝗴𝘀
# 𝗠𝗮𝘆 𝗼𝘃𝗲𝗿𝗿𝗶𝗱𝗲 𝘁𝗵𝗲 𝗮𝗯𝗼𝘃𝗲 𝘀𝗲𝘁𝘁𝗶𝗻𝗴𝘀
######################################################################################

# 2634B : 192.168.11.11
# 2636A : 192.168.11.12
# 2636B : 192.168.11.13

if hostname in ('pciwe46', 'iwe21705'):

    datafolder = r'D:\data\{}'.format(username)
    db_path = 'D:\metadata.db'

    inst_connections = [('ps', instruments.Picoscope),
                        ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'),
                        ('rigol2', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T182500117::INSTR'),
                        #('teo', instruments.TeoSystem),
                        #('daq', instruments.USB2708HS),
                        ('ts', instruments.EugenTempStage),
                        ('dp', instruments.WichmannDigipot),
                        ('cam', instruments.MikrOkular, 0, camSettings),
                        ('amb', instruments.AmbientModule, "COM15"),
                        # ('keith', instruments.Keithley2600, 'TCPIP::192.168.11.11::inst0::INSTR'),
                        # ('keith', instruments.Keithley2600, 'TCPIP::192.168.11.12::inst0::INSTR'),
                        ('keith', instruments.Keithley2600)]  # Keithley can be located automatically now

    if username == 'hennen':
        autocommit = True
        datafolder = r'D:\t\ivdata'
        for di in logging_prints.values(): di['all'] = True # print everything

    elif username == 'mohr':
        #inst_connections.append(('teo', instruments.TeoSystem))
        savePicWithMeas = True
        saveAmbient = True

    elif username == 'munoz':
        munoz = 'D:/munoz/'
        datafolder = os.path.join(munoz, 'ivdata')
        db_path = os.path.join(munoz, 'Metadata/munoz_database.db')
        logging_file = os.path.join(munoz, 'ivtools_logging.log')
        logging_prints = {
            'instruments': {'all': None, 'DEBUG':False, 'INFO':True, 'WARNING':True, 'ERROR':True, 'CRITICAL':True},
            'io':          {'all': None, 'DEBUG':False, 'INFO':True, 'WARNING':True, 'ERROR':True, 'CRITICAL':True},
            'plots':       {'all': None, 'DEBUG':False, 'INFO':True, 'WARNING':True, 'ERROR':True, 'CRITICAL':True},
            'analyze':     {'all': None, 'DEBUG':False, 'INFO':True, 'WARNING':True, 'ERROR':True, 'CRITICAL':True},
            'measure':     {'all': None, 'DEBUG':False, 'INFO':True, 'WARNING':True, 'ERROR':True, 'CRITICAL':True},
            'interactive': {'all': None, 'DEBUG':False, 'INFO':True, 'WARNING':True, 'ERROR':True, 'CRITICAL':True}
        }
        inst_connections = [('ps', instruments.Picoscope),
                            ('k', instruments.Keithley2600),
                            ('dp', instruments.WichmannDigipot),
                            ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR')]
    else:
        datafolder = r'D:\{}\ivdata'.format(username)

elif hostname in ('pciwe38', 'iwe21407'):
    # Moritz computer
    datafolder = r'C:\Messdaten'
    inst_connections =  [('k', instruments.Keithley2600, 'GPIB0::27::INSTR'),
    ('ttx', instruments.TektronixDPO73304D ,'GPIB0::1::INSTR'),
    ('sympuls', instruments.Sympuls ,'ASRL3::INSTR')]
   # ('pg100', instruments.PG100 ,'ASRL3::INSTR')]

elif hostname == 'pcluebben2':
    datafolder = r'C:\data'
    inst_connections = [('k', instruments.Keithley2600, 'GPIB0::27::INSTR'),]

elif hostname == 'pciwe34':
    # Mark II
    # This computer and whole set up is a massive irredeemable piece of shit
    # computer crashes when you try to access the data drive
    # Data drive gets mounted on different letters for some reason
    # Therefore I will use the operating system drive..
    # datafolder = r'G:\Messdaten\hennen'
    datafolder = r'C:\Messdaten\hennen'
    inst_connections = [('et', instruments.Eurotherm2408),
                        ('k', instruments.Keithley2600, 'GPIB0::27::INSTR')]

elif hostname == 'CHMP2':
    datafolder = r'C:\data'
    inst_connections = [('ps', instruments.Picoscope),
                        ('rigol', instruments.RigolDG5000, 'USB0::0x1AB1::0x0640::DG5T161750020::INSTR'),
                        ('p', instruments.UF2000Prober, 'GPIB0::5::INSTR')]

elif username == 'alexgar':
    munoz = '/Users/alexgar/sciebo/munoz/'
    datafolder = os.path.join(munoz, 'ivdata')
    db_path = os.path.join(munoz, 'Metadata/munoz_database.db')
    logging_file = os.path.join(munoz, 'ivtools_logging.log')
    logging_prints = {
        'instruments': {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'io':          {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'plots':       {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'analyze':     {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'measure':     {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True},
        'interactive': {'all': None, 'DEBUG': True, 'INFO': True, 'WARNING': True, 'ERROR': True, 'CRITICAL': True}
    }

else:
    print(f'No Hostname specific settings found for {hostname}.  Using defaults.')
