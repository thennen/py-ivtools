import logging.config
from colorama import Fore, Back, Style
import os.path
import ivtools
from ivtools import settings
# Order matters, because of crazy circular imports..

#__all__ = ['settings', 'io', 'plot', 'analyze', 'measure', 'instruments']

# This holds the BORG instance states, to protect them from reload
# Often just for reusing the instrument connections
instrument_states = {}
# For MetaHandler, InteractiveFigs, ...
class_states = {}

# TODO: some way to export and load instrument states

def clear_instrument_states():
    global instrument_states
    instrument_states = {}

### Logging module configuration ###
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


datafolder = ivtools.settings.datafolder
username = ivtools.settings.username
logging_file = os.path.join(datafolder, 'logging_file.log')
os.makedirs(datafolder, exist_ok=True)
logging_prints = ivtools.settings.logging_prints
logging_format = f'%(levelname)s : {username} : %(asctime)s : %(message)s'
logging_config = {
    'version': 1,
    'filters': {
        'level1': {
            '()': LevelFilter,
            'level': 5,
            'show': True
        },
        'level2': {
            '()': LevelFilter,
            'level': 15,
            'show': True
        },
        'level3': {
            '()': LevelFilter,
            'level': 25,
            'show': True
        },
        'level4': {
            '()': LevelFilter,
            'level': 35,
            'show': True
        },
        'level5': {
            '()': LevelFilter,
            'level': 45,
            'show': True
        },
        'debug': {
            '()': LevelFilter,
            'level': 10,
            'show': True
        },
        'info': {
            '()': LevelFilter,
            'level': 20,
            'show': True
        },
        'warning': {
            '()': LevelFilter,
            'level': 30,
            'show': True
        },
        'error': {
            '()': LevelFilter,
            'level': 40,
            'show': True
        },
        'critical': {
            '()': LevelFilter,
            'level': 50,
            'show': True
        }
    },
    'formatters': {
        'standard': {
            'format': logging_format
        },
        'level1': {
            'format': Fore.BLACK + logging_format + Style.RESET_ALL
        },
        'level2': {
            'format': Fore.CYAN + logging_format + Style.RESET_ALL
        },
        'level3': {
            'format': Fore.YELLOW + logging_format + Style.RESET_ALL
        },
        'level4': {
            'format': Fore.RED + logging_format + Style.RESET_ALL
        },
        'level5': {
            'format': Fore.MAGENTA + logging_format
        }
    },
    'handlers': {
        'level1_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['level1'],
            'formatter': 'level1'
        },
        'level2_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['level2'],
            'formatter': 'level2'
        },
        'level3_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['level3'],
            'formatter': 'level3'
        },
        'level4_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['level4'],
            'formatter': 'level4'
        },
        'level5_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['level5'],
            'formatter': 'level5'
        },
        'debug_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['debug'],
            'formatter': 'level1'
        },
        'info_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['info'],
            'formatter': 'level2'
        },
        'warning_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['warning'],
            'formatter': 'level3'
        },
        'error_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['error'],
            'formatter': 'level4'
        },
        'critical_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['critical'],
            'formatter': 'level5'
        },
        'file': {
            'level': 5,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': logging_file
        }
    },
    'root': {
        'level': 5,
        'handlers': ['level1_stream', 'level2_stream', 'level3_stream', 'level4_stream', 'level5_stream',
                     'debug_stream', 'info_stream', 'warning_stream', 'error_stream', 'critical_stream', 'file']
    }
}
for lvl, val in logging_prints.items():
    logging_config['filters'][lvl]['show'] = val
logging.config.dictConfig(logging_config)


def level1(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    self._log(5, message, args, **kws)


def level2(self, message, *args, **kws):
    self._log(15, message, args, **kws)


def level3(self, message, *args, **kws):
    self._log(25, message, args, **kws)


def level4(self, message, *args, **kws):
    self._log(35, message, args, **kws)


def level5(self, message, *args, **kws):
    self._log(45, message, args, **kws)


logging.Logger.level1 = level1
logging.Logger.level2 = level2
logging.Logger.level3 = level3
logging.Logger.level4 = level4
logging.Logger.level5 = level5

logging.addLevelName(5, 'Level 1')
logging.addLevelName(15, 'Level 2')
logging.addLevelName(25, 'Level 3')
logging.addLevelName(35, 'Level 4')
logging.addLevelName(45, 'Level 5')


# this is so you can do
# import ivtools
# and then you have ivtools.plot, ivtools.analyze etc.
#from . import settings
#from . import analyze
#from . import plot
#from . import instruments
#from . import io
#from . import measure
