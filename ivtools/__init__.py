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

#### Logging module configuration ####
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
logging_format = f'%(levelname)s : %(name)s : {username} : %(asctime)s : %(message)s'
logging_config = {
    'version': 1,
    'filters': {
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
        'debug': {
            'format': Fore.BLACK + logging_format + Style.RESET_ALL
        },
        'info': {
            'format': Fore.CYAN + logging_format + Style.RESET_ALL
        },
        'warning': {
            'format': Fore.YELLOW + logging_format + Style.RESET_ALL
        },
        'error': {
            'format': Fore.RED + logging_format + Style.RESET_ALL
        },
        'critical': {
            'format': Fore.MAGENTA + logging_format
        }
    },
    'handlers': {
        'debug_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['debug'],
            'formatter': 'debug'
        },
        'info_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['info'],
            'formatter': 'info'
        },
        'warning_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['warning'],
            'formatter': 'warning'
        },
        'error_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['error'],
            'formatter': 'error'
        },
        'critical_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['critical'],
            'formatter': 'critical'
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
        'handlers': ['debug_stream', 'info_stream', 'warning_stream', 'error_stream', 'critical_stream', 'file']
    },
    'loggers': {
        'settings': {
            'handlers': ['debug_stream', 'info_stream', 'warning_stream', 'error_stream', 'critical_stream', 'file'],
            'propagate': False
        },
        'interactive': {
            'handlers': ['debug_stream', 'info_stream', 'warning_stream', 'error_stream', 'critical_stream', 'file'],
            'propagate': False
        },
        'test': {
            'handlers': ['debug_stream', 'info_stream', 'warning_stream', 'error_stream', 'critical_stream', 'file'],
            'propagate': False
        }
    }
}
for lvl, val in logging_prints.items():
    logging_config['filters'][lvl]['show'] = val
logging.config.dictConfig(logging_config)


# this is so you can do
# import ivtools
# and then you have ivtools.plot, ivtools.analyze etc.
#from . import settings
#from . import analyze
#from . import plot
#from . import instruments
#from . import io
#from . import measure
