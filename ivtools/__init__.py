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

username = ivtools.settings.username
logging_format = f'%(levelname)s : {username} : %(asctime)s : %(message)s'

# Define custom logging levels and the format that they will be printed
# levelname: print format
custom_levels = {'instruments': Fore.BLACK + logging_format + Style.RESET_ALL,
                 'io':          Fore.CYAN + logging_format + Style.RESET_ALL,
                 'plots':       Fore.YELLOW + logging_format + Style.RESET_ALL,
                 'analysis':    Fore.RED + logging_format + Style.RESET_ALL,
                 'interactive': Fore.MAGENTA + logging_format + Style.RESET_ALL,
                }

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
logging_file = ivtools.settings.logging_file
logging_dir = os.path.split(logging_file)[0]
os.makedirs(logging_dir, exist_ok=True)
logging_prints = ivtools.settings.logging_prints
logging_config = {
    'version': 1,
    'filters': {
        **{k:{'()': LevelFilter,
               'level':60+i,
               'show':True} # prints all on by default
                for i,k in enumerate(custom_levels.keys())},
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
        **{k:{'format':v} for k,v in custom_levels.items()},
        'standard': {
            'format': logging_format
        }
    },
    'handlers': {
        **{f'{k}_stream':
           {'class': 'logging.StreamHandler',
            'filters':[k],
            'formatter': k}
            for k in custom_levels.keys()},
        'debug_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['debug'],
            'formatter': 'standard'
        },
        'info_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['info'],
            'formatter': 'standard'
        },
        'warning_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['warning'],
            'formatter': 'standard'
        },
        'error_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['error'],
            'formatter': 'standard'
        },
        'critical_stream': {
            'class': 'logging.StreamHandler',
            'filters': ['critical'],
            'formatter': 'interactive'
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
        'handlers': [f'{k}_stream' for k in custom_levels.keys()] +
                    ['debug_stream', 'info_stream', 'warning_stream', 'error_stream', 'critical_stream', 'file']
    }
}

for lvl, val in logging_prints.items():
    filter_config = logging_config['filters'].get(lvl)
    if filter_config is not None:
        filter_config['show'] = val

logging.config.dictConfig(logging_config)

# Monkeypatch Logger to have the custom level names as methods
for i,k in enumerate(custom_levels.keys()):
    def monkeymethod(self, message, *args, **kws):
        self._log(60+i, message, args, **kws)
    setattr(logging.Logger, k, monkeymethod)
for i,k in enumerate(custom_levels.keys()):
    logging.addLevelName(60+i, k)

log = logging.getLogger('root')

# this is so you can do
# import ivtools
# and then you have ivtools.plot, ivtools.analyze etc.
#from . import settings
#from . import analyze
#from . import plot
#from . import instruments
#from . import io
#from . import measure
