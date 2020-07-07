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
datafolder = ivtools.settings.datafolder
logging_file = ivtools.settings.logging_file
logging_dir = os.path.split(logging_file)[0]
logging_prints = ivtools.settings.logging_prints
os.makedirs(logging_dir, exist_ok=True)
logging_levels = {
    'DEBUG':       Fore.RED + logging_format + Style.RESET_ALL,
    'INFO':        Fore.RED + logging_format + Style.RESET_ALL,
    'WARNING':     Fore.RED + logging_format + Style.RESET_ALL,
    'ERROR':       Fore.RED + logging_format + Style.RESET_ALL,
    'CRITICAL':    Fore.RED + logging_format + Style.RESET_ALL,
    'instruments': Fore.BLACK + logging_format + Style.RESET_ALL,
    'io':          Fore.CYAN + logging_format + Style.RESET_ALL,
    'plots':       Fore.YELLOW + logging_format + Style.RESET_ALL,
    'analysis':    Fore.BLUE + logging_format + Style.RESET_ALL,
    'interactive': Fore.MAGENTA + logging_format + Style.RESET_ALL
}


class LevelFilter(logging.Filter):
    def __init__(self, name=None):
        self.name = name

    def filter(self, record):
        if record.levelname == self.name:
            allow = logging_prints[record.levelname]
        else:
            allow = False
        return allow


log = logging.getLogger('my_logger')
log.setLevel(1)

# File Handler
file_handler = logging.FileHandler(logging_file)
file_handler.setLevel(1)
file_formatter = logging.Formatter(logging_format)
file_handler.setFormatter(file_formatter)
log.addHandler(file_handler)

# Stream Handlers
for level in logging_levels.keys():
    handler = logging.StreamHandler()
    handler.setLevel(1)
    formatter = logging.Formatter(logging_levels[level])
    handler.setFormatter(formatter)
    handler.addFilter(LevelFilter(level))
    log.addHandler(handler)

custom_levels = list(logging_levels.keys())[5:]

for index, level in enumerate(custom_levels):
    def monkeymethod(self, message, index=index, *args, **kws):
        self._log(60 + index, message, args, **kws)
    setattr(logging.Logger, level, monkeymethod)
    logging.addLevelName(60 + index, level)

# this is so you can do
# import ivtools
# and then you have ivtools.plot, ivtools.analyze etc.
#from . import settings
#from . import analyze
#from . import plot
#from . import instruments
#from . import io
#from . import measure
