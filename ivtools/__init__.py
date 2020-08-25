import logging.config
from colorama import Fore, Back, Style
import os.path
import ivtools
import ivtools.settings

# Order matters, because of crazy circular imports..

#__all__ = ['settings', 'io', 'plot', 'analyze', 'measure', 'instruments']

# Use this if you want ivtools to import all its modules right away
# e.g.:
# import ivtools (then you have ivtools.plot, ivtools.analyze etc.)
#from . import settings
#from . import analyze
#from . import plot
#from . import instruments
#from . import io
#from . import measure

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
stream_format = f'%(message)s'
file_format = f'%(asctime)s\t{username}\t%(name)s\t%(levelname)s\t\"%(message)s\"'
datafolder = ivtools.settings.datafolder
logging_file = ivtools.settings.logging_file
logging_dir = os.path.split(logging_file)[0]
logging_prints = ivtools.settings.logging_prints
os.makedirs(logging_dir, exist_ok=True)
'''
Colorama options:
    Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
    Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
    Style: DIM, NORMAL, BRIGHT, RESET_ALL
'''

loggers = {
    'instruments': Fore.GREEN + stream_format + Style.RESET_ALL,
    'io':          Fore.CYAN + stream_format + Style.RESET_ALL,
    'plots':       Fore.YELLOW + stream_format + Style.RESET_ALL,
    'analyze':     Fore.BLUE + stream_format + Style.RESET_ALL,
    'interactive': Fore.MAGENTA + stream_format + Style.RESET_ALL,
    'measure':     Fore.BLACK + stream_format + Style.RESET_ALL
}

logging_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


class LevelFilter(logging.Filter):
    def __init__(self, logger_name, level_name):
        self.level_name = level_name
        self.logger_name = logger_name

    def filter(self, record):
        if record.levelname == self.level_name:
            if self.logger_name not in logging_prints:
                return True
            if logging_prints[self.logger_name]['all'] is True:
                allow = True
            elif logging_prints[self.logger_name]['all'] is False:
                allow = False
            else:
                allow = logging_prints[self.logger_name][self.level_name]
        else:
            allow = False
        return allow

class LogFileFilter(logging.Filter):
    def filter(self, record):
        msg = record.msg
        if type(msg) is not str:
            msg = str(msg)
        record.msg = msg.replace('\n', '\\n').replace('\t', '\\t')
        return True


for logger in list(loggers.keys()):
    log = logging.getLogger(logger)
    log.setLevel(1)

    # Stream Handlers
    for level in logging_levels:
        handler = logging.StreamHandler()
        handler.setLevel(1)
        formatter = logging.Formatter(loggers[logger])
        handler.setFormatter(formatter)
        handler.addFilter(LevelFilter(logger, level))
        log.addHandler(handler)

    # File Handler
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(1)
    file_formatter = logging.Formatter(file_format)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(LogFileFilter())
    log.addHandler(file_handler)

