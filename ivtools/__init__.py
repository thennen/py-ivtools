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

# this is so you can do
# import ivtools
# and then you have ivtools.plot, ivtools.analyze etc.
#from . import settings
#from . import analyze
#from . import plot
#from . import instruments
#from . import io
#from . import measure
