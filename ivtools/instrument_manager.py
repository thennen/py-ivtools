'''
Instrument connections should be reused when the class is instantiated again,
so that connections can be easily accessed from different parts of the code

But should different parts of the code have their own instances, so that they can have different state?

Want to enable changing the class code, and seemlessly updating the existing instances
The instance may also have some state that needs to be preserved

still want to retain the ability to have two instruments that use the same class (two connected keithley's for example)
'''
import visa
from collections import deque
#import weakref
#registrants = weakref.WeakValueDictionary()

visa_rm = visa.ResourceManager()
pico_state = {}

registrants = []
# We need a shared state that persists even after the class is redefined!
shared_state = {}

# function instrument_provider?

# function update_instances?  which manually overwrites the __class__ attribute of all registrants

def detect_instruments():
    ''' find which instruments are available to connect to '''
    # TCPIP instruments might be a pain
    pass

# Define parent class which registers all instances and defines behavior for reloading the class
class Instrument(object):
    ''' Writing default methods for visa type instruments.  Overload them for others '''
    def __init__(self, *args, **kwargs):
        # TODO:
        # Check for existing, connected instances with the same class name and the same init arguments
        # If one is found, this instance will be borg
        try:
            self.connect()
        except:
            # Say which instrument failed
            print('failed')

    def connect(self):
        # Could already be connected
        if hasattr(self, 'conn'):
            self.conn.whatever

        # Pass through some methods of the connection
        self.conn = None
        if hasattr(self.conn, 'close'):
            setattr(self, 'close', self.conn.close)
        if worked:
            pass
        else:
            pass

    def _reload(self):
        # Some default reload behavior
        pass

    def idn(self):
        return self.query('*IDN?').replace('\n', '')

    def isconnected():
        # is there a better way?
        #try:
            #self.idn()
            #return True
        #except:
            #return False
        return True if self.conn._session else False
