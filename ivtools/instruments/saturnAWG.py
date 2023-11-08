# Jari Klinkmann (c) 2023

from saturnpy.saturn_system import Saturn_System
from saturnpy.saturn_system_enum import Globaltrigger, Trigger_sources, System_state, Rhea_state
from saturnpy import rhea
from saturnpy.helpers import send_ascii_file
from typing import Final
from decimal import Decimal
import logging
log = logging.getLogger('instruments')


class saturn_0360(Saturn_System):
    # Forward declaration of channel objects
    # (not strictly necessary, but makes it easier to work with, 
    #  especially editors which support function autocompletion etc.)
    S1M1: rhea.DA_module
    S1M1R1: rhea.DA_channel
    S1M1R2: rhea.DA_channel
    S1M1R3: rhea.DA_channel
    S1M1R4: rhea.DA_channel

    def __init__(self, verbose=False):

        # register in ivtools
        #statename = self.__class__.__name__
        #if statename not in ivtools.instrument_states:
        #    ivtools.instrument_states[statename] = {}
        #self.__dict__ = ivtools.instrument_states[statename]

        # init the saturn
        super().__init__(verbose = verbose)
        self.connect_S2(ip=ss2_host_ip, port=ss2_host_port )

        # Add modules and/or channels to system object
        # Rhea module
        self.S1M1: Final[rhea.DA_module] = self.add_DA_module('S1M1', samplerate=Decimal('1e9'))