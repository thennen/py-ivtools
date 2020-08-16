import logging
log = logging.getLogger('instruments')


class USB2708HS(object):
    # Measurement Computing USB1208HS DAQ ###################
    def __init__(self):
        # Import here because I don't want the entire module to error if you don't have mcculw installed
        from mcculw import ul
        from mcculw import enums
        self.ul = ul
        self.enums = enums

    def analog_out(self, ch, dacval=None, volts=None):
        '''
        I found a USB-1208HS so this is how you use it I guess.
        Pass a digital value between 0 and 2**12 - 1
        0 is -10V, 2**12 - 1 is 10V
        Can also pass volts instead of dacval
        Voltage values that don't make sense for my current set up are disallowed.
        '''
        board_num = 0
        ao_range = self.enums.ULRange.BIP10VOLTS

        # Can pass dacval or volts.  Prefer dacval.
        if dacval is None:
            # You better have passed volts...
            dacval = self.ul.from_eng_units(board_num, ao_range, volts)
        else:
            dacval = int(dacval)
            volts = self.ul.to_eng_units(board_num, ao_range, dacval)

        log.info('Setting analog out {} to {} ({} V)'.format(ch, dacval, volts))

        try:
            self.ul.a_out(board_num, ch, ao_range, int(dacval))
        except ULError as e:
            # Display the error
            log.error("A UL error occurred. Code: " + str(e.errorcode)
                  + " Message: " + e.message)


    def digital_out(self, ch, val):
        #ul.d_config_port(0, DigitalPortType.AUXPORT, DigitalIODirection.OUT)
        self.ul.d_config_bit(0, self.enums.DigitalPortType.AUXPORT, 8, self.enums.DigitalIODirection.OUT)
        self.ul.d_bit_out(0, self.enums.DigitalPortType.AUXPORT, ch, val)
