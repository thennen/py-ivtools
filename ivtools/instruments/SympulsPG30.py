# Copyright 2023 Faisal Munir, Jari Klinkmann, RWTH Aachen University

import logging
log = logging.getLogger('instruments')
import pyvisa as visa
visa_rm = visa.visa_rm # stored here by __init__
import time
import math
import inspect

def _sendCmd(interface, cmd): 
    """ Sends a command to an instrument by first encoding it into utf-8 bytes
        and then using the pyvisa write_raw function to send it.
        Parameters
        ----------
        interface : <class 'pyvisa.resources.serial.SerialInstrument'>
        instrument instance to which to send the command.
        cmd : str
        the command for the instrument (usually SCPI).
        Returns
        -------
        None.
    """
    # cmd must be complete with '\n' ending
    interface.write_raw(bytes(bytearray(cmd+"\n", 'utf-8')))

def _query (interface, queryString):
    """
    Sends a query to an instrument and collects response using pyvisa
    query function.

    Parameters
    ----------
    interface : <class 'pyvisa.resources.serial.SerialInstrument'>
        instrument instance to which to send the command.
    queryString : str
        the query command for the instrument (usually SCPI).

    Returns
    -------
    response : str
        Response received from the instrument.

    """
    response = interface.query(queryString)
    time.sleep(0.1)
    return response

def _sendBlockDataCmd (interface, cmd: str, databytes: bytearray): 
    """
    Sends a command with block data to an instrument by first encoding it into
    utf-8 bytes and then appending " #DL" (also encoded into utf-8) where D
    stands for the number of digits of L where L is the number of bytes of 
    the block data. 

    Parameters
    ----------
    interface : <class 'pyvisa.resources.serial.SerialInstrument'>
        instrument instance to which to send the command.
    cmd : str
        the command for the instrument.
    databytes : bytearray
        the block data to send to instrument.

    Returns
    -------
    None.

    """
    # cmd must not include '\n'
    L=len(databytes)
    ba = bytearray(cmd, 'utf-8')
    ba += bytearray(' #' + str(len(str(L))) + str(L), 'utf-8')
    ba += databytes
    ba += bytes('\n','utf-8')
    interface.write_raw(bytes(ba))
    time.sleep(0.02)

def _queryBlockData (interface, cmd: str):
    """
    Sends a query to an instrument and collects response as block data (in 
    bytes).

    Parameters
    ----------
    interface : <class 'pyvisa.resources.serial.SerialInstrument'>
        instrument instance to address query to.
    cmd : str
        the query command to send to instrument.

    Returns
    -------
    response : raw data in bytes (bytearray?)
        Response data received from the instrument.

    """
    _sendCmd(interface, cmd)
    response = interface.read_raw()
    return response

def _readErrorQueue (interface):
    """
    Query error queue of device and then clear it.

    Parameters
    ----------
    interface : <class 'pyvisa.resources.serial.SerialInstrument'>
        instrument instance to which to address query to.

    Returns
    -------
    error : str
        response received from instrument

    """
    error = _query (interface, ":system:error:next?")
    _sendCmd(interface, "*CLS\n")
    return error
 
def _to_patternbytes(pattern):
    """
    Convert a pattern encoded into a string or a list to a bytearray. If
    the length does not fully fill one (or multiple) words, the input is
    padded with zeros. One word corresponds to 128 digits or 32 bytes.
    
    For STRING INPUT the expected format is a binary number sequence, e.g.
    "01011000". Here "00" stands for no voltage, "10" stands for a negative 
    pulse, "01" stands for a positive pulse, and "11" is invalid.
    
    For LIST INPUT the expected format is a list with elements 1, -1, 0. Here
    1 is a positive pulse, 0 is no pulse, -1 is a negative pulse. An example
    list would be [1, -1, 0, 0]

    Parameters
    ----------
    pattern : str OR list
        pattern represented as string or list as described above.

    Returns
    -------
    bytearray
        pattern as bytearray.

    """
    wordWidth=256 # Internal word width of Data in bits (32bytes)
    
    if isinstance(pattern, bytearray):
        return pattern
    
    # for list expect digits
    elif isinstance(pattern, list):
        # there is 128 digits in 1 word, every unfilled word is padded with 0s
        L = math.ceil(len(pattern)/128)
        n = L*128 - len(pattern)
        pattern += [0]*n # pad pattern with zeros
        # create bytearray to fill with bytes
        d = bytearray(int(wordWidth/8) * L)
        for index, byte in enumerate(list(_chunks(pattern, 4))):
            # convert each 4 digits to string with 8 binary numbers
            # as in the binary number string encoding of the pattern
            # we reverse the byte order because join joins the opposite
            # direction from the way we read the number
            byte = ''.join(map(_digit_to_bits, byte[::-1]))
            # convert 8 digits of binary number string to byte
            d[index] = int(byte, 2)
            
    # for string expect bits
    elif isinstance(pattern, str):
        # there is 256 bits in 1 word, every unfilled word is padded with 0s
        L = math.ceil(len(pattern)/256)
        n = L*256 - len(pattern)
        pattern += "0"*n # pad pattern with zeros
        # create bytearray to fill with bytes
        d = bytearray(int(wordWidth/8) * L)
        for index, byte in enumerate(list(_chunks(pattern, 8))):
            # convert 8 digits of binary number string to byte
            d[index] = int(byte, 2)
            
    return d

def _digit_to_bits(digit: int) -> str:
    """
    Convert a digit to its binary number representation. 
    1 : "01"
    0 : "00"
    -1 : "10"

    Parameters
    ----------
    digit : int
        The digit to be converted. Can be 1, 0, or -1.

    Raises
    ------
    Exception
        Raises an exception if the input is not 1, 0, or -1.

    Returns
    -------
    str
        Returns the binary number representation of the digit.

    """
    if digit == 1:
        return "01"
    elif digit == 0:
        return "00"
    elif digit == -1:
        return "10"
    else:
        raise Exception(f"{digit} is not a valid digit")
            
# TODO: for large patterns this function is not feasible, use multiprocessing
# or find more efficient algorithm        
def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    if len(lst) % n != 0:
        print(
            "Input has invalid length. It should be divisible by chunk length."
        )
        lst = lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _read_errors(func):
    """
    If debug attribute of class instance is True, then for any function 
    decorated with @_read_errors calls the error method of the class after 
    executing that function.
    For SympulsPPG this reads and prints the error queue subsequently to the
    action of the decorated function so that one can know whether the action
    (the command) was succesful.

    Parameters
    ----------
    func : function
        The function to be decorated with @_read_errors.

    Returns
    -------
    function
        Returns function that calls the error method of the class after
        executing the decorated function.

    """
    def wrapper_func(*args, **kwargs):
        # Invoke the wrapped function first
        retval = func(*args, **kwargs)
        # Now do something here with retval and/or action
        if args[0].debug:
            args[0].error()
        return retval
    wrapper_func.__doc__ = func.__doc__
    wrapper_func.__signature__ = inspect.signature(func)
    return wrapper_func
    
class SympulsPG30(object):
    
    def __init__(self, addr='ASRL5::INSTR', debug=False):
        try:
            self.connect(addr)
            self.debug = debug
            self.past_errors = ""
            print(f"Past errors: {self.error()}")
        except:
            log.error('Sympuls connection failed at {}'.format(addr))
        self.debug = debug
        self.past_errors = ""

    def connect(self, addr):
        self.conn = visa_rm.get_instrument(addr)
        # Expose a few methods directly to self
        self.write = self.conn.write
        self.query = self.conn.query
        self.ask = self.query
        self.read = self.conn.read
        self.read_raw = self.conn.read_raw
        self.close = self.conn.close
        self.conn.write_termination ='\n'
        
    @_read_errors    
    def set_upattern(self, pattern):
        '''
        sets upattern of Sympuls PPG. pattern can be string of form
        01010000, list of form [1, -1, 0, 0] or bytearray.
        Note that upattern length is fixed to 1 word = 32 bytes

        '''
        # convert pattern to block data
        pattern = _to_patternbytes(pattern)
        # check that length is 32 bytes
        if not len(pattern) == 32:
            raise Exception(
                "Please note that upattern length is fixed to"
                f" 1 word = 32 bytes but got {len(pattern)}bytes"
            )
        # transfer pattern
        _sendBlockDataCmd(
            self.conn, "source:upattern:data", 
            pattern
        )
    
    @_read_errors    
    def set_lupattern(self, pattern):
        '''sets lupattern of Sympuls PPG. pattern can be string of form
        "01010000", list of form [1, -1, 0, 0] or bytearray.
        Lupattern length can be between 1...4 194 304 / divisor words'''
        # convert pattern to block data
        pattern = _to_patternbytes(pattern)
        # check that length requirements are met
        if not len(pattern) % 32 == 0:
            raise Exception(
                "Please note that lupattern length must be multiple of 32"
                " bytes i.e. an integer number of words,"
                f" but got {len(pattern)} bytes"
            )
        num_words = len(pattern) // 32
        divisor = int(self.get_lupattern_divisor())
        if not num_words <= 4194304 // divisor:
            raise Exception(
                "Please not that lupattern length must not exceed"
                " 4 194 304 / divisor words but got {num_words} >"
                f" 4194304 / {divisor} = {4194304/divisor}"
            )
        # transfer pattern
        _sendBlockDataCmd(
            self.conn, "source:lupattern:data", 
            pattern
        )
        # adjust the pattern length
        self.set_lupattern_length(num_words)


    @_read_errors
    def get_upattern(self)->str:
        '''queries upattern from Sympuls PPG and returns it as string.'''
        data = _queryBlockData (self.conn, ":source:upattern:data?")
        # cut off introductory bytes at the beginning and \n at the end
        # beginning has form #(number digits of length)(length of pattern)
        # end is \n (0x0a in utf-8)
        data = data[4:-1]
        # make pattern data string
        pattern = ' '.join(format(x, '02x') for x in data)      
        return f"{pattern=}, length={len(data)}bytes={len(data)//32}word(s)"
        
    @_read_errors
    def idn(self):
        '''asks name from Sympuls PPG and returns it as string.'''
        idn = self.query('*IDN?')
        # self.read()
        return idn.replace('\n', '')

    def error(self):
        '''prints the last error'''
        error_msg = self.query(':SYST:ERR:NEXT?')
        new_error = time.asctime() + ": " + error_msg
        log.info(error_msg)
        self.write('*CLS')
        self.past_errors += new_error
        return new_error

    def get_error_log(self):
        print(self.past_errors)

    @_read_errors
    def freq(self, freq):
        '''Define Freqeuncy betwwen 200MHz -- 15GHz'''
        self.write(':SOUR:FREQ '+ str(freq))
        ''' .format is a method to call string, {} is used to add space'''

    @_read_errors
    def set_pulse_width(self, pulse_width):
        '''Define pulse width betwwen 200MHz (400MHz/2.5ns for 1 digit) -- 15GHz (33.33ps for 1 digit) Clock= bit clock/2'''
        ''' Device capability is between 66ps-5ns, which 15GHz-200MHz in freq'''
        ### We divide pulse width by 2, so that we can avoid 2 digits in pattern
        if pulse_width < 33e-12 or pulse_width > 2.5e-9:
            raise Exception('pulse width should be between 5ns and 66ps')
        self.write(':SOUR:FREQ '+ str(1/(2*pulse_width)))
        Frequency = str((1/(2*pulse_width))/1e9)+(str(' GHz'))
        return Frequency

    @_read_errors
    def amplitude1(self, Amp1):
        '''Define Postive amplitude betwwen 1000mV --  2000mV for 
        example Amplitude1('1000mV') or (1000e-3)'''
        #self.write(':OUTput3:AMPLitude1  {}' .format(Amp1))
        self.write(':OUTput3:AMPLitude1 '+ str(Amp1))
        ''' .format is a method to call string, {} is used to add space'''

    @_read_errors
    def amplitude2(self, Amp2):
        '''Define negative amplitude betwwen 600mV -- 1200mV for example 
        Amplitude2('1000mV')'''
        #self.write(':OUTput3:AMPLitude1  {}' .format(Amp2))
        self.write(':OUTput3:AMPLitude2 '+ str(Amp2))
        ''' .format is a method to call string, {} is used to add space'''      

    @_read_errors
    def format(self, form):
        '''Define format as an BIP or NRZ output for example Format('NRZ') 
        and BIP for bipolar'''
        #self.write(':SOUR:FORM {}' .format(form))
        self.write(':SOUR:FORM '+ str(form))
        ''' .format is a method to call string, {} is used to add space'''

    @_read_errors
    def pattern(self, pattern):
        '''Define Pattern as an LUPattern (for FWRM) or UPATtern (for FWRAM 120) 
        or TESTPATTERN example Pattern('LUPATTERN')'''
        if pattern == 'LU':
            self.write(':SOUR:FUNC LUP')
        elif pattern == 'UP':
            self.write(':SOUR:FUNC UPAT')
        elif pattern == 'Test':
            self.write(':SOUR:FUNC TESTPATTERN')
        else:
            log.info('Unknown trigger type. Make sure it is \'LUP\', \'UPAT\' or \'TEST\'')

    
    @_read_errors
    def sub_pattern(self, subpattern):
        '''Define format as an LUPATTERN (for FWRM) or UPATTERN (for FWRAM 120)
        or TESTPATTERN example Pattern('LUPATTERN')'''
        self.write(':SOUR:FUNC {}' .format(subpattern))
        ''' .format is a method to call string, {} is used to add space'''
    
    @_read_errors
    def set_lupattern_length(self, num_words):
        '''Define length of the long pattern (lupattern) in words.'''
        '''1 word = 128 digits, 1 digits = 2bits, 1 byte = 4 digits = 8 bits
        minimum word = 1 , maximum word = 4194304'''
        divisor = int(self.get_lupattern_divisor())
        if num_words > 4194304 // divisor or num_words < 1:
            raise Exception('lupattern length should be between 1 and 4194304 / divisor length')

        self.write(f"source:lupattern:length {num_words}")
    
    @_read_errors
    def get_lupattern_length(self):
        '''Query length of the long pattern (lupattern) in words'''
        return _query(self.conn, "source:lupattern:length?").replace("\n", "")
    
    @_read_errors
    def set_lupattern_divisor(self, divisor):
        '''Set divisor (number of patterns in lupattern storage).
        Divisor can be 1,2,4, or 8.'''
        self.write(f":source:lupattern:divisor {divisor}")
    
    @_read_errors
    def get_lupattern_divisor(self):
        '''Query divisor (number of patterns in lupattern storage).
        Divisor can be 1,2,4, or 8'''
        return _query(self.conn, "source:lupattern:divisor?").replace("\n", "")

    @_read_errors
    def trigger_source(self, trig):
        '''Define trigger as AUTO or IMMediate or EXTernal example 
        TrigSource('IMM')'''
        self.write(':TRIG2:SOUR {}' .format(trig))
        ''' .format is a method to call string, {} is used to add space'''

    @_read_errors
    def trigger(self):
        '''When Trigger Ssource is IMM, then you can trigger manually by just 
            Trigger()
        '''
        self.write(':INIT:IMM')
        ''' .format is a method to call string, {} is used to add space''' 
    @_read_errors 
    def reset(self):
        '''Reset sympuls to factory defaults.'''
        self.write("*RST")
        

            
