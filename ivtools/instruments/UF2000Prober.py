import itertools
import time
import logging
log = logging.getLogger('instruments')
import visa
visa_rm = visa.visa_rm # stored here by __init__

class UF2000Prober(object):
    '''
    T Hennen modified 2018-12-11

    !!! Important !!!
    There are two ways to move the prober: By index, and by micron
    These two reference frames are centered on different locations
    indexing system is centered on a home device, so depends on how the wafer/coupon is loaded
    micron system is centered somewhere far away from the chuck

    The coordinate systems sound easy, but will confuse you for days
    in part because the coordinate system and units used for setting
    and getting the position are sometimes different and sometimes the same!!
    e.g. when reading the position in microns, the x and y axes are reflected!!

    I attempted to hide all of this nonsense from the user of this class
    !!!!!!

    UF2000 has its own device indexing system which requires some probably horrible setup that you need to do for each wafer.
    But we also have the option to specify directly position in micrometers, then we can handle the positioning here in the python universe.

    The indexing system is referenced to some home device, but the micron coordinate system is referenced to the chuck and centered god knows where.

    There are a few ways we can deal with this.  Right now I choose to deal with it outside of the class, so that it does not have any hidden state.

    UF2000Prober has no concept of what a device is or where they are located, except the home position is on the home device.

    Prober coordinate system is not intuitive, because the prober moves the chuck/wafer, not the probe, and has inverted Y axis
    But I like to think about moving the probe, and +X should be right, +Y should be up, in other words, the lab frame
    I will attempt to shield the user completely from the probers coordinate system, and always use the lab frame
    '''

    def __init__(self, idstring = 'GPIB0::5::INSTR'):
        self.inst = visa_rm.open_resource(idstring)
        self.inst.timeout = 3000
        # UF2000 seems to call this position home, could depend on the setup!
        self.home_indices = (128, 128)
        # Very roughly the center of the chuck...
        self.center_position_um = (160_126.3, 388_264.5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.zDn()
        log.debug('Closing UF2000')
        return False

    ####### Communication ########

    def write_8min(self, message, stbList):
        #will wait 8 min for stblist, with no breaking for stalls
        self.inst.write(message)
        self.errorCheck()
        return self.waitforStatusByte_8min(stbList)

    def write(self, message, stbList):

        self.inst.write(message)
        #self.errorCheck()
        return self.waitForStatusByte(stbList)

    def waitForSRQ_readStatus(self, timeout):
        self.inst.timeout = timeout
        self.inst.wait_for_srq()
        del self.inst.timeout
        return self.inst.read_stb()

    def waitforStatusByte_8min(self, stb ):
        #this function has no provision for 'stalls' it will wait for the stb until the for loop completes
        # this is necessary so that the 'stall' doesn't terminate the loop before the next wafer is loaded
        if type(stb) == int:
            stb = [stb]
        #
        old_a = 0
        #
        for _ in range(int(1E6)):
            a = self.inst.read_stb()


            """
            if a == 76:
                #Error!
                self.errorCheck()
                self.errorClearanceReq()

            """
            if a != old_a:
                log.debug('STB '+str(a)+': '+self.getSTBMessage(a))
                old_a = a
            if a in stb:
                return a

            time.sleep(.0005)
            pass
        else:
            #this will execute if the above for loop executes sucessfully--that is, 1e6 iterations were completed, and STB
            raise UF2000ProberError

    def waitForStatusByte(self, stb):
        if type(stb) == int:
            stb = [stb]
        #
        old_a = 0
        #
        for _ in range(int(1E6)):
            a = self.inst.read_stb()


            """
            if a == 76:
                #Error!
                self.errorCheck()
                self.errorClearanceReq()

            """
            if a != old_a:
                log.debug('STB '+str(a)+': '+self.getSTBMessage(a))
                old_a = a
            if (a in stb) :
                return a
            if a==3 and _ > 6000:
                log.info('**********\n**********\nSTB 3 Stall\n**********\n**********\n')
                return(a)

            if a==4 and _ >6000:  #ONLY FOR 'makeContact = No --comment out otherwise
                log.info('**********\n**********\nSTB 4 Stall\n**********\n**********\n')
                return(a)

            time.sleep(.0005)
            pass
        else:
            #this will execute if the above for loop executes sucessfully--that is, 1e6 iterations were completed, and STB was never 3 or 4
            raise UF2000ProberError

    def query(self, queryStr):
        for retry_nr in range(4):
            try:
                result = self.inst.query(queryStr)
                if result[0] != queryStr and queryStr != 'ms':
                    raise UF2000ProberError('Return String not well-formed:%s %s' %(queryStr, result))
                self.errorCheck()
                return result
            except pyvisa.errors.VisaIOError as e:
                #emailData("alexander.elias@gmail.com")
                traceback.print_exc()
                time.sleep(10)
                continue

    def waitForSTB(self):
        stb1 = str(self.inst.read_stb())
        #self.inst.wait_for_srq()

        stb2 = str(self.inst.read_stb())
        return int(stb1)

    def errorCheck(self):
        errorTypeDict = {'S': 'System Error: ',
                        'E': 'Error State: ',
                        'O': 'Operator Call: ',
                        'W': 'Warning Condition: ',
                        'I': 'Information: '}
        errorCodeDict = {'0650' : 'GPIB Receive Error ',
                         '0651': 'GPIB Transmit Error ',
                         '0660': 'GPIB Command Format Invalid ',
                         '0661': 'GPIB Command Execution Error ',
                         '0665': 'GPIB Stop Command Received ',
                         '0667': 'GPIB Communication Timeout Error ',
                         '0669': 'GPIB Timeout Error '}
        for retry_nr in range(1):
            try:
                rawError = self.inst.query('E')
                break
            except pyvisa.errors.VisaIOError as e:
                #emailData("alexander.elias@gmail.com")
                traceback.print_exc()
                time.sleep(10)
                continue

        if rawError[0] != 'E':
            raise UF2000ProberError('Return String not well-formed:%s %s' %(rawError, 'E'))

        errorTypeCode = rawError[1]
        errorCode = rawError[2:6]
        errorString = errorTypeDict.get(errorTypeCode , 'Unknown Type') + errorCodeDict.get(errorCode, 'Unknown Code: ') + errorCode
        #self.errorClearanceReq()
        return errorString

    def getID(self):
        '''returns prober ID string'''
        return self.query('B')
    def getSTBMessage(self, stbIntStr):
        stbDict = {'64': 'GPIB inital setting done',
                   '65': 'Absolute Value Travel Done',
                   '66': 'Coordinate Travel Done',
                   '67': 'Z-Up (Test Start)',
                   '68': 'Z-Down',
                   '69': 'Marking Done',
                   '70': 'Wafer Loading Done',
                   '71': 'Wafer Unloading Done',
                   '72': 'Lot End',
                   '74': 'Out of Probing Area',
                   '75': 'Prober Initial Setting Done',
                   '76': 'Error: Lock/Unlock cassette',
                   '77': 'Index Setting Done',
                   '78': 'Pass Counting Up/Execution Error',
                   '79': 'Fail Counting Up Done',
                   '80': 'Wafer Unloaded',
                   '81': 'Wafer End',
                   '82': 'Cassette End',
                   '84': 'Alignment Rejection Error',
                   '85': 'Stop Command Received',
                   '86': 'Print Data Receiving Done',
                   '87': 'Warning Error',
                   '88': 'Test Start (Count Not Needed)',
                   '89': 'Needle Cleaning Done',
                   '90': 'Probing Stop',
                   '91': 'Probing Start',
                   '92': 'Z-Up/Down Done',
                   '93': 'Hot Chuck Cont. Command Received',
                   '94': 'Lot Done',
                   '98': 'Command Normally Done',
                   '99': 'Command Abnormally Done',
                   '100': 'Test Done Received',
                   '101': '(em command correct end)',
                   '103': 'Map Data Downloading Normally Done',
                   '104': 'Map Data Downloading Abormally Done',
                   '105': 'Able To Adjust Needle Height',
                   '107': 'Binary Data Uploading',
                   '108': 'Binary Data Uploading Finish',
                   '110': 'Needle Mark OK',
                   '111': 'Needle Mark NG',
                   '112': 'Cassette Sensing Done',
                   '113': 'Re-Alignment Done',
                   '114': 'Auto Needle Alignment Normally Done',
                   '115': 'Auto Needle Alignment Abnormally Done',
                   '116': 'Chuck Height Setting Done',
                   '117': 'Continuous Fail Error',
                   '118': 'Wafer Loading Done',
                   '119': 'Error Recovery Done (Wafer Centering Complete)',
                   '120': 'Start Normally Done',
                   '121': 'Start Abnormally Done',
                   '122': 'Probe Mark Insapection Finish',
                   '123': 'Fail Mark Inspection Finish',
                   '124': 'Preload Done',
                   '125': 'Probing Stop by GEM Host',
                   '127': 'Travel Done',
                   '6': '6: Probing...',
                   '16': '16: Wafer Loading...',
                   '17': '17: Wafer Unloading...',
                   '30': '30: Waiting For New Cassette...',
                   '34': '34: Cassette Ready...'}
        return stbDict.get(str(stbIntStr), 'Unknown Status Byte: '+str(stbIntStr))

    def proberStatusReq(self):
        rawStatus =  self.query('ms')
        status = rawStatus[2]
        statusDict = {'W': 'Cassette Process Going on',
                      ' ':'Status process Not going on',
                      'I': 'Waiting for Lot Start',
                      'C': 'Card Replacement Going On',
                      'R': 'Lot Process Going On',
                      'E': 'Waiting For Operator\'s Help With an Error '}

        return statusDict[status], status

    def probeClean(self):
        '''intiate tip clean'''
        self.write('W', [89])

    def pushStart(self):
        self.write('st', [120,121])

    def stopTesting(self):
        self.write('K', [90,85,26])

    def errorClearanceReq(self):
        self.inst.write('es')

    def lotEndReq(self):
        self.write('le', [98,99])

    def getWaferID(self):
        '''returns wafer ID as scanned by prober OCR, removes leading/trailing character'''
        return self.query('b')[1:-1]

    def pollStatusByte(self, doneResponse):
        done = False
        while done == False:
            response = self.waitForSTB()
            time.sleep(1)
            log.info(response)
            ent = raw_input
            if response == doneResponse:
                done = True
        return


    ######## Wafer loading ########
    def loadWafer(self):
        return self.write_8min('L', [70,94])
        """
        if self.waitForStatusByte([2,17]) == 2:
            #print ('Needle Cleaning on Unit...')
        elif self.waitForStatusByte([2,17]) == 17:
            #print("Wafer Unloading...")
        else:
            #print('Loading Next Wafer...')
        """
    def unLoadWafer(self):
        #TODO increase timeout -- wafer takes a while to unload
        self.write('U', [71])

    ######## Temperature control ########
    def getChuckTemp(self):
        temp = self.inst.query('f1')
        return float(temp)

    def setChuckTemp(self, temp):
        if temp < 15 or temp > 150:
            log.warning('Temperature out of range, must be 15->150C')
            return

        temp = temp*10 #conver degree C to 0.1 C
        temp = str(temp).zfill(4)
        self.write('h{}'.format(temp), 93)

    def waitForTemp(self):
        self.inst.timeout = None
        tempStr = self.inst.query('f')
        if len(tempStr)!=11:
            log.warning('Hot Chuck not enabled!')
            raise UF2000ProberError


        currTemp =  float(tempStr[1:5])/10.
        setTemp =   float(tempStr[5:9])/10.
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        #sys.stdout.write('Waiting on STB ')
        #log.warning(('Waiting for Temperature = {}...'.format(setTemp)))
        while abs(currTemp-setTemp)>0.2:

            time.sleep(10)
            # sys.stdout.write(spinner.next())  # write the next character
            # sys.stdout.flush()                # flush stdout buffer (actual character display)
            # time.sleep(0.33)
            # sys.stdout.write('\b')            # erase the last written char
            tempStr = self.inst.query('f')
            log.debug(tempStr)
            currTemp =  float(tempStr[1:5])/10.
            setTemp =   float(tempStr[5:9])/10.
            log.info(('Set Temp: {}, CurrentTemp: {}, Diff: {}'.format(setTemp, currTemp, abs(currTemp-setTemp))))
        self.inst.timeout = 3000
        log.info('Set Temperature Achieved!')


    ######## Movement ########
    def zDn(self):
        '''moves chuck to NO_CONTACT position'''
        #hp.shortAll()
        self.write('D', [68])
        #self.pollStatusByte(68)
        #self.waitForSTB()
        pass

    def zUp(self):
        '''moves chuck unto CONTACT position'''
        #hp.shortAll()
        log.debug('Zupping...')
        self.write('Z', [67])
        #self.pollStatusByte(67)
        #self.waitForSTB()
        pass

    def goHome(self):
        # Prober seems to call home position 128, 128.  Could be wrong!
        # I think it depends on the set up
        self.moveAbsolute(0, 0)

    # Reference frame conversion
    def prober_to_lab_indices(self, xprober, yprober):
        xlab = -xprober + self.home_indices[0]
        ylab = yprober - self.home_indices[1]
        return xlab, ylab

    def lab_to_prober_indices(self, xlab, ylab):
        xprober = -xlab + self.home_indices[0]
        yprober = ylab + self.home_indices[1]
        return xprober, yprober

    def prober_to_lab_um(self, xprober, yprober):
        xlab = xprober - self.center_position_um[0]
        ylab = -yprober + self.center_position_um[1]
        return xlab, ylab

    def lab_to_prober_um(self, xlab, ylab):
        xprober = xlab + self.center_position_um[0]
        yprober = -ylab + self.center_position_um[1]
        return xprober, yprober

    # Index based -- moves by unit cell and has only integer values
    def getPosition(self):
        '''
        Get position indices
        Home position is subtracted
        converted to +Y up, +X right in the lab frame
        '''
        rawPosString = self.query('Q')
        y = int(rawPosString[2:5])
        x = int(rawPosString[6:9])
        return self.prober_to_lab_indices(x, y)

    def moveAbsolute(self, absX, absY):
        '''
        Move to a given index in the lab frame
        '''
        X0, Y0 = self.getPosition()
        if (X0, Y0) == (absX, absY):
            return (X0, Y0)
        Xrel = absX - X0
        Yrel = absY - Y0
        self.moveRelative(Xrel, Yrel)
        newPos = self.getPosition()
        return newPos

    def moveRelative(self, x_rel, y_rel):
        '''
        Moves by whatever the prober thinks is the unit cell distance
        with respect to a view of the top of the wafer from the front of the machine:
        +X moves probe right
        +Y moves probe up
        '''
        if((x_rel, y_rel) == (0,0)):
            return self.getPosition()
        x_rel_prober = -x_rel
        y_rel_prober = y_rel
        strX = '%+04d' % x_rel_prober
        strY = '%+04d' % y_rel_prober
        moveString = 'SY'+ strY + 'X' + strX
        self.write(moveString, [66,67,74])
        return self.getPosition()


    # Micron based
    def getHomePosition_um(self):
        '''
        This is to center the micron coordinate system on the home device
        Does not set self.center_position_um, but you could do that
        '''
        self.goHome()
        home = self.getPosition_um()
        return home

    def getPosition_um(self):
        '''
        The position UF2000 thinks it is in, in um
        '''
        pos_str = self.query('R')
        # Manual says the unit is 1e-7 meter
        y, x = int(pos_str[2:9])/10, int(pos_str[10:-2])/10
        return self.prober_to_lab_um(x, y)

    def moveRelative_um(self, xum_rel, yum_rel):
        '''
        with respect to a view of the top of the wafer from the front of the machine:
        +X moves probe right
        +Y moves probe up
        '''
        xum_rel_prober = -xum_rel
        yum_rel_prober = yum_rel
        str_xum = '{:+07d}'.format(int(round(xum_rel_prober)))
        str_yum = '{:+07d}'.format(int(round(yum_rel_prober)))
        # manual says the unit is 1e-6 meter
        moveString = 'AY{}X{}'.format(str_yum, str_xum)
        self.write(moveString, [65, 67, 74])

    def moveAbsolute_um(self, xum_abs, yum_abs):
        xum_curr, yum_curr = self.getPosition_um()
        log.info(('Current position:     {}, {}'.format(xum_curr, yum_curr)))
        log.info(('Destination position: {}, {}'.format(xum_abs, yum_abs)))
        xum_rel = int(xum_abs - xum_curr)
        yum_rel = int(yum_abs - yum_curr)
        self.moveRelative_um(xum_rel, yum_rel)
