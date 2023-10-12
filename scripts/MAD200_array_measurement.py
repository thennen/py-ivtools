# wifi setup to measure tobias MAD200 chip

# First,
# run -i interactive.py
# run/import this script
# initialize()

# "SL" is switched internally on the chip (1-512), the pin is called vin1
# "BL" each go to a separate bin on the package (1-32)
# SL connects to the transistor source, BL connects to the RRAM terminal. (neglecting whatever select circuitry)
# Two possibilities:
# 1. Source on SL and measure on BL --> form with negative voltage
# 2. Source on BL and measure on SL --> form with positive voltage

# 1 seems to have less current noise, but forming operation would require an extreme bias of chip ground which could really change the transistor characteristic
# I also have in my notes that the biasing "didn't work" as expected

# 2 has higher noise but with massive oversampling has worked

## CONNECTIONS ##
# rigol
# Rigol1, Ch1 → Picoscope A → (a bit line BL)
# Rigol1, Ch2 → Vgate    (Into header pin)

# rigol2
# Rigol2, Ch1 → V ground offset (-1.5 V to chip ground)
# Rigol2, Ch2 → split into back panel trigger inputs of both Rigol1 and Rigol2 (only way to trigger simultaneously)

# Picoscope A → sample of Vapplied
# Picoscope C → 50 ohm current measurement on Source Line SL (labeled Vin)


# Arduino MKR Wifi handles the communication with the million switches in the chip
# This is important for isolating measurement ground from chip ground so we can bias the transistor in a strange way
# import sys
# sys.path.append('.') # breaks the next rigol command??? probably because of the _pycache_ folder.

mad = MAD200_Wifi()


def chip_gnd(V=0):
    """ Push chip ground below measurement ground (e.g. V=-1.5) """
    # TODO check that other signals are not lower than the new chip_gnd value
    rigol2.DC(V, 1)
    meta['chip_gnd'] = V

def Vgate(V):
    rigol.DC(V, 2)
    meta['Vgate'] = V

def zero():
    rigol.DC(0, 1)
    rigol.DC(0, 2)
    rigol2.DC(0, 1)
    rigol2.DC(0, 2)

def output(state=True):
    rigol.output(state, 1)
    rigol.output(state, 2)
    rigol2.output(state, 1)
    rigol2.output(state, 2)

def trigger():
    # send trigger, split into back panel of rigol
    rigol2.pulse_builtin('SQU', duration=1e-6, amp=5, offset=5, ch=2)

# Initialize
def initialize():
    zero()
    chip_gnd(0)
    Vgate(1)
    output()

    ps.coupling.a = 'DC'
    ps.range.a = 2
    ps.coupling.c = 'DC50'
    ps.range.c = 0.05

    # To save storage/data transfer times, these can be recomputed
    settings.drop_cols = ['V', 'I', 'Vneedle', 't', 'Vd']

    setup_meta()


def setup_meta(wafer='MAD200', samplename='dev1', polarity='+BL', block=1, SL=0, BL=2):
    meta['wafer'] = wafer
    meta['samplename'] = samplename
    meta['polarity'] = polarity
    meta['block'] = block
    meta['SL'] = SL
    meta['BL'] = BL
    meta.filenamekeys = ['wafer', 'samplename', 'block', 'SL', 'BL']


Vpos=2; Vneg=-1.4
Vpos = np.abs(Vpos)
Vneg = -np.abs(Vneg)
wfmA = tri(Vpos, Vneg, n=2**10)

def synchropulse(wfmA:'wfm', Vgpos=1.3, Vgneg=3, duration=1e-4, n=100, nsamples=1e5):
    """
    pulsing module for picoiv that constructs and sends a synchronized waveform on ch2
    """
    if any(wfmA < meta['chip_gnd']):
        raise Exception("Trying to apply a voltage below chip gnd!")

    meta.static.update() # lol
    
    wfmB = np.float64(wfmA >= 0) * (Vgpos - Vgneg) + Vgneg
    wfmB = ivtools.analyze.smooth(wfmB, 10) # try to reduce crossover distortion

    rigol.load_volatile_wfm(wfmA, duration, ch=1, interp=True)
    rigol.load_volatile_wfm(wfmB, duration, ch=2, interp=True) # smoother transition reduces cap coupling

    rigol.setup_burstmode(n=n, trigsource='EXT', ch=1)
    rigol.setup_burstmode(n=n, trigsource='EXT', ch=2)
    # This delay seems to be necessary the first time you set up burst mode?
    time.sleep(1)

    ps.capture(['A', 'C'], duration=dur*n, nsamples=nsamples*n, chrange=RANGE, choffset=OFFS)
    rigol.trigsource('EXT')

    trigger()

    d = ps.get_data(['A', 'C'])
    savedata(d)

    d = ivtools.settings.pico_to_iv(d)

    iplots.newline(d)
    #sd = splitiv(smoothimate(settings.pico_to_iv(d), 10, 1), n)
    #plotiv(sd[::5], alpha=.4)

    return d

#IV = interactive_wrapper(IV...)

def IV(Vpos=2, Vneg=-1.4, Vgpos=1.3, Vgneg=3, dur=1e-4, n=100, nsamples=1e5):
    """
    IV loops that use a synchronized gate signal
    TODO: Integrate with interactive picoiv somehow, instead of reimplementing those features
           Just add an argument for trigger that can be a function?
    """
    Vpos = np.abs(Vpos)
    Vneg = -np.abs(Vneg)
    
    meta.static.update(locals()) # lol

    # Autorange of monitor channel
    Arange, Aoffs = ps.best_range(np.array([Vneg, Vpos]), 0, ps.atten.a, ps.coupling.a)
    RANGE = dict(A=Arange)
    OFFS = dict(A=Aoffs)

    if Vneg < meta['chip_gnd']:
        raise Exception("Trying to apply a voltage below chip gnd!")
    

    wfmA = tri(Vpos, Vneg, n=2**10)
    wfmB = np.float64(wfmA >= 0) * (Vgpos - Vgneg) + Vgneg
    wfmB = ivtools.analyze.smooth(wfmB, 10) # try to reduce crossover distortion

    rigol.load_volatile_wfm(wfmA, dur, ch=1, interp=True)
    rigol.load_volatile_wfm(wfmB, dur, ch=2, interp=True) # smoother transition reduces cap coupling

    rigol.setup_burstmode(n=n, trigsource='EXT', ch=1)
    rigol.setup_burstmode(n=n, trigsource='EXT', ch=2)
    # This delay seems to be necessary the first time you set up burst mode?
    time.sleep(1)

    ps.capture(['A', 'C'], duration=dur*n, nsamples=nsamples*n, chrange=RANGE, choffset=OFFS)
    rigol.trigsource('EXT')

    trigger()

    d = ps.get_data(['A', 'C'], raw=True)
    savedata(d)

    d = ivtools.settings.pico_to_iv(d)

    iplots.newline(d)
    #sd = splitiv(smoothimate(settings.pico_to_iv(d), 10, 1), n)
    #plotiv(sd[::5], alpha=.4)

    return d

class block1():
    """
    Turns block1 on in the beginning and off at the end

    however, block1 is not guaranteed to be on during the entire block, as other functions might need toggle it (e.g. SL())..
    """
    def __enter__(self):
        meta['block'] = 1
        mad.setDIOs([MAD200_Wifi.block1])
        #time.delay(0.001)
    def __exit__(self, type, value, traceback):
        mad.resetDIOs([MAD200_Wifi.block1])
        #time.delay(0.001)

class block2():
    def __enter__(self):
        meta['block'] = 2
        mad.setDIOs([MAD200_Wifi.block2])
        #time.delay(0.001)
    def __exit__(self, type, value, traceback):
        mad.resetDIOs([MAD200_Wifi.block2])
        #time.delay(0.001)


def SL(n):
    '''
    For block1, SLs are columns in the schematic
    '''
    mad.resetFF()
    mad.setFF(n)
    # SL is the thing that switches internally, BL is the thing that you have to switch externally..
    meta['SL'] = n

def SLs(ns):
    mad.resetFF()
    for n in ns:
        mad.setFF(n)
    meta['SL'] = ns

def SL2(n, bl):
    '''
    Not tested!
    For block2, SLs are rows in the schematic?
    '''
    mad.resetFF()
    mad.setFF2(n, bl)
    # SL is the thing that switches internally, BL is the thing that you have to switch externally..
    meta['SL'] = n
    meta['block'] = 2

column = SL

def measurement():
    # Measure loops on all devices in the row
    meta['BL'] = 2
    for n in range(512):
        column(n)
        time.sleep(3.5) # TODO: replace with wifi confirmation -- should not take nearly this long
        with block1():
                time.sleep(.1)
                IV(Vpos=1.5, Vneg=-1.4, Vgpos=1.3, Vgneg=3, dur=1e-4, n=100, nsamples=1e5)