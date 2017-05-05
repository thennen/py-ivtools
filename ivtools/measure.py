""" Functions for measuring IV data with picoscope 6403C and Rigol AWG """

from picoscope import ps6000
import visa
from fractions import Fraction
from math import gcd
import numpy as np

# These are None until the instruments are connected
ps = None
rigol = None

# Settings for picoscope channels.
# These can be modified by the user, but should not be modified by functions.
COUPLINGS = {'A': 'DC50', 'B': 'DC', 'C': 'DC', 'D': 'DC'}
ATTENUATION = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
OFFSET = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
RANGE = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}

def connect_picoscope():
    global ps
    if ps is None:
        try:
            ps = ps6000.PS6000()
            print('Picoscope connection succeeded.')
            print(ps.getAllUnitInfo())
            print('Channel settings:')
            print('Couplings: {}'.format(COUPLINGS))
            print('Attenuation: {}'.format(ATTENUATION))
            print('Offset: {}'.format(OFFSET))
            print('Range: {}'.format(RANGE))
        except:
            print('Connection to picoscope failed.  Could be an unclosed session.')
    else:
        try:
            info = ps.getAllUnitInfo()
            print('Picoscope already connected.\n')
            print(info)
        except:
            print('ps variable is not None.  Doing nothing.')


def connect_rigolawg():
    global rigol
    rigolstr = 'USB0::0x1AB1::0x0640::DG5T155000186::INSTR'
    if rigol is None:
        try:
            rigol = visa.ResourceManager().open_resource(rigolstr)
            print('Rigol connection succeeded.')
            idn = rigol.query('*IDN?')
            print('*IDN?  {}'.format(idn))
        except:
            print('Connection to Rigol AWG failed.')
    else:
        try:
            # Check if rigol is already defined and connected
            idn = rigol.query('*IDN?')
            print('Rigol AWG already connected\n')
            print(idn)
        except:
            print('rigol variable is not None.  Doing nothing.')


def pico_capture(ch='A', freq=1e6, duration=0.04, nsamples=None,
                 trigsource='TriggerAux', triglevel=0.5):
    '''
    Set up picoscope to capture from specified channel(s).
    Won't actually do it until it receives the specified trigger event.
    I think it will trigger automatically after a timeout of 30s
    ch can be a list of characters, i.e. ch=['A','B'].
    # TODO: provide a way to override the global variable channel settings
    '''

    # If ch not iterable, just put it in a list by itself
    if not hasattr(ch, '__iter__'):
        ch = [ch]

    # Maximum sample rate is different depending on the number of channels that are enabled.
    # Therefore, if you want the highest possible rate, you should keep unused channels disabled.
    # Enable only the channels being used, disable the rest
    existing_channels = COUPLINGS.keys()
    for c in existing_channels:
        if c in ch:
            ps.setChannel(c, enabled=True)
        else:
            ps.setChannel(c, enabled=False)

    # This will return actual sample frequency, then we can determine
    # the number of samples needed.
    freq, _ = ps.setSamplingFrequency(freq, 0)
    if nsamples is None:
        nsamples = duration * freq
    freq, max_samples = ps.setSamplingFrequency(freq, nsamples)
    print('Actual picoscope sampling frequency: {:,}'.format(freq))
    if nsamples > max_samples:
        raise(Exception('Trying to sample more than picoscope memory capacity'))
    # Set up the channels
    for c in ch:
        ps.setChannel(c,
                      COUPLINGS[c],
                      RANGE[c],
                      probeAttenuation=ATTENUATION[c],
                      VOffset=OFFSET[c])
    # Set up the trigger.  Will timeout in 30s
    ps.setSimpleTrigger(trigsource, triglevel, timeout_ms=30000)
    ps.runBlock()


def pulse(waveform, duration, n=1, ch=1, trigsource='ScopeTrig'):
    '''
    Generate n pulses of the input waveform on Rigol AWG.
    Trigger immediately.
    '''

    # Rigol uses stupid SCPI commands.
    # Construct a string out of the waveform
    waveform = np.array(waveform, dtype=np.float32)
    maxamp = np.max(np.abs(waveform))
    normwaveform = waveform/maxamp
    wfm_str = ','.join([str(w) for w in normwaveform])
    freq = 1 / duration

    # TODO: toggling output state is slow and might not be necessary.
    #       it might also cause some spiking that damages the device.
    #       see if you can load a trace without leaving burst mode.
    rigol.write(':OUTPUT:STATE OFF')
    # This command switches out of burst mode...
    rigol.write(':TRAC:DATA VOLATILE,{}'.format(wfm_str))
    rigol.write(':SOURCE{}:FREQ:FIX {}'.format(ch, freq))
    rigol.write(':SOURCE{}:VOLTAGE:AMPL {}'.format(ch, 2*maxamp))
    rigol.write(':SOURCE{}:BURST:MODE TRIG'.format(ch))
    rigol.write(':SOURCE{}:BURST:NCYCLES {}'.format(ch, n))
    rigol.write(':SOURCE{}:BURST:TRIG:SOURCE MAN'.format(ch))
    rigol.write(':SOURCE{}:BURST:STATE ON'.format(ch))
    rigol.write(':OUTPUT{}:STATE ON'.format(ch))
    # Trigger rigol
    rigol.write(':SOURCE{}:BURST:TRIG IMM'.format(ch))


def get_data(ch='A', raw=False):
    '''
    Wait for data and transfer it from pico memory.
    ch can be a list of channels
    TODO: if raw is True, do not convert from ADC value - this saves a lot of memory
    return dict of arrays. TODO and metadata
    '''
    # Wait for data
    while(not ps.isReady()):
        time.sleep(0.01)

    if not hasattr(ch, '__iter__'):
        data = ps.getDataV(ch)
    else:
        data = []
        for c in ch:
            data.append(ps.getDataV(c))

    return data


def close():
    global ps
    global rigol
    # Close connection to pico
    ps.close()
    ps = None
    # Close connection to rigol
    rigol.close()
    rigol = None


def pulse_and_capture(waveform, ch=['A', 'B'], fs=1e6, duration=1e-3, n=1):
    '''
    Send n pulses of the input waveform and capture on specified channels of picoscope.
    Duration determines the length of one repetition of waveform.
    '''

    # Set up to capture for n times the duration of the pulse
    pico_capture(ch, freq=fs, duration=n*duration)
    # Pulse the waveform n times, this will trigger the picoscope capture.
    pulse(waveform, duration, n=n)

    data = get_data(ch)

    return data


def _rate_duration(vmax, vmin, rate=None, duration=None):
    '''
    Determines the duration or sweep rate of a triangle type pulse with constant sweep rate.
    Pass rate XOR duration, return (rate, duration).
    '''
    if not (bool(duration) ^ bool(rate)):
        raise Exception('Must give either duration or rate, and not both')
    if duration is not None:
        duration = float(duration)
        rate = 2 * (vmax - vmin) / duration
    elif rate is not None:
        rate = float(rate)
        duration = 2 * (vmax - vmin) / rate

    return rate, duration


def tripulse(n=1, vmax=1.0, vmin=-1.0, duration=None, rate=None):
    '''
    Generate n bipolar triangle pulses.
    Voltage sweep rate will  be constant.
    Trigger immediately
    '''

    rate, duration = _rate_duration(rate, duration, vmax, vmin)

    wfm = tri_wfm(vmax, vmin)

    pulse(wfm, duration, n=n)


def sinpulse(n=1, vmax=1.0, vmin=-1.0, duration=None):
    '''
    Generate n sine pulses.
    Trigger immediately
    If you pass vmin != -vmax, will not start at zero!
    '''

    wfm = (vmax - vmin) / 2 * np.sin(np.linspace(0, 2*pi, ps.AWGMaxSamples)) + ((vmax + vmin) / 2)

    pulse(wfm, duration, n=n)


def tri_wfm(vmax, vmin):
    '''
    Generate a triangle pulse waveform.

    This is a slightly tricky because the AWG takes equally spaced samples,
    so finding the shortest waveform that truly reaches vmax and vmin with
    constant sweep rate involves finding the greatest common divisor of
    vmin and vmax.
    '''
    vmin = -abs(vmin)

    fmax = Fraction(str(vmax))
    fmin = Fraction(str(vmin))
    # This is depreciated for some reason
    #vstep = float(abs(fractions.gcd(fmax, fmin)))
    # Doing it this other way.. Seems faster by a large factor.
    a, b = fmax.numerator, fmax.denominator
    c, d = fmin.numerator, fmin.denominator
    commond = float(b * d)
    vstep = gcd(a*d, b*c) / commond
    dv = vmax - vmin
    wfm = np.concatenate((np.linspace(0, vmax, vmax/vstep + 1),
                         np.linspace(vmax, vmin, dv / vstep + 1)[1:-1],
                         np.linspace(vmin, 0, -vmin/vstep + 1)))

    # Filling the AWG record length with probably take more time than it's worth.
    # Interpolate to a "Large enough" waveform size
    enough = 2**16
    x = np.linspace(0, 1, enough)
    xp = np.linspace(0, 1, len(wfm))
    wfm = np.interp(x, xp, wfm)

    return wfm
