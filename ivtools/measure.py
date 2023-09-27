"""
This module contains code that is used to coordinate control and data collection
from setups involving multiple instruments, focused on (I,V) measurements.

Module also contains other utilities that are just generally useful for doing measurements
"""
import logging
import os
import functools
import signal
import time
from fractions import Fraction
from math import gcd
from numbers import Number
import inspect
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ivtools.analyze
import ivtools.instruments as instruments
import ivtools.settings
import ivtools.plot

log = logging.getLogger('measure')

########### Picoscope + Rigol AWG testing #############

def pulse_and_capture(wfm, ch=['A', 'B'], fs=1e6, duration=1e-3, n=1, interpwfm=True, pretrig=0, posttrig=0,
                      **kwargs):
    '''
    Send n pulses of the input waveform and capture on specified channels of picoscope.
    Duration determines the length of one repetition of waveform.

    Basically a minimal version of picoiv that does no post-processing or other fancy business
    '''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    # Set up to capture for n times the duration of the pulse
    # TODO have separate arguments for pulse duration and frequency, sampling frequency, number of samples per pulse
    sampling_factor = (n + pretrig + posttrig)

    ps.capture(ch, freq=fs,
                   duration=duration * sampling_factor,
                   pretrig=pretrig / sampling_factor,
                   **kwargs)
    # Pulse the waveform n times, this will trigger the picoscope capture.
    rigol.pulse_arbitrary(wfm, duration, n=n, interp=interpwfm)

    data = ps.get_data(ch)

    return data

def pulse_and_capture_builtin(ch=['A', 'B'], shape='SIN', amp=1, freq=None, offset=0, phase=0, duration=None,
                              ncycles=10, samplespercycle=None, fs=None, extrasample=0, **kwargs):
    '''
    same as pulse_and_capture but using a built-in waveform
    TODO: this should just be a special case of pulse_and_capture, not repeated code
    '''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    if not (bool(samplespercycle) ^ bool(fs)):
        raise Exception('Must give either samplespercycle, or sampling frequency (fs), and not both')
    if not (bool(freq) ^ bool(duration)):
        raise Exception('Must give either freq or duration, and not both')

    if fs is None:
        fs = freq * samplespercycle
    if freq is None:
        freq = 1. / duration

    ps.capture(ch=ch, freq=fs, duration=(ncycles+extrasample)/freq, **kwargs)

    rigol.pulse_builtin(freq=freq, amp=amp, offset=offset, phase=phase, shape=shape, n=ncycles)

    data = ps.get_data(ch)

    return data

def rigol_pulse(wfm, duration, n, interpwfm=True, ch=1):
    '''
    This is the default pulsing component for picoiv -- not so useful on its own
    '''
    rigol = instruments.RigolDG5000()
    rigol.pulse_arbitrary(wfm, duration=duration, interp=interpwfm, n=n, ch=ch)

def picoiv(wfm, duration=1e-3, n=1, fs=None, nsamples=None, smartrange=1, autosplit=True,
           termination=None, channels=None, autosmoothimate=False, splitbylevel=None,
           savewfm=False, pretrig=0, posttrig=0, picoresolution=8, pico_to_iv=None, monitor_ch=None,
           trigsource='TriggerAux', triglevel=0.1, trigdirection='Rising', 
           timeout_ms=30000, pulsefunc=rigol_pulse, **kwargs):
    '''
    Pulse a waveform (n repeats), capture on picoscope channels, and return data with some conversion/post-processing.

    Provide either fs (picoscope sample frequency) or nsamples (total number of samples per shot)

    smartrange squeezes the range on the monitor channel, since we know the
    waveform we are going to apply (wfm)

    autosplit will split the capture into chunks in post-processing

    termination=50 will double the waveform amplitude to cancel resistive losses when using terminator

    by default we sample for exactly the length of the waveform,
    use "pretrig" and "posttrig" to sample before and after the waveform
    units are fraction of one pulse duration

    Some of the default arguments come from the settings module (pico_to_iv, monitor_ch, channels)

    To use other AWGs and/or triggers, pass a different pulsefunc.
    If pulsefunc takes arguments wfm, duration, or n, or are annotated to correspond to those,
    the picoiv arguments are passed through to pulsefunc, as well as any extra kwargs
    '''
    ps = instruments.Picoscope()

    if not (bool(fs) ^ bool(nsamples)):
        raise Exception('Must pass either fs or nsamples, and not both')

    if fs is None:
        fs = nsamples / duration

    if pico_to_iv is None:
        pico_to_iv = ivtools.settings.pico_to_iv

    if channels is None:
        channels = sorted(probe_channels(pico_to_iv))
        if not channels:
            channels = ['A', 'C']

    if monitor_ch is None:
        monitor_ch = ivtools.settings.MONITOR_PICOCHANNEL

    # Let pretrig and posttrig refer to the fraction of a single pulse, not the whole pulsetrain
    sampling_factor = (n + pretrig + posttrig)

    if not type(wfm) == np.ndarray:
        wfm = np.array(wfm)

    if smartrange:
        # Smart range the monitor channel
        smart_range(np.min(wfm), np.max(wfm), ch=[monitor_ch])

    # Set picoscope to capture
    # Sample frequencies have fixed values, so it's likely the exact one requested will not be used
    # TODO: ps.capture has some arguments that are not accessible by a picoiv() call..  add them?
    actual_fs = ps.capture(ch=channels,
                           freq=fs,
                           duration=duration * sampling_factor,
                           pretrig=pretrig / sampling_factor,
                           resolution=picoresolution,
                           trigsource=trigsource,
                           triglevel=triglevel,
                           direction=trigdirection,
                           timeout_ms=timeout_ms)

    # This makes me feel good, but I didn't test whether it's really necessary
    time.sleep(.05)

    if termination:
        # Account for terminating resistance
        # e.g. multiply applied voltages by 2 for 50 ohm termination
        wfm *= (50 + termination) / termination

    ### Send a pulse and trigger the measurement
    # pulsefunc can "request" data from picoiv by signalling with its signature
    # this could potentially break in strange ways but it's worth it to avoid a major code overhaul
    # These are shared arguments I think pulsefunc might want us to pass through
    pulseargs = {'wfm':wfm, 'duration':duration, 'n':n}
    # I'm going to ignore that it is technically possible that the arguments are positional-only
    for k,v in inspect.signature(pulsefunc).parameters.items():
        # bind by function annotation first, so we don't have to rename anything that could break compatibility
        if v.annotation in pulseargs:
            kwargs[k] = pulseargs[v.annotation]
        elif k in pulseargs:
            kwargs[k] = pulseargs[k]
    # Send a pulse (should be accompanied by a trigger for picoscope)
    pulsefunc(**kwargs)
    trainduration = n * duration
    log.info('Applying pulse(s) ({:.2e} seconds).'.format(trainduration))
    ivtools.plot.mybreakablepause(n * duration)

    log.debug('Getting data from picoscope.')
    # Raw data (int type matching scope resolution)
    chdata = ps.get_data(channels, raw=True)
    log.debug('Got data from picoscope.')

    # Convert to IV data (keeps channel data)
    ivdata = pico_to_iv(chdata)

    ivdata['nshots'] = n

    if savewfm:
        # Measured voltage has noise sometimes it's nice to plot vs the programmed waveform.
        # You will need to interpolate it, however.. Or can we instead read the interpolation off the rigol?
        ivdata['Vwfm'] = wfm

    if autosmoothimate:
        # This is largely replaced by putting autosmoothimate in the preprocessing list for the interactive figures!
        # if you do that, the data still gets written in its raw form, which is usually preferable
        # Below, we irreversibly drop data.
        nsamples_shot = ivdata['nsamples_capture'] / n
        # Smooth by 0.3% of a shot
        window = max(int(nsamples_shot * 0.003), 1)
        # End up with about 1000 data points per shot
        # This will be bad if you send in a single shot waveform with multiple cycles
        # In that case, you shouldn't be using autosmoothimate or autosplit
        # TODO: make a separate function for IV trains?
        if autosmoothimate is True:
            # yes I meant IS true..
            npts = 1000
        else:
            # Can pass the number of data points you would like to end up with
            npts = autosmoothimate
        factor = max(int(nsamples_shot / npts), 1)
        log.debug('Smoothimating data with window {}, factor {}'.format(window, factor))
        ivdata = ivtools.analyze.smoothimate(ivdata, window=window, factor=factor, columns=None)

    if autosplit: # True by default
        if (splitbylevel is None) and (n > 1):
            log.debug(f'Splitting data into n={n} individual pulses')
            nsamples = duration * actual_fs
            if 'downsampling' in ivdata:
                # Not exactly correct but I hope it's close enough
                nsamples /= ivdata['downsampling']
            ivdata = ivtools.analyze.splitiv(ivdata, nsamples=nsamples)
        elif splitbylevel:
            # splitbylevel can split loops even if they are not the same length
            # Rising edge or falling edge? Could take another argument for that..
            # This is a not-genius way to determine whether to split at + or - dV/dt
            increasing = bool(sign(argmax(wfm) - argmin(wfm)) + 1)
            ltgt = '>' if increasing else '<'
            log.debug(f'Splitting data at V={splitbylevel} V, where dV/dt {ltgt} 0')
            ivdata = ivtools.analyze.split_by_crossing(ivdata, V=splitbylevel, increasing=increasing, smallest=20)
        else:
            # log.debug('Nothing to split')
            pass
    elif splitbylevel:
            log.debug('splitbylevel does nothing if autosplit is False')

    return ivdata

def freq_response(ch='A', fstart=10, fend=1e8, n=10, amp=.3, offset=0, trigsource='TriggerAux'):
    ''' Apply a series of sine waves with rigol, and sample the response on picoscope. Return data without analysis.'''
    rigol = instruments.RigolDG5000()
    ps = instruments.Picoscope()

    if fend > 1e8:
        raise Exception('Rigol can only output up to 100MHz')

    freqs = np.geomspace(fstart, fend, n)
    data = []
    for freq in freqs:
        # Figure out how many cycles to sample and at which sample rate.
        # In my tests with FFT:
        # Number of cycles did not matter much, as long as there was exactly an integer number of cycles
        # Higher sampling helps a lot, with diminishing returns after 10^5 total data points.

        # I don't know what the optimum sampling conditions for the sine curve fitting method.
        # Probably takes longer for big datasets.  And you need a good guess for the number of cycles contained in the dataset.
        # (Actually is much faster, surprisingly)

        # How many cycles you want to have per frequency
        target_cycles = 100
        # How many data points you want to have
        target_datapoints = 1e5
        # Max amount of time (s) you are willing to wait for a measurement of a single frequency
        max_time_per_freq = 10
        # Capture at least this many cycles
        minimum_cycles = 1

        # Can sample 5 MS/s, divided among the channels
        # ** To achieve 2.5 GS/s sampling rate in 2-channel mode, use channel A or B and channel C or D.
        if len(ch) == 1:
            maxrate = 5e9
        elif len(ch) == 2:
            # 4 channel combinations allow 2.5 GS/s sampling rate
            if set(ch) in (set(('A', 'B')), set(('C', 'D'))):
                maxrate = 1.25e9
            else:
                maxrate = 2.5e9
        else:
            maxrate = 1.25e9

        cycles_per_maxtime = freq * max_time_per_freq
        time_for_target_cycles = target_cycles / freq

        # TODO: use hardware oversampling to increase resolution
        if cycles_per_maxtime < minimum_cycles:
            # We still need to capture at least certain number of whole cycles, so it will take longer.  Sorry.
            ncycles = minimum_cycles
            fs = target_datapoints * freq / ncycles
        elif cycles_per_maxtime < target_cycles:
            # Cycle a reduced number of (integer) times in order to keep measurement time down
            ncycles = int(cycles_per_maxtime)
            fs = target_datapoints * freq / ncycles
        elif target_datapoints / time_for_target_cycles < maxrate:
            # Excluding the possibility that someone set a really dumb max_time_per_freq,
            # this means that we acquire our target number of cycles, and our target number of samples.
            ncycles = target_cycles
            fs = target_datapoints * freq / ncycles
        else:
            # We are limited by the sampling rate of picoscope.
            # Capture the target number of cycles but with a reduced number of samples
            ncycles = target_cycles
            fs = maxrate
            # Or would it be better to capture an increased number of cycles?  To be determined..

        # Pico triggering appears to have about 6 ns of jitter.
        # To avoid capturing zeros at the end of the pulses, we will do an extra pulse at higher frequencies
        # Don't do it at low frequencies because it could lock up the AWG for an extra 1/freq
        if freq > 1e4:
            npulses = ncycles + 1
        else:
            npulses = ncycles

        duration = ncycles / freq

        # TODO: Should I apply the signal for a while before sampling?  Here I am sampling immediately from the first cycle.
        # The aux trigger has a delay and jitter for some reason.  Maybe triggering directly on the channel is better?
        if trigsource == 'TriggerAux':
            triglevel = 0.05
        else:
            triglevel = 0

        # This one pulses exactly npulses and tries to capture them from the beginning
        #ps.capture(ch, freq=fs, duration=duration, pretrig=0, triglevel=triglevel, trigsource=trigsource)
        #rigol.pulse_builtin(freq=freq, amp=amp, offset=offset, shape='SIN', n=npulses, ch=1)
        rigol.continuous_builtin(freq=freq, amp=amp, offset=offset, shape='SIN', ch=1)
        #time.sleep(.1)
        # I have seen this take a while on the first shot
        time.sleep(.5)
        ps.capture(ch, freq=fs, duration=duration, pretrig=0, triglevel=triglevel, trigsource=trigsource)
        d = ps.get_data(ch)
        d['ncycles'] = ncycles
        data.append(d)

        # TODO: make some plots that show when debug=True is passed
        rigol.output(False, ch=1)

    return data

def tripulse(n=1, v1=1.0, v2=-1.0, duration=None, rate=None):
    '''
    Generate n bipolar triangle pulses.
    Voltage sweep rate will be constant.
    Trigger immediately
    '''
    rigol = instruments.RigolDG5000()
    rate, duration = _rate_duration(v1, v2, rate, duration)
    wfm = tri(v1, v2)
    rigol.pulse_arbitrary(wfm, duration, n=n)

def smart_range(v1, v2, R=None, ch=['A', 'B']):
    '''
    Tries to choose the best range, offset for input channels, given the minimum and maximum values of the applied waveform
    This is easy to do for the monitor channel, but hard to do for channels that depend on the load (DUT)
    TODO: Somehow have reconfigurable auto-ranging functions for different setups.  Function attributes on pico_to_iv?
    TODO: don't let this function change the pico state.  Just return the calculated ranges.
    '''
    ps = instruments.Picoscope()
    # Auto offset for current input
    possible_ranges = np.array((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
    # Each range has a maximum possible offset
    max_offsets = np.array((.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20))

    # TODO allow multiple monitors?
    monitor_channel = ivtools.settings.MONITOR_PICOCHANNEL
    if monitor_channel in ch:
        # Assuming CHA is directly sampling the output waveform, we can easily optimize the range
        arange, aoffs = ps.best_range((v1, v2), atten=ps.atten[monitor_channel], coupling=ps.coupling[monitor_channel])
        ps.range[monitor_channel] = arange
        ps.offset[monitor_channel] = aoffs

def raw_to_V(datain, dtype=np.float32):
    '''
    Convert digitized values to voltage values.
    datain should be a dict with the 8/10/12 bit channel arrays and the RANGE and OFFSET values.
    return a new dict with updated channel arrays

    Doesn't use the state of picoscope like ps.rawToV() would, but rather the metadata stored in datain
    '''

    # picoscope int16 does not go +-32768, but depends on resolution setting!
    # this is probably because they only use 2**nbits - 1 bins 
    # so that it is symmetric and can represent zero volts with zero ADC count
    maxval = {8:32512, 10:32704, 12:32736}

    channels = ['A', 'B', 'C', 'D']
    dataout = {}
    for c in channels:
        if (c in datain.keys()):
            if (datain[c].dtype == np.int8):
                # ivtools converted this in Picoscope.get_data(..., raw=True)
                dataout[c] = datain[c] / 127 * dtype(datain['RANGE'][c]) - dtype(datain['OFFSET'][c])
            elif (datain[c].dtype == np.int16):
                # might be 10 or 12 bit
                # have to hope the information is included in datain, which it should be..
                res = datain['resolution']
                dataout[c] = datain[c] / dtype(maxval[res]) * dtype(datain['RANGE'][c]) - dtype(datain['OFFSET'][c])
    for k in datain.keys():
        if k not in dataout.keys():
            dataout[k] = datain[k]
    return dataout

def V_to_raw(datain):
    '''
    Inverse of raw_to_V (for 8 bit)
    '''
    channels = ['A', 'B', 'C', 'D']
    dataout = {}
    for c in channels:
        if (c in datain.keys()) and (datain[c].dtype in (np.float32, np.float64)):
            dataout[c] = np.int8(np.round((datain[c] + datain['OFFSET'][c]) * 127 / datain['RANGE'][c]))
    for k in datain.keys():
        if k not in dataout.keys():
            dataout[k] = datain[k]
    return dataout

def _rate_duration(v1, v2, rate=None, duration=None):
    '''
    Determines the duration or sweep rate of a triangle type pulse with constant sweep rate.
    Pass rate XOR duration, return (rate, duration).
    '''
    if not (bool(duration) ^ bool(rate)):
        raise Exception('Must give either duration or rate, and not both')
    if duration is not None:
        duration = float(duration)
        rate = 2 * (v1 - v2) / duration
    elif rate is not None:
        rate = float(rate)
        duration = 2 * (v1 - v2) / rate

    return rate, duration


########### Picoscope + Teo testing ###################

def picoteo(wfm, n=1, duration=None, fs=None, nsamples=None, smartrange=None, autosplit=False,
            HFV_ch=None, V_MONITOR_ch='B', HF_LIM_ch='C', HF_FUL_ch='D',
            splitbylevel=None, termination=None, autosmoothimate=False,
            savewfm=False, save_teo_int=True, pretrig=0, posttrig=0):
    '''
    Pulse a waveform with teo, measure on picoscope and teo, and return data

    Parameters:
        wfm: Array of voltage values to be applied. Or the name of a waveform loaded in Teo.
        n: Number of repetitions of the waveform
        duration: Duration of the wfm. If None, wfm values will be applied at Teo frequency: 500 MHz.
        fs: Picoscope sample frequency
        nsamples: Picoscope number of samples (alternative to fs)
        smartrange: =1 autoranges the monitor channel. =2 tries some other fancy shit to autorange the current
            measurement channel
        autosplit: Automatically split data
        HFV_ch: Picoscope channel used to monitor teo HFV
        V_MONITOR_ch: Picoscope channel used to monitor teo V_MONITOR
        HF_LIM_ch: Picoscope channel used to monitor teo HF_LIMITED_BW
        HF_FUL_ch: Picoscope channel used to monitor teo HF_FULL_BW
        splitbylevel: no idea
        termination: termination=50 will double the waveform amplitude to cancel resistive losses when using terminator
        autosmoothimate: Automatically smooth and decimate
        savewfm: save original waveform
        save_teo_int: save Teo internal measurements
        pretrig: sample before the waveform. Units are fraction of one pulse duration
        posttrig: sample after the waveform. Units are fraction of one pulse duration


    TODO: substantial amount of this code is shared with picoiv. Refactor to share the same code.
    '''

    teo = instruments.TeoSystem()
    ps = instruments.Picoscope()

    # decide what sample rate to use
    teo_freq = 500e6

    if type(wfm) is str:
        wfm_name = wfm
        if wfm_name in teo.waveforms:
            wfm = teo.waveforms[wfm_name][0]
        else:
            wfm = teo.download_wfm(wfm_name)[0]
        if duration is not None:
            raise Exception("You can't pass 'duration' when using a saved waveform")
        lenw = len(wfm)
        duration = (lenw-1)/teo_freq
    else:
        wfm_name = None
        if not type(wfm) == np.ndarray:
            wfm = np.array(wfm)
        lenw = len(wfm)
        if duration is not None:
            wfm = teo.interp_wfm(wfm, duration)
            lenw = len(wfm)
        else:
            duration = (lenw - 1) / teo_freq


    if (bool(fs) * bool(nsamples)):
        raise Exception('Can not pass fs and nsamples, only one of them')

    if fs is None:
        if nsamples is None:
            fs = teo_freq
        else:
            fs = nsamples / duration

    if smartrange:
        # Smart range the monitor channel
        # TODO: Are we assuming that picoscope is taking its own sample of the HFV output?
        # the Vmonitor channel can also be autoranged. Just needs to be adapted for the
        # offset/gain of the teo monitor output!
        smart_range(np.min(wfm), np.max(wfm), ch=[ivtools.settings.MONITOR_PICOCHANNEL])
    channels = [HFV_ch, V_MONITOR_ch, HF_LIM_ch, HF_FUL_ch]
    channels = [ch for ch in channels if ch is not None]
    log.info(channels)

    chunksize = 2 ** 11
    npad = chunksize - (lenw % chunksize)
    pad_duration = (npad - 1) / fs

    # Let pretrig and posttrig refer to the fraction of a single pulse, not the whole pulsetrain
    sampling_factor = (n + pretrig + posttrig)

    # There is a delay of some ns on the triggering, so that has to passed to ps.capture, but it is passed
    # in clock cycles units.
    # Actually, each channel has its own delay, V_MONITOR is 4 ns, HF_LIMITED_BW is 13 ns, and HF_FULL_BW is 9 ns
    pico_clock_freq = 1e9
    delay_sec = 4e-9
    delay = int(pico_clock_freq * delay_sec)

    # Set picoscope to capture
    # Sample frequencies have fixed values, so it's likely the exact one requested will not be used
    actual_pico_freq = ps.capture(ch=channels,
                                  freq=fs,
                                  duration=(duration+pad_duration) * sampling_factor,
                                  pretrig=pretrig / sampling_factor,
                                  delay=delay)

    pico_nsamples = int(duration * actual_pico_freq)

    log.debug(f"Teo frequency: 500.0 MHz\n"
              f"Picoscope frequency: {actual_pico_freq*1e-6} MHz\n"
              f"Teo number of samples: {lenw}\n"
              f"Picoscope number of samples: {pico_nsamples}")


    # This makes me feel good, but I don't think it's really necessary
    time.sleep(.05)
    if termination:
        # Account for terminating resistance
        # e.g. multiply applied voltages by 2 for 50 ohm termination
        wfm *= (50 + termination) / termination

    # Send a pulse
    trainduration = (duration+pad_duration) * sampling_factor
    log.info('Applying pulse(s) ({:.2e} seconds).'.format(trainduration))
    teo.output_wfm(wfm, n=n)

    time.sleep(trainduration * 1.05)
    #ps.waitReady()
    log.debug('Getting data from picoscope.')
    # Get the picoscope data
    # This goes into a global strictly for the purpose of plotting the (unsplit) waveforms.
    chdata = ps.get_data(channels, raw=False)
    log.debug('Got data from picoscope.')
    # Convert to IV data (keeps channel data)
    ivdata = ivtools.settings.pico_to_iv(chdata)

    ivdata['nshots'] = n

    if savewfm:
        # Measured voltage has noise sometimes it's nice to plot vs the programmed waveform.
        # You will need to interpolate it, however..
        # Or can we read it off the rigol??
        ivdata['Vwfm'] = wfm

    if autosmoothimate:
        # This is largely replaced by putting autosmoothimate in the preprocessing list for the interactive figures!
        # if you do that, the data still gets written in its raw form, which is preferable usually
        # Below, we irreversibly drop data.
        nsamples_shot = ivdata['nsamples_capture'] / n
        # Smooth by 0.3% of a shot
        window = max(int(nsamples_shot * 0.003), 1)
        # End up with about 1000 data points per shot
        # This will be bad if you send in a single shot waveform with multiple cycles
        # In that case, you shouldn't be using autosmoothimate or autosplit
        # TODO: make a separate function for IV trains?
        if autosmoothimate is True:
            # yes I meant IS true..
            npts = 1000
        else:
            # Can pass the number of data points you would like to end up with
            npts = autosmoothimate
        factor = max(int(nsamples_shot / npts), 1)
        log.debug('Smoothimating data with window {}, factor {}'.format(window, factor))
        ivdata = ivtools.analyze.smoothimate(ivdata, window=window, factor=factor, columns=None)

    if save_teo_int:
        teo_data = teo.get_data()
        ivdata['V_teo'] = teo_data['V']
        ivdata['I_teo'] = teo_data['I']
        ivdata['t_teo'] = teo_data['t']
        ivdata['wfm_teo'] = teo_data['Vwfm']
        ivdata['idn_teo'] = teo_data['idn']
        ivdata['sample_rate_teo'] = teo_data['sample_rate']
        ivdata['gain_step_teo'] = teo_data['gain_step']
        if 'units' not in ivdata:
            ivdata['units'] = {}
            print('Hi')
        print(ivdata['units'])
        ivdata['units']['V_teo'] = teo_data['units']['V']
        ivdata['units']['I_teo'] = teo_data['units']['I']
        ivdata['units']['t_teo'] = teo_data['units']['t']
        ivdata['units']['wfm_teo'] = teo_data['units']['Vwfm']
        print(ivdata['units'])
        if 'calibration_teo' not in ivdata:
            ivdata['calibration_teo'] = {}
        ivdata['calibration_teo']['V_teo'] = teo_data['calibration']['V']
        ivdata['calibration_teo']['I_teo'] = teo_data['calibration']['I']


    if autosplit and (n > 1):
        log.debug('Splitting data into individual pulses')
        if splitbylevel is None:
            nsamples = duration * actual_pico_freq
            if 'downsampling' in ivdata:
                # Not exactly correct but I hope it's close enough
                nsamples /= ivdata['downsampling']
            ivdata = ivtools.analyze.splitiv(ivdata, nsamples=nsamples)
        elif splitbylevel is not None:
            # splitbylevel can split loops even if they are not the same length
            # Could take more time though?
            # This is not a genius way to determine to split at + or - dV/dt
            increasing = bool(sign(argmax(wfm) - argmin(wfm)) + 1)
            ivdata = ivtools.analyze.split_by_crossing(ivdata, V=splitbylevel, increasing=increasing, smallest=20)

    return ivdata


########### Rehan amplifier ##########################

def measure_dc_gain(Vin=1, ch='C', R=10e3):
    import ivtools.plot as plot
    # Measure dc gain of rehan amplifier
    # Apply voltage
    rigol = instruments.RigolDG5000()

    print('Outputting {} volts on Rigol CH1'.format(Vin))
    rigol.pulse_arbitrary(np.repeat(Vin, 100), 1e-3)
    time.sleep(1)
    # Measure output
    measurechannels = ['A', ch]
    ps.capture(measurechannels, freq=1e6, duration=1, timeout_ms=1)
    time.sleep(.1)
    chdata = ps.get_data(measurechannels)
    plot.plot_channels(chdata)
    chvalue = np.mean(chdata[ch])
    print('Measured {} volts on picoscope channel {}'.format(chvalue, ch))

    gain = R * chvalue / Vin
    # Set output back to zero
    rigol.pulse_arbitrary([Vin, 0,0,0,0], 1e-3)
    return gain

def measure_ac_gain(R=1000, freq=1e4, ch='C', outamp=1):
    # Probably an obsolete function
    # Send a test pulse to determine better range to use
    import ivtools.plot as plot
    arange, aoffset = best_range([outamp, -outamp])
    RANGE = {}
    OFFSET = {}
    RANGE['A'] = arange
    OFFSET['A'] = aoffset
    # Power supply is 5V, so this should cover the whole range
    RANGE[ch] = 5
    OFFSET[ch] = 0
    sinwave = outamp * sin(np.linspace(0, 1, 2**12)*2*pi)
    chs = ['A', ch]
    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1, chrange=RANGE, choffset=OFFSET)
    data = ps.get_data(chs)
    plot.plot_channels(data)

    # will change the range and offset after all
    squeeze_range(data, [ch])

    pulse_and_capture(sinwave, ch=chs, fs=freq*100, duration=1/freq, n=1000)
    data = ps.get_data(chs)

    plot.plot_channels(data)

    return max(abs(fft.fft(data[ch]))[1:-1]) / max(abs(fft.fft(data['A']))[1:-1]) * R


########### Compliance circuit ###################

def calibrate_compliance_with_keithley(Rload=1000, kch='A'):
    '''
    Use keithley to calibrate current compliance levels
    Attach keithley channel A to input of compliance circuit through a resistor (Rload)
    Use rigol channel 2 for the compliance control voltage
    this sets different compliance levels and measures the whole I-V curve for that setting
    takes a while, but will be accurate
    '''
    k = instruments.Keithley2600()
    rigol = instruments.RigolDG5000()

    Remitter = 2050
    Vneg = 9.6 # negative power rail
    def approxVc(Ic):
        # first approximation for calibration of current source
        return Ic*Remitter - Vneg + 0.344

    def approxIc(Vc):
        return (Vc + Vneg - 0.344) / Remitter

    # Measure IV sweeps with different voltages applied to the current control
    approxImax = 2e-3
    n = 30
    approxVcmax = approxVc(approxImax)
    vlist = np.linspace(-9.6, approxVcmax, n)
    # what V is necessary to reach the highest I? apply 1.25 times that
    Vmax = approxIc(max(vlist)) * Rload * 1.25
    Vmax = min(Vmax, 5)
    Vmin = Rload * 500e-6
    data = []
    for v in vlist:
        rigol.DC(v, 2)
        d = k.iv(tri(Vmin, Vmax, n=100), ch=kch)
        k.waitready()
        d = k.get_data(ch=kch)
        data.append(d)
    data = pd.DataFrame(data)

    # calculate the current saturation level and slope for every control voltage
    # early effect slope (hundreds of kohm)
    data['R'] = ivtools.analyze.resistance(data, Vmax/1.25, Vmax)
    data['Ic'] = ivtools.analyze.slicebyvalue(data, 'V', Vmax/1.25, Vmax).I.apply(np.mean)
    data['Vc'] = vlist
    #plt.figure()
    #plt.plot(vc, np.polyval(p, vc))

    # fit diode equation
    from scipy.optimize import curve_fit
    # params Is=1e-10, Ioff=-2.075e-6, Re=2050, Vneg=-9.6
    guess = (1e-10, -2e-6, 2000)
    try:
        p = curve_fit(compliance_voltage, data.Ic, data.Vc, guess)[0]
        Is, Ioff, Re = p
    except RuntimeError as e:
        log.error('compliance current calibration fit failed!')
        Is = 2.5825e-10
        Ioff = -2.063e-6
        Re = 1997.3

    #guess = (1e-10, -2e-6)
    #p = curve_fit(compliance_voltage, data.Ic, data.Vc, guess)[0]
    #Is, Ioff = p
    #Re = 2000

    # There's not a great way to store these scalars in the dataframe
    data._metadata = ['Is', 'Ioff', 'Re']
    data.Is = Is
    data.Ioff = Ioff
    data.Re = Re

    calfp = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    os.makedirs(os.path.split(calfp)[0], exist_ok=True)
    data.to_pickle(calfp)

    plot_compliance_calibration()

    return data

def plot_compliance_calibration(calfile=None):
    cal = read_compliance_calibration_file(calfile)

    plt.figure()
    plt.plot(cal.Vc, cal.Ic, marker='.')
    I = np.linspace(-4e-6, np.max(cal.Ic), 1000)
    V = compliance_voltage(I, cal.Is, cal.Ioff, cal.Re)
    plt.plot(V, I)
    plt.xlabel('V_base [V]')
    plt.ylabel('I_limit [A]')

    plt.figure()
    for i,r in cal.iterrows():
        plt.plot(r.V, r.I, label=r.Vc)
    plt.xlabel('V_input')
    plt.ylabel('I_input')

def read_compliance_calibration_file(calfile=None):
    if calfile is None:
        calfile = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    if os.path.isfile(calfile):
        return pd.read_pickle(calfile)
    else:
        log.error('Calibration file not found!')

def set_compliance(cc_value):
    '''
    Setting a DC value for the current limit
    '''
    rigol = instruments.RigolDG5000()
    Vc = compliance_voltage_lookup(cc_value)
    log.debug(f'Setting compliance control voltage to {Vc}')
    rigol.DC(Vc, ch=2)
    rigol.output(True, ch=2)

def compliance_voltage_lookup(I):
    '''
    Interpolates a calibration table to convert between Icompliance and Vcontrol
    if you ask for a current outside of the calibration range, the output will be clipped
    # TODO Use a spline
    '''
    calfile = ivtools.settings.COMPLIANCE_CALIBRATION_FILE
    if os.path.isfile(calfile):
        cal = pd.read_pickle(calfile)
        # interpolate for best value for vc
        Vc = np.interp(I, cal['Ic'], cal['Vc'])
    else:
        log.warning('No calibration file! Using diode equation.')
        #Remitter = 2050
        #Vneg = 9.6
        #Vc = Ic*Remitter - Vneg + 0.344
        Vc = compliance_voltage(I)
    return Vc

def compliance_voltage(I, Is=2.5825e-10, Ioff=-2.063e-6, Re=1997.3, Vneg=-9.6):
    '''
    Use continuous diode equation to calculate the control voltage needed for a target compliance current (Icc)
    default values are from calibration
    depends on a good fit! plot_compliance_calibration() to check it!

    it's quite linear above 100 uA or so.  This is more important for lesser values

    Inputs are calibration constants
    Is -- diode reverse saturation current
    Re -- emitter resistance
    Ioff -- input offset current
    Vneg -- negative rail of power supply
    '''
    # System that needs to be solved for I(V):
    # I = Is(exp(Vd/VT) - 1) + Ioff
    # Vd = V - IRe
    # where V is the voltage level above the negative rail

    # Thermal Voltage
    V_T = 25.69e-3
    V = V_T * np.log((I - Ioff + Is) / Is) + I*Re + Vneg
    if type(V) is np.ndarray:
        V[np.isnan(V)] = Vneg
    return V

def hybrid_IV(Imax=500e-6, Vmin=-3, dur=1e-3):
    '''
    this does a current sweep in the positive direction, followed by a voltage sweep in the
    negative direction using compliance circuit.  All I,V points are measured

    will work better with compliance circuit that sits on the sourcing side

    tested a bit but it's a work in progress
    TODO: make a pulse2ch_and_capture for these synchronized signals, and use that
    '''

    # To synchronize rigol channels, you need an external trigger split into the two sync ports on the back.
    # This is provided by a second rigol here
    rigol = instruments.RigolDG5000('USB0::0x1AB1::0x0640::DG5T155000186::INSTR')
    rigol2 = instruments.RigolDG5000('USB0::0x1AB1::0x0640::DG5T182500117::INSTR')
    ps = instruments.Picoscope()

    I_to_Vc = compliance_voltage_lookup

    # TODO we should transition more smoothly at the crossover
    #      might need to carefully decide how to do it
    wfm_len = 2**12
    mid = wfm_len // 2
    t = np.linspace(0, 2*np.pi, wfm_len)
    # current source during voltage sweep (should be high enough to bring the collector
    # current up and therefore emitter resistance down)
    Iidle = I_to_Vc(Imax*1.3)
    # voltage source during current sweep (needs to be enough to supply all the currents)
    # but Vneedle can only go up to 4v or so!
    # meaning that we can't get an accurate Vdevice unless Vneedle < 4
    Vidle = 4
    iwfm = I_to_Vc(Imax*np.sin(t)**2)
    iwfm[mid:] = Iidle
    vwfm = -np.abs(Vmin)*np.sin(t)**2
    vwfm[:mid] = Vidle
    vwfm[0] = vwfm[-1] = 0
    iwfm[0] = iwfm[-1] = Iidle
    # just add some smoother linear transitions
    # sorry this is ugly af
    # when converting to current source, drop the current before increasing the voltage limit
    # when converting to voltage source, drop the voltage before increasing the current limit
    trans_len = 2**6
    iwfm = np.concatenate(([iwfm[0]], np.linspace(iwfm[0], iwfm[1], trans_len), np.repeat(iwfm[1], trans_len),
                           iwfm[1:mid], np.repeat(iwfm[mid-1], trans_len),
                           np.linspace(iwfm[mid-1], [iwfm[mid]], trans_len), iwfm[mid:]))
    vwfm = np.concatenate(([vwfm[0]], np.repeat(vwfm[0], trans_len), np.linspace(vwfm[0], vwfm[1], trans_len), vwfm[1:mid],
                           np.linspace(vwfm[mid-1], [vwfm[mid]], trans_len), np.repeat(vwfm[mid], trans_len), vwfm[mid:]))
    '''
    plt.figure()
    plt.plot(iwfm)
    plt.plot(vwfm)
    '''

    rigol.load_volatile_wfm(vwfm, dur, ch=1)
    time.sleep(.1)
    # Use rigol offset, or else the zero idle level will destroy everything
    iwfm -= Iidle
    rigol.load_volatile_wfm(iwfm, dur, ch=2, offset=Iidle)
    rigol.setup_burstmode(n=1, trigsource='EXT', ch=1)
    rigol.setup_burstmode(n=1, trigsource='EXT', ch=2)
    # This delay seems to be necessary the first time you set up burst mode?
    time.sleep(1)

    ps.capture(['A', 'B', 'C'], 1e8, dur, timeout_ms=3000, pretrig=0)
    # trigger everything
    rigol2.pulse_builtin('SQU', duration=1e-6, amp=5, offset=5)
    d = ps.get_data(['A', 'B', 'C'])
    d = ccircuit_to_iv(d)
    #plot_channels(d)
    #savedata(d)

    d['t'] = ivtools.analyze.maketimearray(d)
    '''
    sd = smoothimate(d, 300, 10)
    # TODO cut out the crossover distortion
    #plotiv(sd, 'Vd', 'I')

    # if no linear transitions
    #plotiv(slicebyvalue(sd, 't', dur*.0015, dur*(0.5-0.001)), 'Vd', 'I')
    #plotiv(slicebyvalue(sd, 't', dur*(0.5 + 0.001)), 'Vd', 'I', ax=plt.gca())

    dplus = slicebyvalue(sd, 't', dur*.017, dur*(0.5-0.001))
    dminus = slicebyvalue(sd, 't', dur*(0.5 + 0.017))
    plotiv(dplus, 'Vd', 'I')
    hplotiv(dminus, 'Vd', 'I')

    plotiv(dplus, 'Vd', 'I', plotfunc=arrowpath, maxsamples=100)
    hplotiv(dminus, 'Vd', 'I', plotfunc=arrowpath, maxsamples=100, color='C1')
    '''

    return d


########### Digipot ####################

def digipot_test(plot=True):
    '''
    This rapidly makes sure everything is working properly with the digipot board, since it lacks built-in diagnostics

    Put Rigol ch1 into the digipot input,
    split rigol into pico chA for monitoring
    connect digipot voltage output into pico chB
    monitor digipot output current on rigol chC
    '''
    # Use these settings but don't change the state of picoscope
    coupling = dict(A='DC', B='DC50', C='DC50')
    ranges = dict(A=5, B=5, C=1)
    ps = instruments.Picoscope()
    dp = instruments.WichmannDigipot()
    rigol = instruments.RigolDG5000()
    dur = 1e-2
    if plot:
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        plt.show()
    data = []
    channels = ['A', 'B', 'C']

    # Check Bypass first
    dp.set_bypass(1)
    ps.capture(channels, freq=10000/dur, duration=dur, chrange=ranges, chcoupling=coupling)
    rigol.pulse_builtin('SIN', duration=dur, amp=1)
    d = ps.get_data(channels)
    d = digipot_to_iv(d)
    d = ivtools.analyze.moving_avg(d,1000)
    data.append(d)
    if plot:
        #ax.plot(d['V'], d['I'])
        ax.plot(d['V'], d['V']/d['I'], color='black')

    dp.set_bypass(0)

    for w,Rnom in enumerate(dp.R2list):
        dp.set_state(wiper2=w) # Should have the necessary delay built in
        # Put 10 milliwatt through each resistor
        #A = np.sqrt(10e-3 * Rnom)
        #A = min(A, 5)
        A = 3
        Iexpected = A/Rnom
        ranges['C'] = ps.best_range([-Iexpected*50, Iexpected*50])[0]
        ps.capture(channels, freq=10000/dur, duration=dur, chrange=ranges, chcoupling=coupling)
        rigol.pulse_builtin('SIN', duration=dur, amp=A)
        d = ps.get_data(channels)
        d = digipot_to_iv(d)
        d = ivtools.analyze.moving_avg(d,100)
        data.append(d)
        if plot:
            #ax.plot(d['V'], d['I'], label=w)
            #color = ax.lines[-1].get_color()
            #ax.plot(d['V'], d['V']/Rnom, label=w, linestyle='--', alpha=.2, color=color)
            # Or
            ax.plot(d['V'], d['V']/d['I'], label=w)
            color = ax.lines[-1].get_color()
            ax.plot([-5,5], [Rnom+50, Rnom+50], label=w, linestyle='--', alpha=.2, color=color)
            plt.pause(.1)
            plt.xlim(-3, 3)
            plt.ylim(40, 60000)

    return data

def digipot_calibrate(plot=True):
    '''
    Connect keithley to digipot to measure the resistance values
    Can be done by contacting something conductive or smashing the needles together in the air

    configured for the single pot mode, not parallel or series
    '''
    dp = instruments.WichmannDigipot()
    k = instruments.Keithley2600()
    if plot:
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(np.arange(34), dp.Rlist, marker='.', label='current calibration')
        ax.set_xlabel('Pot setting')
        ax.set_ylabel('Resistance [Ohm]')
        plt.show()
        plt.pause(.1)

    dp.set_bypass(0)
    data = []
    for w,Rnom in enumerate(dp.R2list):
        dp.set_state(wiper2=w) # Should have the necessary delay built in
        # Apply a volt, measure current
        k.iv([1], Irange=0, Ilimit=10e-3, nplc=1)
        while not k.done():
            plt.pause(.1)
        d = k.get_data()
        d['R'] = d['V']/d['I']
        data.append(d)
        if plot:
            plt.scatter(w, d['R'])
            plt.pause(.1)

    print([d['R'][0].round(2) for d in data])
    return data

def digipotiv(V1, V2, R1=0, R2=0,
              duration=1e-3, fs=None, nsamples=100_000, smartrange=1, termination=None, autosmoothimate=False,
              savewfm=False, pretrig=0, posttrig=0, interpwfm=True):

    '''
    Uses a Rigol, Picoscope and Digipot to measure ReRAM loops with triangle sweeps and different
    series resistance values e.g. during SET and RESET.

    just a slightly more convenient way to write a for-loop
    TODO: wrap picoiv automatically and add arguments to it

    @param V_set: Voltage to applied during SET, it can be a list of values or 0 if you only want to do RESET.
    If V_set is a list and V_reset a number, V_reset will be repeated for every V_set value, and viceversa.
    @param V_reset: Same as V_set.
    @param R_set: Series Resistance during SET, it can be a list of values.
    @param R_reset: Series Resistance during RESET, it can be a list of values.

    everything else goes to picoiv
    '''
    ps = instruments.Picoscope()
    dp = instruments.WichmannDigipot()

    V1, V2, R1, R2 = arg_broadcast(V1, V2, R1, R2)

    sweeps = []
    for v1, v2, r1, r2 in zip(V1, V2, R1, R2):
        if v1 != 0:
            wfm_set = tri(0, v1)
            dp.set_R(r1)
            d_set = picoiv(wfm=wfm_set, duration=duration, fs=fs, nsamples=nsamples, smartrange=smartrange,
                           termination=termination, channels=['A', 'B', 'C'], autosmoothimate=autosmoothimate,
                           savewfm=savewfm, pretrig=pretrig, posttrig=posttrig, interpwfm=interpwfm)
            sweeps.append(d_set)
        if v2 != 0:
            wfm_reset = tri(0, -v2)
            dp.set_R(r2)
            d_reset = picoiv(wfm=wfm_reset, duration=duration, fs=fs, nsamples=nsamples, smartrange=smartrange,
                             termination=termination, channels=['A', 'B', 'C'], autosmoothimate=autosmoothimate,
                             savewfm=savewfm, pretrig=pretrig, posttrig=posttrig, interpwfm=interpwfm)
            sweeps.append(d_reset)

    return sweeps


########### Setup-dependent conversions from picoscope channel data to IV data ###################

def ccircuit_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    For the newer version of compliance circuit, which compensates itself and amplifies the needle voltage
    chA should monitor the applied voltage
    chB (optional) should be the needle voltage
    chC should be the amplified current
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here (input channel data will not be changed!)
    if datain['A'].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    C = datain['C']
    #C = datain['C']
    gain = ivtools.settings.CCIRCUIT_GAIN
    dataout['V'] = dtype(A)
    #dataout['V_formula'] = 'CHA - IO'
    #dataout['I'] = 1e3 * (B - C) / R
    if 'B' in datain:
        B = datain['B']
        # Vneedle is basically never used.  if needed, can be calculated easily
        #dataout['Vneedle'] = dtype(B)
        dataout['Vd'] = dataout['V'] - dtype(B) # Does not take phase shift into account!
    dataout['I'] = dtype(C / gain)
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    #units = {'V':'V', 'Vd':'V', 'Vneedle':'V', 'I':'A'}
    units = {'V':'V', 'Vd':'V', 'I':'A'}
    dataout['units'] = {k:v for k,v in units.items() if k in dataout}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
    dataout['gain'] = gain
    return dataout

def ccircuit_to_iv_split(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    For the newer version of compliance circuit, which compensates itself and amplifies the needle voltage
    chA should monitor the applied voltage
    chB (optional) should be the needle voltage
    The amplified current from the ccircuit is splitted by an RF-splitter into chC and chD
    chC should be the amplified current (set to a high ps.range for measuring large current ranges)
    chD should be the amplified current (set to the minimal ps.range for high resolution at low currents)

    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    C = datain['C']
    D = datain['D']
    #C = datain['C']
    gain = ivtools.settings.CCIRCUIT_GAIN
    dataout['V'] = dtype(A)
    #dataout['V_formula'] = 'CHA - IO'
    #dataout['I'] = 1e3 * (B - C) / R
    if 'B' in datain:
        B = datain['B']
        # Vneedle is basically never used.  if needed, can be calculated easily
        #dataout['Vneedle'] = dtype(B)
        dataout['Vd'] = dataout['V'] - dtype(B) # Does not take phase shift into account!
    dataout['I']=dtype(2*C/gain) 
    dataout['I_low']=dtype(2*D/gain)
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    units = {'V':'V', 'Vd':'V', 'I':'A', 'I_low':'A'}
    dataout['units'] = {k:v for k,v in units.items() if k in dataout}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
    dataout['gain'] = gain
    return dataout

def ccircuit_yellow_to_iv(datain, dtype=np.float32):
    '''
    For the redesigned circuit that sources voltage
    needle voltage is the device voltage
    no need to sample AWG waveform -- don't have a cow if there's no CHA

    Convert picoscope channel data to IV dict
    For the newer version of compliance circuit, which compensates itself and amplifies the needle voltage

    chA should monitor the applied voltage
    chB should be the needle voltage
    chC should be the amplified current
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)
    B = datain['B']
    C = datain['C']
    if 'A' in datain:
        A = datain['A']
        dataout['V'] = dtype(A)
    else:
        dataout['V'] = dtype(B)
    gain = ivtools.settings.CCIRCUIT_GAIN
    #dataout['V_formula'] = 'CHA - IO'
    #dataout['I'] = 1e3 * (B - C) / R
    dataout['I'] = dtype(C / gain)
    dataout['Vd'] = dtype(B) # actually should subtract 50*I
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    dataout['units'] = {'V':'V', 'Vd':'V', 'I':'A'}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
    dataout['gain'] = gain
    return dataout

def rehan_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    for Rehan amplifier
    Assumes constant gain vs frequency.  This isn't right.
    Careful! Scope input couplings will affect the gain!
    if x10 channel has 50 ohm termination, then gain of x200 channel reduced by 2!
    everything should be terminated with 50 ohms obviously
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    # Volts per amp
    gainC = 10.6876 * 50
    gainD = 113.32 * 50
    # 1 Meg, 33,000

    # I think these depend a lot on the scope input range/offset..
    offsD = 0.073988
    offsC = 0.006025

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    C = datain['C']

    dataout['V'] = A
    dataout['I'] = (C - offsC) / gainC
    dataout['units'] = {'V':'V', 'I':'A'}
    dataout['Cgain'] = gainC
    dataout['Coffs'] = offsC

    if 'D' in datain:
        D = datain['D']
        dataout['I2'] = (D - offsD) / gainD
        dataout['Dgain'] = gainD
        dataout['Doffs'] = offsD
        dataout['units'].update({'I2':'A'})

    return dataout

def femto_log_to_iv(datain, dtype=np.float32):
    # Adjust output offset so that 0.1V in --> 1V out on the 2V input setting
    # Then 0.1V in --> 1.25V out on the 200mV setting
    # Input offset also important. should minimize the output signal when input is zero
    dataout = datain

    # If data is raw, convert it here
    if datain['A'].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    B = datain['B']

    dataout['V'] = A
    # 2V setting
    #dataout['I'] = 10**((B - 1) / 0.25) * 0.1 / 50
    dataout['I'] = 10**((B - 1) / 0.25) * 0.01 / 50
    dataout['units'] = {'V':'V', 'I':'A'}

    return dataout

def TEO_HFext_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    for TEO HF output channels

    HFV → A
    V_MONITOR → B
    HF_LIMITED_BW → C
    HF_FULL_BW → D
    '''
    # TODO: How can we know what gain setting was used??
    #       For now we will just ask Teo what the current state is..
    teo = instruments.TeoSystem()
    gain_step = teo.gain()

    HFV='A'
    V_MONITOR='B'
    HF_LIMITED_BW='C'
    HF_FULL_BW='D'

    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    chs = [ch for ch in ('A', 'B', 'C', 'D') if ch in datain]
    if datain[chs[0]].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)

    if 'units' not in dataout:
        dataout['units'] = {}

    if 'calibration_teo' not in dataout:
        dataout['calibration_teo'] = {}

    if HFV and (HFV in datain):
        dataout['units']['HFV'] = 'V'
        dataout['HFV'] = datain[HFV]

    if V_MONITOR and (V_MONITOR in datain):
        dataout['units']['V'] = 'V'
        Vdata, Vcal = teo.apply_calibration(datain[V_MONITOR], 'V_MONITOR', gain_step)
        dataout['V'] = Vdata
        dataout['calibration_teo']['V'] = Vcal


    if HF_LIMITED_BW and (HF_LIMITED_BW in datain):
        dataout['units']['I'] = 'A'
        Idata, Ical = teo.apply_calibration(datain[HF_LIMITED_BW], 'HF_LIMITED_BW', gain_step)
        dataout['I'] = Idata
        dataout['calibration_teo']['I'] = Ical

    if HF_FULL_BW and (HF_FULL_BW in datain):
        dataout['units']['I2'] = 'A'
        I2data, I2cal = teo.apply_calibration(datain[HF_FULL_BW], 'HF_FULL_BW', gain_step)
        dataout['I2'] = I2data
        dataout['calibration_teo']['I2'] = I2cal

    # TODO if only one of HF_LIMITED or HF_FULL is used, call the signal I, and indicate somehow where it came from

    return dataout

def Rext_to_iv(datain, R=50, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    This is for the configuration where you are using a series resistance
    for a shunt current measurement

    e.g. using the scope inputs (R = 50Ω or 1MΩ)

    Measure current on channel C (for highest sample rate)
    '''
    Ichannel='C'

    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)

    # Use channel A and C as input, because simultaneous sampling is faster than using A and B
    A = datain['A']
    C = datain[Ichannel]

    # V device
    # No propagation delay considered!
    dataout['V'] = A - C
    dataout['I'] = C / R
    dataout['units'] = {'V':'V', 'I':'A'}
    dataout['Rs_ext'] = R

    return dataout

def digipot_to_iv(datain, gain=1/50, Vd_gain=0.5, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    for digipot circuit with device voltage probe
    gain is in A/V, in case you put an amplifier on the output

    Vd_gain 0.5 for V follower and 50 ohm output/input

    Simultaneous sampling is faster when not using adjacent channels (i.e. A&B)

    Vdevice → ch B
    current → ch C
    '''
    Vd_channel='B'
    I_channel='C'
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    chs_sampled = [ch for ch in ['A', 'B', 'C', 'D'] if ch in datain]
    if not chs_sampled:
        log.error('No picoscope data detected')
        return datain

    dataout = datain
    # If data is raw, convert it here
    if datain[chs_sampled[0]].dtype in (np.int8, np.int16):
        datain = raw_to_V(datain, dtype=dtype)

    if 'units' not in dataout:
        dataout['units'] = {}

    monitor_channel = ivtools.settings.MONITOR_PICOCHANNEL
    if monitor_channel in datain:
        V = datain[monitor_channel]
        dataout['V'] = V # Subtract voltage on output?  Don't know what it is necessarily.
        dataout['units']['V'] = 'V'
    if I_channel in datain:
        I = datain[I_channel] * gain
        dataout['I'] = I
        dataout['units']['I'] = 'A'
    if Vd_channel in datain:
        Vd = datain[Vd_channel]/Vd_gain
        if I_channel in datain: # subtract voltage across 50 ohm to ground
            dataout['Vd'] = Vd - dataout['I']
        else:
            dataout['Vd'] = Vd
        dataout['units']['Vd'] = 'V'

    dataout['Igain'] = gain

    return dataout

def probe_channels(something_to_iv):
    """
    Sends in a probe to the iv conversion function to determine which channels it actually uses.

    If the probe fails it might give a partial set of channels.

    I use this to determine which channels to sample from on picoscope by default
    because I'm too lazy to always remember and type channels=['A', 'B', 'C']
    """
    class Probe(defaultdict):
        touched = set()
        def __getitem__(self, x):
            #print("ouch!")
            self.touched.add(x)
            return super().__getitem__(x)

    a = np.array([1])
    probe = Probe(lambda: a)

    try:
        something_to_iv(probe)
    except:
        log.warning("Probe failed")

    channels = {'A', 'B', 'C', 'D'}
    return probe.touched.intersection(channels)


############# Waveforms ###########################

# TODO add pulse trains

def tri(v1, v2=0, n=None, step=None, repeat=1):
    '''
    Create a triangle pulse waveform with a constant sweep rate.  Starts and ends at zero.

    Can optionally pass number of data points you want, or the voltage step between points.

    If neither n or step is passed, return the shortest waveform which reaches v1 and v2.

    Building these waveforms has some unexpected nuances.  Check the code if unsure.
    '''
    # How this waveform is constructed depends on our value system.
    # For example, when we pass n, do we care most that the waveform:
    # 1. actually has n points?
    # 2. has constant datapoint spacing?
    # 3. contains the extrema?
    # 4. hits all the same values on the up and down sweep?

    # For IV sweeps I would say containing the extrema is important,
    # In order to make sure the waveform actually contains 0, v1 and v2, we need to allow
    # uneven spacing for the extreme points (v1, v2, last 0).
    # I think this is better than strictly enforcing equal sweep rate, because you can then get
    # deformed undersampled triangle waveforms

    # Within this scheme, there does not necessarily exist a step size that leads to n data points.
    # (e.g. going from zero to one extreme and back to zero can't be done in an even number of data points)
    # When a step size does exist, it is not unique.
    # What to do when a step size doesn't exist, and how to find the optimum step size when it does exist?
    # It's a fun problem to think about, but hard to solve.

    #nulls = sum(np.array((v1, v2-v1, v2)) == 0)
    #endpts = 3 - nulls
    if n is not None:
        dv = abs(v1) + abs(v2 - v1) + abs(v2)
        step = dv / (n - 1)
        # If I take the above step, I will end up between 1 and "endpts" extra points
        # Dumb stuff below
        #sstart = dv / (n-1)
        #sstop = dv / (n-endpts-1)
        #stest = np.linspace(sstart, sstop, 1024)
        # number of points you will end up with
        #ntest = abs(v1)//stest + (abs(v2-v1))//stest + abs(v2)//stest + 1 + endpts
        #step = stest[np.where(ntest == n)[0][0]]
    if step is not None:
        def sign(num):
            npsign = np.sign(num)
            return npsign if npsign !=0 else 1
        wfm = np.concatenate((np.arange(0, v1, sign(v1) * step),
                             np.arange(v1, v2, sign(v2 - v1) * step),
                             np.arange(v2, 0, -sign(v2) * step),
                             [0]))
    else:
        # Find the shortest waveform that truly reaches v1 and v2 with constant spacing
        # I don't think we need better than 1 mV resolution
        v1 = round(v1, 3)
        v2 = round(v2, 3)
        f1 = Fraction(str(v1))
        f2 = Fraction(str(v2))
        # This is depreciated for some reason
        #vstep = float(abs(fractions.gcd(fmax, fmin)))
        # Doing it this other way.. Seems faster by a large factor.
        a, b = f1.numerator, f1.denominator
        c, d = f2.numerator, f2.denominator
        # not a typo
        commond = float(b * d)
        vstep = gcd(a*d, b*c) / commond
        dv = v1 - v2
        # Using round because floating point errors suck
        # e.g. int(0.3 / 0.1) = int(2.9999999999999996) = 2
        n1 = round(abs(v1) / vstep + 1)
        n2 = round(abs(dv) / vstep + 1)
        n3 = round(abs(v2) / vstep + 1)
        wfm = np.concatenate((np.linspace(0 , v1, n1),
                            np.linspace(v1, v2, n2)[1:],
                            np.linspace(v2, 0 , n3)[1:]))

        # Filling the AWG record length will probably take more time than it's worth.
        # Interpolate to a "Large enough" waveform size
        #enough = 2**16
        #x = np.linspace(0, 1, enough)
        #xp = np.linspace(0, 1, len(wfm))
        #wfm = np.interp(x, xp, wfm)

        # Let AWG do the interpolation

    if repeat > 1:
        wfm = np.concatenate([wfm[:-1]]*(repeat-1) + [wfm])

    return wfm

def square(vpulse, duty=.5, length=2**12, startval=0, endval=0, startendratio=1):
    '''
    A single square pulse waveform.
    '''
    ontime = int(duty * length)
    remainingtime = length - ontime
    pretime = int(startendratio * remainingtime / (startendratio + 1))
    pretime = max(1, pretime)
    posttime = remainingtime - pretime
    posttime = max(1, posttime)
    prearray = np.ones(pretime) * startval
    pulsearray = np.ones(ontime) * vpulse
    postarray = np.ones(posttime) * endval
    return np.concatenate((prearray, pulsearray, postarray))


############# Misc ##############################

def arg_broadcast(*args):
    '''
    When you have a function that takes multiple list-like arguments of equal length
    it is fairly common to want to broadcast all the arguments to the same length

    e.g.
    arg_broadcast(2, [3,4,5], ...) = [2,2,2], [3,4,5], ...

    will also tile:
    arg_broadcast(1, [2,3], [4,5,6,7]) = [1,1,1,1], [2,3,2,3], [4,5,6,7]

    outputs are numpy arrays even if inputs are not.  should be fine.
    '''
    length = lambda x: 1 if isinstance(x, Number) else len(x)
    lengths = list(map(length, args))
    N = max(lengths)
    return [np.tile(arg, int(N/l)+1)[:N] for arg, l in zip(args, lengths)]

def beep(freq=500, ms=300):
    import winsound
    winsound.Beep(freq, ms)

def tts(text=''):
    # immediately speak the text, does not return until finished
    import pyttsx3
    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty('voices')       #getting details of current voice
    tts_engine.setProperty('voice', voices[1].id)
    tts_engine.say(text)
    tts_engine.runAndWait()

def tts_queue(text):
    # Put some utterances in a queue, then you can call tts_thread() to start it
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.say(text)

def tts_thread(text=None):
    '''
    runs in a separate thread, but you can't call it again until the previous one is complete

    sometimes it gives an error and some unknown condition can get it to work again..

    Seems to be that if you run it outside of a thread first, then it works inside a thread forever.
    therefore you can initialize with text_to_speech('')

    You can't run it again until it finishes -- check the done flag
    '''
    from threading import Thread
    import pyttsx3
    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty('voices')       #getting details of current voice
    tts_engine.setProperty('voice', voices[1].id)
    #tts_engine.say('')
    class Threader(Thread):
        def __init__(self, *args, **kwargs):
            Thread.__init__(self, *args, **kwargs)
            self.daemon = True
            self.start()
        def run(self):
            # Doesn't work
            #while tts_engine.isBusy():
                #time.sleep(.1)
            if text:
                tts_engine.say(text)
            tts_engine.runAndWait()
            # Is this an abuse of a function attribute?
            tts_thread.done = True
    tts_thread.done = False
    ttsthread = Threader()

class telegram_bot:
    '''
    This class enables use of a telegram bot for basic remote control of the measurement setup
    e.g.: receive interactive figures, microscope pictures, move the sample stage to the next device or start pre-configured electrical measurements

    you need to do 'pip install python-telegram-bot'
    https://docs.python-telegram-bot.org/en/stable/

    
    the bot_token is used to control the telegram bot 'TS-Bot' with the bot username: 'TS_controller_bot'
    use 'await' before async functions like 'await tb.send_hello_message()'

    to make the bot communicate with you in your personal chat:
    search telegram app for TS-Bot
    send any message to this bot
    get your personal chat_id (this returns the chat_id of the most recent received message):
    tb = telegram_bot()
    chat_id = await tb.get_recent_chat_id()

    set up the bot to talk with you on your personal chat:
    tb = telegram_bot(chat_id=chat_id)
    await tb.send_hello_message()
    '''

    def __init__(self, chat_id=None, bot_token='5927560730:AAEXhbOeRxhKoyb9xBmeF6PrrRNC5SR5-yc'):
        # Avoid having telegram as an ivtools dependency
        import telegram
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
        from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

        telegram_bot.telegram = telegram
        telegram_bot.Update = Update
        telegram_bot.InlineKeyboardMarkup = InlineKeyboardMarkup
        telegram_bot.InlineKeyboardButton = InlineKeyboardButton
        telegram_bot.Application = Application
        telegram_bot.CallbackQueryHandler = CallbackQueryHandler
        telegram_bot.CommandHandler = CommandHandler
        telegram_bot.ContextTypes = ContextTypes
    
        self.chat_id = chat_id
        self.bot_token = bot_token

        self.bot = telegram.Bot(token=self.bot_token)
        

    async def get_recent_chat_id(self):
        '''
        this will return the chatid from the most recently received chat message
        send the bot any text message before running this function to get your own personal chat_id of your conversation with the bot
        you need the chat_id to let the bot send any message to the given chat
        '''
        async with self.bot:
            update = await self.bot.get_updates()
        
        # this gets the chat_id 
        chat_id = update[-1].message.chat_id

        return chat_id

    async def get_recent_text(self):
        '''
        this will return the text string from the most recently received chat message
        '''
        async with self.bot:
            update = await self.bot.get_updates()
        
        # this gets the chat_id 
        text = update[-1].message.text

        return text
    
    async def get_recent_chat_user_name(self):
        '''
        this will return the username from the most recently received chat
        '''
        # get most recent update from chatid
        async with self.bot:
            update = await self.bot.get_updates()

        # this gets the username from the chatid
        name = update[-1].message.from_user.first_name

        return name
        
    async def get_bot_user_name(self):
        '''
        this will return the username of the bot
        '''
        # get info about bot
        async with self.bot:
            bot_info = await self.bot.get_me()

        # this gets the username of the bot
        bot_user_name = bot_info.username

        return bot_user_name

    async def get_bot_name(self):
        '''
        this will return the name of the bot
        '''
        # get info about bot
        async with self.bot:
            bot_info = await self.bot.get_me()

        # this gets the username of the bot
        bot_name = bot_info.full_name

        return bot_name

    async def get_name_of_user(self):
        '''
        this will return the (first) name of the user from the given chatid
        '''
        # get user info
        async with self.bot:
            user_info = await self.bot.get_chat(chat_id=self.chat_id)
        
        # get first name from user info
        name = user_info.first_name

        return name

    async def send_hello_message(self):
        '''
        this will send a hello message to the chatid with the name of the receiver in the text
        '''
        # get first name of user from the given chat_id
        name = await self.get_name_of_user()

        async with self.bot:
            await self.bot.sendMessage(text=f'Hello {name}', chat_id=self.chat_id)
    
    async def send_message(self, text=None):
        '''
        this will send a message with the given text to the chatid
        '''
        async with self.bot:
            await self.bot.sendMessage(text=text, chat_id=self.chat_id)

    async def send_picture(self, filepath=None, web_link=None):
        '''
        this will send a picture to the chatid
        pass local filepath or web_link to a picture
        '''
        async with self.bot:
            if filepath:
                # from local drive
                await self.bot.send_photo(photo=open(filepath, 'rb'), chat_id=self.chat_id)
            elif web_link:
                # from internet
                await self.bot.send_photo(photo='https://telegram.org/img/t_logo.png', chat_id=self.chat_id)
            else:
                log.warning('you should pass a filepath or an image web-link')
    
    async def start(self, update, context) -> None:

        """Sends a message with three inline buttons attached."""

        keyboard = [
            [
                telegram_bot.InlineKeyboardButton("Option 1", callback_data="1"),
                telegram_bot.InlineKeyboardButton("Option 2", callback_data="2"),
            ],

            [telegram_bot.InlineKeyboardButton("Option 3", callback_data="3")],
        ]

        reply_markup = telegram_bot.InlineKeyboardMarkup(keyboard)

        # await update.message.reply_text("Please choose:", reply_markup=reply_markup)
        await telegram_bot.Update.message.reply_text("Please choose:", reply_markup=reply_markup)

    async def send_menu(self):
        '''
        This will send a button menu and return what button was pressed
        '''
        async def button(update, context) -> None:
            """Parses the CallbackQuery and updates the message text."""

            query = update.callback_query

            # CallbackQueries need to be answered, even if no notification to the user is needed
            # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery

            await query.answer()

            await query.edit_message_text(text=f"Selected option: {query.data}")

        application = telegram_bot.Application.builder().token('5927560730:AAEXhbOeRxhKoyb9xBmeF6PrrRNC5SR5-yc').build()
        application.add_handler(telegram_bot.CommandHandler("start", start))
        application.add_handler(telegram_bot.CallbackQueryHandler(button))
        application.run_polling()



class controlled_interrupt():
    '''
    Allows you to protect code from keyboard interrupt using a context manager.
    Potential safe break points can be individually specified by breakpoint().
    If a ctrl-c is detected during protected execution, the code will be interrupted at the next break point.
    You can always press ctrl-c TWICE to bypass this protection and interrupt immediately.
    ctrl-c behavior will be returned to normal at the end of the with block
    '''
    def __init__(self):
        self.interruptable = None
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, *args):
        self.stop()
    def start(self):
        signal.signal(signal.SIGINT, self.int_handler)
        self.interruptable = False
    def stop(self):
        # Now you can use "ctrl+c" as usual.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    def int_handler(self, signalNumber, frame):
        if self.interruptable:
            signal.default_int_handler()
        else:
            print('Not safe to interrupt! Will interrupt at next opportunity.')
            print('Press ctrl-c again to override')
            self.interruptable = True
    def breakpoint(self):
        if self.interruptable:
            raise KeyboardInterrupt
