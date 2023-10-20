# Functions that Alejandro split up and I haven't had a chance to review yet - TH

def pico_measure(wfm, n, channels, fs, duration, pretrig, termination, V_source,
                 interpwfm, savewfm, autosmoothimate, autosplit, splitbylevel):

    ps = instruments.Picoscope()

    # Set picoscope to capture
    # Sample frequencies have fixed values, so it's likely the exact one requested will not be used
    actual_fs = ps.capture(ch=channels,
                           freq=fs,
                           duration=duration,
                           pretrig=pretrig)

    # This makes me feel good, but I don't think it's really necessary
    time.sleep(.05)
    if termination:
        # Account for terminating resistance
        # e.g. multiply applied voltages by 2 for 50 ohm termination
        wfm *= (50 + termination) / termination

    # Send a pulse
    if V_source == 'rigol':
        rigol = instruments.RigolDG5000()
        rigol.pulse_arbitrary(wfm, duration=duration, interp=interpwfm, n=n, ch=1)
    elif V_source == 'teo':
        teo = instruments.TeoSystem()
        teo.output_wfm(wfm, n=n)
    else:
        raise Exception(f"V_source '{V_source}' not recognized.")


    log.info('Applying pulse(s) ({:.2e} seconds).'.format(duration))
    time.sleep(duration * 1.05)
    #ps.waitReady()
    log.debug('Getting data from picoscope.')
    # Get the picoscope data
    # This goes into a global strictly for the purpose of plotting the (unsplit) waveforms.
    chdata = ps.get_data(channels, raw=True)
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

    if autosplit and (n > 1):
        log.debug('Splitting data into individual pulses')
        if splitbylevel is None:
            nsamples = duration * actual_fs
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


def picoiv_new(wfm, duration=1e-3, n=1, fs=None, nsamples=None, smartrange=1, autosplit=True,
            termination=None, channels=['A', 'B'], autosmoothimate=False, splitbylevel=None,
            savewfm=False, pretrig=0, posttrig=0, interpwfm=True, **kwargs):
    ''' Alejandro's attempt to split up picoiv so that we can reuse the code while using a different AWG. Not tested yet. '''
    rigol = instruments.RigolDG5000()

    wfm = np.array(wfm) if not type(wfm) == np.ndarray else wfm

    if (bool(fs) * bool(nsamples)):
        raise Exception('Can not pass fs and nsamples, only one of them')
    if fs is None:
        fs = nsamples / duration

    if smartrange:
        # Smart range the monitor channel
        smart_range(np.min(wfm), np.max(wfm), ch=[ivtools.settings.MONITOR_PICOCHANNEL])

    sampling_factor = (n + pretrig + posttrig)

    sampling_factor = (n + pretrig + posttrig)

    ivdata = pico_measure(wfm=wfm, n=n, channels=channels, fs=fs, duration=duration*sampling_factor,
                          pretrig=pretrig / sampling_factor,
                          termination=termination, V_source='rigol', interpwfm=interpwfm, savewfm=savewfm,
                          autosmoothimate=autosmoothimate, autosplit=autosplit, splitbylevel=splitbylevel)

    return ivdata


def picoteo_new(wfm, duration=None, n=1, fs=None, nsamples=None, smartrange=True, autosplit=False,
             termination=None, autosmoothimate=False, splitbylevel=None,
             savewfm=False, save_teo_int=True, pretrig=0, posttrig=0,
             V_MONITOR='B', HF_LIMITED_BW='C', HF_FULL_BW='D', Irange=None):
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
    '''

    teo = instruments.TeoSystem()
    ps = instruments.Picoscope()

    if type(wfm) is str:
        wfm_name = wfm
        wfm = teo.waveforms[wfm_name][0] if wfm_name in teo.waveforms else teo.download_wfm(wfm_name)[0]
    else:
        wfm_name = None
        wfm = np.array(wfm) if not type(wfm) == np.ndarray else wfm
    len_wfm = len(wfm)

    if duration is not None:
        wfm = teo.interp_wfm(wfm, duration)
        wfm_name = None  # New wfm, not in memory
    else:
        duration = (len_wfm - 1) / teo.freq


    channels = [ch for ch in [V_MONITOR, HF_LIMITED_BW, HF_FULL_BW] if ch is not None]

    if Irange is not None:
        gain_step = teo.Irange(I=Irange, LBW=bool(HF_LIMITED_BW), HBW=bool(HF_FULL_BW), INT=bool(save_teo_int))
    else:
        gain_step = teo.gain()

    log.info(f"gain_step = {gain_step}")

    if smartrange:
        Vrange = max(abs(np.min(wfm)), np.max(wfm))

        def find_ps_range(range, cahnnel, gain_step):
            range_pos, cal = teo.apply_calibration(datain=range, channel=cahnnel, gain_step=gain_step, reverse=True)
            range_neg, cal = teo.apply_calibration(datain=-range, channel=cahnnel, gain_step=gain_step, reverse=True)
            return max(abs(range_pos), abs(range_neg))

        if V_MONITOR is not None:
            ch = V_MONITOR.lower()
            setattr(ps.coupling, ch, 'DC')
            setattr(ps.range, ch, find_ps_range(range=Vrange, cahnnel='V_MONITOR', gain_step=gain_step))

        if Irange is not None:
            if HF_LIMITED_BW is not None:
                ch = HF_LIMITED_BW.lower()
                setattr(ps.coupling, ch, 'DC50')
                setattr(ps.range, ch, find_ps_range(range=Irange, cahnnel='HF_LIMITED_BW', gain_step=gain_step))
            if HF_FULL_BW is not None:
                ch = HF_FULL_BW.lower()
                setattr(ps.coupling, ch, 'DC50')
                setattr(ps.range, ch, find_ps_range(range=Irange, cahnnel='HF_FULL_BW', gain_step=gain_step))


    if fs is None and nsamples is None:
        fs = teo.freq
    elif fs is None:
        fs = nsamples / duration
    elif nsamples is None:
        raise Exception('Can not pass fs and nsamples, only one of them')


    chunksize = 2 ** 11
    npad = chunksize - (len_wfm % chunksize)
    pad_duration = (npad - 1) / fs

    # There is a delay of some ns on the triggering, so that has to passed to ps.capture, but it is passed
    # in clock cycles units.
    # Actually, each channel has its own delay, V_MONITOR is 4 ns, HF_LIMITED_BW is 13 ns, and HF_FULL_BW is 9 ns
    pico_clock_freq = 1e9
    delay_sec = 4e-9
    delay = int(pico_clock_freq * delay_sec)

    # Let pretrig and posttrig refer to the fraction of a single pulse, not the whole pulsetrain
    sampling_factor = (n + pretrig + posttrig)

    wfm_ = wfm if wfm_name is None else wfm_name

    ivdata = pico_measure(wfm=wfm_, n=n, channels=channels, fs=fs, duration=(duration+pad_duration) * sampling_factor,
                          pretrig=pretrig / sampling_factor,
                          termination=termination, V_source='teo', interpwfm=False, savewfm=savewfm,
                          autosmoothimate=autosmoothimate, autosplit=autosplit, splitbylevel=splitbylevel)

    if save_teo_int:
        teo_data = teo.get_data()
        # renaming teo names like: name -> name_teo
        for k, v in teo_data.items():
            if k in ['calibration', 'units']:
                if k not in ivdata:
                    ivdata[k] = {}
                for kk, vv in v.items():
                    ivdata[k][f"{kk}_teo"] = vv
            else:
                ivdata[f"{k}_teo"] = v
        if not savewfm:
            del ivdata['Vwfm_teo']
            del ivdata['units']['Vwfm_teo']

    return ivdata
