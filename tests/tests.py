# Here are some test functions probably not conforming to any software engineering conventions

import sys
import os
ivtoolsdir = os.path.join(os.path.split(sys.path[0])[0], 'ivtools')

def name_collisions():
    # Interactive script imports all names into the same namespace
    # So make sure there are no name collisions
    import ast

    filenames = ['measure.py', 'io.py', 'plot.py', 'analyze.py']
    names = []
    for fn in filenames:
        with open(os.path.join(ivtoolsdir, fn)) as file:
            node = ast.parse(file.read())

        functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        funcnames = [f.name for f in functions]
        classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
        classnames = [c.name for c in classes]
        names.append(set(funcnames + classnames))

    fail = False
    from itertools import combinations
    for (i, j) in combinations(range(4), 2):
        intersect = names[i].intersection(names[j])
        if any(intersect):
            print('Name collision!')
            print(filenames[i])
            print(filenames[j])
            print(intersect)
            fail = True

    return fail



def test_rigol_1():
    '''
    trigger jitter and delay with usb loaded waveforms
    For some reason I see about a microsecond delay between trigger and waveform
    Does it depend on number of samples?
    '''
    from ivtools import instruments
    ps = instruments.Picoscope()
    rigol = instruments.RigolDG5000()
    rigol.setup_burstmode()
    rigol.output(1)

    # just resample this
    # Something recognizable with a sharp edge
    def wfm(t):
        w = 0.8 + 0.2*np.sin(t*2*np.pi)
        w[0] = 0
        w[-1] = 0
        return w
    lens = np.int32(np.geomspace(1000, 100000, 20))
    wfms = [wfm(np.linspace(0,1,n)) for n in lens]
    freq = 1e6
    filenames = [f'wfm{i}.RAF' for i in range(len(wfms))]

    ans = input('Write wfms to usb drive at F: ?')
    if ans.lower() == 'y':
        print('Writing waveforms to USB drive at F:')
        for wfm, fn in zip(wfms, filenames):
            rigol.write_wfm_file(wfm, os.path.join('F:', fn))
        input('Put the usb drive into rigol. (press enter)')

    sleep = 3
    amp = 1
    offs = 0
    dur = 1e-6
    fs = 1e9
    data = []
    for wfm, fn in zip(wfms, filenames):
        rigol.load_wfm_usbdrive(fn, wait=False)
        # Long wait to be absolutely sure that rigol is finished loading wfm and had time to rest...
        time.sleep(sleep)
        # x2 for idiot definition of amplitude, x2 for 50 ohm termination
        # There could still be some error here, both in the measurement by picoscope and by rigol..
        rigol.amplitude(2*2*(amp + abs(offs)))
        rigol.period(dur)
        time.sleep(.05) # I don't think it's needed, just trying to stop crash
        ps.capture(ch=['A'], freq=fs, duration=3*dur, chrange={'A':1}, choffset={'A':0}, chcoupling={'A':'DC50'})
        rigol.trigger(1)
        d = ps.get_data(['A'], raw=False)
        d['t'] = analyze.maketimearray(d)
        d['len'] = len(wfm)
        data.append(d)
    df = pd.DataFrame(data)
    return df

def test_rigol_2():
    '''
    How long does waveform loading take from USB stick?
    vs rear loading binary?
    '''
    from ivtools import instruments
    rigol = instruments.RigolDG5000()

    # just resample this
    # Something recognizable with a sharp edge
    def wfm(t):
        w = 0.8 + 0.2*np.sin(t*2*np.pi)
        w[0] = 0
        w[-1] = 0
        return w
    #lens = np.int32(np.geomspace(1000, 16e6, 40))
    lens = 2**np.arange(12, 25)
    #lens = np.int32(2**np.arange(12, 24, .5))
    #lens = np.int32(2**np.arange(19, 22, .1))
    wfms = [wfm(np.linspace(0,1,n)) for n in lens]
    freq = 1e6
    #filenames = [f'wfm{i}.RAF' for i in range(len(wfms))]
    filenames = [io.hash_array(a)[:8]+'.RAF' for a in wfms]

    ans = input('Write wfms to usb drive at F: ?')
    if ans.lower() == 'y':
        print('Writing waveforms to USB drive at F:')
        for wfm, fn in zip(wfms, filenames):
            rigol.write_wfm_file(wfm, os.path.join('F:', fn))
        input('Put the usb drive into rigol. (press enter)')

    times = []
    for wfm, fn in zip(wfms, filenames):
        l = len(wfm)
        time.sleep(1)
        print(f'loading length {l} waveform')
        t0 = time.time()
        rigol.load_wfm_usbdrive(fn, wait=True)
        t1 = time.time()
        times.append(t1-t0)

    plt.figure()
    plt.plot(np.log(lens)/np.log(2), times, '.-')
    plt.xlabel('waveform length (log2)')
    plt.ylabel('time to load from front usb [s]')

    return lens, times

def test_rigol_3():
    '''
    What is the distribution of loading times?
    '''
    from ivtools import instruments
    rigol = instruments.RigolDG5000()

    # just resample this
    # Something recognizable with a sharp edge
    def wfm(t):
        w = 0.8 + 0.2*np.sin(t*2*np.pi)
        w[0] = 0
        w[-1] = 0
        return w
    lens = 2**16, 2**18
    wfms = [wfm(np.linspace(0,1,n)) for n in lens]
    freq = 1e6
    filenames = [f'wfm{i}.RAF' for i in range(len(wfms))]

    ans = input('Write wfms to usb drive at F: ?')
    if ans.lower() == 'y':
        print('Writing waveforms to USB drive at F:')
        for wfm, fn in zip(wfms, filenames):
            rigol.write_wfm_file(wfm, os.path.join('F:', fn))
        input('Put the usb drive into rigol. (press enter)')

    times = []
    nrepeats = 50
    for wfm, fn in zip(wfms, filenames):
        tt = []
        for n in range(nrepeats):
            l = len(wfm)
            time.sleep(1)
            print(f'loading length {l} waveform. {n+1}/{nrepeats}')
            t0 = time.time()
            rigol.load_wfm_usbdrive(fn, wait=True)
            t1 = time.time()
            tt.append(t1-t0-1) # load_wfm_usbdrive waits a second
        times.append(tt)
    return lens, times

def test_rigol_wfms_1(sleep=8):
    '''
    Rigol is a flaky piece of crap
    we need to reliably output long waveforms
    only way to do this is from the front USB port
    rigol crashes after a certain number of waveforms

    send a bunch of waveforms and see if they are getting output correctly
    connect Rigol output to picoscope channel A
    connect Rigol sync output to picoscope ext trigger
    load a bunch of waveforms to the usb drive
    '''
    from ivtools import instruments
    ps = instruments.Picoscope()
    rigol = instruments.RigolDG5000()
    rigol.setup_burstmode()
    rigol.output(1)

    # Generate a few long waveforms
    # modulated sinewaves
    awg_fs = 4e8
    nwfms = 10
    nsamples = 2**20
    dur = nsamples / awg_fs
    # Carrier freq
    # If you keep the frequency far below 1/trigger_jitter ~ 100 MHz, we maybe don't have to cross-correlate
    f1 = 1e6
    # Modulation freqs
    f2 = np.linspace(1e4, 5e4, nwfms)
    t = np.linspace(0, dur, nsamples)
    wfms = [np.sin(f1*2*np.pi*t)*np.sin(ff2*2*np.pi*t) for ff2 in f2]
    filenames = [io.hash_array(a)[:8]+'.RAF' for a in wfms]
    #filenames = [f'wfm{i}.RAF' for i in range(len(wfms))]
    amp = 1
    offs = 0
    # same as awg, so that we can compare the waveforms easily
    # Oh wait I can't do 1.0 GS/s
    scope_fs = 1e9

    ans = input('Write wfms to usb drive at F: ?')
    if ans.lower() == 'y':
        print('Writing waveforms to USB drive at F:')
        for wfm, fn in zip(wfms, filenames):
            rigol.write_wfm_file(wfm, os.path.join('F:', fn))
        input('Put the usb drive into rigol. (press enter)')

    def is_wfm_good(programmed, measured):
        # Cross correlation to tell if you got the waveform you asked for
        # programmed and measured are in general sampled at different frequencies
        # There is trigger jitter which could be important depending on the sampling freq
        #lags = 
        #crosscorr = [np.correlate(programmed, measured)]
        # I assume they correspond to the same amount of time at least
        nsamples = len(programmed)
        interp = np.interp(np.linspace(0,1,nsamples), np.linspace(0, 1, len(measured)), measured)
        corr = np.correlate(programmed, interp) / np.correlate(programmed, programmed)
        plt.figure()
        plt.plot(programmed)
        plt.plot(interp)
        return 0.8 < corr < 1.2

    for wfm, fn in zip(wfms, filenames):
        rigol.load_wfm_usbdrive(fn, wait=False)
        # Long wait to be absolutely sure that rigol is finished loading wfm and had time to rest...
        for _ in range(int(sleep)):
            time.sleep(1)
        # x2 for idiot definition of amplitude, x2 for 50 ohm termination
        # There could still be some error here, both in the measurement by picoscope and by rigol..
        rigol.amplitude(2*2*(amp + abs(offs)))
        rigol.period(dur)
        time.sleep(.05) # I don't think it's needed, just trying to stop crash
        ps.capture(ch=['A'], freq=scope_fs, duration=dur, chrange={'A':1}, choffset={'A':0}, chcoupling={'A':'DC50'})
        rigol.trigger(1)
        d = ps.get_data(['A'], raw=False)
        worked = is_wfm_good(wfm, d['A'])
        break
        if worked:
            print('Worked!')
        else:
            print('Failed!')
            return wfm, d

def test_rigol_wfms_2(sleep=8):
    '''
    difference between waveforms when loaded different ways (strings, ints, binary, front usb)
    limits of lengths?
    strings < ~20,000
    ints < ~40,000 ??
    binary <= 2^16 = 65,563
    front < 2^24 = 16 MS
    '''
    from ivtools import instruments
    ps = instruments.Picoscope()
    rigol = instruments.RigolDG5000()
    rigol.setup_burstmode()
    rigol.output(1)



if __name__ == '__main__':
    fail = name_collisions()
    if not fail:
        print('No name collisions detected!')
