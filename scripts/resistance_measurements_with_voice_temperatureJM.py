import winsound


###
meta.load_lassen(dep_code='Birke', sample_number=[5, 6, 7, 8], module=['001I','001H'], die_rel=2)
meta.static['polarity'] = 'HFV_right'
meta.static['Keithley_connection'] = 'LFI_ground'

Vmax = 5.0
Vmin = -Vmax
npts = 60
Ilimit = 300e-6

temp = 60

### Setup
setup_picoteo()
teo.LF_mode()
meta.i = -1


ts.set_temperature(temp)
tts("Set temperature " + str(temp) + " C")

### Some functions

def autorange(Vmin, Vmax, npts=2, Irange0 = 1e-10):
    kiv_silent = interactive_wrapper(k.iv, k.get_data, donefunc = k.done, autosave=False, shared_kws=["ch"])
    
    wfm_auto = tri(Vmin, Vmax, n=npts)
    
    Irange = Irange0
    while True:    
        iplots.enable = False
        #old_beep = measure.beep()
        measure.beep = lambda: 0
        d = kiv_silent(wfm_auto, measure_range=Irange, i_limit=Ilimit, ch="B")
        if not any(np.isnan(d["I"])):
            break
        Irange = 10*Irange
    
    iplots.enable = True
    return Irange

def measure_device(Vmin, Vmax, npts, Ilimit):
    wfm = tri(Vmin, Vmax, n=npts)
    Irange = autorange(Vmin, Vmax)
    tts("Using range " + metric_prefix_longname(Irange, 0) + " amp ")
    d = kiv(wfm, measure_range=Irange, i_limit=Ilimit, ch="B")
    winsound.Beep(500, 300)
    return d


    
### Begin meatframe for loop

#meta.print()
module = None
sample = None
while meta.i < len(meta.df):
    meta.next()
    if meta['sample_number'] != sample:
        tts('Sample ' + str(meta['sample_number']))
    if meta['module'] != module:
        tts('Module ' + meta['module'])
    tts('Device ' + str(meta['device']))
    module = meta['module']
    sample = meta['sample_number']
    # Switch the sample contact now!!!
    #plt.pause(.1)
    ans = input('Switch to the next sample.  Press enter to measure (q to quit) ')
    if ans == 'q':
        break
    elif ans == 's':
        continue
    elif ans == 'r':
        meta.previous()
        continue
    print('Measuring')
    tts('Measuring')
    d = measure_device(Vmin, Vmax, npts, Ilimit)
    try:
        R = resistance(d)
        tts(metric_prefix_longname(R, 0) + ' ohms')
    except:
        tts('W T F')
