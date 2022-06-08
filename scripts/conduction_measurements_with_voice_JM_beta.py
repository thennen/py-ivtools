import winsound


###
meta.load_lassen(dep_code='Ente', sample_number=[1], module=['001'], die_rel=1, device=2)
#meta.static['polarity'] = 'HFV_right'
#meta.static['Keithley_connection'] = 'LFI_ground'

# for V2O3 conduction measurements
Vmax = 0.3
Vmin = -Vmax
npts = 60
Ilimit = 300e-6

temps = [20] #, 30, 40, 50, 60, 70, 80]

### Setup - don't use TEO for now
#setup_picoteo()
setup_keithley()
#teo.LF_mode()
meta.i = -1

### Some functions

# Try to find current measurement range for Keithley
# start with low range and apply triangle
# increase range until all readings in range
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

# Stuff to be done on each device
def measure_device(Vmin, Vmax, npts, Ilimit, temp):
    #wfm = tri(Vmin, Vmax, n=npts)
    wfm = tri(Vmin, Vmax, step=0.05)
    
    set_temperature(temp, delay=10)
    tts("Set temperature " + str(temp) + " C")
    # Best range might depend on temperature
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
    for T in temps:
        d = measure_device(Vmin, Vmax, npts, Ilimit, T)
        try:
            R = resistance(d)
            tts(metric_prefix_longname(R, 0) + ' ohms')
        except:
            tts('W T F')
