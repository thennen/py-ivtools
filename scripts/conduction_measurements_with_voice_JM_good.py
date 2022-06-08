import winsound
import time

# Select device
#   Select voltage - must be outer loop as devices might break
#       Select temperature
#           IV sweep


### Devices to measure
meta.load_lassen(dep_code='Ente', sample_number=[3], module=['001'], die_rel=1, device = [7, 8])

### How are the needles connected to the Keithley?
meta.static['left_needle'] = 'ground'
meta.static['right_needle'] = 'voltage'

### Configuration for conductance measurements
T_meas = list(range(30, 110, 10))    # 10 to 80 C in 10 C steps
#V_peak = [round(1/3, 2), round(2/3, 2), 1]
V_peak = [0.034, 0.1, 0.3, 0.9, 2.7] # If used with 10, 30, 90 nm this gives at least 3 identical max field strengths
# But is that needed?
#V_step = 5e-3
n_steps = 85

Ilimit = 5e-3   # Not really important, should not be in effect for this


### Setup - Keithley directly connected
setup_keithley()

### Autorange helper function
# Try to find current measurement range for Keithley
# start with low range and apply triangle
# increase range until all readings in range
def autorange(Vmin, Vmax, npts=2, Irange0 = 1e-10):
    # Setup so the measurement doesn't beep or save data
    kiv_silent = interactive_wrapper(k.iv, k.get_data, donefunc = k.done, autosave=False, shared_kws=["ch"])
    iplots.enable = False
    measure.beep = lambda: 0
    
    wfm_auto = tri(Vmin, Vmax, n=npts)
    Irange = Irange0
    while True:    
        d = kiv_silent(wfm_auto, measure_range=Irange, i_limit=Ilimit, ch="B")
        if not any(np.isnan(d["I"])):
            break
        Irange = 10*Irange
    
    iplots.enable = True
    return Irange

# Input to check if the stage is on

### Begin meatframe for loop
print('Starting measurement!')
tts('Starting measurement!')

meta.i = -1
module = None
sample = None
while meta.i < len(meta.df)-1:
    set_temperature(T_meas[0], delay=20)
    tts("Set temperature " + str(T_meas[0]) + " C")
    
    
    meta.next()
    print(meta.i)
    #meta.i += 1
    if meta['sample_number'] != sample:
        tts('Sample ' + str(meta['sample_number']))
    if meta['module'] != module:
        tts('Module ' + meta['module'])
    tts('Device ' + str(meta['device']))
    module = meta['module']
    sample = meta['sample_number']

    ans = input('Switch to the next sample.  Press enter to measure (q to quit) ')
    if ans == 'q':
        break
    elif ans == 's':
        continue
    elif ans == 'r':
        meta.previous()
        continue
    
    print('Measuring device')
    tts('Measuring device')
    
    # Figure out how long this runs
    s = time.time()
    
    for Vmax in V_peak:
        #wfm = tri(-Vmax, Vmax, step = V_step)
        wfm = tri(-Vmax, Vmax, n = n_steps)
                
        for T in T_meas:
            set_temperature(T, delay=15)
            tts("Set temperature " + str(T) + " C")
            
            Irange = autorange(-Vmax, Vmax)
            tts("Using range " + metric_prefix_longname(Irange, 0) + " amp ")
            
            tts("Sweep started!")
            d = kiv(wfm, measure_range=Irange, i_limit=Ilimit, ch="B")
            winsound.Beep(500, 300)
            
        set_temperature(T_meas[0], delay=30)
        
    print(f"Measureing one device took {time.time()-s} seconds!")
    tts(f"Measureing one device took {time.time()-s} seconds!")

print('Measurement complete!')
tts('Measurement complete!')