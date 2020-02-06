meta.load_lassen(dep_code='Dorsch', sample_number=[1,2,3,4], module=['001I','001E','001B','001'], die_rel=1)
meta.static['polarity'] = '+left'
Vmax = 0.3
Vmin = -Vmax
npts = 30
Ilimit = 2e-3
meta.i = -1

tts('')

#### Begin meatframe for loop

#meta.print()
module = None
sample = None
while meta.i < len(meta.df):
    meta.next()
    if meta['sample_number'] != sample:
        tts_queue('Sample ' + str(meta['sample_number']))
    if meta['module'] != module:
        tts_queue('Module ' + meta['module'])
    tts_queue('Device ' + str(meta['device']))
    tts_thread()
    module = meta['module']
    sample = meta['sample_number']
    # Switch the sample contact now!!!
    #plt.pause(.1)
    ans = input('Switch to the next sample.  Press enter to measure to measure (q to quit) ')
    if ans == 'q':
        break
    elif ans == 's':
        continue
    elif ans == 'r':
        meta.previous()
        continue
    print('Measuring')
    tts_thread('Measuring')
    wfm = tri(Vmin, Vmax, n=npts)
    wfm = wfm[abs(wfm) > 0.005] # so outrange doesn't take a decade
    d = kiv(wfm, Irange=0, Ilimit=Ilimit)
    try:
        R = resistance(d)
        tts_thread(metric_prefix_longname(R, 0) + ' ohms')
    except:
        tts_thread('W T F')
