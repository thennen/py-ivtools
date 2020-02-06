# Expects you to be in the interactive environment (run -i interactive.py)
# You can call this script with 
sys.argv
#meta.load_lassen(dep_code='Esel', sample_number=[1,2,3,4,5], module=['001G','001H','001I','001E'], die_rel=1)
meta.load_lassen(dep_code='Lachs', sample_number=[2], module=['001G','001H','001I','001E'], die_rel=1)
meta.i = -1

#### Begin meatframe for loop

#meta.print()
module = None
sample = None
while meta.i < len(meta.df):
    meta.next()
    if meta['sample_number'] != sample:
        engine.say('Sample ' + str(meta['sample_number']))
    if meta['module'] != module:
        engine.say('Module ' + meta['module'])
    engine.say('Device ' + str(meta['device']))
    engine.runAndWait()
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
    engine.say('Measuring')
    engine.runAndWait()
    wfm = tri(-.2, .2, n=20)
    wfm = wfm[abs(wfm) > 0.01] # so outrange doesn't take a decade
    d = kiv(wfm, Irange=0, Ilimit=2e-3)
    try:
        R = resistance(d)
        engine.say(metric_prefix_longname(R) + ' ohms')
    except:
        engine.say('W T F')
    engine.runAndWait()
    #winsound.Beep(500, 400)
# End meatframe forloop
