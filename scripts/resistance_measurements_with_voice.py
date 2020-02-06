import winsound
# pip install pyttsx3
# pip install pypiwin32
import pyttsx3
engine = pyttsx3.init()

voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)

def metric_prefix_longname(x):
    longnames = ['exa', 'peta', 'tera', 'giga', 'mega', 'kilo', '', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto']
    prefix = ['E', 'P', 'T', 'G', 'M', 'k', '', 'm', '$\mu$', 'n', 'p', 'f', 'a']
    values = [1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18]
    if abs(x) < min(values):
        return '{:n}'.format(x)
    for v, p in zip(values, longnames):
        if abs(x) >= v:
            return '{:n} {}'.format(round(x/v), p)

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
