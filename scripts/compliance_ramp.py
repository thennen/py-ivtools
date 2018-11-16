# For compliance ramp without changing voltage waveform
# Not for huge datasets, will keep everything in memory to try to speed up measurement time
# then smoothimate, split, and save at the end
import winsound

t0 = time.time()
############################################# Settings

cc_start = 20e-6
cc_end = 900e-6
cc_step = 10e-6

# Voltage waveform to apply
wfm = tri(5, 0)
# Number of pulses for each compliance current
npulses = 100
pulseduration = 1e-3

# Picoscope channels to measure
channels = ['A', 'B']
# Approximate number of (pre-decimation) samples to capture per loop (determines sampling frequency)
nsamples = 50000
downsampling = 40
smoothing = 100
passes = 1

############################################## Script

trainduration = npulses * pulseduration

# Desired sampling frequency
fs = nsamples / pulseduration

load_volatile_wfm(wfm, pulseduration, n=npulses, ch=1, interp=True)

# Loop that takes all the data
ivdatalist = []
for Ic in arange(cc_start, cc_end, cc_step):
    set_compliance(Ic)

    smart_range(np.min(wfm), np.max(wfm), ch=['A', 'B'])

    # Set picoscope to capture
    # Sample frequencies have fixed values, so it's likely the exact one requested will not be used
    actual_fs = pico_capture(ch=channels, freq=fs, duration=trainduration)
    # Send a pulse
    trigger_rigol(ch=1)

    print('Applying pulse(s) ({:.2e} seconds).'.format(trainduration))
    time.sleep(trainduration * 1.05)

    print('Getting data from picoscope.')
    # Get the picoscope data
    # Convert to IV data (keeps channel data)
    ivdatalist.append(pico_to_iv(get_data(channels, raw=True)))
    print('Got data from picoscope.')

t1 = time.time()
print('All data collected and stored in memory.')
print('Smoothimating and splitting loops...')
print('Move to the next sample if you want...')
winsound.Beep(500, 300)

actual_nsamples = pulseduration * actual_fs

# Just split data
#splitdf = []
#for ivdata in ivdatalist:
#    splitdf.extend(splitiv(ivdata, nsamples=actual_nsamples))

# Loop that processes all the data
processeddata = []
for ivdata in ivdatalist:
    ivdata = smoothimate(ivdata, window=smoothing, factor=downsampling, passes=passes, columns=None)
    ivdata = splitiv(ivdata, nsamples=actual_nsamples / downsampling / passes)
    processeddata.extend(ivdata)

t2 = time.time()
print('Writing data to disk ...')

#s = processeddata[0]
s = devicemeta
filename = datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]
for fnkey in filenamekeys:
    if fnkey in s.keys():
        # Use metadata from first dict.  Should all be same
        filename += '_{}'.format(s[fnkey])
savedata(processeddata, filename + '.df')
t3 = time.time()

convert_to_uA(processeddata)
lines = plotiv(processeddata[::npulses], alpha=.7)
plt.title('{}_{}_{}_{}: {}, t={}nm, w={}nm'.format(s['dep_code'], s['sample_number'], s['module'], s['device'], s['layer_1'], s['thickness_1'], s['width_nm'], ))
plt.tight_layout()
plt.savefig(pjoin(datafolder, subfolder, filename))
plt.close()
nextdevice()

winsound.Beep(1000, 100)
