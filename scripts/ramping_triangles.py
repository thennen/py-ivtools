# This script tests a bunch of devices using long pulsetrains
# Need to load the waveforms via usb drive because Rigol is retarded

# Split rigol channel A into picoscope channel A (1 Mohm)
# put Rehan amplifier to pico C (x10) D (x200)

# Run the interactive.py script first.
import winsound

rigol = instruments.RigolDG5000()
ps = instruments.Picoscope()

### Create the waveforms that we will use

# I want to try using sine pulses because no one on the planet has ever tried that wild and crazy idea on reram
# This will eliminate sharp edges with crazy frequency components

'''
plt.figure()
plt.plot(pulse)
'''

###
### Settings !
###
# Maximum amplitude of pulse train
train_amp = 6
# Starting amplitude as a fraction of ending amplitude
amp_start = .1
#trainsamples = 2**24 # 16 MS
trainsamples = 2**22 # 4 MS
npulses = 1024 # you should pick something that divides trainsamples
pulsesamples = int(trainsamples / npulses)
pulseduration = 1e-4
trainduration = pulseduration * npulses
# Picoscope channels to measure
channels = ['A', 'C', 'D']
nchannels = len(channels)
# Desired picoscope sampling frequency
# fs = np.min((5e9/nchannels, 512e6 / trainduration)) # Tries to get the maximum number of samples
fs = 1e8
# Picoscope
ps.coupling.a = 'DC'
ps.coupling.c = 'DC50'
ps.coupling.d = 'DC50'
# Ranges change during the measurement (see script)
###
###
###

# This is what the pulses look like
pulse = (1 + np.cos(np.linspace(-pi, pi, pulsesamples))) / 2

# Positive polarity
pulsematrix = np.tile(pulse, npulses).reshape([npulses, len(pulse)])
# Modulate the amplitude, not continuously but per pulse
pos_train = (pulsematrix.T * np.linspace(amp_start, 1, npulses)).T.flatten()
# Negative polarity
neg_train = -pos_train
# Both polarities
# This one has twice the frequency
#bipolar_train = ((pulsematrix.T - .5)*2 * np.linspace(.1, 1, npulses)).T.flatten()
# This one is kind of strange, only 512 pulses per polarity.  Whatever I'll use it.
bipolar_train = (pulsematrix.T * np.linspace(.1, 1, npulses) * np.sign(np.cos(np.arange(0, pi*npulses, pi)))).T.flatten()

'''
plt.figure()
x = np.arange(len(pos_train))
plt.step(x, pos_train)
plt.step(x, neg_train)
plt.step(x, bipolar_train)
'''

# Write the waveforms to usb drive.
ans = input('Write waveforms to usb drive F:? (y/n)')
if ans == 'y':
    rigol.write_wfm_file(pos_train, 'F:/ptrain.RAF')
    rigol.write_wfm_file(neg_train, 'F:/ntrain.RAF')
    rigol.write_wfm_file(bipolar_train, 'F:/btrain.RAF')

    input('Move the USB drive to rigol!')

# Load metadata
modules = [f'014{l}' for l in ['B', 'C', 'F', 'G', 'I', 'E']]
meta.load_lassen(dep_code='Pferd', sample_number=1, die_rel=1, module=modules, device=[5,6,8,9])

meta.static['polarity'] = '+left'

############################################## Script
#for i,m in meta.df.iterrows():
# TODO Define what to do to each device, so we just keep hitting enter and moving the probes until done
# input('Press Enter after you moved the probes to the next sample')

# Set up waveform
rigol.setup_burstmode(n=1)
print('Loading ptrain waveform..')
rigol.load_wfm_usbdrive('ptrain.RAF')
# Amplitude is peak-to-peak because rigol is chinese or whatever
rigol.amplitude(train_amp * 2)
rigol.period(trainduration)
rigol.output(True)

rangeA, offsetA = ps.best_range([0, train_amp])
ps.range.a = rangeA
ps.offset.a = offsetA
# Currents should be positive
# For normal Rehan setup, x10 channel gives 1V / 2mA, x100 gives 1V / 175 uA
ps.range.c = .2
ps.offset.c = -.19
ps.range.d = .2
ps.offset.d = -.19

ps.capture(channels, freq=fs, duration=trainduration, pretrig=0)
rigol.trigger()
d = ps.get_data(channels)
# Resample and save the input waveform.  There could be a time offset.
d['wfm'] = np.interp(np.linspace(0,1,len(d['A'])), np.linspace(0,1,len(pos_train)), pos_train)
# save the raw data
savedata(d)
d = rehan_to_iv(d)
splitd = splitiv(d, npulses)
smoothd = smoothimate(d, 50, 50)
ssd = splitiv(smoothd, npulses)

winsound.Beep(500, 300)
meta.next()
# Plot something?
plotiv(ssd[::10])
