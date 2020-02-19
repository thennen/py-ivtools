# Tested on 1kohm resistor
# this does a current sweep in the positive direction, followed by a voltage sweep in the negative direction, all in 1 ms
# using compliance circuit 
dur = 1e-3

# To synchronize rigol channels, you need an external trigger split into the two sync ports on the back.
# This is provided by a second rigol here
rigol2 = instruments.RigolDG5000('USB0::0x1AB1::0x0640::DG5T182500117::INSTR')

def I_to_Vc(I):
    #made up calibration for current source
    return I*1960 - 9.6 +0.55

t = np.linspace(0, 2*np.pi, 2**10)
# current source during voltage sweep (should be high enough to bring the collector
# current up and therefore emitter resistance down)
Iidle = I_to_Vc(800e-6)
# voltage source during current sweep (needs to be enough to supply all the currents)
Vidle = 2
iwfm = I_to_Vc(500e-6*np.sin(t)**2)
iwfm[2**9:] = Iidle
vwfm = -np.sin(t)**2
vwfm[:2**9] = Vidle
vwfm[0] = vwfm[-1] = 0
iwfm[0] = iwfm[-1] = Iidle
'''
plt.figure()
plt.plot(iwfm)
plt.plot(vwfm)
'''

rigol.load_volatile_wfm(vwfm, dur, ch=1)
rigol.load_volatile_wfm(iwfm, dur, ch=2)
rigol.setup_burstmode(n=1, trigsource='EXT', ch=1)
rigol.setup_burstmode(n=1, trigsource='EXT', ch=2)
# This delay seems to be necessary the first time you set up burst mode?
time.sleep(1)

ps.capture(['A', 'B', 'C'], 1e8, dur, timeout_ms=3000, pretrig=0)
# trigger everything
rigol2.pulse_builtin('SQU', duration=1e-6, amp=5, offset=5)
d = ps.get_data(['A', 'B', 'C'])
d = settings.pico_to_iv(d)
plot_channels(d)
savedata(d)

d['t'] = maketimearray(d)
sd = smoothimate(d, 100)
# TODO cut out the crossover distortion
plotiv(sd, 'Vd', 'I')

plotiv(slicebyvalue(sd, 't', dur*.0015, dur*(0.5-0.001)), 'Vd', 'I')
plotiv(slicebyvalue(sd, 't', dur*(0.5 + 0.001)), 'Vd', 'I', ax=plt.gca())
