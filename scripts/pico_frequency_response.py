# measure phase shift and fit for constant delay
d = freq_response(ch=['A', 'C'], fstart=1e3, fend=1e8, n=30, amp=1, trigsource='A')
sinefit = fit_sine(d, ['A', 'C'], guess_ncycles=100)
sinefit = pd.DataFrame(sinefit)
sinefit.loc[27:29, 'C_phase'] -= 2*pi
dtheta = sinefit['C_phase'] - sinefit['A_phase']
freqs = sinefit['A_freq']
from scipy.optimize import curve_fit
def phaseshift(freqs, dt):
    return -dt * 2 * pi * freqs
(dthetafit,), _ = curve_fit(phaseshift, freqs, dtheta, 15e-9)
fig, ax = plt.subplots()
ax.semilogx(freqs, dtheta, '.')
ax.semilogx(freqs, phaseshift(freqs, dthetafit), alpha=.2, color='red', linestyle='--')
ax.legend(['Measured', 'Constant delay {:.2f} ns'.format(dthetafit/1e-9)])
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Phase shift [radians]')


# Measure gain and phase shift
d = freq_response(ch=['A', 'C'], fstart=1e3, fend=1e8, n=30, amp=1, trigsource='A')
sinefit = fit_sine(d, ['A', 'C'], guess_ncycles=100)
sinefit = pd.DataFrame(sinefit)
sinefit.loc[27:29, 'C_phase'] -= 2*pi
dtheta = sinefit['C_phase'] - sinefit['A_phase']
freqs = sinefit['A_freq']
from scipy.optimize import curve_fit
def phaseshift(freqs, dt):
    return -dt * 2 * pi * freqs
(dthetafit,), _ = curve_fit(phaseshift, freqs, dtheta, 15e-9)
fig, ax = plt.subplots()
ax.semilogx(freqs, dtheta, '.')
ax.semilogx(freqs, phaseshift(freqs, dthetafit), alpha=.2, color='red', linestyle='--')
ax.legend(['Measured', 'Constant delay {:.2f} ns'.format(dthetafit/1e-9)])
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Phase shift [radians]')
