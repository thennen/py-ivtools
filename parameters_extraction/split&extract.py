import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ivtools.plot import plotiv, colorbar_manual, engformatter
from ivtools.measure import raw_to_V
from ivtools.analyze import splitiv, smoothimate

from ivtools.analyze import *
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.signal import find_peaks

def ccircuit_to_iv(datain, dtype=np.float32):
    '''
    Convert picoscope channel data to IV dict
    For the newer version of compliance circuit, which compensates itself and amplifies the needle voltage
    chA should monitor the applied voltage
    chB (optional) should be the needle voltage
    chC should be the amplified current
    '''
    # Keep all original data from picoscope
    # Make I, V arrays and store the parameters used to make them

    dataout = datain
    # If data is raw, convert it here
    if datain['A'].dtype == np.int8:
        datain = raw_to_V(datain, dtype=dtype)
    A = datain['A']
    C = datain['C']
    #C = datain['C']
    gain = 1930
    dataout['V'] = dtype(A)
    #dataout['V_formula'] = 'CHA - IO'
    #dataout['I'] = 1e3 * (B - C) / R
    if 'B' in datain:
        B = datain['B']
        dataout['Vneedle'] = dtype(B)
        dataout['Vd'] = dataout['V'] - dataout['Vneedle'] # Does not take phase shift into account!
    dataout['I'] = dtype(C / gain)
    #dataout['I_formula'] = '- CHB / (Rout_conv * gain_conv) + CC_conv'
    units = {'V':'V', 'Vd':'V', 'Vneedle':'V', 'I':'A'}
    dataout['units'] = {k:v for k,v in units.items() if k in dataout}
    #dataout['units'] = {'V':'V', 'I':'$\mu$A'}
    # parameters for conversion
    #dataout['Rout_conv'] = R
    dataout['gain'] = gain
    return dataout

print('Loading data')
d = pd.read_pickle('parameters_extraction/example_data.s')
d = ccircuit_to_iv(d)
#d['I'] *= 1e6
d['V'] += .01 # remove offset

# First and last loops were a bit different
# still, 95,000 ≈ 100,000
for i in ('A', 'C', 'V', 'I'):
    d[i] = d[i][4000:99000]


print('Spliting loops')
#timestamp('loaded data')
nloops = 100000 # Number of loops
npts = len(d['V']) # Number of points in total
nppl = npts / nloops # Number of points per loop


# Concatenate the IV data before smoothing
Vconc = d['V']
Iconc = d['I']


# smooth loops
window = 5
Vconc_smooth = smooth(Vconc, window)
Iconc_smooth = smooth(Iconc, window)


# split loops at extrema
# could otherwise assume where the loops should be split, but won't be that robust for new datasets
stride = 100 # Separation to calculate dV
dV = Vconc_smooth[:-stride] - Vconc_smooth[stride:]
icross = np.where(np.diff(np.int8(dV > 0)))[0] + stride//2 # Indicates where dV crosses zero
hys = nppl/2 * 0.9 # If separation between loops is lower than 'hys' it's considered an error
icross = icross[np.insert(np.diff(icross) > hys, 0, True)]
if np.mod(len(icross),2) == 0:
    icross = np.delete(icross, -1) # To have the same number of ups and downs
ssd = [dict(I=Iconc_smooth[i:j+1], V=Vconc_smooth[i:j+1]) for i,j in zip(icross[0:-1], icross[1:])]


print('Extracting parameters')
# EVERYTHING IS ASSUMING A CERTAIN I, V ORIENTATION W.R.T SET/RESET

SET = ssd[0::2]
RESET = ssd[1::2]


################## Reset parameters #########################

# finding these sucks because there are a variable number of peaks.
# for now, only worry about the first one..

def find_reset_peaks(curve, nmax=3, promthresh=5e-6):
    # returns array(V,I) of up to nmax most prominent peaks
    #ipeaks, peakparams  = find_peaks(-I, prominence=5e-6)
    I = curve['I']
    V = curve['V']
    ipeaks, peakparams  = find_peaks(-I, prominence=0)

    prom = peakparams['prominences']
    # don't consider peaks with V > 0
    mask = V[ipeaks] < 0
    ipeaks = ipeaks[mask]
    prom = prom[mask]
    promenough = np.where(prom >= promthresh)[0]
    if not any(ipeaks):
        # unlikely, but still have to handle it
        #return np.empty(0, dtype=np.int64)
        return np.empty(2) * np.nan
    else:
        if len(promenough) > 0:
            # return peaks in the same order as the input array
            #return ipeaks[promenough[:nmax]]
            i = ipeaks[promenough[0]]
        else:
            # return the most prominent one even if it is below threshold
            #mostprom = np.argsort(prom)[::-1]
            i = ipeaks[np.argmax(prom)]
            #return np.array((ipeaks[mostprom],))
        return np.array((V[i], I[i]))

# pretty fast actually
IVreset_arr = np.vstack([find_reset_peaks(r, 3, 5e-6) for r in RESET])
IVreset = pd.DataFrame(IVreset_arr, columns=['Vreset', 'Ireset'])


################## Set parameters #########################

# simple thresholding to half the compliance level
# do something more sophisticated later if you want
#iset = [np.where(s['I'] >= Ithresh)[0][:1] for s in SET]
def setpoint(curve, Ithresh, lag=1):
    # lag = 1 gives the datapoint before threshold crossing
    i = np.where(curve['I'] >= Ithresh)[0]
    if any(i):
        i0 = i[0] - lag
        return np.array((curve['V'][i0], curve['I'][i0]))
    else:
        return np.empty(2) * np.nan

# How about detecting steps in the unfiltered data? will it be too digitized?
#SETraw = sd[0::2]

I25 = d['CC'] / 4 #25e-6
IVset25_arr = np.vstack([setpoint(s, I25, 1) for s in SET])
IVset25 = pd.DataFrame(IVset25_arr, columns=['Vset25', 'Iset25'])

# smoothed curves at half compliance
I50 = d['CC'] / 2
IVset50_arr = np.vstack([setpoint(s, I50, 1) for s in SET])
IVset50 = pd.DataFrame(IVset50_arr, columns=['Vset50', 'Iset50'])

I50 = d['CC'] / 2
IVset50lag_arr = np.vstack([setpoint(s, I50, window//2) for s in SET])
IVset50lag = pd.DataFrame(IVset50lag_arr, columns=['Vset50lag', 'Iset50lag'])


'''
n=100
n = slice(0, -1, 4000)
plotiv(SET[n], marker='.')
plotiv(SETraw[n], marker='.', hold=1)
plt.scatter(*IVset50_arr[n].T, marker='x', s=50, color='red')
plt.scatter(*IVset25_arr[n].T, marker='x', s=50, color='pink')
'''

################## polynomial fits to resistance states #########################
# define fit ranges (I and V?)

def fitpoly(curve, order, V0=None, V1=None, I0=None, I1=None):
    # much much faster if you downsample the curves a bit
    I = curve['I'][::window//2]
    V = curve['V'][::window//2]
    mask = np.ones(len(V), dtype=bool)
    if V0 is not None:
        mask &= V >= V0
    if V1 is not None:
        mask &= V <= V1
    if I0 is not None:
        mask &= I >= I0
    if I1 is not None:
        mask &= I <= I1

    if sum(mask) > order:
        Ifit = I[mask]
        Vfit = V[mask]
        pf = np.polyfit(Vfit, Ifit, order)
    else:
        # Don't fit if "poorly conditioned"
        pf = [np.nan] * (order + 1)
    return pf

# polyfit for HRS -- fit up until set starts
polyuporder = 5
polyup = np.vstack(fitpoly(s, polyuporder, -1.5, v, -80e-6, 25e-6) for s,(v,i) in zip(SET, IVset50lag_arr))
# polyfit for LRS -- fut down until reset starts
polydownorder = 3
polydown = np.vstack(fitpoly(r, polydownorder, v, 0.7, -120e-6, 80e-6) for r,(v,i) in zip(RESET, IVreset_arr))

# convert result to dataframe
polyup_colnames = [f'polyup{polyuporder - n}' for n in range(polyuporder + 1)]
polydown_colnames = [f'polydown{polydownorder - n}' for n in range(polydownorder + 1)]
polys = pd.DataFrame(np.hstack((polyup, polydown)), columns=polyup_colnames + polydown_colnames)

'''
# values of the polynomials
polyvalup = [dict(V=v, I=np.polyval(p, v)) for p in polyup]
polyvaldown = [dict(V=v, I=np.polyval(p, v)) for p in polydown]

step = 1000
plotiv(SET[::step])
v = np.linspace(-1.5, 1.5, 50)
plotiv(polyvalup[::step], alpha=.5, hold=1)

step = 1000
plotiv(RESET[::step])
v = np.linspace(-1.5, 1.5, 50)
plotiv(polyvaldown[::step], alpha=.5, hold=1)
'''

# make dataframe of all the parameters I want to export
datasets = (IVset25, IVset50, IVset50lag, IVreset, polys)
df = pd.concat(datasets, 1)

df.to_pickle('Lamprey_params.df')



