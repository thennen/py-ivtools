""" Functions for making plots with IV data """

import matplotlib.pyplot as plt
import numpy as np
from dotdict import dotdict

def _plot_single_iv(iv, ax=None, x='V', y='I', maxsamples=10000, **kwargs):
    ''' Plot an array vs another array contained in iv object '''
    if ax is None:
        fig, ax = plt.subplots()

    if type(y) == str:
        Y = iv[y]
    else:
        Y = y
    l = len(Y)

    if x is None:
        X = np.arange(l)
    elif type(x) == str:
        X = iv[x]
    else:
        X = x
    if maxsamples is not None and maxsamples < l:
        # Down sample data
        step = int(l/maxsamples)
        X = X[np.arange(0, l, step)]
        Y = Y[np.arange(0, l, step)]

    # Try to name the axes according to metadata
    # Will error right now if you pass array as x or y
    if x == 'V': longnamex = 'Voltage'
    elif x is None:
        longnamex = 'Data Point'
    elif type(x) == str:
        longnamex = x
    if y == 'I': longnamey = 'Current'
    else: longnamey = y
    if 'longnames' in iv.keys():
        if x in iv['longnames'].keys():
            longnamex = iv['longnames'][x]
        if y in iv['longnames'].keys():
            longnamey = iv['longnames'][y]
    if x is None: unitx = '#'
    else: unitx = '?'
    unity = '?'
    if 'units' in iv.keys():
        if x in iv['units'].keys():
            unitx = iv['units'][x]
        if y in iv['units'].keys():
            unity = iv['units'][y]

    ax.set_xlabel('{} [{}]'.format(longnamex, unitx))
    ax.set_ylabel('{} [{}]'.format(longnamey, unity))

    return ax.plot(X, Y, **kwargs)[0]

def plotiv(data, x='V', y='I', ax=None, maxsamples=10000, cm='jet', **kwargs):
    '''
    IV loop plotting which can handle single or multiple loops.
    maxsamples : downsample to this number of data points if necessary
    kwargs passed through to ax.plot
    New figure is created if ax=None

    Maybe pass an arbitrary plotting function
    '''
    if ax is None:
        fig, ax = plt.subplots()

    dtype = type(data)
    if dtype == np.ndarray:
        # There are many loops
        if x is None or hasattr(data[0][x], '__iter__'):
            line = []
            # Pick colors
            # you can pass a list of colors, or a colormap
            if isinstance(cm, list):
                colors = cm
            else:
                if isinstance(cm, str):
                    # Str refers to the name of a colormap
                    cmap = plt.cm.get_cmap(cm)
                elif isinstance(cm, mpl.colors.LinearSegmentedColormap):
                    cmap = cm
                colors = [cmap(c) for c in np.linspace(0, 1, len(data))]
            for iv, c in zip(data, colors):
                kwargs.update(c=c)
                line.append(_plot_single_iv(iv, ax=ax, x=x, y=y, maxsamples=maxsamples, **kwargs))
        else:
            # Probably referencing scalar values.
            # No tests to make sure both x and y scalar values for all loops.
            X = extract(data, x)
            Y = extract(data, y)
            line = ax.plot(X, Y, **kwargs)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
    elif dtype == dotdict:
        line = _plot_single_iv(data, ax, x=x, y=y, maxsamples=maxsamples, **kwargs)
    else:
        print('plotiv did not understand the input datatype {}'.format(dtype))

    return ax, line


def plot_channels(chdata, ax=None):
    '''
    Plot the channel data of picoscope
    '''
    if ax is None:
        fig, ax = plt.subplots()
    # Colors match the code on the picoscope
    colors = dict(A='Blue', B='Red', C='Green', D='Yellow')
    channels = ['A', 'B', 'C', 'D']
    # Remove the previous range indicators
    ax.collections = []
    for c in channels:
        if c in chdata.keys():
            ax.plot(chdata[c], color=colors[c], label=c)
            # lightly indicate the channel range
            choffset = chdata['OFFSET'][c]
            chrange = chdata['RANGE'][c]
            ax.fill_between((0, len(chdata[c])), -choffset - chrange, -choffset + chrange, alpha=0.05, color=colors[c])
    ax.legend(title='Channel')
