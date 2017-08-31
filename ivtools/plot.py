""" Functions for making plots with IV data """

import matplotlib.pyplot as plt
import numpy as np
from dotdict import dotdict
import pandas as pd

def _plot_single_iv(iv, ax=None, x='V', y='I', maxsamples=100000, **kwargs):
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

    if dtype in (np.ndarray, list, pd.DataFrame):
        # There are several loops
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

        if dtype == pd.DataFrame:
            if x is None or hasattr(data.iloc[0][x], '__iter__'):
                # Plot x array vs y array.  x can be none, then it will just be data point number
                line = []
                for (row, iv), c in zip(data.iterrows(), colors):
                    kwargs.update(c=c)
                    line.append(_plot_single_iv(iv, ax=ax, x=x, y=y, maxsamples=maxsamples, **kwargs))
            else:
                line = plot(data[x], data[y], **kwargs)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
        else:
            if x is None or hasattr(data[0][x], '__iter__'):
                line = []
                for iv, c in zip(data, colors):
                    kwargs.update(c=c)
                    line.append(_plot_single_iv(iv, ax=ax, x=x, y=y, maxsamples=maxsamples, **kwargs))
            else:
                # Probably referencing scalar values.
                # No tests to make sure both x and y scalar values for all loops.
                X = [d[x] for d in data]
                Y = [d[y] for d in data]
                line = ax.plot(X, Y, **kwargs)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
    elif dtype in (dict, dotdict, pd.Series):
        # Just one IV
        line = _plot_single_iv(data, ax, x=x, y=y, maxsamples=maxsamples, **kwargs)
    else:
        print('plotiv did not understand the input datatype {}'.format(dtype))
        return

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
            if chdata[c].dtype == np.int8:
                # Convert to voltage for plot
                chplotdata = chdata[c] / 2**8 * chdata['RANGE'][c] * 2 - chdata['OFFSET'][c]
            else:
                chplotdata = chdata[c]
            ax.plot(chplotdata, color=colors[c], label=c)
            # lightly indicate the channel range
            choffset = chdata['OFFSET'][c]
            chrange = chdata['RANGE'][c]
            ax.fill_between((0, len(chdata[c])), -choffset - chrange, -choffset + chrange, alpha=0.05, color=colors[c])
    ax.legend(title='Channel')
    ax.set_xlabel('Data Point')
    ax.set_ylabel('Voltage [V]')

def interactive_figures(n=2):
    # Determine nice place to put some plots, and make the figures
    # Need to get monitor information
    # Only works in windows ...
    import ctypes
    user32 = ctypes.windll.user32
    wpixels, hpixels = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    #aspect = hpixels / wpixels
    dc = user32.GetDC(0)
    LOGPIXELSX = 88
    LOGPIXELSY = 90
    hdpi = ctypes.windll.gdi32.GetDeviceCaps(dc, LOGPIXELSX)
    vdpi = ctypes.windll.gdi32.GetDeviceCaps(dc, LOGPIXELSY)
    ctypes.windll.user32.ReleaseDC(0, dc)
    bordertop = 79
    borderleft = 7
    borderbottom = 28
    taskbar = 40
    figheight = (hpixels - bordertop*2 - borderbottom*2 - taskbar) / 2
    # Nope
    #figwidth = wpixels * .3
    #figwidth = 500
    figwidth = figheight * 1.3
    figsize = (figwidth / hdpi, figheight / vdpi)
    fig1loc = (wpixels - figwidth - 2*borderleft, 0)
    fig2loc = (wpixels - figwidth - 2*borderleft, figheight + bordertop + borderbottom)

    fig1, ax1 = plt.subplots(figsize=figsize, dpi=hdpi)
    fig1.canvas.manager.window.move(*fig1loc)
    fig2, ax2 = plt.subplots(figsize=figsize, dpi=hdpi)
    fig2.canvas.manager.window.move(*fig2loc)

    fig1.set_tight_layout(True)
    fig2.set_tight_layout(True)

    plt.show()

    return (fig1, ax1), (fig2, ax2)
