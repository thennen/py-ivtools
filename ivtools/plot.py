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

def plot_R_states(data, v0=.1, v1=None, **kwargs):
    resist_states = resistance_states(data, v0, v1)
    resist1 = resist_states[0]
    resist2 = resist_states[1]
    if type(resist1) is pd.Series:
        cycle1 = resist1.index
        cycle2 = resist2.index
    else:
        cycle1 = cycle2 = len(resist1)

    fig, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    scatterargs.update(kwargs)
    ax.scatter(cycle1, resist1, c='royalblue', **scatterargs)
    ax.scatter(cycle2, resist2,  c='seagreen', **scatterargs)
    #ax.legend(['HRS', 'LRS'], loc=0)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Resistance / $\\Omega$')



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

def write_frames(data, directory, splitbranch=True, shadow=True, extent=None, startloopnum=0):
    '''
    Write set of ivloops to disk to make a movie which shows their evolution nicely
    I rewrote this ten times before decided to make it a function
    probably there's a better version in ipython history
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    if shadow:
        # Plot them all on top of each other transparently for reference
        plotiv(data, color='gray', linewidth=.5, alpha=.05, ax=ax)
    #colors = plt.cm.rainbow(arange(len(data))/len(data))
    colors = ['black'] * len(data)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    if type(data) is pd.DataFrame:
        thingtoloop = data.iterrows()
    else:
        thingtoloop = enumerate(data)
    for i,l in thingtoloop:
        if splitbranch:
            # Split branches
            plotiv(increasing(l, sort=True), ax=ax, color='C0', label='>>')
            plotiv(decreasing(l, sort=True), ax=ax, color='C2', label='<<')
            legend(title='Sweep Direction')
        else:
            # Colors will mess up if you pass a dataframe with non range(0, ..) index
            plotiv(l, ax=ax, color=colors[i])
        title('Loop {}'.format(i+startloopnum))
        plt.savefig(os.path.join(directory, 'Loop_{:03d}'.format(i)))
        del ax.lines[-1]
        del ax.lines[-1]

def frames_to_mp4(directory, fps=10, prefix='Loop', outname='out'):
    # Send command to create video with ffmpeg
    # TODO: have it recognize the file prefix
    # Don't know difference between -framerate and -r options, but it
    # seems both need to be set to the desired fps.  Even the order matters.  Don't change it.

    cmd = (r'cd "{0}" & ffmpeg -framerate {1} -i {3}_%03d.png -c:v libx264 '
            '-r {2} -pix_fmt yuv420p -crf 18 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{4}.mp4').format(directory, fps, fps+5, prefix, outname)
    os.system(cmd)

def mpfunc(x, pos):
    #longnames = ['exa', 'peta', 'tera', 'giga', 'mega', 'kilo', '', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto']
    prefix = ['E', 'P', 'T', 'G', 'M', 'k', '', 'm', '$\mu$', 'n', 'p', 'f', 'a']
    values = [1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18]
    if abs(x) < min(values):
        return '{:1.1f}'.format(x)
    for v, p in zip(values, prefix):
        if abs(x) >= v:
            return '{:1.1f}{}'.format(x/v, p)

metricprefixformatter = mpl.ticker.FuncFormatter(mpfunc)
