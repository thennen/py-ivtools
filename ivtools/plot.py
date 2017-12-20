""" Functions for making plots with IV data """

import matplotlib.pyplot as plt
import numpy as np
from dotdict import dotdict
import pandas as pd
from matplotlib.widgets import SpanSelector

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


def plotiv(data, x='V', y='I', c=None, ax=None, maxsamples=10000, cm='jet', **kwargs):
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
            if c is None:
                colors = [cmap(c) for c in np.linspace(0, 1, len(data))]
            elif type(c) is str:
                # color by the column given
                normc = (data[c] - np.min(data[c])) / (np.max(data[c]) - np.min(data[c]))
                colors = cmap(normc)
            else:
                # It's probably an array of values?  Map them to colors
                normc = (c - np.min(c)) / (np.max(c) - np.min(c))
                colors = cmap(normc)

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


def auto_title(data, keys=None, ax=None):
    '''
    Label an axis to identify the device.
    Quickly written, needs improvement
    '''
    if ax is None:
        ax = plt.gca()
    if keys is None:
        if type(data) is pd.DataFrame:
            meta = data.iloc[0]
        else:
            meta = data
        id = '{}_{}_{}_{}'.format(*list(meta[['dep_code','sample_number','module','device']]))
        layer = meta['layer_1']
        thickness = meta['thickness_1']
        width = meta['width_nm']
        title = '{}, {}, t={}nm, w={}nm'.format(id, layer, thickness, width)
    else:
        title = ', '.join(['{}:{}'.format(k, data[k]) for k in keys if k in data])

    ax.set_title(title)


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


def plot_span(data=None, ax=None, plotfunc=plotiv, **kwargs):
    '''
    Select index range from plot of some parameter vs index.  Plot the loops there.
    To use selector, just make sure you don't have any other gui widgets active
    Will remain active as long as the return value is not garbage collected
    Ipython keeps a reference to all outputs, so this will stay open forever if you don't assign it a value
    # TODO pass the plotting function (could be different than plotiv)
    '''
    if data is None:
        # Check for global variables ...
        # Sorry if this offends you ..
        print('No data passed. Looking for global variable d')
        try:
            data = d
        except:
            print('No global variable d. Looking for global variable df')
            try:
                data = df
            except:
                raise Exception('No data can be found')

    if ax is None:
        ax = plt.gca()
    def onselect(xmin, xmax):
        # Plot max 1000 loops
        xmin = int(xmin)
        xmax = int(xmax)
        n = xmax - xmin
        step = max(1, int(n / 1000))
        print('Plotting loops {}:{}:{}'.format(xmin, xmax+1, step))
        plotfunc(data[xmin:xmax+1:step], **kwargs)
        plt.show()
    rectprops = dict(facecolor='blue', alpha=0.3)
    return SpanSelector(ax, onselect, 'horizontal', useblit=True, rectprops=rectprops)


def paramplot(df, y, x, parameters, yerr=None, cmap=plt.cm.gnuplot, labelformatter=None,
              sparseticks=True, xlog=False, ylog=False, sortparams=False, paramvals=None, **kwargs):
    '''
    Plot y vs x for any number of parameters
    Can choose a subset of the parameter values to plot, and the colors will be the same as if the
    subset was not passed.  does that make any sense? sorry.
    '''
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if type(parameters) == str:
        parameters = [parameters]
    grp = df.groupby(parameters, sort=sortparams)
    ngrps = len(grp)
    colors = cmap(np.linspace(.1, .9, ngrps))
    colordict = {k:c for k,c in zip(grp.groups.keys(), colors)}
    for k, g in grp:
        if paramvals is None or k in paramvals:
            # Figure out how to label the lines
            if type(k) == tuple:
                if labelformatter is not None:
                    label = labelformatter.format(*k)
                else:
                    label = ', '.join(map(str, k))
            else:
                if labelformatter is not None:
                    label = labelformatter.format(k)
                else:
                    label = str(k)
            plotkwargs = dict(color=colordict[k],
                            marker='.',
                            label=label)
            plotkwargs.update(kwargs)
            plotg = g.sort_values(by=x)
            plt.plot(plotg[x], plotg[y], **plotkwargs)
            if yerr is not None:
                plt.errorbar(plotg[x], plotg[y], plotg[yerr], color=colordict[k], label=None)
    if sparseticks:
        # Only label the values present
        ux = np.sort(df[x].unique())
        ax.xaxis.set_ticks(ux)
        ax.xaxis.set_ticklabels(ux)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.legend(loc=0, title=', '.join(parameters))
    return fig, ax


def plot_channels(chdata, ax=None):
    '''
    Plot the channel data of picoscope
    '''
    if ax is None:
        fig, ax = plt.subplots()
    # Colors match the code on the picoscope
    # Yellow is too hard to see
    colors = dict(A='Blue', B='Red', C='Green', D='Gold')
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

def colorbar_manual(vmin=0, vmax=1, cmap='jet', **kwargs):
    ''' Usually you need a "mappable" to create a colormap on a plot.  This function lets you create one manually. '''
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, **kwargs)
    return cb

def write_frames(data, directory, splitbranch=True, shadow=True, extent=None, startloopnum=0, title=None, axfunc=None, **kwargs):
    '''
    Write set of ivloops to disk to make a movie which shows their evolution nicely
    axfunc gets called on the axis every frame -- if you want to modify it in some way
    kwargs get passed through to plotiv, which can get passed through to plt.plot
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    if shadow:
        # Plot them all on top of each other transparently for reference
        plotiv(data, color='gray', linewidth=.5, alpha=.03, ax=ax)
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
            colorup = 'Red'
            colordown = 'DarkBlue'
            plotiv(increasing(l, sort=True), ax=ax, color=colorup, label=r'$\rightarrow$', **kwargs)
            plotiv(decreasing(l, sort=True), ax=ax, color=colordown, label=r'$\leftarrow$', **kwargs)
            ax.legend(title='Sweep Direction')
        else:
            # Colors will mess up if you pass a dataframe with non range(0, ..) index
            plotiv(l, ax=ax, color=colors[i], **kwargs)
        if title is None:
            ax.set_title('Loop {}'.format(i+startloopnum))
        else:
            # Make title from the key indicated in argument
            # Could make it a function and then call the function on the data
            # Yeah let's do that
            ax.set_title(title(l))
        if axfunc is not None:
            axfunc(ax)
        plt.savefig(os.path.join(directory, 'Loop_{:03d}'.format(i)))
        del ax.lines[-1]
        del ax.lines[-1]


def write_frames_2(data, directory, persist=5, framesperloop=50, extent=None):
    ''' Temporary name, make a movie showing iv loops as they are swept'''
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fig, ax = plt.subplots(figsize=(8,6))
    fig.set_tight_layout(True)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    #colors = plt.cm.rainbow(arange(len(data))/len(data))
    colors = ['black'] * len(data)
    if type(data) is pd.DataFrame:
        thingtoloop = data.iterrows()
    else:
        thingtoloop = enumerate(data)
    frame = 0
    for i,l in thingtoloop:
        npts = len(l['V'])
        step = max(1, int(npts / framesperloop))
        # empty plots to update with data
        # Assuming there is ONE loop, starting and ending at zero
        # Up to first extreme
        C1 = 'forestgreen'
        C2 = 'darkmagenta'
        line1 = ax.plot([], color=C1, alpha=1)[0]
        # Down to min
        line2 = ax.plot([], color=C2, alpha=1)[0]
        # Back up to zero
        line3 = ax.plot([], color=C1, alpha=1)[0]

        plt.title('Loop {}'.format(i))
        ax.set_xlabel('Applied Voltage [V]')
        ax.set_ylabel('Device Current [$\mu$A]')
        imax = np.argmax(l['V'])
        imin = np.argmin(l['V'])
        firstextreme = min(imax, imin)
        secondextreme = max(imax, imin)
        for endpt in np.int32(np.linspace(step, npts+1, framesperloop)):
            end1 = min(firstextreme, endpt)
            line1.set_data(l['V'][:end1], l['I'][:end1])
            end2 = min(secondextreme, endpt)
            line2.set_data(l['V'][firstextreme:end2], l['I'][firstextreme:end2])
            line3.set_data(l['V'][secondextreme:endpt], l['I'][secondextreme:endpt])
            plt.savefig(os.path.join(directory, 'Loop_{:03d}'.format(frame)))
            frame += 1
            # Reduce opacity of all previous loops, every frame (because cool)
            for part in ax.lines[:-3]:
                part.set_alpha(part.get_alpha() * 0.96)
                part.set_color('black')
                part.set_linewidth(1)
        # Suddenly reduce opacity of previous loop
        line1.set_alpha(.75)
        line2.set_alpha(.75)
        line3.set_alpha(.75)
        # I hope i starts at zero, it won't if you pass a dataframe slice
        if i > persist:
            # Remove the oldest loop
            del ax.lines[0:3]


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

# Use it like this: ax.yaxis.set_major_formatter(metricprefixformatter)
metricprefixformatter = mpl.ticker.FuncFormatter(mpfunc)
# Note I might be stupid and this could already be built in, using mpl.ticker.EngFormatter()

def plot_log_reference_lines(ax, slope=-2):
    ''' Put some reference lines on a log-log plot indicating a certain power dependence'''
    ylims = ax.get_ylim()
    ymin, ymax = ylims
    logymin, logymax = np.log10(ymin), np.log10(ymax)
    xlims = ax.get_xlim()
    xmin, xmax = xlims
    # Starting y points for the lines
    y = np.logspace(logymin, logymax + np.log10(ymax - ymin), 20)
    # Plot one at a time so you can just label one (for legend)
    for yi in y[:-1]:
        ax.plot(xlims, (yi, yi + yi/xmin**slope *(xmax**slope - xmin**slope)), '--', alpha=.2, color='black')
    # Label the last one
    ax.plot(xlims, (y[-1], y[-1] + y[-1]/xmin**slope *(xmax**slope - xmin**slope)), '--', alpha=.2, color='black', label='Area scaling')
    # Put the limits back
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)


def plot_power_lines(pvals=None, ax=None):
    '''
    Plot lines of constant power on the indicated axis  (should be I vs V)
    TODO: Label power values
    TODO: detect power range from axis range
    '''
    if ax is None:
        ax = plt.gca()

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    if pvals is None:
        # Determine power range from axis limits
        minp = x0 * y0
        maxp = x1 * y1
        if (ax.get_yscale() == 'log') or (ax.get_xscale() == 'log'):
            if minp < 0:
                # Sometimes scale of axis includes negative powers.  Data shouldn't.
                # Assuming here the voltage axis is the problem.  Might need to fix it later.
                minp = y0 * (x0 + 0.1 * (x1 - x0))
            pvals = np.logspace(np.log10(minp), np.log10(maxp), 10)[1:-1]
        else:
            pvals = np.linspace(minp, maxp, 10)[1:-1]

    # Easiest to space equally in x.  Could change this later so that high slope areas get enough data points.
    x = linspace(x0, x1, 1000)
    #pvals = linspace(pmin, pmax, nlines)
    ylist = [p/x for p in pvals]
    for y in ylist:
        ax.plot(x, y, '--', color='black', alpha=.3)
