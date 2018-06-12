""" Functions for making plots with IV data """

# Local imports
from . import analyze

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
from matplotlib.widgets import SpanSelector
from inspect import signature
import os

def _plot_single_iv(iv, ax=None, x='V', y='I', maxsamples=100000, xfunc=None, yfunc=None, **kwargs):
    '''
    Plot an array vs another array contained in iv object
    if None is passed for x, then plots vs range(len(x))
    '''
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
    if x == 'Vcalc': longnamex = 'Device Voltage'
    elif x is None:
        longnamex = 'Data Point'
    elif type(x) == str:
        longnamex = x
    if y == 'I': longnamey = 'Current'
    else: longnamey = y
    if ('longnames' in iv.keys()) and (type(iv['longnames']) == dict):
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

    xlabel = '{} [{}]'.format(longnamex, unitx)
    ylabel = '{} [{}]'.format(longnamey, unity)

    if xfunc is not None:
        # Apply a function to x array
        X = xfunc(X)
        xlabel = '{}({})'.format(xfunc.__name__, xlabel)
    if yfunc is not None:
        # Apply a function to y array
        Y = yfunc(Y)
        ylabel = '{}({})'.format(yfunc.__name__, ylabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax.plot(X, Y, **kwargs)[0]

def plotiv(data, x='V', y='I', c=None, ax=None, maxsamples=10000, cm='jet', xfunc=None, yfunc=None,
           plotfunc=_plot_single_iv, autotitle=False, **kwargs):
    '''
    IV loop plotting which can handle single or multiple loops.
    Can plot any column vs any other column
    Can assign a colormap to the lines, or use all one color using e.g. color='black'
    Can transform the x and y data by passing functions to xfunc and/or yfunc arguments
    if x and y specify a scalar values, then just one scatter plot is made
    if x is None, then y is plotted vs range(len(y))
    maxsamples : downsample to this number of data points if necessary
    kwargs passed through to ax.plot (can customize line properties this way)
    New figure is created if ax=None

    Can pass an arbitrary plotting function, which defaults to _plot_single_iv, haven't tested it yet
    Could then define some other plots that take IV data and reuse plotiv functionality.
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
                    line.append(plotfunc(iv, ax=ax, x=x, y=y, maxsamples=maxsamples, xfunc=xfunc, yfunc=yfunc, **kwargs))
            else:
                line = plt.plot(data[x], data[y], **kwargs)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
        else:
            if x is None or hasattr(data[0][x], '__iter__'):
                line = []
                for iv, c in zip(data, colors):
                    kwargs.update(c=c)
                    line.append(plotfunc(iv, ax=ax, x=x, y=y, maxsamples=maxsamples, xfunc=xfunc, yfunc=yfunc, **kwargs))
            else:
                # Probably referencing scalar values.
                # No tests to make sure both x and y scalar values for all loops.
                # Will break for dataframes right now.
                X = [d[x] for d in data]
                Y = [d[y] for d in data]
                line = ax.scatter(X, Y, **kwargs)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
    elif dtype in (dict, pd.Series):
        # Just one IV
        line = plotfunc(data, ax=ax, x=x, y=y, maxsamples=maxsamples, xfunc=xfunc, yfunc=yfunc, **kwargs)
    else:
        print('plotiv did not understand the input datatype {}'.format(dtype))
        return

    # Use EngFormatter if the plotted values are small or large
    xlims = np.array(ax.get_xlim())
    if any(xlims > 1e3) or all(xlims < 1e-1):
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ylims = np.array(ax.get_ylim())
    if any(ylims > 1e3) or all(ylims < 1e-1):
        ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

    if autotitle:
        auto_title(data, keys=None, ax=ax)

    # should I really return this?  usually I don't assign the values and then they get cached by ipython forever
    return ax, line

def auto_title(data, keys=None, ax=None):
    '''
    Label an axis to identify the device.
    Quickly written, needs improvement
    '''
    if ax is None:
        ax = plt.gca()

    if type(data) is pd.DataFrame:
        meta = data.iloc[0]
    else:
        meta = data

    def safeindex(data, key):
        if key in data:
            return data[key]
        else:
            return '?'

    if keys is None:
        # Default behavior
        idkeys = ['dep_code','sample_number','module','device']
        id = '_'.join([format(safeindex(meta, idk)) for idk in idkeys])

        otherkeys = ['layer_1', 'thickness_1', 'width_nm', 'R_series']
        othervalues = [safeindex(meta, k) for k in otherkeys]
        # use kohm if necessary
        if othervalues[3] != '?' and othervalues[3] >= 1000:
            othervalues[3] = str(int(othervalues[3]/1000)) + 'k'
        formatstr = '{}, {}, t={}nm, w={}nm, Rs={}$\Omega$'
        title = formatstr.format(id, *othervalues)
    else:
        title = ', '.join(['{}:{}'.format(k, safeindex(meta, k)) for k in keys])

    ax.set_title(title)

def plot_R_stajes(data, v0=.1, v1=None, **kwargs):
    resist_states = analyze.resistance_states(data, v0, v1)
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

def paramplot(df, y, x, parameters, yerr=None, cmap=plt.cm.gnuplot, labelformatter=None,
              sparseticks=True, xlog=False, ylog=False, sortparams=False, paramvals=None,
              ax=None, **kwargs):
    '''
    Plot y vs x for any number of parameters
    Can choose a subset of the parameter values to plot, and the colors will be the same as if the
    subset was not passed.  does that make any sense? sorry.
    '''
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
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
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return fig, ax

def plot_channels(chdata, ax=None):
    '''
    Plot the channel data of picoscope
    Includes an indication of the measurement range used
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
            if 'sample_rate' in chdata:
                # If sample rate is available, plot vs time
                x = analyze.maketimearray(chdata)
                ax.set_xlabel('Time [s]')
                ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
            else:
                x = range(len(chdata[c]))
                ax.set_xlabel('Data Point')
            ax.plot(x, chplotdata, color=colors[c], label=c)
            # lightly indicate the channel range
            choffset = chdata['OFFSET'][c]
            chrange = chdata['RANGE'][c]
            ax.fill_between((0, np.max(x)), -choffset - chrange, -choffset + chrange, alpha=0.05, color=colors[c])
    ax.legend(title='Channel')
    ax.set_ylabel('Voltage [V]')

def colorbar_manual(vmin=0, vmax=1, cmap='jet', ax=None, **kwargs):
    ''' Normally you need a "mappable" to create a colorbar on a plot.  This function lets you create one manually. '''
    if ax is None:
        ax = plt.gca()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, **kwargs)
    return cb


    # Fill the whole plot with lines.  Find points to go through
    if ax.get_yscale() == 'linear':
        yp = np.linspace(ymin, ymax , n)
    else:
        # Sorry if this errors.  Negative axis ranges are possible on a log plot.
        logymin, logymax = np.log10(ymin), np.log10(ymax)
        yp = np.logspace(logymin, logymax + np.log10(ymax - ymin), n)
    if ax.get_xscale() == 'linear':
        xp = np.linspace(xmin, xmax , n)
    else:
        logxmin, logxmax = np.log10(xmin), np.log10(xmax)
        xp = np.logspace(logxmin, logxmax + np.log10(xmax - xmin), n)

    # Load lines aren't lines on log scale, so plot many points
    x = np.linspace(xmin, xmax, 500)
    # Plot one at a time so you can just label one (for legend)
    slope = 1 / R / Iscale
    for xi,yi in zip(xp, yp):
        ax.plot(x, yi - slope * (x - xi), **plotargs)
    # Label the last one
    ax.lines[-1].set_label('{}$\Omega$ Load Line'.format(mpfunc(R, None)))
    # Put the limits back
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

def mypause(interval):
    ''' plt.pause calls plt.show, which steals focus on some systems.  Use this instead '''
    backend = plt.rcParams['backend']
    if backend in mpl.rcsetup.interactive_bk:
        figManager = mpl._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def plot_cumulative_dist(data, ax=None, **kwargs):
    ''' Because I always forget how to do it'''
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.sort(data), np.arange(len(data))/len(data), **kwargs)

# Could be a module, but I want it to keep its state when the code is reloaded
class interactive_figs(object):
    '''
    A class to manage the figures used for automatic plotting of IV data while it is measured.
    Right now we are limited to one axis per figure ...  could be extended.
    can have several plotting functions per axis though ..
    '''
    # TODO: save/load configurations to disk?
    def __init__(self, n=4, oldinstance=None):
        if oldinstance is None:
            # Find nice sizes and locations for the figures
            # Need to get monitor information. Only works in windows ...
            import ctypes
            user32 = ctypes.windll.user32
            wpixels, hpixels = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
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
            figwidth = figheight * 1.3
            self.hdpi = hdpi
            self.vdpi = vdpi
            self.figsize = (figwidth / hdpi, figheight / vdpi)
            self.figlocs = [(wpixels - figwidth - 2*borderleft, 0),
                    (wpixels - figwidth - 2*borderleft, figheight + bordertop + borderbottom),
                    (wpixels - 2*figwidth - 4*borderleft, 0),
                    (wpixels - 2*figwidth - 4*borderleft, figheight + bordertop + borderbottom),
                    (wpixels - 3*figwidth - 6*borderleft, 0),
                    (wpixels - 3*figwidth - 6*borderleft, figheight + bordertop + borderbottom)]
            self.figs = []
            self.axs = []
            for i in range(n):
                self.createfig(n=i)
            # To be implemented..
            #self.colorcycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
            self.plotters = []
        else:
            # This is to completely reload the class code but keep the same state
            self.__dict__ = oldinstance.__dict__

    def createfig(self, n):
        '''
        Create the nth figure and move it into position.
        Store the fig, ax objects in self.figs, self.axs
        '''
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.hdpi)
        fig.set_tight_layout(True)
        fig.canvas.set_window_title('Interactive Plot {}'.format(n))
        if len(self.figs) <= n:
            self.figs.extend([None] * (n - len(self.figs) + 1))
        if len(self.axs) <= n:
            self.axs.extend([None] * (n - len(self.axs) + 1))
        self.figs[n] = fig
        self.axs[n] = ax
        # Give attribute names to each figure and ax: self.fig1 self.ax1 etc.
        setattr(self, 'fig'+str(n), fig)
        setattr(self, 'ax'+str(n), ax)
        self.tile(n)
        plt.show()

    def tile(self, n=None):
        '''
        Move figures to their default positions
        '''
        if n is None:
            for fig, loc in zip(self.figs, self.figlocs):
                fig.canvas.manager.window.move(*loc)
        else:
            if (len(self.figs) > n) and (len(self.figlocs) > n):
                self.figs[n].canvas.manager.window.move(*self.figlocs[n])

    def del_plotters(self, axnum):
        ''' Delete the plotters for the specified axis '''
        self.plotters = [p for p in self.plotters if p[0] != axnum]

    def add_plotter(self, plotfunc, axnum, **kwargs):
        '''
        Assign a function which creates a plot to an axis.
        Function should take data (dictionary-like) as first argument and have an ax keyword
        '''
        self.plotters.append((axnum, plotfunc))

    def newline(self, data=None):
        ''' Update the plots with new data. '''
        for axnum, plotter in self.plotters:
            ax = self.axs[axnum]
            if data is None:
                ax.plot([])
            else:
                try:
                    plotter(data, ax=ax)
                    color = ax.lines[-1].get_color()
                    ax.set_xlabel(ax.get_xlabel(), color=color)
                    ax.set_ylabel(ax.get_ylabel(), color=color)
                except:
                    ax.plot([])
                    print('Plotter number {} failed!'.format(axnum))
                ax.get_figure().canvas.draw()
        mypause(0.05)

    def updateline(self, data):
        '''
        Update the data in the previous plots.
        '''
        # Since I can't know how to get to the actual plotted data without calling the plot
        # function again, this works by deleting the last line and plotting a new one with
        # the same colors
        # I am assuming for now that the plot functions each produce one line.
        for axnum, plotter in self.plotters:
            ax = self.axs[axnum]
            if any(ax.lines):
                color = ax.lines[-1].get_color()
                del ax.lines[-1]
            else:
                color = None
            argspec = inspect.getfullargspec(plotter)
            if (argspec.varkw is not None) or ('color' in argspec.kwonlyargs) or ('color' in argspec.args):
                # plotter won't error if we pass this keyword argument
                # it might even work ..
                try:
                    plotter(data, ax, color=color)
                except:
                    print('Plotter number {} failed!'.format(axnum))
            else:
                # Simply set the line color after plotting
                # could mess up the color cycle.
                try:
                    plotter(data, ax)
                    ax.lines[-1].set_color(color)
                except:
                    print('Plotter number {} failed!'.format(axnum))
            ax.get_figure().canvas.draw()
        mypause(0.05)

    def clear(self):
        ''' Clear all the axes '''
        for fig in self.figs:
            for ax in fig.axes:
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                title = ax.get_title()
                ax.cla()
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)
                ax.set_title(title)

    def write(self, directory):
        ''' Write the figures to disk. '''
        pass

    def show(self):
        ''' Bring all the interactive plots to the foreground. '''
        import win32gui
        import win32com.client
        shell = win32com.client.Dispatch('WScript.Shell')
        # This is really messed up, but if you don't send this key, windows will
        # Refuse to bring another window to the foreground.
        # Its effect is like pressing the alt key.
        shell.SendKeys('%')
        console_hwnd = win32gui.GetForegroundWindow()
        # This alone doesn't work in windows with qt5 backend.  Don't know why.
        for fig in self.figs:
            fig.show()
            fig.canvas.manager.window.activateWindow()
            fig.canvas.manager.window.raise_()
        windowtitles = [f.canvas.manager.get_window_title() for f in self.figs]
        def enum_callback(hwnd, *args):
            txt = win32gui.GetWindowText(hwnd)
            if txt in windowtitles:
                #win32gui.SetForegroundWindow(hwnd)
                win32gui.ShowWindow(hwnd, True)
        win32gui.EnumWindows(enum_callback, None)
        # Put console back in foreground
        win32gui.SetForegroundWindow(console_hwnd)

    def close(self):
        ''' Close all the figures and stop doing anything '''
        for fig in self.figs:
            plt.close(fig)
        # Delete references
        self.figs = []
        self.axs = []


### These are supposed to be for the live plotting
def plottertemplate(data, ax, **kwargs):
    '''
    Minimal template for defining a new plotter.
    Needs kwargs if you want live updating.
    '''
    ax.plot(data['x'], data['y'], **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def ivplotter(data, ax=None, maxloops=100, smooth=False, **kwargs):
    # Smooth data a bit and give it to plotiv
    # Make sure not too much data gets plotted, or it slows down the program a lot.
    # Would be better to smooth before splitting ...
    # kwargs gets passed through to plotiv, which passes them through to plt.plot
    if ax is None:
        fig, ax = plt.subplots()
    if smooth:
        data = analyze.moving_avg(data, window=10)
    if type(data) is list:
        nloops = len(data)
    else:
        nloops = 1
    if nloops > maxloops:
        print('You captured {} loops.  Only plotting {} loops'.format(nloops, maxloops))
        loopstep = int(nloops / 99)
        data = data[::loopstep]
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    plotiv(data, ax=ax, maxsamples=5000, **kwargs)

def chplotter(data, ax=None, **kwargs):
    # data might contain multiple loops because of splitting, but we want the unsplit arrays
    # To avoid pasting them back together again, there is a global variable called chdata
    if ax is None:
        fig, ax = plt.subplots()
    # Remove previous lines
    for l in ax.lines[::-1]: l.remove()
    # Plot at most 100000 datapoints of the waveform
    channels = [ch for ch in ['A', 'B', 'C', 'D'] if ch in data]
    if any(channels):
        lendata = len(data[channels[0]])
        if lendata > 100000:
            print('Captured waveform has {} pts.  Downsampling data.'.format(lendata))
            plotdata = analyze.sliceiv(data, step=10)
        else:
            plotdata = data
        plot_channels(plotdata, ax=ax)

def dVdIplotter(data, ax=None, **kwargs):
    ''' Plot dV/dI vs V'''
    if ax is None:
        fig, ax = plt.subplots()
    mask = np.abs(d['V']) > .01
    vmasked = d['V'][mask]
    imasked = d['I'][mask]
    dv = np.diff(vmasked)
    di = np.diff(imasked)
    ax.plot(vmasked[1:], dv/di, **kwargs)
    ax.set_yscale('log')
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('V/I [$\Omega$]')
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

# Keithley ones
def Rfitplotter(data, ax=None, **kwargs):
    ''' Plot a line of resistance fit'''
    if ax is None:
        fig, ax = plt.subplots()
    mask = abs(data['V']) <= .1
    if sum(mask) > 1:
        line = np.polyfit(data['V'][mask], data['I'][mask], 1)
    else:
        line = [np.nan, np.nan]
    # Plot line only to max V or max I
    R = 1 / line[0]
    vmin = max(min(data['V']), min(data['I'] * R))
    vmax = min(max(data['V']), max(data['I'] * R))
    # Do some points in case you put it on a log scale later
    fitv = np.linspace(1.1 * vmin, 1.1 * vmax, 10)
    fiti = np.polyval(line, fitv)
    plotkwargs = dict(color='black', alpha=.3, linestyle='--')
    plotkwargs.update(kwargs)
    ax.plot(fitv, 1e6 * fiti, **plotkwargs)
    return R

def complianceplotter(data, ax=None, **kwargs):
    # Plot a dotted line indicating compliance current
    pass

def vtplotter(data, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ''' This defines what gets plotted on ax2'''
    if 't' not in data:
        t = analyze.maketimearray(data)
    else:
        t = data['t']
    ax.plot(t, data['V'], **kwargs)
    #color = ax.lines[-1].get_color()
    #ax.set_ylabel('Voltage [V]', color=color)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

def itplotter(data, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if 't' not in data:
        t = analyze.maketimearray(data)
    else:
        t = data['t']
    ax.plot(t, data['I'], **kwargs)
    #color = ax.lines[-1].get_color()
    #ax.set_ylabel('Current [$\mu$A]', color=color)
    ax.set_ylabel('Current [A]')
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

def VoverIplotter(data, ax=None, **kwargs):
    ''' Plot V/I vs V, like GPIB control program'''
    if ax is None:
        fig, ax = plt.subplots()
    # Mask small currents, since V/I will blow up
    # There's definitely a better way.
    #if len(data['I'] > 0):
    #    max_current = np.max(np.abs(data['I']))
    #    mask = np.abs(data['I']) > .01 * max_current
    #else:
    #    mask = []
    mask = np.abs(data['V']) > .01
    V = data['V'][mask]

    if 'Vmeasured' in data:
        VoverI = data['Vmeasured'] / data['I']
    elif 'Imeasured' in data:
        VoverI = data['V'] / data['Imeasured']
    else:
        VoverI = data['V'] / data['I']

    VoverI = VoverI[mask]
    VoverI[VoverI <= 0] = np.nan

    ax.plot(V, VoverI, **kwargs)
    #color = ax.lines[-1].get_color()

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ax.set_xlabel('Voltage [V]')
    #ax.set_ylabel('V/I [$\Omega$]', color=color)
    ax.set_ylabel('V/I [$\Omega$]')

def vcalcplotter(data, ax=None, R=8197, **kwargs):
    '''
    Subtract internal series resistance voltage drop
    For Lassen R = 143, 2164, 8197, 12857
    '''
    if ax is None:
        fig, ax = plt.subplots()
    # wtf modifies the input data?  Shouldn't do that.
    if type(data) is list:
        for d in data:
            d['Vcalc'] = d['V'] - R * d['I']
    else:
        data['Vcalc'] = data['V'] - R * data['I']
    plotiv(data, ax=ax, x='Vcalc', **kwargs)
    ax.set_xlabel('V device (calculated assuming Rseries={}$\Omega$) [V]'.format(R))
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())


### Widgets
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


### Animation
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
        x = 'V'
        y = 'I'
        if 'x' in kwargs:
            x = kwargs['x']
        if 'y' in kwargs:
            y = kwargs['y']
        plotiv(data, color='gray', linewidth=.5, alpha=.03, x=x, y=y, ax=ax)
    #colors = plt.cm.rainbow(arange(len(data))/len(data))
    colors = ['black'] * len(data)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    if type(data) is pd.DataFrame:
        # reset index so that we don't skip file names
        thingtoloop = data.reset_index().iterrows()
    else:
        thingtoloop = enumerate(data)
    for i,l in thingtoloop:
        if splitbranch:
            # Split branches
            colorup = 'Red'
            colordown = 'DarkBlue'
            plotiv(analyze.increasing(l, sort=True), ax=ax, color=colorup, label=r'$\rightarrow$', **kwargs)
            plotiv(analyze.decreasing(l, sort=True), ax=ax, color=colordown, label=r'$\leftarrow$', **kwargs)
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
            # Can do an operation on the axis, or on whatever you want.
            # First argument will be the axis, second argument the loop being plotted
            sig = signature(axfunc)
            nparams = len(sig.parameters)
            if nparams > 1:
                axfunc(ax, l)
            elif len(sig) == 1:
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

    os.system(cmd)

def frames_to_mp4(directory, fps=10, prefix='Loop', crf=5, outname='out'):
    ''' Send command to create video with ffmpeg
    crf controls quality. 1 is the best. 18 is not that bad...
    '''
    #cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
    #        '-r 15 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
    #        '{}.mp4').format(directory, prefix, outname)
    # Should be higher quality still compatible with outdated media players
    # And ppt....
    cmd = (r'cd "{0}" & ffmpeg -framerate {1} -i {3}_%03d.png -c:v libx264 '
            '-r {2} -pix_fmt yuv420p -crf {5} -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{4}.mp4').format(directory, fps, fps+5, prefix, outname, crf)
    # Need elite player to see this one, but it should be better in all ways
    #cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
            #' -crf 17 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            #'{}.mp4').format(directory, prefix, outname)
    os.system(cmd)


### Reference marks
def plot_powerlaw_lines(ax, slope=-2, num=20, label='Area Scaling', **kwargs):
    '''
    Put some reference lines on a log-log plot indicating a certain power law dependence
    y = a * x^slope
    values of a chosen to fill the plot
    '''
    ylims = ax.get_ylim()
    ymin, ymax = ylims
    logymin, logymax = np.log10(ymin), np.log10(ymax)
    xlims = ax.get_xlim()
    xmin, xmax = xlims
    logxmin, logxmax = np.log10(xmin), np.log10(xmax)
    # y and x points that the lines should pass through
    y = np.logspace(logymin, logymax, num)
    x = np.logspace(logxmin, logxmax, num)
    xplot = np.logspace(logxmin, logxmax, 100)
    plotargs = dict(linestyle='--', alpha=.2, color='black', label=None)
    plotargs.update(kwargs)
    for xi, yi in zip(x, y[::-1]):
        #ax.plot(xlims, (yi, yi + yi/xmin**slope *(xmax**slope - xmin**slope)), '--', alpha=.2, color='black')
        ax.plot(xplot, yi/xi**slope * xplot**slope, **plotargs)
    ax.lines[-1].set_label(label)
    # Put the limits back
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

# Used to be called this.  Leaving it here to not break old scripts
plot_log_reference_lines = plot_powerlaw_lines
def plot_load_lines(R, n=20, Iscale=1, ax=None, **kwargs):
    '''
    Put some reference lines indicating load lines for a particular series resistance.
    Assumes V is in volts and I is in amps.  Use I=1e-6 to scale to microamps
    '''

    plotargs = dict(linestyle='--', alpha=.2, c='black')
    plotargs.update(kwargs)

    if ax is None:
        ax = plt.gca()
    ylims = ax.get_ylim()
    ymin, ymax = ylims
    xlims = ax.get_xlim()
    xmin, xmax = xlims

    # Fill the whole plot with lines.  Find points to go through
    if ax.get_yscale() == 'linear':
        yp = np.linspace(ymin, ymax , n)
    else:
        # Sorry if this errors.  Negative axis ranges are possible on a log plot.
        logymin, logymax = np.log10(ymin), np.log10(ymax)
        yp = np.logspace(logymin, logymax + np.log10(ymax - ymin), n)
    if ax.get_xscale() == 'linear':
        xp = np.linspace(xmin, xmax , n)
    else:
        logxmin, logxmax = np.log10(xmin), np.log10(xmax)
        xp = np.logspace(logxmin, logxmax + np.log10(xmax - xmin), n)

    # Load lines aren't lines on log scale, so plot many points
    x = np.linspace(xmin, xmax, 500)
    # Plot one at a time so you can just label one (for legend)
    slope = 1 / R / Iscale
    for xi,yi in zip(xp, yp):
        ax.plot(x, yi - slope * (x - xi), **plotargs)
    # Label the last one
    ax.lines[-1].set_label('{}$\Omega$ Load Line'.format(mpfunc(R, None)))
    # Put the limits back
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

def plot_power_lines(pvals=None, npvals=10, ax=None, xmin=None):
    '''
    Plot lines of constant power on the indicated axis  (should be I vs V)
    pass either pvals (explicitly set the power levels) or npvals (set the number of power values to plot)
    TODO: Label power values
    '''
    if ax is None:
        ax = plt.gca()

    x0, x1 = ax.get_xlim()
    if xmin is not None:
        x0 = xmin
    y0, y1 = ax.get_ylim()

    if pvals is None:
        # Determine power range from axis limits
        if (x0 < 0) or (y0 < 0):
            minp = 0
        else:
            minp = x0 * y0
        maxp = x1 * y1
        if (ax.get_yscale() == 'log') or (ax.get_xscale() == 'log'):
            if minp < 0:
                # Sometimes scale of axis includes negative powers.  Data shouldn't.
                # Assuming here the voltage axis is the problem.  Might need to fix it later.
                minp = y0 * (x0 + 0.1 * (x1 - x0))
            pvals = np.logspace(np.log10(minp), np.log10(maxp), 10)[1:-1]
        else:
            pvals = np.linspace(minp, maxp, npvals)[1:-1]

    # Easiest to space equally in x.  Could change this later so that high slope areas get enough data points.
    x = np.linspace(x0, x1, 1000)
    #pvals = np.linspace(pmin, pmax, nlines)
    ylist = [p/x for p in pvals]
    for y in ylist:
        ax.plot(x, y, '--', color='black', alpha=.3)

    # Put the limits back
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def engformatter(axis='y', ax=None):
    if ax is None:
        ax = plt.gca()
    if axis.lower() == 'x':
        axis = ax.xaxis
    else:
        axis = ax.yaxis
    axis.set_major_formatter(mpl.ticker.EngFormatter())
