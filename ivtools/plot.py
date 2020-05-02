""" Functions for making plots with IV data """

import ivtools
import ivtools.analyze

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
from matplotlib.widgets import SpanSelector, RectangleSelector
from inspect import signature
import os
from numbers import Number
from functools import wraps
from collections import deque


def arrowpath(x, y, ax=None, **kwargs):
    # make a quiver style plot along a path
    # Draws one arrow per pair of data points
    # Should use interpolation or downsampling beforehand so the arrows are not too small
    if ax is None:
        ax = plt.gca()
    qkwargs = dict(scale_units='xy', angles='xy', scale=1, width=.005)
    if 'c' in kwargs:
        qkwargs['color'] = kwargs['c']
    # only pass these keywords through
    kws = ['alpha', 'scale', 'scale_units', 'width', 'headwidth', 'headlength',
           'headaxislength', 'minshaft', 'minlength', 'color', 'pivot', 'label',
           'clim', 'cmap', 'linestyle', 'zorder']
    for k,v in kwargs.items():
        if k in kws:
            qkwargs[k] = v
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], **qkwargs)

def plot_multicolor(x, y, c=None, cmap='rainbow', vmin=None, vmax=None, ax=None, **kwargs):
    '''
    line plot whose color changes along its length
    '''
    from matplotlib.collections import LineCollection
    if ax is None:
        fig, ax = plt.subplots()
    cm = plt.get_cmap(cmap)
    if c is None:
        #c = np.arange(len(x))
        c = cm(np.linspace(0, 1, len(x)))
    else:
        # be able to scale/clip the range of colors using vmin and vmax (like imshow)
        cmin = np.min(c)
        cmax = np.max(c)
        if vmin is None:
            vmin = cmin
        if vmax is None:
            vmax = cmax

        scaledc = (c - vmin) / (vmax - vmin)
        c = cm(np.clip(scaledc, 0, 1))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap(cmap))
    #lc.set_array(c)
    lc.set_color(c)
    lc.set_linewidth(2)
    lc.set(**kwargs)

    ax.add_collection(lc)
    ax.autoscale()

def plotiv(data, x='V', y='I', c=None, ax=None, maxsamples=500000, cm='jet', xfunc=None, yfunc=None,
           plotfunc=plt.plot, autotitle=False, labels=None, labelfmt=None, colorbyval=True,
           hold=False , **kwargs):
    '''
    IV loop plotting which can handle single or multiple loops.
    the point is mainly to do coloring and labeling in a nice way.

    data structure should be dict-like or list-like of dict-like
    Can plot any column vs any other column
    Automatically labels the axes, with name and units if they are in the data structure
    Can assign a colormap to the lines
    if you want a single color for all lines, use the "color" keyword
    Can transform the x and y data by passing functions to xfunc and/or yfunc arguments
    if x and y specify a scalar value, then just one scatter plot is made
    if x is None, then y is plotted vs range(len(y))
    maxsamples : downsample to this number of data points if necessary
    kwargs passed through to ax.plot (can customize line properties this way)

    Maybe unexpected behavior: A new figure is created if ax=None

    Can pass an arbitrary plotting function, which defaults to plt.plot
    Could then define some other plots that take IV data and reuse plotiv functionality.
    '''
    if hold:
        # you can type hold=1 instead of ax=plt.gca()
        fig = plt.gcf()
        ax = plt.gca()
    if ax is None:
        fig, ax = plt.subplots()

    # check if plotfunc uses any of the same keywords as plotiv
    plotiv_args = inspect.getfullargspec(plotiv).args
    plotfunc_args = inspect.getfullargspec(plotfunc).args
    overlap_args = set(plotiv_args) & set(plotfunc_args)
    # We know how to deal with these ones
    overlap_args -= set(('x', 'y', 'c', 'ax'))
    if any(overlap_args):
        print(f'the following args are used by both plotiv and plotfunc, and will not pass through: {overlap_args}')

    # might be one curve, might be many
    dtype = type(data)
    assert dtype in (list, dict, pd.Series, pd.DataFrame)
    # Convert to a list of dict-like, so we can use a consistent syntax below
    if dtype in (dict, pd.Series):
        data = [data]
    elif dtype == pd.DataFrame:
        data = data.to_dict(orient='records')

    lendata = len(data)

    ##### Line coloring #####
    if lendata > 1:
        # There are several loops
        # Pick colors for each line
        # you can either pass a list of colors, or a colormap
        if isinstance(cm, list):
            colors = cm
        elif 'color' in kwargs:
            # color overrides everything
            colors = [kwargs['color']] * len(data)
        else:
            if isinstance(cm, str):
                # Str refers to the name of a colormap
                cmap = plt.cm.get_cmap(cm)
            elif type(cm) in (mpl.colors.LinearSegmentedColormap, mpl.colors.ListedColormap):
                cmap = cm
            # TODO: add vmin and vmax arguments to stretch the color map
            if c is None:
                colors = [cmap(c) for c in np.linspace(0, 1, len(data))]
            elif type(c) is str:
                cdata = np.array([d[c] for d in data])
                if colorbyval:
                    # color by value of the column given
                    cmax = np.max(cdata)
                    cmin = np.min(cdata)
                    normc = (cdata - cmin) / (cmax - cmin)
                    colors = cmap(normc)
                else:
                    # this means we want to color by the category of the value in the column
                    # Should put in increasing order, but equally spaced on the color map,
                    # not proportionally spaced according to the value of the data column
                    uvals, category = np.unique(cdata, return_inverse=True)
                    colors = cmap(category / max(category))
            else:
                # It should be either a list of colors or a list of values
                # Cycle through it if it's not long enough
                firstval = next(iter(c))
                if hasattr(firstval, '__iter__'):
                    #it's a list of strings, or of RGB values or something
                    colors = [c[i%len(c)] for i in range(lendata)]
                else:
                    # It's probably an array of values?  Map them to colors
                    normc = (c - np.min(c)) / (np.max(c) - np.min(c))
                    colors = cmap(normc)
    else:
        # Use default color cycling
        colors = [None]

    ##### Line labeling #####
    if labels is not None:
        if type(labels) is str:
            # TODO: allow passing a list of strings, which can label by multiple values
            #       but there is an ambiguity if the length of that list happens to be the
            #       same as the length of the data..
            # label by the key with this name
            if type(data) == list:
                label_list = [d[labels] for d in data]
            else:
                #should be dataframe
                label_list = list(data[labels])
        else:
            # otherwise we will iterate through labels directly (so you can pass a list of labels)
            # make np.nan count as None (not labelled)
            label_list = list(map(lambda v: None if (isinstance(v, Number) and np.isnan(v)) else v, labels))
        assert len(label_list) == len(data)
        # reformat the labels in case they are numbers
        if labelfmt:
            label_list = list(map(lambda v: None if v is None else format(v, labelfmt), label_list))
    else:
        # even if we did not specify labels, we will still iterate through a list of labels
        # Make them all None (unlabeled)
        label_list = [None] * len(data)

    # Drop repeat labels that have the same line style, because we don't need hundreds of repeat labeled objects
    # right now only the color identifies the line style
    # Python loop style.. will not be efficient, but it's the first solution I thought of.
    lineset = set()
    if lendata > 1:
        for i in range(len(data)):
            l = label_list[i]
            cc = colors[i]
            if type(cc) is np.ndarray:
                # need a hashable type...
                cc = tuple(cc)
            if (l,cc) in lineset:
                label_list[i] = None
            else:
                lineset.add((l,cc))

    ##### Come up with axis labels #####
    if type(y) == str:
        yname = y
    elif hasattr(y, '__call__'):
        yname = y.__name__
    elif hasattr(y, '__iter__'):
        # Don't know if this is a good idea
        yname = '[{}, ..., {}]'.format(y[0], y[-1])
    else:
        raise exception('I do not know wtf you are trying to plot')
    if x is None:
        xname = None
    elif type(x) == str:
        xname = x
    elif hasattr(x, '__call__'):
        xname = x.__name__
    elif hasattr(x, '__iter__'):
        # Don't know if this is a good idea
        xname = '[{}, ..., {}]'.format(x[0], x[-1])
    else:
        raise exception('I do not know wtf you are trying to plot')

    defaultunits = {'V':     ('Voltage', 'V'),
                    'Vcalc': ('Device Voltage', 'V'),
                    'Vd':    ('Device Voltage', 'V'),
                    'I':     ('Current', 'A'),
                    'G':     ('Conductance', 'S'),
                    'R':     ('Resistance', '$\Omega$'),
                    't':     ('Time', 's'),
                    None:    ('Data Point', '#')}

    longnamex, unitx = x, '?'
    longnamey, unity = y, '?'
    if x in defaultunits.keys():
        longnamex, unitx = defaultunits[x]
    if y in defaultunits.keys():
        longnamey, unity = defaultunits[y]

    # Overwrite the label guess with value from dict if it exists
    # Only consider the first data -- hopefully there are not different units passed at once
    iv0 = data[0]
    if ('longnames' in iv0.keys()) and (type(iv0['longnames']) == dict):
        if x in iv0['longnames'].keys():
            longnamex = iv0['longnames'][x]
        if y in iv0['longnames'].keys():
            longnamey = iv0['longnames'][y]
    if ('units' in iv0.keys()) and (type(iv0['units']) == dict):
        if x in iv0['units'].keys():
            unitx = iv0['units'][x]
        if y in iv0['units'].keys():
            unity = iv0['units'][y]

    xlabel = longnamex
    if unitx != '?':
        xlabel += f' [{unitx}]'
    ylabel = longnamey
    if unity != '?':
        ylabel += f' [{unity}]'

    if xfunc is not None:
        xlabel = '{}({})'.format(xfunc.__name__, xlabel)
    if yfunc is not None:
        ylabel = '{}({})'.format(yfunc.__name__, ylabel)
    # We will label the axes at the end, in case the plotfunc tries to set its own labels


    ##### Make the lines #####
    lines = []
    for iv, c, l in zip(data, colors, label_list):
        ivtype = type(iv)
        if ivtype not in (dict, pd.Series):
            # what the F, you passed a list with something weird in it
            print('plotiv did not understand the input datatype {}'.format(ivtype))
            continue

        ## construct the x and y arrays that you actually want to plot
        # Can pass non dict keys to plot on the x,y axes (func, list..)
        if type(y) == str:
            Y = iv[y]
        elif hasattr(y, '__call__'):
            # can pass a function as y, this will be called on the whole data structure
            Y = y(iv)
        else:
            Y = y

        if hasattr(Y, '__iter__'):
            lenY = len(Y)
            Yscalar = False
        else:
            lenY = 1
            Yscalar = True

        if x is None:
            X = np.arange(lenY)
        elif type(x) == str:
            X = iv[x]
        elif hasattr(x, '__call__'):
            X = x(iv)
        else:
            X = x

        if hasattr(X, '__iter__'):
            lenX = len(X)
            Xscalar = False
        else:
            lenX = 1
            Xscalar = True

        if xfunc is not None:
            X = xfunc(X)
        if yfunc is not None:
            Y = yfunc(Y)

        # X and Y should be the same length, if they are not, truncate one
        if lenX != lenY:
            print('_plot_single_iv: X and Y arrays are not the same length! Truncating the longer one.')
            if lenX > lenY:
                X = X[:lenY]
                lenX = lenY
            else:
                Y = Y[:lenX]
                lenY = lenX

        if maxsamples is not None and maxsamples < lenX:
            # Down sample data
            print('Downsampling data for plot!!')
            step = int(lenX/maxsamples)
            X = X[np.arange(0, lenX, step)]
            Y = Y[np.arange(0, lenY, step)]

        if Xscalar and Yscalar:
            # there's only one datapoint per iv loop
            # the way this is set up, we plot one line per iv loop
            # so we cannot easily connect the points in the plot
            # Will be invisible if e.g. plotfunc == plt.plot and there's no marker
            #if plotfunc == plt.plot
            #    plotfunc = plt.scatter
            pass

        plotfunc_kwargs = dict(c=c, label=l)
        if 'ax' in plotfunc_args:
            # If the plotfunc takes an axis argument, pass it through,
            # otherwise have to assume it plots on the current axis..
            plotfunc_kwargs['ax'] = ax
        # all plotiv kwargs get passed through to plotfunc, even if they overwrite something (i.e. label)
        plotfunc_kwargs.update(kwargs)
        if ('x' not in plotfunc_args) or ('y' not in plotfunc_args):
            # stupid plotfunc didn't label x and y keywords, assume they are the first two arguments
            newline = plotfunc(X, Y, **plotfunc_kwargs)
        else:
            newline = plotfunc(x=X, y=Y, **plotfunc_kwargs)
        # probably going to be a list of length 1 lists...
        lines.append(newline)

    # Use EngFormatter if the plotted values are small or large
    xlims = np.array(ax.get_xlim())
    if any(xlims > 1e3) or all(xlims < 1e-1):
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ylims = np.array(ax.get_ylim())
    if any(ylims > 1e3) or all(ylims < 1e-1):
        ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # put on a legend if there are labels
    if labels and ((type(labels) is str) or any(labels)):
        leg = ax.legend()
        if type(labels) is str:
            leg.set_title(labels)
        elif hasattr(labels, 'name'):
            leg.set_title(labels.name)

    if autotitle:
        auto_title(data, keys=None, ax=ax)

    # should I really return this?  usually I don't assign the values and then they get cached by ipython forever
    # Can always get them with plt.gca()...
    # return ax, line

## Linearized plots for conduction mechanisms
def schottky_plot(data, V='V', I='I', T=None):
    # Linearizes schottky mechanism
    # log(I) or log(I)/T^2 vs sqrt(v)
    # Should I use ln on the data or use the log scale?
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    if T is not None:
        # Assuming data is a dataframe.  Will kick myself later.
        data = data.assign(**{'I/T2': data['I'] / data['T']**2})
        plotiv(data, V, 'I/T2', xfunc=np.sqrt, ax=ax)
        ax.set_ylabel(f'{I} / {T}$^2$')
    else:
        plotiv(data, V, I, xfunc=np.sqrt, ax=ax)
        ax.set_ylabel(f'{I}')
    ax.set_xlabel(f'sqrt({V})')

def poole_frenkel_plot(data, V='V', I='I', T=None):
    # Linearizes P-F mechanism
    # log(G) or log(G)/T^2 vs sqrt(v)
    # Should I use ln on the data or use the log scale?
    data = data.assign(G=data[I]/data[V])
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    if T is not None:
        # Assuming data is a dataframe.  Will kick myself later.
        data = data.assign(**{'G/T2': data['G'] / data['T']**2})
        plotiv(data, V, 'G/T2', xfunc=np.sqrt, ax=ax)
        ax.set_ylabel(f'G / {T}$^2$')
    else:
        plotiv(data, V, 'G', xfunc=np.sqrt, ax=ax)
        ax.set_ylabel('G')
    ax.set_xlabel(f'sqrt({V})')

def arrhenius_plot(data, V='V', I='I', T='T', numv=20, minv=None, maxv=None, cm=plt.cm.viridis, **kwargs):
    # Thermal activation plot -- needs some work though
    # log(I) or log(G) vs 1000/T
    # This is a little tricky because voltage values need to be interpolated in general
    # If I is multi-valued in voltage then it can be a pain in the ass to interpolate
    # for each interpolated value of V, we plot a line
    # Not using plotiv because I couldn't think of the smart way to "pivot" the nested dataframe
    # Should output the "interpolated pivot" data for fitting
    plt.figure()
    if maxv is None:
        maxv =  np.max(data[V].apply(np.max))
    if minv is None:
        minv = 0.05
    vs = np.linspace(minv, maxv, numv)
    colors = cmap(np.linspace(0, 1, len(vs)))
    fits = []
    for v,c in zip(vs, colors):
        it = ivtools.analyze.interpiv(data, v, column=V, left=np.nan, right=np.nan, findmonotonic=True)
        plt.plot(1000/it['T'], np.log(it[I]), marker='.', color=c, label=format(v, '.2f'), **kwargs)
        #notnan = ~it['I'].isnull()
        #fits.append(polyfit(1/it['T'][notnan], log(it['G'][notnan]), 1))
        #fitx = np.linspace(1/300, 1/81)
        #color = ax.lines[-1].get_color()
        #plt.plot(fitx, np.polyval(fits[-1], fitx), color=color, alpha=.7)
        #ax.lines[-1].set_label(None)
    #plt.legend(title='Device Voltage')
    colorbar_manual(minv, maxv, cmap=cmap, label='Applied Voltage [V]')
    plt.xlabel('Temperature [K] (scale 1/T)')
    plt.ylabel('log(I)')
    formatter = mpl.ticker.FuncFormatter(lambda x, y: format(1000/x, '.0f'))
    plt.gca().xaxis.set_major_formatter(formatter)

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
    return title

def plot_R_states(data, v0=.1, v1=None, **kwargs):
    resist_states = ivtools.analyze.resistance_states(data, v0, v1)
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
    engformatter('y', ax)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Resistance [$\\Omega$]')


def violinhist(data, x, range, bins=50, alpha=.8, color='C0', logbin=True, logx=True, ax=None):
    # histogram version of violin plot (when there's not a lot of data so the KDE looks weird)
    # data should be a list of arrays of values
    if ax is None:
        ax = plt.gca()
    order = np.argsort(x)
    x = np.array(x)[order]
    data = [data[o] for o in order]
    # we could scale the histograms so that their max value is half the minimum x distance
    scale = 0.45
    if logx:
        assert(all(x>0))
        # how to make them look the same amplitude on log scale, without any possible overlapping?
        # a slightly tricky problem. we e.g. need different amplitudes for the left and right bar for them to look equal
        logx = np.log10(x)
        logamp = np.min(np.diff(logx)) * scale
        ampleft = x - 10**(logx - logamp)
        ampright = 10**(logx + logamp) - x
        ax.set_xscale('log')
    else:
        amp = np.min(np.diff(x)) * scale
        ampleft = ampright = [amp] * len(x)
    for d,xi,aleft,aright in zip(data, x, ampleft, ampright):
        if logbin:
            logd = np.log10(d[d > 0])
            logrange = [np.log10(r) for r in range]
            hist, edges = np.histogram(logd, bins=bins, range=logrange)
            edges = 10**edges
            ax.set_yscale('log')
        else:
            hist, edges = np.histogram(d, bins=bins, range=range)
        hist = hist / np.max(hist)
        heights = np.diff(edges)
        ax.barh(edges[:-1], aright*hist, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1)
        ax.barh(edges[:-1], -aleft*hist, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1)
    ax.xaxis.set_ticks(histx)
    ax.xaxis.set_ticklabels(histx)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.vlines(x, *range, color='black', alpha=.5)
    ax.set_ylim(range)

def loghist(data, bins=50, range=None, density=False, ax=None, **kwargs):
    '''
    Just like a plt.hist except the binning is done on the logged data
    kwargs to go plt.bar
    '''
    # TODO it could also be ok if they are ALL negative
    data = data[data > 0]
    if ax is None:
        ax = plt.gca()
    ax.set_xscale('log')
    logd = np.log10(data)
    if range is not None:
        range = [np.log10(r) for r in range]
    hist, edges = np.histogram(logd, bins=bins, density=density, range=range)
    edges = 10**edges
    hist = hist
    widths = np.diff(edges)
    ax.bar(edges[:-1], hist, width=widths, align='edge', **kwargs)

def grouped_hist(df, col, groupby=None, range=None, bins=30, logx=True, ax=None):
    '''
    Histogram where the bars are split into colors by group
    I don't know if this is a good idea or not.  maybe for < 4 groups
    '''
    if ax is None:
        ax = plt.gca()

    if range is None:
        # every subset needs to have the same range
        if logx:
            range = np.nanpercentile(df[col][df[col] > 0], (0, 100))
        else:
            range = np.nanpercentile(df[col], (0, 100))

    params = []
    hists = []
    for k, g in df.groupby(groupby):
        if logx:
            x = np.log10(g[col][g[col] > 0])
            logrange = [np.log10(v) for v in range]
            hist, edges = np.histogram(x[~np.isnan(x)], bins=bins, range=logrange)
            if any(hist < 0):
                print('wtf')
            edges = 10**edges
            ax.set_xscale('log')
        else:
            hist, edges = np.histogram(g[col][~np.isnan(g[col])], bins=bins, range=range)
        params.append(k)
        hists.append(hist)
    # edges will all be the same
    widths = np.diff(edges)
    heights = np.stack(hists)
    bottoms = np.vstack((np.zeros(bins), np.cumsum(heights, 0)[:-1]))
    tops = bottoms + heights
    for p,bot,height in zip(params, bottoms, heights):
        ax.bar(edges[:-1], height, widths, align='edge', bottom=bot, label=p, edgecolor='white')

    ax.legend(title=groupby)
    ax.set_xlabel(col)
    ax.set_ylabel('N')



def paramplot(df, x, y, parameters, yerr=None, cmap=plt.cm.gnuplot, labelformatter=None,
              sparseticks=True, xlog=False, ylog=False, sortparams=False, paramvals=None,
              ax=None, **kwargs):
    '''
    Plot y vs x grouping by any number of parameters
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

def plot_channels(chdata, ax=None, alpha=.8, **kwargs):
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
                x = ivtools.analyze.maketimearray(chdata, c)
                ax.set_xlabel('Time [s]')
                ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
            else:
                x = range(len(chdata[c]))
                ax.set_xlabel('Data Point')
            ax.plot(x, chplotdata, color=colors[c], label=c, alpha=alpha, **kwargs)
            # lightly indicate the channel range
            choffset = chdata['OFFSET'][c]
            chrange = chdata['RANGE'][c]
            ax.fill_between((0, np.max(x)), -choffset - chrange, -choffset + chrange, alpha=0.05, color=colors[c])
    ax.legend(title='Channel')
    ax.set_ylabel('Voltage [V]')

def colorbar_manual(vmin=0, vmax=1, cmap='jet', ax=None, cax=None, **kwargs):
    ''' Normally you need a "mappable" to create a colorbar on a plot.  This function lets you create one manually. '''
    if ax is None:
        ax = plt.gca()
    if hasattr(vmin, '__iter__'):
        # I think you meant to send in the values directly instead of min and max
        vmin = np.min(vmin)
        vmax = np.max(vmin)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Sometimes you want to specify the axis for the colorbar itelf.
    cb = plt.colorbar(sm, ax=ax, cax=cax, **kwargs)
    return cb



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

def plot_ivt(d, phaseshift=14, fig=None, **kwargs):
    ''' A not-so-refined subplot of current and voltage vs time'''
    if fig is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    else:
        axs = fig.get_axes()
        if len(axs) == 2:
            ax1, ax2 = axs
        elif len(axs) == 0:
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
    if 't' not in d:
        d['t'] = ivtools.analyze.maketimearray(d)
    ax1.plot(d['t'], d['V'], c='blue', label='V')
    ax2.plot(d['t'] - phaseshift* 1e-9, d['I'], c='green', label='I')
    ax2.set_ylabel('Current [A]', color='green')
    ax1.set_ylabel('Applied Voltage [V]', color='blue')
    ax1.set_xlabel('Time [s]')
    ax1.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ax2.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

# Could be a module, but I want it to keep its state when the code is reloaded
class InteractiveFigs(object):
    '''
    A class to manage the figures used for automatic plotting of IV data while it is measured.
    Right now we are limited to one axis per figure ...  could be extended.
    can have several plotting functions per axis though ..
    '''
    # TODO: save/load configurations to disk?
    def __init__(self, n=4, clear_state=False):
        statename = self.__class__.__name__
        if statename not in ivtools.class_states:
            ivtools.class_states[statename] = {}
        self.__dict__ = ivtools.class_states[statename]
        if not self.__dict__ or clear_state:
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
            # if False, disables updateline, newline
            self.enable = True
            # Put a list of functions here to pass the data through before plotting (e.g. smoothing)
            self.preprocessing = []
            self.processed_data = None

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
        if self.enable:
            if data is not None:
                if any(self.preprocessing):
                    for pp in self.preprocessing:
                        # Just run the data through all the functions
                        try:
                            data = pp(data)
                        except:
                            print('Pre-processing failed!')
                        # In case you want to access it without running the processing again
                    self.processed_data = data
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
                    except Exception as e:
                        ax.plot([])
                        print('Plotter number {} failed!: {}'.format(axnum, e))
                    ax.get_figure().canvas.draw()
            mypause(0.05)

    def set_maxlines(self, maxlines=None):
        for ax in self.axs:
            if maxlines is None:
                ax.lines = list(ax.lines)
            else:
                ax.lines = deque(ax.lines, maxlen=maxlines)

    def updateline(self, data):
        '''
        Update the data in the previous plots.
        '''
        # Since I can't know how to get to the actual plotted data without calling the plot
        # function again, this works by deleting the last line and plotting a new one with
        # the same colors
        # I am assuming for now that the plot functions each produce one line.
        if self.enable:
            if any(self.preprocessing):
                for pp in self.preprocessing:
                    # Just run the data through all the functions
                    data = pp(data)
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
                    except Exception as e:
                        print('Plotter number {} failed!: {}'.format(axnum, e))
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
# They should take the data and an axis to plot on
# Should handle single or multiple loops
# TODO: Can I make a wrapper that makes that easier?
# TODO: don't have each plot function downsample themselves, just do it once and share the result
def parametrized(dec):
    ''' This is a meta-decorator to create a parametrized decorator.  You got a better idea? '''
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def plotter(plotfunc, cmap='jet', maxloops=100, maxsamples=5000, clear=False):
    '''
    Plotting functions decorated with this can be written as though they are plotting a single loop,
    but will automatically plot multiple loops if passed.
    Will also try to avoid plotting too much data by decimating (not implemented) it and plotting a maximum number of loops at a time
    if the plotter does not plot the arrays directly, then you might not need to decimate/limit the number
    I wish python had pattern matching...
    '''
    # TODO: Usually the data is in list or dict form when this is used. extend to series and dataframes?
    cmap = plt.get_cmap(cmap)
    @wraps(plotfunc)
    def wrap(data, ax=None, *args, **kwargs):
        if clear:
            ax.cla()
        typein = type(data)
        if typein is dict:
            # if data is length 1, simply call plotfunc
            plotfunc(data, ax, *args, **kwargs)
        elif typein is list:
            # if data is longer than length 1, call plotfunc several times, and try to apply a color map
            # use inspect to check if plotfunc can take keywords
            lendata = len(data)
            if lendata > maxloops:
                data = [data[int(n)] for n in np.round(np.linspace(0, lendata - 1, maxloops))]
            passcolor = False
            argspec = inspect.getfullargspec(plotfunc)
            if (argspec.varkw is not None) or ('color' in argspec.kwonlyargs) or ('color' in argspec.args):
                # plotter won't error if we pass the color keyword argument
                # it might even work ..
                colors = cmap(np.linspace(0, 1, len(data)))
                passcolor = True
            for i,d in enumerate(data):
                if passcolor:
                    kwargs['color'] = colors[i]
                plotfunc(d, ax, *args, **kwargs)
        else:
            print('Cannot plot that kind of data')
    return wrap

def plottertemplate(data, ax, **kwargs):
    '''
    Minimal template for defining a new plotter.
    Needs kwargs if you want live updating.
    '''
    ax.plot(data['x'], data['y'], **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def ivplotter(data, ax=None, maxloops=100, smooth=False, **kwargs):
    # Maybe smooth data a bit and give it to plotiv
    # Make sure not too much data gets plotted, or it slows down the program a lot.
    # Would be better to smooth before splitting ...
    # kwargs gets passed through to plotiv, which passes them through to plt.plot
    if ax is None:
        fig, ax = plt.subplots()
    if smooth:
        data = ivtools.analyze.moving_avg(data, window=10)
    if type(data) is list:
        nloops = len(data)
    else:
        nloops = 1
    if nloops > maxloops:
        print('You captured {} loops.  Only plotting {} loops'.format(nloops, maxloops))
        loopstep = int(nloops / 99)
        data = data[::loopstep]
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    #ax.plot(data['V'], data['I'], **kwargs)
    plotiv(data, ax=ax, maxsamples=5000, **kwargs)

def R_vs_cycle_plotter(data, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # Using line plot because that's all InteractiveFigs knows about

    if len(data['V'] > 1):
        v0 = np.min(data['V'])
        v1 = np.max(data['V'])
        R = ivtools.analyze.resistance(data, v0, v1)
    else:
        R = np.nan
    # Try to always make a data point after the largest one already on the plot
    if len(ax.lines) > 0:
        lastline = ax.lines[-1]
        lastx = np.max(lastline.get_data()[0])
    else:
        lastx = -1
    ax.plot(lastx + 1, R, marker='.', markersize=12, **kwargs)
    ax.set_xlabel('Cycle #')
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_ylabel('Resistance (from line fit) [$\Omega$]')
    engformatter('y')

#@plotter(clear=True)
def chplotter(data, ax=None, **kwargs):
    # basically just plot_channels with downsampling
    if ax is None:
        fig, ax = plt.subplots()
    # Remove previous lines
    for l in ax.lines[::-1]: l.remove()
    # Plot at most 100000 datapoints of the waveform
    channels = [ch for ch in ['A', 'B', 'C', 'D'] if ch in data]
    if len(channels) > 0:
        lendata = len(data[channels[0]])
        if lendata > 100000:
            print('Captured waveform has {} pts.  Downsampling data.'.format(lendata))
            step = lendata // 50000
            #plotdata = ivtools.analyze.decimate(data, step, columns=channels)
            plotdata = ivtools.analyze.sliceiv(data, step=step, columns=channels)
            if 'downsampling' in plotdata:
                plotdata['downsampling'] *= step
            else:
                plotdata['downsampling'] = step
        else:
            plotdata = data
        plot_channels(plotdata, ax=ax)

#@plotter
def dVdIplotter(data, ax=None, **kwargs):
    ''' Plot dV/dI vs V'''
    if ax is None:
        fig, ax = plt.subplots()
    mask = np.abs(data['V']) > .01
    vmasked = data['V'][mask]
    imasked = data['I'][mask]
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
        t = ivtools.analyze.maketimearray(data)
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
    if 't' in data:
        t = data['t']
    else:
        t = ivtools.analyze.maketimearray(data)
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

    def calc_VoverI(data):
        mask = np.abs(data['V']) > .01

        if 'Vmeasured' in data:
            VoverI = data['Vmeasured'] / data['I']
        elif 'Imeasured' in data:
            VoverI = data['V'] / data['Imeasured']
        else:
            VoverI = data['V'] / data['I']

        VoverI[~mask] = np.nan
        VoverI[VoverI <= 0] = np.nan

        return VoverI

    ax.set_yscale('log')

    #ax.plot(V, VoverI, **kwargs)
    # should work with multiple loops
    plotiv(data, y=calc_VoverI, ax=ax, **kwargs)
    #color = ax.lines[-1].get_color()

    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ax.set_xlabel('Voltage [V]')
    #ax.set_ylabel('V/I [$\Omega$]', color=color)
    # Also called Chordal resistance
    ax.set_ylabel('Static Resistance (V/I) [$\Omega$]')

# TODO differential resistance plotter

def vcalcplotter(data, ax=None, R=None, **kwargs):
    '''
    Subtract internal series resistance voltage drop
    For Lassen R = 143, 2164, 8197, 12857
    '''
    # Sorry this is shitty ...
    dtype = type(data)
    # i don't want to modify input data, make shallow copy
    if dtype in (pd.Series, pd.DataFrame):
        data = data.copy(deep=False)
    else:
        data = data.copy()
    Rmap = {0:143, 1000:2164, 5000:8197, 9000:12857}
    if ax is None:
        fig, ax = plt.subplots()

    if 'Vd' in data:
        plotiv(data, ax=ax, x='Vd', **kwargs)
    elif 'Vcalc' in data:
        plotiv(data, ax=ax, x='Vcalc', **kwargs)
    else:
        # Desperately try to figure out the series resistance and calculate Vd
        if hasattr(R, '__call__'):
            R = R()
        if R is None:
            # Look for R in the meta data
            # Check normal meta
            R = data.get('R_series')
            if R is not None:
                # If it is a lassen coupon, then convert to the measured values of series resistors
                if dtype is list:
                    # Fuck
                    wafer_code = data.get('wafer_code')[0]
                elif dtype is pd.DataFrame:
                    wafer_code = data.get('wafer_code').iloc[0]
                else:
                    wafer_code = data.get('wafer_code')

                if wafer_code == 'Lassen':
                    if (dtype == pd.Series) or (not hasattr(R, '__iter__')):
                        R = int(R)
                        if R in Rmap:
                            R = Rmap[R]
                    else:
                    # lol
                        R = np.array([Rmap[r] if r in Rmap else r for r in R])
            else:
                # Assumption for R_series if there's nothing in the meta data
                R = 0
        # wtf modifies the input data?  Shouldn't do that.

        # Determine the units of I data.  Assume the units are uniform.
        Iunit = None
        # This is STUPID
        if dtype is list:
            representative = data[0]
        elif dtype is pd.DataFrame:
            representative = data.iloc[0]
        elif dtype in (pd.Series, dict):
            representative = data
        if 'units' in representative:
            if 'I' in representative['units']:
                Iunit = representative['units']['I']
        if Iunit == '$\mu$A':
            Iunit = 1e-6
        else:
            Iunit = 1

        if type(data) is list:
            for d in data:
                d['Vcalc'] = d['V'] - R * d['I'] * Iunit
        else:
            # Works for Series, dict, DataFrame
            data['Vcalc'] = data['V'] - R * data['I'] * Iunit

        plotiv(data, ax=ax, x='Vcalc', **kwargs)

        if hasattr(R, '__iter__'):
            R = R[0]
        ax.set_xlabel('V device (calculated assuming Rseries={}$\Omega$) [V]'.format(R))

    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())


### Widgets
# TODO: more of these!
def plot_span(data=None, ax=None, plotfunc=plotiv, **kwargs):
    '''
    Select index range from plot of some parameter vs index.  Plot the loops there.
    To use selector, just make sure you don't have any other gui widgets active
    Will remain active as long as the return value is not garbage collected
    Ipython keeps a reference to all outputs, so this will stay open forever if you don't assign it a value
    '''
    if data is None:
        # Check for global variables ...
        # Sorry if this offends you ..  it offends me
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

# Data selector
def plot_selector(data=None, ax=None, plotfunc=plotiv, x='V', y='I', **kwargs):
    '''
    Plot all the data, x vs y
    Select 2D range from plot of x vs y.
    print the data indices that have data in the range selected
    # Or should it return the data subset?

    # TODO: use lasso tool instead

    Will remain active as long as the return value is not garbage collected
    Ipython keeps a reference to all outputs, so this will stay open forever if you don't assign it a value
    '''
    if data is None:
        # Check for global variables ...
        # Sorry if this offends you ..  it offends me
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

    plotfunc(data, x=x, y=y, ax=ax)
    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        #print("(%.2e, %.2e) --> (%.2e, %.2e)" % (x1, y1, x2, y2))
        #print("The button you used were: %s %s" % (eclick.button, erelease.button))
        # Find the data that has values in the selected range
        print(f'[{xmin}, {xmax}, {ymin}, {ymax}]')
        def inside(d):
            X = d[x]
            Y = d[y]
            return np.any((X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax))
        if type(data) is pd.DataFrame:
            print(data.index[data.apply(inside, 1)])
        else:
            # Should be a list of dicts/Series
            print([i for i,d in enumerate(data) if inside(d)])
    rectprops = dict(facecolor='blue', alpha=0.3)
    RS = RectangleSelector(ax, onselect, 'box', useblit=True, rectprops=rectprops)
    return RS


### Animation
# TODO: check out the library "celluloid"
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
            plotiv(ivtools.analyze.increasing(l, sort=True), ax=ax, color=colorup, label=r'$\rightarrow$', **kwargs)
            plotiv(ivtools.analyze.decreasing(l, sort=True), ax=ax, color=colordown, label=r'$\leftarrow$', **kwargs)
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
def plot_powerlaw_lines(ax=None, slope=-2, num=20, label='Area$^{-1}$ Scaling', **kwargs):
    '''
    Put some reference lines on a log-log plot indicating a certain power law dependence
    y = a * x^slope
    values of 'a' chosen to fill the current plot limits
    For now, will not work properly if there are negative numbers on the x or y limits
    '''
    if ax is None:
        ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    if (ymin < 0) or (xmin < 0):
        raise Exception('I cannot do anything intelligent with negative axis limits yet')
    logymin, logymax = np.log10(ymin), np.log10(ymax)
    logxmin, logxmax = np.log10(xmin), np.log10(xmax)
    # y and x points that the lines should pass through
    y = np.logspace(logymin, logymax, num)
    x = np.logspace(logxmin, logxmax, num)
    if slope > 0:
        y = y[::-1]
    xplot = np.logspace(logxmin, logxmax, 100)
    plotargs = dict(linestyle='--', alpha=.2, color='black')
    plotargs.update(kwargs)
    for xi, yi in zip(x, y):
        ax.plot(xplot, yi/xi**slope * xplot**slope, label=None, **plotargs)
    # Label only one line (so legend does not repeat)
    ax.lines[-1].set_label(label)
    # Put the limits back to where they were initially
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

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
    ax.lines[-1].set_label('{}$\Omega$ Load Line'.format(metric_prefix(R)))
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


### Other kinds of plotting utilities
def metric_prefix(x):
    #longnames = ['exa', 'peta', 'tera', 'giga', 'mega', 'kilo', '', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto']
    prefix = ['E', 'P', 'T', 'G', 'M', 'k', '', 'm', '$\mu$', 'n', 'p', 'f', 'a']
    values = [1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18]
    if abs(x) < min(values):
        return '{:n}'.format(x)
    for v, p in zip(values, prefix):
        if abs(x) >= v:
            return '{:n}{}'.format(x/v, p)

def metric_prefix_longname(x, decimals=1):
    longnames = ['exa', 'peta', 'tera', 'giga', 'mega', 'kilo', '', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto']
    prefix = ['E', 'P', 'T', 'G', 'M', 'k', '', 'm', '$\mu$', 'n', 'p', 'f', 'a']
    values = [1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18]
    if abs(x) < min(values):
        return '{:n}'.format(x)
    for v, p in zip(values, longnames):
        if abs(x) >= v:
            return f'{x/v:.{decimals}f} {p}'

def engformatter(axis='y', ax=None):
    if ax is None:
        ax = plt.gca()
    if axis.lower() == 'x':
        axis = ax.xaxis
    else:
        axis = ax.yaxis
    axis.set_major_formatter(mpl.ticker.EngFormatter())

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, int(256*(maxval-minval)))), N=n)
    return new_cmap

def xylim():
    # return the command to set a plot xlim,ylim to the xlim and ylim of the current plot
    # also put it on the clipboard
    # got sick of repeating this over and over
    xlim = plt.xlim()
    ylim = plt.ylim()
    cmd = 'plt.xlim({:.5e}, {:.5e})\nplt.ylim({:.5e}, {:.5e})'.format(*xlim, *ylim)
    print(cmd)
    # I don't know how to copy a new line onto the clipboard
    df = pd.DataFrame([cmd.replace('\n', ';')])
    df.to_clipboard(index=False,header=False)
