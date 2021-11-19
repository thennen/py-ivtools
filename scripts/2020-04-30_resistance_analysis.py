# resistance analysis for lassen coupons. does all the violin histograms vs thickness/width
# and also vs any other parameter that varied in your dataset

import pandas as pd
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from numbers import Number
import os
import time
import numpy as np
from shutil import copyfile

# there are lots of devices, you might not want to repeat the individual plots
overwrite_individuals = False

# Copy the script whenever it is run
scriptpath = os.path.realpath(__file__)
scriptdir, scriptfile = os.path.split(scriptpath)
scriptcopydir = os.path.join(scriptdir, 'Script_copies')
scriptcopyfn = time.strftime("%y_%m_%d_%H_%M_%S") + scriptfile
scriptcopyfp = os.path.join(scriptcopydir, scriptcopyfn)
if not os.path.isdir(scriptcopydir):
    os.makedirs(scriptcopydir)
copyfile(scriptpath, scriptcopyfp)

plotdir = '2020_04_30_Plots'
def writefig(filename, subdir='', plotdir=plotdir, overwrite=True, savefig=False, fig=None):
    # write the current figure to disk
    if fig is None:
        fig = plt.gcf()
    plotsubdir = os.path.join(plotdir, subdir)
    if not os.path.isdir(plotsubdir):
        os.makedirs(plotsubdir)
    plotfp = os.path.join(plotsubdir, filename)
    if os.path.isfile(plotfp + '.png') and not overwrite:
        print('Not overwriting {}'.format(plotfp))
    else:
        fig.savefig(plotfp, transparent=True)
        print('Wrote {}.png'.format(plotfp))
        if savefig:
            with open(plotfp + '.plt', 'wb') as f:
                pickle.dump(plt.gcf(), f)
            print('Wrote {}.plt'.format(plotfp))

################# Read in raw data #########################

# Import data
from ivtools.io import read_pandas, glob
datadir = '.'
df = read_pandas(glob('*.s', datadir))

#df = read_pandas_glob('.', '*.s')
#df['thickness_1'] = df.thickness_1.astype(int)



def datalabel(row):
    return f'{row.dep_code}_{row.sample_number} {row.layer_1} {row.thickness_1}nm'

df['label'] = df.apply(datalabel, 1)

# TODO change the fit range to correspond to a constant electric field, not
# voltage?

def calculate_r(ivloop):
    # fit a line to determine resistance
    # Tricky to choose fit range.  Try a small one and extend if the value is high
    maxv = 0.1
    mask = (abs(ivloop.V) <= maxv) & ~np.isnan(ivloop.I)
    ifit = ivloop.I[mask]
    vfit = ivloop.V[mask]
    if len(ifit) == 0:
        print('IV data at index {} has no datapoints in the fit range!'.format(ivloop.name))
        return np.nan
    # needs at least a couple different voltage values
    if len(np.unique(vfit)) < 2:
        print('IV data at index {} has only one voltage value'.format(ivloop.name))
        return vfit[0]/np.mean(ifit)
    # Don't do this because it finds the least squared error in V,
    # but V is well known relative to I, so it sometimes does not give a reciprocal fit
    #line = np.polyfit(ifit, vfit, 1)
    line = np.polyfit(vfit, ifit, 1)
    poly = np.polyfit(vfit, ifit, 3)
    r = 1/line[0]
    # another definition could be the linear part of the polynomial
    #r = 1/poly[2]
    ### if you know the leakage resistance, you can correct for it
    #Rleakage = 9.09e8
    #r_corrected = Rleakage*r / (Rleakage - r)
    r_corrected = r
    ###

    Iline = np.polyval(line, ivloop.V)
    Ipoly = np.polyval(poly, ivloop.V)
    return pd.Series({'Resistance': r_corrected, 'Iline':Iline, 'Rfit':r, 'Ipoly':Ipoly})

fitinfo = df.apply(calculate_r, 1)
fitcols = fitinfo.columns
df[fitcols] = fitinfo

# Calculate resistivity assuming uniform cuboid devices
def resistivity(R, w, t):
    # return resistivity in ohm cm
    # w in nm
    # t in nm
    return R * (w * 1e-9)**2 / (t * 1e-9) * 100

df['Resistivity'] = resistivity(df.Resistance, df.width_nm, df.thickness_1)
df['logResistance'] = np.log10(df.Resistance)
df['logResistivity'] = np.log10(df.Resistivity)

sampleidentifiers = ['dep_code', 'sample_number']
samplename_formatter = '{}_{:.0f}'

# This groups measurements by every individual device
deviceidentifiers = ['dep_code', 'sample_number', 'die_rel', 'module', 'device']
deviceid_titleformatter = '{0}_{1:.0f}, die_{2:.0f}, mod {3}, device {4:.0f}'
deviceid_fnformatter = '{0}_{1:.0f}_die{2:.0f}_mod{3}_{4:.0f}'

# This groups by devices which are nominally of the same type
devicetypeidentifiers = ['dep_code', 'sample_number', 'width_nm', 'layer_1', 'thickness_1', 'R_series']
devicetype_titleformatter = '{0}_{1:.0f}, {3}, w={2:.0f}nm, t={4:.0f}nm'
devicetype_fnformatter = '{0}_{1:.0f}_{3}_t{2:.0f}nm_w{4:.0f}nm_Rs{5:.0f}'

df['R_series'] = df.R_series.astype(int)

devicegrp = df.groupby(deviceidentifiers)
typegrp = df.groupby(devicetypeidentifiers)

# Give repeated measurements a unique index
u = devicegrp.groups.keys()
d = {n:0 for n in u}
count = []
for k, id in df[deviceidentifiers].iterrows():
    dkey = tuple(id)
    count.append(d[dkey])
    d[dkey] += 1
df['repeat'] = count


##################### Rearrange columns #############################

# Put identifying columns first.  Could use multiindex..
# put these columns next
nextcolumns = ['Resistance']
columns = df.columns
df = df[deviceidentifiers + nextcolumns + [c for c in columns if c not in deviceidentifiers + nextcolumns]]

df.to_pickle('ivs_with_resistance.pickle')

##################### Select subset of data, do aggregation #############################

####### Collapse any repeated measurements into one resistance value per device #########
# there are different ways you might want to do this.
# you could average them, or pick one based on whatever criteria you want

def selectloop(loops):
    # Simply pick the last measurement
    selection = loops.iloc[-1].name
    reason = 'LastMeasurement'
    return pd.Series({'selection':selection, 'reason':reason})

selections = devicegrp.apply(selectloop)
# New column marks the selected loops
df.loc[:, 'Selected'] = False
df.loc[selections.selection, 'Selected'] = True
# Dunno this doesn't work anymore
#df.loc[selections.selection, 'SelectionReason'] = selections['reason']

# Take just the selected loops
df = df[df.Selected == True]
# This was the sorting before I changed something..
df = df.sort_values(by=deviceidentifiers, ascending=True)

# regroup after selection
devicegrp = df.groupby(deviceidentifiers)
typegrp = df.groupby(devicetypeidentifiers)

# this is just so that some extra columns survive the aggregation.  they can
# not change with devicetype
agg_grp = df.groupby(devicetypeidentifiers + ['layer_2', 'thickness_2', 'dep_temp', 'cr'])

# these dfs have values averaged over devices of same type

# Get mean and std of resistance for each device type
# include log because log(mean()) is not the same as mean(log())
# because they should be closer to lognormal, it's better to do the log first
def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
R_df = agg_grp[['Resistance', 'Resistivity', 'logResistance', 'logResistivity']].agg([np.mean, np.std, percentile(.2), percentile(99.8)])
# Collapse multiindex
R_df.columns = R_df.columns.to_series().str.join('_')
# Reset index
R_df.reset_index(inplace=True)
# still want to use real logscale, so convert log values back to not log
# sorry if this is confusing later..
for k in R_df:
    if k.startswith('log'):
        R_df[k] = 10**R_df[k]
# Sort the devices by roughly decreasing resistance
R_df = R_df.sort_values(['thickness_1', 'width_nm'], ascending=[False, False])


#just 001 module
df0 = R_df[R_df.R_series == 0]


############################## Dataframe is ready, make plots  ########################


###############################################################################
###############################################################################
# Plots that should apply to every sample, (e.g. width, module dependence)
###############################################################################
###############################################################################

def paramplot(df, x, y, parameters, yerr=None, cmap=plt.cm.gnuplot, labelformatter=None,
              sparseticks=True, xlog=False, ylog=False, sortparams=False, paramvals=None,
              ax=None, **kwargs):
    '''
    line plot y vs x, grouping lines by any number of parameters

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


def violinhist(data, x, range=None, bins=50, alpha=.8, color=None, logbin=True, logx=True, ax=None,
               label=None, fixscale=None, sharescale=True, hlines=False, vlines=False, **kwargs):
    '''
    histogram version of violin plot (when there's not a lot of data so the KDE looks weird)
    Can handle log scaling the x-axis, which plt.violinplot cannot do
    widths are automatically scaled, attempting to make them visible and not overlapping
    data should be a list of arrays of values
    kwargs go to plt.bar
    This was pretty difficult to write -- mostly because I want the log ticks..
    TODO: could extend to make a real violin plot by increasing # of bins, adding some gaussian noise to the data (dithering), and doing line plots
    '''
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    # sort data
    order = np.argsort(x)
    x = np.array(x)[order]
    data = [data[o] for o in order]
    if range is None:
        #range = np.percentile(np.concatenate(data), (1,99))
        alldata = np.concatenate(data)
        range = (np.min(alldata), np.max(alldata))

    # values will be converted back to linear scale before plotting
    # so that we can use the log-scale axes
    if logbin:
        data = [np.log(d) for d in data]
        range = np.log(range)
        ax.set_yscale('log')
    if logx:
        x = np.log(x)
        ax.set_xscale('log')

    dx = np.diff(x)
    # Calculate stats
    a = np.array
    n = a([len(d) for d in data])
    means = a([np.mean(d) for d in data])
    medians = a([np.median(d) for d in data])
    maxs = a([np.max(d) for d in data])
    mins = a([np.min(d) for d in data])
    p99s = a([np.percentile(d, 99) for d in data])
    p01s = a([np.percentile(d, 1) for d in data])

    # Calculate the histograms
    hists = []
    for d,xi in zip(data, x):
        edges = np.linspace(*range, bins+1)
        hist, edges = np.histogram(d, bins=edges)
        #if logbin:
            # normalize to account for different bin widths
            # this only needs to be done if we are putting a log binned dataset back onto a linear scale! we are not!
            # hist = hist / np.diff(np.exp(edges))
        hists.append((hist, edges))

    # Figure out how to scale the bin heights
    if fixscale:
        # fixscale overrides everything, useful for manual tuning or if you need to plot many
        # different violinhists on the same plot and want all the bin heights to be consistent
        hists = [(h*fixscale, e) for h,e in hists]
    else:
        # we don't want violins to overlap, and we don't want one to look much bigger than any other
        maxscale = 0.49
        maxamp = np.min(dx) * maxscale
        if sharescale:
            # Usually I will want 1 sample to correspond to the same height everywhere (sharescale)
            # scale all hists by the same factor so that the globally maximum bin reaches maxamp
            maxbin = np.max([h for h,e in hists])
            hists = [(h*maxamp/maxbin, e) for h,e in hists]
        else:
            # scale every hist to maxscale
            hists = [(h*maxamp/np.max(h), e) for h,e in hists]

    # return data to normal scale by overwriting. This is awful.
    if logx:
        x = np.exp(x)
        hists = [(np.exp(hist), edges) for (hist, edges) in hists]
    if logbin:
        hists = [(hist, np.exp(edges)) for (hist, edges) in hists]
        maxs = np.exp(maxs)
        mins = np.exp(mins)
        means = np.exp(means)
        medians = np.exp(medians)
        p99s = np.exp(p99s)
        p01s = np.exp(p01s)
        range = np.exp(range)

    # Plot the histograms
    for (hist, edges), xi in zip(hists, x):
        heights = np.diff(edges)
        if logx:
            ax.barh(edges[:-1], xi*hist - xi, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1, label=label, **kwargs)
            label = None # only label the first one
            ax.barh(edges[:-1], xi/hist - xi, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1, label=label, **kwargs)
        else:
            ax.barh(edges[:-1], hist, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1, label=label, **kwargs)
            label = None # only label the first one
            ax.barh(edges[:-1], -hist, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1, label=label, **kwargs)

    # Plot hlines (why not just a plt.box? I forgot why.)
    if hlines:
        barwidth = np.min(dx * .2)
        midscale = .5
        if logx:
            # I don't understand how I ever ended up with this code..
            #ax.hlines([mins, means ,maxs], x*np.exp(-barwidth) - x, x*np.exp(barwidth) - x, colors=color)
            #ax.hlines(mins, x*np.exp(-barwidth), x*np.exp(barwidth), colors=color)
            #ax.hlines(maxs, x*np.exp(-barwidth), x*np.exp(barwidth), colors=color)
            ax.hlines(p01s, x*np.exp(-barwidth), x*np.exp(barwidth), colors=color)
            ax.hlines(p99s, x*np.exp(-barwidth), x*np.exp(barwidth), colors=color)
            ax.hlines(medians, x*np.exp(-barwidth*midscale), x*np.exp(barwidth*midscale), colors=color)
        else:
            #ax.hlines([mins, means ,maxs], x-barwidth, x+barwidth, colors=color)
            ax.hlines(mins, x-barwidth, x+barwidth, colors=color)
            ax.hlines(maxs, x-barwidth, x+barwidth, colors=color)
            ax.hlines(medians, x-barwidth*.5, x+barwidth*.5, colors=color)
    if vlines:
        if vlines == 'full':
            for xx in x:
                ax.axvline(xx, color=color)
        else:
            #ax.vlines(x, mins, maxs, colors=color)
            ax.vlines(x, p01s, p99s, colors=color)

    # only label the x axis where there are histograms
    ax.xaxis.set_ticks(x)
    ax.xaxis.set_ticklabels(x)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    r0, r1 = range
    if logbin:
        ax.set_ylim(r0, r1)
    else:
        m = (r0 + r1) / 2
        ax.set_ylim((r0-m)*1.05 + m, (r1-m)*1.05 +m)

def violinhist_old(data, x, range, bins=50, alpha=.8, color=None, logbin=True, logx=True, ax=None, label=None, **kwargs):
    # histogram version of violin plot (when there's not a lot of data so the KDE looks weird)
    # data should be a list of arrays of values
    # kwargs go to plt.bar
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
            logrange = [np.log10(r) if r > 0 else 1 for r in range]
            hist, edges = np.histogram(logd, bins=bins, range=logrange)
            edges = 10**edges
            ax.set_yscale('log')
        else:
            hist, edges = np.histogram(d, bins=bins, range=range)
        hist = hist / np.max(hist)
        heights = np.diff(edges)
        rightbar = ax.barh(edges[:-1], aright*hist, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1, label=label, **kwargs)
        # this makes sure all the subsequent colors are the same
        # even if color was initially None
        color = rightbar.get_children()[0].get_facecolor()
        label = None # only label the first one
        ax.barh(edges[:-1], -aleft*hist, height=heights, align='edge', left=xi, color=color, alpha=alpha, linewidth=1, label=label, **kwargs)
    # only label the x axis where there are histograms
    ax.xaxis.set_ticks(x)
    ax.xaxis.set_ticklabels(x)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.vlines(x, *range, color='black', alpha=.3)
    ax.set_ylim(range)

def violinhist_fromdf(df, col, xcol, **kwargs):
    histd = []
    histx = []
    for k, g in df.groupby(xcol):
        histx.append(k)
        histd.append(g[col].values)
    violinhist(histd, histx, **kwargs)

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
    Histogram where the bars are split into colors
    I don't know if this is a good idea or not.
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
            logrange = [np.log10(r) if r > 0 else 1 for r in range]
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

def reference_lines(ax, slope=-2, label='Area scaling'):
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
    ax.plot(xlims, (y[-1], y[-1] + y[-1]/xmin**slope *(xmax**slope - xmin**slope)), '--', alpha=.2, color='black', label=label)
    # Put the limits back
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

def engformatter(axis='y', ax=None):
    if ax is None:
        ax = plt.gca()
    if axis.lower() == 'x':
        axis = ax.xaxis
    else:
        axis = ax.yaxis
    axis.set_major_formatter(mpl.ticker.EngFormatter())


def plot_iv(ivloop, **kwargs):
    plt.plot(1e3 * ivloop.V, 1e6 * ivloop.I, **kwargs)
    plt.xlabel('Voltage [mV]')
    plt.ylabel('Current [$\mu$A]')

############### IV Loops/fits for each individual measurement ################
subdir='IV_Loop_fits'
for k, g in df.groupby(deviceidentifiers):
    filename = deviceid_fnformatter.format(*k) + '.png'
    filepath = os.path.join(plotdir, subdir, filename)
    if (not os.path.isfile(filepath)) or overwrite_individuals:
        plt.figure()
        for row, loop in g.iterrows():
            if loop.Selected:
                plot_iv(loop, marker='.', label='Measurement')
                plt.plot(1e3 * loop.V, 1e6 * loop.Iline, '--', color='red', alpha=.6, label='Resistance Fit: {:.1e}$\Omega$'.format(loop['Resistance']))
                plt.plot(1e3 * loop.V, 1e6 * loop.Ipoly, '--', color='blue', alpha=.6, label='Polyfit(3): ...')
                # Use the same name as the datafile
                #filename = os.path.splitext(os.path.split(loop.filepath)[1])[0]
            else:
                # plot non selected loops transparently
                plot_iv(loop, marker='.', alpha=.2, label='{} (R={:.1e}$\Omega$)'.format(loop['repeat'], loop['Resistance']))
        title = deviceid_titleformatter.format(*k)
        plt.title(title)
        plt.legend()
        writefig(filename, subdir)
        plt.close()

############### IV Loops by device type  ###############
for k, g in df.groupby(devicetypeidentifiers):
    plt.figure()
    filename = devicetype_fnformatter.format(*k)
    title = devicetype_titleformatter.format(*k)
    for row, loop in g.iterrows():
        plot_iv(loop, marker='.', label='die {}, {}_{}'.format(loop.die_rel, loop.module, loop.device))
    plt.title(title)
    if len(g) < 10:
        plt.legend()
    # Strange attempt to cut out outliers
    range = np.nanmean(np.abs(np.nanpercentile(np.hstack(g.I).flatten(), (5,95))*1.2*1e6))
    #range = np.clip(range, -10e-3, 10e-3)
    plt.ylim(-range, range)
    writefig(filename, subdir='IV_Loop_Device_Type')
    plt.close()


############## log resistance histograms by device type #############
for k, g in df.groupby(devicetypeidentifiers):
    fig, ax = plt.subplots()
    title = devicetype_titleformatter.format(*k)
    plt.title(title)
    loghist(g.Resistance, bins=50, alpha=.8)
    engformatter('x')
    filename = devicetype_fnformatter.format(*k)
    writefig(filename, subdir='R_hists_device_type')
    plt.close()


############## Resistivity histograms ###########
# colored by device width and stacked

for k,g in df[df.module_num == 1].groupby(sampleidentifiers):
    fig, ax = plt.subplots()
    grouped_hist(g, 'Resistivity', groupby='width_nm', logx=True, ax=ax)
    ax.set_xlabel('Resistivity [Ohm cm]')
    title = samplename_formatter.format(*k)
    ax.set_title(title)
    filename = title
    writefig(filename, subdir='Resistivity_hists')


####################################  R vs width/area  #############################
# log-log R vs width
# For zero series resistance devices

# Individual samples (same color):
# range changes with sample
# labeled only by sample name, later we will label by the parameters that were varied
for k,g in df0.groupby(sampleidentifiers):
    fig, ax = plt.subplots()
    # Do violinhist
    # get whole dataframe corresponding to these sampleidentifiers
    range = g['logResistance_percentile_0.2'].min(), g['logResistance_percentile_99.8'].max()
    dfg = df[df[sampleidentifiers].apply(tuple, 1) == k]
    violinhist_fromdf(dfg, 'Resistance', 'width_nm', bins=60, range=range, logbin=True, logx=True, alpha=.7, ax=ax, hlines=True, vlines=True)

    ax.set_xlabel('Device width [nm]')
    ax.set_ylabel('Resistance [$\Omega$]')

    reference_lines(ax, slope=-2)
    engformatter('y')
    title = samplename_formatter.format(*k)
    ax.set_title(title)
    filename = title
    writefig(filename, subdir='logResistance_vs_logwidth_individual')

    # plot mean values on top
    # error bar is miscalculated somehow.  not needed here anyway
    fig, ax = paramplot(g, y='logResistance_mean', yerr='logResistance_std', x='width_nm', parameters=sampleidentifiers,
                        xlog=True, ylog=True, ax=ax)
    # only one sample, no legend needed
    ax.legend_.set_visible(False)
    engformatter('y')
    ax.set_xlabel('Device width [nm]')
    ax.set_ylabel('Resistance [$\Omega$]')

    writefig(filename+'_with_meanline', subdir='logResistance_vs_logwidth_individual')
    plt.close()

# Resistivity vs width
# range changes with sample
# labeled only by sample name, later we will label by the parameters that were varied
for k,g in df0.groupby(sampleidentifiers):
    fig, ax = plt.subplots()
    # Do violinhist
    # get whole dataframe corresponding to these sampleidentifiers
    range = g['logResistivity_percentile_0.2'].min(), g['logResistivity_percentile_99.8'].max()
    dfg = df[df[sampleidentifiers].apply(tuple, 1) == k]

    violinhist_fromdf(dfg, 'Resistivity', 'width_nm', bins=60, range=range, logbin=True, logx=True, alpha=.7, ax=ax, hlines=True, vlines=True)

    ax.set_xlabel('Device width [nm]')
    ax.set_ylabel('Resistivity [$\Omega$ cm]')
    ax.grid(axis='y')

    title = samplename_formatter.format(*k)
    ax.set_title(title)
    filename = title
    writefig(filename, subdir='logResistivity_vs_logwidth_individual')

    # plot mean values on top
    # error bar is miscalculated somehow.  not needed here anyway
    fig, ax = paramplot(g, y='logResistivity_mean', yerr='logResistivity_std', x='width_nm', parameters=sampleidentifiers,
                        xlog=True, ylog=True, ax=ax)
    # only one sample, no legend needed
    ax.legend_.set_visible(False)
    ax.set_xlabel('Device width [nm]')
    ax.set_ylabel('Resistivity [$\Omega$ cm]')

    writefig(filename+'_with_meanline', subdir='logResistivity_vs_logwidth_individual')
    plt.close()


# dep_codes usually exist to denote a particular experiment where a parameter
# or two was varied
# There should always be plots comparing samples with the same depcode with
# respect to any parameter variations

# Additionally, it would be nice to detect similar samples with one parameter
# difference and campare those even if they do not belong to the same dep_code

# These should be handled separately, because we don't want some weird sample
# interfering with the plots of the dep_code variations, which are most
# important


# these won't use the aggregated values
# only robust way is to look at the distributions directly
# because resistance outliers are a bitch
for dep_code in df.dep_code.drop_duplicates():
    d = df[df.dep_code == dep_code][df.module_num == 1]
    #R_d = R_df[R_df.dep_code == dep_code]
    #d0 = df0[df0.dep_code == dep_code]
    print(dep_code)

    #what is varied?
    look_for = ['thickness_1', 'thickness_2', 'dep_temp', 'cr']
    varied = []
    for col in look_for:
        if col in d:
            if len(d[col].drop_duplicates()) > 1:
                varied.append(col)

    if not any(varied):
        # If nothing appears to be varied, just plot vs sample number
        varied = ['sample_number']

    print(varied)

    for var in varied:
        uniq = d[var].drop_duplicates()

        # outer group (other varied parameters)
        if len(varied) > 1:
            outer_grpby = [v for v in varied if v != var]
            if len(outer_grpby) == 1:
                outer_grpby = outer_grpby[0]
            outergrp = d.groupby(outer_grpby)
        else:
            # no outer group (only var was varied)
            outer_grpby = None
            outergrp = ((None, d),)


        # violin R vs w grouping by varied parameter
        # all samples have a w variation so we know this can always be done
        for k,g in outergrp:
            fig, ax = plt.subplots()
            innergrp = g.groupby(var)
            if len(innergrp) > 1:
                # Resistance vs w
                range = np.nanpercentile(g['Resistance'], (.5, 99.5))
                for kk,gg in g.groupby(var):
                    violinhist_fromdf(gg, 'Resistance', 'width_nm', bins=80, range=range, logbin=True, logx=True, alpha=.7, ax=ax, label=kk)
                engformatter('y')
                ax.set_ylabel('Resistance [$\Omega$]')
                ax.set_xlabel('Device width [nm]')
                ax.legend(title=var, loc=0)
                reference_lines(ax, -2)
                if outer_grpby is None:
                    title = dep_code
                else:
                    title = f'{dep_code} {outer_grpby}={k}'
                ax.set_title(title)
                plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                filename = title.replace(' ','_').replace('.', 'p').replace('p0', '')
                writefig(filename, subdir='logResistance_vs_logwidth_variations')
                plt.close()

                # Resistivity vs w
                fig, ax = plt.subplots()
                range = np.nanpercentile(g['Resistivity'], (.5, 99.5))
                for kk,gg in g.groupby(var):
                    violinhist_fromdf(gg, 'Resistivity', 'width_nm', bins=80, range=range, logbin=True, logx=True, alpha=.7, ax=ax, label=kk)
                ax.set_ylabel('Resistivity [$\Omega$ cm]')
                ax.set_xlabel('Device width [nm]')
                ax.legend(title=var, loc=0)
                ax.set_title(title)
                plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                writefig(filename, subdir='logResistivity_vs_logwidth_variations')
                plt.close()


        # Violin with x the varied parameter (i.e. thickness)
        # is the thing varied numerical? otherwise just skip violining it
        if isinstance(d[var].iloc[0], Number):
            # violin vs varied parameter
            if var == 'thickness_1':
                logx = True
            else:
                logx = False

            for k,g in outergrp:
                range = np.nanpercentile(g['Resistance'], (.5, 99.5))

                if outer_grpby is None:
                    title = dep_code
                else:
                    title = f'{dep_code} {outer_grpby}={k} vs {var}'
                filename = title.replace(' ','_').replace('.', 'p').replace('p0', '')

                ## Resistance vs parameter
                fig, ax = plt.subplots()
                # One plot with all widths included, all the same color
                violinhist_fromdf(g, 'Resistance', var, bins=80, range=range, logbin=True, logx=logx, alpha=.7, ax=ax)
                ax.set_xlabel(var)
                ax.set_ylabel('Resistance [$\Omega$]')
                engformatter('y')
                if var =='thickness_1':
                    reference_lines(ax, 1)
                plt.title(title)
                plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                writefig(filename, subdir='logResistance_vs_param')
                plt.close()

                # plots with widths separated by color
                fig, ax = plt.subplots()
                for kk,gg in g.groupby('width_nm'):
                    violinhist_fromdf(gg, 'Resistance', var, bins=80, range=range, logbin=True, logx=logx, alpha=.7, ax=ax, label=kk)
                ax.legend(title='Width [nm]', loc=0)
                ax.set_xlabel(var)
                ax.set_ylabel('Resistance [$\Omega$]')
                engformatter('y', ax=ax)
                if var =='thickness_1':
                    reference_lines(ax, 1)
                #if outer_grpby != var
                plt.title(title)
                plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                writefig(filename+'_width_separated', subdir='logResistance_vs_param')
                plt.close()


                # and separate plots for each width
                for kk,gg in g.groupby('width_nm'):
                    fig, ax = plt.subplots()
                    violinhist_fromdf(gg, 'Resistance', var, bins=80, range=range, logbin=True, logx=logx, alpha=.7, ax=ax, label=kk)
                    ax.legend(title='Width [nm]', loc=0)
                    ax.set_xlabel(var)
                    ax.set_ylabel('Resistance [$\Omega$]')
                    engformatter('y', ax=ax)
                    if var =='thickness_1':
                        reference_lines(ax, 1)
                    #if outer_grpby != var
                    plt.title(title)
                    plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                    writefig(filename+f'_width_{kk:.0f}', subdir='logResistance_vs_param')
                    plt.close()


                ## Resistivity vs parameter
                range = np.nanpercentile(g['Resistivity'], (.5, 99.5))

                ## Resistance vs parameter
                # One plot with all widths included, all the same color

                fig, ax = plt.subplots()
                violinhist_fromdf(g, 'Resistivity', var, bins=80, range=range, logbin=True, logx=logx, alpha=.7, ax=ax)
                ax.set_xlabel(var)
                ax.set_ylabel('Resistivity [$\Omega$ cm]')
                plt.title(title)
                plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                writefig(filename, subdir='logResistivity_vs_param')
                plt.close()


                # plots with widths separated by color
                fig, ax = plt.subplots()
                for kk,gg in g.groupby('width_nm'):
                    violinhist_fromdf(gg, 'Resistivity', var, bins=80, range=range, logbin=True, logx=logx, alpha=.7, ax=ax, label=kk)
                ax.legend(title='Width [nm]', loc=0)
                ax.set_xlabel(var)
                ax.set_ylabel('Resistivity [$\Omega$ cm]')
                plt.title(title)
                plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                writefig(filename+'_width_separated', subdir='logResistivity_vs_param')
                plt.close()

                # and separate plots for each width
                for kk,gg in g.groupby('width_nm'):
                    fig, ax = plt.subplots()
                    violinhist_fromdf(gg, 'Resistivity', var, bins=80, range=range, logbin=True, logx=logx, alpha=.7, ax=ax, label=kk)
                    ax.legend(title='Width [nm]', loc=0)
                    ax.set_xlabel(var)
                    ax.set_ylabel('Resistivity [$\Omega$ cm]')
                    #if outer_grpby != var
                    plt.title(title)
                    plt.grid(True, 'major', 'x', color='black', linestyle='solid', alpha=.9)
                    writefig(filename+f'_width_{kk:.0f}', subdir='logResistivity_vs_param')
                    plt.close()


# color by module?

# R vs 1/width?
# Has intercept = series resistance

# Shorts and opens? hard to define in a general way

# Other modules?

############## End plots
plt.show()
