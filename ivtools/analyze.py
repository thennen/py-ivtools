""" Functions for doing data analysis on IV data """

# Local imports
from . import plot as ivplot

from functools import wraps
import numpy as np
from itertools import groupby
from scipy import signal
from numbers import Number
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
import sys

# TODO make some functions that index/iterate rows of dataframe AND list, because they have different syntax
#      this would avoid testing for the datatype in many different parts of the code

def ivfunc(func):
    '''
    Decorator which allows the same function to be used on a single loop, as
    well as a container of loops.

    Decorated function should take a single loop and return anything
    Then this function will also take multiple loops, and return a list/dataframe of the outputs
    The point is to make the analysis functions easier to write, and not repeat a bunch of code

    Handles dicts and pd.Series for singular input data, as well as "list of dict" and DataFrames for multiple input data
    An attempt is made to return the most reasonable type, given the input and output types

    Some preliminary testing indicates that operating on dataframes can be much slower (~100x) than list of dicts.

    If any of the arguments are instances of "paramlist", this tells ivfunc to also index into this list when
    calling the wrapped function, so that you can pass a list of parameters to use for each data row.

    If you pass as an argument a function wrapped with the paramfunc function, that function will get called on each of
    the data to determine the argument individually.
    (Maybe we don't need to wrap functions, and assume this is the intent whenever any function is passed..)
    '''
    def paramtransform(param, i, data):
        if type(param) == paramlist:
            # Index param if it is a paramlist. Otherwise, don't.
            return param[i]
        #elif hasattr(param, '__call__'):
        elif hasattr(param, '__name__') and (param.__name__ == 'paramfunc'):
            # Call function with data as argument, and use the return value as the parameter
            return param(data)
        else:
            return param
    @wraps(func)
    def func_wrapper(data, *args, **kwargs):
        dtype = type(data)
        ###################################
        ###  IF A DATAFRAME IS PASSED   ###
        ###################################
        if dtype == pd.DataFrame:
            # Apply the function to the columns of the dataframe
            # Sadly, the line below will error if your function returns an array, because pandas.
            # Although it can be significantly faster if your function does not return an array..
            # return data.apply(func, axis=1, args=args, **kwargs)
            # Since we don't already know what type the wrapped function will return,
            # we need to explicitly loop through the dataframe rows, and store the results in an intermediate list
            resultlist = []
            for i, (rownum, row) in enumerate(data.iterrows()):
                #resultlist.append(func(row, *args, **kwargs))
                result = func(row, *[paramtransform(arg, i, row) for arg in args],
                              **{k:paramtransform(v, i, row) for k,v in kwargs.items()})
                resultlist.append(result)
            ### Decide how to return the values based on the datatype that the wrapped function returned
            if type(resultlist[0]) in (pd.Series, dict):
                # Each row returns a dict or series
                if type(resultlist[0][list(resultlist[0].keys())[0]]) is dict:
                    # Each dict probably just contains another dict.
                    # Return a dataframe with multiindex columns
                    # It hurts my brain to think about how to accomplish this, so for now it does the same
                    df_out = pd.DataFrame(resultlist)
                    df_out.index = data.index
                else:
                    df_out = pd.DataFrame(resultlist)
                    df_out.index = data.index
                return df_out
            elif (type(resultlist[0]) is list):
                if (type(resultlist[0][0]) is dict):
                    # Each row returns a list of dicts, stack the lists into new dataframe
                    # Mainly used for splitting loops, so that everything stays flat
                    # Maybe return a panel instead?
                    # Index will get reset ...
                    # Unless ...
                    index = np.repeat(data.index, [len(sublist) for sublist in resultlist])
                    return pd.DataFrame([item for sublist in resultlist for item in sublist], index=index)
                elif isinstance(resultlist[0][0], Number):
                    # Each row returns a list of numbers
                    # Make dataframe
                    df_out = pd.DataFrame(resultlist)
                    df_out.index = data.index
                    return df_out
            elif all([r is None for r in resultlist]):
                # If ivfunc returns nothing, return nothing
                return None
            # For all other cases
            # Keep the index the same!
            series_out = pd.Series(resultlist)
            series_out.index = data.index
            return series_out
        #######################################
        ### IF A LIST (OF DICTS) IS PASSED  ###
        #######################################
        elif dtype == list:
            # Assuming it's a list of iv dicts
            #resultlist = [func(d, *args, **kwargs) for d in data]
            resultlist = []
            for i, d in enumerate(data):
                result = func(d, *[paramtransform(arg, i, d) for arg in args],
                              **{k:paramtransform(v, i, d) for k,v in kwargs.items()})
                resultlist.append(result)
            if (type(resultlist[0]) is list):
                if (type(resultlist[0][0]) is dict):
                    # Each func(dict) returns a list of dicts, stack the lists
                    return [item for sublist in resultlist for item in sublist]
                elif isinstance(resultlist[0][0], Number):
                    # Each iv dict returns a list of numbers
                    # "Unpack" them
                    return list(zip(*resultlist))
            elif all([r is None for r in resultlist]):
                # If ivfunc returns nothing, return nothing
                return None
            # For all other return types
            return resultlist
        elif dtype is pd.Series:
            # It's just one IV Series
            # If it returns a dict, convert it back to a series
            result = func(data, *args, **kwargs)
            if type(result) is dict:
                return(pd.Series(result))
            else:
                return result
        elif dtype in (dict,):
            # It's just one IV dict
            return(func(data, *args, **kwargs))
        else:
            print('ivfunc did not understand the input datatype {}'.format(dtype))
    return func_wrapper


class paramlist(list):
    # Only a class so that ivfunc can know what you want to do with it
    # Which is pass a list of parameters to use for each individual loop
    pass


def paramfunc(func):
    # Wraps a function to identify itself to ivfunc as a function to be called on the data to determine input parameters
    func.__name__ = 'paramfunc'
    return func

'''
Functions that return a new value/new array per IV loop should just return that value
Functions that modify the IV data should return a copy of the entire input structure
'''

#@ivfunc
def find_data_arrays(data):
    '''
    Try to find the names of arrays in a dataframe that have the same length
    If the keys 'I' or 'V' exist, use the arrays that have the same length as those.
    If those keys are not present, and there are arrays different sizes, choose the one
    that has the most arrays of that size
    '''
    # Get lengths of all arrays
    #arraykeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    arraykeys = [k for k,v in data.items() if (type(v) == np.ndarray)]
    lens = [len(data[a]) for a in arraykeys]
    # If 'I' or 'V' is in array keys, use that length array
    if 'I' in arraykeys:
        Ilen = len(data['I'])
        return [ak for ak,l in zip(arraykeys, lens) if l == Ilen]
    elif 'V' in arraykeys:
        Vlen = len(data['I'])
        return [ak for ak,l in zip(arraykeys, lens) if l == Vlen]
    else:
        Alen = max(lens, key=lens.count)
        return [ak for ak,l in zip(arraykeys, lens) if l == Alen]


@ivfunc
def diffiv(data, stride=1, columns=None):
    ''' Calculate first difference of arrays.'''
    if columns is None:
        columns = find_data_arrays(data)
    arrays = [data[c] for c in columns]
    diffarrays = [ar[stride:] - ar[:-stride] for ar in arrays]
    dataout = {c:diff for c,diff in zip(columns, diffarrays)}
    add_missing_keys(data, dataout)
    return dataout


### Determine "switching" thresholds ###
# Basically use some criteria to identify a single point on the IV curve
# having the data point index is useful, so there's a separate function for doing that
# TODO: rename things (don't use the word threshold) for the general concept of choosing a datapoint
# select?

@ivfunc
def select_by_maxdiff(data, column='I', stride=1):
    '''
    Find switching thresholds by finding the maximum differences.
    Return index of the threshold datapoint
    '''
    diff = data[column][stride:] - data[column][:-stride]
    argmaxdiff = np.argmax(diff)

    return argmaxdiff

@ivfunc
def thresholds_bydiff(data, stride=1):
    ''' Find switching thresholds by finding the maximum differences. '''
    argmaxdiffI = select_by_maxdiff(data, column='I', stride=stride)

    return indexiv(data, argmaxdiffI)


@ivfunc
def select_by_crossing(data, column='I', thresh=0.5, direction=True):
    '''
    Determine threshold datapoint index the first to cross a certain value
    return the whole datastructure with the data arrays replaced by the value at the threshold point
    If data in column is decreasing, specify direction=False
    Return a new iv structure with the values at the threshold in place of data arrays
    '''
    # Find threshold
    if direction:
        threshside = np.where(data[column] >= thresh)
    else:
        threshside = np.where(data[column] <= thresh)


    if any(threshside[0]):
        index = threshside[0][0]
    else:
        index = np.nan

    return index


@ivfunc
def threshold_bycrossing(data, column='I', thresh=0.5, direction=True):
    '''
    Determine threshold datapoint by the first to cross a certain value
    return the whole datastructure with the data arrays replaced by the value at the threshold point
    If data in column is decreasing, specify direction=False
    Return a new iv structure with the values at the threshold in place of data arrays
    '''
    index = select_by_crossing(data, column=column, thresh=thresh, direction=direction)
    return indexiv(data, index)


@ivfunc
def select_by_derivative(data, threshold=None, debug=False):
    '''
    Selects the first point where the derivative dI/dV crosses the threshold
    (Where the sign of dI/dV - threshold changes for the first time)
    normalizing the threshold value by resistance is a good idea. (e.g. pass threshold=100 / R)
    '''
    # I don't want the real dI/dV, because dV can be close to zero, or negative on a transition due to source resistance.
    # I will assume the array is ~equally spaced in voltage and use the mean value for dV
    # slope will therefore be considered negative for decreasing voltages
    dV = np.mean(np.abs(np.diff(data['V'])))
    dI = np.diff(data['I'])

    if threshold is None:
        # TODO: guess a threshold based on histogram of dV/dI, maybe just a percentile or something
        # TODO: Or try a fit for resistance and use a standard value normalized by resistance
        raise Exception('Autothreshold not implemented!')

    sideofthresh = np.sign(dI/dV - threshold)
    index = np.where(np.diff(sideofthresh))
    if any(index[0]):
        # If side of threshold ever changed, use the first time
        # Do I need to add one?
        index = index[0][0] + 1
    else:
        index = np.nan

    # Using values of debug to show different figures, because I couldn't think of a nice way to make both figures
    # The problem is that this function gets called separately for each row,
    # and the function doesn't know if it has already been called on previous rows
    # So it doesn't know to add the lines to the appropriate figure.

    # Don't make a figure because this function gets called for every row.  Need to make an empty figure manually before calling.
    ax = plt.gca()
    if debug == 1:
        #fig, ax = plt.subplots()
        ax.plot(dI/dV)
        xmin, xmax = ax.get_xlim()
        ax.hlines(threshold, xmin, xmax, alpha=.3, linestyle='--')
        ax.set_xlim(xmin, xmax)
    elif debug == 2:
        ivplot.plotiv(data, ax=ax)
        ax.scatter(data['V'][index], data['I'][index])

    return index


@ivfunc
def threshold_byderivative(data, threshold=None, interp=False, debug=False):
    '''
    Find thresholds by derivative method
    Selects the first point where the derivative crosses the threshold
    (Where the sign of dI/dV - threshold changes for the first time)
    normalizing the threshold value by resistance is a good idea. (e.g. pass threshold=100 / R)
    If interp=False, will return the nearest point in the actual dataset
    interp=True is not implemented yet.
    '''
    if interp:
        raise Exception('Interpolation not implemented')
    else:
        index = select_by_derivative(data, threshold=threshold, debug=debug)

    dataout = indexiv(data, index)


    return dataout

@ivfunc
def select_from_func():
    ''' Select data index by function '''
    pass


### End determine switching thresholds ###


'''
@ivfunc
def thresholds_byval(data, value):
    pindex(data, 'I', value)
'''

@ivfunc
def moving_avg(data, window=5, columns=None):
    ''' Smooth data arrays with moving avg '''
    if columns is None:
        columns = find_data_arrays(data)
    else:
        columns = [c for c in columns if c in data.keys()]
    arrays = [data[c] for c in columns]
    lens = [len(ar) for ar in arrays]
    if not all([l - lens[0] == 0 for l in lens]):
        raise Exception('Arrays to be smoothed have different lengths!')
    if lens[0] == 0:
        raise Exception('No data to smooth')
    #weights = np.repeat(1.0, window)/window
    #smootharrays = [np.convolve(ar, weights, 'valid') for ar in arrays]
    smootharrays = [smooth(ar, window) for ar in arrays]
    dataout = {c:sm for c,sm in zip(columns, smootharrays)}
    add_missing_keys(data, dataout)
    dataout['smoothing'] = window
    return dataout


@ivfunc
def medfilt(data, window=5, columns=('I', 'V')):
    ''' Smooth data arrays with moving median '''
    if columns is None:
        columns = find_data_arrays(data)
    arrays = [data[c] for c in columns]
    lens = [len(ar) for ar in arrays]
    if not all([l - lens[0] == 0 for l in lens]):
        raise Exception('Arrays to be smoothed have different lengths!')
    if lens[0] == 0:
        raise Exception('No data to smooth')
    smootharrays = [signal.medfilt(ar, window) for ar in arrays]
    dataout = {c:sm for c,sm in zip(columns, smootharrays)}
    add_missing_keys(data, dataout)
    return dataout


@ivfunc
def decimate(data, factor=5, columns=('I', 'V')):
    ''' Decimate data arrays '''
    if columns is None:
        columns = find_data_arrays(data)
    arrays = [data[c] for c in columns]
    lens = [len(ar) for ar in arrays]
    if not all([l - lens[0] == 0 for l in lens]):
        raise Exception('Arrays to be decimated have different lengths!')
    if lens[0] == 0:
        raise Exception('No data to decimate')
    decarrays = [signal.decimate(ar, factor, zero_phase=True) for ar in arrays]
    dataout = {c:dec for c,dec in zip(columns, decarrays)}
    add_missing_keys(data, dataout)
    if 'downsampling' in dataout:
        dataout['downsampling'] *= factor
    else:
        dataout['downsampling'] = factor
    return dataout


@ivfunc
def smoothimate(data, window=10, factor=2, passes=1, columns=None):
    ''' Smooth with moving avg and then decimate the data'''
    if columns is None:
        columns = find_data_arrays(data)
    arrays = [data[c] for c in columns]
    lens = [len(ar) for ar in arrays]
    if not all([l - lens[0] == 0 for l in lens]):
        raise Exception('Arrays to be smoothimated have different lengths!')
    if lens[0] == 0:
        raise Exception('No data to smooth')
    dataout = {}
    decarrays = arrays
    dtypes = [type(ar[0]) for ar in arrays]
    for _ in range(passes):
        smootharrays = [smooth(ar, window) for ar in decarrays]
        # After all that work to keep the same datatype, signal.decimate converts them to float64
        # I will ignore the problem for now and just convert back in the end...
        # IIR filter has really bad step response!
        #decarrays = [signal.decimate(ar, factor, zero_phase=True) for ar in smootharrays]
        # FIR filter seems more appropriate
        #decarrays = [signal.decimate(ar, factor, type='fir', n=30, zero_phase=True) for ar in smootharrays]
        # But I see no reason not to simply downsample the array
        if factor > 1:
            decarrays = [ar[::factor] for ar in smootharrays]
        else:
            decarrays = smootharrays
    for c, ar, dtype in zip(columns, decarrays, dtypes):
        if dtype == np.float64:
            # Datatype was already float64, don't convert float64 to float64
            dataout[c] = ar
        #elif dtype == np.int8:
            # Maybe we should allow low resolution data (like scope samples) turn high resolution when smoothed
            #dataout[c] = ar
        else:
            # Convert back to original data type (like float32)
            dataout[c] = dtype(ar)
    add_missing_keys(data, dataout)
    dataout['downsampling'] = factor
    dataout['smoothing'] = window
    dataout['smoothimate_passes'] = passes
    return dataout


@ivfunc
def maketimearray(data):
    # TODO: need to account for any possible downsampling!
    # Don't know what data columns exist
    columns = find_data_arrays(data)
    t = np.arange(len(data[columns[0]])) * 1/data['sample_rate']
    if 'downsampling' in data:
        t *= data['downsampling']
    return t


@ivfunc
def indexiv(data, index):
    '''
    Index all the data arrays inside an iv loop container at once.
    Index can be anything that works with np array __getitem__
    if index is np.nan, return np.nan
    '''
    colnames = find_data_arrays(data)

    if hasattr(index, '__call__'):
        # If index is a function, call it on the data
        index = index(data)

    isarray = hasattr(index, '__iter__')
    if not isarray and np.isnan(index):
        # If it's just a plain old nan
        dataout = {c:np.nan for c in colnames}
    else:
        if not isarray and type(index) is not int:
            # Maybe you passed a float, which will error.
            index = int(index)
        dataout = {c:data[c][index] for c in colnames}

    add_missing_keys(data, dataout)
    return dataout


@ivfunc
def sliceiv(data, stop=None, start=0, step=1):
    '''
    Slice all the data arrays inside an iv loop container at once.
    start, stop can be functions that take the iv loop as argument
    if those functions return nan, start defaults to 0 and stop to -1
    '''
    slicekeys = find_data_arrays(data)
    if callable(start):
        start = start(data)
        if np.isnan(start): start = 0
    if callable(stop):
        stop = stop(data)
        if np.isnan(stop): stop = -1
    dataout = {}
    def int_or_none(ind):
        return None if ind is None else int(ind)
    for sk in slicekeys:
        # Apply the filter to all the relevant items
        dataout[sk] = data[sk][slice(int_or_none(start), int_or_none(stop), int_or_none(step))]
    add_missing_keys(data, dataout)
    return dataout


@ivfunc
def slicefraction(data, stop=1/2, start=0, step=1):
    '''
    Slice all the data arrays inside an iv loop container at once.
    start and stop point given as fraction of data length
    '''
    slicekeys = find_data_arrays(data)
    lendata = len(data[slicekeys[0]])
    dataout = {}
    for sk in slicekeys:
        # Slice all the relevant arrays
        dataout[sk] = data[sk][slice(int(start * lendata), int(stop * lendata), int(step))]
    add_missing_keys(data, dataout)
    return dataout


@ivfunc
def split_by_crossing(data, column='V', thresh=0, direction=None, hyspts=1):
    '''
    Split loops into multiple loops, by threshold crossing
    Only implemented V threshold crossing
    return list of input type
    Noisy data is hard to split this way
    hyspts will require that on a crossing, the value of column was above/below threshold hyspts ago
    set it to less than half of the minimum loop length

    Hitting the threshold value and turning around counts as crossing.

    If direction is None, will trigger on either rising or falling edge
    If direction is True, will trigger on the rising edge
    If direction is False, will trigger on the falling edge

    # TODO: find the indices in a separate function.  Could be useful.
    '''
    # V threshold crossing
    side = data[column] >= thresh
    crossings = np.diff(np.int8(side))
    if direction is None:
        # Trigger at either rising or falling edege
        # Determined by the first datapoint which is on one side of the threshold,
        # and hyspts ago was on the other side of the threshold

        # There's a flaw in the logic here.  Right now:
        # trigger point needs to be one which JUST crossed the threshold 1 pt ago
        # AND was on the other side hyspts ago
        # What we need is the first point which is on side A and was on side B hyspts ago
        # but this is hard because it can result in triggers that are clustered
        # we need the first point of each cluster, so need to define cluster length..

        # Probably there's a nice single line way to do this but I'm in a hurry!
        trigger1 = np.where((crossings[hyspts-1:] == 1) & (side[:-hyspts] == False))[0] + hyspts
        trigger2 = np.where((crossings[hyspts-1:] == -1) & (side[:-hyspts] == True))[0] + hyspts
        trigger = np.sort(np.concatenate((trigger1, trigger2)))
    elif direction:
        trigger = np.where((crossings[hyspts-1:] == 1) & (side[:-hyspts] == False))[0] + hyspts
    else:
        trigger = np.where((crossings[hyspts-1:] == -1) & (side[:-hyspts] == True))[0] + hyspts
    # Put the endpoints in
    trigger = np.concatenate(([0], trigger, [len(data['V'])]))
    # Delete triggers that are too close to each other
    # This is not perfect.
    trigthresh = np.diff(trigger) > hyspts
    trigmask = np.insert(trigthresh, 0, 1)
    trigger = trigger[trigmask]

    outlist = []
    splitkeys = find_data_arrays(data)
    for i, j in zip(trigger[:-1], trigger[1:]):
        splitloop = {}
        for k in splitkeys:
            splitloop[k] = data[k][i:j]
        add_missing_keys(data, splitloop)
        outlist.append(splitloop)

    return outlist


@ivfunc
def splitbranch(data, columns=None, dupe_endpoints=True, make_increasing=False):
    '''
    Split a loop into two branches
    Assumptions are that loop starts at intermediate V (like zero), goes to one extremum, to another extremum, then back to zero.
    Can also just go to one extreme and back to zero
    Not sure how to extend to splitting multiple loops.  Should it return interleaved branches or two separate dataframes/lists?
    right now it returns one dataframe of interleaved branches
    '''
    if columns is None:
        columns = find_data_arrays(data)
    imax = np.argmax(data['V'])
    imin = np.argmin(data['V'])
    firstextreme = min(imax, imin)
    secondextreme = max(imax, imin)
    start = data['V'][0]
    firstxval = data['V'][firstextreme]
    secondxval = data['V'][secondextreme]
    pp = abs(firstxval - secondxval)
    branch1 = {}
    branch2 = {}

    # Determine if loop goes to two extremes or just one
    # Is the start value close to one of the extremes?
    if (abs(start - firstxval) < .01 * pp):
        singleextreme = secondextreme
    elif (abs(start - firstxval) < .01 * pp):
        singleextreme = firstextreme
    else:
        singleextreme = None

    if singleextreme is None:
        # Assume there are two extremes..
        for c in columns:
            if dupe_endpoints:
                branch1[c] = np.concatenate((data[c][secondextreme:], data[c][:firstextreme + 1]))
                branch2[c] = data[c][firstextreme:secondextreme + 1]
            else:
                branch1[c] = np.concatenate((data[c][secondextreme:], data[c][:firstextreme]))
                branch2[c] = data[c][firstextreme:secondextreme]
    else:
        # Only one extreme
        for c in columns:
            if dupe_endpoints:
                branch1[c] = data[c][:singleextreme + 1]
                branch2[c] = data[c][singleextreme:]
            else:
                branch1[c] = data[c][:singleextreme]
                branch2[c] = data[c][singleextreme:]

    if make_increasing:
        for c in columns:
            if firstextreme > start:
                branch2[c] = branch2[c][::-1]
            else:
                branch1[c] = branch1[c][::-1]

    add_missing_keys(data, branch1)
    add_missing_keys(data, branch2)

    return [branch1, branch2]


def split_updown():
    ''' dunno I am sick of writing updown = splitbranch(data), up =updown[::2], down=updown[1::2]'''
    pass


@ivfunc
def splitiv(data, nloops=None, nsamples=None, indices=None, dupe_endpts=True):
    '''
    Split data into individual loops, specifying somehow the length of each loop
    if you pass nloops, it splits evenly into n loops.
    if you pass nsamples, it makes each loop have that many samples (except possibly the last one)
    pass nsamples = PulseDuration * SampleFrequency if you don't know nsamples
    '''
    splitkeys = find_data_arrays(data)

    l = len(data[splitkeys[0]])

    if indices is None:
        if nloops is not None:
            nsamples = float(l / int(nloops))
        if nsamples is None:
            raise Exception('You must pass nloops, nsamples, or indices')
        # nsamples need not be an integer.  Will correct for extra time.
        trigger = [int(n) for n in np.arange(0, l, nsamples)]
        # If array is not evenly split, return the last fragment as well
        if trigger[-1] != l - 1:
            trigger.append(l - 1)
    else:
        # Trigger was passed directly as a list of indices
        trigger = indices
        # Put in endpoints
        if trigger[0] != 0:
            trigger = np.append([0], trigger)
        if trigger[-1] not in [-1, l - 1]:
            trigger = np.append(trigger, [l - 1])

    outlist = []
    for i, j in zip(trigger[:-1], trigger[1:]):
        splitloop = {}
        for k in splitkeys:
            if dupe_endpts:
                splitloop[k] = data[k][i:j+1]
            else:
                splitloop[k] = data[k][i:j]
        add_missing_keys(data, splitloop)
        outlist.append(splitloop)

    # If you pass a Series, you get a list of dicts anyway
    # This is slightly complicated to fix and I don't want to do it now
    return outlist


def concativ(data):
    ''' Inverse of splitiv.  Can only be called on multiple loops.  Keeps only keys from 0th loop.'''
    if type(data) is pd.DataFrame:
        firstrow = data.iloc[0]
    else:
        firstrow = data[0]

    concatkeys = find_data_arrays(firstrow)

    out = {}
    for k in concatkeys:
        if type(data) is pd.DataFrame:
            out[k] = np.concatenate(list(data[k]))
        else:
            out[k] = np.concatenate([d[k] for d in data])
    add_missing_keys(firstrow, out)

    if type(data) == pd.DataFrame:
        return pd.Series(out)
    else:
        return out


def meaniv(data, truncate=False, columns=None):
    '''
    Return the average of all iv columns.
    No interpolation at the moment
    not an ivfunc -- takes multiple loops and returns one
    '''
    if type(data) is pd.DataFrame:
        isdf = True
        firstrow = data.iloc[0]
    else:
        isdf = False
        firstrow = data[0]
    if columns is None:
        columns = find_data_arrays(firstrow)

    if isdf:
        lens = data[columns[0]].apply(len)
    else:
        lens = np.array([len(d[columns[0]]) for d in data])

    if truncate:
        # If the arrays are different sizes, truncate them all to the smallest size so that they can be averaged
        if not np.all(np.diff(lens) == 0):
            trunc = np.min(lens)
            print(f'Truncating data to length {trunc}')
            data = sliceiv(data, stop=trunc)

    dataout = {}
    for k in columns:
        if isdf:
            dataout[k] = data[k].mean()
        else:
            dataout[k] = np.mean([d[k] for d in data], axis=0)
    add_missing_keys(firstrow, dataout)
    if isdf:
        return pd.Series(dataout)
    else:
        return dataout


# TODO: Rename this, because it doesn't return a true "slice" of the data
# Maybe put "mask" in the name.  maskbyvalue? selectbyvalue?
@ivfunc
def slicebyvalue(data, column='V', minval=0, maxval=None):
    # This is so commonly done that I will make a function for it, though it's just a special case of indexiv
    # Including the endpoints in interval.  Change it later if you care.
    if (minval is None) and (maxval is not None):
        index = data[column] <= maxval
    elif (maxval is None) and (minval is not None):
        index = data[column] >= minval
    elif (maxval is not None):
        index = (minval <= data[column]) & (data[column] < maxval)
    else:
        return data

    dataout = {}
    keys = find_data_arrays(data)
    for k in keys:
        dataout[k] = data[k][index]
    add_missing_keys(data, dataout)

    return dataout


"""
@ivfunc
def slicebyvalue(data, column='V', startval=None, endval=None, occurance=0):
    '''
    Return a single continous slice of the IV data, between threshold values of a column.

    '''
"""


@ivfunc
def sortvalues(data, column='V', ascending=True):
    # Sort the iv data points by a certain column
    sortkeys = find_data_arrays(data)
    reindex = np.argsort(data['V'])
    if not ascending:
        reindex = reindex[::-1]
    dataout = {}
    for k in sortkeys:
        dataout[k] = data[k][reindex]
    add_missing_keys(data, dataout)
    return dataout


@ivfunc
def reversearrays(data, columns=None):
    ''' Reverse the direction of arrays.
    Faster than sorting if you know that they are reverse sorted'''
    if columns == None:
        columns = find_data_arrays(data)
    dataout = {}
    for k in columns:
        dataout[k] = data[k][::-1]
    add_missing_keys(data, dataout)
    return dataout


@ivfunc
def diffsign(data, column='V'):
    '''
    Return boolean array indicating if V is increasing, decreasing, or constant.
    Will not handle noisy data.  Have to dig up the code that I wrote to do that.
    '''
    direction = np.sign(np.diff(data[column]))
    # Need the same size array as started with. Categorize the last point same as previous 
    return np.append(direction, direction[-1])


@ivfunc
def nanmask(data, column='I', value=9.9100000000000005e+37):
    ''' Replace a value with nan.  Wrote this for replacing keithley special nan value ..'''
    mask = data[column] == value
    dataout = data.copy()
    dataout[column][mask] = np.nan
    return dataout

# I guess a func that just calls ivfuncs doesn't need to be an ivfunc itself
#@ivfunc

def decreasing(data, column='V', sort=False):
    decreased = indexiv(data, lambda l: diffsign(l, column) < 0)
    if sort:
        return sortvalues(decreased, column='V', ascending=True)
    else:
        return decreased


#@ivfunc
def increasing(data, column='V', sort=False):
    increased = indexiv(data, lambda l: diffsign(l, column) > 0)
    if sort:
        return sortvalues(increased, column='V', ascending=True)
    else:
        return increased


@ivfunc
def interpiv(data, interpvalues, column='I', reverse=False, findmonotonic=False, left=None, right=None):
    '''
    Interpolate all the arrays in ivloop to new values of one of the columns
    Right now this sorts the arrays according to "column"
    would be nice if newvalues could be a function, or an array of arrays ...
    '''
    lenI = len(data[column])
    interpkeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    interpkeys = [ik for ik in interpkeys if ik != column]

    # Get the largest monotonic subsequence of data, with 'column' increasing
    if findmonotonic:
        data = largest_monotonic(data)

    # not doing this anymore, but might want the code for something else
    #saturated = abs(dataout[column]/dataout[column][-1]) - 1 < 0.0001
    #lastindex = np.where(saturated)[0][0]
    #dataout[column] = dataout[column][:lastindex

    dataout = {}
    for ik in interpkeys:
        if reverse:
            dataout[ik] = np.interp(interpvalues, data[column][::-1], data[ik][::-1], left=left, right=right)
        else:
            dataout[ik] = np.interp(interpvalues, data[column], data[ik], left=left, right=right)
    dataout[column] = interpvalues
    add_missing_keys(data, dataout)

    return dataout


@ivfunc
def largest_monotonic(data, column='I'):
    ''' returns the segment of the iv loop that is monotonic in 'column', and
    spans the largest range of values.  in output, 'column' will be increasing
    Mainly used for interpolation function.

    Could pass in a function that operates on the segments to determine which one is "largest"
    '''
    lenI = len(data[column])
    keys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]

    # interp has a problem if the function is not monotonically increasing.
    # Find all monotonic sections of the data, use the longest section,
    # reversing it if it's decreasing This will have problems if 'column'
    # contains noisy data.  Deal with this for now by printing some warnings if
    # no segment of significant length is monotonic

    sign = np.sign(np.diff(data[column]))
    # Group by the sign of the first difference to get indices
    gpby = groupby(enumerate(sign, 0), lambda item: sign[item[0]])
    # Making some lists because I can't think of a better way at the moment
    # Sorry for these horrible lines. It's a list of tuples, (direction, (i,n,d,i,c,e,s))
    monolists = [(gp[0], *zip(*list(gp[1]))) for gp in gpby if abs(gp[0]) == 1]
    directions, indices, _ = zip(*monolists)
    segment_endpoints = [(ind[0], ind[-1] + 2) for ind in indices]
    #return segment_endpoints
    #start_indices, end_indices = zip(*[(ind[0], ind[-1] + 1) for ind in indices])
    # Finally, list of (direction, startindex, endindex) for all monotonic segments
    columnsegments = [data[column][start:end] for (start, end) in segment_endpoints]
    segment_spans = [max(vals) - min(vals) for vals in columnsegments]
    largest = np.argmax(segment_spans)
    direction = int(directions[largest])
    startind, endind = segment_endpoints[largest]

    dataout = {}
    for k in keys:
        dataout[k] = data[k][startind:endind][::direction]
    add_missing_keys(data, dataout)

    return dataout

@ivfunc
def jumps(loop, column='I', thresh=0.25, normalize=True, abs=True):
    ''' Find jumps in the data.
    if normalize=True, give thresh as fraction of maximum absolute value.
    return (indices,), (values of jumps,)
    pass abs=False if you care about the sign of the jump
    '''
    d = np.diff(loop[column])
    if normalize:
        thresh = thresh * np.max(np.abs(loop[column]))
    # Find jumps greater than thresh * 100% of the maximum
    if abs:
        jumps = np.where(np.abs(d) > thresh )[0]
    elif thresh < 0:
        jumps = np.where(d < thresh)[0]
    else:
        jumps = np.where(d > thresh)[0]

    return {'jumpind': jumps, 'jumpval':d[jumps]}
    #return [jumps, d[jumps]]

@ivfunc
def njumps(loop, **kwargs):
    j = jumps(loop, **kwargs)
    njumps = len(j[0])
    loop['njumps'] = njumps
    return njumps


@ivfunc
def first_jump(loop, **kwargs):
    j = jumps(loop, **kwargs)
    if np.any(j):
        first_jump = j[0][0]
    else:
        first_jump = np.nan
    #loop['first_jump'] = first_jump
    return first_jump


@ivfunc
def last_jump(loop, **kwargs):
    j = jumps(loop, **kwargs)
    if np.any(j):
        last_jump = j[0][-1]
    else:
        last_jump = np.nan
    #loop['last_jump'] = last_jump
    return last_jump 

@ivfunc
def nth_jump(loop, n, **kwargs):
    j = jumps(loop, **kwargs)
    if np.any(j) and len(j[0]) > n:
        last_jump = j[0][n]
    else:
        last_jump = np.nan
    return last_jump 

'''
Can't do this because ivfunc doesn't know it also iterate through the "index", which is different for every loop
could do it if index was a function (i.e. first_jump)
@ivfunc
def pindex(loop, column, index):
    if np.isnan(index):
        return np.nan
    else:
        return loop[column][index]
'''

# These are dumb names.  Supposed to be pindex for parallel index
# What it is doing is indexing a single column of multiple dicts (dataframe) at the same time

@ivfunc
def pindex_fromfunc(loop, column, indexfunc):
    index = indexfunc(loop)
    if np.isnan(index):
        return np.nan
    else:
        return loop[column][index]


def pindex_fromlist(loops, column, indexlist):
    # Index some column of all the ivloops in parallel
    # "index" is a list of indices with same len as loops
    # Understands list[nan] --> nan
    # TODO: index by a number contained in the ivloop object
    # TODO: rename to something that makes sense
    vals = []
    if type(loops) == pd.DataFrame:
        loops = loops.iterrows()
    else:
        # Because I don't know how to iterate through the df rows without enumerating them
        # Also enumerate through list
        loops = enumerate(loops)

    if not hasattr(indexlist, '__iter__'):
        indexlist = [indexlist] * len(loops)

    for (_, l), i in zip(loops, indexlist):
        if np.isnan(i):
            vals.append(np.nan)
        else:
            vals.append(l[column][int(i)])
    return np.array(vals)


@ivfunc
def pindex(data, column, index):
    '''
    Index some column of all the ivloops in parallel
    '''
    # This one replaces _fromfunc and _fromlist using paramlist() and paramfunc()
    # Dunno didn't test it..
    if np.isnan(index):
        return np.nan
    else:
        return data[column][index]


@ivfunc
def longest_monotonic(data, column='I'):
    ''' returns the largest segment of the iv loop that is monotonic in
    'column'.  in output, 'column' will be increasing Mainly used for
    interpolation function. '''
    lenI = len(data[column])
    keys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]

    # interp has a problem if the function is not monotonically increasing.
    # Find all monotonic sections of the data, use the longest section,
    # reversing it if it's decreasing This will have problems if 'column'
    # contains noisy data.  Deal with this for now by printing some warnings if
    # no segment of significant length is monotonic

    sign = np.sign(np.diff(data[column]))
    # Group by the sign of the first difference to get indices
    gpby = groupby(enumerate(sign, 0), lambda item: sign[item[0]])
    # Making some lists because I can't think of a better way at the moment
    monolists = [(gp[0], list(gp[1])) for gp in gpby if abs(gp[0]) == 1]
    segment_lengths = [len(gp[1]) for gp in monolists]
    longest = np.argmax(segment_lengths)
    if segment_lengths[longest] < lenI * 0.4:
        print('No monotonic segments longer than 40% of the {} array were found!'.format(column))
    direction = int(monolists[longest][0])
    startind = monolists[longest][1][0][0]
    endind = monolists[longest][1][-1][0] + 2

    dataout = {}
    for k in keys:
        dataout[k] = data[k][startind:endind][::direction]
    add_missing_keys(data, dataout)

    return dataout


@ivfunc
def normalize(data):
    ''' Normalize by the maximum current '''
    dataout = {}
    maxI = np.max(data['I'])
    dataout['I'] = data['I'] / maxI
    add_missing_keys(data, dataout)
    if 'units' in dataout:
        dataout['units']['I'] = 'Normalized'
    return dataout


def add_missing_keys(datain, dataout):
    for k in datain.keys():
        if k not in dataout.keys():
            dataout[k] = datain[k]


@ivfunc
def resistance(data, v0=0.5, v1=None, x='V', y='I'):
    '''
    Fit a line to IV data to find R.
    if v1 not given, fit from -v0 to +v0
    TODO: call polyfitiv
    '''
    if v1 is None:
        v1 = -v0
    vmin = min(v0, v1)
    vmax = max(v0, v1)
    V = data[x]
    I = data[y]
    mask = (V <= vmax) & (V >= vmin)
    poly = np.polyfit(I[mask], V[mask], 1)
    if 'units' in data:
        if y in data['units']:
            Iunit = data['units'][y]
            if Iunit == 'A':
                return poly[0]
            elif Iunit == '$\mu$A':
                return poly[0] * 1e6
            elif Iunit == 'mA':
                return poly[0] * 1e3
            else:
                print('Did not understand current unit!')
    return poly[0]


@ivfunc
def polyfitiv(data, order=1, x='V', y='I', xmin=None, xmax=None, ymin=None, ymax=None, extendrange=False, mask=None):
    '''
    Fit a polynomial to IV data.  Can specify the value range of x and y to use
    xmin < xmax,  ymin < ymax
    if extend == True, the range will be extended to include at least order+1 data points
    '''
    X = data[x]
    Y = data[y]

    if mask is None:
        mask = np.ones(len(X), dtype=bool)
        if xmin is not None:
            mask &= X >= xmin
        if xmax is not None:
            mask &= X <= xmax
        if ymin is not None:
            mask &= Y >= ymin
        if ymax is not None:
            mask &= Y <= ymax


    if sum(mask) > order:
        pf = np.polyfit(X[mask], Y[mask], order)
    elif extendrange:
        # There were not enough data points in the passed fit range
        # so use the "nearest" datapoints outside the range.
        # Can think of a few different ways to do it, all dumb
        def find_center(vmin, vmax):
            if (vmin is None) and (vmax is None):
                return None
            elif vmin is None:
                return vmax
            elif vmax is None:
                return vmin
            else:
                return (vmin + vmax) / 2

        xc = find_center(xmin, xmax)
        yc = find_center(ymin, ymax)
        mask = select_nclosest(data, x=xc, y=yc, n=order + 1, xarr=x, yarr=y)
        return polyfitiv(data, order=order, mask=mask, x=x, y=y)
    else:
        # Don't fit if "poorly conditioned"
        pf = [np.nan] * (order + 1)

    return pf

@ivfunc
def select_nclosest(data, n=2, x=None, y=None, xarr='V', yarr='I'):
    # Return the indices of the n data points nearest to the indicated x, y value
    X = data[xarr]
    Y = data[yarr]
    if x is None:
        distance = (Y - y)**2
    elif y is None:
        distance = (X - x)**2
    else:
        distance = (X - x)**2 + (Y - y)**2

    nclosest = np.argsort(distance)[:n]

    return nclosest


@ivfunc
def resistance_states(data, v0=0.1, v1=None):
    ''' Calculate resistance for increasing/decreasing branches '''
    RS1, RS2 = resistance(splitbranch(data), v0, v1)
    return [RS1, RS2]


@ivfunc
def convert_unit(column='I', prefix='u'):
    #longnames = ['exa', 'peta', 'tera', 'giga', 'mega', 'kilo', '', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto']
    prefix = ['E', 'P', 'T', 'G', 'M', 'k', '', 'm', '$\mu$', 'n', 'p', 'f', 'a']


@ivfunc
def downsample_dumb(data, nsamples, columns=None):
    ''' Downsample arrays with equal spacing. Probably won't be exactly nsamples'''
    if columns is None:
        columns = find_data_arrays(data)
        l = len(data[columns[0]])
        step = round(l / (nsamples - 1))
    if step <= 1:
        return data
    dataout = {c:data[c][::step] for c in columns}
    add_missing_keys(data, dataout)
    return dataout


def df_to_listofdicts(df):
    return df.to_dict(orient='records')


@ivfunc
def set_dict_kv(data, dictname, key, value):
    '''
    Set a dict key:value pair for all dicts in iv list.
    basically setting second level k:v for nested dicts
    i.e. set_dict_kv(df, 'longnames', 'T', 'Temperature')
    Won't work for dataframes that don't already have the dictname column, because it tries to add data to the individual series, without ever seeing the real dataframe
    '''
    if dictname in data:
        data[dictname][key] = value
    else:
        data[dictname] = {key: value}

def set_unit(data, name, unit):
    # Need to make the column if it's a dataframe and doesn't have it
    if type(data) == pd.DataFrame:
        if 'units' not in data:
            data['units'] = [{}] * len(data)
    set_dict_kv(data, 'units', name, unit)

def set_longname(data, name, longname):
    # Need to make the column if it's a dataframe and doesn't have it
    if type(data) == pd.DataFrame:
        if 'longnames' not in data:
            data['longnames'] = [{}] * len(data)
    set_dict_kv(data, 'longnames', name, longname)

def set_unit_info(data, colname, unit=None, longname=None):
    '''
    Set information about the units in a column (array)
    This information can then be used to label plots automatically
    '''
    if unit is not None:
        set_unit(data, colname, unit)
    if longname is not None:
        set_longname(data, colname, longname)

# These are not needed for pandas types obviously
# Just trying to get some of that functionality into list of dicts
@ivfunc
def apply(data, func, column):
    '''
    This applies func to one column of the ivloop, and leaves the rest the same.
    func should take an array and return an array of the same size
    '''
    dataout = {}
    dataout[column] = func(dataout[column])
    add_missing_keys(data, dataout)
    return dataout


def insert(data, key, vals):
    '''
    Insert key:values into list of dicts
    Like a pandas column
    '''
    if hasattr(vals, '__getitem__'):
        for d,v in zip(data, vals):
            d[key] = v
    else:
        for d in data:
            d[key] = vals


def extract(data, key):
    '''
    Get array of values from list of dicts
    '''
    return array([d[key] for d in data])


# I like typing smooth instead of Rolling/running/moving average/mean
def smooth(x, N):
    '''
    Efficient rolling mean for arrays
    Faster than numpy.convolve for most situations (window > 10)
    Floating point errors will accumulate if you use lower precision!
    Converts to and back from float64.  Still seems to be an issue using float16.
    '''
    if N <= 10:
        # Use convolve
        return smooth_conv(x, N)
    dtypein = type(x[0])
    converted = False
    if dtypein in (np.float32, np.float16):
        if len(x) > 100000:
            # Convert to and back from float64
            converted = True
            x = np.float64(x)
    # Do the smoothing
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    movingavg = (cumsum[N:] - cumsum[:-N]) / N
    if converted:
        return dtypein(movingavg)
    else:
        return movingavg


def smooth_conv(x, N, mode='valid'):
    ''' Smooth (moving avg) with convolution '''
    dtypein = type(x[0])
    return np.convolve(x, np.ones(N, dtype=dtypein)/dtypein(N), mode)


@ivfunc
def convert_to_uA(data):
    ''' Works in place and returns nothing.  Sorry for inconsistency'''
    data['I'] *= 1e6
    data['units']['I'] = '$\mu$A'


@ivfunc
def drop_arrays(data):
    ''' Drop all numpy arrays from the data '''
    keys = data.keys()
    types = [type(data[k]) for k in keys]
    arrays = [k for k,t in zip(keys, types) if t == np.ndarray]
    savekeys = [k for k in keys if k not in arrays]
    return {sk:data[sk] for sk in savekeys}
    # In pandas you can do df.drop(arrays, 1), which is much much faster

def fit_sine_array(array, dt=1, guess_freq=1, debug=False):
    ''' Fit a sine function to array.  Assumes equal spacing in time.  Guess has to be pretty close.'''
    #guess_amplitude = 3*np.std(array)/(2**0.5)
    guess_amplitude = (np.max(array) - np.min(array)) / 2
    guess_offset = np.mean(array)
    startval = (array[0] - guess_offset) / guess_amplitude
    startval = np.clip(startval, -1, 1)
    guess_phase = np.arcsin(startval)
    if array[1] < array[0]:
        # Probably there is a big phase shift, because curve is decreasing at first
        # But np.arcsin will always return between +- pi/2, which is where sine is increasing.
        # Need the other solution.
        guess_phase = np.sign(guess_phase) * (np.pi - abs(guess_phase))

    # Could guess freq by fft, but that would probably take longer than the fit itself.

    p0=[guess_freq, guess_amplitude, guess_phase, guess_offset]

    # Define the function we want to fit
    def my_sin(x, freq, amplitude, phase, offset):
        return np.sin(x * 2 * np.pi * freq + phase) * amplitude + offset

    # Now do the fit
    t = np.linspace(0, dt, len(array))
    fit, cov = curve_fit(my_sin, t, array, p0=p0)

    if debug:
        data_first_guess = my_sin(t, *p0)
        data_fit = my_sin(t, *fit)
        plt.figure()
        plt.plot(array, '.')
        plt.plot(data_fit, label='after fitting', linewidth=2)
        plt.plot(data_first_guess, label='first guess')
        plt.legend()
        plt.show()

    return {'freq':fit[0],
            'amp':fit[1],
            'phase':fit[2],
            'offset':fit[3]}

@ivfunc
def fft_iv(data, columns=None):
    ''' Calculate fft of arrays.  Return with same name as original arrays. '''
    if columns is None:
        columns = find_data_arrays(data)

    if 'sample_rate' in data:
        # Use sample rate to scale frequencies
        pass
    else:
        pass

    # Make dict of dicts
    dataout = {}
    for c in columns:
        dataout[c] = np.fft.fft(data[c])
    add_missing_keys(data, dataout)

    # TODO: Units change due to fft...

    return dataout

@ivfunc
def largest_fft_component(data):
    ''' Find the amplitude and phase of the largest single frequency component'''
    # Calculate fft
    fftdata = fft_iv(data)
    columns = find_data_arrays(fftdata)

    dataout = {}
    for c in columns:
        l = len(fftdata[c])
        halfl = int(l/2)
        # I still don't understand what the other half of the fft array is.
        mag = np.abs(fftdata[c][:halfl])
        phase = np.angle(fftdata[c][:halfl])
        # Find which harmonic is the largest
        fundi = np.argmax(mag)
        #Want a single index dataframe.  Name the columns like this:
        dataout[c + '_freq'] = fundi
        if 'sample_rate' in data:
            dataout[c + '_freq'] *= data['sample_rate'] / l
        dataout[c + '_amp'] = mag[fundi] * 2 / l
        # fft gives phase of cos, but we apply signals starting from zero, so phase of sine is clearer
        dataout[c + '_phase'] = phase[fundi] + np.pi/2
        dataout[c + '_offset'] = mag[0]

    return dataout

@ivfunc
def fit_sine(data, columns=None, guess_ncycles=None, debug=False):
    ''' Fit a sine function to some data.  Return phase and amplitude of the fit. '''
    if columns is None:
        columns = find_data_arrays(data)

    if 'sample_rate' in data:
        dt = len(data[columns[0]]) / data['sample_rate']
    else:
        # Output frequency will be number of cycles in the input waveform
        dt = 1

    if 'ncycles' in data:
        # freq_response function adds this key so we don't need to guess
        guess_ncycles = data['ncycles']
    elif guess_ncycles is None:
        # TODO: could guess based on fft or something, but could be slow
        raise Exception('If data does not contain ncycles key, then guess_ncycles must be passed!')

    guess_freq = guess_ncycles / dt

    # Make dict of dicts
    dataout = {}
    for c in columns:
        sinefit = fit_sine_array(data[c], dt=dt, guess_freq=guess_freq, debug=debug)

        # Don't want negative amplitudes.  But constraining fit function always has bad consequences.
        if sinefit['amp'] < 0:
            sinefit['amp'] *= -1
            # Keep phase in range of (-np.pi:np.pi]
            if sinefit['phase'] > 0:
                sinefit['phase'] -= np.pi
            else:
                sinefit['phase'] += np.pi


        #Want a single index dataframe.  Name the columns like this:
        dataout[c + '_freq'] = sinefit['freq']
        dataout[c + '_amp'] = sinefit['amp']
        dataout[c + '_phase'] = sinefit['phase']
        dataout[c + '_offset'] = sinefit['offset']
        # This returns something a little weird - nested dicts or dicts as dataframe elements ..
        # dataout[c] = fit_sine_array(data[c], dt=dt, guess_freq=guess_freq, debug=debug)

    return dataout

@ivfunc
def freq_analysis(data):
    '''
    Input data (e.g. collected by freq_response function) containing sinusoid arrays.
    This will use curve fitting and fft methods to determine amplitude and phase.
    '''
    pass

def replace_nanvals(array):
    # Keithley returns this special value when the measurement is out of range
    # replace it with a nan so it doesn't mess up the plots
    # They aren't that smart at Keithley, so different models return different special values.
    nanvalues = (9.9100000000000005e+37, 9.9099995300309287e+37)
    for nv in nanvalues:
        array[array == nv] = np.nan
    return array


### Interactive

def hand_select(df, groupby=None, **kwargs):
    '''
    Select a subset of loops by hand.
    Right now it selects one loop from each group
    '''
    print('\n\n')
    print('left arrow: previous loop')
    print('right arrow: next loop')
    print('down arrow: truncate n data points')
    print('left arrow: untruncate n data points')
    print('[1-9]: Set n')
    print('Enter: select loop and')
    print('\n\n')

    # Shitty manual loop selection written as fast as I could
    def selectloop(data):
        fig, ax = plt.subplots()
        fignum = fig.number
        n = 0
        class thinger(object):
            def __init__(self):
                self.n = 0
                self.l = None
                self.step = 1
            def press(self, event):
                print('press', event.key)
                if event.key == 'right':
                    # plot next loop
                    self.l = None
                    self.n = (self.n + self.step) % len(data)
                elif event.key == 'left':
                    # plot previous loop
                    self.l = None
                    self.n = (self.n - self.step) % len(data)
                elif event.key == 'enter':
                    # select the loop and return
                    plt.close(fig)
                    return
                elif event.key == 'down':
                    # Take a datapoint off the end
                    if self.l is None:
                        self.l = -self.step
                    else:
                        self.l -= self.step
                elif event.key == 'up':
                    # add another datapoint to the end
                    if self.l is not None:
                        self.l += self.step
                elif event.key in '0123456789':
                    # change step size
                    self.step = 2**int(event.key)
                    return

                if len(data) >= self.n + 1:
                    del ax.lines[-1]
                    print(self.n, self.l)
                    #print(data.iloc[self.n].Irange[0])
                    ivplot.plotiv(sliceiv(data.iloc[self.n], stop=self.l), x='Vcalc', color='red', ax=ax)
                else:
                    print('no more data')
                    self.n -= 1

                sys.stdout.flush()
                fig.canvas.draw()

        ivplot.plotiv(data, x='Vcalc', alpha=.1, color='black', ax=ax)
        ivplot.plotiv(data.iloc[0], x='Vcalc', color='red', ax=ax)

        thing = thinger() # lol object oriented
        cid = fig.canvas.mpl_connect('key_press_event', thing.press)
        # loop until figure closed?
        while plt.fignum_exists(fignum):
            plt.pause(.1)

        return sliceiv(data.iloc[thing.n], stop=thing.l)

    if groupby is None:
        return selectloop(df)
    else:
        selected = []
        for k,g in df.groupby(groupby):
            selected.append(selectloop(g))
        return pd.DataFrame(selected)


### Not ivfuncs, useful to have around

def category(alist):
    unique, category = np.unique(alist, return_inverse=True)
    return category
