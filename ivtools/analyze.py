""" Functions for doing data analysis on IV data """

from functools import wraps
import numpy as np
from itertools import groupby
from dotdict import dotdict
from scipy import signal
from numbers import Number

def ivfunc(func):
    '''
    Decorator which allows the same function to be used on a single loop, as
    well as a container of loops.

    Decorated function should take a single loop and return anything
    Then this function will also take multiple loops, and return a list/dataframe of the outputs

    Handles dicts and pd.Series as IV data, and "list of dict" and DataFrames for multiple IV data
    An attempt is made to return the most reasonable type, given the input and output types

    If any of the arguments instances of "paramlist", this tells ivfunc to also index into this list when
    calling the wrapped function, so that you can pass a list of parameters to use for each iv loop.

    If you pass a function wrapped with the paramfunc function, that function will get called on the data to
    determine the argument.
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
        ###  If a DataFrame is passed
        if dtype == pd.DataFrame:
            # Apply the function to the columns of the dataframe
            # Sadly, the line below will error if your function returns an array
            # return data.apply(func, axis=1, args=args, **kwargs)
            resultlist = []
            for i, (rownum, row) in enumerate(data.iterrows()):
                #resultlist.append(func(row, *args, **kwargs))
                result = func(row, *[paramtransform(arg, i, row) for arg in args],
                              **{k:paramtransform(v, i, row) for k,v in kwargs.items()})
                resultlist.append(result)
            ### Decide how to return the values based on the datatype that the wrapped function returned
            if type(resultlist[0]) in (pd.Series, dict):
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
            # For all other cases
            # Keep the index the same!
            series_out = pd.Series(resultlist)
            series_out.index = data.index
            return series_out
        ### If a list (of dicts) is passed
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
        elif dtype in (dotdict, dict):
            # It's just one IV dict
            return(func(data, *args, **kwargs))
        else:
            print('ivfunc did not understand the input datatype {}'.format(dtype))
    return func_wrapper

class paramlist(list):
    # Only a class so that ivfunc can know what you want to do with it
    # Which is pass a list of parameters to use for each loop
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
    if columns is None:
        columns = find_data_arrays(data)
    arrays = [data[c] for c in columns]
    diffarrays = [ar[stride:] - ar[:-stride] for ar in arrays]
    dataout = {c:diff for c,diff in zip(columns, diffarrays)}
    add_missing_keys(data, dataout)
    return dataout

@ivfunc
def thresholds_bydiff(data, stride=1):
    ''' Find switching thresholds by finding the maximum differences. '''
    diffI = data['I'][stride:] - data['I'][:-stride]
    argmaxdiffI = np.argmax(diffI)
    vset = data['V'][argmaxdiffI]
    maxdiffI = diffI[argmaxdiffI]
    argmindiffI = np.argmin(diffI)
    vreset = data['V'][argmindiffI]
    mindiffI = diffI[argmindiffI]
    # TODO: This is breaking the pattern of other ivfuncs -- list of dict will return list of series...
    return pd.Series({'Vset':vset, 'Vreset':vreset, 'Idiffmax':maxdiffI, 'Idiffmin':mindiffI})

'''
@ivfunc
def thresholds_byval(data, value):
    pindex(data, 'I', value)
'''

@ivfunc
def moving_avg(data, window=5, columns=('I', 'V')):
    ''' Smooth data arrays with moving avg '''
    if columns is None:
        columns = find_data_arrays(data)
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
def smoothimate(data, window=10, factor=2, passes=1, columns=('I', 'V')):
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
        decarrays = [ar[::factor] for ar in smootharrays]
    for c, ar, dtype in zip(columns, decarrays, dtypes):
        if dtype is np.float64:
            # Datatype was already float64, don't convert float64 to float64
            dataout[c] = ar
        else:
            # Convert back to original data type
            dataout[c] = dtype(ar)
    add_missing_keys(data, dataout)
    dataout['downsampling'] = factor
    dataout['smoothing'] = window
    dataout['smoothimate_passes'] = passes
    return dataout

@ivfunc
def maketimearray(data):
    # TODO: need to account for any possible downsampling!
    return np.arange(len(data['V'])) * 1/data['sample_rate']

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
    if not hasattr(index, '__iter__') and np.isnan(index):
        dataout = {c:np.nan for c in colnames}
    else:
        dataout = {c:data[c][index] for c in colnames}

    add_missing_keys(data, dataout)
    return dataout

@ivfunc
def sliceiv(data, stop=-1, start=0, step=1):
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
    for sk in slicekeys:
        # Apply the filter to all the relevant items
        dataout[sk] = data[sk][slice(int(start), int(stop), int(step))]
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


# NOT an ivfunc -- can only be called on single IV
# Would it make sense to collapse a list of IVs into a flattened list of list of IVs?
@ivfunc
def split_by_crossing(data, V=0, increasing=True, hyspts=50):
    '''
    Split loops into multiple loops, by threshold crossing
    Only implemented V threshold crossing
    return list of input type
    Noisy data is hard to split this way
    hyspts will require that on a crossing, the value of V was above/below threshold hyspts ago
    set it to less than half of the minimum loop length
    '''
    # V threshold crossing
    side = data['V'] >= V
    crossings = np.diff(np.int8(side))
    if increasing:
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
def splitbranch(data, columns=None):
    '''
    Split a loop into two branches
    Assumptions are that loop starts at intermediate V (like zero), goes to one extremum, to another extremum, then back to zero.
    Can also just go to one extreme and back to zero
    Not sure how to extend to splitting multiple loops.  Should it return interleaved branches or two separate dataframes/lists?
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
            branch1[c] = np.concatenate((data[c][secondextreme:], data[c][:firstextreme]))
            branch2[c] = data[c][firstextreme:secondextreme]
    else:
        for c in columns:
            branch1[c] = data[c][:singleextreme]
            branch2[c] = data[c][singleextreme:]

    add_missing_keys(data, branch1)
    add_missing_keys(data, branch2)

    return [branch1, branch2]


@ivfunc
def splitiv(data, nloops=None, nsamples=None):
    '''
    Split data into individual loops, specifying somehow the length of each loop
    if you pass nloops, it splits evenly into n loops.
    if you pass nsamples, it makes each loop have that many samples (except possibly the last one)
    pass nsamples = PulseDuration * SampleFrequency if you don't know nsamples
    '''
    l = len(data['V'])
    if nloops is not None:
        nsamples = float(l / int(nloops))
    if nsamples is None:
        raise Exception('You must pass nloops or nsamples')
    # nsamples need not be an integer.  Will correct for extra time.
    trigger = [int(n) for n in np.arange(0, l, nsamples)]
    # If array is not evenly split, return the last fragment as well
    if trigger[-1] != l - 1:
        trigger.append(l - 1)

    splitkeys = find_data_arrays(data)
    outlist = []
    for i, j in zip(trigger[:-1], trigger[1:]):
        splitloop = {}
        for k in splitkeys:
            splitloop[k] = data[k][i:j]
        add_missing_keys(data, splitloop)
        outlist.append(splitloop)

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

def meaniv(data, columns=None):
    '''
    Return the average of all iv columns.
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
    dataout = {}
    for k in columns:
        if isdf:
            dataout[k] = data[k].mean()
        else:
            dataout[k] = np.mean([d[k] for d in data])
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
def interpolate(data, interpvalues, column='I', reverse=False, findmonotonic=False):
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
            dataout[ik] = np.interp(interpvalues, data[column][::-1], data[ik][::-1])
        else:
            dataout[ik] = np.interp(interpvalues, data[column], data[ik])
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
# just gets a single value of a single column determined by a function


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
    pass

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
def resistance(data, v0=0.1, v1=None, x='V', y='I'):
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


"""
# Lol I wrote nearly the same function twice, keeping it here to laugh at myself
@ivfunc
def polyfitiv(data, Vmax=None, Imax=None, Vmin=None, Imin=None, order=1, x='V', y='I'):
    ''' Fit polynomials to range of voltage and current values '''
    ones = np.ones(len(data[x]), dtype=bool)
    if Vmin is None:
        xminmask = ones
    else:
        xminmask = Vmin <= data[x]
    if Vmax is None:
        xmaxmask = ones
    else:
        xmaxmask = data[x] <= Vmax
    if Imin is None:
        yminmask = ones
    else:
        yminmask = Imin <= data[y]
    if Imax is None:
        ymaxmask = ones
    else:
        ymaxmask = data[y] <= Imax
    mask = xminmask & xmaxmask & yminmask & ymaxmask
    yfit = data[y][mask]
    xfit = data[x][mask]
    return np.polyfit(xfit, yfit, order)
"""

@ivfunc
def polyfitiv(data, order=1, x='V', y='I', xmin=None, xmax=None, ymin=None, ymax=None):
    '''
    Fit a polynomial to IV data.  Can specify the value range of x and y to use
    xmin < xmax,  ymin < ymax
    '''
    X = data[x]
    Y = data[y]
    mask = np.ones(len(X), dtype=bool)
    if xmin is not None:
        mask &= X >= xmin
    if xmax is not None:
        mask &= X <= xmax
    if ymin is not None:
        mask &= Y >= ymin
    if ymax is not None:
        mask &= Y <= ymax

    pf = polyfit(X[mask], Y[mask], order)

    return pf

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
    return df.to_dict('records')

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
    Faster than numpy.convolve for most situations (window < 10)
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
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
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

