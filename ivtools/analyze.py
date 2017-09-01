""" Functions for doing data analysis on IV data """

from functools import wraps
import numpy as np
from itertools import groupby
from dotdict import dotdict

def ivfunc(func):
    '''
    Decorator which allows the same function to be used on a single loop, as
    well as a container of loops.

    Don't know if this is a good idea or not ...

    Decorated function should take a single loop and return anything

    Then this function will also take multiple loops, and return an array of the outputs
    '''
    @wraps(func)
    def func_wrapper(data, *args, **kwargs):
        dtype = type(data)
        if dtype == pd.DataFrame:
            # Apply the function to the columns of the dataframe
            # This will error if your function returns an array
            #return data.apply(func, axis=1, args=args, **kwargs)
            resultlist = []
            for i, row in data.iterrows():
                resultlist.append(func(row, *args, **kwargs))
            # Let's decide how to return the values based on the datatype that the wrapped function returned
            if type(resultlist[0]) == pd.Series:
                return pd.DataFrame(resultlist)
            else:
                # Keep the index the same!
                series_out = pd.Series(resultlist)
                series_out.index = data.index
                return series_out
        elif dtype == list:
            # Assuming it's a list of iv dicts
            return [func(d, *args, **kwargs) for d in data]
        elif dtype in (dotdict, dict, pd.Series):
            # It's just one IV dict
            return(func(data, *args, **kwargs))
        else:
            print('ivfunc did not understand the input datatype {}'.format(dtype))
    return func_wrapper

'''
Functions that return a new value/new array per IV loop should just return that value
Functions that modify the IV data should return a copy of the entire input structure
# TODO: don't copy all the data arrays if you are returning new ones
'''

@ivfunc
def find_data_arrays(data):
    # Determine the names of arrays that have same length as 'I' ('V', 'R', 'P')
    # We will select them now just based on which values are arrays with same size as I and V
    lenI = len(data['I'])
    arraykeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    return arraykeys

@ivfunc
def moving_avg(data, columns=('I', 'V'), window=5):
    ''' Smooth data arrays with moving avg '''
    arrays = [data[c] for c in columns]
    lens = [len(ar) for ar in arrays]
    if not all([l - lens[0] == 0 for l in lens]):
        raise Exception('Arrays to be smoothed have different lengths!')
    if lens[0] == 0:
        raise Exception('No data to smooth')
    weights = np.repeat(1.0, window)/window
    smootharrays = [np.convolve(ar, weights, 'valid') for ar in arrays]

    dataout = type(data)()
    for c, smooth in zip(columns, smootharrays):
        dataout[c] = smooth
    add_missing_keys(data, dataout)
    return dataout


@ivfunc
def indexiv(data, index_function):
    '''
    Index all the data arrays inside an iv loop container at once.
    Condition specified by index function, which should take an iv dict/series and return an indexing array
    '''
    splitkeys = find_data_arrays(data)

    dataout = type(data)()
    for sk in splitkeys:
        # Apply the filter to all the relevant items
        index = np.array(index_function(data))
        dataout[sk] = data[sk][index]
    add_missing_keys(data, dataout)

    return dataout

@ivfunc
def sliceiv(data, stop, start=0, step=None):
    '''
    Slice all the data arrays inside an iv loop container at once.
    start, stop can be functions that take the iv loop as argument
    if those functions return nan, start defaults to 0 and stop to -1
    '''
    slicekeys = find_data_arrays(data)
    dataout = type(data)()
    if callable(start):
        start = start(data)
        if np.isnan(start): start = 0
    if callable(stop):
        stop = stop(data)
        if np.isnan(stop): stop = -1
    for sk in slicekeys:
        # Apply the filter to all the relevant items
        dataout[sk] = data[sk][slice(start, stop, step)]
    add_missing_keys(data, dataout)
    return dataout

# NOT an ivfunc -- can only be called on single IV
# Would it make sense to collapse a list of IVs into a flattened list of list of IVs?
def split_by_crossing(data, V=0, increasing=True, hys=1e-3):
    '''
    Split loops into multiple loops
    Only implemented V threshold crossing
    return list of input type
    '''
    # V threshold crossing
    # Noisy data is hard to split this way
    side = data['V'] >= V
    crossings = np.diff(np.int8(side))
    if increasing:
        trigger = np.where(crossings == 1)
    else:
        trigger = np.where(crossings == -1)
    # Put the endpoints in
    trigger = np.concatenate(([0], trigger[0], [-1]))

    outlist = []
    splitkeys = find_data_arrays(data)
    for i, j in zip(trigger[:-1], trigger[1:]):
        splitloop = type(data)()
        for k in splitkeys:
            splitloop[k] = data[k][i:j]
        add_missing_keys(data, splitloop)
        outlist.append(splitloop)

    if type(data) == pd.Series:
        return pd.DataFrame(outlist)
    else:
        return outlist


def splitiv(data, nloops=None, nsamples=None, fs=None, duration=None):
    '''
    Split data into individual loops, specifying somehow the length of each loop
    if you pass nloops, it splits evenly into n loops.
    if you pass nsamples, it makes each loop have that many samples (except possibly the last one)
    pass nsamples = PulseDuration * SampleFrequency if you don't know nsamples
    '''
    l = len(data['V'])
    if nloops is not None:
        nsamples = float(l / int(nloops))
    # nsamples need not be an integer.  Will correct for extra time.
    trigger = [int(n) for n in arange(0, l, nsamples)]
    # If array is not evenly split, return the last fragment as well
    if trigger[-1] != l - 1:
        trigger.append(l - 1)

    splitkeys = find_data_arrays(data)
    outlist = []
    for i, j in zip(trigger[:-1], trigger[1:]):
        splitloop = type(data)()
        for k in splitkeys:
            splitloop[k] = data[k][i:j]
        add_missing_keys(data, splitloop)
        outlist.append(splitloop)

    if type(data) == pd.Series:
        return pd.DataFrame(outlist)
    else:
        return outlist


@ivfunc
def slicebyvalue(data, column='V', minval=0, maxval=None):
    # This is so commonly done that I will make a function for it, though it's just a special case of indexiv
    # Including the endpoints in interval.  Change it later if you care.
    keys = find_data_arrays(data)
    dataout = type(data)()
    if (minval is None) and (maxval is not None):
        index = data[column] <= maxval
    elif (maxval is None) and (minval is not None):
        index = data[column] >= minval
    elif (maxval is not None):
        index = minval <= data[column] < maxval
    else:
        return data
    for k in keys:
        dataout[k] = data[k][index]
    add_missing_keys(data, dataout)

    return dataout


@ivfunc
def sortvalues(data, column='V', ascending=True):
    # Sort the iv data points by a certain column
    sortkeys = find_data_arrays(iv)
    reindex = np.argsort(data['V'])
    if not ascending:
        reindex = reindex[::-1]
    dataout = type(data)()
    for k in sortkeys:
        dataout[k] = data[k][reindex]
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
def decreasing(data, column='V'):
    # Could sort afterward, but that could lead to undesired behavior
    return indexiv(data, lambda l: diffsign(l, column) < 0)


@ivfunc
def increasing(data, column='V'):
    # Could sort afterward, but that could lead to undesired behavior
    return indexiv(data, lambda l: diffsign(l, column) > 0)


@ivfunc
def interpolate(data, interpvalues, column='I'):
    '''
    Interpolate all the arrays in ivloop to new values of one of the columns
    Right now this sorts the arrays according to "column"
    would be nice if newvalues could be a function, or an array of arrays ...
    '''
    lenI = len(data[column])
    interpkeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    interpkeys = [ik for ik in interpkeys if ik != column]

    # Get the largest monotonic subsequence of data, with 'column' increasing
    dataout = largest_monotonic(data)

    # not doing this anymore, but might want the code for something else
    #saturated = abs(dataout[column]/dataout[column][-1]) - 1 < 0.0001
    #lastindex = np.where(saturated)[0][0]
    #dataout[column] = dataout[column][:lastindex

    for ik in interpkeys:
        dataout[ik] = np.interp(interpvalues, dataout[column], dataout[ik])
    dataout[column] = interpvalues

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

    dataout = type(data)()
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
    return jumps, d[jumps]

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
        loops = enumerate(loops)
    for (_, l), i in zip(loops, indexlist):
        if np.isnan(i):
            vals.append(np.nan)
        else:
            vals.append(l[column][int(i)])
    return np.array(vals)


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

    dataout = type(data)()
    for k in keys:
        dataout[k] = data[k][startind:endind][::direction]
    add_missing_keys(data, dataout)

    return dataout

@ivfunc
def normalize(data):
    ''' Normalize by the maximum current '''
    dataout = type(data)()
    maxI = np.max(data['I'])
    dataout['I'] = data['I'] / maxI
    add_missing_keys(data, dataout)

    return dataout

def add_missing_keys(datain, dataout):
    for k in datain.keys():
        if k not in dataout.keys():
            dataout[k] = datain[k]

# These are not needed for pandas types obviously
@ivfunc
def apply(data, func, column):
    '''
    This applies func to one column of the ivloop, and leaves the rest the same.
    func should take an array and return an array of the same size
    '''
    dataout = type(data)()
    dataout[column] = func(dataout[column])
    add_missing_keys(data, dataout)
    return dataout

def insert(data, key, vals):
    # Insert values into ivloop objects
    for d,v in zip(data, vals):
        d[key] = v

def extract(data, key):
    # Get array of values from ivloop objects
    return array([d[key] for d in data])
