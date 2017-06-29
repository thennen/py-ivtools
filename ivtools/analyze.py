""" Functions for doing data analysis on IV data """

from functools import wraps
import numpy as np
from itertools import groupby
from .dotdict import dotdict

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
        if dtype == np.ndarray:
            # Assuming it's an ndarray of iv dicts
            return np.array([func(d, *args, **kwargs) for d in data])
        elif dtype == dotdict:
            return(func(data, *args, **kwargs))
        else:
            print('ivfunc did not understand the input datatype {}'.format(dtype))
    return func_wrapper


@ivfunc
def moving_avg(data, window=5):
    ''' Smooth data with moving avg '''
    V = data['V']
    I = data['I']
    lenV = len(V)
    lenI = len(I)
    if lenI != lenV:
        print('I and V arrays have different length!')
        return data
    if lenI == 0:
        return data
    weights = np.repeat(1.0, window)/window
    smoothV = np.convolve(V, weights, 'valid')
    smoothI = np.convolve(I, weights, 'valid')

    new_data = data.copy()
    new_data.update({'I':smoothI, 'V':smoothV})
    return new_data


@ivfunc
def index_iv(data, index_function):
    '''
    Index all the data arrays inside an iv loop container at once.
    Condition specified by index function, which should take an iv dict and return an indexing array
    '''
    # Determine the arrays that will be split
    # We will select them now just based on which values are arrays with same size as I and V
    lenI = len(data['I'])
    splitkeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    dataout = data.copy()
    for sk in splitkeys:
        # Apply the filter to all the relevant items
        index = np.array(index_function(data))
        dataout[sk] = dataout[sk][index]
    return dataout

@ivfunc
def slice_iv(data, stop, start=0, step=None):
    '''
    Slice all the data arrays inside an iv loop container at once.
    start, stop can be functions that take the iv loop as argument
    if those functions return nan, start defaults to 0 and stop to -1
    '''
    lenI = len(data['I'])
    splitkeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    dataout = data.copy()
    if callable(start):
        start = start(data)
        if np.isnan(start): start = 0
    if callable(stop):
        stop = stop(data)
        if np.isnan(stop): stop = -1
    for sk in splitkeys:
        # Apply the filter to all the relevant items
        dataout[sk] = dataout[sk][slice(start, stop, step)]
    return dataout


@ivfunc
def apply(data, func, column):
    '''
    This applies func to one column of the ivloop, and leaves the rest the same.
    func should take an array and return an array of the same size
    '''
    dataout = data.copy()
    dataout[column] = func(dataout[column])
    return dataout

def insert(data, key, vals):
    # Insert values into ivloop objects
    for d,v in zip(data, vals):
        d[key] = v

def extract(data, key):
    # Get array of values from ivloop objects
    return array([d[key] for d in data])

@ivfunc
def dV_sign(iv):
    '''
    Return boolean array indicating if V is increasing, decreasing, or constant.
    Will not handle noisy data.  Have to dig up the code that I wrote to do that.
    '''
    direction = np.sign(np.diff(iv['V']))
    # Need the same size array as started with. Categorize the last point same as previous 
    return np.append(direction, direction[-1])

@ivfunc
def decreasing(iv):
    return index_iv(iv, lambda l: dV_sign(iv) < 0)


@ivfunc
def increasing(iv):
    return index_iv(iv, lambda l: dV_sign(iv) > 0)


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

    dataout = data.copy()
    for k in keys:
        dataout[k] = dataout[k][startind:endind][::direction]

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
    loop['first_jump'] = first_jump
    return first_jump

@ivfunc
def last_jump(loop, **kwargs):
    j = jumps(loop, **kwargs)
    if np.any(j):
        last_jump = j[0][-1]
    else:
        last_jump = np.nan
    loop['last_jump'] = last_jump
    return last_jump 


def pindex(loops, column, index):
    # Index some column of all the ivloops in parallel
    # "index" is a list of indices with same len as loops
    # Understands list[nan] --> nan
    # TODO: index by a number contained in the ivloop object
    vals = []
    for l,i in zip(loops, index):
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

    dataout = data.copy()
    for k in keys:
        dataout[k] = dataout[k][startind:endind][::direction]

    return dataout


@ivfunc
def normalize(data):
    ''' Normalize by the maximum current '''
    dataout = data.copy()
    maxI = np.max(data['I'])
    dataout['I'] = dataout['I'] / maxI
    return dataout


def split(data):
    ''' Split one loop into many loops '''
    pass


def pico_to_iv(datain):
    ''' Convert picoscope channel data to IV structure '''
    # TODO: A lot

    # Keep all the data from picoscope
    dataout = dotdict(datain)
    A = datain['A']
    B = datain['B']
    #C = datain['C']
    R = 5e3
    dataout['V'] = A
    #dataout['I'] = 1e3 * (B - C) / R
    dataout['I'] = 1e3 * B / R
    dataout['units'] = {'V':'V', 'I':'mA'}
    return dataout
