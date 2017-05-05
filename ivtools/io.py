""" Functions for saving and loading data """
# TODO: def read_txt, and make read_txts call it repeatedly
from . import dotdict
import os
import re
import fnmatch
from pandas import read_csv
import sys
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

# Current datatype uses a small subclass of dict "dotdict"
# This class lives in ivtools, and is not installed to normal package directory
# Need to do this fancy thing so that pickle recognizes it
sys.modules['dotdict'] = dotdict
# It's better, though, to just convert all the dotdicts back to normal dicts so that
# loading the data does not depend on having this module at all.

def read_pickle(fp):
    ''' Read data from a pickle file '''
    with open(fp, 'rb') as f:
        normaldict = pickle.load(f)
    # Convert all the normal dicts to dotdicts
    out = dotdict.dotdict(normaldict)
    out['iv'] = np.array([dotdict.dotdict(l) for l in normaldict['iv']])
    return out


def write_pickle(data, fp):
    ''' Write data to a pickle file '''
    # Convert all dotdicts into normal dicts
    # Hopefully this does not take a huge amount of time
    normaldict = dict(data)
    normaldict['iv'] = np.array([dict(dd) for dd in data.iv])
    with open(fp, 'wb') as f:
        pickle.dump(normaldict, f)


def read_txts(directory, pattern, **kwargs):
    ''' Load list of loops from separate text files. Specify files by glob
    pattern.  kwargs are passed to loadtxt'''
    fnames = fnmatch.filter(os.listdir(directory), pattern)

    # Try to sort by file number, even if fixed width numbers are not used
    # For now I will assume the filename ends in _(somenumber)
    try:
        fnames.sort(key=lambda fn: int(splitext(fn.split('_')[-1])[0]))
    except:
        print('Failed to sort files by file number')

    print('Loading the following files:')
    print('\n'.join(fnames))

    fpaths = [pjoin(directory, fn) for fn in fnames]

    ### List of np arrays version ..
    # Load all the data
    # loadtxt_args = {'unpack':True,
    #                 'usecols':(0,1),
    #                 'delimiter':'\t',
    #                 'skiprows':1}
    # loadtxt_args.update(kwargs)
    # return [np.loadtxt(fp, **loadtxt_args) for fp in fpaths]

    ### Array of DataFrames version
    readcsv_args = dict(sep='\t', decimal='.')
    readcsv_args.update(kwargs)
    def txt_iter():
        # Iterate through text files, load data, and modify in some way
        # Using pandas here only because its read_csv can handle comma decimals easily..
        # Will convert back to numpy arrays.
        for fp in fpaths:
            # TODO: Guess which column has Voltage and Current based on various
            # different names people give them.  Here it seems the situation
            # is very bad and sometimes there are no delimiters in the header.
            # Even this is not consistent.

            # For now, read the first row and try to make sense of it
            with open(fp, 'r') as f:
                header = f.readline()
                if header == '*********** I(V) ***********\n':
                    skiprows = 8
                    for _ in range(7):
                        header = f.readline()
                else:
                    skiprows = 1
                # Try to split it by the normal delimiter
                splitheader = header.split(readcsv_args['sep'])
                if len(splitheader) > 1:
                    # Probably these are the column names?
                    colnames = splitheader
                else:
                    # The other format that I have seen is like 'col name [unit]'
                    # with a random number of spaces interspersed. Split after ].
                    colnames = re.findall('[^\]]+\]', header)
                    colnames = [c.strip() for c in colnames]

            df = read_csv(fp, skiprows=skiprows, names=colnames, index_col=False, **readcsv_args)

            # These will be recognized as the Voltage and Current columns
            Vnames = ['Voltage Source (V)', 'Voltage [V]']
            Inames = ['Current Probe (A)', 'Current [A]']
            # Rename columns
            dfcols = df.columns
            if 'V' not in dfcols:
                for Vn in Vnames:
                    if Vn in dfcols:
                        df.rename(columns={Vn:'V'}, inplace=True)
            if 'I' not in dfcols:
                for In in Inames:
                    if In in dfcols:
                        df.rename(columns={In:'I'}, inplace=True)
            yield df
    # Have to make an intermediate list?  Hopefully this does not take too much time/memory
    # Probably it is not a lot of data if it came from a csv ....
    # This doesn't work because it tries to cast each dataframe into an array first ...
    #return np.array(list(txt_iter()))
    #return (list(txt_iter()), dict(source_directory=directory), [dict(filepath=fp) for fp in fpaths])
    datalist = []
    for i, (fp, df) in enumerate(zip(fpaths, txt_iter())):
        mtime = os.path.getmtime(fp)
        ctime = os.path.getctime(fp)
        longnames = {'I':'Current', 'V':'Voltage'}
        units = {'I':'A', 'V':'V'}
        dd = dotdict(I=np.array(df['I']), V=np.array(df['V']), filepath=fp,
                     mtime=mtime, ctime=ctime, units=units, longnames=longnames,
                     index=i)
        datalist.append(dd)
    iv = np.array(datalist)

    # regular dict version
    #iv = np.array([{'I':np.array(df['I']), 'V':np.array(df['V']), 'filepath':fp} for fp, df in zip(fpaths, txt_iter())])
    return dotdict(iv=iv, source_dir=directory)
