""" Functions for saving and loading data """
from dotdict import dotdict
import os
import re
import fnmatch
import pandas as pd
import sys
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as spio
from scipy.io import savemat
pjoin = os.path.join
splitext = os.path.splitext

def validvarname(varStr):
    # Make valid variable name from string
    sub_ = re.sub('\W|^(?=\d)','_', varStr)
    sub_strip = sub_.strip('_')
    if sub_strip[0].isdigit():
       # Can't start with a digit
       sub_strip = 'm_' + sub_strip
    return sub_strip


def getGitRevision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        # Either there is no git or you are not in the py-ivtools directory.
        # Don't error because of this
        return 'Dunno'

def read_pickle(fp):
    ''' Read data from a pickle file '''
    import dotdict
    # Current datatype uses a small subclass of dict "dotdict"
    # This class lives in ivtools, and is not installed to normal package directory
    # Need to do this fancy thing so that pickle recognizes it
    sys.modules['dotdict'] = dotdict
    # It's better, though, to just convert all the dotdicts back to normal dicts so that
    # loading the data does not depend on having this module at all.
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


def read_txt(filepath, **kwargs):
    '''
    Function to read IV data from text files, trying to accomodate all the ridiculous formats I have encountered at IWEII.
    All columns will be loaded, but can be renamed to a standard value such as 'I' and 'V', or to a more reasonable name.
    Return pd.Series with additional information, like mtime, filepath, etc.
    kwargs passes through to pd.readcsv, so you can pass decimal = ',' or sep=' ', etc..
    '''
    # Different headers
    # Decimal commas instead of decimal points
    # Inconsistent use of delimiters within one file
    # Delimiters in the column names
    # ...

    # Here is a dict which constructs a mapping between various column names I have seen and a standard column name
    colnamemap = {'I': ['Current Probe (A)', 'Current [A]'],
                  'V': ['Voltage Source (V)', 'Voltage [V]'],
                  'T': ['Temperature  (K)', 'Temperature', 'Temperature [K]'],
                  't': ['time', 'Time [S]'],
                  'Vmeasured': ['Voltage Probe (V)']}

    # Default arguments for readcsv
    readcsv_args = dict(sep='\t', decimal='.')
    readcsv_args.update(kwargs)

    # Need to do different things for different file formats
    # Can't rely on file extensions.
    # For now, read the first row of the file to determine readcsv arguments
    with open(filepath, 'r') as f:
        firstline = f.readline()
        if firstline == '*********** I(V) ***********\n':
            # From Lakeshore labview monstrosity
            readcsv_args['skiprows'] = 8
            more_header = []
            for _ in range(7):
                more_header.append(f.readline())
            header = [firstline]
            header.extend(more_header)
            # Save this line to parse later
            colname_line = more_header[-1]
            # Single string version
            header = ''.join(header)
        else:
            # Assume first row contains column names
            # But cannot trust that they are properly delimited
            readcsv_args['skiprows'] = 1
            colname_line = firstline
            header = firstline

        # Try to split the colname line by the normal delimiter
        splitnames = colname_line.strip().split(readcsv_args['sep'])

        if len(splitnames) > 1:
            # Probably these are the column names?
            readcsv_args['names'] = splitnames
        else:
            # Another format that I have seen is like 'col name [unit]'
            # with a random number of spaces interspersed. Split after ].
            colnames = re.findall('[^\]]+\]', header)
            readcsv_args['names'] = [c.strip() for c in colnames]

    df = pd.read_csv(filepath, index_col=False, **readcsv_args)

    # Rename recognized columns to standard names
    dfcols = df.columns
    for k in colnamemap:
        if k not in dfcols:
            # If a column is not already named with the standard name
            for altname in colnamemap[k]:
                if altname in dfcols:
                    df.rename(columns={altname:k}, inplace=True)

    # My preferred format for a single IV loop is a dict with arrays and scalars and whatever else
    # Pandas equivalent is a pd.Series.

    longnames = {'I':'Current', 'V':'Voltage', 't':'Time', 'T':'Temperature'}
    # Note that the unit names are simply assumed here -- no attempt to read the units from the file
    units = {'I':'A', 'V':'V', 't':'s', 'T':'K'}

    dataout = {k:df[k].as_matrix() for k in df.columns}
    dataout['mtime'] = os.path.getmtime(filepath)
    dataout['units'] = {k:v for k,v in units.items() if k in dataout.keys()}
    dataout['longnames'] = {k:v for k,v in longnames.items() if k in dataout.keys()}
    dataout['filepath'] = os.path.abspath(filepath)
    dataout['header'] = header

    # Replace Keithley nan values with real nans
    nanmask = dataout['I'] == 9.9100000000000005e+37
    dataout['I'][nanmask] = np.nan

    return pd.Series(dataout)

def read_txts(directory, pattern='*', exclude=None, **kwargs):
    ''' Load list of loops from separate text files. Specify files by glob
    pattern.  kwargs are passed to loadtxt'''
    pattern = pattern.join('**')
    fnames = fnmatch.filter(os.listdir(directory), pattern)
    if exclude is not None:
        exclude = exclude.join('**')
        excludefiles = fnmatch.filter(fnames, exclude)
        fnames = [mf for mf in fnames if mf not in excludefiles]

    # Try to sort by file number, even if fixed width numbers are not used
    # For now I will assume the filename ends in _(somenumber)
    try:
        fnames.sort(key=lambda fn: int(splitext(fn.split('_')[-1])[0]))
    except:
        print('Failed to sort files by file number. Sorting by mtime instead.')
        fnames.sort(key=lambda fn: os.path.getmtime(pjoin(directory, fn)))

    print('Loading the following files:')
    print('\n'.join(fnames))

    fpaths = [pjoin(directory, fn) for fn in fnames]

    datalist = []
    for fp in fpaths:
        datalist.append(read_txt(fp, **kwargs))

    return pd.DataFrame(datalist)


def read_pandas_files(filepaths, concat=True, dropcols=None):
    '''
    Load in dataframes and/or series in list of filepaths
    return concatenated dataframe
    series will all have index 0 ...
    '''
    pdlist = []
    # Try to get pandas to read the files, but don't give up if some fail
    for f in filepaths:
        try:
            # pdlist may have some combination of Series and DataFrames.  Series should be rows
            pdobject = pd.read_pickle(f)
        except:
            print('Failed to interpret {} as a pandas pickle!'.format(f))
            continue

        if type(pdobject) is pd.DataFrame:
            if dropcols is not None:
                realdropcols = [dc for dc in dropcols if dc in pdobject]
                pdobject = pdobject.drop(realdropcols, 1)
            pdlist.append(pdobject)
        elif type(pdobject) is pd.Series:
            if dropcols is not None:
                realdropcols = [dc for dc in dropcols if dc in pdobject]
                pdobject = pdobject.drop(realdropcols)
            # Took me a while to figure out how to convert series into single row dataframe
            pdlist.append(pd.DataFrame.from_records([pdobject]))
            # This resets all the datatypes to object !!
            #pdlist.append(pd.DataFrame(pdobject).transpose())
        else:
            print('Do not know wtf this file is:')
        print('Loaded {}.'.format(f))
    if concat:
        return pd.concat(pdlist)
    else:
        return pdlist

def read_pandas_glob(directory='.', pattern='*', exclude=None, concat=True):
    '''
    Load in all dataframes and series matching a glob pattern
    return concatenated dataframe
    '''
    # Put wildcards at the ends of pattern
    pattern = pattern.join('**')
    files = os.listdir(directory)
    matchfiles = fnmatch.filter(files, pattern)
    if exclude is not None:
        exclude = exclude.join('**')
        excludefiles = fnmatch.filter(matchfiles, exclude)
        matchfiles = [mf for mf in matchfiles if mf not in excludefiles]
    matchfilepaths = [os.path.join(directory, f) for f in matchfiles]

    return read_pandas_files(matchfilepaths, concat=concat)

def read_pandas_recent(directory='.', pastseconds=60, concat=True):
    ''' Read files in directory which were made in the last pastseconds '''
    now = time.time()
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory)]
    ctimes = [os.path.getctime(fp) for fp in filepaths]
    recentfps = [fp for fp,ct in zip(filepaths, ctimes) if now - ct < pastseconds]
    return read_pandas_files(recentfps, concat=concat)


def write_matlab(data, filepath, varname=None, compress=True):
   # Write dict, list of dict, series, or dataframe to matlab format for the neanderthals
   # Haven't figured out what sucks less to work with in matlab
   # Each IV loop is a struct, has to be
   # For multiple IV loops, can either make a cell array of structs (plot(cell{1,1}.V, cell{1,1}.I))
   # Or just dump a whole bunch of structs into the namespace (plot(loop1.V, loop1.I))
   if varname is None:
      varname = validvarname(splitext(os.path.split(filepath)[-1])[0])
      print(varname)
   dtype = type(data)
   if dtype is list:
      savemat(filepath, {varname:data}, do_compression=compress)
   elif dtype is dict:
      # This will dump a bunch of names into namespace unless encapsulated in a list
      savemat(filepath, {varname:[data]}, do_compression=compress)
   elif dtype is pd.Series:
      # Same
      savemat(filepath, {varname:[dict(data)]}, do_compression=compress)
   elif dtype is pd.DataFrame:
      savemat(filepath, {varname: data.to_dict('records')}, do_compression=compress)

def read_matlab(filepath):
   # Read matlab file into dataframe or series
   '''
   These functions solve the problem of not properly recovering python dictionaries
   from mat files. It calls the function check keys to cure all entries
   which are still mat-objects

   Stolen from
   https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
   Tyler's edit only recurses into level 0 np.arrays
   '''
   def _check_keys(d):
      '''
      checks if entries in dictionary are mat-objects. If yes
      todict is called to change them to nested dictionaries
      '''
      for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
               d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
               d[key] = _tolist(d[key])
      return d

   def _todict(matobj):
      '''
      A recursive function which constructs from matobjects nested dictionaries
      '''
      d = {}
      for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
               d[strg] = _todict(elem)
            # Don't do this, in my case I want to preserve nd.arrays that are not lists containing dicts
            #elif isinstance(elem, np.ndarray):
            #    d[strg] = _tolist(elem)
            else:
               d[strg] = elem
      return d

   def _tolist(ndarray):
      '''
      A recursive function which constructs lists from cellarrays
      (which are loaded as numpy ndarrays), recursing into the elements
      if they contain matobjects.
      '''
      elem_list = []
      for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
               elem_list.append(_todict(sub_elem))
            # Only the first level -- in case it's a list of dicts with np arrays
            # Better not be a list of dicts of list of dicts ....
            #elif isinstance(sub_elem, np.ndarray):
            #    elem_list.append(_tolist(sub_elem))
            else:
               elem_list.append(sub_elem)
      return elem_list

   # Squeeze me gets rid of dimensions that have length 1
   # So if you saved a 1x1 cell array, you just get back the element
   mat_in = spio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
   mat_in = _check_keys(mat_in)
   # Should only be one key
   mat_vars = [k for k in mat_in.keys() if not k.startswith('__')]
   if len(mat_vars) > 1:
      print('More than one matlab variable stored in {}. Returning dict.'.format(filepath))
      return mat_in
   else:
      # List of dicts
      #return mat_in[mat_vars[0]]
      # DataFrame
      var_in = mat_in[mat_vars[0]]
      if type(var_in) is list:
         # More than one loop
         return pd.DataFrame(var_in)
      else:
         return pd.Series(var_in)


def write_csv(data, filepath, columns=['I', 'V']):
    # For true dinosaurs
    pass

def write_meta_csv(data, filepath):
    ''' Write the non-array data to a text file.  Only first row of dataframe considered!'''
    dtype = type(data)
    if dtype is pd.Series:
        s = pd.read_pickle(pjoin(root, f))
    elif dtype is pd.Dataframe:
        # Only save first row metadata -- Usually it's the same for all
        df = pd.read_pickle(pjoin(root, f))
        s = df.iloc[0]
        s['nloops'] = len(df)
    elif dtype is list:
        s = pd.Series(data[0])
    elif dtype is dict:
        s = pd.Series(data)
    # Drop all arrays from data
    arrays = s[s.apply(type) == np.ndarray].index
    s.drop(arrays).to_csv(filepath, sep='\t', encoding='utf-8')


def plot_datafiles(datadir, maxloops=500, x='V', y='I', smoothpercent=1):
   # Make a plot of all the .s and .df files in a directory
   # Save as pngs with the same name
   # TODO: Optionally group by sample, making one plot per sample
   files = os.listdir(datadir)
   series_fns = [pjoin(datadir, f) for f in files if f.endswith('.s')]
   dataframe_fns = [pjoin(datadir, f) for f in files if f.endswith('.df')]

   fig, ax = plt.subplots()

   for sfn in series_fns:
      s = pd.read_pickle(sfn)
      s.I *= 1e6
      s.units['I'] = '$\mu$A'
      smoothn = max(int(smoothpercent * len(s.V) / 100), 1)
      plotiv(moving_avg(s, smoothn, columns=None), x=x, y=y, ax=ax)
      pngfn = sfn[:-2] + '.png'
      pngfp = os.path.join(datadir, pngfn)
      if 'thickness_1' in s:
          plt.title('{}, Width={}nm, Thickness={}nm'.format(s['layer_1'], s['width_nm'], s['thickness_1']))
      plt.savefig(pngfp)
      print('Wrote {}'.format(pngfp))
      ax.cla()

   for dffn in dataframe_fns:
      df = pd.read_pickle(dffn)
      df.I *= 1e6
      df['units'] = len(df) * [{'V':'V', 'I':'$\mu$A'}]
      step = int(ceil(len(df) / maxloops))
      smoothn = max(int(smoothpercent * len(df.iloc[0].V) / 100), 1)
      plotiv(moving_avg(df[::step], smoothn), alpha=.6, ax=ax)
      pngfn = dffn[:-3] + '.png'
      pngfp = os.path.join(datadir, pngfn)
      s = df.iloc[0]
      if 'thickness_1' in s:
          plt.title('{}, Width={}nm, Thickness={}nm'.format(s['layer_1'], s['width_nm'], s['thickness_1']))
      plt.savefig(pngfp)
      print('Wrote {}'.format(pngfp))
      ax.cla()

   plt.close(fig)

def change_devicemeta(filepath, newmeta, deleteold=False):
    ''' For when you accidentally write a file with the wrong sample information attached '''
    filedir, filename = os.path.split(filepath)
    filename, extension = os.path.splitext(filename)
    datain = pd.read_pickle(filepath)
    if type(datain) == pd.Series:
        datain[newmeta.index] = newmeta
        s = datain
    if type(datain) == pd.DataFrame:
        datain[newmeta.index] = pd.DataFrame([newmeta] * len(datain)).reset_index(drop=True)
        s = datain.iloc[0]
    # Retain time information in filename
    newfilename = filename[:21]
    for fnkey in filenamekeys:
        if fnkey in s.index:
            newfilename += '_{}'.format(s[fnkey])
    newpath = os.path.join(filedir, newfilename + extension)
    print('writing new file {}'.format(newpath))
    datain.to_pickle(newpath)
    if deleteold:
        print('deleting old file {}'.format(filepath))
        os.remove(filepath)


def writefig(filename, subdir='', plotdir='Plots', overwrite=True, savefig=False):
    # write the current figure to disk
    # Can also write a pickle of the figure
    plotsubdir = os.path.join(plotdir, subdir)
    if not os.path.isdir(plotsubdir):
        os.makedirs(plotsubdir)
    plotfp = os.path.join(plotsubdir, filename)
    if os.path.isfile(plotfp + '.png') and not overwrite:
        print('Not overwriting {}'.format(plotfp))
    else:
        plt.savefig(plotfp)
        print('Wrote {}.png'.format(plotfp))
        if savefig:
            with open(plotfp + '.plt', 'wb') as f:
                pickle.dump(plt.gcf(), f)
            print('Wrote {}.plt'.format(plotfp))
