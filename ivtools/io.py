""" Functions for saving and loading data """
# TODO: def read_txt, and make read_txts call it repeatedly
from dotdict import dotdict
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
import scipy.io as spio
from scipy.io import savemat
pjoin = os.path.join
splitext = os.path.splitext

# Make valid variable name from string
def validvarname(varStr):
    sub_ = re.sub('\W|^(?=\d)','_', varStr)
    sub_strip = sub_.strip('_')
    if sub_strip[0].isdigit():
       # Can't start with a digit
       sub_strip = 'm_' + sub_strip
    return sub_strip

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


def read_txts(directory, pattern, exclude=None, **kwargs):
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
        print('Failed to sort files by file number. Sorting by mtime.')
        fnames.sort(key=lambda fn: os.path.getctime(pjoin(directory, fn)))

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
        dd = dict(I=np.array(df['I']), V=np.array(df['V']), filepath=fp,
                  mtime=mtime, ctime=ctime, units=units, longnames=longnames,
                  index=i)
        datalist.append(dd)

    # regular dict version
    #iv = np.array([{'I':np.array(df['I']), 'V':np.array(df['V']), 'filepath':fp} for fp, df in zip(fpaths, txt_iter())])
    #return dotdict(iv=iv, source_dir=directory)
    return pd.DataFrame(datalist)


def read_pandas(directory='.', pattern='*', exclude=None, concat=True):
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
    pdlist = []
    # Try to get pandas to read the files, but don't give up if some fail
    for f in matchfiles:
        fp = os.path.join(directory, f)
        try:
            # pdlist may have some combination of Series and DataFrames.  Series should be rows
            pdobject = pd.read_pickle(fp)
            if type(pdobject) is pd.DataFrame:
                pdlist.append(pdobject)
            elif type(pdobject) is pd.Series:
                # Took me a while to figure out how to convert series into single row dataframe
                pdlist.append(pd.DataFrame.from_records([pdobject]))
                # This resets all the datatypes to object !!
                #pdlist.append(pd.DataFrame(pdobject).transpose())
            else:
                print('Do not know wtf this file is:')
            print('Loaded {}.'.format(f))
        except:
            print('Failed to interpret {} as a pandas pickle!'.format(f))
    if concat:
        return pd.concat(pdlist)
    else:
        return pdlist


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
