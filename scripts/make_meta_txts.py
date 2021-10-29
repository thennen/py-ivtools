# Because I'm apparently too busy to make my program write to a proper database, my solution is to maintain a bunch of metadata text files, then combine them periodically.  This is dumb.  It is nice, though, to have simple text files that describe the data set in the same folder as the data files

# This script walks through the data directory, loads every data file into memory just so it can read the non-array data, then it writes a txt file containing that information.

# when done with that, it loads all the txt files it wrote and combines them into one database, that can be searched to make sense of what measurements were done.


# Update 2021-10-03 we have a database now but it's inconvenient enough to query it that I still want txts next to the datafiles
# working on box drive, needs an option to target a specific directory otherwise we will download countless gigabytes

import os
pjoin = os.path.join
import pandas as pd
import numpy as np
from shutil import copyfile
import fnmatch
import sys

# Not used - just keep the metadata with the data file
#metatxtfolder = 'D:/t/ivdata/metatxts'
#if not os.path.isdir(metatxtfolder):
#    os.makedirs(metatxtfolder)

ignore = ['metatxts', 'hgst_reram_cycles']

# I added an important column 'die_rel' at some point
# Need to create it if it wasn't in some of the earlier data files
lassen = pd.read_pickle('../ivtools/sampledata/all_lassen_device_info.pkl')
# Get this mapping
dierel = lassen[['die', 'die_rel']]

colnamemap = {'Ic': ['Icomp', 'CC', 'CComp'],
              'cr': ['cr_%'],
              'dep_date': ['date'],
              'dep_code': ['deposition_code'],
              'dep_temp': ['temperature'],
              't': ['t[s]'],
              'datafilepath': ['filepath']}

def newcolnames(old):
    new = []
    for o in old:
        rename = False
        for k,v in colnamemap.items():
            if (o in v) and (k not in old):
                new.append(k)
                rename = True
        if not rename:
            new.append(o)
    return new


def create_meta_txts(datadir=r'D:/t/ivdata/'):
    for root, folders, files in os.walk(datadir):
        folders[:] = [fo for fo in folders if fo not in ignore]
        for f in files:
            metafile = pjoin(root, os.path.splitext(f)[0] + '.meta')
            if os.path.exists(metafile):
                continue
            if f.endswith('.s'):
                try:
                    s = pd.read_pickle(pjoin(root, f))
                except:
                    continue
            elif f.endswith('.df'):
                try:
                    df = pd.read_pickle(pjoin(root, f))
                except:
                    continue
                # Only save first row metadata -- should be the same for all
                s = df.iloc[0]
                if 'nloops' not in s:
                    s['nloops'] = len(df)
            else:
                continue
            # Drop all arrays from data
            arrays = s[s.apply(type) == np.ndarray].index
            # Update filepath
            if 'datafilepath' not in s:
                s['datafilepath'] = os.path.abspath(pjoin(root, f))
            # Rename columns
            s.index = newcolnames(s.index)
            s = s.drop(arrays)
            # Make sure die_rel has a value
            if ('die' in s) and ('die_rel' not in s):
                s['die_rel'] = dierel['die_rel'].iloc[np.where(dierel['die'] == int(s['die']))[0][0]]

            # Write file
            s.to_csv(metafile, sep='\t', encoding='utf-8')
            print('Wrote ' + metafile)

def collect_all_meta_txts(datadir=r'D:/t/ivdata/'):
    # Go back and read all the metadata txts and write them to one file
    metas = []
    for root, folders, files in os.walk(datadir):
        folders[:] = [fo for fo in folders if fo not in ignore]
        for f in files:
            metafile = pjoin(root, os.path.splitext(f)[0] + '.meta')
            if f.endswith('.meta'):
                # bad idea
                #copyfile(metafile, os.path.join(metatxtfolder, f))
                # insert contents into dataframe
                with open(pjoin(root, f), 'r') as thisfile:
                    # Using file pointer because from_csv pukes if you have utf-8 characters in the filename
                    meta = pd.Series.from_csv(thisfile, sep='\t')
                if ('die' in meta.dropna()) and ('die_rel' not in meta.dropna()):
                    meta['die_rel'] = dierel['die_rel'].iloc[np.where(dierel['die'] == int(meta['die']))[0][0]]
                metas.append(meta)


    # Drop any duplicated keys that may exist in each series...
    metadf = pd.DataFrame([m[~m.index.duplicated()] for m in metas])
    #metadf = pd.DataFrame(metas)
    # Excel because people might be retarded -- takes a long time
    #metadf.to_excel('D:/t/ivdata/metadata.xls', encoding='utf-8')
    outpath = os.path.join(datadir, 'metadata.pkl')
    print(f'writing {outpath}')
    metadf.to_pickle(outpath)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        create_meta_txts(sys.argv[1])
    else:
        create_meta_txts()
