''' Functions for saving and loading data '''
import fnmatch
import os
import re
import subprocess
import sys
import time
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# we don't make heavy use of these other modules
# don't reference them on the top level
# this is to avoid circular import problems
import ivtools.analyze
import ivtools.plot
from ivtools import settings

try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as spio
from scipy.io import savemat
import sqlite3
import logging

from pathlib import Path

from PIL import Image


log = logging.getLogger('io')

pjoin = os.path.join
splitext = os.path.splitext
psplit = os.path.split

# py-ivtools git repo
# Directory above the one containing this file
repoDir = psplit(psplit(__file__)[0])[0]

# Find the git executable.  Should be on the path.
gitexe = shutil.which('git')
if gitexe is None:
    maybegit = 'C:\\Program Files\\Git\\cmd\\git.EXE'
    if os.path.isfile(maybegit):
        gitexe = maybegit


db_path = settings.db_path


class MetaHandler(object):
    '''
    Stores, indexes, and prints meta data (stored in dicts, or pd.Series)
    for attaching sample information to data files, with interactive use in mind.

    savedata method writes data along with the selected metadata to disk and makes a database entry

    Can generate filenames with timestamps and keys from the metadata

    df attribute holds the list of metadata as a list-of-dicts or pandas dataframe
    meta holds the currently selected row of metadata
    static holds additional metadata which will not cycle
    static values will override meta values if the keys collide

    __repr__ will print the concatenated meta and static

    you can set/get items directly on the MetaHandler instance instead of on its static attribute
    eg meta = MetaHandler(); meta['key'] = value

    Attach the currently selected meta data and the static meta data with the attach() function

    MetaHandler is Borg.  Its state lives in an separate module.
    This is so if the io module is reloaded, a new Metahandler instance keeps the metadata state
    '''

    def __init__(self, clear_state=False):
        statename = self.__class__.__name__
        if statename not in ivtools.class_states:
            ivtools.class_states[statename] = {}
        self.__dict__ = ivtools.class_states[statename]
        if not self.__dict__ or clear_state:
            self.clear()

    def clear(self):
        ''' Clear all the information from the MetaHandler instance '''
        # This stores the currently selected metadata.  It's a dict or a pd.Series
        # Go ahead and overwrite or modify it if you want
        self.meta = {}
        # This is the index of the selected data
        self.i = 0
        # This is the dataframe holding all of the metadata -- one row per device
        self.df = None
        # These key:values are always appended
        self.static = {}
        # This controls which keys will be used to construct a filename
        self.filenamekeys = []
        # TODO: This will be called with str.format
        # self.filenameformatter = None
        # These keys get printed when you step through the list of metadata
        self.prettykeys = []
        self.moduledir = os.path.split(__file__)[0]

    def __repr__(self):
        # return self.meta.__repr__()
        return pd.Series({**self.meta, **self.static}).__repr__()

    def asdict(self):
        '''
        meta.meta has the steppable metadata
        meta.static has the constant metadata
        this returns the combined one
        '''
        return {**self.meta, **self.static}

    def __iter__(self):
        # Implementing this because then you can just write dict(meta)
        for k, v in self.asdict().items():
            yield k, v

    def __getitem__(self, key):
        return self.asdict()[key]

    def __setitem__(self, key, value):
        # Set a key:value to the static part of the meta data
        self.static[key] = value

    def __delitem__(self, key):
        if key in self.meta:
            del self.meta[key]
        if key in self.static:
            del self.static[key]

    def select(self, i):
        # select the ith row of the metadataframe
        self.i = i
        if type(self.df) == pd.DataFrame:
            self.meta = dict(self.df.iloc[self.i])
        else:
            self.meta = self.df[self.i]

    def load_sample_table(self, fpath, sheet=0, header=0, skiprows=None, **filters):
        ''' load data (pd.read_excel) from some tabular format'''
        if not os.path.isfile(fpath):
            # Maybe it's a relative path
            fpath = os.path.join(self.moduledir, fpath)
        df = pd.read_excel(fpath, sheet, header=header, skiprows=skiprows)
        # TODO: Apply filters
        for name, value in filters.items():
            if name in df:
                if isinstance(value, str) or not hasattr(value, '__iter__'):
                    value = [value]
                df = df[df[name].isin(value)].dropna(axis=1, how='all')
            else:
                df[name] = [value] * len(df)
        filenamekeys = []
        if 'sample_name' in df:
            filenamekeys = ['sample_name'] + filenamekeys
        self.prettykeys = None
        self.df = df
        self.select(0)


    # TODO: Unified metadata loader that just loads every possible sample

    def load_nanoxbar(self, **kwargs):
        '''
        Load nanoxbar metadata
        use keys X, Y, width_nm, device
        Making no attempt to load sample information, because it's a huge machine unreadable excel file mess.
        all kwargs will just be added to all metadata
        # TODO add formatter for filename '{}_{}_{}_{}_{}' or whatever
        '''
        nanoxbarfile = os.path.join(self.moduledir, 'sampledata/nanoxbar.pkl')
        nanoxbar = pd.read_pickle(nanoxbarfile)
        devicemetalist = nanoxbar
        for name, value in kwargs.items():
            if name in nanoxbar:
                if not hasattr(value, '__iter__'):
                    value = [value]
                devicemetalist = devicemetalist[devicemetalist[name].isin(value)]
            else:
                devicemetalist[name] = [kwargs[name]] * len(devicemetalist)
        # filenamekeys = ['X', 'Y', 'width_nm', 'device']
        filenamekeys = ['id']
        if 'sample_name' in kwargs:
            filenamekeys = ['sample_name'] + filenamekeys
        self.df = devicemetalist
        self.select(0)
        self.prettykeys = filenamekeys
        self.filenamekeys = filenamekeys
        log.info('Loaded {} devices into metadata list'.format(len(devicemetalist)))
        self.print()

    def load_lassen(self, **kwargs):
        '''
        Load wafer information for Lassen
        if a key is specified which is in the deposition sheet, then try to merge in deposition data
        Specify lists of keys to match on. e.g. coupon=[23, 24], module=['001H', '014B']
        '''
        deposition_file = os.path.join(self.moduledir, 'sampledata/CeRAM_Depositions.xlsx')
        lassen_file = os.path.join(self.moduledir, 'sampledata/all_lassen_device_info.pkl')

        deposition_df = pd.read_excel(deposition_file, header=8, skiprows=[9])
        # Only use info for Lassen wafers
        deposition_df = deposition_df[deposition_df['wafer_code'] == 'Lassen']
        lassen_df = pd.read_pickle(lassen_file)
        # Merge data
        merge_deposition_data_on = ['coupon']

        # If someone neglected to write the coupon number in the deposition sheet
        # Merge the non-coupon specific portion of lassen_df
        coupon_cols = ['coupon', 'die_x', 'die_y', 'die']
        non_coupon_cols = [c for c in lassen_df.columns if c not in coupon_cols]
        non_coupon_specific = lassen_df[lassen_df.coupon == 42][non_coupon_cols]
        # sort breaks reverse compatibility with old pandas versions
        # not passing sort can cause an annoying warning
        lassen_df = pd.concat((lassen_df, non_coupon_specific))  # , sort=False)

        if any([(k in deposition_df) for k in kwargs.keys()]):
            meta_df = pd.merge(lassen_df, deposition_df, how='left', on=merge_deposition_data_on, sort=False)
        else:
            meta_df = lassen_df

        # Check that function got valid arguments
        for key, values in kwargs.items():
            if key not in meta_df.columns:
                raise Exception('Key must be in {}'.format(meta_df.columns))
            if isinstance(values, str) or not hasattr(values, '__iter__'):
                kwargs[key] = [values]

        #### Filter kwargs ####
        for key, values in kwargs.items():
            meta_df = meta_df[meta_df[key].isin(values)]
        #### Filter devices to be measured #####
        devices001 = [2, 3, 4, 5, 6, 7, 8]
        devices014 = [4, 5, 6, 7, 8, 9]
        meta_df = meta_df[~((meta_df.module_num == 1) & ~meta_df.device.isin(devices001))]
        meta_df = meta_df[~((meta_df.module_num == 14) & ~meta_df.device.isin(devices014))]
        meta_df = meta_df.dropna(axis=1, how='all')
        # Sort values so that they are in the same order as you would probe them
        # Which is a strange order, since the mask is a disaster
        sortby = [k for k in ('dep_code', 'sample_number', 'die_rel', 'wX', 'wY') if k in meta_df.columns]
        meta_df = meta_df.sort_values(by=sortby)

        # Try to convert data types
        typedict = dict(coupon=np.uint8,
                        sample_number=np.uint16,
                        number_of_dies=np.uint8,
                        cr=np.uint8,
                        thickness_1=np.uint16,
                        thickness_2=np.uint16,
                        dep_temp=np.uint16,
                        etch_time=np.float32,
                        etch_depth=np.float32)
        for k, v in typedict.items():
            if k in meta_df:
                # int arrays don't support missing data, because python sucks and computers suck
                if not any(meta_df[k].isnull()):
                    meta_df[k] = meta_df[k].astype(v)

        self.df = meta_df
        self.select(0)
        self.prettykeys = ['dep_code', 'sample_number', 'coupon', 'die_rel', 'module', 'device', 'width_nm', 'R_series',
                           'layer_1', 'thickness_1']
        self.filenamekeys = ['dep_code', 'sample_number', 'die_rel', 'module', 'device']
        log.info('Loaded metadata for {} devices'.format(len(self.df)))
        self.print()

    def load_DomeB(self, **kwargs):
        '''
        Load wafer information for DomeB
        if a key is specified which is in the deposition sheet, then try to merge in deposition data
        Specify lists of keys to match on. e.g. coupon=[23, 24], module=['001H', '014B']
        '''
        # Left to right, bottom to top!
        columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b',
                   'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                   'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AP', 'AQ',
                   'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                   'A7', 'A8', 'A9', 'A0']
        rows = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '00', '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
        deposition_file = os.path.join(self.moduledir, 'sampledata/CeRAM_Depositions.xlsx')
        domeB_file = os.path.join(self.moduledir, 'sampledata/domeB.pkl')

        deposition_df = pd.read_excel(deposition_file, header=8, skiprows=[9])
        # Only use info for DomeB wafers
        deposition_df = deposition_df[deposition_df['wafer_code'] == 'DomeB']
        domeB_df = pd.DataFrame(pd.read_pickle(domeB_file))
        # Merge data
        if any([(k in deposition_df) for k in kwargs.keys()]):
            # Why is cartesian merge not just available in pandas?
            domeB_df['key'] = 0
            deposition_df['key'] = 0
            meta_df = domeB_df.merge(deposition_df, how='outer', sort=False).drop(columns=['key'])
        else:
            meta_df = domeB_df

        # Check that function got valid arguments
        for key, values in kwargs.items():
            if key not in meta_df.columns:
                raise Exception('Key must be in {}'.format(meta_df.columns))
            if isinstance(values, str) or not hasattr(values, '__iter__'):
                kwargs[key] = [values]

        #### Filter kwargs ####
        for key, values in kwargs.items():
            meta_df = meta_df[meta_df[key].isin(values)]
        #### Filter devices to be measured #####
        meta_df = meta_df.dropna(axis=1, how='all')

        # Sort top to bottom, left to right
        meta_df['icol'] = meta_df.col.apply(columns.index)
        meta_df['irow'] = meta_df.row.apply(rows.index)
        meta_df = meta_df.sort_values(by=['icol', 'irow'], ascending=[True, False])  # .drop(columns=['icol', 'irow'])

        self.df = meta_df
        self.select(0)
        self.prettykeys = ['dep_code', 'sample_number', 'die_rel', 'row', 'col', 'Resistance', 'gap', 'radius']
        self.filenamekeys = ['dep_code', 'sample_number', 'row', 'col']
        log.info('Loaded metadata for {} devices'.format(len(self.df)))
        self.print()

    def move_domeb(self, direction='l'):
        '''
        Assumes you have domeB metadata loaded into self.df
        Lets you move left, right, up, down by passing l r u d
        For interactive use
        '''
        lastmeta = self.meta
        # Left to right, bottom to top!
        columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b',
                   'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                   'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AP', 'AQ',
                   'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                   'A7', 'A8', 'A9', 'A0']
        rows = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '00', '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
        col = self.meta.col
        row = self.meta.row
        icol = columns.index(col)
        irow = rows.index(row)
        # dumb loop because it's the first thing I thought of
        i = None
        while i is None:
            if direction.lower() in ('left', 'l'):
                icol -= 1
            if direction.lower() in ('right', 'r'):
                icol += 1
            if direction.lower() in ('up', 'u'):
                irow += 1
            if direction.lower() in ('down', 'd'):
                irow -= 1

            if (icol < 0) or (icol >= len(columns)) or (irow < 0) or (irow >= len(rows)):
                irow %= len(rows)
                icol %= len(col)
                log.warning('Went over edge of coupon -- wrapping around')
                #return
            newcol = columns[icol]
            newrow = rows[irow]
            w = np.where((self.df.col == newcol) & (self.df.row == newrow))[0]
            if any(w):
                i = w[0]
            else:
                # TODO: don't check every single row/column in between, this can print lots of times in a row
                log.warning('skipping a device that is not loaded into memory')
        self.select(i)


        # Highlight keys that have changed
        isnan = lambda x: isinstance(x, float) and np.isnan(x)
        hlkeys = []
        for key in self.meta.keys():
            if key not in lastmeta.keys() or self.meta[key] != lastmeta[key]:
                # don't count nan → nan as a changed value even though nan ≠ nan ..
                if not (isnan(self.meta[key]) and isnan(lastmeta[key])):
                    hlkeys.append(key)
        log.info('You have selected this device (index {}):'.format(self.i))
        # Print some information about the device
        self.print(hlkeys=hlkeys)

    def step(self, n):
        ''' Select the another device by taking a step through meta df '''
        lastmeta = self.meta
        meta_i = self.i + n
        if meta_i < 0:
            log.info('You are at the beginning of metadata list')
            return
        elif meta_i >= len(self.df):
            log.info('You are at the end of metadata list')
            return
        else:
            self.select(meta_i)

        # Highlight keys that have changed
        isnan = lambda x: isinstance(x, float) and np.isnan(x)
        hlkeys = []
        for key in self.meta.keys():
            if key not in lastmeta.keys() or self.meta[key] != lastmeta[key]:
                # don't count nan → nan as a changed value even though nan ≠ nan ..
                if not (isnan(self.meta[key]) and isnan(lastmeta[key])):
                    hlkeys.append(key)
        log.info('You have selected this device (index {}):'.format(self.i))
        # Print some information about the device
        self.print(hlkeys=hlkeys)

    def next(self):
        self.step(1)

    def previous(self):
        self.step(-1)

    def goto(self, **kwargs):
        ''' Assuming you loaded metadata already, this goes to the first row that matches the keys'''
        # TODO: if there is more than one device with those parameters, don't select the first, instead, raise exception
        mask = np.ones(len(self.df), bool)
        for k, v in kwargs.items():
            mask &= self.df[k] == v
        w = np.where(mask)
        if any(mask):
            i = w[0][0]
            self.select(i)
            log.debug('You have selected this device (index {}):'.format(self.i))
            return self.i
        else:
            log.error('No matching devices found')
            return None

    def print(self, keys=None, hlkeys=None):
        ''' Print the selected metadata '''
        # Print some information about the device
        if self.prettykeys is None or len(self.prettykeys) == 0:
            # Print all the information
            prettykeys = self.meta.keys()
        else:
            prettykeys = self.prettykeys
        for key in prettykeys:
            if key in self.meta.keys():
                if hlkeys is not None and key in hlkeys:
                    print('{:<18}\t{:<8} <----- Changed'.format(key[:18], self.meta[key]))
                else:
                    print('{:<18}\t{}'.format(key[:18], self.meta[key]))

    def attach(self, data):
        '''
        Attach the currently selected metadata to input data
        If the data is a "list" of data, this will append the metadata to all the elements
        Might overwrite existing keys!
        May modify the input data in addition to returning it
        # TODO make it always modify the input data, or never
        '''
        dataout = self.attach_keys(data, **self.meta)
        dataout = self.attach_keys(dataout, **self.static)
        return dataout

    def attach_keys(self, data, **kwargs):
        '''
        Return shallow copy of input data with metadata keys attached
        data can be list, list of dict, pd.Series, or pd.DataFrame
        '''
        dtype = type(data)
        if dtype is dict:
            dataout = {**data, **kwargs}
        elif dtype is pd.Series:
            dataout = pd.Series({**dict(data), **kwargs})
        elif dtype is list:
            # should be a list of dicts
            dataout = [{**d, **kwargs} for d in data]
        elif dtype is pd.DataFrame:
            # Faster than .iterrows()?
            datadict = data.to_dict(orient='records')
            dataout = pd.DataFrame([{**d, **kwargs} for d in datadict])
        return dataout

    def timestamp(self):
        return datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]

    def filename(self):
        ''' Create a timestamped filename from selected metadata '''
        filename = self.timestamp()
        for fnkey in self.filenamekeys:
            if fnkey in self.meta.keys():
                filename += '_{}'.format(self.meta[fnkey])
            elif fnkey in self.static.keys():
                filename += '_{}'.format(self.static[fnkey])
        return filename

    def savedata(self, data, folder_path=None, database_path=None, table_name='meta', drop=None):
        '''
        Save data + selected metadata to disk and write a row of metadata to an sqlite3 database

        :param data: Row of data to be add to the database.
        :param folder_path: Folder where all data will be saved. If None, data will be saved in Desktop.
        :param database_path: Path of the database where data will be saved. If None, data will be saved in Desktop.
        :param table_name: Name of the table in the database. If the table doesn't exist, create a new one.
        :param drop: drop columns to save disk space.
        '''

        # save in current directory by default
        if folder_path is None:
            folder_path = '.'
        if database_path is None:
            database_path = 'metadata.db'

        file_name = self.filename()
        file_path = os.path.abspath(os.path.join(folder_path, file_name))
        file_timestamp = file_name[:len(self.timestamp())]

        ext = pandas_pickle_extension(data)
        file_path += ext

        # could be some inefficiency here?
        data = self.attach(data)
        data = self.attach_keys(data, filepath=file_path)
        data = self.attach_keys(data, file_timestamp=file_timestamp)

        datatype = type(data)
        if datatype in (list, pd.DataFrame):
            # You are saving a list of data
            # only the first element will be used for saving the metadata
            metadata = ivtools.analyze.iloc(data, 0)
            rowtype = type(metadata)
            if rowtype not in (dict, pd.Series):
                raise Exception(f"List of {rowtype} is not compatible.")
        elif datatype in (dict, pd.Series):
            metadata = data
        else:
            raise Exception(f"Data type {datatype} is not compatible.")

        write_pandas_pickle(data, file_path, drop=drop)

        db_conn = db_connect(database_path)
        exist = db_exist_table(db_conn, table_name)
        if exist:
            db_insert_row(db_conn, table_name, metadata)
        else:
            db_create_table(db_conn, table_name, metadata)
        db_commit(db_conn)



###### Database stuff ######

def db_create_table(db_conn, table_name, data):
    '''
    Creates a table from the 'pandas.series' array of data.
    It names columns and inserts the first row of data.
    To apply this change, the function "db_commit()" must be ran.

    :param db_conn: Connection with the database estiblished by db_connect()
    :param table_name: Name of the table in the database.
    :param data: pandas.series array from which to create the table
    :return: None
    '''

    c = db_conn.cursor()

    # Creating table and naming columns.
    col_names = list(data.keys())

    # It is not possible to have two column names that only differ in case.
    # To solve that, '&' is added at the end of the second name
    col_names_encoded = db_encode(col_names)

    def blacklist_filter(col_name):
        val = data[col_name]
        if val is not None:
            val_ch = db_change_type(val)
        else:
            val_ch = None

        dtype = type(val)
        if val_ch is None:

            log.debug(f"Data type {dtype} is not allowed in database, '{col_name}' will be dropped.")

            return None
        else:
            return col_names_encoded[col_names.index(col_name)]

    # Here all empty columns are deleted, one could save them just removing the None filter below.
    params = tuple(filter(None, [blacklist_filter(name) for name in col_names]))
    col_names_encoded = list(params)
    col_names = db_decode(col_names_encoded)
    if len(params) == 0:
        raise Exception("An empty table can't be used to create a table")
    log.debug(f"CREATE TABLE {table_name} {params}")
    c.execute(f"CREATE TABLE {table_name} {params}")

    # Adding values to the first row
    params = tuple([db_change_type(data[col_name]) for col_name in col_names])
    qmarks = "(?" + ", ?" * (len(params) - 1) + ")"
    log.debug(f"INSERT INTO {table_name} VALUES {qmarks}", params)
    c.execute(f"INSERT INTO {table_name} VALUES {qmarks}", params)


def db_insert_row(db_conn, table_name, row):
    '''
    Insert a row of data of any length, creating new columns if necessary.
    To apply this change, the function "db_commit()" must be ran.

    :param db_conn: Connection with the database estiblished by db_connect()
    :param table_name: Name of the table in the database.
    :param row: Row to be added to the table.
    :return: None
    '''

    datatype = type(row)
    if datatype not in (dict, pd.core.series.Series):
        raise Exception(f'Data type {datatype} is not compatible. Use "dict" or "pandas.core.series.Series".')

    c = db_conn.cursor()

    prev_col_names_encoded = db_get_col_names(db_conn, table_name)
    prev_col_names = db_decode(prev_col_names_encoded)
    in_col_names = list(row.keys())

    # This loop fills the cells of the existing columns.
    def db_fill_cols(col_name):
        if col_name in in_col_names:
            val = row[col_name]
            dtype = type(val)
            val_ch = db_change_type(val)
            return val_ch
        else:
            return None

    params = [db_fill_cols(name) for name in prev_col_names]

    # This loop adds new columns if needed.
    prev_col_names_encoded_low = [i.lower() for i in prev_col_names_encoded]
    for name in in_col_names:
        if name not in prev_col_names:
            name_low = name.lower()
            name_encoded = name
            while name_low in prev_col_names_encoded_low:
                name_low += '&'
                name_encoded += '&'
            val = row[name]
            val_ch = db_change_type(val)
            if val_ch is not None:
                db_add_col(db_conn, table_name, name_encoded)

                log.debug(f"New column added: {name}")
                params.append(val_ch)
            else:
                dtype = type(val)
                NoneType = type(None)
                #if dtype is NoneType:
                #    log.debug(f"'{name}' is empty so won't be saved")
                #else:
                #    log.debug(f"Data type '{dtype}' not supported. '{name}' won't be saved")


    qmarks = "(?" + ", ?" * (len(params) - 1) + ")"
    params = tuple(params)

    # log.debug(f"INSERT INTO {table_name} VALUES {qmarks}", params)
    c.execute(f"INSERT INTO {table_name} VALUES {qmarks}", params)


def db_get_col_names(db_conn, table_name):
    '''
    Return a list with the name of the columns.

    :param db_conn: Connection with the database established by db_connect()
    :param table_name:
    :return: List of the names.
    '''

    c = db_conn.cursor()

    get_names = c.execute(f"select * from {table_name} limit 1")
    col_names = [i[0] for i in get_names.description]
    return list(col_names)


def db_add_col(db_conn, table_name, col_name):
    '''
    Add a new columns at the end of the table.

    :param db_conn: Connection with the database established by db_connect().
    :param table_name: Table i nthe database.
    :param col_name: Name of the new column
    :return: None
    '''

    c = db_conn.cursor()

    c.execute(f"ALTER TABLE {table_name} ADD '{col_name}'")


def db_change_type(var):
    '''
    Change the type of a variable to the best option to be in the database.

    :param var: variable to change its type
    :return: Changed variable
    '''
    types_dict = {np.ndarray: None, list: None, dict: str, pd._libs.tslibs.timestamps.Timestamp: str,
                  np.float64: float,
                  np.float32: float, np.int64: int, np.int32: int, np.int16: int, str: str, int: int, float: float,
                  np.uint8: int, np.uint16: int, np.uint32: int,
                  pd.core.series.Series: None}
    dtype = type(var)
    if dtype in types_dict.keys():
        if types_dict[dtype] == float:
            var = float(var)
        elif types_dict[dtype] == str:
            var = str(var)
        elif types_dict[dtype] == None:
            var = None
        elif types_dict[dtype] == datetime:
            var = var.to_pydatetime()
        elif types_dict[dtype] == int:
            var = int(var)
    elif var is None:
        pass
    else:
        log.debug(f"Data type {dtype} is not registered, it will be save as str")
        var = repr(var)
    return var


def db_load(db_path=db_path, table_name='meta'):
    '''
    Load a dataframe from a database table.

    :param db_path: Path of the database
    :param table_name: name of the table
    :return: Table of the database as a pandas.DataFrame.
    '''
    db_conn = sqlite3.connect(db_path)
    try:
        query = db_conn.execute(f"SELECT * From {table_name}")
    except Exception as e:
        db_conn.close()
        raise(e)
    col_names_encoded = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data=query.fetchall(), columns=col_names_encoded)
    col_names = db_decode(col_names_encoded)
    changes = {}
    for name_encoded in col_names_encoded:
        if name_encoded[-1] == '&':
            i = col_names_encoded.index(name_encoded)
            name = col_names[i]
            changes[name_encoded] = name
    df = df.rename(columns=changes)
    # Empty cells in a column of numbers are load as numpy.nan; and in a column of strings, as None.
    # In next line I left all empty cells as None.
    df = df.replace({np.nan: None})
    db_conn.close()
    return df


def db_filter(db, **kwargs):
    '''
    Filter a pandas.dataframe by column name and delete all the empty columns

    :param db: pandas.dataframe
    :param kwargs: like (username='munoz', color= ['blue', 'red'])
    :return: Processed dataframe
    '''
    newdb = db.copy()

    for k in kwargs.keys():
        a = kwargs[k]
        if type(a) is not list:
            a = [a]
        newdb = newdb[newdb[k].isin(a)]

    for k in newdb.keys():
        if all(i is None for i in newdb[k]):
            del newdb[k]

    return newdb


def db_encode(col_names):
    '''
    Sqlite cannot store keys that only differ in case
    Return a list of names where names that only differed in case are encoded like This, this&, THIS&&, tHIs&&&...

    :param col_names_encoded: list of names
    :return: encoded list of names
    '''
    col_names_low = [col.lower() for col in col_names]
    col_names_encoded_low = list(col_names_low)
    col_names_encoded = list(col_names)
    removed = 0
    N = len(col_names)
    # It's ugly but since I'm changing names in lists this is the easier way I've found
    for n in range(N):
        name = col_names_low[n]
        del col_names_encoded_low[0]
        removed += 1
        rep = 0
        while name in col_names_encoded_low:
            rep += 1
            i = col_names_encoded_low.index(name)
            col_names_encoded_low[i] += '&' * rep
            col_names_encoded[i + removed] += '&' * rep

            log.debug(f'Name of column {i + removed} was changed from {col_names[i + removed]} to'
                      f' {col_names_encoded[i + removed]} in the database file.')

    return col_names_encoded


def db_decode(col_names_encoded):
    '''
    Remove the & from the encoded names (This, this&, THIS&&, tHIs&&&).

    :param col_names_encoded: encoded list of names
    :return: decoded list of names
    '''
    return [cn.strip('&') for cn in col_names_encoded]


def db_exist_table(db_conn, table_name):
    '''
    Check if a table exists in a databse.

    :param db_conn: Connection with the database established by db_connect()
    :param table_name: table to check if exists
    :return: bool
    '''

    c = db_conn.cursor()
    # get the count of tables with the name
    c.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    # if the count is 1, then table exists
    if c.fetchone()[0] == 1:
        return True
    else:
        return False


def db_connect(db_path):
    '''
    Establish a connection with the database.

    :param db_path: Path of the database.
    :return: Connection var.
    '''
    db_conn = sqlite3.connect(db_path)
    return db_conn


def db_commit(db_conn):
    '''
    Commit changes to a database, and close connection.
    It has an independent function to avoid using it unnecessarily, since it takes too long.

    :param db_conn: Name of the connection established previously
    :return: None
    '''
    db_conn.commit()
    db_conn.close()


load_metadb = db_load

###### Git ######

def getGitRevision():
    if gitexe is None:
        log.error('Cannot find git executable. Put git.exe on your path.')
        return 'Dunno'
    rev = subprocess.getoutput(f'cd \"{repoDir}\" & "{gitexe}" rev-parse --short HEAD')
    # return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    return rev


def getGitStatus():
    # attempt to parse the git status
    if gitexe is None:
        log.error('Cannot find git executable. Put git.exe on your path.')
        return
    status = subprocess.check_output([gitexe, 'status', '--porcelain'], cwd=repoDir).decode().strip()
    status = [l.strip().split(' ', maxsplit=1) for l in status.split('\n')]
    # I like dict of lists better
    output = {}
    if any(status[0]):
        for l in status:
            k, v = l
            if k in output:
                output[k].append(v)
            else:
                output[k] = [v]
    return output


def gitCommit(message='AUTOCOMMIT'):
    if gitexe is None:
        log.error('Cannot find git executable. Put git.exe on your path.')
        return
    # I think it will give an error if there is nothing to commit..
    output = subprocess.check_output([gitexe, 'commit', '-a', f'-m {message}'], cwd=repoDir).decode()
    return output


def log_ipy(start=True, logfilepath=None, mode='over'):
    '''
    Append ipython and std in/out to a text file
    it seems the ipython magic %logstart logs output but not stdout, so this is a naive attempt to add it

    There are some strange bugs involved
    spyder just crashes, for example
    don't be surprised if it messes something up
    '''
    #magic = get_ipython().magic
    ipy = get_ipython()
    magic = ipy.run_line_magic

    if logfilepath == ipy.logfile: pass
    magic('logstop', '')

    # Sorry, I just don't know a better way to do this.
    # I want to store the normal standard out somewhere it's safe
    # But this can only run ONCE
    try:
        sys.stdstdout
    except:
        sys.stdstdout = sys.stdout

    class Logger(object):
        ''' Something to split stdout into both the terminal and the log file'''
        def __init__(self):
            self.terminal = sys.stdstdout
            self.log = open(logfilepath, 'a')

        def write(self, message):
            self.terminal.write(message)
            # Comment the lines and append them to ipython log file
            # with open(logfilepath, 'a') as f:
            self.log.writelines(['#[Stdout]# {}\n'.format(line) for line in message.split('\n') if line != ''])

        def flush(self):
            # self.log.flush()
            # This needs to be here otherwise there's no line break in the terminal.  Don't worry about it.
            self.terminal.flush()

    # No idea what I was doing here
    #global logger
    #if logger is not None:
    #    logger.log.close()

    if start:
        # logfilepath = os.path.join(datafolder, subfolder, datestr + '_IPython.log')
        #magic('logstart -o {} append'.format(logfilepath))
        # over is not a good mode
        # append is not a good mode because we will duplicate the log if the same session starts it more than once
        magic('logstart', f'-o {logfilepath} {mode}')
        logger = Logger()
        sys.stdout = logger
    else:
        sys.stdout = sys.stdstdout


###### File/variable naming ######

def validvarname(varStr):
    # Make valid variable name from string
    sub_ = re.sub('\W|^(?=\d)', '_', varStr)
    sub_strip = sub_.strip('_')
    if sub_strip[0].isdigit():
        # Can't start with a digit
        sub_strip = 'm_' + sub_strip
    return sub_strip


def valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def hash_array(arr):
    import hashlib
    # There is also this?
    # hash(arr.tostring())
    return hashlib.md5(arr).hexdigest()


def timestamp(date=True, time=True, ms=True, us=False):
    now = datetime.now()
    datestr = now.strftime('%Y-%m-%d')
    timestr = now.strftime('%H%M%S')
    msstr = now.strftime('%f')[:-3]
    usstr = now.strftime('%f')[-3:]
    parts = []
    if date:
        parts.append(datestr)
    if time:
        parts.append(timestr)
    if ms:
        parts.append(msstr)
    if us:
        parts[-1] += usstr

    return '_'.join(parts)


def insert_file_num(filepath, number, width=3):
    return '_{{:0{}}}'.format(width).format(number).join(os.path.splitext(filepath))


###### File finding / File system utilities ######

def glob(pattern='*', directory='.', subdirs=False, exclude=None):
    pattern = pattern.join('**')
    if subdirs:
        fpaths = []
        for root, folders, files in os.walk(directory):
            fpaths.extend([os.path.join(root, f) for f in files])
    else:
        fpaths = [os.path.join(directory, f) for f in os.listdir(directory)]

    # filter by filename only, but keep absolute path
    def condition(fpath):
        filename = os.path.split(fpath)[-1]
        # Should it be excluded
        if exclude is not None:
            if isinstance(exclude, str):
                return not fnmatch.fnmatch(filename, exclude.join('**'))
            else:
                for arg in exclude:
                    if fnmatch.fnmatch(filename, arg.join('**')):
                        return False
        # Does it match
        return fnmatch.fnmatch(filename, pattern)

    filtfpaths = [fp for fp in fpaths if condition(fp)]
    abspaths = [os.path.abspath(fp) for fp in filtfpaths]
    return abspaths


def multiglob(names, *patterns):
    ''' filter list of names with potentially multiple glob patterns '''
    # TODO make it like glob(), so it searches for files
    filtered = []
    for patt in patterns:
        filter = '*' + patt + '*'
        filtered.extend(fnmatch.filter(names, filter))
    return filtered


def recentf(directory='.', n=None, seconds=None, maxlen=None, pattern=None, subdirs=False):
    '''
    Return filepaths of recently created files
    specify n to limit search to the last n files created
    '''
    now = time.time()
    if subdirs:
        filepaths = []
        for root, folders, files in os.walk(directory):
            filepaths.extend([os.path.join(root, f) for f in files])
    else:
        filepaths = [os.path.join(directory, f) for f in os.listdir(directory)]
    if pattern is not None:
        pattern = pattern.join('**')
        filepaths = fnmatch.filter(filepaths, pattern)
    ctimes = [os.path.getctime(fp) for fp in filepaths]
    # Sort by ctime
    order = np.argsort(ctimes)
    ctimes = [ctimes[i] for i in order]
    filepaths = [filepaths[i] for i in order]
    if n is not None:
        filepaths = filepaths[-n:]
        ctimes = ctimes[-n:]
    if seconds is not None:
        filepaths = [fp for fp, ct in zip(filepaths, ctimes) if now - ct < seconds]
    if maxlen is not None:
        filepaths = filepaths[:maxlen]
    return [os.path.abspath(fp) for fp in filepaths]


def set_readonly(filepath):
    from stat import S_IREAD, S_IRGRP, S_IROTH
    os.chmod(filepath, S_IREAD | S_IRGRP | S_IROTH)


def makefolder(*args):
    ''' Make a folder if it doesn't already exist. All args go to os.path.join '''
    subfolder = os.path.join(*args)
    if not os.path.isdir(subfolder):
        log.info('Making folder: {}'.format(subfolder))
        os.makedirs(subfolder)
    else:
        log.info('Folder already exists: {}'.format(subfolder))


def psplitall(path):
    # get all parts of the filepath why the heck isn't this in os.path?
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


###### IV data IO to and from different formats ######

def read_pandas(filepaths, concat=True, dropcols=None):
    '''
    Load in any number of pickled dataframes and/or series
    return concatenated dataframe
    e.g.
    read_pandas(glob('2019-02*.s'))
    read_pandas(recentf(n=3))

    # TODO I ran short on time, so dropcols is not implemented if you pass a single filepath
    '''
    if type(filepaths) is str:
        # Single series or df
        return pd.read_pickle(filepaths)
    else:
        N = len(filepaths)
        # Should be a list of filepaths
        pdlist = []
        # Try to get pandas to read the files, but don't give up if some fail
        for i,f in enumerate(filepaths):
            try:
                # pdlist may have some combination of Series and DataFrames.  Series should be rows
                pdobject = pd.read_pickle(f)
            except KeyboardInterrupt:
                print('KeyboardInterrupt. Returning whatever was loaded so far.')
                break
            except:
                log.error('Failed to interpret {} as a pickle!'.format(f))
                continue

            if type(pdobject) is pd.DataFrame:
                if dropcols is not None:
                    realdropcols = [dc for dc in dropcols if dc in pdobject]
                    pdobject = pdobject.drop(realdropcols, 1)
                if 'filepath' not in pdobject:
                    pdobject['filepath'] = [f] * len(pdobject)
                pdlist.append(pdobject)
            elif type(pdobject) is pd.Series:
                if dropcols is not None:
                    realdropcols = [dc for dc in dropcols if dc in pdobject]
                    pdobject = pdobject.drop(realdropcols)
                if 'filepath' not in pdobject:
                    pdobject['filepath'] = f
                # Took me a while to figure out how to convert series into single row dataframe
                pdlist.append(pd.DataFrame.from_records([pdobject]))
                # This resets all the datatypes to object !!
                # pdlist.append(pd.DataFrame(pdobject).transpose())
            else:
                log.warning('Do not know wtf this file is:')
            log.info(f'{i+1}/{N} Loaded {f}.')
        if concat:
            return pd.concat(pdlist).reset_index()
        else:
            return pdlist


# to not break old scripts
read_pandas_files = read_pandas

def write_pandas_pickle(data, filepath=None, drop=None):
    ''' Write a dict, list of dicts, Series, or DataFrame to pickle. '''
    if filepath is None:
        filepath = timestamp()

    filedir = os.path.split(filepath)[0]
    if (filedir != '') and not os.path.isdir(filedir):
        os.makedirs(filedir)

    # give it a standard type-dependent extension if one isn't specified
    filename, ext = os.path.splitext(filepath)
    if not ext:
        ext = pandas_pickle_extension(data)
        filepath += ext

    dtype = type(data)
    if dtype in (dict, pd.Series):
        if dtype == dict:
            #log.info('Converting data to pd.Series for storage.')
            data = pd.Series(data)
        if drop is not None:
            todrop = [c for c in drop if c in data]
            if any(todrop):
                log.info('Dropping data keys: {}'.format(todrop))
                data = data.drop(todrop)
    elif dtype in (list, pd.DataFrame):
        if dtype == list:
            #log.info('Converting data to pd.DataFrame for storage.')
            data = pd.DataFrame(data)
        if drop is not None:
            todrop = [c for c in drop if c in data]
            if any(todrop):
                log.info('Dropping data keys: {}'.format(todrop))
                data = data.drop(todrop, 1)
    data.to_pickle(filepath)
    set_readonly(filepath)
    size = os.path.getsize(filepath)
    if size > 2**30:
        size = f'{size/2**30:.2f} GB'
    elif size > 2**20:
        size = f'{size/2**20:.2f} MB'
    elif size > 2**10:
        size = f'{size/2**10:.2f} KB'
    else:
        size = f'{size:.2f} B'
    abspath = os.path.abspath(filepath)
    log.info(f'Wrote {abspath}\n{size}')
    return abspath


def pandas_pickle_extension(data):
    # defines how we name the extension of dataframes and series
    dtype = type(data)
    if dtype in (dict, pd.Series):
        return '.s'
    elif dtype in (list, pd.DataFrame):
        return '.df'
    else:
        return ''


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
    colnamemap = {'I': ['Current Probe (A)', 'Current [A]', 'Current[A]', 'I1'],
                  'V': ['Voltage Source (V)', 'Voltage [V]', 'Voltage[V]', 'V1'],
                  'T': ['Temperature  (K)', 'Temperature', 'Temperature [K]'],
                  't': ['time', 'Time [S]', 't[s]'],
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
        elif firstline.startswith('linestoskip:'):
            # GPIB control monstrosity
            skiprows = int(firstline[12:].strip()) + 1
            readcsv_args['skiprows'] = skiprows
            more_header = []
            for _ in range(skiprows - 1):
                more_header.append(f.readline())
            header = [firstline]
            header.extend(more_header)
            colname_line = more_header[-1]
            # Single string version
            header = ''.join(header)
        elif firstline.startswith('#Temperature'):
            # no header ....
            header = firstline
            readcsv_args['header'] = None
            readcsv_args['comment'] = '#'
            colname_line = 't\tV\tI\tV/I'
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
                    df.rename(columns={altname: k}, inplace=True)

    # My preferred format for a single IV loop is a dict with arrays and scalars and whatever else
    # Pandas equivalent is a pd.Series.

    longnames = {'I': 'Current', 'V': 'Voltage', 't': 'Time', 'T': 'Temperature'}
    # Note that the unit names are simply assumed here -- no attempt to read the units from the file
    units = {'I': 'A', 'V': 'V', 't': 's', 'T': 'K'}

    dataout = {k: df[k].values for k in df.columns}
    dataout['mtime'] = os.path.getmtime(filepath)
    dataout['units'] = {k: v for k, v in units.items() if k in dataout.keys()}
    dataout['longnames'] = {k: v for k, v in longnames.items() if k in dataout.keys()}
    dataout['filepath'] = os.path.abspath(filepath)
    dataout['header'] = header

    # Replace Keithley nan values with real nans
    if 'I' in dataout:
        nanmask = dataout['I'] == 9.9100000000000005e+37
        dataout['I'][nanmask] = np.nan

    return pd.Series(dataout)


def read_txts(filepaths, sort=True, **kwargs):
    # Try to sort by file number, even if fixed width numbers are not used
    # For now I will assume the filename ends in _(somenumber)
    # this function used to contain the globbing code, now you should use the glob() function
    fnames = [psplit(fp)[-1] for fp in filepaths]
    if sort:
        try:
            filepaths.sort(key=lambda fn: int(splitext(fn.split('_')[-1])[0]))
        except:
            log.warning('Failed to sort files by file number. Sorting by mtime instead.')
            filepaths.sort(key=lambda fn: os.path.getmtime(fn))

    log.info('Loading the following files:')
    log.info('\n'.join(fnames))

    datalist = []
    for fp in filepaths:
        datalist.append(read_txt(fp, **kwargs))

    return pd.DataFrame(datalist)


def write_matlab(data, filepath, varname=None, compress=True):
    # Write dict, list of dict, series, or dataframe to matlab format for the neanderthals
    # Haven't figured out what sucks less to work with in matlab
    # Each IV loop is a struct, has to be
    # For multiple IV loops, can either make a cell array of structs (plot(cell{1,1}.V, cell{1,1}.I))
    # Or just dump a whole bunch of structs into the namespace (plot(loop1.V, loop1.I))
    # There's no DataFrame equivalent in matlab as far as I know, but they might get around to adding one in 2050
    if varname is None:
        varname = validvarname(splitext(os.path.split(filepath)[-1])[0])
        log.info(varname)
    dtype = type(data)
    if dtype is list:
        savemat(filepath, {varname: data}, do_compression=compress)
    elif dtype is dict:
        # This will dump a bunch of names into namespace unless encapsulated in a list
        savemat(filepath, {varname: [data]}, do_compression=compress)
    elif dtype is pd.Series:
        # Same
        savemat(filepath, {varname: [dict(data)]}, do_compression=compress)
    elif dtype is pd.DataFrame:
        savemat(filepath, {varname: data.to_dict('records')}, do_compression=compress)


def read_matlab(filepath):
    # Read matlab file into dataframe or series
    '''
    These functions solve the problem of not properly recovering python dictionaries
    from mat files. It calls the function _check_keys to cure all entries
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
            # elif isinstance(elem, np.ndarray):
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
            # elif isinstance(sub_elem, np.ndarray):
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
        log.warning('More than one matlab variable stored in {}. Returning dict.'.format(filepath))
        return mat_in
    else:
        # List of dicts
        # return mat_in[mat_vars[0]]
        # DataFrame
        var_in = mat_in[mat_vars[0]]
        if type(var_in) is list:
            # More than one loop
            return pd.DataFrame(var_in)
        else:
            return pd.Series(var_in)


def write_csv(data, filepath, columns=None, overwrite=False):
    # For true dinosaurs
    # TODO: Don't write hundreds of thousands of files
    # TODO: Don't write a 10 GB text file
    if type(data) in (dict, pd.Series):
        if hasattr(filepath, '__call__'):
            filepath = filepath(data)
        # Write header of non-array data
        isarray = [k for k in data.keys() if type(data[k]) == np.ndarray]
        notarray = [k for k in data.keys() if k not in isarray]
        if (not overwrite) and os.path.isfile(filepath):
            raise Exception('File already exists!')
        else:
            # Replace any newlines with literal \n
            def replacenewline(line):
                if type(line) is str:
                    return line.replace('\n', '\\n')
                else:
                    return line

            header = '\n'.join(['# {}\t{}'.format(k, replacenewline(data[k])) for k in notarray])
            if columns is None:
                columns = isarray
            directory = os.path.split(filepath)[0]
            if (directory != '') and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(filepath, 'w') as f:
                f.write(header)
                f.write('\n')
            pd.DataFrame({k: data[k] for k in columns}).to_csv(filepath, sep='\t', index=False, mode='a')
            # np.savetxt(f, np.vstack([data[c] for c in columns]).T, delimiter='\t', header=header)
    elif type(data) in (list, pd.DataFrame):
        # Come up with unique filenames, and pass it back to write_csv one by one
        if type(data) == pd.DataFrame:
            iterdata = data.iterrows()
        else:
            iterdata = enumerate(data)
        for i, d in iterdata:
            if hasattr(filepath, '__call__'):
                write_csv(d, filepath=filepath, columns=columns, overwrite=overwrite)
            else:
                fn = insert_file_num(filepath, i, 3)
                write_csv(d, filepath=fn, columns=columns, overwrite=overwrite)


def write_csv_multi(data, filepath, columns=None, overwrite=False):
    '''
    NOT IMPLEMENTED
    Write a list (or df) of loops to one csv file.
    Could write a ragged list, for dinosaurs who want to copy paste into origin
    Or could stack all the arrays vertically, which would be easier to read in programmatically maybe
    '''
    if type(data) in (dict, pd.Series):
        # You called the wrong function
        write_csv(data, filepath, columns=None, overwrite=False)
    elif type(data) is list:
        pass


def write_meta_csv(data, filepath):
    ''' Write the non-array data to a text file.  Only first row of dataframe considered!'''
    dtype = type(data)
    if dtype is pd.Series:
        # s = pd.read_pickle(pjoin(root, f))
        s = data
    elif dtype is pd.DataFrame:
        # Only save first row metadata -- Usually it's the same for all
        # df = pd.read_pickle(pjoin(root, f))
        s = data.iloc[0]
        s['nloops'] = len(data)
    elif dtype is list:
        s = pd.Series(data[0])
    elif dtype is dict:
        s = pd.Series(data)
    # Drop all arrays from data
    arrays = s[s.apply(type) == np.ndarray].index
    s.drop(arrays).to_csv(filepath, sep='\t', encoding='utf-8')


###### other things

def read_exampledata():
    fp = os.path.join(repoDir, 'ivtools', 'sampledata', 'example_ivloops.df')
    return read_pandas(fp)


def change_devicemeta(filepath, newmeta, filenamekeys=None, deleteold=False):
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
    if filenamekeys is not None:
        for fnkey in filenamekeys:
            if fnkey in s.index:
                newfilename += '_{}'.format(s[fnkey])
    newpath = os.path.join(filedir, newfilename + extension)
    log.info('writing new file {}'.format(newpath))
    datain.to_pickle(newpath)
    if deleteold:
        log.info('deleting old file {}'.format(filepath))
        os.remove(filepath)


def plot_datafiles(datadir, maxloops=500, smoothpercent=0, overwrite=False, groupby=None,
                   plotfunc=None, **kwargs):
    # Make a plot of all the .s and .df files in a directory
    # Save as pngs with the same name
    # kwargs go to plotfunc
    # TODO: move to plot.py

    if plotfunc is None:
        plotfunc = ivtools.plot.plotiv

    files = glob('*.[s,df]', datadir)

    fig, ax = plt.subplots()

    def processgroup(g):
        if smoothpercent > 0:
            col = ivtools.analyze.find_data_arrays(g)[0]
            smoothn = max(int(smoothpercent * len(g.iloc[0][col]) / 100), 1)
            g = ivtools.analyze.moving_avg(g, smoothn)
        if ('R_series' in g) and ('I' in g) and ('V' in g) and ('Vd' not in g):
            g['Vd'] = g['V'] - g['R_series'] * g['I']
        # ivtools.analyze.convert_to_uA(g)
        return g

    def plotgroup(g):
        fig, ax = plt.subplots()
        step = int(np.ceil(len(g) / maxloops))
        plotfunc(g[::step], alpha=.6, ax=ax, **kwargs)
        ivtools.plot.auto_title(g)

    def writefig(pngfp):
        plt.savefig(pngfp)
        log.info('Wrote {}'.format(pngfp))

    if groupby is None:
        # Load each file individually and plot
        for fn in files:
            log.info(f'Reading {fn}')
            pngfn = os.path.splitext(fn)[0] + '.png'
            pngfp = os.path.join(datadir, pngfn)
            if overwrite or not os.path.isfile(pngfp):
                try:
                    df = pd.read_pickle(fn)
                    # if type(df) is pd.Series:
                    # df = ivtools.analyze.series_to_df(df)
                    df = processgroup(df)
                    log.info('plotting')
                    plotgroup(df)
                    writefig(pngfp)
                except Exception as e:
                    log.warning('Failed to plot')
                    log.error(e)
            elif not overwrite:
                log.info(f'not overwriting file {pngfp}')
    else:
        # Read all the data in the directory into memory at once
        df = read_pandas(files)
        for k, g in df.groupby(groupby):
            # Fukken thing errors if you are only grouping by one key
            pngfn = 'group_' + '_'.join(format(val) for pair in zip(groupby, k) for val in pair) + '.png'
            pngfp = os.path.join(datadir, pngfn)
            if overwrite or not os.path.isfile(pngfp):
                processgroup(g)
                plotgroup(g)
                # Maybe title with the thing you grouped by
                writefig(pngfp)

    plt.close(fig)


def writefig(filename, subdir='', plotdir='Plots', overwrite=True, savefig=False):
    # write the current figure to disk
    # Can also write a pickle of the figure
    plotsubdir = os.path.join(plotdir, subdir)
    if not os.path.isdir(plotsubdir):
        os.makedirs(plotsubdir)
    plotfp = os.path.join(plotsubdir, filename)
    if os.path.isfile(plotfp + '.png') and not overwrite:
        log.info('Not overwriting {}'.format(plotfp))
    else:
        plt.savefig(plotfp)
        log.info('Wrote {}.png'.format(plotfp))
        if savefig:
            with open(plotfp + '.plt', 'wb') as f:
                pickle.dump(plt.gcf(), f)
            log.info('Wrote {}.plt'.format(plotfp))


def tile_figs(pattern='*_loops.png', out_fn='grid.png', folder='.', scale=.5, aspect=16/9, crop=True, pad=0, rect=False, keeplabels=True):
    '''
    for when you have too many subplots that would bring matplotlib to its knees
    this stitches pngs together to make a giant grid

    pngs are tiled from the top to bottom, left to right, in the order of the sorted filenames

    hopefully they are all the same size and have the same axis ranges.
    if not, all hell will probably break loose
    '''
    pngfiles = [fp for fp in glob(pattern, folder)]
    n = len(pngfiles)
    # get close to the right aspect ratio
    W = int(np.sqrt(n*aspect)) # number of plots in width direction
    H = int(np.ceil(n / W))    # number of plots in height direction
    # extra = W*H - n
    if rect:
        H -= 1 # skip the last row to make a nice tidy rectangle
        n = W*H
        pngfiles = pngfiles[:n]

    pngs = [Image.open(fn) for fn in pngfiles]

    # all pngs should have the same size/axis frame as this representative
    rep = pngs[0]

    if crop:
        #crop_pngs = [png.crop([91, 15, 615, 428]) for png in pngs]
        # Try to find the axis frames automatically -- should be easy
        pixsum = np.sum(np.array(rep)[:,:,:3], -1)
        framethresh = 200 # mean of the sum of the RGB channels
        wframe = np.where(np.mean(pixsum, 0) < framethresh)[0]
        hframe = np.where(np.mean(pixsum, 1) < framethresh)[0]
        area = [wframe[0], hframe[0], wframe[1]+1, hframe[1]+1]
        if keeplabels:
            # don't cut labels if plot is on left or bottom
            crop_pngs = []
            for i,png in enumerate(pngs):
                area2 = area.copy()
                if i % W == 0: area2[0] = 0
                if i // W == H - 1:  area2[3] = rep.height
                crop_pngs.append(png.crop(area2))
        else:
            crop_pngs = [png.crop(area) for png in pngs]
    else:
        crop_pngs = pngs

    crop_resize_pngs = [im.resize([int(s*scale) for s in im.size]) for im in crop_pngs]

    grid_width = sum(im.width for im in crop_resize_pngs[:W]) + pad*(W-1)
    grid_height = sum(im.height for im in crop_resize_pngs[::W]) + pad*(H-1)
    grid = Image.new('RGB', (grid_width, grid_height), color=(255,255,255))

    x = 0
    y = 0
    for i,p in enumerate(crop_resize_pngs):
        grid.paste(p, (x, y))
        if (i+1) % W == 0:
            y += p.height + pad
            x = 0
        else:
            x += p.width + pad

    grid.save(os.path.join(folder, out_fn))


def update_depsheet():
    # Try to get the new deposition sheet
    moduledir = os.path.split(__file__)[0]
    localfile = os.path.join(moduledir, r'sampledata\CeRAM_Depositions.xlsx')
    sourcefile = r'X:\emrl\Pool\Projekte\HGST-CERAM\CeRAM_Depositions.xlsx'
    log.info(f'copy {sourcefile} {localfile}')
    return subprocess.getoutput(f'copy {sourcefile} {localfile}')





##############################################
### Functions for processing camera images ###
##############################################

# TODO: Maybe don't call this jpg if we don't now if this works with loaded jpgs?

def mat2jpg(mat, scale=1, quality=95):
    """

    Converts matrix of color values to jpg formatted byte vector.

    Parameters
    ----------
    mat : ndarray
        2D/3D matrix with byte (unit8) values.
    scale : TYPE, optional
        DESCRIPTION. The default is 1.
    quality : TYPE, optional
        DESCRIPTION. The default is 95.

    Returns
    -------
    jpg : ndarray
        Vector with byte values, formatted as jpg.

    """
    import cv2 as cv
    if scale != 1:
        height = int(mat.shape[0] * scale)
        width = int(mat.shape[1] * scale)

        mat = cv.resize(mat, (width, height))
    # TODO: Warn if scale > 1 ?

    # Quality is jpg image quality, it's between 0 and 100,
    # both values inclusive, higher is better
    # 95 is opencv default
    quality = round(quality)
    if quality not in range(0, 101):
        raise Exception("Quality must be between 0 and 100 (inclusive)!")

    # This returns a vector of bytes, which seems to be the compressed image
    # and also contains information on its format
    # So far no documentation was found that the quality flag and the value are
    # to be a list, this is by analogy to the C++ example
    succ, jpg = cv.imencode(".jpg", mat, [cv.IMWRITE_JPEG_QUALITY, quality])

    if not succ:
        raise Exception("Failed to encode image!")

    return jpg

def jpg2mat(jpg):
    """
    Converts jpg formatted data to matrix of color values.

    Parameters
    ----------
    jpg : ndarray
        Vector with byte values, formatted as jpg.

    Returns
    -------
    mat : ndarray
        2D/3D matrix with byte (unit8) values.

    """
    import cv2 as cv
    # This apparently doesn't have a success return value,
    # if not succesfull mat is empty
    mat = cv.imdecode(jpg, cv.IMREAD_UNCHANGED)

    if mat is None:
        raise Exception("Could not decode image!")

    return mat

def extractJpg(files=None, throw=False, name="cameraImage"):
    """
    Extracts the stored camera image from a measurement file and save it to disc.

    Parameters
    ----------
    files : str,[str], optional
        Path(es) to the datafiles. If None, all files in the current folder
        are processed. The default is None.
    throw : bool, optional
        Should there be an exception if no image is found in a file?
        Otherwise these are silently ignored. The default is False.
    name : str, optional
        The index of the image in the datafile. The default is "CameraImage".

    Returns
    -------
    None.

    """
    if type(files) == str:
        d = pd.read_pickle(files)
        try:
            jpg = d[name]
        except:
            if throw:
                raise Exception("Could not extract image for " +
                      d["file_timestamp"] + "!")

            return

        p = Path(files)
        with open(p.with_suffix(".jpg"), "wb") as w:
                w.write(jpg)
    else:
        if files is None:
            files = [f for f in os.listdir(".") if f.endswith(".s")]

        for f in files:
            d = pd.read_pickle(f)
            try:
                jpg = d[name]
            except:
                if throw:
                    raise Exception("Could not extract image for " +
                          d["file_timestamp"] + "!")

                continue

            p = Path(f)

            # So the thing encoded by opencv is apparently really a valid jpg!
            with open(p.with_suffix(".jpg"), "wb") as w:
                w.write(jpg)
