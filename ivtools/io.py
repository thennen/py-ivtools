""" Functions for saving and loading data """
# we don't make heavy use of these other modules
# don't reference them on the top level
# this is to avoid circular import problems
import ivtools.analyze
import ivtools.plot

import os
import re
import fnmatch
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import sys
import subprocess
import numpy as np
import time
from matplotlib import pyplot as plt

try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as spio
from scipy.io import savemat
import sqlite3

pjoin = os.path.join
splitext = os.path.splitext
psplit = os.path.split

gitdir = os.path.split(__file__)[0]

logger = None


class MetaHandler(object):
    '''
    Stores, cycles through, prints meta data (stored in dicts, or pd.Series)
    for attaching sample information to data files, with interactive use in mind.

    Can generate filenames

    df attribute holds the list of metadata as a list-of-dicts or pandas dataframe
    meta holds the currently selected row of metadata
    static holds additional metadata which will not cycle
    static values will override meta values if the keys collide

    __repr__ will print the concatenated meta and static

    you can set/get items directly on the MetaHandler instance instead of on its meta attribute

    Attach the currently selected meta data and the static meta data with the attach() function

    MetaHandler is Borg.  Its state lives in an separate module.
    This is so if io module is reloaded, Metahandler instance keeps the metadata
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
        return self.meta[key]

    def __setitem__(self, key, value):
        self.meta[key] = value

    def __delitem__(self, key):
        self.meta[key].__delitem__

    def select(self, i):
        # select the ith row of the metadataframe
        self.i = i
        if type(self.df) == pd.DataFrame:
            self.meta = self.df.iloc[self.i]
        else:
            self.meta = self.df[self.i]

    def load_sample_table(self, **filters):
        ''' load data (pd.read_excel) from some tabular format'''
        fpath = 'sampledata/CeRAM_Depositions.xlsx'
        if not os.path.isfile(fpath):
            # Maybe it's a relative path
            fpath = os.path.join(self.moduledir, fpath)
        df = pd.read_excel(fpath, header=8, skiprows=[9])
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
        self.select(0)
        self.df = df

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
        print('Loaded {} devices into metadata list'.format(len(devicemetalist)))
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
        meta_df = meta_df.dropna(1, 'all')
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
        print('Loaded metadata for {} devices'.format(len(self.df)))
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
        meta_df = meta_df.dropna(1, 'all')

        # Sort top to bottom, left to right
        meta_df['icol'] = meta_df.col.apply(columns.index)
        meta_df['irow'] = meta_df.row.apply(rows.index)
        meta_df = meta_df.sort_values(by=['icol', 'irow'], ascending=[True, False])  # .drop(columns=['icol', 'irow'])

        self.df = meta_df
        self.select(0)
        self.prettykeys = ['dep_code', 'sample_number', 'die_rel', 'row', 'col', 'Resistance', 'gap', 'radius']
        self.filenamekeys = ['dep_code', 'sample_number', 'row', 'col']
        print('Loaded metadata for {} devices'.format(len(self.df)))
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
                print('Went over edge of coupon -- wrapping around')
                return
            newcol = columns[icol]
            newrow = rows[irow]
            w = np.where((self.df.col == newcol) & (self.df.row == newrow))[0]
            if any(w):
                i = w[0]
            else:
                # TODO: don't check every single row/column in between, this can print lots of times in a row
                print('skipping a device that is not loaded into memory')
        self.select(i)

        # Highlight keys that have changed
        hlkeys = []
        for key in self.meta.keys():
            if key not in lastmeta.keys() or self.meta[key] != lastmeta[key]:
                hlkeys.append(key)
        print('You have selected this device (index {}):'.format(self.i))
        # Print some information about the device
        self.print(hlkeys=hlkeys)

    def step(self, n):
        ''' Select the another device by taking a step through meta df '''
        lastmeta = self.meta
        meta_i = self.i + n
        if meta_i < 0:
            print('You are at the beginning of metadata list')
            return
        elif meta_i >= len(self.df):
            print('You are at the end of metadata list')
            return
        else:
            self.select(meta_i)

        # Highlight keys that have changed
        hlkeys = []
        for key in self.meta.keys():
            if key not in lastmeta.keys() or self.meta[key] != lastmeta[key]:
                hlkeys.append(key)
        print('You have selected this device (index {}):'.format(self.i))
        # Print some information about the device
        self.print(hlkeys=hlkeys)

    def next(self):
        self.step(1)

    def previous(self):
        self.step(-1)

    def goto(self, **kwargs):
        ''' Assuming you loaded metadata already, this goes to the first row that matches the keys'''
        mask = np.ones(len(self.df), bool)
        for k, v in kwargs.items():
            mask &= self.df[k] == v

        w = np.where(mask)
        if any(w):
            i = w[0][0]
            self.select(i)
            print('You have selected this device (index {}):'.format(self.i))
            self.print()
        else:
            print('No matching devices found')

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
        # if len(self.meta) > 0:
        # print('Attaching the following metadata:')
        # TODO this does not consider meta.static, which in fact can overwrite the values of meta.meta
        # self.print()
        dtype = type(data)
        if dtype is dict:
            # Make shallow copy
            dataout = data.copy()
            dataout.update(self.meta)
            dataout.update(self.static)
        elif dtype is list:
            dataout = [d.copy() for d in data]
            for d in dataout:
                d.update(self.meta)
                d.update(self.static)
        elif dtype is pd.Series:
            # Series can't be updated by dicts
            dataout = data.append(pd.Series(self.meta)).append(pd.Series(self.static))
        elif dtype is pd.DataFrame:
            dupedmeta = pd.DataFrame([self.meta] * len(data), index=data.index)
            dupedstatic = pd.DataFrame([self.static] * len(data), index=data.index)
            allmeta = dupedmeta.join(dupedstatic)
            dataout = data.join(allmeta)
        else:
            print('MetaHandler does not understand what kind of data you are trying to attach to.')
            dataout = data.append(self.meta).append(self.static)
        return dataout

    def filename(self):
        ''' Create a timestamped filename from selected metadata '''
        filename = datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]
        for fnkey in self.filenamekeys:
            if fnkey in self.meta.keys():
                filename += '_{}'.format(self.meta[fnkey])
            elif fnkey in self.static.keys():
                filename += '_{}'.format(self.static[fnkey])
        return filename

    def attach_filepath(self, data, filepath):
        dict_file = {'File': filepath}
        pd_file = pd.Series(dict_file)
        print(pd_file, type(pd_file))
        dtype = type(data)
        if dtype is dict:
            # Make shallow copy
            dataout = data.copy()
            dataout.update(pd_file)
        elif dtype is list:
            dataout = [d.copy() for d in data]
            for d in dataout:
                d.update(pd_file)
        elif dtype is pd.Series:
            # Series can't be updated by dicts
            dataout = data.append(pd.Series(pd_file))
        elif dtype is pd.DataFrame:
            dupedmeta = pd.DataFrame([pd_file] * len(data), index=data.index)
            dataout = data.join(dupedmeta)
        else:
            print('MetaHandler does not understand what kind of data you are trying to attach to.')
            dataout = data.append(pd_file)
        return dataout

    def savedata(self, data=None, file_path = None,
                 database_path='C:/Users/munoz/Desktop/py-ivtools/ivtools/saves/DataBase.db',
                 table_name='Meta', drop=None):
        """
        :param data: If no data is passed, try to use the global variable d.
        :param folder_path: Folder where the file will be saved. If None, save it in the current directory.
        :param database_path: Path of the database. If it doesn't exist create a new one.
        :param table_name: Name of the table in the database. If the table doesn't exist, create a new one.
        :param drop: drop columns to save disk space.
        """
        # TODO: Think a better default folder_path and database_path
        if data is None:
            global d
            if type(d) in (dict, list, pd.Series, pd.DataFrame):
                print('No data passed to savedata(). Using global variable d.')
                data = d
        data = self.attach(data)
        file_name = self.filename()
        file_path = folder_path + '/' + file_name
        write_pandas_pickle(data, file_path, drop=drop)
        data = self.attach_filepath(data, file_path)
        if db_exist_table(database_path, table_name) is False:
            db_create_table(database_path, table_name, data)
        else:
            db_insert_row(database_path, table_name, data)



def db_create_table(db_name, table_name, data):
    """
    Creates a table from the 'pandas.series' array of data.
    It names columns and insert the first row of data.
    All column's names will be in lower case
    """

    db_file = sqlite3.connect(db_name)
    c = db_file.cursor()

    # Creating table and naming columns.
    col_names = list(data.keys())

    def blacklist_filter(col_name):
        val = data[col_name]
        val_ch = db_change_type(val)
        dtype = type(val)
        if val_ch is None:
            print(f"Data type {dtype} is not allowed, '{col_name}' won't de saved.")
            return None
        else:
            return col_name

    params = tuple(filter(None, [blacklist_filter(name) for name in col_names]))
    col_names = list(params)
    # print(f"CREATE TABLE {table_name} {params}")
    c.execute(f"CREATE TABLE {table_name} {params}")

    # Adding values to the first row
    params = [db_change_type(data[col_name]) for col_name in col_names]
    qmarks = "(?" + ", ?" * (len(params) - 1) + ")"
    # print(f"INSERT INTO {table_name} VALUES {qmarks}", params)
    c.execute(f"INSERT INTO {table_name} VALUES {qmarks}", params)
    db_file.commit()


def db_insert_row(db_name, table_name, data):
    '''
    Insert a row of data of any length, creating new columns if necessary.
    data must be pandas.Series
    '''

    db_file = sqlite3.connect(db_name)
    c = db_file.cursor()

    old_col_names = db_get_col_names(db_name, table_name)
    new_col_names = list(data.keys())

    # This loop fills the cells of the existing columns.
    def fill_cols(col_name):
        if col_name in new_col_names:
            val = data[col_name]
            dtype = type(val)
            val_ch = db_change_type(val)
            if val_ch is None:
                print(f"Data type {dtype} not suported.'{col_name}' was saved as 'None'")
                return None
            else:
                return val_ch
        else:
            return None

    params = [fill_cols(name) for name in old_col_names]

    # This loop add new columns if needed.
    for name in new_col_names:
        if name not in old_col_names:
            val = data[name]
            dtype = type(val)
            val_ch = db_change_type(val)
            if val_ch is not None:
                db_add_col(db_name, table_name, name)
                print(f"New column added: {name}")
                params.append(val_ch)
            else:
                print(f"Data type '{dtype}' not suported.'{name}' won't be saved")

    qmarks = "(?" + ", ?" * (len(params) - 1) + ")"
    params = tuple(params)

    # print(f"INSERT INTO {table_name} VALUES {qmarks}", params)
    c.execute(f"INSERT INTO {table_name} VALUES {qmarks}", params)
    db_file.commit()


def db_get_col_names(db_name, table_name):
    '''Returns a list with the name of the columns, in lower case.'''

    db_file = sqlite3.connect(db_name)
    c = db_file.cursor()

    get_names = c.execute(f"select * from {table_name} limit 1")
    col_names = [i[0] for i in get_names.description]
    return list(col_names)


def db_add_col(db_name, table_name, col_name):
    '''Adds a new column at the end of the table'''

    db_file = sqlite3.connect(db_name)
    c = db_file.cursor()

    c.execute(f"ALTER TABLE {table_name} ADD '{col_name}'")


def db_change_type(val):
    types_dict = {np.ndarray: None, list: None, dict: str, pd._libs.tslibs.timestamps.Timestamp: str,
                  np.float64: float,
                  np.float32: float, np.int64: int, np.int32: int, np.int16: int, str: str, int: int, float: float,
                  np.uint8: int, np.uint16: int, np.uint32: int}
    dtype = type(val)
    if dtype in types_dict.keys():
        if types_dict[dtype] == None:
            val = None
        elif types_dict[dtype] == str:
            val = str(val)
        elif types_dict[dtype] == datetime:
            val = val.to_pydatetime()
        elif types_dict[dtype] == float:
            val = float(val)
        elif types_dict[dtype] == int:
            val = int(val)
    else:
        print(f"Data type {dtype} is not registered, it will be save as str")
        val = repr(val)
    return val


def db_load_db(db_name, table_name):
    db_file = sqlite3.connect(db_name)
    query = db_file.execute(f"SELECT * From {table_name}")
    cols = [column[0] for column in query.description]
    return pd.DataFrame.from_records(data=query.fetchall(), columns=cols)


def db_exist_table(db_name, table_name):
    db_file = sqlite3.connect(db_name)
    c = db_file.cursor()
    # get the count of tables with the name
    c.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    # if the count is 1, then table exists
    if c.fetchone()[0] == 1:
        return True
    else:
        return False
    # close the connection
    db_file.close()


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
    return hashlib.md5(arr).hexdigest()
    # There is also this?
    # hash(arr.tostring())


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


def getGitRevision():
    rev = subprocess.getoutput('cd \"{}\" & git rev-parse --short HEAD'.format(gitdir))
    # return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    if 'not recognized' in rev:
        # git not installed probably
        return 'Dunno'
    else:
        return rev


def getGitStatus():
    # attempt to parse the git status
    status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
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
    # I think it will give an error if there is nothing to commit..
    output = subprocess.check_output(['git', 'commit', '-a', f'-m {message}']).decode()
    return output


def log_ipy(start=True, logfilepath=None):
    '''
    Append ipython and std in/out to a text file
    There are some strange bugs involved
    spyder just crashes, for example
    don't be surprised if it messes something up
    '''
    global logger
    magic = get_ipython().magic
    magic('logstop')

    # Sorry, I just don't know a better way to do this.
    # I want to store the normal standard out somewhere it's safe
    # But this can only run ONCE
    try:
        sys.stdstdout
    except:
        sys.stdstdout = sys.stdout

    class Logger(object):
        ''' Something to replace stdout '''

        def __init__(self):
            self.terminal = sys.stdstdout
            self.log = open(logfilepath, 'a')

        def write(self, message):
            self.terminal.write(message)
            # Comment the lines and append them to ipython log file
            self.log.writelines(['#[Stdout]# {}\n'.format(line) for line in message.split('\n') if line != ''])

        def flush(self):
            self.log.flush()
            # This needs to be here otherwise there's no line break in the terminal.  Don't worry about it.
            self.terminal.flush()

    if logger is not None:
        logger.log.close()

    if start:
        # logfilepath = os.path.join(datafolder, subfolder, datestr + '_IPython.log')
        magic('logstart -o {} append'.format(logfilepath))
        logger = Logger()
        sys.stdout = logger
    else:
        sys.stdout = sys.stdstdout


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

    dataout = {k: df[k].as_matrix() for k in df.columns}
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
            print('Failed to sort files by file number. Sorting by mtime instead.')
            filepaths.sort(key=lambda fn: os.path.getmtime(fn))

    print('Loading the following files:')
    print('\n'.join(fnames))

    datalist = []
    for fp in filepaths:
        datalist.append(read_txt(fp, **kwargs))

    return pd.DataFrame(datalist)


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
            if fnmatch.fnmatch(filename, exclude.join('**')):
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
        # Should be a list of filepaths
        pdlist = []
        # Try to get pandas to read the files, but don't give up if some fail
        for f in filepaths:
            try:
                # pdlist may have some combination of Series and DataFrames.  Series should be rows
                pdobject = pd.read_pickle(f)
            except:
                print('Failed to interpret {} as a pickle!'.format(f))
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
                print('Do not know wtf this file is:')
            print('Loaded {}.'.format(f))
        if concat:
            return pd.concat(pdlist).reset_index()
        else:
            return pdlist


# to not break old scripts
read_pandas_files = read_pandas


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


def write_pandas_pickle(data, filepath=None, drop=None):
    ''' Write a dict, list of dicts, Series, or DataFrame to pickle. '''
    if filepath is None:
        filepath = timestamp()

    filedir = os.path.split(filepath)[0]
    if (filedir != '') and not os.path.isdir(filedir):
        os.makedirs(filedir)

    dtype = type(data)
    if dtype in (dict, pd.Series):
        filepath += '.s'
        if dtype == dict:
            print('Converting data to pd.Series for storage.')
            data = pd.Series(data)
        if drop is not None:
            todrop = [c for c in drop if c in data]
            if any(todrop):
                print('Dropping data keys: {}'.format(todrop))
                data = data.drop(todrop)
    elif dtype in (list, pd.DataFrame):
        filepath += '.df'
        if dtype == list:
            print('Converting data to pd.DataFrame for storage.')
            data = pd.DataFrame(data)
        if drop is not None:
            todrop = [c for c in drop if c in data]
            if any(todrop):
                print('Dropping data keys: {}'.format(todrop))
                data = data.drop(todrop, 1)
    data.to_pickle(filepath)
    set_readonly(filepath)
    print('Wrote {}'.format(os.path.abspath(filepath)))


def write_matlab(data, filepath, varname=None, compress=True):
    # Write dict, list of dict, series, or dataframe to matlab format for the neanderthals
    # Haven't figured out what sucks less to work with in matlab
    # Each IV loop is a struct, has to be
    # For multiple IV loops, can either make a cell array of structs (plot(cell{1,1}.V, cell{1,1}.I))
    # Or just dump a whole bunch of structs into the namespace (plot(loop1.V, loop1.I))
    # There's no DataFrame equivalent in matlab as far as I know, but they might get around to adding one in 2050
    if varname is None:
        varname = validvarname(splitext(os.path.split(filepath)[-1])[0])
        print(varname)
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
        print('More than one matlab variable stored in {}. Returning dict.'.format(filepath))
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
        # TODO: Don't write thousands of files
        # TODO: Don't write a 10 GB text file


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


def insert_file_num(filepath, number, width=3):
    return '_{{:0{}}}'.format(width).format(number).join(os.path.splitext(filepath))


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


def write_sql(data, db):
    ''' Append some data to sql table.  Didn't actually write the function yet.'''
    con = 'wtf'
    data.to_sql(db, con, if_exists='append')
    # TODO: figure out what to do if data has columns that aren't already in the database.


def set_readonly(filepath):
    from stat import S_IREAD, S_IRGRP, S_IROTH
    os.chmod(filepath, S_IREAD | S_IRGRP | S_IROTH)


def plot_datafiles(datadir, maxloops=500, smoothpercent=0, overwrite=False, groupby=None,
                   plotfunc=ivtools.plot.plotiv, **kwargs):
    # Make a plot of all the .s and .df files in a directory
    # Save as pngs with the same name
    # kwargs go to plotfunc
    # TODO: move to plot.py
    files = glob('*.[s,df]', datadir)

    fig, ax = plt.subplots()

    def processgroup(g):
        if smoothpercent > 0:
            smoothn = max(int(smoothpercent * len(g.iloc[0].V) / 100), 1)
            g = ivtools.analyze.moving_avg(g, smoothn)
        if 'R_series' in g:
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
        print('Wrote {}'.format(pngfp))

    if groupby is None:
        # Load each file individually and plot
        for fn in files:
            print(f'Reading {fn}')
            pngfn = os.path.splitext(fn)[0] + '.png'
            pngfp = os.path.join(datadir, pngfn)
            if overwrite or not os.path.isfile(pngfp):
                df = pd.read_pickle(fn)
                # if type(df) is pd.Series:
                # df = ivtools.analyze.series_to_df(df)
                df = processgroup(df)
                print('plotting')
                plotgroup(df)
                writefig(pngfp)
            elif not overwrite:
                print(f'not overwriting file {pngfp}')
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


def makefolder(*args):
    ''' Make a folder if it doesn't already exist. All args go to os.path.join'''
    subfolder = os.path.join(*args)
    if not os.path.isdir(subfolder):
        print('Making folder: {}'.format(subfolder))
        os.makedirs(subfolder)
    else:
        print('Folder already exists: {}'.format(subfolder))


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


def update_depsheet():
    # Try to get the new deposition sheet
    moduledir = os.path.split(__file__)[0]
    localfile = os.path.join(moduledir, r'sampledata\CeRAM_Depositions.xlsx')
    sourcefile = r'X:\emrl\Pool\Projekte\HGST-CERAM\CeRAM_Depositions.xlsx'
    print(f'copy {sourcefile} {localfile}')
    return subprocess.getoutput(f'copy {sourcefile} {localfile}')
