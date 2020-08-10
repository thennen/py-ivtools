import math
import os
import statistics

import time
from ivtools import io
import interactive
import numpy as np
import pandas as pd
# from line_profiler import LineProfiler

meta = io.MetaHandler()
folder = "C:/Users/munoz/Desktop/database/dbtests"
db = "C:/Users/munoz/Desktop/database/dbtests/dbtest.db"
bigdb = "D:/metadata.db"

def to_time_test(row):
    col_names = list(row.keys())
    print(col_names)
    col_name = col_names[100]
    val = row[col_name]
    print(val)


def time_test(data):
    lp = LineProfiler()
    lp_wrapper = lp(to_time_test)
    lp_wrapper(data)
    lp.print_stats()


def fake_measurement():
    measurement = {'V': np.random.rand(10), 'I': np.random.rand(10) / 10, 'Instrument': 'fake',
                   'Some rand': 100 * np.random.rand()}

    meta.load_nanoxbar()
    meta.step(3)
    meta.savedata(measurement)

    '''
    #Code to run in console
    cd C:/Users/munoz/Desktop/py-ivtools
    run -i interactive
    d = {'V': np.random.rand(10), 'I': np.random.rand(10) / 10, 'Instrument': 'fake', 'Some rand': 100*np.random.rand()}
    meta.load_nanoxbar()
    meta.step(3)
    savedata(d)
    '''


def create_big_db():
    """
    This has been made especially for matadata.pkl, but can be change to anything easily.
    Since it could take long, it plots expected remaining time every 1000 rows added.
    """

    data = pd.read_pickle('C:/Users/munoz/Desktop/database/data_examples/2019-04-26_151601_693_Lama_3_1_001G_2.s')
    row0 = data.iloc[0]
    print(len(data))
    print(type(data))
    print(data)
    db_path = "C:/Users/munoz/Desktop/database/dbtests.db"
    db_conn = io.db_connect(db_path)
    io.db_create_table(db_conn, "none_test", row0)
    prev = 0
    start = time.time()
    times = []
    N = 1000
    for n in range(N):
        row = data.iloc[n]
        io.db_insert_row(db_conn, "none_test", row)
        if int(n/1000) > prev:
            elapsed = time.time() - start
            times.append(elapsed)
            mean = statistics.mean(times)
            mins = int(mean / 1000 * (len(data) - n) / 60)
            secs = int((mean / 1000 * (len(data) - n) / 60 - mins) * 60)
            prev += 1
            print(f"Row {n} loaded. Expected remaining time: {mins} minutes and {secs} seconds")
            start = time.time()
    io.db_commit(db_conn)


def merge_repeated():
    """
    This has been made especially for matadata.pkl, but it could work in general with small changes.
    """
    data = pd.read_pickle('C:/Users/munoz/Desktop/database/data_examples/metadata.pkl')
    columns = list(data.columns)
    columns_pop = [column.lower() for column in columns]
    repeated = []
    for i in range(len(columns)):
        col = columns_pop.pop(0)
        if col in columns_pop:
            i2 = columns_pop.index(col) + i + 1
            tup = (i, f'{columns[i]}', i2, f'{columns[i2]}')
            repeated.append(tup)

    print(repeated)
    # tuple 0, OFFSET and offset have 3655 overlaps, so by the moment both columns will be kept
    # tuple 4, t and T are not actually repeated columns, t is for thickness and T for Temperature

    for tup in [1, 2, 3, 5, 6, 7, 8]:
        overlaps = 0
        moves = 0
        for i in range(len(data)):
            val1 = data.at[i, repeated[tup][1]]
            if type(val1) is float:
                b1 = math.isnan(val1)
            else:
                b1 = False
            val2 = data.at[i, repeated[tup][3]]
            if type(val2) is float:
                b2 = math.isnan(val2)
            else:
                b2 = False

            if b1 is False and b2 is False:
                # print(f'Overlap in row {i}: {val1} | {val2}')
                overlaps += 1
            elif b1 is False and b2 is True:
                data.at[i, repeated[tup][3]] = data.at[i, repeated[tup][1]]
                moves += 1

        data = data.drop(columns=[repeated[tup][1]])
        print(f'{overlaps} total overlaps with {repeated[tup][1]} and {repeated[tup][3]}')
        print(f'{moves} items moved from {repeated[tup][1]} to {repeated[tup][3]}')

    data.to_pickle('C:/Users/munoz/Desktop/database/data_examples/metadata2.pkl')

    return data


def test1():
    meta = io.MetaHandler()
    table = 'test'
    db_conn = io.db_connect(db)
    data = {'hola': 1, 'adios': 2, 'Hola': None, 'Hello': 3}
    if io.db_exist_table(db_conn, table):
        c = db_conn.cursor()
        c.execute(f"DROP TABLE {table}")
    meta.savedata(data, folder, db, table)

    table = 'test2'
    db_conn = io.db_connect(db)
    if io.db_exist_table(db_conn, table):
        c = db_conn.cursor()
        c.execute(f"DROP TABLE {table}")
    io.db_create_table(db_conn, table, data)
    io.db_commit(db_conn)
<<<<<<< HEAD

def test2():
    data = io.db_load(db, 'none_test')
    data = data.replace({'None': None})
    data = data.replace({np.nan: None})
    data = data.iloc[:,0:15]

    db_conn = io.db_connect(db)
    table = 'test'
    if io.db_exist_table(db_conn, table):
        c = db_conn.cursor()
        c.execute(f"DROP TABLE {table}")
=======

def test2():
    data = io.db_load(db, 'none_test')
    data = data.iloc[:, 0:20]
    print(data)
    print('+' * 100)
    print(f"{data.iloc[8]['X']} -> {type(data.iloc[8]['X'])}")
    print('+' * 100)

    data = data.replace({'None': None})
    data = data.replace({np.nan: None})
    print(data)
    print('+' * 100)
    print(f"{data.iloc[8]['X']} -> {type(data.iloc[8]['X'])}")
    print('+' * 100)

    db_conn = io.db_connect(db)
    if io.db_exist_table(db_conn, 'none_nones'):
        c = db_conn.cursor()
        c.execute("DROP TABLE none_nones")

    row0 = data.iloc[0]
    io.db_create_table(db_conn, "none_nones", row0)
    N = len(data)
    for n in range(N):
        row = data.iloc[n]
        io.db_insert_row(db_conn, "none_nones", row)
    io.db_commit(db_conn)
>>>>>>> f0ff6d633b730154bd6a718593721e9510a852bc

    row0 = data.iloc[0]
    io.db_create_table(db_conn, table, row0)
    N = len(data)
    for n in range(N-1):
        row = data.iloc[n+1]
        io.db_insert_row(db_conn, table, row)
    io.db_commit(db_conn)

<<<<<<< HEAD
=======
    data = io.db_load(db, 'none_nones')
    print(data)
    print('+' * 100)
    # print(f"{data.iloc[8]['X']} -> {type(data.iloc[8]['X'])}")
    # print('+' * 100)

>>>>>>> f0ff6d633b730154bd6a718593721e9510a852bc
def change_None():
    data = io.db_load('D:\metadata.db', 'meta')
    data = data.replace({'None': None})
    data = data.replace({np.nan: None})
    db_conn = io.db_connect('D:\metadata2.db')
    table = 'meta'
    row0 = data.iloc[0]
    io.db_create_table(db_conn, table, row0)
    prev = 0
    start = time.time()
    times = []
    N = len(data)
    for n in range(N):
        row = data.iloc[n]
        io.db_insert_row(db_conn, table, row)
        if int(n / 1000) > prev:
            elapsed = time.time() - start
            times.append(elapsed)
            mean = statistics.mean(times)
            mins = int(mean / 1000 * (len(data) - n) / 60)
            secs = int((mean / 1000 * (len(data) - n) / 60 - mins) * 60)
            prev += 1
            print(f"Row {n} loaded. Expected remaining time: {mins} minutes and {secs} seconds")
            start = time.time()
    io.db_commit(db_conn)



if __name__ == '__main__':

<<<<<<< HEAD
    test2()
=======
    change_None()
>>>>>>> f0ff6d633b730154bd6a718593721e9510a852bc


