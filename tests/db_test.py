import math
import os
import statistics

import time
from ivtools import io
import interactive
import numpy as np
import pandas as pd
from line_profiler import LineProfiler

meta = io.MetaHandler()


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

    data = pd.read_pickle('C:/Users/munoz/Desktop/database/data_examples/metadata2.pkl')
    row0 = data.iloc[0]
    if os.path.isfile("metadata.db") is True:
        os.remove("metadata.db")
    db_conn = io.db_connect("metadata.db")
    io.db_create_table(db_conn, "metadata", row0)
    prev = 0
    start = time.time()
    times = []
    N = 1000
    for n in range(N):
        row = data.iloc[n]
        io.db_insert_row(db_conn, "metadata", row)
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
    ser = pd.Series([11, 12, 13, 14], index=['Hola', 'Ey', 'hola', 'Adios'])
    dct = {'Hola': 7, 'hola': 9, 'Adios': 8, 'List': [1, 2, 3]}
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=['Hola', 'Adios', 'hola'])
    ld = [{'Hola': 21, 'hola': 22, 'Adios': 23}, {'Hola': 24, 'hola': 25, 'Adios': 26}]

    data = ser
    print(data)

    # if os.path.isfile("test.db") is True:
    #     os.remove("test.db")

    interactive.savedata(data)

    # df = io.db_load("test.db", "test")
    # print(df)
    # print(type(df))


def test2(data):
    data = pd.read_pickle('C:/Users/munoz/Desktop/database/data_examples/metadata2.pkl')



if __name__ == '__main__':

    test1()



