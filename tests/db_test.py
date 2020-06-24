import math
import os
import statistics

import time
from ivtools import io
import numpy as np
import pandas as pd
from line_profiler import LineProfiler

meta = io.MetaHandler()


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
    '''
    This has been made especially for matadata.pkl
    It takes around 15 mins
    Some columns are repeated in metadata.pkl so merge_repeated() should be run first.
    '''

    data = pd.read_pickle('C:/Users/munoz/Desktop/database/data_examples/metadata2.pkl')
    row0 = data.iloc[0]
    io.db_create_table("metadata.db", "metadata", row0)
    prev = 0
    start = time.time()
    times = []
    for N in range(len(data)):
        row = data.iloc[N]
        io.db_insert_row("metadata.db", "metadata", row)
        if int(N/1000) > prev:
            elapsed = time.time() - start
            times.append(elapsed)
            mean = statistics.mean(times)
            mins = int(mean / 1000 * (len(data) - N) / 60)
            secs = int((mean / 1000 * (len(data) - N) / 60 - mins) * 60)
            prev += 1
            print(f"Row {N} loaded. Expected remaining time: {mins} minutes and {secs} seconds")
            start = time.time()

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

def small_test1():
    # TODO: check what happens with 'pandas.core.series.Series' data type
    data1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=['Hola', 'Adios', 'hola'])
    data2 = pd.DataFrame(np.array([[7, 8, 9, 10]]), columns=['Hola', 'Adios', 'hola', 'adios'])
    data3 = pd.DataFrame(np.array([[11, 12, 13]]), columns=['adios', 'hola', 'HOLA'])
    print(data1)
    print(data2)
    print(data3)

    if os.path.isfile("test.db") is True:
        os.remove("test.db")

    row = data1.iloc[0]
    io.db_create_table("test.db", "test", row)
    row = data1.iloc[1]
    io.db_insert_row("test.db", "test", row)
    row = data2.iloc[0]
    io.db_insert_row("test.db", "test", row)
    row = data3.iloc[0]
    io.db_insert_row("test.db", "test", row)

    df = io.db_load("test.db", "test")
    print(df)

def small_test2():
    data = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=['Hola', 'Adios', 'hola'])
    print(data)
    print(type(data))
    print('-'*20)
    a = data.iloc[0]
    print(a)
    print(type(a))
    print('-' * 20)
    a = a['Hola']
    print(a)
    print(type(a))

def time_test():
    data = pd.read_pickle('C:/Users/munoz/Desktop/database/data_examples/metadata2.pkl')
    row0 = data.iloc[0]
    row = data.iloc[3056]
    io.db_insert_row("metadata.db", "metadata", row)

#### Functions to run ####

# start = time.time()
# create_big_db()
# elapsed = time.time() - start
# mins = int(elapsed/60)
# secs = int((elapsed/60 - mins)*60)
# print(f'Elapsed time: {mins} minutes and {secs} seconds')

lp = LineProfiler()
lp.add_function(io.db_insert_row)
lp_wrapper = lp(time_test)
lp_wrapper()
lp.print_stats()

