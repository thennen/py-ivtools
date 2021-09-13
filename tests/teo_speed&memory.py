import logging
import os
import time
from datetime import datetime
from multiprocessing import Process, Queue

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from ivtools import instruments

"""
Uploads waveforms with increasing lenght until the system complains.
Tracks memory usage of the python and TSX processes.
Measures times to upload a waveform to TSX, output that waveform, output the same waveform again (so it is
already in the device memory) and data collection.
Saves the results at 'X:\emrl\Pool\Bulletin\Handbücher.Docs\TS_Memory_Tester/tests/memory_test'.
Parameters
"""






def track_memory(q, python_pid, TSX_DM_pip, sample_time=0.01):
    """
    Tracks the memory usage

    """
    memory_time = np.array([])
    memory_total = np.array([])
    memory_python = np.array([])
    memory_tsx = np.array([])
    python_process = psutil.Process(python_pid)
    TSX_DM_process = psutil.Process(TSX_DM_pip)
    while True:
        t = time.time()
        vm = psutil.virtual_memory().used
        pvm = python_process.memory_info().rss
        tvm = TSX_DM_process.memory_info().rss
        memory_time = np.append(memory_time, t)
        memory_total = np.append(memory_total, vm)
        memory_python = np.append(memory_python, pvm)
        memory_tsx = np.append(memory_tsx, tvm)
        time.sleep(sample_time)
        if not q.empty():
            q.get()
            break
    q.put(memory_time)
    q.put(memory_total)
    q.put(memory_python)
    q.put(memory_tsx)


def test():
    teo = instruments.TeoSystem()
    log = logging.getLogger('instruments')

    # Creates the folder to save the data and plots
    folder_path = 'X:\emrl\Pool\Bulletin\Handbücher.Docs\TS_Memory_Tester/tests/memory_test/test_1/'
    t = 1
    while os.path.exists(folder_path):
        t += 1
        folder_path = f'X:\emrl\Pool\Bulletin\Handbücher.Docs\TS_Memory_Tester/tests/memory_test/test_{t}/'
    os.mkdir(folder_path)

    # Here you can specify the pips of the precesses to track in case the script is not finding the properly.
    python_pip = None
    TSX_DM_pip = None

    file = open(os.path.join(folder_path, 'info.txt'), "w")


    def log_and_write(msg):
        file.write(msg + '\n')
        log.info(msg)


    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]
    log_and_write(f"Teo speed and memory test\n{timestamp}")


    # Finding the Processes to track
    if python_pip is None or TSX_DM_pip is None:

        python_pids = []
        tsx_pids = []
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == 'python.exe':
                python_pids.append(proc.info['pid'])
            elif proc.info['name'] == 'TSX_DM.exe':
                tsx_pids.append(proc.info['pid'])
        if len(python_pids) > 1 and python_pip is None:
            python_vms = []
            # Creates a big array so and choose the one with more memory usage
            a = np.ones(100_000_000)
            for pip in python_pids:
                try:
                    python_vms.append(psutil.Process(pip).memory_info().vms)
                except:
                    python_vms.append(0)
            del a
            python_vms = np.array(python_vms)
            python_pip = python_pids[python_vms.argmax()]

        elif len(python_pids) == 1 and python_pip is None:
            python_pip = python_pids[0]
        else:
            raise Exception("No Python processes found")

        if len(tsx_pids) == 1 and TSX_DM_pip is None:
            TSX_DM_pip = tsx_pids[0]
        else:
            raise Exception("No TSX_DM processes found")

    log_and_write(f"Python pip: {python_pip}")
    log_and_write(f"TSX_DM pip: {TSX_DM_pip}")
    t_start = time.time()

    # Using multiprocessing to track the memory vs time
    q = Queue()
    p = Process(target=track_memory, args=(q, python_pip, TSX_DM_pip, 0.01))

    p.start()
    time.sleep(1.7)


    teo_freq = 500e6
    teo.delete_all_wfms()
    n = 5
    upload_time_0 = np.array([])
    upload_time_1 = np.array([])
    output1_time_0 = np.array([])
    output1_time_1 = np.array([])
    output2_time_0 = np.array([])
    output2_time_1 = np.array([])
    data_time_0 = np.array([])
    data_time_1 = np.array([])
    wfm_nsamples = []
    log_and_write("-" * 50)
    while True:
        teo.delete_all_wfms()
        wfm = [1] * 2 ** n
        wfm_nsamples.append(2 ** n)
        name = f'smt{n}'
        samples = 2 ** n
        duration = samples / teo_freq
        try:
            log_and_write(f"Waveform with 2^{n} = {samples} samples and {duration}s")

            t0 = time.time()
            teo.upload_wfm(wfm, name)
            t1 = time.time()
            upload_time_0 = np.append(upload_time_0, t0 - t_start)
            upload_time_1 = np.append(upload_time_1, t1 - t_start)
            log_and_write(f"Upload completed in {t1-t0}s")
        except Exception as E:
            t1 = time.time()
            log_and_write(f"Teo raised the following error:\n\t{E}")
            teo.delete_all_wfms()
            upload_time_0 = np.append(upload_time_0, t0 - t_start)
            upload_time_1 = np.append(upload_time_1, t1 - t_start)
            output1_time_0 = np.append(output1_time_0, np.nan)
            output1_time_1 = np.append(output1_time_1, np.nan)
            output2_time_0 = np.append(output2_time_0, np.nan)
            output2_time_1 = np.append(output2_time_1, np.nan)
            data_time_0 = np.append(data_time_0, np.nan)
            data_time_1 = np.append(data_time_1, np.nan)
            break

        try:
            t0 = time.time()
            teo.output_wfm(name)
            t1 = time.time()
            output1_time_0 = np.append(output1_time_0, t0 - t_start)
            output1_time_1 = np.append(output1_time_1, t1 - t_start)
            log_and_write(f"First output completed in {t1-t0}s")
        except Exception as E:
            t1 = time.time()
            log_and_write(f"Teo raised the following error:\n\t{E}")
            teo.delete_all_wfms()
            output1_time_0 = np.append(output1_time_0, t0 - t_start)
            output1_time_1 = np.append(output1_time_1, t1 - t_start)
            output2_time_0 = np.append(output2_time_0, np.nan)
            output2_time_1 = np.append(output2_time_1, np.nan)
            data_time_0 = np.append(data_time_0, np.nan)
            data_time_1 = np.append(data_time_1, np.nan)
            break

        try:
            t0 = time.time()
            teo.output_wfm(name)
            t1 = time.time()
            output2_time_0 = np.append(output2_time_0, t0 - t_start)
            output2_time_1 = np.append(output2_time_1, t1 - t_start)
            log_and_write(f"Second output completed in {t1-t0}s")
        except Exception as E:
            t1 = time.time()
            log_and_write(f"Teo raised the following error:\n\t{E}")
            teo.delete_all_wfms()
            output2_time_0 = np.append(output2_time_0, t0 - t_start)
            output2_time_1 = np.append(output2_time_1, t1 - t_start)
            data_time_0 = np.append(data_time_0, np.nan)
            data_time_1 = np.append(data_time_1, np.nan)
            break

        try:
            t0 = time.time()
            teo.get_data()
            t1 = time.time()
            data_time_0 = np.append(data_time_0, t0 - t_start)
            data_time_1 = np.append(data_time_1, t1 - t_start)
            log_and_write(f"Data collection completed in {t1-t0}s")
            teo.delete_all_wfms()
        except Exception as E:
            t1 = time.time()
            log_and_write(f"Teo raised the following error:\n\t{E}")
            teo.delete_all_wfms()
            data_time_0 = np.append(data_time_0, t0 - t_start)
            data_time_1 = np.append(data_time_1, t1 - t_start)
            break

        log_and_write("-"*50)
        n += 1

    q.put('stop')
    time.sleep(2)

    memory_time = q.get()
    memory_total = q.get()
    memory_python = q.get()
    memory_tsx = q.get()
    p.join()
    memory_time = memory_time - t_start  # seconds
    memory_total = memory_total * 10 ** -9  # GB
    memory_python = memory_python * 10 ** -9  # GB
    memory_tsx = memory_tsx * 10 ** -9  # GB

    memory_results = pd.DataFrame(dict(memory_time=memory_time, memory_total=memory_total,
                                       memory_python=memory_python, memory_tsx=memory_tsx))

    time_results = pd.DataFrame(dict(upload_time_0=upload_time_0, upload_time_1=upload_time_1,
                                     output1_time_0=output1_time_0, output1_time_1=output1_time_1,
                                     output2_time_0=output2_time_0, output2_time_1=output2_time_1,
                                     data_time_0=data_time_0, data_time_1=data_time_1,
                                     wfm_nsamples=wfm_nsamples))

    pd.to_pickle(memory_results, os.path.join(folder_path, 'memory_results.df'))
    pd.to_pickle(time_results, os.path.join(folder_path, 'time_results.df'))

    fig = plt.figure('Teo memory Test')
    fig.clf()
    ax = fig.add_subplot()
    cmap = mpl.cm.get_cmap('Set2')
    for i, v in time_results.iterrows():
        if i == 0:
            ax.axvspan(v['upload_time_0'], v['upload_time_1'],
                       alpha=0.9, color=cmap(0), label="Upload")
            ax.axvspan(v['output1_time_0'], v['output1_time_1'],
                       alpha=0.9, color=cmap(1), label="First Output")
            ax.axvspan(v['output2_time_0'], v['output2_time_1'],
                       alpha=0.9, color=cmap(2), label="Second Output")
            ax.axvspan(v['data_time_0'], v['data_time_1'],
                       alpha=0.9, color=cmap(3), label="Data collection")
        else:
            ax.axvspan(v['upload_time_0'], v['upload_time_1'],
                       alpha=0.9, color=cmap(0))
            ax.axvspan(v['output1_time_0'], v['output1_time_1'],
                       alpha=0.9, color=cmap(1))
            ax.axvspan(v['output2_time_0'], v['output2_time_1'],
                       alpha=0.9, color=cmap(2))
            ax.axvspan(v['data_time_0'], v['data_time_1'],
                       alpha=0.9, color=cmap(3))
        text = f"2^{int(np.log2(v['wfm_nsamples']))} samples"
        ax.text(v['upload_time_0'], 0, text, horizontalalignment='right',
                verticalalignment='bottom', rotation=90)

    cmap = mpl.cm.get_cmap('tab10')
    ax.plot(memory_results['memory_time'], memory_results['memory_total'],
            alpha=0.9, linewidth=2, color=cmap(0), label="Total memory used")
    ax.plot(memory_results['memory_time'], memory_results['memory_python'],
            alpha=0.9, linewidth=2, color=cmap(1), label="Python memory used")
    ax.plot(memory_results['memory_time'], memory_results['memory_tsx'],
            alpha=0.9, linewidth=2, color=cmap(2), label="TSX_DM memory used")

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Memory used [GB]')
    ax.set_xscale('log', base=2)
    t_max = memory_results['memory_time'].iloc[-1]
    ticks = [2 ** i for i in range(int(np.log2(t_max)))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(folder_path, 'results.png'))
    fig.show()

    log_and_write(f"Data saved at {folder_path}")
    file.close()


if __name__ == '__main__':
    __spec__ = None  # Fixes this on iPython: AttributeError: module '__main__' has no attribute '__spec__'
    test()

