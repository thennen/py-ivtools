

from ivtools.instruments import Keithley2600
import numpy as np
import matplotlib.pyplot as plt
import logging

log = logging.getLogger('instruments')
"""
This test is meant to have a red led in channel A and a blue one in channel B
"""
k = Keithley2600()

v_red = np.linspace(1.5, 3, 100)
i_red = np.linspace(0.0001, 0.03, 100)
v_blue = np.linspace(2.5, 3, 100)


def iv():
    plt.figure()

    log.info("Testing IV sourcing V")
    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.plot(data['Vmeasured'], data['I'], label='sourcing V')

    log.info("Testing IV sourcing I")
    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.plot(data['V'], data['Imeasured'], label="sourcing I")

    log.info("Testing IV sourcing V in channel B")
    k.iv(source_list=v_blue, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='b')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='b')
    plt.plot(data['Vmeasured'], data['I'], label='channel b')

    plt.legend()
    plt.title("You should see 3 diode lines:\n'sourcing V' and 'sourcing I' must be almost identical.\nAnd 'channel b' "
              "should be similar but displaced horizontally")
    plt.show()


def iv_limits():
    log.info("Testing limiting in IV")
    v_limit = 1.85
    i_limit = 0.005
    p_limit = 0.04

    log.info("\tSourcing I, limiting I")
    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=i_limit, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.hlines(i_limit, xmin=np.min(data['V']), xmax=np.max(data['V']))
    plt.plot(data['V'], data['Imeasured'], label=f'I sourced\nI limited to {i_limit}A')
    plt.legend()
    plt.title(f"You should see one line lower than y={i_limit}A")
    plt.show()

    log.info("\tSourcing V, limiting V")
    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=v_limit, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.vlines(v_limit, ymin=np.min(data['I']), ymax=np.max(data['I']))
    plt.plot(data['Vmeasured'], data['I'], label=f'V sourced\nV limited to {v_limit}V')
    plt.legend()
    plt.title(f"You should see one line lower than x={v_limit}V")
    plt.show()

    log.info("\tSourcing I, limiting V")
    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=v_limit, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.vlines(v_limit, ymin=np.min(data['I']), ymax=np.max(data['I']))
    plt.plot(data['V'], data['Imeasured'], label=f'I sourced\nV limited to {v_limit}V')
    plt.legend()
    plt.title(f"You should see one line lower than x={v_limit}V")
    plt.show()

    log.info("\tSourcing V, limiting I")
    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=i_limit, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.hlines(i_limit, xmin=np.min(data['V']), xmax=np.max(data['V']))
    plt.plot(data['Vmeasured'], data['I'], label=f'V sourced\nI limited to {i_limit}A')
    plt.legend()
    plt.title(f"You should see one line lower than y={i_limit}A")
    plt.show()


    log.info("\tSourcing I, limiting P")
    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=p_limit,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    Imax = p_limit / data['V']
    plt.figure()
    plt.plot(data['V'], data['Imeasured'], label=f'I sourced\nP limited to {p_limit}W')
    plt.plot(data['V'], Imax, label='Imax')
    plt.legend()
    plt.title(f"Diode line should be under 'Imax'")
    plt.show()

    log.info("\tSourcing V, limiting P")
    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=p_limit,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    Imax = p_limit / data['Vmeasured']
    plt.figure()
    plt.plot(data['Vmeasured'], data['I'], label=f'V sourced\nP limited to {p_limit}W')
    plt.plot(data['Vmeasured'], Imax, label='Imax')
    plt.legend()
    plt.title(f"Diode line should be under 'Imax'")
    plt.show()

def low_limits():
    log.info("Testing limiting in IV")
    v_limit = 1.65
    i_limit = 0.002
    p_limit = 0.01

    log.info("\tSourcing I, limiting I")
    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=i_limit, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.hlines(i_limit, xmin=np.min(data['V']), xmax=np.max(data['V']))
    plt.plot(data['V'], data['Imeasured'], label=f'I sourced\nI limited to {i_limit}A')
    plt.legend()
    plt.title(f"You should see one line lower than y={i_limit}A")
    plt.show()

    log.info("\tSourcing V, limiting V")
    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=v_limit, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.vlines(v_limit, ymin=np.min(data['I']), ymax=np.max(data['I']))
    plt.plot(data['Vmeasured'], data['I'], label=f'V sourced\nV limited to {v_limit}V')
    plt.legend()
    plt.title(f"You should see one line lower than x={v_limit}V")
    plt.show()

    log.info("\tSourcing I, limiting V")
    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=v_limit, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.vlines(v_limit, ymin=np.min(data['I']), ymax=np.max(data['I']))
    plt.plot(data['V'], data['Imeasured'], label=f'I sourced\nV limited to {v_limit}V')
    plt.legend()
    plt.title(f"You should see one line lower than x={v_limit}V")
    plt.show()

    log.info("\tSourcing V, limiting I")
    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=i_limit, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.hlines(i_limit, xmin=np.min(data['V']), xmax=np.max(data['V']))
    plt.plot(data['Vmeasured'], data['I'], label=f'V sourced\nI limited to {i_limit}A')
    plt.legend()
    plt.title(f"You should see one line lower than y={i_limit}A")
    plt.show()


    log.info("\tSourcing I, limiting P")
    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=p_limit,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    Imax = p_limit / data['V']
    plt.figure()
    plt.plot(data['V'], data['I'], label=f'I sourced\nP limited to {p_limit}W')
    plt.plot(data['V'], Imax, label='Imax')
    plt.legend()
    plt.title(f"Diode line should be under 'Imax'")
    plt.show()

    log.info("\tSourcing V, limiting P")
    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=0.00001,
         v_limit=None, i_limit=None, p_limit=p_limit,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    Imax = p_limit / data['Vmeasured']
    plt.figure()
    plt.plot(data['Vmeasured'], data['I'], label=f'V sourced\nP limited to {p_limit}W')
    plt.plot(data['Vmeasured'], Imax, label='Imax')
    plt.legend()
    plt.title(f"Diode line should be under 'Imax'")
    plt.show()

def vi():
    log.info("Testing VI")
    k.vi(source_list=i_red, source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.plot(data['V'], data['Imeasured'], label='vi')

    plt.legend()
    plt.title("You should see one diode line")
    plt.show()


def iv_2ch():
    log.info("Testing IV 2 channels")
    k.iv_2ch(a_source_list=v_red, b_source_list=v_blue,
             a_source_func='v', b_source_func='v',
             a_source_range=None, b_source_range=None,
             a_measure_range=None, b_measure_range=None,
             a_v_limit=None, b_v_limit=None,
             a_i_limit=None, b_i_limit=None,
             a_p_limit=None, b_p_limit=None,
             a_nplc=1, b_nplc=1,
             a_delay=None, b_delay=None,
             a_point4=False, b_point4=False)
    k.waitready()
    data = k.get_data_2ch(start=1, end=None, history=True)
    plt.figure()
    plt.plot(data['V_A'], data['I_A'], label='2ch A')
    plt.plot(data['V_B'], data['I_B'], label='2ch B')
    plt.legend()
    plt.title("You should see two diode lines")
    plt.show()

    log.info(f"Red and Blue leds should alternate in pulses of 3 seconds approx, if you see short pulses, or both leds "
             f"brighting at the same time, something is going wrong.")
    v_red2 = [2, 0.01, 2, 0.01, 2, 0.01]
    v_blue2 = [0.01, 3, 0.01, 3, 0.01, 3]
    k.iv_2ch(a_source_list=v_red2, b_source_list=v_blue2,
             a_source_func='v', b_source_func='v',
             a_source_range=None, b_source_range=None,
             a_measure_range=None, b_measure_range=None,
             a_v_limit=None, b_v_limit=None,
             a_i_limit=None, b_i_limit=None,
             a_p_limit=None, b_p_limit=None,
             a_nplc=1, b_nplc=1,
             a_delay=None, b_delay=None,
             a_point4=False, b_point4=False,
             sync=True)
    k.waitready()
    log.info("Done")

    log.info(f"Now, sync is off, and you should see a mess of lights")
    k.iv_2ch(a_source_list=v_red2, b_source_list=v_blue2,
             a_source_func='v', b_source_func='v',
             a_source_range=None, b_source_range=None,
             a_measure_range=None, b_measure_range=None,
             a_v_limit=None, b_v_limit=None,
             a_i_limit=None, b_i_limit=None,
             a_p_limit=None, b_p_limit=None,
             a_nplc=1, b_nplc=1,
             a_delay=None, b_delay=None,
             a_point4=False, b_point4=False,
             sync=False)
    k.waitready()
    log.info("Done")


if __name__ == '__main__':
    iv()
    iv_limits()
    # low_limits()
    vi()
    iv_2ch()

    log.info("Test completed!")
