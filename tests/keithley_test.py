from ivtools.instruments import Keithley2600
import numpy as np
import matplotlib.pyplot as plt
import logging

log = logging.getLogger('instruments')
"""
This test is ment to have a red led in channel A and a blue one in channel B
"""
k = Keithley2600()

v_red = np.linspace(1.5, 2, 100)
i_red = np.linspace(0.0001, 0.015, 100)
v_blue = np.linspace(2.5, 3, 100)


def iv():
    plt.figure()

    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.plot(data['Vmeasured'], data['I'], label='iv')

    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.plot(data['V'], data['Imeasured'], label="source = I")

    k.iv(source_list=v_blue, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='b')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='b')
    plt.plot(data['Vmeasured'], data['I'], label='channel b')

    plt.legend()
    plt.show()


def iv_limits():
    v_limit = 1.85
    i_limit = 0.004
    p_limit = v_limit*i_limit

    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=i_limit, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.hlines(i_limit, xmin=np.min(data['V']), xmax=np.max(data['V']))
    plt.plot(data['V'], data['I'], label='I sourced, I limited')
    plt.legend()
    plt.show()

    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=v_limit, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.vlines(v_limit, ymin=np.min(data['I']), ymax=np.max(data['I']))
    plt.plot(data['V'], data['I'], label='I sourced, V limited')
    plt.legend()
    plt.show()


    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=v_limit, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.vlines(v_limit, ymin=np.min(data['I']), ymax=np.max(data['I']))
    plt.plot(data['V'], data['I'], label='V sourced, V limited')
    plt.legend()
    plt.show()


    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=i_limit, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.hlines(i_limit, xmin=np.min(data['V']), xmax=np.max(data['V']))
    plt.plot(data['V'], data['I'], label='V sourced, I limited')
    plt.legend()
    plt.show()

    k.iv(source_list=v_red, source_func='v', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=p_limit,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.plot(data['V'], data['I'], label='V sourced, P limited')
    plt.legend()
    plt.show()

    k.iv(source_list=i_red, source_func='i', source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=p_limit,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.plot(data['V'], data['I'], label='I sourced, P limited')
    plt.legend()
    plt.show()


def vi():
    k.vi(source_list=i_red, source_range=None, measure_range=None,
         v_limit=None, i_limit=None, p_limit=None,
         nplc=1, delay=None, point4=False, ch='a')
    k.waitready()
    data = k.get_data(start=1, end=None, history=True, ch='a')
    plt.figure()
    plt.plot(data['V'], data['Imeasured'], label='vi')

    plt.legend()
    plt.show()


def iv_2ch():
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
    plt.plot(data['V_B'], data['I_B'], label='2ch A')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # iv()
    iv_limits()
    # vi()
    # iv_2ch()
