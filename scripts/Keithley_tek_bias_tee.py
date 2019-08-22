from matplotlib.widgets import Button
import tkinter as tk
import sys
from pathlib import Path
from collections import defaultdict
from scipy.optimize import curve_fit

def where(*args):
    return np.where(*args)[0]

def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def setup_pcm_plots():
    def plot0(data, ax=None, **kwargs):
        ax.cla()
        ax.set_title('Answer')
        if data['t_scope']:
            ax.plot(data['t_scope'][-1], data['v_answer'][-1], **kwargs)    
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        ax.cla()
        ax.semilogy(data['t'], data['V'] / data['I'], **kwargs)
        if data['t_event']:
            ax.vlines(data['t_event'],ax.get_ylim()[0]*1.2,ax.get_ylim()[1]*0.8, alpha = 0.5)
        ax.set_ylabel('Resistance [V/A]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot2(data, ax=None, **kwargs):
        ax.cla()
        ax.set_title('Pulse')
        if data['t_scope']:
            ax.plot(data['t_scope'][-1], data['v_pulse'][-1], **kwargs)
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot3(data, ax=None, **kwargs):
        ax.cla()
        ax.plot(data['t'], data['I'], **kwargs)
        if data['t_event']:
            ax.vlines(data['t_event'],ax.get_ylim()[0]*1.2,ax.get_ylim()[1]*0.8, alpha = 0.5)
        ax.set_ylabel('Current [A]')
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2],
                       [3, plot3]]
                 
    iplots.newline()

def setup_vcm_plots():

    def plot0(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_hrs).any():
            i+=1
            line = data.iloc[-2]
        ax.set_title('Read HRS #' + str(len(data)-i))
        if not np.isnan(line.t_hrs).any():
            ax.cla()
            ax.set_title('Read HRS #' + str(len(data)-i))
            ax.plot(line.t_hrs,  line.V_hrs /  line.I_hrs - 50)
            ax.set_ylabel('Resistance HRS [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_ttx).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Answer #' + str(len(data)-i))
        if not np.isnan(line.t_ttx).any():
            ax.cla()
            ax.set_title('Answer #' + str(len(data)-i))
            ax.plot(line.t_ttx, line.V_ttx, **kwargs)    
            ax.set_ylabel('Voltage [V]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot2(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_lrs).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Read LRS #' + str(len(data)-i))
        if not np.isnan(line.t_lrs).any():
            ax.cla()
            ax.set_title('Read LRS #' + str(len(data)-i))
            ax.plot(line.t_lrs,  line.V_lrs /  line.I_lrs - 50)
            ax.set_ylabel('Resistance LRS [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot3(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.V_sweep).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Sweep #' + str(len(data)-i))
        if not np.isnan(line.V_sweep).any():
            ax.cla()
            ax.set_title('Sweep #' + str(len(data)-i))
            ax.semilogy(line.V_sweep,  line.V_sweep /  line.I_sweep - 50)
            ax.set_ylabel('Resistance Sweep [V/A]')
            ax.set_xlabel('Voltage [V]')
            ax.set_ylim(bottom = 1e2)
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())   
        

        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2],
                       [3, plot3]]
                 
    iplots.newline()  

def setup_pcm_plots_2():

    def plot0(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_pre).any():
            i+=1
            line = data.iloc[-2]
        ax.set_title('Read PRE #' + str(len(data)-i))
        if not np.isnan(line.t_pre).any():
            ax.cla()
            ax.set_title('Read PRE #' + str(len(data)-i))
            ax.plot(line.t_pre,  line.V_pre /  line.I_pre)
            ax.set_ylabel('Resistance PRE [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
            ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_ttx).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Answer #' + str(len(data)-i))
        if not np.isnan(line.t_ttx).any():
            ax.cla()
            ax.set_title('Answer #' + str(len(data)-i))
            ax.plot(line.t_ttx, line.V_ttx, **kwargs)    
            ax.set_ylabel('Voltage [V]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    def plot2(data, ax=None, **kwargs):
        line = data.iloc[-1]
        i=0
        if np.isnan(line.t_post).any():
            line = data.iloc[-2]
            i+=1
        ax.set_title('Read POST #' + str(len(data)-i))
        if not np.isnan(line.t_post).any():
            ax.cla()
            ax.set_title('Read POST #' + str(len(data)-i))
            ax.plot(line.t_post,  line.V_post /  line.I_post)
            ax.set_ylabel('Resistance Post [V/A]')
            ax.set_xlabel('Time [s]')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
            ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2]]
                 
    iplots.newline()  
    
def set_keithley_plotters():
    iplots.plotters = keithley_plotters
    iplots.ax0.cla()
    iplots.ax1.cla()
    iplots.ax2.cla()
    iplots.ax3.cla()

def pcm_measurement(samplename,
padname,
amplitude = np.nan,
bits = np.nan,
sourceVA = -0.2,
points = 250, 
interval = 0.1,
trigger = 0.1,
two_channel = False,
rangeI = 0,
nplc = 1,
pulse_width = 50e-12,
attenuation =np.nan,
polarity = 1,
answer_position = -2.5,
pulse_postition = -4,
answer_scale = 0.12,
pulse_scale = 0.12,
continious = False):
    '''run a measurement during which the Keithley2600 applies a constants voltage and measures the current. 
    Pulses applied during this measurement are also recorded. '''
    setup_pcm_plots()

    number_of_events =0
    data = {}
    data['t_scope'] = []
    data['v_pulse'] = []
    data['v_answer'] = []
    data['t_event'] = []
    data['amplitude'] = amplitude
    data['bits'] = bits
    data['padname'] = padname
    data['samplename'] = samplename
    data['attenuation'] = attenuation
    iplots.show()    

    k.set_channel_state(channel = 'A', state = True)
    k.set_channel_voltage(channel = 'A', voltage = sourceVA)

    plt.pause(1)

    k.it(sourceVA = sourceVA, sourceVB = 0, points = points, interval = interval, rangeI = rangeI, limitI = 1, nplc = nplc, reset_keithley = False)

    ttx.inputstate(1, False)
    ttx.inputstate(2, True)
    ttx.inputstate(3, False)
    if two_channel:
        ttx.inputstate(4, True)
        ttx.scale(4, pulse_scale)
        ttx.position(4, -pulse_position*polarity)
    else:
        ttx.inputstate(4, False)
    ttx.scale(2, answer_scale)
    ttx.position(2, -answer_position*polarity)
    ttx.change_samplerate_and_recordlength(100e9, 5000)
    trigger = trigger*polarity
    if two_channel:
        ttx.arm(source = 4, level = trigger, edge = 'e')
    else:
        ttx.arm(source = 2, level = trigger, edge = 'e')
    
    if pg5:
        pg5.set_pulse_width(pulse_width)
        plt.pause(1)
        pg5.trigger()
    
    while not k.done():
        
        if pg5 and continious:
            pg5.trigger()

        data.update(k.get_data())
        status = ttx.triggerstate()
        while status == True:
            plt.pause(0.1)
            status = ttx.triggerstate()
        plt.pause(0.5)
        else:

            number_of_events +=1
            if two_channel:
                data_scope1 = ttx.get_curve(4)

            data_scope2 = ttx.get_curve(2)
            
            time_array = data['t']
            data['t_scope'].append(data_scope2['t_ttx'])
            if two_channel:
                data['v_pulse'].append(data_scope1['V_ttx'])
            data['v_answer'].append(data_scope2['V_ttx'])
            '''Moritz: last current data point measured after last trigger event so the entry one before
             will be used as time reference (-2 instead of -1, which is the last entry)'''
            data['t_event'].append(time_array[len(time_array)-2])
            print(time_array[len(time_array)-2])
            if two_channel:
                ttx.arm(source = 4, level = trigger, edge = 'e')
            else:
                ttx.arm(source = 2, level = trigger, edge = 'e')

        iplots.updateline(data)

    data.update(k.get_data())
    iplots.updateline(data)
    k.set_channel_state('A', False)
    k.set_channel_state('B', False)
    ttx.disarm()
    
    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, 'pcm_measurement_'+str(i))
    while os.path.isfile(filepath + '.s'):
        i +=1
        filepath = os.path.join(datafolder, subfolder, 'pcm_measurement_'+str(i))
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data

def vcm_pg5_measurement(samplename,
padname,
v1,
v2,
step = 0.01,
step2 = 0.01,
V_read = 0.2,
range_lrs = 1e-3,
range_hrs = 1e-4,
range_sweep = 1e-2,
range_sweep2 = 1e-3,
cycles = 1,
pulse_width = 50e-12,
attenuation = 0,
automatic_measurement = True,
sweep = True,
two_sweeps = False,
scale = 0.12,
position = -3,
trigger_level = 0.05,
nplc = 10,
limitI = 3e-4,
limitI2 = 3e-4,
r_window = False,
r_lower = 1e3,
r_upper = 2e3,
cc_step = 25e-6):

    setup_vcm_plots()
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    hrs_list = []
    lrs_list = []
    sweep_list = []
    scope_list = []
    vlist = tri(v1 = v1, v2 = v2, step = step)

    abort = False
    for i in range(cycles):
        if not abort:
            ### Reading HRS resistance ############################################################################
            k.set_channel_state(channel = 'A', state = True)
            k.set_channel_voltage(channel = 'A', voltage = V_read)

            plt.pause(1)

            k.it(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.1, rangeI = range_hrs , limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.set_channel_state('A', False)
            k.set_channel_state('B', False)
            hrs_data = k.get_data()
            hrs_list.append(add_suffix_to_dict(hrs_data,'_hrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)
            ### Setting up scope  ################################################################################

            ttx.inputstate(1, False)
            ttx.inputstate(2, False)
            ttx.inputstate(3, True)
            ttx.inputstate(4, False)

            ttx.scale(3, scale)
            ttx.position(3, position)


            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength=250)
            if pulse_width < 100e-12:
                ttx.trigger_position(40)
            elif pulse_width < 150e-12:
                ttx.trigger_position(30)
            else:
                ttx.trigger_position(20)

            plt.pause(0.1)

            ttx.arm(source = 3, level = trigger_level, edge = 'r')


            ### Applying pulse and reading scope data #############################################################
            pg5.set_pulse_width(pulse_width)
            if not automatic_measurement:
                input('Connect the RF probes and press enter')
                plt.pause(0.5)
            else:
                plt.pause(0.1)
                
            pg5.trigger()
            plt.pause(0.1)
            plt.pause(0.2)
            status = ttx.triggerstate()
            while status == True:
                plt.pause(0.1)
                status = ttx.triggerstate()
            plt.pause(0.5)
            scope_list.append(ttx.get_curve(3))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### Reading LRS resistance #############################################################################

            if not automatic_measurement:
                input('Connect the DC probes and press enter')

            k.set_channel_state(channel = 'A', state = True)
            k.set_channel_voltage(channel = 'A', voltage = V_read)

            plt.pause(1)
            k.it(sourceVA = V_read, sourceVB = 0, points = 10, interval = 0.1, rangeI = range_lrs, limitI = 1, nplc = nplc)
            while not k.done():
                plt.pause(0.1)
            k.set_channel_state('A', False)
            k.set_channel_state('B', False)
            lrs_data = k.get_data()
            lrs_list.append(add_suffix_to_dict(lrs_data,'_lrs'))
            data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
            iplots.updateline(data)

            ### performing sweep ###################################################################################
            if sweep:
                if two_sweeps:
                    dates_dict = defaultdict(list)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)
                    k.iv(vlist1, Irange = range_sweep, Ilimit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    sweep_data = k.get_data()
                    k.iv(vlist2, Irange = range_sweep2, Ilimit = limitI2) 
                    while not k.done():
                        plt.pause(0.1)
                    data_2nd_sweep = k.get_data()
                    for key in data_2nd_sweep:
                        data_to_append = data_2nd_sweep[key]
                        if not isinstance(data_to_append,dict) and not isinstance(data_to_append, str):
                            sweep_data[key] = np.append(sweep_data[key], data_to_append)
                else:  
                    k.iv(vlist, Irange = range_sweep) 
                    while not k.done():
                        plt.pause(0.1)
                    k.set_channel_state('A', False)
                    k.set_channel_state('B', False)
                    sweep_data = k.get_data()
                sweep_list.append(add_suffix_to_dict(sweep_data,'_sweep'))
                data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
                iplots.updateline(data)
            if r_window:
                current_compliance = limitI2
                window_hit = False
                u=0
                d=0
                while not window_hit:
                    
                    k.set_channel_state(channel = 'A', state = True)
                    k.set_channel_voltage(channel = 'A', voltage = V_read)

                    plt.pause(1)
                    k.it(sourceVA = V_read, sourceVB = 0, points = 5, interval = 0.1, rangeI = range_lrs, limitI = 1, nplc = nplc)
                    while not k.done():
                        plt.pause(0.1)
                    k.set_channel_state('A', False)
                    k.set_channel_state('B', False)
                    r_data = k.get_data()
                    resistance = np.mean(r_data['V']/r_data['I']) - 50
                    print('Compliance = ' + str(current_compliance))
                    print('Resistance = ' + str(resistance))

                    if resistance >= r_lower and resistance <= r_upper:
                        window_hit = True
                        break
                    elif resistance < r_lower:
                        current_compliance -= cc_step
                        u = 0
                        d += 1
                    elif resistance > 3.5e4 or u >=50:
                        vlist2 = tri(v1 = 0, v2 = -2, step = step2)
                        current_compliance = 2e-3
                    elif d >= 50:
                        vlist1 = tri(v1 = 2, v2 = 0, step = step)
                    else:
                        current_compliance += cc_step
                        u += 1
                        d = 0

                    if current_compliance < cc_step:
                        current_compliance =cc_step

                    if u > 51 or d > 51:
                        print('Failed hitting resistance window, aborting measurement')
                        window_hit = True
                        abort = True
                        break

                    k.iv(vlist1, Irange = range_sweep, Ilimit = limitI) 
                    while not k.done():
                        plt.pause(0.1)
                    
                    k.iv(vlist2, Irange = range_sweep2, Ilimit = current_compliance) 
                    while not k.done():
                        plt.pause(0.1)
                    vlist1 = tri(v1 = v1, v2 = 0, step = step)
                    vlist2 = tri(v1 = 0, v2 = v2, step = step2)

                    if current_compliance > 1e-3:
                        current_compliance = limitI2
            k.set_channel_state('A', False)
            k.set_channel_state('B', False)
  
    data['attenuation'] = attenuation
    data['pulse_width'] = pulse_width
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level

    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(pulse_width*1e12)) + 'ps_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data, abort


def pcm_resistance_measurement(samplename,
padname,
bits,
amplitude,
V_read = 0.2,
start_range = 1e-3,
cycles = 1,
scale = 0.12,
position = 3,
trigger_level = -0.1,
recordlength=2000,
points = 10
):

    setup_pcm_plots_2()
    data = {}
    data['padname'] = padname
    data['samplename'] = samplename

    pre_list = []
    post_list = []
    scope_list = []


    abort = False
    for i in range(cycles):
        if not abort:
            ### Reading Pre resistance ############################################################################
            _, pre_data = k.read_resistance(start_range = start_range, voltage = V_read, points = points)
            pre_list.append(add_suffix_to_dict(pre_data,'_pre'))
            data = combine_lists_to_data_frame(pre_list, post_list, scope_list)
            iplots.updateline(data)
            ### Setting up scope  ################################################################################

            ttx.inputstate(1, False)
            ttx.inputstate(2, True)
            ttx.inputstate(3, False)
            ttx.inputstate(4, False)

            ttx.scale(2, scale)
            ttx.position(2, position)

            ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength=recordlength)
            ttx.trigger_position(20)

            plt.pause(0.1)
            input('Connect RF probes')
            ttx.arm(source = 2, level = trigger_level, edge = 'e')

            ### Applying pulse and reading scope data #############################################################

            print('Apply pulse')
            plt.pause(0.2)
            status = ttx.triggerstate()
            while status == True:
                plt.pause(0.1)
                status = ttx.triggerstate()
            plt.pause(0.5)
            while ttx.busy():
                plt.pause(0.1)
            scope_data = ttx.get_curve(2)
            scope_list.append(scope_data)
            data = combine_lists_to_data_frame(pre_list, post_list, scope_list)
            iplots.updateline(data)

            ### Reading Post resistance ########################
            input('Connect DC probes')
            _, post_data = k.read_resistance(start_range = start_range, voltage = V_read, points = points)
            post_list.append(add_suffix_to_dict(post_data,'_post'))
            data = combine_lists_to_data_frame(pre_list, post_list, scope_list)
            iplots.updateline(data)
  
    data['amplitude'] = amplitude
    data['bits'] = bits
    data['scale'] = scale
    data['position'] = position
    data['trigger_level'] = trigger_level

    datafolder = os.path.join('C:\Messdaten', samplename, padname)
    subfolder = datestr
    file_exits = True
    i=1
    amplitude_decimal = (amplitude % 1)*10
    filepath = os.path.join(datafolder, subfolder, str(int(amplitude)) + 'p' + str(int(amplitude_decimal)) + '_amplitude_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, str(int(amplitude)) + 'p' + str(int(amplitude_decimal)) + '_amplitude_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data
def eval_pcm_measurement(data, manual_evaluation = False):
    '''evaluates saved data (location or variable) from an  measurements. In case of a two channel measurement it determines pulse amplitude and width'''
    setup_pcm_plots()
    ########## declareation of buttons ###########
    def agree(self):
        waitVar1.set(True)

    def threhsold_visible(self):
        pulse_minimum =min(v_answer)
        pulse_index = where(np.array(v_answer) < 0.5* pulse_minimum)
        pulse_start_index = pulse_index[0]
        pulse_start = t_scope[pulse_start_index]
        print(pulse_start)
        ax_dialog.set_title('Please indicate threshold')
        ax_dialog.plot(np.array([pulse_start,pulse_start]),np.array([-1,0.3]))
        ax_agree = plt.axes([0.59, 0.05, 0.1, 0.075])
        b_agree = Button(ax_agree,'Agree')
        b_agree.on_clicked(agree)
        cid = figure_handle.canvas.mpl_connect('pick_event', onpick)
        root.wait_variable(waitVar1)
        if not threshold_written_class.state:
            print(threshold_class.threshold)
            data['t_threshold'].append(threshold_class.threshold-pulse_start)
            threshold_written_class.state = True

        waitVar.set(True)


    def threshold_invisible(self):
        #print(threshold_written_class.state)
        if not threshold_written_class.state:
            data['t_threshold'].append(numpy.nan)
            threshold_written_class.state = True
        #print(threshold_written_class.state)
        waitVar.set(True)

    def onpick(event):
        ind = event.ind
        t_threshold = np.take(x_data, ind)
        print('onpick3 scatter:', ind, t_threshold, np.take(y_data, ind))
        threshold_class.set_threshold(t_threshold)
        if len(ind) == 1:
            ax_dialog.plot(np.array([t_threshold,t_threshold]),np.array([-1,0.3]))
            
            plt.pause(0.1)

    ######## beginning of main evalution #############
    if(type(data) == str):
        data = pd.read_pickle(data)
        iplots.show()    
    iplots.updateline(data)
    data['pulse_width'] = []
    data['pulse_amplitude'] = []
    data['t_threshold'] = []
    ########## if two channel experiment: ################
    if data['v_pulse']:       
        for t_scope, v_pulse in zip(data['t_scope'], data['v_pulse']):
            pulse_minimum =min(v_pulse)
            pulse_index = where(np.array(v_pulse) < 0.5* pulse_minimum)
            #pulse_end = t_scope[pulse_index[-1]]
            #pulse_start = t_scope[pulse_index[0]]
            v_max = max(v_pulse)
            v_min = min(v_pulse)
            if v_max > -v_min:
                pulse_width = calc_fwhm(valuelist = v_pulse, time = t_scope)
            else:
                pulse_width = calc_fwhm(valuelist = -v_pulse, time = t_scope)
            data['pulse_width'].append(pulse_width)
            data['pulse_amplitude'].append(np.mean(v_pulse[pulse_index])*2)
        
    ########## if one channel experiment: ################       
    else:
        for t_scope, v_answer in zip(data['t_scope'],data['v_answer']):
            #pulse_minimum =min(v_answer)
            #pulse_index = where(np.array(v_answer) < 0.5* pulse_minimum)
            #pulse_start_index = pulse_index[0]
            #pulse_start = t_scope[pulse_start_index]
            #print(pulse_start)
            #pulse_end_index = pulse_start_index + where(np.array(v_answer[pulse_start_index:-1]) >= 0)[0]
            #pulse_end = t_scope[pulse_end_index]
            
            # '''for short pulses the width is determined as FWHM, otherwise from pulse start until 
            #  the zero line is crossed for the first time '''
            # if pulse_end - pulse_start < 1e-9:
            #     pulse_end = t_scope[pulse_index[-1]]
            #     pulse_width = pulse_end - pulse_start
            # else:

            v_max = max(v_answer)
            v_min = min(v_answer)
            if v_max > -v_min:
                pulse_width = calc_fwhm(valuelist = v_answer, time = t_scope)
            else:
                pulse_width = calc_fwhm(valuelist = -v_answer, time = t_scope)

            data['pulse_width'].append(pulse_width)
            data['pulse_amplitude'].append(get_pulse_amplitude_of_PSPL125000(amplitude = data['amplitude'], bits = data['bits']))
            #import pdb; pdb.set_trace()
    ######## detection of threshold event by hand ###########
    if manual_evaluation:
        threshold_class = tmp_threshold()
        threshold_written_class = threshold_written()
        
        root = tk.Tk()
        root.withdraw()
        waitVar = tk.BooleanVar()
        waitVar1 = tk.BooleanVar()
        for t_scope, v_answer in zip(data['t_scope'], data['v_answer']):
            threshold_written_class.state = False
            x_data = t_scope
            y_data = v_answer/max(abs(v_answer))
            figure_handle, ax_dialog = plt.subplots()
            plt.title('Is a threshold visible?')
            plt.subplots_adjust(bottom=0.25)
            ax_dialog.plot(x_data,y_data, picker = True)
            ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
            ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])
            b_yes = Button(ax_yes, 'Yes')
            b_yes.on_clicked(threhsold_visible)
            b_no = Button(ax_no, 'No')
            b_no.on_clicked(threshold_invisible)
            root.wait_variable(waitVar)
            plt.close(figure_handle)
            #print(len(data['pulse_amplitude'])-len(data['t_threshold']))         
        plot_pcm_threshold(data)   
        root.destroy()
    return data


def eval_pcm_r_measurement(data, manual_evaluation = False, t_cap = np.nan, v_cap = np.nan):
    '''evaluates saved data (location or variable) from an  measurements. In case of a two channel measurement it determines pulse amplitude and width'''
    setup_pcm_plots_2()
    ########## declareation of buttons ###########
    # def agree(self):
    #     waitVar1.set(True)


    def threhsold_visible(self):
        user_aproove = False
        pulse_minimum =min(v_answer)
        pulse_index = where(np.array(v_answer)[1:-1] < 0.2* pulse_minimum) #ignoring first value which is ofter just wrong
        pulse_start_index = pulse_index[0]
        pulse_start = t_scope[pulse_start_index]
        #print(pulse_start)
        ax_dialog.set_title('Please indicate threshold')
        #plt.autoscale('False')
        ax_dialog.autoscale(False)
        ax_dialog.plot(np.array([pulse_start,pulse_start]),np.array([-1,0.3]))
        # ax_agree = plt.axes([0.59, 0.05, 0.1, 0.075])
        # b_agree = Button(ax_agree,'Agree')
        # b_agree.on_clicked(agree)
        above_threshold_level = where(np.array(v_diff/50 < -200e-6))
        try:
            threshold_event = where(above_threshold_level>pulse_start_index+6)[0]
        except:
            print('No threshold')
        while not user_aproove:
            threshold_index = above_threshold_level[threshold_event]-1
            t_threshold = t_scope[threshold_index]-pulse_start
            print(t_threshold)
            ax_dialog.plot(np.array([t_scope[threshold_index],t_scope[threshold_index]]),np.array([-1,0.3]))
            plt.pause(0.1)
            user_answer = input('Do you approve? ')            
            if user_answer == 'n':
                user_aproove = False
                threshold_event = where(above_threshold_level>above_threshold_level[threshold_event])[0]
            elif user_answer == 'd':
                t_threshold = np.nan
                user_aproove = True
            elif user_answer == 'y':
                user_aproove = True

        data['t_threshold'][x].append(t_threshold)
        #cid = figure_handle.canvas.mpl_connect('pick_event', onpick)
        # if not threshold_written_class.state:
        #     print(threshold_class.threshold)
        #     data['t_threshold'][x].append(threshold_class.threshold-pulse_start)
        #     threshold_written_class.state = True

        waitVar.set(True)


    def threshold_invisible(self):
        #print(threshold_written_class.state)
        if not threshold_written_class.state:
            data['t_threshold'][x].append(numpy.nan)
            threshold_written_class.state = True
        #print(threshold_written_class.state)
        waitVar.set(True)

    def onpick(event):
        ind = event.ind
        t_threshold = np.take(x_data, ind)
        #print('onpick3 scatter:', ind, t_threshold, np.take(y_data, ind))
        threshold_class.set_threshold(t_threshold)
        if len(ind) == 1:
            ax_dialog.plot(np.array([t_threshold,t_threshold]),np.array([-1,0.3]))
            
            plt.pause(0.1)

    ######## beginning of main evalution #############
    if(type(data) == str):
        data = pd.read_pickle(data)
        iplots.show()    
    iplots.updateline(data)
    data['pulse_width'] = [list() for x in range(len(data.index))]
    data['pulse_amplitude'] = [list() for x in range(len(data.index))]
    data['t_threshold'] = [list() for x in range(len(data.index))]
    ########## if two channel experiment: ################
    for x in range(len(data.index)):
        if 'v_pulse' in data.keys():       
            for t_scope, v_pulse in zip(data['t_scope'], data['v_pulse']):
               # pulse_minimum =min(v_pulse)
                #pulse_index = where(np.array(v_pulse) < 0.15* pulse_minimum)
                #pulse_end = t_scope[pulse_index[-1]]
                #pulse_start = t_scope[pulse_index[0]]
                v_max = max(v_pulse)
                v_min = min(v_pulse)
                if v_max > -v_min:
                    pulse_width = calc_fwhm(valuelist = v_pulse, time = t_scope)
                else:
                    pulse_width = calc_fwhm(valuelist = -v_pulse, time = t_scope)
                data['pulse_width'][x].append(pulse_width)
                data['pulse_amplitude'][x].append(np.mean(v_pulse[pulse_index])*2)
            
        ########## if one channel experiment: ################       
        else:
            for t_scope, v_answer in zip(data['t_ttx'],data['V_ttx']):

                v_max = max(v_answer)
                v_min = min(v_answer)
                if v_max > -v_min:
                    pulse_width = fwhm(valuelist = v_answer, time = t_scope)
                else:
                    pulse_width = fwhm(valuelist = -v_answer, time = t_scope)

                data['pulse_width'][x].append(pulse_width)
                data['pulse_amplitude'][x].append(get_pulse_amplitude_of_PSPL125000(amplitude = data['amplitude'][x], bits = data['bits'][x]))
                #import pdb; pdb.set_trace()
        ######## detection of threshold event by hand ###########
        if manual_evaluation:
            above_threshold_level = np.array(np.nan)
            threshold_event =np.nan
            threshold_class = tmp_threshold()
            threshold_written_class = threshold_written()
            
            root = tk.Tk()
            root.withdraw()
            waitVar = tk.BooleanVar()
            waitVar1 = tk.BooleanVar()
            for t_scope, v_answer in zip(data['t_ttx'], data['V_ttx']):
                threshold_written_class.state = False
                x_data = t_scope
                y_data = v_answer
                figure_handle, ax_dialog = plt.subplots()
                plt.title('Is a threshold visible?')
                plt.subplots_adjust(bottom=0.25)
                if type(t_cap) == float:
                    ax_dialog.plot(x_data,y_data, picker = True)
                else:
                    ax_dialog.plot(x_data,y_data/50, label = '$I_{\mathrm{Meas.}}$')
                    ax_dialog.plot(t_cap,v_cap/50, label = '$I_{\mathrm{Cap.}}$')
                    v_diff = subtract_capacitive_current(y_data, v_cap)
                    ax_dialog.plot(x_data, v_diff/50, label = '$I_{\mathrm{Diff.}}$', picker = True)
                    ax_dialog.set_xlabel('Time [s]')
                    ax_dialog.set_ylabel('Curreent [A]')
                    ax_dialog.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
                    ax_dialog.yaxis.set_major_formatter(mpl.ticker.EngFormatter())

                ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
                ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])
                b_yes = Button(ax_yes, 'Yes')
                b_yes.on_clicked(threhsold_visible)
                b_no = Button(ax_no, 'No')
                b_no.on_clicked(threshold_invisible)
                root.wait_variable(waitVar)
                plt.close(figure_handle)
                #print(len(data['pulse_amplitude'])-len(data['t_threshold']))         
            root.destroy()
    return data

def eval_vcm_measurement(data, 
hrs_upper = np.nan, 
hrs_lower = np.nan, 
lrs_upper = np.nan,
lrs_lower = np.nan,
do_plots = True):
    impedance = 50
    if(type(data) == str):
        data = pd.read_pickle(data)
    
    if do_plots == True:
        setup_vcm_plots()
        iplots.updateline(data)
        iplots.show()
    #print(type(data))
    fwhm_list = []
    pulse_amplitude = []
    R_hrs = []
    R_lrs = []

    ###### Eval Reads ##########################

    for I_hrs, V_hrs, I_lrs, V_lrs in zip(data['I_hrs'], data['V_hrs'], data['I_lrs'], data['V_lrs']):
        r_hrs = determine_resistance(v = V_hrs, i = I_hrs)-impedance
        r_lrs = determine_resistance(v = V_lrs, i = I_lrs)-impedance
        if r_hrs > hrs_upper:
            r_hrs = np.nan
        if r_lrs > lrs_upper:
            r_lrs = np.nan
        if r_hrs < hrs_lower:
            r_hrs = np.nan
        if r_lrs < lrs_lower:
            r_lrs = np.nan
        R_hrs.append(r_hrs)
        R_lrs.append(r_lrs)

    ##### Eval Pulses ##########################

    for t_ttx, V_ttx, pulse_width in zip(data['t_ttx'], data['V_ttx'], data['pulse_width']):
        fwhm_value = calc_fwhm(valuelist = V_ttx, time = t_ttx)
        if fwhm_value < pulse_width:
            fwhm_list.append(pulse_width)
        else:
            fwhm_list.append(calc_fwhm(valuelist = V_ttx, time = t_ttx))

   
    data['R_hrs'] = R_hrs
    data['R_lrs'] = R_lrs
    data['fwhm'] = fwhm_list
    return data



def eval_all_pcm_measurements(filepath):
    ''' executes all eval_pcm_measurements in one directory and bundles the results'''
    if filepath[-1] != '/':
        filepath = filepath + '/'
    files = os.listdir(filepath)
    all_data = []
    for f in files:
        filename = filepath+f
        print(filename)
        all_data.append(eval_pcm_measurement(filename, manual_evaluation = True))
    t_threshold = np.array(all_data[0]['t_threshold'])
    pulse_amplitude = np.array(all_data[0]['pulse_amplitude'])
    t_threshold = []
    for data in all_data:
        if len(t_threshold)>0:
            t_threshold = np.append(t_threshold,np.array(data['t_threshold']))
            pulse_amplitude = np.append(pulse_amplitude,np.array(data['pulse_amplitude']))
        else:
            t_threshold = np.array(data['t_threshold'])
            pulse_amplitude = np.array(data['pulse_amplitude'])
    plot_pcm_vt(pulse_amplitude, t_threshold)
    return all_data, t_threshold, pulse_amplitude

def eval_all_vcm_measurements(filepath, **kwargs):
    ''' executes all eval_vcm_measurements in one directory and bundles the results. Also error propagation is included.'''
    if filepath[-1] != '/':
        filepath = filepath + '/'
    files = os.listdir(filepath)
    all_data = []
    R_hrs_mean = []
    R_hrs_std = []
    R_lrs_mean = []
    R_lrs_std = []
    fwhm_mean = []
    fwhm_std = []
    R_ratio_mean =[]
    R_ratio_std = []

    for f in files:
        filename = filepath+f
        print(filename)
        data = eval_vcm_measurement(filename, **kwargs)
        all_data.append(data)
        R_hrs_mean.append(np.mean(data['R_hrs']))
        R_hrs_std.append(np.std(data['R_hrs']))
        R_lrs_mean.append(np.mean(data['R_lrs']))
        R_lrs_std.append(np.std(data['R_lrs']))
        fwhm_mean.append(np.mean(data['fwhm']))
        fwhm_std.append(np.std(data['fwhm']))
        R_ratio_mean.append(np.mean(data['R_lrs']/data['R_hrs']))
        R_ratio_std.append(R_ratio_mean[-1]*np.sqrt(np.power(R_hrs_std[-1]/R_hrs_mean[-1], 2)+np.power(R_lrs_std[-1]/R_lrs_mean[-1], 2)))

    return all_data, R_hrs_mean, R_hrs_std, R_lrs_mean, R_lrs_std, fwhm_mean, fwhm_std, R_ratio_mean, R_ratio_std

def eval_all_pcm_r_measurements(filepath, t_cap = np.nan, v_cap = np.nan):
    if filepath[-1] != '/':
        filepath = filepath + '/'
    files = os.listdir(filepath)
    all_data = []
    for f in files:
        if 'data' in f or 'Thumbs' in f or '.png' in f:
            pass
        else:
            filename = filepath+f
            print(filename)
            all_data.append(eval_pcm_r_measurement(filename, manual_evaluation = True, t_cap = t_cap, v_cap = v_cap))
    
    t_threshold = []
    pulse_amplitude = []
    R_pre = []
    R_post = []
    for data in all_data:
        t_threshold.append(np.array(data['t_threshold'][0])[0])
        pulse_amplitude.append(data['pulse_amplitude'][0])
        R_pre.append(np.mean(data['V_pre'][0]/data['I_pre'][0]))
        R_post.append(np.mean(data['V_post'][0]/data['I_post'][0]))
    plot_R_threshold(R_pre, t_threshold)
    print('Amplitude = ' + str(pulse_amplitude[0]) + 'V')
    
    export_data = {}
    export_data['all_data'] = all_data
    export_data['t_threshold'] = t_threshold
    export_data['pulse_amplitude'] = pulse_amplitude
    export_data['R_pre'] = R_pre
    export_data['R_post'] = R_post
    export_data = pd.DataFrame(export_data)
    

    file_name = os.path.join(filepath + 'data')
    file_link = Path(file_name + '.df')
    i=0
    while file_link.is_file():
        i +=1
        file_name = os.path.join(filepath + 'data_' + str(i))
        file_link = Path(file_name+ '.df')

    write_pandas_pickle(data, file_name)

    return all_data, t_threshold, pulse_amplitude, R_pre, R_post


def get_pulse_amplitude_of_PSPL125000(amplitude, bits):
    '''returns pulse amplitude in Volts depending on the measured output of the PSPL12500'''
    pulse_amplitude = np.nan
    if np.isnan(amplitude):
        return np.nan

    if bits == 1:
        amplitude_array = [0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return_values = [2*0.728, 2*0.776, 2*0.8356, 2*1.1088, 2*1.5314, 2*2.0028, 
        2*2.306727, 2*2.622, 2*2.8624, 2*3.144727, 2*3.378, 2*3.652, 2*4.184]

    else:
        amplitude_array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return_values = [2*0.5956, 2*0.6481, 2*0.7188, 2*0.7757, 2*0.8182, 2*0.8952, 2*1.1693, 2*1.7592, 2*2.2008, 
        2*2.605455, 2*2.9248, 2*3.2552, 2*3.541818, 2*3.872, 2*4.2144, 2*4.7756]

    index = where(np.array(amplitude_array) == amplitude)

    if index.size > 0:
        pulse_amplitude = return_values[int(index)]
    else:
        print('Unknown amplitude')
        index_pre = where(np.array(amplitude_array) > amplitude)[0]

        x= amplitude%1
        pulse_amplitude = (x*return_values[int(index_pre+1)]+(1-x)*return_values[int(index_pre)])

    return pulse_amplitude

def plot_pcm_amp_comp(data, i = 0):
    fig1 = plt.figure()
    ax_cmp = plt.gca()

    for j in range(0,len(data)):
        ax_cmp.plot(data[j]['t_scope'][i],data[j]['v_answer'][i]/max(abs(data[j]['v_answer'][i])), 
        label=str(round(data[j]['pulse_amplitude'][i],2)) + ' V')

    fig1.show()
    ax_cmp.set_ylabel('Norm. voltage [a.u.]')
    ax_cmp.set_xlabel('Time [s]')
    ax_cmp.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    handles, labels = ax_cmp.get_legend_handles_labels()
    ax_cmp.legend(handles, labels, loc = 'lower right')

def plot_pcm_vt(pulse_amplitude, t_threshold):
    fig = plt.figure()
    ax_vt = plt.gca()
    ax_vt.semilogy(pulse_amplitude, t_threshold,'.k')
    ax_vt.set_ylabel('t_Threshold [s]')
    ax_vt.set_xlabel('Votlage [V]')
    ax_vt.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    fig.show()

class tmp_threshold():
    '''Allows to save the threshold obtained by clicking eval_pcm_measurement => there maybe a better solultion'''
    threshold = np.nan
    def set_threshold(self, threshold_value):
        if len(threshold_value) > 1:
            print('More than one point selected. Zoom closer to treshold event')
            self.threshold = numpy.nan
        else:
            self.threshold = threshold_value[0]

def add_suffix_to_dict(data, suffix):
    return {k+suffix:v for k,v in data.items()}

class threshold_written():
    state = False

def combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list = np.nan):
    hrs_df = pd.DataFrame(hrs_list)
    lrs_df = pd.DataFrame(lrs_list)
    scope_df = pd.DataFrame(scope_list)
    if sweep_list is not np.nan:
        sweep_df = pd.DataFrame(sweep_list)
        return_frame = pd.concat([hrs_df, lrs_df, scope_df, sweep_df] , axis = 1)
    else: 
        return_frame = pd.concat([hrs_df, lrs_df, scope_df] , axis = 1)
    return return_frame

def calc_fwhm(valuelist, time, peakpos=-1):
    """calculates the full width at half maximum (fwhm) of some curve.
    the function will return the fwhm with sub-pixel interpolation. It will start at the maximum position and 'walk' left and right until it approaches the half values.
    INPUT: 
    - valuelist: e.g. the list containing the temporal shape of a pulse 
    OPTIONAL INPUT: 
    -peakpos: position of the peak to examine (list index)
    the global maximum will be used if omitted.
    OUTPUT:
    -fwhm (value)
    """
    if peakpos== -1: #no peakpos given -> take maximum
        peak = np.max(valuelist)
        peakpos = np.min( np.nonzero( valuelist==peak  )  )

    peakvalue = valuelist[peakpos]
    phalf = peakvalue / 2.0

    # go left and right, starting from peakpos
    ind1 = peakpos
    ind2 = peakpos   

    while ind1>2 and valuelist[ind1]>phalf:
        ind1=ind1-1
    while ind2<len(valuelist)-1 and valuelist[ind2]>phalf:
        ind2=ind2+1  
    #ind1 and 2 are now just below phalf
    grad1 = valuelist[ind1+1]-valuelist[ind1]
    grad2 = valuelist[ind2]-valuelist[ind2-1]
    #calculate the linear interpolations
    p1interp= ind1 + (phalf -valuelist[ind1])/grad1
    p2interp= ind2 + (phalf -valuelist[ind2])/grad2
    #calculate the width
    width = p2interp-p1interp

    ### calculate pulse widht
    time_step = time[1]-time[0]
    fwhm = width*time_step
    if np.isinf(fwhm):
        return np.nan 
    return fwhm

def determine_resistance(i, v):
    '''returns average resistance of all entries'''
    i = np.array(i)
    v = np.array(v)
    r = np.mean(v/i)
    if r < 0:
        return np.nan
    else:
        return r

def deb_to_atten(deb):
    return np.power(10, -deb/20)

def savefig2(fig_handle, location):
    location = location + '.fig.pickle'
    pickle.dump(fig_handle, open(location, 'wb'))

def openfig(location):
    fig = pickle.load(open(location, 'rb'))
    fig.show()
    return fig

def plot_pcm_transients(data):
    i=1
    fig, ax = plt.subplots()
    for time, voltage in zip(data['t_scope'], data['v_answer']):
        ax.plot(time, np.array(voltage)/50, label = str(i))
        i+=1
    ax.legend(loc = 'lower right')
    ax.set_ylabel('Current [A]')
    ax.set_xlabel('Time [s]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
    fig.tight_layout()
    fig.show()

def plot_pcm_threshold(data):
    fig, ax = plt.subplots()
    ax.semilogy(data['t_threshold'],'.')
    ax.set_xlabel('Pulse No')
    ax.set_ylabel('t_threshold [s]')
    fig.tight_layout()
    fig.show()


def transition_time(fwhm, R_ratio, upper_limit = 0.9, lower_limit = 0.1, reset = False):
    R_ratio = np.array(R_ratio)
    fwhm = np.array(fwhm)

    #sorting
    sorted_index = np.argsort(fwhm)
    fwhm=fwhm[sorted_index]
    R_ratio = R_ratio[sorted_index]

    if not reset:
        index = where(R_ratio < upper_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        start_index = index[0]-1    #last entry at which all values are above the upper limit
        index = where(R_ratio > lower_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        end_index = index[-1] + 1   #first entry at which all values are below the lower limit
    else:
        index = where(R_ratio > lower_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        start_index = index[0]-1    #last entry at which all values are below the lower limit
        index = where(R_ratio < below_limit)
        if index.size < 1: 
            return np.nan, np.nan, np.nan
        end_index = index[-1] + 1   #first entry at which all values are below the upper limit
    try:    
        t_start = fwhm[start_index]    
        t_end = fwhm[end_index]
    except:
        print('Out of array error => returning nan')
        return np.nan, np.nan, np.nan

    transition_time = t_end-t_start
    return transition_time, t_start, t_end

def roundup10(value):
    '''Rounds to the next higher order of magnitude. E.g. for a value of 3.17e-3 this function 
    would return 1e-2'''
    #Usefull for detecting the right range of a smu.
    log_value = np.log10(value)
    exponent = np.ceil(log_value)
    return np.power(10,exponent) 

def set_time(fwhm, R_ratio, limit = 0.5, reset = False):
    fwhm = np.array(fwhm)
    R_ratio = np.array(R_ratio)
    
    #sorting
    sorted_index = np.argsort(fwhm)
    fwhm=fwhm[sorted_index]
    R_ratio = R_ratio[sorted_index]

    if not reset:
        index = where(R_ratio > limit)
        if index.size < 1: 
            return fwhm[0]
        t_set_index = index[-1] + 1
    else:
        index = where(R_ratio < limit)
        if index.size < 1: 
            return fwhm[0]
        t_set_index = index[-1] + 1

    try:
        t_set = fwhm[t_set_index]
    except:
        print('Out of array error => returning nan')
        return np.nan
    return t_set

def get_R_median(all_data):
    R = []
    fwhm_array =  []
    for data in all_data:
        ratio = np.array(data['R_lrs']/data['R_hrs'])
        index = where(~np.isnan(ratio))
        ratio = ratio[index]
        R.append(np.median(ratio))
        fwhm_array.append(np.mean(data['fwhm']))
    fwhm_array = np.array(fwhm_array)
    R = np.array(R)   
    sorted_index = np.argsort(np.array(fwhm_array))
    fwhm_array = fwhm_array[sorted_index]
    R = R[sorted_index]     
    return fwhm_array, R

def Boxplot_array(all_data):
    R = []
    fwhm_array = []
    for data in all_data:
        ratio = np.array(data['R_lrs']/data['R_hrs'])
        index = where(~np.isnan(ratio))
        ratio = ratio[index]
        if np.size(ratio)>0:
            R.append(ratio)
            fwhm_array.append(np.mean(data['fwhm']))
    fwhm_array = np.array(fwhm_array)
    R = np.array(R)
    sorted_index = np.argsort(np.array(fwhm_array))
    fwhm_array = fwhm_array[sorted_index]
    R = R[sorted_index]
    R = np.ndarray.tolist(R)
    fwhm_array = np.ndarray.tolist(fwhm_array)

    return fwhm_array, R

def plot_R_threshold(r, t):
    fig, ax = plt.subplots()
    ax.loglog(r,t,'.')
    ax.set_xlabel('Resistance [$\Omega$]')
    ax.set_ylabel('$t_{\mathrm{Threshold}}$ [s]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    fig.tight_layout()
    fig.show()
    return fig, ax

def plot_R_threshold_color(r, t):
    fig, ax = plt.subplots()
    sc = ax.scatter(r,t,cmap = 'rainbow', c= range(len(t)))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(left=1e4, right =1.2e7)
    ax.set_ylim(bottom = 1e-10, top = 12e-9)
    ax.set_xlabel('Resistance [$\Omega$]')
    ax.set_ylabel('$t_{\mathrm{Threshold}}$ [s]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    plt.colorbar(sc)
    fig.tight_layout()
    fig.show()
    return fig, ax

def subtract_capacitive_current(v_meas, v_cap, trigger_position = 20):
    start_index_meas = int(trigger_position/100*len(v_meas))
    start_index_cap = int(trigger_position/100*len(v_cap))
    begin_cap = start_index_cap-start_index_meas
    end_cap = len(v_meas) + start_index_cap -start_index_meas
    try:
        v_diff = v_meas - v_cap[begin_cap:end_cap]
    except:
        v_diff = v_meas
        print('v_cap to short')
    return v_diff

def hs(x=0):
    return np.heaviside(x,1)

def threshold(time, t_start= 7.7e-9, t_diff = 0.7e-9, r_start = 1e6, r_end = 1200):
    r_diff = r_start-r_end
    t_end = t_start + t_diff
    r_slope = r_diff/t_diff
    return r_start - hs(t_end-time)*hs(time-t_start)*r_slope*(time-t_start)

