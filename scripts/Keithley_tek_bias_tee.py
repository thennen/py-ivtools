from matplotlib.widgets import Button
import tkinter as tk
import sys
from pathlib import Path

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
            ax.plot(line.t_hrs,  line.V_hrs /  line.I_hrs)
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
        ax.set_title('Read LRS #' + str(len(data)))
        if not np.isnan(line.t_lrs).any():
            ax.cla()
            ax.set_title('Read LRS #' + str(len(data)))
            ax.plot(line.t_lrs,  line.V_lrs /  line.I_lrs)
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
    
def set_keithley_plotters():
    iplots.plotters = keithley_plotters
    iplots.ax0.cla()
    iplots.ax1.cla()
    iplots.ax2.cla()
    iplots.ax3.cla()

def pcm_measurement(samplename, samplepad, amplitude = 10, bits = 256, sourceVA = -0.2, points = 250, 
 interval = 0.1, trigger = -0.3, two_channel = False, rangeI = 0):
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
    data['samplepad'] = samplepad
    data['samplename'] = samplename
    iplots.show()    

    datafolder = 'C:/Messdaten/' + samplename + '/' + samplepad + '/'

    k.it(sourceVA = sourceVA, sourceVB = 0, points = points, interval = interval, rangeI = rangeI, limitI = 1, nplc = 1)

    ttx.inputstate(1, False)
    ttx.inputstate(2, True)
    ttx.inputstate(3, False)
    if two_channel:
        ttx.inputstate(4, True)
        ttx.scale(4, 0.4)
        ttx.position(4, 4)
    else:
        ttx.inputstate(4, False)
    ttx.scale(2, 0.04)
    ttx.position(2, 3.5)
    ttx.change_samplerate_and_recordlength(100e9, 5000)
    if two_channel:
        ttx.arm(source = 4, level = trigger, edge = 'e')
    else:
        ttx.arm(source = 2, level = trigger, edge = 'e')
    while not k.done():
        data.update(k.get_data())
        if ttx.triggerstate():
            plt.pause(0.1)
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
             will be used as time reference (-2 instead of -1, which be the last entry)'''
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
    
    datafolder = os.path.join('C:\Messdaten', samplename, samplepad)
    subfolder = datestr
    file_exits = True
    i=1
    filepath = os.path.join(datafolder, subfolder, '_pcm_measurement_'+str(i))
    file_link = Path(filepath + '.df')
    while file_link.is_file():
        i +=1
        filepath = os.path.join(datafolder, subfolder, '_pcm_measurement_'+str(i))
        file_link = Path(filepath + '.df')
    io.write_pandas_pickle(meta.attach(data), filepath)

    return data

def vcm_pg5_measurement(samplename, samplepad, v1, v2, step = 0.02, V_read = 0.2, range_lrs = 1e-3, range_hrs = 1e-4, range_sweep = 1e-2, 
    cycles = 1, pulse_width = 50e-12, attenuation = 0,  automatic_measurement = True):
    setup_vcm_plots()

    data['samplepad'] = samplepad
    data['samplename'] = samplename

    hrs_list = []
    lrs_list = []
    sweep_list = []
    scope_list = []
    vlist = tri(v1 = v1, v2 = v2, step = step)

    for i in range(cycles):

        ### Reading HRS resistance ############################################################################

        k.it(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.01, rangeI = range_hrs , limitI = 1, nplc = 1)
        while not k.done():
            plt.pause(0.1)
        k.set_channel_state('A', False)
        k.set_channel_state('B', False)
        hrs_data = k.get_data()
        hrs_list.append(add_suffix_to_dict(hrs_data,'_hrs'))
        data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
        iplots.updateline(data)
        ### RSetting up scope  ################################################################################

        ttx.inputstate(1, False)
        ttx.inputstate(2, True)
        ttx.inputstate(3, False)
        ttx.inputstate(4, False)
        if attenuation == 3:
            trigger_level = 0.04
            ttx.scale(2, 0.1)
            ttx.position(2, -4)
        elif attenuation == 6:
            trigger_level = 0.03
            ttx.scale(2, 0.07)
            ttx.position(2, -2)
        elif attenuation ==10:
            trigger_level = 0.02
            ttx.scale(2, 0.05)
            ttx.position(2, -2)
        elif attenuation == 13:
            trigger_level = 0.02
            ttx.scale(2, 0.04)
            ttx.position(2, -2)
        elif attenuation == 16:
            trigger_level = 0.02
            ttx.scale(2, 0.03)
            ttx.position(2, -2)
        else:
            trigger_level = 0.05
            ttx.scale(2, 0.12)
            ttx.position(2, -4.5)

        ttx.change_samplerate_and_recordlength(samplerate = 100e9, recordlength=250)
        if pulse_width < 100e-12:
            ttx.trigger_position(40)
        elif pulse_width < 150e-12:
            ttx.trigger_position(30)
        else:
            ttx.trigger_position(20)



        plt.pause(0.1)

        ttx.arm(source = 2, level = trigger_level, edge = 'e')


        ### Applying pulse and reading scope data #############################################################

        if not automatic_measurement:
            input('Connect the RF probes and press enter')
        pg5.set_pulse_width(pulse_width)
        plt.pause(0.1)
        pg5.trigger()
        plt.pause(0.1)
        if not ttx.triggerstate:
            plt.pause(0.1)
        scope_list.append(ttx.get_curve(2))
        data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
        iplots.updateline(data)

        ### Reading LRS resistance #############################################################################

        if not automatic_measurement:
            input('Connect the DC probes and press enter')
        k.it(sourceVA = V_read, sourceVB = 0, points =10, interval = 0.01, rangeI = range_lrs, limitI = 1, nplc = 1)
        while not k.done():
            plt.pause(0.1)
        k.set_channel_state('A', False)
        k.set_channel_state('B', False)
        lrs_data = k.get_data()
        lrs_list.append(add_suffix_to_dict(lrs_data,'_lrs'))
        data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
        iplots.updateline(data)

        ### performing sweep ###################################################################################
        
        k.iv(vlist, Irange = range_sweep) 
        while not k.done():
            plt.pause(0.1)
        k.set_channel_state('A', False)
        k.set_channel_state('B', False)
        sweep_data = k.get_data()
        sweep_list.append(add_suffix_to_dict(sweep_data,'_sweep'))
        data = combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list)
        iplots.updateline(data)
        #savedata(data,datafolder + str(pulse_width) + 'ps')

    data['attenuation'] = attenuation
    data['pulse_width'] = pulse_width

    datafolder = os.path.join('C:\Messdaten', samplename, samplepad)
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

    return data

def eval_pcm_measurement(data, manual_evaluation = False):
    '''evaluates saved data (location or variable) from an  measurements. In case of a two channel measurement it determines pulse amplitude and width'''
    setup_pcm_plots()
    ########## declareation of buttons ###########
    def agree(self):
        waitVar1.set(True)

    def threhsold_visible(self):
        ax_dialog.set_title('Please indicate threshold')
        ax_agree = plt.axes([0.59, 0.05, 0.1, 0.075])
        b_agree = Button(ax_agree,'Agree')
        b_agree.on_clicked(agree)
        cid = figure_handle.canvas.mpl_connect('pick_event', onpick)
        root.wait_variable(waitVar1)
        if not threshold_written_class.state:
            data['t_threshold'].append(threshold_class.threshold-t_scope[pulse_index[0]])
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
            ax_dialog.plot(np.array([pulse_start,pulse_start]),np.array([-1,0.3]))
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
            pulse_end = t_scope[pulse_index[-1]]
            pulse_start = t_scope[pulse_index[0]]

            data['pulse_width'].append(pulse_end-pulse_start)
            data['pulse_amplitude'].append(np.mean(v_pulse[pulse_index])*2)
        
    ########## if one channel experiment: ################       
    else:
        for t_scope, v_answer in zip(data['t_scope'],data['v_answer']):
            pulse_minimum =min(v_answer)
            pulse_index = where(np.array(v_answer) < 0.5* pulse_minimum)
            pulse_start_index = pulse_index[0]
            pulse_start = t_scope[pulse_start_index]
            pulse_end_index = pulse_start_index + where(np.array(v_answer[pulse_start_index:-1]) >= 0)[0]
            pulse_end = t_scope[pulse_end_index]
            '''for short pulses the width is determined as FWHM, otherwise from pulse start until 
             the zero line is crossed for the first time '''
            if pulse_end - pulse_start < 1e-9:
                pulse_end = t_scope[pulse_index[-1]]
            data['pulse_width'].append(pulse_end-pulse_start)
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
        root.destroy()
    return data

def eval_vcm_measurement(data):
    impedance = 50
    setup_vcm_plots()
    if(type(data) == str):
        data = pd.read_pickle(data)
    iplots.show()
    iplots.updateline(data)
    print(type(data))
    fwhm_list = []
    pulse_amplitude = []
    R_hrs = []
    R_lrs = []

    ###### Eval Reads ##########################

    for I_hrs, V_hrs, I_lrs, V_lrs in zip(data['I_hrs'], data['V_hrs'], data['I_lrs'], data['V_lrs']):
        R_hrs.append(determine_resistance(v = V_hrs, i = I_hrs)-impedance)
        R_lrs.append(determine_resistance(v = V_lrs, i = I_lrs)-impedance)

    ##### Eval Pulses ##########################

    for t_ttx, V_ttx, pulse_width in zip(data['t_ttx'], data['V_ttx'], data['pulse_width']):
        fwhm_value = fwhm(valuelist = V_ttx, time = t_ttx)
        if fwhm_value < pulse_width:
            fwhm_list.append(pulse_width)
        else:
            fwhm_list.append(fwhm(valuelist = V_ttx, time = t_ttx))

   
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

def eval_all_vcm_measurements(filepath):
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
        data = eval_vcm_measurement(filename)
        all_data.append(data)
        R_hrs_mean.append(np.mean(data['R_hrs']))
        R_hrs_std.append(np.std(data['R_hrs']))
        R_lrs_mean.append(np.mean(data['R_lrs']))
        R_lrs_std.append(np.std(data['R_lrs']))
        fwhm_mean.append(np.mean(data['fwhm']))
        fwhm_std.append(np.std(data['fwhm']))
        R_ratio_mean.append(R_lrs_mean[-1]/R_hrs_mean[-1])
        R_ratio_std.append(R_ratio_mean[-1]*np.sqrt(np.power(R_hrs_std[-1]/R_hrs_mean[-1], 2)+np.power(R_lrs_std[-1]/R_lrs_mean[-1], 2)))

    return all_data, R_hrs_mean, R_hrs_std, R_lrs_mean, R_lrs_std, fwhm_mean, fwhm_std, R_ratio_mean, R_ratio_std

def get_pulse_amplitude_of_PSPL125000(amplitude, bits):
    '''returns pulse amplitude in Volts depending on the measured output of the PSPL12500'''
    pulse_amplitude = numpy.nan
    if bits == 1:
        if amplitude == 0.3:
            pulse_amplitude = 2*0.728
        elif amplitude == 0.4:
            pulse_amplitude = 2*0.776
        elif amplitude == 0.5:
            pulse_amplitude = 2*0.8356
        elif amplitude == 1:
            pulse_amplitude = 2*1.1088
        elif amplitude == 2:
            pulse_amplitude = 2*1.5314
        elif amplitude == 3:
            pulse_amplitude = 2*2.0028
        elif amplitude == 4:
            pulse_amplitude = 2*2.306727
        elif amplitude == 5:
            pulse_amplitude = 2*2.622
        elif amplitude == 6:
            pulse_amplitude = 2*2.8624
        elif amplitude == 7:
            pulse_amplitude = 2*3.144727
        elif amplitude == 8:
            pulse_amplitude = 2*3.378
        elif amplitude == 9:
            pulse_amplitude = 2*3.652
        elif amplitude == 10:
            pulse_amplitude = 2*4.184
        else: 
            print('Unknown amplitude')

    elif bits > 1:
        if amplitude == 0.3:
            pulse_amplitude = 2*0.7646
        elif amplitude == 0.4:
            pulse_amplitude = 2*0.8182
        elif amplitude == 0.5:
            pulse_amplitude = 2*0.8952
        elif amplitude == 1:
            pulse_amplitude = 2*1.1693
        elif amplitude == 2:
            pulse_amplitude = 2*1.7592
        elif amplitude == 3:
            pulse_amplitude = 2*2.2008
        elif amplitude == 4:
            pulse_amplitude = 2*2.605455
        elif amplitude == 5:
            pulse_amplitude = 2*2.9248
        elif amplitude == 6:
            pulse_amplitude = 2*3.2552
        elif amplitude == 7:
            pulse_amplitude = 2*3.541818
        elif amplitude == 8:
            pulse_amplitude = 2*3.872
        elif amplitude == 9:
            pulse_amplitude = 2*4.2144
        elif amplitude == 10:
            pulse_amplitude = 2*4.7756
        else: 
            print('Unknown amplitude')

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
            self.threshold = threshold_value

def add_suffix_to_dict(data, suffix):
    return {k+suffix:v for k,v in data.items()}

class threshold_written():
    state = False

def combine_lists_to_data_frame(hrs_list, lrs_list, scope_list, sweep_list):
    hrs_df = pd.DataFrame(hrs_list)
    lrs_df = pd.DataFrame(lrs_list)
    scope_df = pd.DataFrame(scope_list)
    sweep_df = pd.DataFrame(sweep_list)
    return pd.concat([hrs_df, lrs_df, scope_df, sweep_df] , axis = 1)

def fwhm(valuelist, time, peakpos=-1):
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
    return fwhm

def determine_resistance(i, v):
    '''returns average resistance of all entries'''
    i = np.array(i)
    v = np.array(v)
    i_mean = np.mean(i)
    v_mean = np.mean(v)
    r = v_mean/i_mean
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