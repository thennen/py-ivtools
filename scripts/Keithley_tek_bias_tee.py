from matplotlib.widgets import Button
import tkinter as tk


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def setup_plots():
    def plot0(data, ax=None, **kwargs):
        ax.cla()
        ax.set_title('Answer')
        if data['t_scope']:
            ax.plot(data['t_scope'][-1], data['v_answer'][-1], **kwargs)    
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [S]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot1(data, ax=None, **kwargs):
        ax.cla()
        ax.semilogy(data['t'], data['V'] / data['I'], **kwargs)
        if data['t_event']:
            ax.vlines(data['t_event'],ax.get_ylim()[0]*1.2,ax.get_ylim()[1]*0.8, alpha = 0.5)
        ax.set_ylabel('Resistance [V/A]')
        ax.set_xlabel('Time [S]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot2(data, ax=None, **kwargs):
        ax.cla()
        ax.set_title('Pulse')
        if data['t_scope']:
            ax.plot(data['t_scope'][-1], data['v_pulse'][-1], **kwargs)
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [S]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    def plot3(data, ax=None, **kwargs):
        ax.cla()
        ax.plot(data['t'], data['I'], **kwargs)
        if data['t_event']:
            ax.vlines(data['t_event'],ax.get_ylim()[0]*1.2,ax.get_ylim()[1]*0.8, alpha = 0.5)
        ax.set_ylabel('Current [A]')
        ax.set_xlabel('Time [S]')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        
    iplots.plotters = [[0, plot0],
                       [1, plot1],
                       [2, plot2],
                       [3, plot3]]
                 
    iplots.newline()

def pcm_measurement(samplename, samplepad, amplitude = 10, bits = 256, sourceVA = -0.2, points = 250, 
 interval = 0.1, trigger = -0.3, two_channel = False):
    setup_plots()
    '''run a measurement during which the Keithley2600 applies a constants voltage and measures the current. 
    Pulses applied during this measurement are also recorded. '''
    number_of_events =0
    data = {}
    data['t_scope'] = []
    data['v_pulse'] = []
    data['v_answer'] = []
    data['t_event'] = []
    data['amplitude'] = amplitude
    data['bits'] = bits
    iplots.show()    

    datafolder = 'C:/Messdaten/' + samplename + '/' + samplepad + '/'

    k.it(sourceVA = sourceVA, sourceVB = 0, points = points, interval = interval, rangeI = 0, limitI = 1, nplc = 1)

    ttx.inputstate(1, False)
    ttx.inputstate(2, True)
    ttx.inputstate(3, False)
    if two_channel:
        ttx.inputstate(4, True)
        ttx.scale(4, 0.4)
        ttx.position(4, 4)
    else:
        ttx.inputstate(4, False)
    ttx.scale(2, 0.03)
    ttx.position(2, 1)
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
            data_scope1 = ttx.get_curve(4)
            data_scope2 = ttx.get_curve(2)
            
            time_array = data['t']
            data['t_scope'].append(data_scope1['t_ttx'])
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
    savedata(data)

def eval_pcm_measurement(data, manual_evaluation = False):
    '''evaluates saved data (location or variable) from an  measurements. In case of a two channel measurement it determines pulse amplitude and width'''
    setup_plots()
    def agree(self):
        waitVar1.set(True)
    def trehsold_visible(self):
        
        ax_agree = plt.axes([0.59, 0.05, 0.1, 0.075])
        b_agree = Button(ax_agree,'Agree')
        b_agree.on_clicked(agree)
        cid = figure_handle.canvas.mpl_connect('pick_event', onpick)
        root.wait_variable(waitVar1)

        waitVar.set(True)
        print(threshold_class.threshold)
        print(data['t_scope'][i][pulse_index[0][0]])
        data['t_threshold'].append(threshold_class.threshold-data['t_scope'][i][pulse_index[0][0]])

    def threshold_invisible(self):
        data['t_threshold'].append(numpy.nan)
        waitVar.set(True)

    def write_threshold(self, threshold_value):
        t_threshold = threshold_value

    def onpick(event):
        ind = event.ind
        t_threshold = np.take(x_data, ind)
        print('onpick3 scatter:', ind, t_threshold, np.take(y_data, ind))
        threshold_class.set_threshold(t_threshold)
        if len(ind) == 1:
            ax_dialog.plot(np.array([t_threshold,t_threshold]),np.array([-1,0.3]))
            plt.pause(0.1)

    if(type(data) == str):
        data = pd.read_pickle(data)
    iplots.show()    
    iplots.updateline(data)
    data['pulse_width'] = []
    if data['v_pulse']:       
        data['pulse_amplitude'] = []
        for i in range(0,len(data['v_pulse'])-1):
            pulse_minimum =min(data['v_pulse'][i])
            pulse_index = np.where(np.array(data['v_pulse'][i]) < 0.5* pulse_minimum)
 
            data['pulse_width'].append(data['t_scope'][i][pulse_index[0][-1]]-data['t_scope'][i][pulse_index[0][0]])
            data['pulse_amplitude'].append(np.mean(data['v_pulse'][i][pulse_index])*2)
    else:
        for i in range(0,len(data['v_answer'])-1):
            pulse_minimum =min(data['v_answer'][i])
            pulse_index = np.where(np.array(data['v_answer'][i]) < 0.5* pulse_minimum)

            data['pulse_width'].append(data['t_scope'][i][pulse_index[0][-1]]-data['t_scope'][i][pulse_index[0][0]])
    if manual_evaluation:
        threshold_class = tmp_threshold()
        data['t_threshold'] = []
        root = tk.Tk()
        root.withdraw()
        waitVar = tk.BooleanVar()
        waitVar1 = tk.BooleanVar()
        for i in range(0,len(data['v_answer'])-1):
            x_data = data['t_scope'][i]
            y_data = data['v_answer'][i]/max(abs(data['v_answer'][i]))
            figure_handle, ax_dialog = plt.subplots()
            plt.title('Is a threshold visible?')
            plt.subplots_adjust(bottom=0.25)
            ax_dialog.plot(x_data,y_data, picker = True)
            ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
            ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])
            b_yes = Button(ax_yes, 'Yes')
            b_yes_conn = b_yes.on_clicked(trehsold_visible)
            b_no = Button(ax_no, 'No')
            b_no_conn = b_no.on_clicked(threshold_invisible)
            root.wait_variable(waitVar)
            plt.close(figure_handle)
    root.destroy()
    return data

def eval_all_pcm_measurements(filepath):
    files = os.listdir(filepath)
    all_data = []
    for f in files:
        filename = filepath+f
        print(filename)
        all_data.append(eval_pcm_measurement(filename, manual_evaluation = True))
    return all_data



class tmp_threshold():
    threshold = numpy.nan
    def set_threshold(self, threshold_value):
        if len(threshold_value) > 1:
            print('More than one object selected')
            self.threshold = numpy.nan
        else:
            self.threshold = threshold_value
            print('in class ' + str(self.threshold))