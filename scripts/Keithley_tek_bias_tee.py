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

def eval_pcm_measurement(data):
    setup_plots()
    '''evaluates saved data (location or variable) from an  measurements. In case of a two channel measurement it determines pulse amplitude and width'''
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
    return data

