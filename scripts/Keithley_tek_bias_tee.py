def plot0(data, ax=None, **kwargs):
    ax.cla()
    ax.set_title('Answer')
    try:
        ax.plot(data_scope2['t_ttx'], data_scope2['V_ttx'], **kwargs)
    except:
        pass    
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot1(data, ax=None, **kwargs):
    ax.cla()
    ax.semilogy(data['t'], data['V'] / data['I'], **kwargs)
    ax.set_ylabel('Resistance [V/A]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot2(data, ax=None, **kwargs):
    ax.cla()
    ax.set_title('Pulse')
    try:
        ax.plot(data_scope1['t_ttx'], data_scope1['V_ttx'], **kwargs)
    except:
        pass
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot3(data, ax=None, **kwargs):
    ax.cla()
    ax.plot(data['t'], data['I'], **kwargs)
    ax.set_ylabel('Current [A]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
iplots.plotters = [[0, plot0],
                   [1, plot1],
                   [2, plot2],
                   [3, plot3]]
             
iplots.newline()
number_of_events =0
data_scope = {}
data_scope_all = {}

datafolder = 'C:/Messdaten/CPW6/x08y13/'
k.it(sourceVA = -0.1, sourceVB = 0, points = 1000, interval = 0.2, rangeI = 0, limitI = 1, nplc = 1)
ttx.inputstate(1, False)
ttx.inputstate(2, True)
ttx.inputstate(3, False)
ttx.inputstate(4, True)
ttx.scale(2, 0.05)
ttx.scale(4, 0.4)
ttx.position(1, 2)
ttx.position(4, 4)
ttx.change_samplerate_and_recordlength(100e9,5000)
ttx.arm(source = 4, level = -0.3, edge = 'e')
while not k.done():
    data = k.get_data()
    if ttx.triggerstate():
        plt.pause(0.1)
    else:
        number_of_events +=1
        data_scope1 = ttx.get_curve(4)
        data_scope2 = ttx.get_curve(2)
        
        time_array = data['t']
        data_scope_all['t_scope'+str(number_of_events)] = data_scope1['t_ttx']
        data_scope_all['v_pulse'+str(number_of_events)] = data_scope1['V_ttx']
        data_scope_all['v_answer'+str(number_of_events)] = data_scope2['V_ttx']
        '''Moritz: last current data point measured after last trigger event so the entry one before
         will be used as time reference (-2 instead of -1, which be the last entry)'''
        data_scope_all['t_event'+str(number_of_events)] = time_array[len(time_array)-2]
        print(time_array[len(time_array)-2])
        data.update(data_scope_all)
        ttx.arm(source = 4, level = -0.3, edge = 'e')
    iplots.updateline(data)

data = k.get_data()
iplots.updateline(data)
k.set_channel_state('A', False)
k.set_channel_state('B', False)
ttx.disarm()
data.update(data_scope_all)
savedata(data)



