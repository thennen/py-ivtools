def plot0(data, ax=None, **kwargs):
    ax.cla()
    ax.set_title('Answer')
    try:
        ax.plot(data_scope2['t'], data_scope2['V'], **kwargs)
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
        ax.plot(data_scope1['t'], data_scope1['V'], **kwargs)
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

k.it(sourceVA = 0.1, sourceVB = 0, points = 10, interval = 0.2, rangeI = 0, limitI = 1, nplc = 1)
ttx.inputstate(1, False)
ttx.inputstate(2, True)
ttx.inputstate(3, False)
ttx.inputstate(4, True)
ttx.scale(2, 0.1)
ttx.scale(4, 0.1)
ttx.position(2, 4)
ttx.position(4, 4)
ttx.change_samplerate_and_recordlength(100e9,5000)
ttx.arm(source = 4, level = -0.1, edge = 'e')
while not k.done():
    trigger_status = ttx.ask('TRIG:STATE?')
    data = k.get_data()
    if trigger_status == 'READY\n':
        plt.pause(0.1)
    else:
        number_of_events +=1
        data_scope1 = ttx.get_curve(4)
        data_scope2 = ttx.get_curve(2)
        print(number_of_events)
        data_scope_all['t_scope'+str(number_of_events)] = data_scope1['t']
        data_scope_all['v_pulse'+str(number_of_events)] = data_scope1['V']
        data_scope_all['v_answer'+str(number_of_events)] = data_scope2['V']
        ttx.arm(channel = 4, level = -0.1, edge = 'e')
    iplots.updateline(data)
data = k.get_data()
iplots.updateline(data)
k.channels_off()   
ttx.write('ACQ:STATE 0')     
data.update(data_scope_all)
savedata(data)



