def plot0(data, ax=None, **kwargs):
    ax.plot(data['t'], data['V'], **kwargs)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot1(data, ax=None, **kwargs):
    ax.semilogy(data['t'], data['V'] / data['I'], **kwargs)
    ax.set_ylabel('Resistance [V/A]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot2(data, ax=None, **kwargs):
    ax.plot(data['t'], data['V'], **kwargs)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot3(data, ax=None, **kwargs):
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
k.it(0.1, 0, 10, 0.5, 0, 1, 1)
t.CH(1, False)
t.CH(2, True)
t.CH(3, False)
t.CH(4, True)
t.Scale(2, 0.1)
t.Scale(4, 0.1)
t.Position(2, 4)
t.Position(4, 4)
t.ChangeSamplerateandRecodlength(100e9,5000)
t.Arm(4,-0.1,'e')
while not k.done():
    trigger_status = t.ask('TRIG:STATE?')
    data = k.get_data()
    if trigger_status == 'READY\n':
        plt.pause(0.1)
    else:
        number_of_events +=1
        data_scope1 = t.get_curve(2)
        data_scope2 = t.get_curve(4)
        print(number_of_events)
        data_scope_all['t_scope'+str(number_of_events)] = data_scope1['t']
        data_scope_all['v_pulse'+str(number_of_events)] = data_scope1['V']
        data_scope_all['v_answer'+str(number_of_events)] = data_scope2['V']
        t.Arm(4,-0.1,'e')
    iplots.updateline(data)

k.channels_off()        
data.update(data_scope_all)
savedata(data)
t.write('ACQ:STATE 0')

# do other measurements ...

