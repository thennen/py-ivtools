def plot0(data, ax=None, **kwargs):
    ax.plot(data['t'], data['V'], **kwargs)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot1(data, ax=None, **kwargs):
    ax.plot(data['t'], data['V'] / data['I'], **kwargs)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot2(data, ax=None, **kwargs):
    ax.plot(data['t'], data['V'], **kwargs)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
def plot3(data, ax=None, **kwargs):
    ax.plot(data['t'], data['V'], **kwargs)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [S]')
    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    
iplots.plotters = [[0, plot0],
                   [1, plot1],
                   [2, plot2],
                   [3, plot3]]
                   
iplots.newline()

# do measurements                   
iplots.updateline(data)
# do other measurements ...

