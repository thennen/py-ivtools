#from ..ivtools import plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import itertools
import Telegram_bot as tb
from tqdm import tqdm

pd.options.mode.chained_assignment = None

possible_ranges = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
max_offsets = [.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20]

def duration_at_const_slew_rate(slew_rate, V_reset_stop, V_set_stop):
    """
    Returns a pulse duration for a constant, preset, slew_rate. For a triangular pulse with a given V_reset_stop and V_set_stop.
    slew_rate in V/s
    Voltages in V
    Returns duration in s
    """
    return (2*abs(V_reset_stop)+2*abs(V_set_stop))/slew_rate


def is_fs_d_n_ratio_ok(fs, duration, cycles):
    """
    Test whether or not the three parameters are too big so they will exceed the memory of the picoscope.
    """
    fs0 = 1.25e9
    n0 = 100
    d0 = 1e-3
    return True if fs*duration*cycles/(fs0*n0*d0) <= 1 else False

def calc_max_fs_d_n(fs=None, duration=None, cycles=None):
    """
    Calculates the maximum sampling rate, duration and amount of cycles depending on the input values.
    """
    fs = fs
    d = duration
    n = cycles
    fs0 = 1.25e9
    n0 = 100
    d0 = 1e-3
    ratio = fs0*n0*d0
    if fs == d == n == None:
        print('Returning standard parameters, due to no input: samplingrate = {}, cycles = {}, duration = {}'.format(1.25e9, 200, 1e-3))
        return 1.25e9, 200, 1e-3
    elif fs == None:
        if (d != None) and (n != None):
            return ratio/(d*n), d, n
        elif d != None:
            print('Choosing maximum sampling rate!')
            return fs0, d, ratio/(d*fs0)
        elif n != None:
            print('Choosing maximum sampling rate!')
            return fs0, ratio/(n*fs0), n
    elif d == None:
        if (fs != None) and (n != None):
            return fs, ratio/(fs*n) ,n        
        elif fs != None:
            print('Choosing standard duration (1e-3s)')
            return fs, d0, ratio/(fs*d0)
        elif n != None: #duerfte nie aufgerufen werden
            print('Choosing maximum sampling rate!')
            return fs0, ratio/(fs0*n), n
    elif n == None:
        if (d != None) and (fs != None):
            return fs, d, ratio/(fs*d)        
        elif d != None: #duerfte nie aufgerufen werden
            print('Choosing maximum sampling rate!')
            return fs0, d, ratio/(fs0*d) 
        elif fs != None: #duerfte nie aufgerufen werden
            print('Choosing standard duration (1e-3s)')
            return fs, d0, ratio/(fs*d0)
    else:
        if is_fs_d_n_ratio_ok(samplingrate, duration, cycles):
            return fs, d, n
        else:
            print('Ratio not okay, too many samples! Please change a parameter!')
            return None, None, None

def adjust_picorange2(V_reset_stop, V_set_stop, duration=1e-3, fs=1.25e9, smartrange=False):
    """
    Performs one measurement with the given parameters to adjust the range and the offset of the picoscope.
    """
    possible_ranges = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    max_offsets = [.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20]
    ps.range.b = 5
    ps.offset.b = 0
    ps.range.a = 5
    ps.offset.a = 0
    
    ps.squeeze_range(temp,padpercent=0.5)
    if max_offsets[possible_ranges.index(ps.range.b)] < ps.offset.b:
        range_index = np.where(possible_ranges>=ps.offset.b)[0][0]
        
        ### alternatively we could just set the range 
        #ps.range.b=5
        #ps.offset.b=0
        
def adjust_picorange(sweep_data):
    """
    Sets the best possible range at a given offset of a sweep.
    """
    possible_ranges = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    max_offsets = [.5, .5, .5, 2.5, 2.5, 2.5, 20, 20, 20]
    
    ps.squeeze_range(sweep_data,padpercent=0.5)
    if max_offsets[possible_ranges.index(ps.range.b)] < ps.offset.b:
        range_index = np.where(possible_ranges>=ps.offset.b)[0][0]
        ps.range.b=possible_ranges[range_index]
        
        ### alternatively we could just set the range 
        #ps.range.b=5
        #ps.offset.b=0 

def squeeze_range_mul(data_list, padpercent=0, ch=['A', 'B', 'C', 'D']):
    for c in ch:
        offset_list = []
        range_list = []
        for data in data_list:
            if c in data:
                usedatten = data['ATTENUATION'][c]
                usedcoupling = data['COUPLINGS'][c]
                usedrange = data['RANGE'][c]
                usedoffset = data['OFFSET'][c]
                if type(data[c][0]) is np.int8:
                    # Need to convert to float
                    # TODO: consider attenuation
                    maximum = np.max(data[c]) / 2**8 * usedrange * 2 - usedoffset
                    minimum = np.min(data[c]) / 2**8 * usedrange * 2 - usedoffset
                    rang, offs = ps.best_range((minimum, maximum), padpercent=padpercent, atten=usedatten, coupling=usedcoupling)
                else:
                    rang, offs = ps.best_range(data[c], padpercent=padpercent, atten=usedatten, coupling=usedcoupling)
                
                range_list.append(rang)
                offset_list.append(offs)
        #print(c, offset_list)
        #print(c, range_list)
        if range_list != []:
            ps.offset[c] = np.mean(offset_list)
            ps.range[c] = max(range_list)
            print('Setting picoscope channel {} range {}, offset {}'.format(c, max(range_list), np.mean(offset_list)))
        

def endurance(I_CC, V_reset_stop, V_set_stop, cycles=10000, cycles_per_sweep = 1000, duration=1e-3, fs=1.25e9):
    '''
    Define Current Compliance in micro-A,
    V_reset_stop and V_set_stop in Volt,
    and regular picoiv-stuff.
    '''
    path = os.path.join(datadir(),"Endurance_V_set_stop{}_V_reset_stop_{}_Icc_{}_cycles_{}_{}".format(int(V_set_stop*100),int(V_reset_stop*100),int(I_CC),cycles,time.strftime('%d%m%y-%H%M%S')))
    if os.path.isfile(path):
        pass
    else:
        os.mkdir(path)
    set_compliance(I_CC*1e-6)
    
    ps.range.b = 5
    ps.offset.b = 0
    ps.range.a = 5
    ps.offset.a = 0
    temp = picoiv(tri(V_set_stop, V_reset_stop), duration=duration, n=1, fs=fs,smartrange=False)
    ps.squeeze_range(temp,padpercent=0.5)
    if max_offsets[possible_ranges.index(ps.range.b)] < ps.offset.b:
        ps.range.b=5
        ps.offset.b=0
    
    if cycles_per_sweep/100 > 1:
        fs=int(fs*100/cycles_per_sweep)
        print('Downsizing sampling rate, due to memory. New sampling rate: {}'.format(fs))
    
    meta.meta['I_cc'] = I_CC
    meta.meta['V_set_stop'] = V_set_stop
    meta.meta['V_reset_stop'] = V_reset_stop
    meta.meta['duration'] = duration
    meta.meta['cycles'] = cycles
    meta.meta['cycles_per_sweep'] = cycles_per_sweep
    meta.meta['fs'] = fs
    meta.meta['slew_rate'] = (2*abs(V_reset_stop)+2*abs(V_set_stop))/duration
    meta.meta['tag'] = 'endurance'
    meta.meta['wfm'] = 'triangular'
    
    for i in trange(int(cycles/cycles_per_sweep)):
        iplots.clear()
        d=picoiv(tri(V_set_stop, V_reset_stop), duration=duration, n=cycles_per_sweep, fs=fs,smartrange=False)
        savedata(meta.attach(d),filepath = os.path.join(path, '{}_Icc_{}_V_reset_stop_{}_V_set_stop_{}_part_{}'.format(meta.filename(),I_CC,int(V_reset_stop*100),int(V_set_stop*100),i)))
    
    del meta.meta['I_cc']
    del meta.meta['V_set_stop']
    del meta.meta['V_reset_stop']
    del meta.meta['duration']
    del meta.meta['cycles']
    del meta.meta['cycles_per_sweep'] 
    del meta.meta['fs']
    del meta.meta['slew_rate']
    del meta.meta['tag']
    del meta.meta['wfm']
    tb.msg_to_nils("Endurance measurement: Done!")

 
def device_destroyer(I_CC, V_reset_stop, V_set_stop, measurement_cycles=1000, cycles_per_sweep=1e5, cycles_total=1e8, measurement_duration=1e-3, duration=1e-4, fs=1.25e6):
    path = os.path.join(datadir(),"Long_Endurance_V_set_stop{}_V_reset_stop_{}_Icc_{}_cycles_total_{}_{}".format(int(V_set_stop*100),int(V_reset_stop*100),int(I_CC),cycles_total,time.strftime('%d%m%y-%H%M%S')))
    if os.path.isfile(path):
        pass
    else:
        os.mkdir(path)
        
    path2 = os.path.join(path,'_low_resolution'.format())
    if os.path.isfile(path2):
        pass
    else:
        os.mkdir(path2)
        
    set_compliance(I_CC*1e-6)
    
    ps.range.b = 5
    ps.offset.b = 0
    ps.range.a = 5
    ps.offset.a = 0
    temp = picoiv(tri(V_set_stop, V_reset_stop), duration=duration, n=1, fs=fs,smartrange=False)
    ps.squeeze_range(temp,padpercent=0.5)
    if max_offsets[possible_ranges.index(ps.range.b)] < ps.offset.b:
        ps.range.b=5
        ps.offset.b=0
    
    #if cycles_per_sweep/100 > 1:
    #    fs=int(fs*100/cycles_per_sweep)
    #    print('Downsizing sampling rate, due to memory. New sampling rate: {}'.format(fs))
    
    packages_of_100 = measurement_cycles/100
    
    meta.meta['I_cc'] = I_CC
    meta.meta['V_set_stop'] = V_set_stop
    meta.meta['V_reset_stop'] = V_reset_stop
    meta.meta['duration'] = duration
    meta.meta['cycles_total'] = cycles_total
    meta.meta['cycles_per_sweep'] = cycles_per_sweep
    meta.meta['fs'] = fs
    meta.meta['slew_rate'] = (2*abs(V_reset_stop)+2*abs(V_set_stop))/duration
    meta.meta['measurement_cycles'] = measurement_cycles
    meta.meta['measurement_slew_rate'] = (2*abs(V_reset_stop)+2*abs(V_set_stop))/measurement_duration
    meta.meta['measurement_duration'] = measurement_duration
    meta.meta['measurement_fs'] = 1.25e9
    meta.meta['measurement_cycles_total'] = measurement_cycles
    meta.meta['measurement_cycles_per_package'] = 100
    meta.meta['tag'] = 'long_endurance'
    meta.meta['wfm'] = 'triangular'
    
    for i in tqdm(range(int(cycles_total//cycles_per_sweep))):
        
        d=measure.picoiv(tri(V_set_stop, V_reset_stop), duration=duration, n=cycles_per_sweep, fs=fs,smartrange=False)
        savedata(meta.attach(d),filepath = os.path.join(path2, '{}_cycles_{}'.format(meta.filename(),(i+1)*cycles_per_sweep).zfill(5)))
        
        iplots.clear()
        for j in range(int(packages_of_100)):
            d=picoiv(tri(V_set_stop, V_reset_stop), duration=measurement_duration, n=100, fs=1.25e9,smartrange=False)
            savedata(meta.attach(d),filepath = os.path.join(path, '{}_Measurement_ after_{}_cycles_part_{}'.format(meta.filename(),(i+1)*cycles_per_sweep, j)))
    
        if i%10 == 0:
            [R1,R2] = analyze.resistance_states_fit(d, v_low=0.1, v_high=0.3)
            tb.msg_to_nils("min HRS {} median HRS {} max HRS {} min LRS {} median LRS {} max LRS {}".format(np.min(R1),np.median(R1),np.max(R1),np.min(R2),np.median(R2),np.max(R2)))
        

    
    del meta.meta['I_cc']
    del meta.meta['V_set_stop']
    del meta.meta['V_reset_stop']
    del meta.meta['duration']
    del meta.meta['cycles_total']
    del meta.meta['cycles_per_sweep']
    del meta.meta['fs']
    del meta.meta['slew_rate']
    del meta.meta['measurement_slew_rate']
    del meta.meta['measurement_duration']
    del meta.meta['measurement_fs']
    del meta.meta['measurement_cycles']
    del meta.meta['tag']
    del meta.meta['wfm']
    del meta.meta['measurement_cycles_per_package']
    
    tb.msg_to_nils("Long Endurance measurement: Done!")
 
def measure_matrix(ICC_Start,ICC_Stop,ICC_inc,Vreset_Start,Vreset_Stop,Vreset_inc,Vset_stop, n_cycle=100, duration=1e-3,fs=1.25e9):
    '''
    Define Current Compliance Start, Stop and Increment in micro-A,
    Reset Voltage start, stop and increment (negative values *100),
    and regular picoiv-stuff.
    '''
    path = os.path.join(datadir(),"MatrixMeasurement_Vset{}_{}".format(int(Vset_stop*100),time.strftime('%d%m%y-%H%M%S')))
    if os.path.isfile(path):
        pass
    else:
        os.mkdir(path)
    appended_data_out=[]
    for I_CC in range(ICC_Start,ICC_Stop+ICC_inc,ICC_inc):
        set_compliance(I_CC*1e-6)
        print('-------------------------------------')
        print('Changed Current Compliance to {}'.format(I_CC))
        print('-------------------------------------')
        for V_RESET_STOP in range(Vreset_Start,Vreset_Stop+Vreset_inc,Vreset_inc):
            ps.range.b = 5
            ps.offset.b = 0
            ps.range.a = 5
            ps.offset.a = 0
            temp = picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=1, fs=fs,smartrange=False)
            ps.squeeze_range(temp,padpercent=0.5)
            if max_offsets[possible_ranges.index(ps.range.b)] < ps.offset.b:
                ps.range.b=5
                ps.offset.b=0
                temp = picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=1, fs=fs,smartrange=False)
                ps.squeeze_range(temp,padpercent=0.5)
            if max_offsets[possible_ranges.index(ps.range.b)] < ps.offset.b:
                ps.range.b=5
                ps.offset.b=0
            print('----------------------------------')
            print('Changed RESET Stop Voltage to {}'.format(V_RESET_STOP/100))
            print('----------------------------------')
            iplots.clear()
            d=picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=n_cycle, fs=fs,smartrange=False)
            #d_in_df=pd.DataFrame(d)
            
            
            meta.meta['wfm'] = 'triangular'
            meta.meta['tag'] = 'matrix'
            meta.meta['I_cc'] = I_CC
            meta.meta['V_reset_stop'] = V_RESET_STOP/100
            meta.meta['Vset_stop'] = Vset_stop
            meta.meta['cycles'] = n_cycle
            meta.meta['duration'] = duration
            meta.meta['fs'] = fs
            
            savedata(meta.attach(d),filepath = os.path.join(path, '{}_Icc_{}_Vreset_{}_Vset_{}'.format(meta.filename(),I_CC,V_RESET_STOP,int(Vset_stop*100))),attach=False)
            
            del meta.meta['wfm']
            del meta.meta['tag']
            del meta.meta['I_cc']
            del meta.meta['V_reset_stop']
            del meta.meta['Vset_stop']
            del meta.meta['cycles']
            del meta.meta['duration']
            del meta.meta['fs']

    
    for I_CC in reversed(range(ICC_Start,ICC_Stop+ICC_inc,ICC_inc)):
        set_compliance(I_CC*1e-6)
        print('-------------------------------------')
        print('Changed Current Compliance to {}'.format(I_CC))
        print('-------------------------------------')
        ps.range.b = 5
        ps.offset.b = 0
        ps.range.a = 5
        ps.offset.a = 0
        temp = picoiv(tri(Vset_stop, Vreset_Stop/100), duration=duration, n=1, fs=fs,smartrange=False)
        if max_offsets[possible_ranges.index(ps.range.b)] > ps.offset.b:
            temp = picoiv(tri(Vset_stop, Vreset_Stop/100), duration=duration, n=1, fs=fs,smartrange=False)
            ps.squeeze_range(temp,padpercent=0.5)
        if max_offsets[possible_ranges.index(ps.range.b)] > ps.offset.b:
            ps.range.b=5
            ps.offset.b=0
        iplots.clear()
        d=picoiv(tri(Vset_stop, Vreset_Stop/100), duration=duration, n=n_cycle, fs=fs,smartrange=False)
        #ivplot.interactive_figs.clear()
        
        meta.meta['wfm'] = 'triangular'
        meta.meta['tag'] = 'matrix'
        meta.meta['I_cc'] = I_CC
        meta.meta['V_reset_stop'] = Vreset_Stop/100
        meta.meta['Vset_stop'] = Vset_stop
        meta.meta['cycles'] = n_cycle
        meta.meta['duration'] = duration
        meta.meta['fs'] = fs
        
        savedata(d,filepath = os.path.join(path, '{}_Icc_{}_Vreset_{}_Vset_{}'.format(meta.filename(),I_CC,Vreset_Stop,int(Vset_stop*100))))
        
        del meta.meta['wfm']
        del meta.meta['tag']
        del meta.meta['I_cc']
        del meta.meta['V_reset_stop']
        del meta.meta['Vset_stop']
        del meta.meta['cycles']
        del meta.meta['duration']
        del meta.meta['fs']
            
    tb.msg_to_nils("Matrix measurement: Done!")
            
    #return(appended_data_out)

def vary_slew_rate(Icc, V_reset_stop, V_set_stop, Duration_start, Duration_stop, Duration_steps, cycles, sampling_rate, log=True):
    """
    Performs multiple measurements, where the duration of the pulses are varied. 
    """
    path = os.path.join(datadir(),"Slewrate_V_set_stop{}_V_reset_stop_{}_Icc_{}_{}".format(int(V_set_stop*100),int(V_reset_stop*100),int(Icc),time.strftime('%d%m%y-%H%M%S')))
    if os.path.isfile(path):
        pass
    else:
        os.mkdir(path)
    set_compliance(Icc*1e-6)
    
    v_range = (2*abs(V_reset_stop)+2*abs(V_set_stop))
    
    if log:
        durations = np.logspace(np.log10(Duration_start), np.log10(Duration_stop), Duration_steps)
    else:
        durations = np.linspace(Duration_start, Duration_stop, Duration_steps)
    
    for i in range(Duration_steps):
        duration = durations[i]
            
        fs, duration, n = calc_max_fs_d_n(fs=None, duration=duration, cycles=cycles)
        
        iplots.clear()
        print('Adjusting Picoscope range and offset:')
        temp = picoiv(tri(V_set_stop, V_reset_stop), duration=duration, n=1, fs=fs,smartrange=False)
        adjust_picorange(temp)
        
        meta.meta['I_cc'] = Icc
        meta.meta['V_set_stop'] = V_set_stop
        meta.meta['V_reset_stop'] = V_reset_stop
        meta.meta['duration'] = duration
        meta.meta['cycles'] = n
        meta.meta['fs'] = fs
        meta.meta['slew_rate'] = (2*abs(V_reset_stop)+2*abs(V_set_stop))/duration
        meta.meta['tag'] = 'slew_rate'
        meta.meta['wfm'] = 'triangular'
        
        print('V_set_stop = {}'.format(V_set_stop))
        print('V_reset_stop = {}'.format(V_reset_stop))
        print('duration = {}'.format(duration))
        print('n = {}'.format(n))
        print('fs = {}'.format(fs))
        print('Icc = {}'.format(Icc))
        print('slew_rate = {}'.format((2*abs(V_reset_stop)+2*abs(V_set_stop))/duration))
        
        print('Starting Measurement')
        d = picoiv(tri(V_set_stop, V_reset_stop), duration=duration, n=n, fs=fs,smartrange=False)
        savedata(meta.attach(d),filepath = os.path.join(path, '{}_Icc_{}_V_reset_stop_{}_V_set_stop_{}_part_{}'.format(meta.filename(),Icc,int(V_reset_stop*100),int(V_set_stop*100),i)))
        
        del meta.meta['I_cc']
        del meta.meta['V_set_stop']
        del meta.meta['V_reset_stop']
        del meta.meta['duration']
        del meta.meta['cycles']
        del meta.meta['fs']
        del meta.meta['slew_rate']
        del meta.meta['wfm']
        del meta.meta['tag']
    tb.msg_to_nils("Slewrate measurement: Done!")
        
    
    

def analyze_matrix(data,vlow_resistance=0.1,vhigh_resistance=0.3,v_lowSET=0.15,v_highSET=5,N_RESETstride=5,v_lowRESET=-5,v_highRESET=-0.1,SET_current_shift=3):
    '''
    Analyze a dataframe and write the data to an excel file so it can be later plotted in Origin or other programms
    TODO cycle duration column
    '''
    
    if type(data) != pd.core.frame.DataFrame:
        data = pd.DataFrame(data)
    
    ###first find the measurement circumstances: V_RESET_stop and I_CC
    #I_CC=pd.DataFrame(analyze.ICC_by_vmax(data, column='V', polarity=True))['I']
    I_CC=pd.DataFrame(data)['CC']
    print('I_CC done')
    #print(len(V_RESET_stop))
    
    ###find out resistances

    V_RESET_stop=pd.DataFrame({'V_RESET_stop':np.floor(10*analyze.V_RESET_stop_by_vmax(data, column='V', polarity=True)['V'])/10})
    print('V_RESET_stop done')
    HRS=pd.DataFrame({'HRS':analyze.resistance_states_fit(data,v_low=vlow_resistance,v_high=vhigh_resistance)[0]})
    print('HRS done')
    
    LRS=pd.DataFrame({'LRS':analyze.resistance_states_fit(data,v_low=vlow_resistance,v_high=vhigh_resistance)[1]})
    print('LRS done')
    # print((HRS))
    # print(len(LRS))
    # ###find out V_SET and V_RESET----OLD
    # V_SET=pd.DataFrame(analyze.thresholds_bydiff((analyze.splitbranch(data)),stride=stride_V_SET))['V']
    # print(len(V_SET))
    # V_SET=V_SET[V_SET <0] 
    # print(len(V_SET))
    # V_RESET=pd.DataFrame(analyze.Reset_by_imax(data, column='I', polarity=False))['V']
    # ###find out I_SET and I_RESET ---- OLD
    # I_SET=pd.DataFrame(analyze.thresholds_bydiff(data,stride=stride_V_SET))['I']
    # I_RESET=pd.DataFrame(analyze.Reset_by_imax(data, column='I', polarity=True))['I']
    
     
    ###find out V_SET and V_RESET, I_SET and I_RESET----NEW
    
    V_SET=pd.DataFrame({'V_SET':analyze.V_SET_by_max_gradient(data=data,v_low=v_lowSET,v_high=v_highSET)})
    print('V_SET done')
    V_RESET=pd.DataFrame({'V_RESET':analyze.V_RESET_by_change_of_gradient(data=data, N=N_RESETstride, v_low=v_lowRESET,v_high=v_highRESET)})
    #V_RESET=pd.DataFrame([0])
    print('V_RESET done')
    I_SET=pd.DataFrame({'I_SET':analyze.I_SET_by_max_gradient_shifted(data=data,v_low=v_lowSET,v_high=v_highSET,shift_index_by=SET_current_shift)})
    print('I_SET done')
    I_RESET=pd.DataFrame({'I_RESET':analyze.I_RESET_by_change_of_gradient(data=data, N=N_RESETstride, v_low=v_lowRESET,v_high=v_highRESET)})
    print('I_RESET done')
    
     
    # print(len(V_SET))
    # print(len(V_RESET))
    # print(len(I_SET))
    # print(len(I_RESET))
    
    
    analysis_data=pd.concat([I_CC, V_RESET_stop,HRS,LRS,V_SET,V_RESET,I_SET,I_RESET], axis=1)
    
    return(analysis_data)
    
def check_analysis_quality(analysis_data):
    badHRS=sum(float(analysis_data.iloc[i]['HRS']) <0 for i in analysis_data.index)
    print('There are {} cases of negative HRS in the analysis'.format(badHRS))
    badLRS=sum(float(analysis_data.iloc[i]['LRS']) <0 for i in analysis_data.index)
    print('There are {} cases of negative LRS in the analysis'.format(badLRS))
    badHRStoLRS=sum(float(analysis_data.iloc[i]['HRS']/analysis_data.iloc[i]['LRS']) <1 for i in analysis_data.index)
    print('There are {} cases of HRS smaller than LRS in the analysis'.format(badHRStoLRS))
    
    badV_SET=sum(float(analysis_data.iloc[i]['V_SET']) <0 for i in analysis_data.index)
    print('There are {} cases of negative V_SET in the analysis'.format(badV_SET))
    badV_RESET=sum(float(analysis_data.iloc[i]['V_RESET']) >0 for i in analysis_data.index)
    print('There are {} cases of positive V_RESET in the analysis'.format(badV_RESET))
    badV_RESET2=sum(float(analysis_data.iloc[i]['V_RESET']) ==0 for i in analysis_data.index)
    print('There are {} cases where V_RESET could not be found in the analysis'.format(badV_RESET2))
    badV_RESET_Currentdefinition=sum(float(analysis_data.iloc[i]['V_RESET']/analysis_data.iloc[i]['V_RESET_stop']) >=0.95 for i in analysis_data.index)
    print('There are {} cases of V_RESET equal to V_RESET_stop in the analysis'.format(badV_RESET_Currentdefinition))
    sumBadQuality = badHRS+badLRS+badHRStoLRS+badV_SET+badV_RESET+badV_RESET2+badV_RESET_Currentdefinition
    print('There are max. {}% bad data points'.format(sumBadQuality/len(analysis_data)*100))

def drop_bad(analysis_data):
    """
    Returns a DataFrame without data with bad quality, accroding to check_analysis_quality. 
    """
    #analysis_data.drop(analysis_data[analysis_data['HRS'] > 0].index, inplace = True) #would be in-place
    #analysis_data.drop(analysis_data[analysis_data['LRS'] > 0].index, inplace = True)
    #analysis_data.drop(analysis_data[analysis_data['HRS'] < analysis_data['LRS']].index, inplace = True)
    #analysis_data.drop(analysis_data[analysis_data['V_SET']<0].index, inplace = True)
    #analysis_data.drop(analysis_data[analysis_data['V_RESET']>0].index, inplace = True)
    #analysis_data.drop(analysis_data[analysis_data['V_RESET']==0].index, inplace = True)
    #analysis_data.drop(analysis_data[analysis_data['V_RESET']/analysis_data['V_RESET_stop']<0.95].index, inplace = True)
    return analysis_data[(analysis_data['HRS']>0) & (analysis_data['LRS'] > 0) & (analysis_data['HRS'] > analysis_data['LRS']) & (analysis_data['V_SET']>0) & (analysis_data['V_RESET']<0) & (analysis_data['V_RESET']!=0) & (analysis_data['V_RESET']/analysis_data['V_RESET_stop']<0.95)]
    
    
def save_matrix_analysis(analysis_data,filepath=None,drop=None):

    '''
    saves the analyzed data in the current working folder as dataframe and csv
    ''' 
    if filepath is None:
        filepath_df = os.path.join(datadir(), meta.filename()+'_matrix_analysis_df')
        filepath_csv=os.path.join(datadir(), meta.filename()+'_matrix_analysis_csv.csv')
    #print('Data to be saved in {}'.format(filepath_df))
    io.write_pandas_pickle(meta.attach(analysis_data), filepath_df, drop=drop)
    analysis_data.to_csv(path_or_buf=filepath_csv)
    #print('Wrote ',format(filepath_csv))
     
def plot_flat_matrix_analysis(analysis_data,saveoption='No',filepath='None'):

    flatfig1, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    #scatterargs.update(kwargs)
    ax.scatter(analysis_data.index,analysis_data['HRS'], c='royalblue', **scatterargs)
    ax.scatter(analysis_data.index,analysis_data['LRS'],  c='seagreen', **scatterargs)
    ax.legend(['HRS', 'LRS'], loc=0)
    engformatter('y', ax)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Resistance [$\\Omega$]')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e3, top=1e7)
    
    flatfig2, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    #scatterargs.update(kwargs)
    ax.scatter(analysis_data.index,analysis_data['V_SET'], c='red', **scatterargs)
    ax.scatter(analysis_data.index,analysis_data['V_RESET'],  c='blue', **scatterargs)
    ax.legend(['SET', 'RESET'], loc=0)
    engformatter('y', ax)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Switching voltages [V]')
    ax.set_yscale('linear')
    ax.set_ylim(bottom=-2, top=2)
    
    flatfig3, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    #scatterargs.update(kwargs)
    ax.scatter(analysis_data['V_SET'],abs(analysis_data['V_RESET']), c='cyan', **scatterargs)
    #ax.legend(['SET', 'RESET'], loc=0)
    #engformatter('y', ax)
    ax.set_xlabel('|SET Voltage| [V]')
    ax.set_ylabel('|RESET Voltage| [V]')
    #ax.set_yscale('linear')
    ax.set_xlim(left=0, right=2)
    ax.set_ylim(bottom=0, top=2)
    
    flatfig4, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    #scatterargs.update(kwargs)
    ax.scatter(analysis_data['V_SET'],analysis_data['HRS'], c='red', **scatterargs)
    #ax.legend(['SET', 'RESET'], loc=0)
    #engformatter('y', ax)
    ax.set_xlabel('SET Voltage [V]')
    ax.set_ylabel('HRS [$\\Omega$]')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e3, top=1e7)
    ax.set_xlim(left=0, right=2)
    
    
    flatfig5, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    #scatterargs.update(kwargs)
    ax.scatter(abs(analysis_data['V_RESET']),analysis_data['LRS'], c='blue', **scatterargs)
    #ax.legend(['SET', 'RESET'], loc=0)
    #engformatter('y', ax)
    ax.set_xlabel('|RESET Voltage| [V]')
    ax.set_ylabel('LRS [$\\Omega$]')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e3, top=2e4)
    ax.set_xlim(left=0, right=2)
    
   
    flatfig6, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    #scatterargs.update(kwargs)
    for _, val in analysis_data.groupby('CC'):
        ax.scatter(val['CC'].iloc[0]*1e6,np.mean(val['HRS']), c='red', **scatterargs)
        ax.scatter(val['CC'].iloc[0]*1e6,np.mean(val['LRS']),  c='blue', **scatterargs)
    ax.legend(['HRS', 'LRS'], loc=0)
    engformatter('y', ax)
    ax.set_xlabel('CC [$\\mu$A]')
    ax.set_ylabel('Resistance [$\\Omega$]')
    ax.set_yscale('log', nonposy='clip')
    ax.set_ylim(bottom=1e3, top=1e7)
    plt.show()
    
    if saveoption=='Yes':
        if filepath=='None':
            filepath_savefig = os.path.join(datadir(), meta.filename()+'_matrix_analysis_flatplots_')
        else:
            filepath_savefig=filepath
        flatfig1.savefig(filepath_savefig+'HRS_LRS_vs_cycle')
        flatfig2.savefig(filepath_savefig+'V_sw_vs_cycle')
        flatfig3.savefig(filepath_savefig+'V_RESET_vs_V_SET')
        flatfig4.savefig(filepath_savefig+'HRS_vs_V_SET')
        flatfig5.savefig(filepath_savefig+'LRS_vs_V_RESET')
        
def combine_df(path):
    df = pd.dataframe
    for item in glob.glob(path):
        df.concat(pd.read_pickle(item))
    return df

###Dont use: 
def plot_colored_matrix_analysis(analysis_data,cmap='jet',saveoption='No',filepath='None'):
    unique_ICC=np.unique(np.around(analysis_data['CC'],4))  
    Icc_map = cm.get_cmap(cmap, len(unique_ICC))
    I_cc_colors = [Icc_map(c) for c in np.linspace(0, 1, len(unique_ICC))]
    
    unique_V_RESET_stop=np.unique(analysis_data['V_RESET_stop'])
    Vreset_stop_map = cm.get_cmap(cmap, len(unique_V_RESET_stop))
    Vreset_stop_colors = [Vreset_stop_map(c) for c in np.linspace(0, 1, len(unique_V_RESET_stop))]
    
    cfig1, ax = plt.subplots()
    scatterargs = dict(s=10, alpha=.8, edgecolor='none')
    #scatterargs.update(kwargs)
    #ax.scatter(analysis_data.index,analysis_data['LRS'],  c='seagreen', **scatterargs)
    #ax.legend(['HRS', 'LRS'], loc=0)
    engformatter('y', ax)
    ax.set_xlabel('V_RESET #')
    ax.set_ylabel('Resistance [$\\Omega$]')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e3, top=1e7)
    
    for element in itertools.product(unique_V_RESET_stop, unique_ICC):
    
        mask = (analysis_data['V_RESET_stop']==element[0]) & (analysis_data['CC']==element[1])
        ax.scatter(analysis_data.loc[mask]['V_RESET'],analysis_data['LRS'].loc[mask], c='blue', **scatterargs)
    
    


def split_data_into_matrix(data,doplot=True,cmap='jet',saveplots=False,analyse=True,vlow_resistance=0.1,vhigh_resistance=0.3,v_lowSET=0.15,v_highSET=5,N_RESETstride=5,v_lowRESET=-5,v_highRESET=-0.3,SET_current_shift=3):
    unique_ICC=np.unique(np.around(data['CC'],4))
    print(unique_ICC)
    unique_V_RESET_stop=np.unique(np.floor(10*analyze.V_RESET_stop_by_vmax(data, column='V', polarity=True)['V'])/10)
    print(unique_V_RESET_stop)
    Iccs = unique_ICC
    Vresets = unique_V_RESET_stop
    Icc_map = cm.get_cmap(cmap, len(unique_ICC))
    I_cc_colors = [Icc_map(c) for c in np.linspace(0, 1, len(unique_ICC))]
    Vreset_stop_map = cm.get_cmap(cmap, len(unique_V_RESET_stop))
    Vreset_stop_colors = [Vreset_stop_map(c) for c in np.linspace(0, 1, len(unique_V_RESET_stop))]
    combi_cmap=cm.get_cmap(cmap, len(unique_V_RESET_stop)*len(unique_ICC))
    combi_colors = [combi_cmap(c) for c in np.linspace(0, 1, len(unique_V_RESET_stop)*len(unique_ICC))]
    if doplot==True:
        scatterargs = dict(s=10, alpha=.8, edgecolor='none')
        errorbarargs=dict(alpha=.8)
        lineargs=dict(alpha=.5)
        figIcc, axIcc = plt.subplots()##figIcc sind Sweeps mit constanter CC
        axIcc.set_xlabel('Voltage [V]')
        axIcc.set_ylabel('Current [A]')
        axIcc.set_yscale('log')
        axIcc.set_ylim(bottom=1e-9, top=5e-3)
        axIcc.set_xlim(left=-2, right=2.5)
        legendIcc_handles=[]
        legendIcc_labels=[]
        
        figVres, axVres = plt.subplots()##figVres sind Sweeps mit constanter Vresetstop
        axVres.set_xlabel('Voltage [V]')
        axVres.set_ylabel('Current [A]')
        axVres.set_yscale('log')
        axVres.set_ylim(bottom=1e-9, top=5e-3)
        axVres.set_xlim(left=-2, right=2.5)
        legendVres_handles=[]
        legendVres_labels=[]
        
        fig1, ax1 = plt.subplots()##fig1 ist V_SET vs HRS, mit Farbe durch reset stop
        ax1.set_xlabel('SET Voltage [V]')
        ax1.set_ylabel('HRS [$\\Omega$]')
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1e3, top=1e7)
        ax1.set_xlim(left=0, right=2)
        legend1_handles=[]
        legend1_labels=[]
        fig1med, ax1med = plt.subplots()##fig1 ist V_SET vs HRS, mit Farbe durch reset stop; nur MEDIANE
        ax1med.set_xlabel('SET Voltage [V]')
        ax1med.set_ylabel('HRS [$\\Omega$]')
        ax1med.set_yscale('log')
        ax1med.set_ylim(bottom=1e3, top=1e7)
        ax1med.set_xlim(left=0, right=2)
        legend1med_handles=[]
        legend1med_labels=[]
        
        
        fig2, ax2 = plt.subplots()##fig2 ist V_RESET vs LRS, mit Farbe durch ICC
        ax2.set_xlabel('|RESET Voltage| [V]')
        ax2.set_ylabel('LRS [$\\Omega$]')
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=1e3, top=2e4)
        ax2.set_xlim(left=0, right=2)
        legend2_handles=[]
        legend2_labels=[]
        fig2med, ax2med = plt.subplots()##fig1 ist V_SET vs HRS, mit Farbe durch reset stop; nur MEDIANE
        ax2med.set_xlabel('|RESET Voltage| [V]')
        ax2med.set_ylabel('LRS [$\\Omega$]')
        ax2med.set_yscale('log')
        ax2med.set_ylim(bottom=1e3, top=2e4)
        ax2med.set_xlim(left=0, right=2)
        legend2med_handles=[]
        legend2med_labels=[]
        
        count_combinations=-1
    for i,Icc in enumerate(Iccs[::1]):
        if doplot==True:
             figIcc, axIcc = plt.subplots()##figIcc sind Sweeps mit constanter CC
             axIcc.set_xlabel('Voltage [V]')
             axIcc.set_ylabel('Current [A]')
             axIcc.set_yscale('log')
             axIcc.set_ylim(bottom=1e-9, top=5e-3)
             axIcc.set_xlim(left=-2, right=2.5)
             legendIcc_handles=[]
             legendIcc_labels=[]
            #pass
        #print(Icc)
        for j,Vreset in enumerate(Vresets[::-1]):
            count_combinations=count_combinations+1
            print(count_combinations)
            #print(Vreset)
            mask = (np.around(data['I'].apply(max),4)==Icc) & (np.floor(10*data['V'].apply(min))/10==Vreset)
            current_df= data.loc[mask]
            if analyse==True:
                plot_analysis=True
                current_ana=analyze_matrix(current_df,vlow_resistance,vhigh_resistance,v_lowSET,v_highSET,N_RESETstride,v_lowRESET,v_highRESET,SET_current_shift)
                combined_stats=pd.concat([current_ana.describe(),pd.DataFrame(current_ana.median(), columns = ["median"] ).T,pd.DataFrame(current_ana.mad(), columns = ["mean_ad"] ).T,pd.DataFrame(abs(current_ana-current_ana.median()).median(),columns=["median_ad"]).T])
            else:
                plot_analysis=False
            if doplot==True:
                #plot.plotiv(current_df)
                if plot_analysis==True:
                    scatter1=ax1.scatter(current_ana['V_SET'],current_ana['HRS'], c=[Vreset_stop_colors[j]], **scatterargs)
                    legend1_handles.append(scatter1)
                    legend1_labels.append(str(Vreset))  
                    scatter1med=ax1med.errorbar(x=combined_stats.loc['median']['V_SET'],y=combined_stats.loc['median']['HRS'],yerr=combined_stats.loc['median_ad']['HRS'],xerr=combined_stats.loc['median_ad']['V_SET'],color=combi_colors[count_combinations])
                    legend1med_handles.append(scatter1med)
                    legend1med_labels.append(str(Vreset)+' V| '+str(Icc)+' A')  
                
                    scatter2=ax2.scatter(abs(current_ana['V_RESET']),current_ana['LRS'], c=[I_cc_colors[i]], **scatterargs)
                    legend2_handles.append(scatter2)
                    legend2_labels.append(str(Icc))
                    scatter2med=ax2med.errorbar(x=abs(combined_stats.loc['median']['V_RESET']),y=combined_stats.loc['median']['LRS'],yerr=combined_stats.loc['median_ad']['LRS'],xerr=combined_stats.loc['median_ad']['V_RESET'],color=combi_colors[count_combinations])
                    legend2med_handles.append(scatter2med)
                    legend2med_labels.append(str(Vreset)+' V| '+str(Icc)+' A')  
            for kkk,lelelel in enumerate(current_df.index):
                #print(kkk)
                
                lineIcc,=axIcc.plot(current_df.iloc[kkk]['V'],abs(current_df.iloc[kkk]['I']),c=Vreset_stop_colors[j],**lineargs)
                legendIcc_handles.append(lineIcc)
                legendIcc_labels.append(str(Vreset)) 
            #axIcc.legend(legendIcc_handles[::np.ceil(len(current_df['V'])/len(unique_V_RESET_stop))],legendIcc_labels[::np.ceil(len(current_df['V'])/len(unique_V_RESET_stop))])
            
    
    
    if doplot==True:
        ax1.legend(legend1_handles[0:len(unique_V_RESET_stop)],legend1_labels[0:len(unique_V_RESET_stop)])
        ax1med.legend(legend1med_handles,legend1med_labels)
        ax2.legend(legend2_handles[::len(unique_ICC)],legend2_labels[::len(unique_ICC)])
        ax2med.legend(legend2med_handles,legend2med_labels)
        
    return(current_df)
    
    
    
    