import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#import itertools
import matplotlib as mpl
import time as time
import Telegram_bot as tb
import scipy as sc
import traceback as traceback

pd.options.mode.chained_assignment = None

mpl.rcParams['agg.path.chunksize'] = 10000

def adjust_picorange(sweep_data):
    """
    Sets the best possible range at a given offset of a sweep.
    """
    possible_ranges = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    max_offsets =     [.5,    .5,  .5, 2.5, 2.5, 2.5, 20,    20, 20]
    
    ps.squeeze_range(sweep_data,padpercent=0.5)
    if max_offsets[possible_ranges.index(ps.range.b)] < abs(ps.offset.b):
        range_index = np.where(max_offsets>=abs(ps.offset.b))[0][0]
        ps.range.b=possible_ranges[range_index]
    
def read_resistance(V_read=0.3,t_read=1e-3):
    read_signal=measure.tri(V_read,-V_read)
    default_rng()
    read=picoiv(read_signal,duration=t_read, n=1, fs=1.25e9,smartrange=False)
    #print('\nFirst read: {}'.format(analyze.resistance_states_fit(read,v_low=0.01,v_high=V_read)))
    adjust_picorange(read)
    iplots.clear()
    read=picoiv(read_signal,duration=t_read, n=1, fs=1.25e9,smartrange=False)
    [R1,R2]=analyze.resistance_states_fit(read,v_low=0.01,v_high=V_read)#-0.05)
    print('\nResistances were found to be {} and {}\n'.format(R1, R2))
    R_mean=(R1+R2)/2
    return(R_mean)
    
def save_resistance(real_resistance, V_read=0.3, t_read=1e-3, fs=1.25e9, cycles=100):
    
    filepath = os.path.join(datadir(),'{}_Resistance_Check_{}_Ohm'.format(time.strftime('%d%m%y-%H%M%S'),real_resistance))
    
    if os.path.isfile(filepath):
        pass
    else:
        os.mkdir(filepath)

    read_signal=measure.tri(V_read,-V_read)
    
    for j in range(100,800,100):

        set_compliance(j*1e-6)
        #time.sleep(0.5)
        default_rng()
        read=picoiv(read_signal,duration=t_read, n=1, fs=1.25e9,smartrange=False)
        adjust_picorange(read)
        
        [R1,R2]=analyze.resistance_states_fit(read,v_low=0.01,v_high=V_read)
        print('First read: ',R1, R2, ps.range.b, ps.offset.b)
        
        for i in range(int(cycles)):
            iplots.clear()
            
            meta.meta['offset_b'] = ps.offset.b
            meta.meta['range_b'] = ps.range.b
            meta.meta['actual_resistance'] = real_resistance
            meta.meta['I_cc'] = j
            meta.meta['V_read'] = V_read
            meta.meta['duration'] = t_read
            meta.meta['cycles'] = cycles
            meta.meta['fs'] = fs
            meta.meta['tag'] = 'resistance_accuracy'
            meta.meta['wfm'] = 'triangular'
            #time.sleep(0.5)
            read=picoiv(read_signal,duration=t_read, n=1, fs=1.25e9,smartrange=False)
            [R1,R2]=analyze.resistance_states_fit(read,v_low=0.01,v_high=V_read)
            
            meta.meta['read_1'] = R1
            meta.meta['read_2'] = R2

            savedata(meta.attach(read),filepath = os.path.join(filepath, '{}_Icc_{}'.format(meta.filename(),j)))
            
            print(R1, R2, ps.range.b, ps.offset.b)
            
            del meta.meta['offset_b']
            del meta.meta['range_b']
            del meta.meta['I_cc']
            del meta.meta['V_read']
            del meta.meta['duration']
            del meta.meta['cycles']
            del meta.meta['fs']
            del meta.meta['tag']
            del meta.meta['wfm']
            del meta.meta['read_1']
            del meta.meta['read_2']
            del meta.meta['actual_resistance']

    
def bring_device_to_HRS(HRS_min,HRS_max,filepath, V_RESET_max=-1.8,t_RESET=1e-3,V_read=0.3,t_read=1e-3,V_RESET_start=-0.4, V_RESET_step=0.05, max_iterations=100):
    iterations = 0
    max_iterations = abs((V_RESET_start-V_RESET_max)/V_RESET_step) + max_iterations
    V_RESET=V_RESET_start
    current_res=read_resistance(V_read=V_read,t_read=t_read)
    print('---Resistance is {}---'.format(int(current_res)))
    if current_res<HRS_min or current_res>HRS_max:
        resistance_correct=False
        print('---Start RESET stair---')
        RESET_tracer=[[0,current_res]]
   
        while resistance_correct==False and iterations<=max_iterations and V_RESET>=V_RESET_max:
            #print('Loop 1 resisatnce_correct {} iterations {} maxiterations {} V_reset {} V_resetmax{}'.format(resistance_correct,iterations,max_iterations,V_RESET,V_RESET_max))
            #if current_res>HRS_max:
            if current_res>HRS_max or iterations == max_iterations-1 or V_RESET-V_RESET_step<=V_RESET_max:
                print('---Do one SET---')
                iterations += 1
                
                meta.meta['tag'] = 'bring_device_to_HRS: Set'
                meta.meta['HRS_min'] = HRS_min
                meta.meta['HRS_max'] = HRS_max
                meta.meta['V_RESET_max'] = V_RESET_max
                meta.meta['duration'] = t_RESET
                meta.meta['V_read'] = V_read
                meta.meta['t_read'] = t_read
                meta.meta['V_RESET_start'] = V_RESET_start
                meta.meta['V_RESET_step'] = V_RESET_step
                meta.meta['res_before_pulse'] = current_res
                meta.meta['wfm'] = 'triangular'
                meta.meta['V_set_stop'] = 3
                meta.meta['V_reset_stop'] = -0.2
                meta.meta['fs'] = 1.25e9
                
                default_rng()
                
                d=picoiv(tri(3,-0.2),duration=t_RESET,fs=1.25e9,n=1)
                #meta.attach(pd.DataFrame(d)).to_pickle(os.path.join(filepath,'{}_Reset_Stair_Pulses_Set.df'.format(iterations)))
                savedata(meta.attach(d),filepath=os.path.join(filepath,'{}_Reset_Stair_Pulses_SETpulse_{}'.format(time.strftime('%d%m%y-%H%M%S'),iterations)), attach=False, read_only=False)
                
                #RESET_cyles += d
                current_res=read_resistance(V_read=V_read,t_read=t_read)  
                print('---Resistance is {}---'.format(int(current_res)))
                newentry=[[V_RESET,current_res]]
                RESET_tracer=np.concatenate((RESET_tracer,newentry))#log the V_RESET vs resistance curve
                if current_res>HRS_min and current_res<HRS_max:
                    resistance_correct=True
                    break
                    print('---Resistance is {} and therefore correct.---'.format(int(current_res)))        
                
                del meta.meta['tag']
                del meta.meta['HRS_min']
                del meta.meta['HRS_max']
                del meta.meta['V_RESET_max']
                del meta.meta['duration']
                del meta.meta['V_read']
                del meta.meta['t_read']
                del meta.meta['V_RESET_start']
                del meta.meta['V_RESET_step']
                del meta.meta['res_before_pulse']
                del meta.meta['wfm']
                del meta.meta['V_set_stop']
                del meta.meta['V_reset_stop']
                del meta.meta['fs']

                
            while current_res<HRS_min and V_RESET>=V_RESET_max and resistance_correct==False:
                #print('Loop 2 current_res {} HRS_min {} V_RESET {} V_RESET_max {} resistance_correct{}'.format(current_res,HRS_min,V_RESET,V_RESET_max,resistance_correct))
                print('---Do RESET:{}---'.format(V_RESET))
                iterations += 1

                meta.meta['tag'] = 'bring_device_to_HRS: Reset'
                meta.meta['HRS_min'] = HRS_min
                meta.meta['HRS_max'] = HRS_max
                meta.meta['V_RESET_max'] = V_RESET_max
                meta.meta['duration'] = t_RESET
                meta.meta['V_read'] = V_read
                meta.meta['t_read'] = t_read
                meta.meta['V_RESET_start'] = V_RESET_start
                meta.meta['V_RESET_step'] = V_RESET_step
                meta.meta['res_before_pulse'] = current_res
                meta.meta['wfm'] = 'triangular'
                meta.meta['V_set_stop'] = 0.1
                meta.meta['V_reset_stop'] = V_RESET
                meta.meta['fs'] = 1.25e9
                
                
                default_rng()
                d=picoiv(tri(0.1,V_RESET),duration=t_RESET,fs=1.25e9,n=1)
                #meta.attach(d).to_pickle(os.path.join(filepath,'{}_Reset_Stair_Pulses_Reset.df'.format(iterations)))
                savedata(meta.attach(d),filepath=os.path.join(filepath,'{}_Reset_Stair_Pulses_RESETpulse_{}'.format(time.strftime('%d%m%y-%H%M%S'),iterations)), attach=False, read_only=False)
                
                del meta.meta['tag']
                del meta.meta['HRS_min']
                del meta.meta['HRS_max']
                del meta.meta['V_RESET_max']
                del meta.meta['duration']
                del meta.meta['V_read']
                del meta.meta['t_read']
                del meta.meta['V_RESET_start']
                del meta.meta['V_RESET_step']
                del meta.meta['res_before_pulse']
                del meta.meta['wfm']
                del meta.meta['V_set_stop']
                del meta.meta['V_reset_stop']
                del meta.meta['fs']
                
                #RESET_cyles += d
                current_res=read_resistance(V_read=V_read,t_read=t_read)  
                print('---Resistance is {}---'.format(int(current_res)))
                newentry=[[V_RESET,current_res]]
                RESET_tracer=np.concatenate((RESET_tracer,newentry))#log the V_RESET vs resistance curve
                if current_res>HRS_min and current_res<HRS_max:
                    resistance_correct=True
                    break
                    print('---Resistance is {} and therefore correct.---'.format(int(current_res)))          
                elif current_res<HRS_min:
                    V_RESET=V_RESET-V_RESET_step###increase RESET voltage    
                    
                    
        if iterations>=max_iterations:
            print('Maximal amount of iterations reached!')
            return RESET_tracer, current_res, False
        elif V_RESET<=V_RESET_max:
            print('Resistance window was not reached!')
            return RESET_tracer, current_res, False
        else:
            print('\n---Resistance found {}---\n'.format(current_res))
            return RESET_tracer, current_res, True
    else:
        print('\n---Resistance already was correct.---\n')          
        RESET_tracer=np.array([[0,current_res]])
        return RESET_tracer, current_res, True
        
def save_RESET_stair(RESET_stair_data,filepath=None,drop=None):
    RESET_stair_data=pd.DataFrame(RESET_stair_data,columns={'V_RESET','R_read'})
    if filepath is None:
        filepath_df = os.path.join(datadir(), meta.filename()+'_RESET_stair_df')
        filepath_csv=os.path.join(datadir(), meta.filename()+'_RESET_stair_csv.csv')
    else:
        filepath_df = os.path.join(filepath, meta.filename()+'_RESET_stair_df')
        filepath_csv=os.path.join(filepath, meta.filename()+'_RESET_stair_csv.csv')
        #print('Data to be saved in {}'.format(filepath_df))
    io.write_pandas_pickle(meta.attach(RESET_stair_data), filepath_df, drop=drop)
    RESET_stair_data.to_csv(path_or_buf=filepath_csv)
    #print('Wrote ',format(filepath_csv))


def rem_over_sampling(data, factor, columns=['I','V']):
    """
    Downsample the signal after applying an anti-aliasing filter.
    """
    decarrays = [sc.signal.decimate(x=data[key], q=factor, zero_phase=True) for key in columns]
    dataout = {c:dec for c,dec in zip(columns, decarrays)}
    analyze.add_missing_keys(data, dataout)
    if 'downsampling' in dataout:
        dataout['downsampling'] *= factor
    else:
        dataout['downsampling'] = factor
    return dataout

def _calc_t_set(data):
    #mask_I, pos1_I, pos2_I, half_I = find_I_set(data)
    #mas_V, pos1_V, pos2_V, pos3_V, half_V = V_pulse_start(data)
    max_V = max(data['V'])
    max_I = max(data['I_corr'])
    
    half_I = np.where(data['I_corr']>0.5*max_I)[0][0]
    half_V = np.where(data['V']>0.5*max_V)[0][0]

    a_I = (data['I_corr'][half_I]-data['I_corr'][half_I-1])/(data['t'][half_I]-data['t'][half_I-1])
    b_I = data['I_corr'][half_I]-a_I*data['t'][half_I]
    
    a_V = (data['V'][half_V]-data['V'][half_V-1])/(data['t'][half_V]-data['t'][half_V-1])
    b_V = data['V'][half_V]-a_V*data['t'][half_V]
    
    t_I = (0.5*max_I-b_I)/a_I
    t_V = (0.5*max_V-b_V)/a_V
    
    return t_V, t_I, t_I-t_V
        
def _calc_t_trans(data,trans_low=0.2,trans_high=0.8):
    #max_V = max(data['V'])
    max_I = max(data['I_corr'])
    eighty_I = np.where(data['I_corr']>trans_high*max_I)[0][0]
    #eighty_V = np.where(data['V']>trans_high*max_V)[0][0]
    twenty_I = np.where(data['I_corr']>trans_low*max_I)[0][0]
    #twenty_V = np.where(data['V']>trans_low*max_V)[0][0]
    
    return data['t'][twenty_I], data['t'][eighty_I], data['t'][eighty_I]-data['t'][twenty_I], twenty_I, eighty_I

def analysis_Nils(SET_pulse, t_pulse,V_SET, resistance_pre,resistance_after,trans_low=0.2,trans_high=0.8, df=None):
    
    if df is None:
        columns = ['t_set','t_trans','SET','t_V','t_I','t_trans_low','t_trans_high','I_preset','t_pulse']
        tmp = pd.DataFrame(columns = columns)
    else:
        tmp = df
    set_condition = 1e-4
    
    if max(SET_pulse['I_corr']) > set_condition:
        #print('If the current is smaller than {} A the data will be acted as if no Set occured'.format(set_condition))

        factor = len(SET_pulse['V'])/t_pulse/SET_pulse['sample_rate']
        factor = np.round(factor,0)
        factor = int(factor)
        x = rem_over_sampling(SET_pulse, factor, columns=['I','V'])
        x['t'] = analyze.maketimearray(x, ignore_downsampling=True)
        
        mask_0V=abs(x['V'])<0.01
        current_offset=np.mean(x['I'][mask_0V])
        x['I_corr']=x['I']-current_offset            
        
        """
        duty = SET_pulse.duty
        pulse_length = SET_pulse.t_pulse

        mask = (x['t'] > (duty-duty*.5-0.03)*pulse_length)*(x['t'] < (duty+duty*.5+0.03)*pulse_length)
        """
        
        t_V, t_I, t_set = _calc_t_set(x)
        t_trans_low, t_trans_high, t_trans, twenty_I, eighty_I = _calc_t_trans(data=x,trans_low=trans_low,trans_high=trans_high)
        
        mask_over0V=x['V']>0.5*np.max(x['V'])
        if twenty_I>np.argmax(mask_over0V==True):
            I_preset=np.mean(x['I_corr'][np.argmax(mask_over0V==True):twenty_I]) #wenn der anstieg des pulses mit einbezogen wird, wird der wert nach oben gezogen
        else:
            print(twenty_I,np.argmax(mask_over0V==True))
            I_preset=np.nan
        
        tmp = tmp.append({'t_set':t_set,'t_trans':t_trans,'SET':True,'t_V':t_V,'t_I':t_I,'t_trans_low':t_trans_low,'t_trans_high':t_trans_high,'I_preset':I_preset,'t_pulse':t_pulse,'V_SET':V_SET,'resistance_pre':resistance_pre,'resistance_after':resistance_after},ignore_index=True)

            
        return tmp
    else:
        my_dict = {'t_set':np.nan,'t_trans':np.nan,'SET':False,'t_V':np.nan,'t_I':np.nan,'t_trans_low':np.nan,'t_trans_high':np.nan,'I_preset':np.nan,'V_SET':V_SET,'resistance_pre':resistance_pre,'resistance_after':resistance_after,'t_pulse':t_pulse}
        tmp = tmp.append(my_dict,ignore_index=True)
        return tmp

def analyze_SET_transient(SET_pulse,trans_low=0.2,trans_high=0.8):
    if np.max(SET_pulse['I_corr'])>1e-4:
        try:
            mask_over0V=SET_pulse['V']>0.5*np.max(SET_pulse['V'])
            i_puls_begin=np.argmax(mask_over0V==True)
            i_set = np.argmax(np.gradient(SET_pulse['I_corr'][mask_over0V]))+i_puls_begin #ist das maximum des Gradienten wirklich der beginn des pulses?
        
            t_SET=SET_pulse['t'][i_set]-SET_pulse['t'][i_puls_begin]
        
            i_trans_begin=np.argwhere(SET_pulse['I_corr'][mask_over0V]>trans_low*(np.max(SET_pulse['I_corr'][mask_over0V])-np.mean(SET_pulse['I_corr'][i_puls_begin:i_set]))) #wieso hier den mittelwert abziehen?
            i_trans_end=np.argwhere(SET_pulse['I_corr'][mask_over0V]>trans_high*(np.max(SET_pulse['I_corr'][mask_over0V])-np.mean(SET_pulse['I_corr'][i_puls_begin:i_set])))
            t_trans=SET_pulse['t'][i_trans_begin]-SET_pulse['t'][i_trans_end]
            
            if i_trans_begin>i_puls_begin:
                I_preset=np.mean(SET_pulse['I_corr'][i_puls_begin:i_trans_begin]) #wenn der anstieg des pulses mit einbezogen wird, wird der wert nach oben gezogen
            else:
                I_preset=0
            return t_SET,t_trans,I_preset, SET_pulse['t'][i_trans_begin],SET_pulse['t'][i_trans_end], SET_pulse['t'][i_set], SET_pulse['t'][i_puls_begin]
        except: np.nan,np.nan,np.nan, np.nan, np.nan, np.nan, np.nan
            
    else:
        return np.nan,np.nan,np.nan, np.nan, np.nan, np.nan, np.nan
    

def repeat_SET_pulse_from_HRS(V_SET,n_SET,t_pulse,HRS_min,HRS_max,V_read=0.3,t_read=1e-3,V_RESET_max=-1.8,t_RESET=1e-3,V_RESET_start=-0.4, V_RESET_step=0.05, max_iterations=20):
    

    filepath = os.path.join(datadir(),'RepeatedPulsing_{}_HRSmin_{}_HRSmax_{}_Vset_{}'.format(time.strftime('%d%m%y-%H%M%S'),HRS_min,HRS_max,round(V_SET,3)))
    
    if os.path.isfile(filepath):
        pass
    else:
        os.mkdir(filepath)
    
    duty = 0.5
    SET_data=pd.DataFrame(columns={'V_SET','t_pulse','R_pre','R_post','t_SET','t_trans','I_preset'})
    pulse_signal=measure.square(vpulse=V_SET, duty=duty, length=16000, startval=0, endval=0, startendratio=1)
    #for i in range(0,n_SET,1):
    
    
    i = 0
    iterator = 0
    Done = False
    
    tmp = pd.DataFrame() #needed for the analysis_Nils func
    
    while (Done == False) and (i <= max_iterations):
        print('Starting cycle {} of minimum {} and maximum {}'.format(int(i),int(n_SET),max_iterations))
        ##first reset the device to the desired HRS and save the staircase data
        RESET_stair, resistance_pre, is_in_HRS_window=bring_device_to_HRS(HRS_min=HRS_min,filepath=filepath,HRS_max=HRS_max,V_RESET_max=V_RESET_max,t_RESET=t_RESET,V_read=V_read,t_read=t_read,V_RESET_start=V_RESET_start,V_RESET_step=V_RESET_step,max_iterations=max_iterations)
        iplots.clear()
        save_RESET_stair(RESET_stair, filepath=filepath)
        if not is_in_HRS_window:
            return None, None, None #in case it was not possible to bring the device to the desired HRS window
            
        ##then apply read-SET-read
        measure.set_compliance(7e-4)
        #resistance_pre=read_resistance(V_read=V_read,t_read=t_read)
        
        default_rng()
        measure.smart_range(-0.1,V_SET,ch=['A'])
        
        #ps.range.b = 1. #ergibt es Sinn die range auf 5 zu setzen?? passiert in default range -> vllt. besser 1, aufgrund des offsets
        
        meta.meta['V_pulse'] = V_SET
        meta.meta['t_pulse'] = t_pulse
        meta.meta['duty'] = duty
        meta.meta['tag'] = 'repeat_SET_pulse_from_HRS'
        meta.meta['HRS_min'] = HRS_min
        meta.meta['HRS_max'] = HRS_max
        meta.meta['V_RESET_max'] = V_RESET_max
        meta.meta['duration'] = t_RESET
        meta.meta['V_read'] = V_read
        meta.meta['t_read'] = t_read
        meta.meta['V_RESET_start'] = V_RESET_start
        meta.meta['V_RESET_step'] = V_RESET_step
        meta.meta['current_res'] = resistance_pre
        meta.meta['resistance_pre'] = resistance_pre
        meta.meta['wfm'] = 'square'
        meta.meta['fs'] = 1.25e9
        
        
        
        #Do SET operation
        SET=picoiv(wfm=pulse_signal,n=1,fs=1.25e9,duration=(1/duty)*t_pulse,autosmoothimate=False)
        
        factor = len(SET['V'])/t_pulse/SET['sample_rate']
        factor = int(np.round(factor,0))
        #factor = int(factor)
        SET = rem_over_sampling(SET, factor, columns=['I','V'])
        SET['t'] = analyze.maketimearray(SET, ignore_downsampling=True)
        
        
        mask_0V=abs(SET['V'])<0.01
        current_offset=np.mean(SET['I'][mask_0V])
        SET['I_corr']=SET['I']-current_offset
        resistance_after=read_resistance(V_read=V_read,t_read=t_read)
        
        meta.meta['resistance_after'] = resistance_after
        
        try:
            t_SET,t_trans,I_preset, t_trans_begin, t_trans_end, t_pulse_begin_I, t_pulse_begin_V = analyze_SET_transient(SET)
        except:
            t_SET,t_trans,I_preset, t_trans_begin, t_trans_end, t_pulse_begin_I, t_pulse_begin_V  = np.nan,np.nan,np.nan, np.nan, np.nan, np.nan, np.nan
        
        
        my_dict = {'V_SET':V_SET,'t_pulse':t_pulse,'R_pre':resistance_pre,'R_post':resistance_after,'t_set':t_SET,'t_trans':t_trans,'I_preset':I_preset,'t_pulse_begin_V':t_trans_begin,'t_pulse_begin_I':t_trans_end,'t_trans_low':t_pulse_begin_I,'t_trans_high':t_pulse_begin_V}
        newentry_df=pd.DataFrame(my_dict, index=[0])
        
        SET_data = pd.concat([SET_data,newentry_df],ignore_index=True, sort=True)
        iplots.clear()

        SET = pd.Series(SET)
        SET = meta.attach(SET)
        #meta.attach(SET).to_pickle(os.path.join(filepath,'Set_{}.s'.format(i)))
        
        tmp = analysis_Nils(SET, t_pulse,V_SET, resistance_pre,resistance_after,trans_low=0.2,trans_high=0.8, df=tmp)
        
        
        i += 1
        
        #if resistance_after < 50000:
        if np.max(SET['I_corr'])>1e-4:
            savedata(SET,filepath=os.path.join(filepath,'Set_{}'.format(i)), attach=False, read_only=False)
            iterator += 1
        else:
            savedata(SET,filepath=os.path.join(filepath,'NoSet_{}'.format(i)), attach=False, read_only=False)
        if iterator >= n_SET:
            Done = True
        

    
    SET_data.to_pickle(os.path.join(filepath,'Analysis_F.df'))
    tmp.to_pickle(os.path.join(filepath,'Analysis_N.df'))
    
    del meta.meta['resistance_pre']
    del meta.meta['resistance_after']
    del meta.meta['V_pulse']
    del meta.meta['t_pulse']
    del meta.meta['duty'] 
    del meta.meta['tag']
    del meta.meta['HRS_min'] 
    del meta.meta['HRS_max']
    del meta.meta['V_RESET_max']
    del meta.meta['duration'] 
    del meta.meta['V_read']
    del meta.meta['t_read'] 
    del meta.meta['V_RESET_start']
    del meta.meta['V_RESET_step']
    del meta.meta['current_res']
    del meta.meta['wfm']
    del meta.meta['fs']
    
    
    return(SET_data, SET, tmp)




def matrix_repeat_SET_pulse_from_HRS(V_SET_start,V_SET_end,V_SET_step,n_SET,t_pulse,HRS_min,HRS_max,HRS_step,V_read=0.3,t_read=1e-3,V_RESET_max=-1.8,t_RESET=1e-3,V_RESET_start=-0.4, V_RESET_step=0.05, max_iterations=20):
    try:
        tb.msg_to_nils("RepeatedPulsing measurement: Started!")
        for j in np.arange(V_SET_start,V_SET_end+V_SET_step,V_SET_step):
            for i in range(HRS_min,HRS_max,HRS_step):
                cur_HRS_min = i
                cur_HRS_max = i+HRS_step
                tb.msg_to_nils("RepeatedPulsing measurement: V_set = {} HRS_min {} HRS_max {}!".format(j,i,i+HRS_step))
                _, _, _ = repeat_SET_pulse_from_HRS(V_SET=j,n_SET=n_SET,t_pulse=t_pulse,HRS_min=cur_HRS_min,HRS_max=cur_HRS_max,V_read=V_read,t_read=t_read,V_RESET_max=V_RESET_max,t_RESET=t_RESET,V_RESET_start=V_RESET_start, V_RESET_step=V_RESET_step, max_iterations=max_iterations)
        tb.msg_to_nils("RepeatedPulsing measurement: Done!")
    except:
        tb.msg_to_nils("RepeatedPulsing measurement: Error!")
        traceback.print_exc()
    