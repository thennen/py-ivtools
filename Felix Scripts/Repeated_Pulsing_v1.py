import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import itertools



    
    
    
    
def read_resistance(V_read=0.3,t_read=1e-3):
    read_signal=measure.tri(V_read,-V_read)
    read=picoiv(read_signal,duration=t_read, n=1, fs=1.25e9,smartrange=False)
    [R1,R2]=analyze.resistance_states_fit(read,v_low=0.05,v_high=V_read-0.05)
    R_mean=(R1+R2)/2
    return(R_mean)
    
def bring_device_to_HRS(HRS_min,HRS_max,V_RESET_max=-1.8,t_RESET=1e-3,V_read=0.3,t_read=1e-3,V_RESET_start=-0.4):
    V_RESET=V_RESET_start
    start_res=read_resistance(V_read=V_read,t_read=t_read)
    print('---Resistance is {}---'.format(int(start_res)))
    if start_res<HRS_min or start_res>HRS_max:
        resistance_correct=False
        print('---Start RESET stair---')
        RESET_tracer=[[0,start_res]]
    
        current_res=start_res
        
        while resistance_correct==False:
    
            if current_res>HRS_max:
                print('---Do one SET---')
                d=picoiv(tri(3,-0.2),duration=t_RESET,fs=1.25e9,n=1)
                current_res=read_resistance(V_read=V_read,t_read=t_read)
    
            while current_res<HRS_min and V_RESET>V_RESET_max and resistance_correct==False:
                print('---Do RESET:{}---'.format(V_RESET))
                d=picoiv(tri(0.1,V_RESET),duration=t_RESET,fs=1.25e9,n=1)
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
                    V_RESET=V_RESET-0.05###increase RESET voltage
    else:
        print('---Resistance already was correct.---')          
        RESET_tracer=np.array([[0,start_res]])
    return(RESET_tracer)
        
def save_RESET_stair(RESET_stair_data,filepath=None,drop=None):
    RESET_stair_data=pd.DataFrame(RESET_stair_data,columns={'V_RESET','R_read'})
    if filepath is None:
        filepath_df = os.path.join(datadir(), meta.filename()+'_RESET_stair_df')
        filepath_csv=os.path.join(datadir(), meta.filename()+'_RESET_stair_csv.csv')
    print('Data to be saved in {}'.format(filepath_df))
    io.write_pandas_pickle(meta.attach(RESET_stair_data), filepath_df, drop=drop)
    RESET_stair_data.to_csv(path_or_buf=filepath_csv)
    print('Wrote ',format(filepath_csv))


def analyze_SET_transient(SET_pulse,trans_low=0.2,trans_high=0.8):
    if np.max(SET_pulse['I_corr'])>1e-4:
    
        mask_over0V=SET_pulse['V']>0.5*np.max(SET_pulse['V'])
        i_puls_begin=np.argmax(mask_over0V==True)
        i_set = np.argmax(np.gradient(SET_pulse['I_corr'][mask_over0V]))+i_puls_begin
    
        t_SET=SET_pulse['t'][i_set]-SET_pulse['t'][i_puls_begin]
    
        i_trans_begin=np.argwhere(SET_pulse['I_corr'][mask_over0V]>trans_low*(np.max(SET_pulse['I_corr'][mask_over0V])-np.mean(SET_pulse['I_corr'][i_puls_begin:i_set])))
        i_trans_end=np.argwhere(SET_pulse['I_corr'][mask_over0V]>trans_high*(np.max(SET_pulse['I_corr'][mask_over0V])-np.mean(SET_pulse['I_corr'][i_puls_begin:i_set])))
        t_trans=SET_pulse['t'][i_trans_begin]-SET_pulse['t'][i_trans_end]
        if i_trans_begin>i_puls_begin:
        
            I_preset=np.mean(SET_pulse['I_corr'][i_puls_begin:i_trans_begin])
        else:
            I_preset=0
        return([[t_SET,t_trans,I_preset]])
    else:
        return([[0,0,0]])
    

def repeat_SET_pulse_from_HRS(V_SET,n_SET,t_pulse,HRS_min,HRS_max,V_read=0.3,t_read=1e-3,V_RESET_max=-1.8,t_RESET=1e-3,V_RESET_start=-0.4):
    SET_data=pd.DataFrame(columns={'V_SET','t_pulse','R_pre','R_post','t_SET','t_trans','I_preset'})
    pulse_signal=measure.square(vpulse=V_SET, duty=0.5, length=16000, startval=0, endval=0, startendratio=1)
    for i in range(0,n_SET,1):
        print('Starting cycle {} of {}'.format(int(i),int(n_SET)))
        ##first reset the device to the desired HRS and save the staircase data
        RESET_stair=bring_device_to_HRS(HRS_min=HRS_min,HRS_max=HRS_max,V_RESET_max=V_RESET_max,t_RESET=t_RESET,V_read=V_read,t_read=t_read,V_RESET_start=V_RESET_start)
        save_RESET_stair(RESET_stair)
        ##then apply read-SET-read
        measure.set_compliance(7e-4)
        resistance_pre=read_resistance(V_read=V_read,t_read=t_read)
        SET=picoiv(wfm=pulse_signal,n=1,fs=1.25e9,duration=2*t_pulse,autosmoothimate=False)
        SET['t']=analyze.maketimearray(SET)
        mask_0V=SET['V']<0.01
        current_offset=np.mean(SET['I'][mask_0V])
        SET['I_corr']=SET['I']-current_offset
        resistance_after=read_resistance(V_read=V_read,t_read=t_read)
        [[t_SET,t_trans,I_preset]]=analyze_SET_transient(SET)
        newentry_df=pd.DataFrame([[V_SET,t_pulse,resistance_pre,resistance_after,t_SET,t_trans,I_preset]], columns={'V_SET','t_pulse','R_pre','R_post','t_SET','t_trans','I_preset'})
        SET_data.append(newentry,ignore_index=True)
        c
    
    return(SET_data)
    
    