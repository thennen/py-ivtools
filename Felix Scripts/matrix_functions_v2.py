def measure_matrix(ICC_Start,ICC_Stop,ICC_inc,Vreset_Start,Vreset_Stop,Vreset_inc,Vset_stop, n_cycle=100, duration=1e-3,fs=1.25e9):
    '''
    Define Current Compliance Start, Stop and Increment in micra-A,
    Reset Voltage start, stop and increment (negative values *100),
    and regular picoiv-stuff.
    '''
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
            ps.squeeze_range(temp,padpercent=0.2)
            print('----------------------------------')
            print('Changed RESET Stop Voltage to {}'.format(V_RESET_STOP))
            print('----------------------------------')
            d=picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=n_cycle, fs=fs,smartrange=False)
            #d_in_df=pd.DataFrame(d)
            appended_data_out += d
    savedata(appended_data_out,filepath = os.path.join(datadir(), meta.filename()+'_combined_matrix'))
            
    return(appended_data_out)
	


def analyze_matrix(data,vlow_resistance=0.1,vhigh_resistance=0.3,v_lowSET=0.15,v_highSET=5,N_RESETstride=5,v_lowRESET=-5,v_highRESET=-0.3,SET_current_shift=3):
    '''
    Analyze a dataframe and write the data to an excel file so it can be later plotted in Origin or other programms
    '''
    ###first find the measurement circumstances: V_RESET_stop and I_CC
    #I_CC=pd.DataFrame(analyze.ICC_by_vmax(data, column='V', polarity=True))['I']
    I_CC=pd.DataFrame(data)['CC']
    
    V_RESET_stop=pd.DataFrame({'V_RESET_stop':analyze.V_RESET_stop_by_vmax(data, column='V', polarity=True)['V']})
    #print(len(V_RESET_stop))
    
    ###find out resistances

    HRS=pd.DataFrame({'HRS':analyze.resistance_states_fit(data,v_low=vlow_resistance,v_high=vhigh_resistance)[0]})
    LRS=pd.DataFrame({'LRS':analyze.resistance_states_fit(data,v_low=vlow_resistance,v_high=vhigh_resistance)[1]})
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
    V_RESET=pd.DataFrame({'V_RESET':analyze.V_RESET_by_change_of_gradient(data=data, N=N_RESETstride, v_low=v_lowRESET,v_high=v_highRESET)})
    I_SET=pd.DataFrame({'I_SET':analyze.I_SET_by_max_gradient_shifted(data=data,v_low=v_lowSET,v_high=v_highSET,shift_index_by=SET_current_shift)})
    I_RESET=pd.DataFrame({'I_RESET':analyze.I_RESET_by_change_of_gradient(data=data, N=N_RESETstride, v_low=v_lowRESET,v_high=v_highRESET)})
    
     
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
    badV_RESET_Currentdefinition=sum(float(analysis_data.iloc[i]['V_RESET']/analysis_data.iloc[i]['V_RESET_stop']) >=0.95 for i in analysis_data.index)
    print('There are {} cases of V_RESET equal to V_RESET_stop in the analysis'.format(badV_RESET_Currentdefinition))
    
    
    
    
    
    
    
def save_matrix_analysis(analysis_data,filepath=None,drop=None):

    '''
    saves the analyzed data in the current working folder as dataframe and csv
    ''' 
    if filepath is None:
        filepath_df = os.path.join(datadir(), meta.filename()+'_matrix_analysis_df')
        filepath_csv=os.path.join(datadir(), meta.filename()+'_matrix_analysis_csv.csv')
    print('Data to be saved in {}'.format(filepath_df))
    io.write_pandas_pickle(meta.attach(analysis_data), filepath_df, drop=drop)
    analysis_data.to_csv(path_or_buf=filepath_csv)
    print('Wrote ',format(filepath_csv))
    
    
    
def plot_flat_matrix_analysis(analysis_data):
    fig, ax = plt.subplots()
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
    
    fig, ax = plt.subplots()
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
    
    fig, ax = plt.subplots()
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
    
    fig, ax = plt.subplots()
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
    
    
    fig, ax = plt.subplots()
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
