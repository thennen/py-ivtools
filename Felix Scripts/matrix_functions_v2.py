
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap




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
    TODO cycle duration column
    '''
    ###first find the measurement circumstances: V_RESET_stop and I_CC
    #I_CC=pd.DataFrame(analyze.ICC_by_vmax(data, column='V', polarity=True))['I']
    I_CC=pd.DataFrame(data)['CC']
    
    #print(len(V_RESET_stop))
    
    ###find out resistances

    V_RESET_stop=pd.DataFrame({'V_RESET_stop':np.floor(10*analyze.V_RESET_stop_by_vmax(data, column='V', polarity=True)['V'])/10})
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
        
    
    
    
    
    
    


def split_data_into_matrix(data,plot=True,cmap='jet',saveplots=False,analyse=True,vlow_resistance=0.1,vhigh_resistance=0.3,v_lowSET=0.15,v_highSET=5,N_RESETstride=5,v_lowRESET=-5,v_highRESET=-0.3,SET_current_shift=3):
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
    if plot==True:
        scatterargs = dict(s=10, alpha=.8, edgecolor='none')
        errorbarargs=dict(alpha=.8)
        lineargs=dict(alpha=.8)
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
        if plot==True:
            # figIcc, axIcc = plt.subplots()##figIcc sind Sweeps mit constanter CC
            # axIcc.set_xlabel('Voltage [V]')
            # axIcc.set_ylabel('Current [A]')
            # axIcc.set_yscale('log')
            # axIcc.set_ylim(bottom=1e-9, top=5e-3)
            # axIcc.set_xlim(left=-2, right=2.5)
            # legendIcc_handles=[]
            # legendIcc_labels=[]
            pass
        print(Icc)
        for j,Vreset in enumerate(Vresets[::-1]):
            count_combinations=count_combinations+1
            print(count_combinations)
            print(Vreset)
            mask = (np.around(data['I'].apply(max),4)==Icc) & (np.floor(10*data['V'].apply(min))/10==Vreset)
            current_df= data.loc[mask]
            if analyse==True:
                plot_analysis=True
                current_ana=analyze_matrix(current_df,vlow_resistance,vhigh_resistance,v_lowSET,v_highSET,N_RESETstride,v_lowRESET,v_highRESET,SET_current_shift)
                combined_stats=pd.concat([current_ana.describe(),pd.DataFrame(current_ana.median(), columns = ["median"] ).T,pd.DataFrame(current_ana.mad(), columns = ["mean_ad"] ).T,pd.DataFrame(abs(current_ana-current_ana.median()).median(),columns=["median_ad"]).T])
            else:
                plot_analysis=False
            if plot==True:
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
            for kkk in current_df.index:
                print(kkk)
                
                #lineIcc=axIcc.plot(current_df.iloc[kkk]['V'],abs(current_df.iloc[kkk]['I']),c=Vreset_stop_colors[j])
                #legendIcc_handles.append(lineIcc)
                #legendIcc_labels.append(str(Vreset)) 
                
            
    
    
    if plot==True:
        ax1.legend(legend1_handles[0:len(unique_V_RESET_stop)],legend1_labels[0:len(unique_V_RESET_stop)])
        ax1med.legend(legend1med_handles,legend1med_labels)
        ax2.legend(legend2_handles[::len(unique_ICC)],legend2_labels[::len(unique_ICC)])
        ax2med.legend(legend2med_handles,legend2med_labels)
    return(current_df)
    
    
    
    