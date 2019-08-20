def measure_matrix(ICC_Start,ICC_Stop,ICC_inc,Vreset_Start,Vreset_Stop,Vreset_inc,Vset_stop, n_cycle=100, duration=1e-3,fs=1.25e9):
    '''
    Define Current Compliance Start, Stop and Increment in micra-A,
    Reset Voltage start, stop and increment (negative values *100),
    and regular picoiv-stuff.
    '''
    appended_data_out=[]
    for I_CC in range(ICC_Start,ICC_Stop+ICC_inc,ICC_inc):
        set_compliance(I_CC*1e-6)
        print('-----------------------------')
        print('Changed Current Compliance to {}'.format(I_CC))
        print('-----------------------------')
        for V_RESET_STOP in range(Vreset_Start,Vreset_Stop+Vreset_inc,Vreset_inc):
            ps.range.b = 5
            ps.offset.b = 0
            ps.range.a = 5
            ps.offset.a = 0
            temp = picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=1, fs=fs,smartrange=False)
            ps.squeeze_range(temp,padpercent=0.2)
            print('-----------------------------')
            print('Changed RESET Stop Voltage to {}'.format(V_RESET_STOP))
            print('-----------------------------')
            d=picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=n_cycle, fs=fs,smartrange=False)
            #d_in_df=pd.DataFrame(d)
            appended_data_out += d
            
    return appended_data_out
	