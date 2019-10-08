
def measure_matrix(ICC_Start,ICC_Stop,ICC_inc,Vreset_Start,Vreset_Stop,Vreset_inc,Vset_stop,Vread=0.4, n_cycle=100,n=1, duration=1e-3,fs=1.25e9):
    '''
    Define Current Compliance Start, Stop and Increment in micra-A,
    Reset Voltage start, stop and increment (negative values *100),
    and regular picoiv-stuff.
    '''
    matrix_out=[]
    hrs_out =  []
    lrs_out =  []
    for I_CC in range(ICC_Start,ICC_Stop+ICC_inc,ICC_inc):
        set_compliance(I_CC*1e-6)
        print('-----------------------------')
        print('Changed Current Compliance to {}'.format(I_CC))
        print('-----------------------------')
        for V_RESET_STOP in range(Vreset_Start,Vreset_Stop+Vreset_inc,Vreset_inc):
            print('-----------------------------')
            print('Changed RESET Stop Voltage to {}'.format(V_RESET_STOP))
            print('-----------------------------')
            for i in range(n):
                ps.range.b = 5; ps.offset.b = 0; ps.range.a = 5; ps.offset.a = 0
                temp = picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=1, fs=fs,smartrange=False)
                ps.squeeze_range(temp,padpercent=0.2)
                d=picoiv(tri(Vset_stop, V_RESET_STOP/100), duration=duration, n=n_cycle, fs=fs,smartrange=False)
                #d_in_df=pd.DataFrame(d)
                matrix_out += d

                #Measure HRS read variability
                ps.range.b=0.5; ps.offset.b=0.75; ps.range.a=0.5; ps.offset.a = 0
                d=picoiv(tri(Vread,-Vread), duration=duration,n=10*n_cycle,fs=fs/10,smartrange=False)
                hrs_out += d

                #Measure LRS read variability
                ps.range.b = 5; ps.offset.b = 0; ps.range.a = 5; ps.offset.a = 0
                temp=picoiv(tri(Vset_stop,-0.1),duration=duration,n=1,fs=fs,smartrange=False)
                temp=picoiv(tri(Vread,-Vread),duration=duration,n=1,fs=fs/10,smartrange=False)
                ps.squeeze_range(temp)
                d=picoiv(tri(Vread,-Vread),duration=duration,n=10*n_cycle,fs=fs/10,smartrange=False)
                lrs_out += d
                ps.range.b = 5; ps.offset.b = 0; ps.range.a = 5; ps.offset.a = 0
                temp=picoiv(tri(0.1, V_RESET_STOP/100), duration=duration,n=1, fs=fs,smartrange=False)




    return matrix_out, hrs_out, lrs_out
