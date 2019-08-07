# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:44:46 2019

@author: CC
"""
def savedata(data=None, filepath=None, drop=None):
    '''
    Save data with metadata attached, as determined by the state of the global MetaHandler instance
    if no data is passed, try to use the global variable d
    filepath automatic by default.
    can drop columns to save disk space.
    '''
    if data is None:
        global d
        if type(d) in (dict, list, pd.Series, pd.DataFrame):
            print('No data passed to savedata(). Using global variable d.')
            data = d
    if filepath is None:
        filepath = os.path.join(datadir(), meta.filename())
    io.write_pandas_pickle(meta.attach(data), filepath, drop=drop)
    # TODO: append metadata to a sql table
# just typing s will save the d variable
#s = autocaller(savedata)
def offset_subtractor(data, vrange=0.05):
    V = data['V']
    vmask = np.where((data['V'])<=vrange)
    if not any(vmask):
        # Just take the .5% nearest data points
        Vclose = np.argsort(np.abs(V))
        N = min(1, 0.005*len(V))
        offset = np.mean(V[Vclose[:N]])
    else:
        offset = np.mean(data['I'][vmask])
    out = data.copy()
    out['I'] -= offset
    return out


SavePlace='E:\\transfer\\CC\\Stephan\\MT31\\X01Y04D1_100nm\\Matrix_Test_2\\'
import datetime
try:
    os.mkdir(SavePlace)
except FileExistsError:
    pass


###ACTUAL MEASUREMENT stuff:###
for ICC in range(100,900,100):
    for VRESET in range(14,25,2):
        set_compliance(ICC*1e-6)
        ps=instruments.Picoscope()
        ps.range['B']=2
        ps.offset['B']=2
        d=picoiv(tri(2, -VRESET/10), duration=1e-2, n=1000, fs=1.25e7, smartrange=True, savewfm=True)
    
    ####
        filename = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')+'_ICC_'+str(ICC)+'_VRESET_'+str(10*VRESET)
        meta=io.MetaHandler()
        savedata(d,filepath=SavePlace+filename)
        
        
        #if type(d)==dict:
        data=pd.DataFrame.from_dict(d)
        #else:
         #   data = pd.DataFrame(d)
        data['I_abs']=data['I'].abs()
        data['R']=data['V']/data['I']
        
        plotiv(data[::10])
        #plt.yscale('lin')
        plt.ylim(-1e-3, 1e-3)
        plt.xlabel('Voltage / V')
        plt.ylabel('Current / A')
        plt.savefig(SavePlace+filename+'_IV_lin.png', dpi=300)
        
        
        plotiv(data[::10], x='V', y='I_abs')
        plt.yscale('log')
        plt.ylim(1e-7, 2e-3)
        #plt.autoscale(enable=True, axis='both', tight=None)
        plt.xlabel('Voltage / V')
        plt.ylabel('Current / A')
        plt.savefig(SavePlace+filename+'_IV_log.png', dpi=300)
        
#        
#        plotiv(data[::1], x='V', y='R')
#        plt.yscale('log')
#        plt.ylim(5e2, 5e7)
#        #plt.autoscale(enable=True, axis='both', tight=None)
#        plt.xlabel('Voltage / V')
#        plt.ylabel('Resistance / Ohm')
#        plt.savefig(SavePlace+filename+'_RV_log.png', dpi=300)
#    
    
    #data_corr=offset_subtractor(data, vrange=0.05)
    #
    #plotiv(data[::1])
    ##plt.yscale('lin')
    #plt.ylim(-1e-3, 1e-3)
    #plt.xlabel('Voltage / V')
    #plt.ylabel('Current_corr / A')
    #plt.savefig(SavePlace+filename+'_IV_lin_corr.png', dpi=300)
    #
    #
    #plotiv(data[::1], x='V', y='I_abs')
    #plt.yscale('log')
    #plt.ylim(1e-7, 2e-3)
    ##plt.autoscale(enable=True, axis='both', tight=None)
    #plt.xlabel('Voltage / V')
    #plt.ylabel('Current_corr / A')
    #plt.savefig(SavePlace+filename+'_IV_log.png_corr', dpi=300)
    #
    #
    #plotiv(data[::1], x='V', y='R')
    #plt.yscale('log')
    #plt.ylim(5e2, 5e7)
    ##plt.autoscale(enable=True, axis='both', tight=None)
    #plt.xlabel('Voltage / V')
    #plt.ylabel('Resistance_corr / Ohm')
    #plt.savefig(SavePlace+filename+'_RV_log_corr.png', dpi=300)
    
