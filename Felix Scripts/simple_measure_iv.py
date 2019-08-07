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

SavePlace='E:\\transfer\\CC\\Felix\\FC_1_32_100nm_1um_Y01_D11\\SimpleIV_test\\'
import datetime
try:
    os.mkdir(SavePlace)
except FileExistsError:
    pass
filename = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

###ACTUAL MEASUREMENT stuff:###
set_compliance(200e-6)
ps=instruments.Picoscope()
ps.range['B']=2
ps.offset['B']=2
d=picoiv(tri(1.5, -1.5), duration=1e-3, n=2, fs=1.25e9, smartrange=True, savewfm=True)

####

meta=io.MetaHandler()
savedata(d,filepath=SavePlace+filename)


#if type(d)==dict:
data=pd.DataFrame.from_dict(d)
#else:
 #   data = pd.DataFrame(d)
data['I_abs']=data['I'].abs()
data['R']=data['V']/data['I']

plotiv(data[::1])
#plt.yscale('lin')
plt.ylim(-1e-3, 1e-3)
plt.xlabel('Voltage / V')
plt.ylabel('Current / A')
plt.savefig(SavePlace+filename+'_IV_lin.png', dpi=300)


plotiv(data[::1], x='V', y='I_abs')
plt.yscale('log')
plt.ylim(1e-7, 2e-3)
#plt.autoscale(enable=True, axis='both', tight=None)
plt.xlabel('Voltage / V')
plt.ylabel('Current / A')
plt.savefig(SavePlace+filename+'_IV_log.png', dpi=300)


plotiv(data[::1], x='V', y='R')
plt.yscale('log')
plt.ylim(5e2, 5e7)
#plt.autoscale(enable=True, axis='both', tight=None)
plt.xlabel('Voltage / V')
plt.ylabel('Resistance / Ohm')
plt.savefig(SavePlace+filename+'_RV_log.png', dpi=300)




