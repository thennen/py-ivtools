import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm
import os
import matplotlib as mpl

def my_func(iv, v_low=0.1, v_high=0.3):
    mask = (iv['V']>=v_low) & (iv['V']<=v_high)
    grad = np.gradient(iv['V'][mask])
    v = iv['V'][mask]
    i = iv['I'][mask]
    r1 = np.polyfit(i[grad>0],v[grad>0],1)[0]
    r2 = np.polyfit(i[grad<0],v[grad<0],1)[0]
    return np.array([r1, r2])

def my_analyze_set(iv, v_low=0.1, v_high=5):
    #v_set max slope in pos half cycle
    mask = (iv['V']>v_low) & (iv['V']<v_high) & (np.gradient(iv['V'])>0)
    mask[-1] = False
    #mask = iv['V']>0
    i_set = np.argmax(np.gradient(iv['I'][mask]))
    #print(np.gradient(iv['I'][mask])
    return iv['V'][mask][i_set]


def my_analyze_reset(iv, N=5, v_low=-0.1, v_high=-5):
    #print(iv)
    #v_reset find I minimum in negativ half cycle
    mask = (iv['V']<v_low) & (iv['V']>v_high) & (np.gradient(iv['V'])<0)
    x = np.convolve(iv['V'][mask], np.ones(N)/N)[:-5]
    y = np.convolve(iv['I'][mask], np.ones(N)/N)[:-5]
    i = np.where(np.r_[True, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:],True])[0][0]
    return iv['V'][mask][i]
    #return i


#data = pd.read_pickle("2019-08-16_154525_739_Icc_700_Vreset_-1.5.df")
#Iccs = [700]
#Vresets = [-1.5]

def my_dataframe(data, Icc, Vreset):
    mask = (np.around(data['I'].apply(max),4)*1e6==Icc) & (np.around(data['V'].apply(min),1)==Vreset)
    d = data.loc[mask,['I','V']]
    return d
