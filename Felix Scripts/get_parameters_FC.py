# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:26:18 2019

@author: CC
"""
from functools import wraps
import numpy as np
from itertools import groupby
from scipy import signal
from numbers import Number
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
import sys


def ivfunc(func):
    '''
    Decorator which allows the same function to be used on a single loop, as
    well as a container of loops.

    Decorated function should take a single loop and return anything
    Then this function will also take multiple loops, and return a list/dataframe of the outputs

    Handles dicts and pd.Series for singular input data, as well as "list of dict" and DataFrames for multiple input data
    An attempt is made to return the most reasonable type, given the input and output types

    Some preliminary testing indicates that operating on dataframes can be much slower (~100x) than list of dicts.

    If any of the arguments are instances of "paramlist", this tells ivfunc to also index into this list when
    calling the wrapped function, so that you can pass a list of parameters to use for each data row.

    If you pass as an argument a function wrapped with the paramfunc function, that function will get called on each of
    the data to determine the argument individually.
    '''
    def paramtransform(param, i, data):
        if type(param) == paramlist:
            # Index param if it is a paramlist. Otherwise, don't.
            return param[i]
        #elif hasattr(param, '__call__'):
        elif hasattr(param, '__name__') and (param.__name__ == 'paramfunc'):
            # Call function with data as argument, and use the return value as the parameter
            return param(data)
        else:
            return param
    @wraps(func)
    def func_wrapper(data, *args, **kwargs):
        dtype = type(data)
        ###################################
        ###  IF A DATAFRAME IS PASSED   ###
        ###################################
        if dtype == pd.DataFrame:
            # Apply the function to the columns of the dataframe
            # Sadly, the line below will error if your function returns an array, because pandas.
            # Although it can be significantly faster if your function does not return an array..
            # return data.apply(func, axis=1, args=args, **kwargs)
            # Since we don't already know what type the wrapped function will return,
            # we need to explicitly loop through the dataframe rows, and store the results in an intermediate list
            resultlist = []
            for i, (rownum, row) in enumerate(data.iterrows()):
                #resultlist.append(func(row, *args, **kwargs))
                result = func(row, *[paramtransform(arg, i, row) for arg in args],
                              **{k:paramtransform(v, i, row) for k,v in kwargs.items()})
                resultlist.append(result)
            ### Decide how to return the values based on the datatype that the wrapped function returned
            if type(resultlist[0]) in (pd.Series, dict):
                # Each row returns a dict or series
                if type(resultlist[0][list(resultlist[0].keys())[0]]) is dict:
                    # Each dict probably just contains another dict.
                    # Return a dataframe with multiindex columns
                    # It hurts my brain to think about how to accomplish this, so for now it does the same
                    df_out = pd.DataFrame(resultlist)
                    df_out.index = data.index
                else:
                    df_out = pd.DataFrame(resultlist)
                    df_out.index = data.index
                return df_out
            elif (type(resultlist[0]) is list):
                if (type(resultlist[0][0]) is dict):
                    # Each row returns a list of dicts, stack the lists into new dataframe
                    # Mainly used for splitting loops, so that everything stays flat
                    # Maybe return a panel instead?
                    # Index will get reset ...
                    # Unless ...
                    index = np.repeat(data.index, [len(sublist) for sublist in resultlist])
                    return pd.DataFrame([item for sublist in resultlist for item in sublist], index=index)
                elif isinstance(resultlist[0][0], Number):
                    # Each row returns a list of numbers
                    # Make dataframe
                    df_out = pd.DataFrame(resultlist)
                    df_out.index = data.index
                    return df_out
            elif all([r is None for r in resultlist]):
                # If ivfunc returns nothing, return nothing
                return None
            # For all other cases
            # Keep the index the same!
            series_out = pd.Series(resultlist)
            series_out.index = data.index
            return series_out
        #######################################
        ### IF A LIST (OF DICTS) IS PASSED  ###
        #######################################
        elif dtype == list:
            # Assuming it's a list of iv dicts
            #resultlist = [func(d, *args, **kwargs) for d in data]
            resultlist = []
            for i, d in enumerate(data):
                result = func(d, *[paramtransform(arg, i, d) for arg in args],
                              **{k:paramtransform(v, i, d) for k,v in kwargs.items()})
                resultlist.append(result)
            if (type(resultlist[0]) is list):
                if (type(resultlist[0][0]) is dict):
                    # Each func(dict) returns a list of dicts, stack the lists
                    return [item for sublist in resultlist for item in sublist]
                elif isinstance(resultlist[0][0], Number):
                    # Each iv dict returns a list of numbers
                    # "Unpack" them
                    return list(zip(*resultlist))
            elif all([r is None for r in resultlist]):
                # If ivfunc returns nothing, return nothing
                return None
            # For all other return types
            return resultlist
        elif dtype is pd.Series:
            # It's just one IV Series
            # If it returns a dict, convert it back to a series
            result = func(data, *args, **kwargs)
            if type(result) is dict:
                return(pd.Series(result))
            else:
                return result
        elif dtype in (dict,):
            # It's just one IV dict
            return(func(data, *args, **kwargs))
        else:
            print('ivfunc did not understand the input datatype {}'.format(dtype))
    return func_wrapper






def get_parameters(data, v0, threshold=0.01, polarity = False):
    
    
    #include cycle number
    data['# Cycle'] = data.index



    #get ICC from current@vamx
    #index_Vmax = select_by_vmax(data, column='V', polarity=False)
    index_ICC = ICC_by_vmax(data, column='V', polarity=polarity)

    data['ICC']=index_ICC['I']
    data['V_ICC']=index_ICC['V']
    
    #get Set Voltage and Current     
    #index_Vset = threshold_byderivative(data, threshold=0.001, interp=False, debug=False)

    
    index_Vset1 = VpolSet(data, polarity=polarity)

    data['VpSet']= index_Vset1['V']
    data['IpSet']= index_Vset1['I']  
    
    index_Vset2 = Set_polarity(data, polarity=True)    
    data['IpSet2']= index_Vset2  
    
    index_Vreset2 = Reset_polarity(data, polarity=True)    
    data['IpReset2']= index_Vreset2  
    
    index_Vreset1 = VpolReset(data, polarity=polarity)    
    data['VpReset']= index_Vreset1['V']
    data['IpReset']= index_Vreset1['I'] 

    index_Vset = thresholds_bydiff2(data, stride=5)
    data['ISet']= index_Vset['IpSet']
    data['VSet']= index_Vset['VpSet']     
         
    #get Reset voltage and Reset current
    #index_Reset = select_by_imax(data, column='I', polarity=False)
    index_VReset = Reset_by_imax(data, column='I', polarity=polarity)

    data['IReset']=index_VReset['I']
    data['VReset']=index_VReset['V']




    
    #get HRS and LRS
    index_HRS_LRS = resistance_states(data, v0, v1=None)
    
    if polarity:
        data['LRS']=index_HRS_LRS[1]
        data['HRS']=index_HRS_LRS[0]
    else:
        data['HRS']=index_HRS_LRS[1]
        data['LRS']=index_HRS_LRS[0]        
    
    index_HRS_pos = resistance_states_pos(data, v0, v1=None)
    data['LRS_pos']=index_HRS_pos[1]
    data['HRS_pos']=index_HRS_pos[0]
    
    index_HRS_neg = resistance_states_neg(data, v0, v1=None)
    data['LRS_neg']=index_HRS_neg[1]
    data['HRS_neg']=index_HRS_neg[0]
    
    
    
@ivfunc 
def select_by_imax(data, column='I', polarity=True):
    '''
    Find Reset voltage by finding the mmaximum (maximum negative for polarity=False) current.
    Return index of the threshold datapoint
    '''
    
    if polarity:
        imax = np.argmax(data[column])
    else:
        imax = np.argmin(data[column])
    return imax



@ivfunc
def Reset_by_imax(data, column='I', polarity=True):
    ''' Find Reset voltage by finding the mmaximum (maximum negative for polarity=False) current. '''

    peter = select_by_imax(data, column='I', polarity=polarity)

    return indexiv(data, peter)


@ivfunc  
def select_by_vmax(data, column='V', polarity=False):
    '''
    Find Icc by finding the mmaximum (maximum negative for polarity=False) voltage.
    Return index of the threshold datapoint
    '''
    
    if polarity:
        vmax = series.values.argmax(data[column])
    else:
        vmax = series.values.argmax(data[column])
    return vmax



@ivfunc
def ICC_by_vmax(data, column='V', polarity=True):
    ''' Find ICC by finding the mmaximum (maximum negative for polarity=False) voltage. '''

    vmax1 = select_by_vmax(data, column='V', polarity=polarity)

    return indexiv(data, vmax1)