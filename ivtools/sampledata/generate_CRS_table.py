import pandas as pd

#Field represents the large Squares on the sample 1=A5, 25=E1 (going from left to right, from top to bottom)
field = pd.DataFrame({'Field': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]})
#xs represents the x value in a field going from left to right 1=a, 12=l
#xs = pd.DataFrame({'X': [1,2,3,4,5,6,7,8,9,10,11,12]})
#ys represents the y value in a field going from top to bottom 1=50nm, 6=500nm
#ys = pd.DataFrame({'Y': [1,2,3,4,5,6]})
#Cell top = 0, Bot = 1
cell = pd.DataFrame({'c':[0,1]})

device = pd.DataFrame({'device': [1,2,3,4,5,6,7,8,9,10,11,12], 'xx':[1,2,3,4,5,6,7,8,9,10,11,12]})
width = pd.DataFrame({'width_nm': [500, 300, 200, 100.2, 100.1, 50], 'yy':[1,2,3,4,5,6]})
#xs['key'] = 0
#ys['key'] = 0
field['key'] = 0
cell['key'] = 0
device['key'] = 0
width['key'] = 0
crs = field.merge(width, on='key').merge(device, on='key').merge(cell, on='key').drop('key', 1)

def makeid(row):
    return 'Field_{}_Row_{}nm_Device_{}_{}'.format(*row[['Field', 'width_nm', 'device', 'c']])

crs['id'] = crs.apply(makeid, 1)


crs.to_pickle('CRSmask.pickle')
crs.to_excel('CRSmask.xls', index=False)
