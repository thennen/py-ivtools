import pandas as pd

xs = pd.DataFrame({'X': [1,2,3,4,5,6,7,8,9,10,11,12]})
ys = pd.DataFrame({'Y': [1,2,3,4,5,6]})
device = pd.DataFrame({'device': [1,2,3,4,5], 'xx':[1,2,3,4,5]})
width = pd.DataFrame({'width_nm': [500, 200, 100, 80, 60], 'yy':[1,2,3,4,5]})
xs['key'] = 0
ys['key'] = 0
device['key'] = 0
width['key'] = 0
nanoxbar = xs.merge(ys, on='key').merge(width, on='key').merge(device, on='key').drop('key', 1)

def makeid(row):
    return 'X{:02}Y{:02}_{}nm_{}'.format(*row[['X', 'Y', 'width_nm', 'device']])

nanoxbar['id'] = nanoxbar.apply(makeid, 1)


nanoxbar.to_pickle('nanoxbar2.pickle')
nanoxbar.to_excel('nanoxbar2.xls', index=False)
