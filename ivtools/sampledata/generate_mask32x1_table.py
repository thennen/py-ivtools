import pandas as pd

block = pd.DataFrame({'Block': [1,2,3,4,5,6,7,8]})
device = pd.DataFrame({'Device': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32], 'xx':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]})
rows = pd.DataFrame({'Row': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32], 'yy':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]})

block['key'] = 0
device['key'] = 0
rows['key'] = 0

mask32x1 = block.merge(rows, on='key').merge(device, on='key').drop('key', 1)

def makeid(row):
    return 'Block{}_Row{}_Device{}'.format(*row[['Block', 'Row', 'Device']])

mask32x1['id'] = mask32x1.apply(makeid, 1)


mask32x1.to_pickle('mask32x1.pkl')
mask32x1.to_excel('mask32x1.xls', index=False)
