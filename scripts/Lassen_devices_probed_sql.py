# To visualize which devices have already been probed
# Using the metadata

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

if len(sys.argv) > 1:
    dep_code = sys.argv[1]
    sample_number = int(sys.argv[2])
else:
    dep_code = 'Ente'
    sample_number = 2

meta = load_metadb().dropna(0, how='any', subset=['dep_code', 'sample_number']).dropna(1, how='all')
meta['sample_number'] = np.int32(meta['sample_number'])
probed_devices = meta[(meta['dep_code'] == dep_code) & (meta['sample_number'] == sample_number)][['module', 'device','die_rel']].drop_duplicates()

probed_devices.device = np.int8(np.float64(probed_devices.device))
probed_devices.die_rel = np.int8(np.float64(probed_devices.die_rel))
# has a row for every device on a wafer
lassen = pd.read_pickle('C:/t/py-ivtools/ivtools/sampledata/all_lassen_device_info.pkl')
lassen = lassen[lassen.coupon == 30]
lassen = lassen[lassen.module_num.isin([1, 14])]
# only care about these
lassen = lassen[['die_rel', 'module', 'device']].reset_index(drop=True)

scriptdir = sys.path[0]
couponpng = os.path.join(scriptdir, 'coupon.png')
im = plt.imread(couponpng)

fig, ax = plt.subplots()
#plt.imshow(im)
# a little easier to see the markings
plt.imshow(np.mean(im, 2), cmap='Greys_r')

#upper left corner of each die (upper left pixel of the upper left most pad).
# dies left to right, top to bottom 1,2,3,4
# (y,x) because indexing is stupid
dielocs = {1:(2, 0),
           2:(2, 929),
           3:(634, 1),
           4:(634, 929)}

#dielocs = {k:np.array(v) for k,v in dielocs.items()}

lassen['pngx'] = [dielocs[i][1] for i in lassen['die_rel']]
lassen['pngy'] = [dielocs[i][0] for i in lassen['die_rel']]

# relative location of 001 modules, ABCDEFGHI
# top left of top left pad
modlocs = {'001' :(156, 902),
           '001B':(156, 863),
           '001C':(313, 863),
           '001D':(  0, 688),
           '001E':(156, 688),
           '001F':(313, 688),
           '001G':(156, 188),
           '001H':(156, 257),
           '001I':(156, 572),
           '014' :(314, 648),
           '014B':(157, 604),
           '014C':(314, 604),
           '014D':(  0, 112),
           '014E':(157, 112),
           '014F':(471, 689),
           '014G':(471, 864),
           '014H':(  0,  80),
           '014I':(157,  80)}

#modlocs = {k:np.array(v) for k,v in modlocs.items()}

lassen['pngx'] += [modlocs[i][1] for i in lassen['module']]
lassen['pngy'] += [modlocs[i][0] for i in lassen['module']]

#relative location of 001 devices 12345678
devicelocs_001 = [(138, 0),
                  (126, 0),
                  (113, 0),
                  (101, 0),
                  (88, 0),
                  (76, 0),
                  (63, 0),
                  (50, 0),]
devicelocs_001 = {i+1:v for i,v in enumerate(devicelocs_001)}

# relative location of 014 devices 123456789
devicelocs_014 = [(138, 0),
                  (113, 0),
                  (100, 0),
                  (88, 0),
                  (63, 0),
                  (50, 0),
                  (38, 0),
                  (13, 0),
                  (0, 0)]
devicelocs_014 = {i+1:v for i,v in enumerate(devicelocs_014)}

def wtf(row):
    if row.module.startswith('001'):
        return devicelocs_001[row.device][0]
    elif row.module.startswith('014'):
        return devicelocs_014[row.device][0]

lassen['pngy'] += lassen.apply(wtf, 1)

# scatter all the devices
#plt.scatter(lassen.pngx, lassen.pngy)


probed_devices = probed_devices.merge(lassen, on=['module', 'device', 'die_rel'], how='left')

#ax.scatter(probed_devices.pngx, probed_devices.pngy, color='red')

for i,r in probed_devices.iterrows():
    ax.add_patch(plt.Rectangle((r.pngx, r.pngy), 22, 10, color='red', alpha=.8))

plt.title(f'{dep_code} {sample_number}')
plt.show()
