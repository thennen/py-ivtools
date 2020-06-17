from ivtools import io
import numpy as np

meta = io.MetaHandler()

measurement = {'V': np.random.rand(10), 'I': np.random.rand(10) / 10, 'Instrument': 'fake', 'Some rand': 100*np.random.rand()}

meta.load_nanoxbar()
meta.step(3)
meta.savedata(measurement)

'''
#Code to run in console for testing
cd C:/Users/munoz/Desktop/py-ivtools
run interactive
d = {'V': np.random.rand(10), 'I': np.random.rand(10) / 10, 'Instrument': 'fake', 'Some rand': 100*np.random.rand()}
meta.load_nanoxbar()
meta.step(3)
savedata(d)
'''