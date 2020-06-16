from ivtools import io
import numpy as np

mh = io.MetaHandler()

measurement = {'V': np.random.rand(10), 'I': np.random.rand(10) / 10, 'Instrument': 'fake', 'Some rand': 100*np.random.rand()}

mh.load_nanoxbar()
mh.step(3)
mh.savedata(measurement, 'C:/Users/munoz/Desktop/py-ivtools/ivtools/saves',
            'C:/Users/munoz/Desktop/py-ivtools/ivtools/saves/DataBase.db', 'Meta')