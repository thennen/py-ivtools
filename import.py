import ivtools
# Reload all the modules in case they changed
from importlib import reload

#importlib.reload(ivtools)
#importlib.reload(ivtools.measure)
#importlib.reload(ivtools.analyze)
#importlib.reload(ivtools.plot)
#importlib.reload(ivtools.io)
#importlib.reload(ivtools.instruments)

import ivtools.analyze as analyze
import ivtools.plot as ivplot
import ivtools.io as io
import ivtools.measure as measure

reload(analyze)
reload(ivplot)
reload(io)
reload(measure)

#from ivtools import measure
#from ivtools import analyze
#from ivtools import plot as ivplot
#from ivtools import io

# Throw everything into the namespace
from ivtools.measure import *
from ivtools.analyze import *
from ivtools.plot import *
from ivtools.io import *
from ivtools.instruments import *
