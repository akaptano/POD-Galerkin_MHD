import compressible_Framework
import sys
import pysindy.optimizers
from importlib import reload
reload(compressible_Framework)
from compressible_Framework import compressible_Framework as framework
del sys.modules['pysindy.optimizers.sr3_enhanced']
reload(pysindy.optimizers)
from pysindy.optimizers import SR3Enhanced
