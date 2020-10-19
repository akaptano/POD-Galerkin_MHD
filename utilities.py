from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.io import loadmat
from matplotlib import rcParams
from numpy import pi
rcParams.update({'figure.autolayout': True})
mu0 = 4*pi*10**(-7)
ts = 18
ls = 26
lw = 6
ms = 10
def grid(x, y, z, resX=200, resY=200):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi,indexing='ij')
    print(np.shape(x),np.shape(y),np.shape(z))
    Z = griddata((x, y), z, (X, Y),'cubic')
    return X, Y, Z
