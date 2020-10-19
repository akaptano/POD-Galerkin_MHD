import numpy as np
from matplotlib import pyplot as plt

temporal_modes = np.loadtxt('trajectories.txt')
for i in range(6):
    plt.figure(i+1)
    plt.plot(temporal_modes[:,0],temporal_modes[:,i+1])

plt.show()
