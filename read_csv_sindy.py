import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from sindy_utils import inner_product, \
    plot_measurement, make_contour_movie, \
    plot_BOD_Fspectrum
from load_incompressible_data import load_data
from compressible_Framework import compressible_Framework

# Truncation number for the SVD
r = 3
# Threshold for SINDy algorithm
threshold = 0.05
# Maximum polynomial order to use in the SINDy library
poly_order = 2
# Start time index
start = 20000
# End time index
end = 30000
# Dump files are written every 10 simulation steps
skip = 10
spacing = float(skip)
# Fraction of the data to use as training data
tfac = 3.0/5.0
time = np.loadtxt('../time.csv')
M = len(time)
# shorten and put into microseconds
time = time[int(start/spacing):int(end/spacing):int(skip/spacing)]*1e6
# Load in only the measurement coordinates for now
X, Y, Z = np.loadtxt('../HITSI_rMHD_HR_00000.csv', \
    delimiter=',',usecols=(0,1,2),unpack=True)
R = np.sqrt(X**2+Y**2)
phi = np.arctan2(Y,X)
dR = np.unique(R.round(decimals=4))[1]
dphi = np.unique(phi.round(decimals=4))[1]- \
    np.unique(phi.round(decimals=4))[0]
dZ = np.unique(Z)[1]-np.unique(Z)[0]
# Load in all the field data
Bx_mat,By_mat,Bz_mat,Vx_mat,Vy_mat,Vz_mat = load_data(start,end,skip)
n_samples = np.shape(Bx_mat)[0]
Q = np.vstack((Bx_mat,By_mat))
Q = np.vstack((Q,Bz_mat))
Q = np.vstack((Q,Vx_mat))
Q = np.vstack((Q,Vy_mat))
Q = np.vstack((Q,Vz_mat))
R = np.ravel([R,R,R,R,R,R])
inner_prod = inner_product(Q,R)*dR*dphi*dZ
# Perform the vector POD method
t_test,x_true,x_sim = compressible_Framework(inner_prod,time,poly_order,threshold,r)
M_test = len(t_test)
Qorig = Q[:,int(len(time)*tfac):]
# need to get put in the radial weights
Q = Q*np.sqrt(dR*dphi*dZ)
for i in range(M):
    Q[:,i] = Q[:,i]*np.sqrt(R)
U,S,Vh = np.linalg.svd(Q,full_matrices=False)
Q = ((U*S)[:,0:r])@Vh[0:r,int(len(time)*tfac):]
Q_true = ((U*S)[:,0:r])@(np.transpose(x_true))
Q_test= ((U*S)[:,0:r])@(np.transpose(x_sim))
# now getting rid of the radial weights
Q = Q/np.sqrt(dR*dphi*dZ)
Q_true = Q_true/np.sqrt(dR*dphi*dZ)
Q_test = Q_test/np.sqrt(dR*dphi*dZ)
for i in range(M_test):
    Q[:,i] = Q[:,i]/np.sqrt(R)
    Q_true[:,i] = Q_true[:,i]/np.sqrt(R)
    Q_test[:,i] = Q_test[:,i]/np.sqrt(R)
print('% Field = ',sum(S[0:r])/sum(S))
plot_BOD_Fspectrum(S)
print(np.shape(Vh),np.shape(Q))
plot_measurement(Qorig,Q,Q_true,Q_test,t_test)
Bx_mat_sim = Q_test[0:1*n_samples,:]
By_mat_sim = Q_test[1*n_samples:2*n_samples,:]
Bz_mat_sim = Q_test[2*n_samples:3*n_samples,:]
Vx_mat_sim = Q_test[3*n_samples:4*n_samples,:]
Vy_mat_sim = Q_test[4*n_samples:5*n_samples,:]
Vz_mat_sim = Q_test[5*n_samples:6*n_samples,:]
#make_contour_movie(x,y,z,Bx_mat[:,int(len(time)*tfac):],Bx_mat_sim,t_test,'Bx')
#make_contour_movie(x,y,z,By_mat[:,int(len(time)*tfac):],By_mat_sim,t_test,'By')
#make_contour_movie(x,y,z,Bz_mat[:,int(len(time)*tfac):],Bz_mat_sim,t_test,'Bz')
#make_contour_movie(x,y,z,Vx_mat[:,int(len(time)*tfac):],Vx_mat_sim,t_test,'Vx')
#make_contour_movie(x,y,z,Vy_mat[:,int(len(time)*tfac):],Vy_mat_sim,t_test,'Vy')
#make_contour_movie(x,y,z,Vz_mat[:,int(len(time)*tfac):],Vz_mat_sim,t_test,'Vz')
plt.show()
