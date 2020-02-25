import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from sindy_utils import inner_product, \
    plot_measurement, make_contour_movie, \
    plot_BOD_Fspectrum
from load_incompressible_data import load_data
from compressible_Framework import compressible_Framework
from scipy.linalg import eig

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
# shorten and put into microseconds
time = time[int(start/spacing):int(end/spacing):int(skip/spacing)]*1e6
M = len(time)
# Load in only the measurement coordinates for now
X, Y, Z = np.loadtxt('../HITSI_rMHD_HR_00000.csv', \
    delimiter=',',usecols=(0,1,2),unpack=True)
X = X*100
Y = Y*100
Z = Z*100
R = np.sqrt(X**2+Y**2)
phi = np.arctan2(Y,X)
dR = np.unique(R.round(decimals=4))[1]
dphi = np.unique(phi.round(decimals=4))[1]- \
    np.unique(phi.round(decimals=4))[0]
dZ = np.unique(Z)[1]-np.unique(Z)[0]
print(dR,dphi,dZ,dR*dphi*dZ,min(R),max(R))
#exit()
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
# checking something here
U1,S1,Vh1 = np.linalg.svd(Q,full_matrices=False)
Q1 = ((U1*S1)[:,0:r])@Vh1[0:r,:]
plt.figure(23842439)
plt.plot(Q[3506,:],label='True')
#for i in range(M):
#    Q[:,i] = Q[:,i]*np.sqrt(R)
#Q = Q*np.sqrt(dR*dphi*dZ)
U2,S2,Vh2 = np.linalg.svd(Q,full_matrices=False)
Q2 = ((U2*S2)[:,0:r])@Vh2[0:r,:]
w2,v3 = eig(inner_prod)
Vh3 = np.transpose(v3)
wr = np.sqrt(np.diag(w2))
U3 = Q@np.transpose(Vh3)@np.linalg.inv(wr)
#V3,S3,Vh3 = np.linalg.svd(inner_prod,full_matrices=False)
#Q3 = ((U2*S2)[:,:])@Vh3[:,:]
Q4 = U3@wr@Vh3
#for i in range(M):
#    Q2[:,i] = Q2[:,i]/np.sqrt(R)
#    Q3[:,i] = Q3[:,i]/np.sqrt(R)
#Q2 = Q2/np.sqrt(dR*dphi*dZ)
#Q3 = Q3/np.sqrt(dR*dphi*dZ)
#plt.plot(Q3[3506,:],label='using method of snapshots on X*X')
plt.plot(Q4[3506,:],label='correctly using method of snapshots on X*X')
plt.legend()
# Perform the vector POD method
#QQ = np.zeros(np.shape(Q))
#for i in range(M):
#    QQ[:,i] = Q[:,i]*np.sqrt(R)
#QQ = QQ*np.sqrt(dR*dphi*dZ)
t_test,x_true,x_sim,S2 = compressible_Framework(Q,inner_prod,time,poly_order,threshold,r)
M_test = len(t_test)
M_train = int(len(time)*tfac)
Qorig = Q[:,M_train:]
Vh_true = np.transpose(x_true)
Vh_sim = np.transpose(x_sim)
Sr = np.sqrt(S2[0:r,0:r])
U_true = Q@np.transpose(Vh3)@np.linalg.inv(wr)
U_sim = Q@np.transpose(Vh3)@np.linalg.inv(wr)
Q_true = U_true[:,0:r]@wr[0:r,0:r]@Vh_true
Q_sim = U_sim[:,0:r]@wr[0:r,0:r]@Vh_sim
# need to put in the radial weights
Q = Q*np.sqrt(dR*dphi*dZ)
for i in range(M):
    Q[:,i] = Q[:,i]*np.sqrt(R)
U,S,Vh = np.linalg.svd(Q,full_matrices=False)
Q = ((U*S)[:,0:r])@Vh[0:r,M_train:]
# now getting rid of the radial weights
Q = Q/np.sqrt(dR*dphi*dZ)
#Q_true = Q_true/np.sqrt(dR*dphi*dZ)
#Q_sim = Q_sim/np.sqrt(dR*dphi*dZ)
#Q = Q/(dR*dphi*dZ)
#Q_true = Q_true/(dR*dphi*dZ)
#Q_sim = Q_sim/(dR*dphi*dZ)
for i in range(M_test):
    Q[:,i] = Q[:,i]/np.sqrt(R)
    #Q_true[:,i] = Q_true[:,i]/np.sqrt(R)
    #Q_sim[:,i] = Q_sim[:,i]/np.sqrt(R)
    #Q[:,i] = Q[:,i]/R
    #Q_true[:,i] = Q_true[:,i]/R
    #Q_sim[:,i] = Q_sim[:,i]/R
print('% Field = ',sum(S[0:r])/sum(S))
#plot_BOD_Fspectrum(S)
print(np.shape(Vh),np.shape(Q))
plot_measurement(Qorig,Q,Q_true,Q_sim,t_test,r)
Bx_mat_sim = Q_sim[0:1*n_samples,:]
By_mat_sim = Q_sim[1*n_samples:2*n_samples,:]
Bz_mat_sim = Q_sim[2*n_samples:3*n_samples,:]
Vx_mat_sim = Q_sim[3*n_samples:4*n_samples,:]
Vy_mat_sim = Q_sim[4*n_samples:5*n_samples,:]
Vz_mat_sim = Q_sim[5*n_samples:6*n_samples,:]
print(Bx_mat_sim)
make_contour_movie(X,Y,Z,Bx_mat[:,M_train:],np.real(Bx_mat_sim),t_test,'Bx')
make_contour_movie(X,Y,Z,By_mat[:,M_train:],np.real(By_mat_sim),t_test,'By')
make_contour_movie(X,Y,Z,Bz_mat[:,M_train:],np.real(Bz_mat_sim),t_test,'Bz')
make_contour_movie(X,Y,Z,Vx_mat[:,M_train:],np.real(Vx_mat_sim),t_test,'Vx')
make_contour_movie(X,Y,Z,Vy_mat[:,M_train:],np.real(Vy_mat_sim),t_test,'Vy')
make_contour_movie(X,Y,Z,Vz_mat[:,M_train:],np.real(Vz_mat_sim),t_test,'Vz')
plt.show()
