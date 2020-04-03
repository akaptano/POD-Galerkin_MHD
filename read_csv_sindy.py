import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from sindy_utils import inner_product, \
    plot_measurement, make_contour_movie, \
    plot_Hc, plot_energy, \
    plot_pod_spatial_modes, \
    plot_density
from load_incompressible_data import \
    load_incompressible_data, load_compressible_data
from compressible_Framework import compressible_Framework
from scipy.linalg import eig
mu0 = 4*pi*1e-7
rho = 2.0e19*2*1.67*1e-27

is_incompressible = False
if is_incompressible:
    # Truncation number for the SVD
    r = 12
    # Threshold for SINDy algorithm
    # There is an issue here, which is that quadratic
    # terms will tend to have large coeffficients because
    # of normalizing the dynamics to be on the unit ball
    threshold = 0.001
    # Maximum polynomial order to use in the SINDy library
    poly_order = 2
    # Start time index
    start = 10000
    # End time index
    end = 34000
    # Dump files are written every 10 simulation steps
    skip = 10
    path_plus_prefix = '../HITSI_rMHD_HR'
    time = np.loadtxt('../time.csv')
else: 
    r = 11 
    poly_order = 2
    threshold = 0.05
    start = 150000
    end = 370000
    skip = 100
    path_plus_prefix = '../compressible1/HITSI_rMHD_HR'
    time = np.loadtxt('../compressible1/time.csv')
contour_animations = False 
spacing = float(skip)
# Fraction of the data to use as training data
tfac = 9.0/10.0
# shorten and put into microseconds
time = time[int(start/spacing):int(end/spacing):int(skip/spacing)]*1e6
M = len(time)
# Load in only the measurement coordinates for now
X, Y, Z = np.loadtxt(path_plus_prefix+'_00000.csv', \
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
# Load in all the field data
if is_incompressible:
    Bx_mat,By_mat,Bz_mat,Vx_mat,Vy_mat,Vz_mat = \
        load_incompressible_data(start,end,skip,path_plus_prefix)
else:
    Bx_mat,By_mat,Bz_mat,Vx_mat,Vy_mat,Vz_mat,dens = \
        load_compressible_data(start,end,skip,path_plus_prefix)
    #plot_density(time,dens)
    sample_skip = 1
    Bx_mat = Bx_mat[::sample_skip,:]
    By_mat = By_mat[::sample_skip,:]
    Bz_mat = Bz_mat[::sample_skip,:]
    Vx_mat = Vx_mat[::sample_skip,:]
    Vy_mat = Vy_mat[::sample_skip,:]
    Vz_mat = Vz_mat[::sample_skip,:]
    X = X[::sample_skip]
    Y = Y[::sample_skip]
    R = R[::sample_skip]
    Z = Z[::sample_skip]
    phi = phi[::sample_skip]
n_samples = np.shape(Bx_mat)[0]
Q = np.vstack((Bx_mat,By_mat))
Q = np.vstack((Q,Bz_mat))
Q = np.vstack((Q,Vx_mat))
Q = np.vstack((Q,Vy_mat))
Q = np.vstack((Q,Vz_mat))
R = np.ravel([R,R,R,R,R,R])
print('D = ',n_samples)
#energy_mat = np.sqrt(0.5*rho)*Q*np.sqrt(dR*dphi*dZ)/1.0e3
#plot_energy(time,energy_mat,R/1.0e2)
#Hc_mat = np.sqrt(rho)*Q*np.sqrt(dR*dphi*dZ)/1.0e3
#plot_Hc(time,Hc_mat,R/1.0e2)
inner_prod = inner_product(Q,R)*dR*dphi*dZ
#Hc = np.sqrt(mion/mu0)*(Bx_mat*Vx_mat + By_mat*Vy_mat + Bz_mat*Vz_mat)*dR*dphi*dZ
Q = Q*1e4
t_test,x_true,x_sim,S2 = \
    compressible_Framework(Q,inner_prod,time,poly_order,threshold,r,tfac)
    #compressible_Framework(Q,inner_prod,time,poly_order,threshold,r,tfac)
M_test = len(t_test)
M_train = int(len(time)*tfac)
Qorig = Q[:,M_train:]
Vh_true = np.transpose(x_true)
Vh_sim = np.transpose(x_sim)
Sr = np.sqrt(S2[0:r,0:r])
#wr = np.real(wr)
#Vh3 = np.real(Vh3)
#Q = np.real(Q)
#U,wr,Vh = np.linalg.svd(Q,full_matrices=False)
#U_true = U[:,0:r]
#U_sim = U_true
w,v = eig(inner_prod)
Vh = np.transpose(v)
wr = np.sqrt(np.diag(w))
#wr = np.diag(wr)
U = Q@np.transpose(Vh)@(np.linalg.inv(wr)[:,0:12])
#plot_pod_spatial_modes(X,Y,Z,Q,Vh,wr,U,time) #,'Bx')
#plot_pod_spatial_modes(X,Y,Z,Q,Vh,wr,U[1*n_samples:2*n_samples,:],time,'By')
#plot_pod_spatial_modes(X,Y,Z,Q,Vh,wr,U[2*n_samples:3*n_samples,:],time,'Bz')
#plot_pod_spatial_modes(X,Y,Z,Q,Vh,wr,U[3*n_samples:4*n_samples,:],time,'Vx')
#plot_pod_spatial_modes(X,Y,Z,Q,Vh,wr,U[4*n_samples:5*n_samples,:],time,'Vy')
#plot_pod_spatial_modes(X,Y,Z,Q,Vh,wr,U[5*n_samples:6*n_samples,:],time,'Vz')
U_true = U[:,0:r] #Q@np.transpose(Vh)@(np.linalg.inv(wr)[:,0:r])
U_sim = U_true #Q@np.transpose(Vh)@(np.linalg.inv(wr)[:,0:r])
Q_true = U_true@wr[0:r,0:r]@Vh_true
Q_sim = U_sim@wr[0:r,0:r]@Vh_sim
plot_measurement(Qorig,Q_true,Q_sim,t_test,r)
if contour_animations:
    Q_true = Q_true/1.0e4
    Q_sim = Q_sim/1.0e4
    Bx_mat_sim = Q_sim[0:1*n_samples,:]
    By_mat_sim = Q_sim[1*n_samples:2*n_samples,:]
    Bz_mat_sim = Q_sim[2*n_samples:3*n_samples,:]
    Vx_mat_sim = Q_sim[3*n_samples:4*n_samples,:]
    Vy_mat_sim = Q_sim[4*n_samples:5*n_samples,:]
    Vz_mat_sim = Q_sim[5*n_samples:6*n_samples,:]
    Bx_mat_pod = Q_true[0:1*n_samples,:]
    By_mat_pod = Q_true[1*n_samples:2*n_samples,:]
    Bz_mat_pod = Q_true[2*n_samples:3*n_samples,:]
    Vx_mat_pod = Q_true[3*n_samples:4*n_samples,:]
    Vy_mat_pod = Q_true[4*n_samples:5*n_samples,:]
    Vz_mat_pod = Q_true[5*n_samples:6*n_samples,:]
    make_contour_movie(X,Y,Z,Bx_mat[:,M_train:],np.real(Bx_mat_pod),np.real(Bx_mat_sim),t_test,'Bx')
    make_contour_movie(X,Y,Z,By_mat[:,M_train:],np.real(By_mat_pod),np.real(By_mat_sim),t_test,'By')
    make_contour_movie(X,Y,Z,Bz_mat[:,M_train:],np.real(Bz_mat_pod),np.real(Bz_mat_sim),t_test,'Bz')
    make_contour_movie(X,Y,Z,Vx_mat[:,M_train:],np.real(Vx_mat_pod),np.real(Vx_mat_sim),t_test,'Bvx')
    make_contour_movie(X,Y,Z,Vy_mat[:,M_train:],np.real(Vy_mat_pod),np.real(Vy_mat_sim),t_test,'Bvy')
    make_contour_movie(X,Y,Z,Vz_mat[:,M_train:],np.real(Vz_mat_pod),np.real(Vz_mat_sim),t_test,'Bvz')
    #plt.show()
