import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from sindy_utils import inner_product, \
    plot_measurement, make_contour_movie, \
    plot_pod_spatial_modes, \
    plot_density, make_poloidal_movie
from load_data import \
    load_incompressible_data, load_compressible_data
from compressible_Framework import compressible_Framework
from scipy.linalg import eig
from spod_plotting import plot_spod
from dmd import dmd, compare_SINDy_DMD

is_incompressible = False
if is_incompressible:
    # Truncation number for the SVD
    r = 12
    # Threshold for SINDy algorithm
    # There is a potential issue here, which is that quadratic
    # terms will tend to have large coeffficients because
    # of normalizing the dynamics to be on the unit ball. Have been dealing
    # with this by adjusting the thresholds in the PySINDy solvers
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
    make_3Dphaseplots = False
else: 
    # for run 2
    #r = 8 
    #start = 80000
    #end = 270000
    #tfac = 6.0/10.0
    # for run 1
    #r = 7 
    #threshold = 0.05
    #start = 150000
    #end = 399000
    #tfac = 8.0/10.0
    r = 8
    poly_order = 2
    threshold = 0.05
    start = 80000
    end = 270000    
    skip = 100
    tfac = 8.0/10.0
    path_plus_prefix = '../compressible2/HITSI_rMHD_HR'
    time = np.loadtxt('../compressible2/time.csv')
    make_3Dphaseplots = True
contour_animations = False
spacing = float(skip)
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
    plot_density(time,dens)
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
inner_prod = inner_product(Q,R)*dR*dphi*dZ
Q = Q*1e4
for rr in np.arange(3,20):
    compressible_Framework(inner_prod,time,poly_order,threshold,rr,tfac,make_3Dphaseplots)
exit()
t_test,x_true,x_sim,S2,x_train_SINDy = \
    compressible_Framework(inner_prod,time,poly_order,threshold,r,tfac,False)
M_test = len(t_test)
M_train = int(len(time)*tfac)
t_train = time[:M_train]
#Q_test = Q[:,:M_train]
Q_test = Q[:,M_train:]
Vh_true = np.transpose(x_true)
Vh_sim = np.transpose(x_sim)
Sr = np.sqrt(S2[0:r,0:r])
w,v = eig(inner_prod)
Vh = np.transpose(v)
wr = np.sqrt(np.diag(w))
# This will crash on some computers... a few GB of memory 
# if I use the full spatio-temporal data
U = Q@(np.transpose(Vh)@(np.linalg.inv(wr)[:,0:12]))
plot_pod_spatial_modes(X,Y,Z,U)
#plot_spod(time,Q[0:n_samples,:])
#exit()
U_true = U[:,0:r]
U_sim = U_true
#Q_pod = U_true@wr[0:r,0:r]@Vh_true
Q_sim = U_sim@wr[0:r,0:r]@Vh_sim
#Qsize = int(np.shape(U_sim)[0]/6)
#Q_train_SINDy = U_sim[324::Qsize,:]@wr[0:r,0:r]@np.transpose(x_train_SINDy)
dmd_Q = dmd(Q,100,time,M_train)
#compare_SINDy_DMD(t_train,t_test,Q[324::Qsize,:M_train],Q_train_SINDy,dmd_Q[:,:M_train],Q_test[324::Qsize,:],Q_sim[324::Qsize,:],dmd_Q[:,M_train:])
#print(np.shape(dmd_Q),print(dmd_Q))
#R = np.sqrt(X**2+Y**2)
#Z0 = np.isclose(Z,np.ones(len(Z))*min(abs(Z)),rtol=1e-3,atol=1e-3)
#ind_Z0 = [i for i, p in enumerate(Z0) if p]
#Q_Z0 = Q[ind_Z0,:]
#SINDy_Q_Z0 = U_sim[ind_Z0,:]@wr[0:r,0:r]@np.transpose(x_sim)
#DMD_Q_Z0 = dmd_Z0(Q,16,time,M_train)
make_contour_movie(X,Y,Z,Bx_mat[:,M_train:],np.real(dmd_Q[0*n_samples:1*n_samples,:]/1e4),np.real(Q_sim[0*n_samples:1*n_samples,:]/1e4
),t_test,'Bx')
make_contour_movie(X,Y,Z,By_mat[:,M_train:],np.real(dmd_Q[1*n_samples:2*n_samples,:]/1e4),np.real(Q_sim[1*n_samples:2*n_samples,:]/1e4
),t_test,'By')
make_contour_movie(X,Y,Z,Bz_mat[:,M_train:],np.real(dmd_Q[2*n_samples:3*n_samples,:]/1e4),np.real(Q_sim[2*n_samples:3*n_samples,:]/1e4
),t_test,'Bz')
make_contour_movie(X,Y,Z,Vx_mat[:,M_train:],np.real(dmd_Q[3*n_samples:4*n_samples,:]/1e4),np.real(Q_sim[3*n_samples:4*n_samples,:]/1e4
),t_test,'Bvx')
make_contour_movie(X,Y,Z,Vy_mat[:,M_train:],np.real(dmd_Q[4*n_samples:5*n_samples,:]/1e4),np.real(Q_sim[4*n_samples:5*n_samples,:]/1e4
),t_test,'Bvy')
make_contour_movie(X,Y,Z,Vz_mat[:,M_train:],np.real(dmd_Q[5*n_samples:6*n_samples,:]/1e4),np.real(Q_sim[5*n_samples:6*n_samples,:]/1e4
),t_test,'Bvz')
exit()
plot_measurement(Q_test,Q_pod,Q_sim,t_test,r)
if contour_animations:
    Q_pod = Q_pod/1.0e4
    Q_sim = Q_sim/1.0e4
    Bx_mat_sim = Q_sim[0:1*n_samples,:]
    By_mat_sim = Q_sim[1*n_samples:2*n_samples,:]
    Bz_mat_sim = Q_sim[2*n_samples:3*n_samples,:]
    Vx_mat_sim = Q_sim[3*n_samples:4*n_samples,:]
    Vy_mat_sim = Q_sim[4*n_samples:5*n_samples,:]
    Vz_mat_sim = Q_sim[5*n_samples:6*n_samples,:]
    Bx_mat_pod = Q_pod[0:1*n_samples,:]
    By_mat_pod = Q_pod[1*n_samples:2*n_samples,:]
    Bz_mat_pod = Q_pod[2*n_samples:3*n_samples,:]
    Vx_mat_pod = Q_pod[3*n_samples:4*n_samples,:]
    Vy_mat_pod = Q_pod[4*n_samples:5*n_samples,:]
    Vz_mat_pod = Q_pod[5*n_samples:6*n_samples,:]
    make_contour_movie(X,Y,Z,Bx_mat[:,M_train:],np.real(Bx_mat_pod),np.real(Bx_mat_sim),t_test,'Bx')
    make_contour_movie(X,Y,Z,By_mat[:,M_train:],np.real(By_mat_pod),np.real(By_mat_sim),t_test,'By')
    make_contour_movie(X,Y,Z,Bz_mat[:,M_train:],np.real(Bz_mat_pod),np.real(Bz_mat_sim),t_test,'Bz')
    make_contour_movie(X,Y,Z,Vx_mat[:,M_train:],np.real(Vx_mat_pod),np.real(Vx_mat_sim),t_test,'Bvx')
    make_contour_movie(X,Y,Z,Vy_mat[:,M_train:],np.real(Vy_mat_pod),np.real(Vy_mat_sim),t_test,'Bvy')
    make_contour_movie(X,Y,Z,Vz_mat[:,M_train:],np.real(Vz_mat_pod),np.real(Vz_mat_sim),t_test,'Bvz')
    make_poloidal_movie(X,Y,Z,Bx_mat[:,M_train:],np.real(Bx_mat_pod),np.real(Bx_mat_sim),t_test,'Bx')
    make_poloidal_movie(X,Y,Z,By_mat[:,M_train:],np.real(By_mat_pod),np.real(By_mat_sim),t_test,'By')
    make_poloidal_movie(X,Y,Z,Bz_mat[:,M_train:],np.real(Bz_mat_pod),np.real(Bz_mat_sim),t_test,'Bz')
    make_poloidal_movie(X,Y,Z,Vx_mat[:,M_train:],np.real(Vx_mat_pod),np.real(Vx_mat_sim),t_test,'Bvx')
    make_poloidal_movie(X,Y,Z,Vy_mat[:,M_train:],np.real(Vy_mat_pod),np.real(Vy_mat_sim),t_test,'Bvy')
    make_poloidal_movie(X,Y,Z,Vz_mat[:,M_train:],np.real(Vz_mat_pod),np.real(Vz_mat_sim),t_test,'Bvz')
