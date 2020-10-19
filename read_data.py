import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from sindy_utils import inner_product, \
    plot_measurement, make_contour_movie, \
    plot_pod_spatial_modes, \
    plot_density
from load_incompressible_data import \
    load_incompressible_data, load_compressible_data
from compressible_Framework import compressible_Framework
from scipy.linalg import eig
from importlib import reload
import pysindy.optimizers
reload(pysindy.optimizers)
from pysindy.optimizers import *
mu0 = 4*pi*1e-7
rho = 2.0e19*2*1.67*1e-27

def read_data():
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
        r = 8 
        poly_order = 2
        threshold = 0.05
        start = 150000
        end = 399900
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
    inner_prod = inner_product(Q,R)*dR*dphi*dZ
    #Q = Q*1e4
    return inner_prod,time,poly_order,threshold,r,tfac
    #t_test,x_true,x_sim,S2 = \
    #compressible_Framework(Q,inner_prod,time,poly_order,threshold,r,tfac)
