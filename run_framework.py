import numpy as np
from sindy_utils import inner_product, \
    plot_measurement, make_toroidal_movie, \
    plot_pod_spatial_modes, \
    plot_density, make_poloidal_movie
from load_data import \
    load_incompressible_data, load_compressible_data
from compressible_Framework import compressible_Framework
from scipy.linalg import eig
from dmd import dmd
"""
    Author: Alan Kaptanoglu

    This is the code used for the analysis in
    https://arxiv.org/pdf/2004.10389.pdf

    Copyright: MIT License

    Description:

    - This code loads in a set of velocity and magnetic field data
      from a uniformly sampled cylindrical mesh, using the NIMROD MHD code.
    - The MHD model used here is an isothermal Hall-MHD model with
      density, velocity, and magnetic field evolution.
    - A set of POD modes are calculated on a part of the data designated
      for training, and then fit with a SINDy model. We search for a system
      of energy-constrained quadratic nonlinear ODEs.
    - The identified model forecasting performance is tested on the remaining
      part of the same time series data, and compared with the ground truth,
      a POD reconstruction of the ground truth, and a DMD model.

    Parameters for simulation 1 in Galerkin paper:
        r = 7
        threshold = 0.05
        start = 150000
        end = 399000
        tfac = 8.0/10.0
    Parameters for simulation 2 in Galerkin paper:
        r = 8
        start = 80000
        end = 270000
        tfac = 6.0/10.0

    Data is available on request.

    Potential issue:
        quadratic terms will tend to have large coeffficients because
        of normalizing the dynamics to be on the unit ball. Have been dealing
        with this by adjusting the thresholds in the PySINDy solvers
"""

# set some flags
is_incompressible = False
do_pareto = False
do_contouranimations = True
do_manifoldplots = True

# set some of the model and SINDy hyperparameters
if is_incompressible:
    r = 12  # Truncation number for the SVD
    threshold = 0.001  # Threshold for SINDy algorithm
    poly_order = 2  # Maximum polynomial order to use in the SINDy library
    start = 10000  # Start time index
    end = 34000  # End time index
    skip = 10  # Dump files are written every 10 simulation steps
    path_plus_prefix = '../HITSI_rMHD_HR'  # path for the data read
    time = np.loadtxt('../time.csv')  # load in the corresponding times
else:
    r = 8
    poly_order = 2
    threshold = 0.05
    start = 80000
    end = 270000
    skip = 100
    tfac = 6.0/10.0
    path_plus_prefix = '../compressible2/HITSI_rMHD_HR'
    time = np.loadtxt('../compressible2/time.csv')

# Define and load in the measurement time sampling and mesh data
spacing = float(skip)
time = time[int(start/spacing):int(end/spacing):int(skip/spacing)]*1e6
M = len(time)
X, Y, Z = np.loadtxt(path_plus_prefix+'_00000.csv',
                     delimiter=',', usecols=(0, 1, 2),
                     unpack=True)
X = X*100
Y = Y*100
Z = Z*100
R = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)
dR = np.unique(R.round(decimals=4))[1]
dphi = np.unique(phi.round(decimals=4))[1] - np.unique(
       phi.round(decimals=4))[0]
dZ = np.unique(Z)[1] - np.unique(Z)[0]

# Load in all the field data and compute <Q,Q> on the mesh
if is_incompressible:
    Bx_mat, By_mat, Bz_mat, Vx_mat, Vy_mat, Vz_mat = \
        load_incompressible_data(start, end, skip, path_plus_prefix)
else:
    Bx_mat, By_mat, Bz_mat, Vx_mat, Vy_mat, Vz_mat, dens = \
        load_compressible_data(start, end, skip, path_plus_prefix)
    plot_density(time, dens)
    sample_skip = 1  # increase for sparser sampling
    Bx_mat = Bx_mat[::sample_skip, :]
    By_mat = By_mat[::sample_skip, :]
    Bz_mat = Bz_mat[::sample_skip, :]
    Vx_mat = Vx_mat[::sample_skip, :]
    Vy_mat = Vy_mat[::sample_skip, :]
    Vz_mat = Vz_mat[::sample_skip, :]
    X = X[::sample_skip]
    Y = Y[::sample_skip]
    R = R[::sample_skip]
    Z = Z[::sample_skip]
    phi = phi[::sample_skip]
n_samples = np.shape(Bx_mat)[0]
Q = np.vstack((Bx_mat, By_mat))
Q = np.vstack((Q, Bz_mat))
Q = np.vstack((Q, Vx_mat))
Q = np.vstack((Q, Vy_mat))
Q = np.vstack((Q, Vz_mat))
R = np.ravel([R, R, R, R, R, R])
inner_prod = inner_product(Q, R)*dR*dphi*dZ
Q = Q*1e4  # convert from Tesla to Gauss

# This generates a pareto landscape in (r, lambda), which is a useful
# way to visualize the space of possible SINDy models.
if do_pareto:
    for rr in np.arange(3, 6):
        compressible_Framework(inner_prod, time, poly_order,
                               threshold, rr, tfac,
                               do_manifoldplots, True)

# Now that data is organized, hand it to the SINDy framework for modeling
t_test, x_true, x_sim, S2, x_train_SINDy = compressible_Framework(
                                            inner_prod, time, poly_order,
                                            threshold, r, tfac,
                                            do_manifoldplots, False)

# Define portions of the data for testing/training
M_test = len(t_test)
M_train = int(len(time)*tfac)
t_train = time[:M_train]
Q_test = Q[:, M_train:]
Vh_true = np.transpose(x_true)
Vh_sim = np.transpose(x_sim)
Sr = np.sqrt(S2[0:r, 0:r])
w, v = eig(inner_prod)
Vh = np.transpose(v)
wr = np.sqrt(np.diag(w))

# Compute the POD and SINDy reconstructions of Q
U = Q@(np.transpose(Vh)@
       (np.linalg.inv(wr)[:, 0:12]))  # our full data requires ~ few GB memory
plot_pod_spatial_modes(X, Y, Z, U)
U_true = U[:, 0:r]
Q_pod = U_true@wr[0:r, 0:r]@Vh_true
Q_sim = U_true@wr[0:r, 0:r]@Vh_sim
dmd_Q = dmd(Q, 16, time, M_train)

# plot probe reconstruction and contour animations
plot_measurement(Q_test, Q_pod, Q_sim, t_test, r)
if do_contouranimations:
    Q_pod = Q_pod/1.0e4
    Q_sim = Q_sim/1.0e4
    Bx_mat_sim = Q_sim[0:1*n_samples, :]
    By_mat_sim = Q_sim[1*n_samples:2*n_samples, :]
    Bz_mat_sim = Q_sim[2*n_samples:3*n_samples, :]
    Vx_mat_sim = Q_sim[3*n_samples:4*n_samples, :]
    Vy_mat_sim = Q_sim[4*n_samples:5*n_samples, :]
    Vz_mat_sim = Q_sim[5*n_samples:6*n_samples, :]
    Bx_mat_pod = Q_pod[0:1*n_samples, :]
    By_mat_pod = Q_pod[1*n_samples:2*n_samples, :]
    Bz_mat_pod = Q_pod[2*n_samples:3*n_samples, :]
    Vx_mat_pod = Q_pod[3*n_samples:4*n_samples, :]
    Vy_mat_pod = Q_pod[4*n_samples:5*n_samples, :]
    Vz_mat_pod = Q_pod[5*n_samples:6*n_samples, :]
    make_toroidal_movie(X, Y, Z, Bx_mat[:, M_train:],
                        np.real(Bx_mat_pod), np.real(Bx_mat_sim),
                        t_test, 'Bx')
    make_toroidal_movie(X, Y, Z, By_mat[:, M_train:],
                        np.real(By_mat_pod), np.real(By_mat_sim),
                        t_test, 'By')
    make_toroidal_movie(X, Y, Z, Bz_mat[:, M_train:],
                        np.real(Bz_mat_pod), np.real(Bz_mat_sim),
                        t_test, 'Bz')
    make_toroidal_movie(X, Y, Z, Vx_mat[:, M_train:],
                        np.real(Vx_mat_pod), np.real(Vx_mat_sim),
                        t_test, 'Bvx')
    make_toroidal_movie(X, Y, Z, Vy_mat[:, M_train:],
                        np.real(Vy_mat_pod), np.real(Vy_mat_sim),
                        t_test, 'Bvy')
    make_toroidal_movie(X, Y, Z, Vz_mat[:, M_train:],
                        np.real(Vz_mat_pod), np.real(Vz_mat_sim),
                        t_test, 'Bvz')

    # repeat for DMD reconstructions
    make_toroidal_movie(X, Y, Z, Bx_mat[:, M_train:],
                        np.real(dmd_Q[0*n_samples:1*n_samples, :]/1e4),
                        np.real(Q_sim[0*n_samples:1*n_samples, :]/1e4),
                        t_test, 'Bx')
    make_toroidal_movie(X, Y, Z, By_mat[:, M_train:],
                        np.real(dmd_Q[1*n_samples:2*n_samples, :]/1e4),
                        np.real(Q_sim[1*n_samples:2*n_samples, :]/1e4),
                        t_test, 'By')
    make_toroidal_movie(X, Y, Z, Bz_mat[:, M_train:],
                        np.real(dmd_Q[2*n_samples:3*n_samples, :]/1e4),
                        np.real(Q_sim[2*n_samples:3*n_samples, :]/1e4),
                        t_test, 'Bz')
    make_toroidal_movie(X, Y, Z, Vx_mat[:, M_train:],
                        np.real(dmd_Q[3*n_samples:4*n_samples, :]/1e4),
                        np.real(Q_sim[3*n_samples:4*n_samples, :]/1e4),
                        t_test, 'Bvx')
    make_toroidal_movie(X, Y, Z, Vy_mat[:, M_train:],
                        np.real(dmd_Q[4*n_samples:5*n_samples, :]/1e4),
                        np.real(Q_sim[4*n_samples:5*n_samples, :]/1e4),
                        t_test, 'Bvy')
    make_toroidal_movie(X, Y, Z, Vz_mat[:, M_train:],
                        np.real(dmd_Q[5*n_samples:6*n_samples, :]/1e4),
                        np.real(Q_sim[5*n_samples:6*n_samples, :]/1e4),
                        t_test, 'Bvz')

    # Repeat for poloidal contour movies
    make_poloidal_movie(X, Y, Z, Bx_mat[:, M_train:],
                        np.real(Bx_mat_pod), np.real(Bx_mat_sim),
                        t_test, 'Bx')
    make_poloidal_movie(X, Y, Z, By_mat[:, M_train:],
                        np.real(By_mat_pod), np.real(By_mat_sim),
                        t_test, 'By')
    make_poloidal_movie(X, Y, Z, Bz_mat[:, M_train:],
                        np.real(Bz_mat_pod), np.real(Bz_mat_sim),
                        t_test, 'Bz')
    make_poloidal_movie(X, Y, Z, Vx_mat[:, M_train:],
                        np.real(Vx_mat_pod), np.real(Vx_mat_sim),
                        t_test, 'Bvx')
    make_poloidal_movie(X, Y, Z, Vy_mat[:, M_train:],
                        np.real(Vy_mat_pod), np.real(Vy_mat_sim),
                        t_test, 'Bvy')
    make_poloidal_movie(X, Y, Z, Vz_mat[:, M_train:],
                        np.real(Vz_mat_pod), np.real(Vz_mat_sim),
                        t_test, 'Bvz')

    # Repeat for DMD
    make_poloidal_movie(X, Y, Z, Bx_mat[:, M_train:],
                        np.real(dmd_Q[0*n_samples:1*n_samples, :]/1e4),
                        np.real(Q_sim[0*n_samples:1*n_samples, :]/1e4),
                        t_test, 'Bx')
    make_poloidal_movie(X, Y, Z, By_mat[:, M_train:],
                        np.real(dmd_Q[1*n_samples:2*n_samples, :]/1e4),
                        np.real(Q_sim[1*n_samples:2*n_samples, :]/1e4),
                        t_test, 'By')
    make_poloidal_movie(X, Y, Z, Bz_mat[:, M_train:],
                        np.real(dmd_Q[2*n_samples:3*n_samples, :]/1e4),
                        np.real(Q_sim[2*n_samples:3*n_samples, :]/1e4),
                        t_test, 'Bz')
    make_poloidal_movie(X, Y, Z, Vx_mat[:, M_train:],
                        np.real(dmd_Q[3*n_samples:4*n_samples, :]/1e4),
                        np.real(Q_sim[3*n_samples:4*n_samples, :]/1e4),
                        t_test, 'Bvx')
    make_poloidal_movie(X, Y, Z, Vy_mat[:, M_train:],
                        np.real(dmd_Q[4*n_samples:5*n_samples, :]/1e4),
                        np.real(Q_sim[4*n_samples:5*n_samples, :]/1e4),
                        t_test, 'Bvy')
    make_poloidal_movie(X, Y, Z, Vz_mat[:, M_train:],
                        np.real(dmd_Q[5*n_samples:6*n_samples, :]/1e4),
                        np.real(Q_sim[5*n_samples:6*n_samples, :]/1e4),
                        t_test, 'Bvz')
