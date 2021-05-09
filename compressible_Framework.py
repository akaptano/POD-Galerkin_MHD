import numpy as np
from pysindy import SINDy
from pysindy.feature_library import CustomLibrary
from pysindy.differentiation import FiniteDifference
from sindy_utils import *
from scipy.integrate import odeint
from scipy.linalg import eig
from pysindy.optimizers import ConstrainedSR3


def compressible_Framework(inner_prod, time, poly_order, threshold,
                           r, tfac, do_manifoldplots):
    """
    Performs the entire vector_POD + SINDy framework for a given polynomial
    order and thresholding for the SINDy method.

    Parameters
    ----------

    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The scaled matrix of inner products X*X

    time: numpy array of floats
    (M = number of time samples)
        Time in microseconds

    poly_order: int
    (1)
        Highest polynomial order to use in the SINDy library

    threshold: float
    (1)
        Threshold in the SINDy algorithm, below which coefficients
        will be zeroed out.

    r: int
    (1)
        Truncation number of the SVD

    tfac: float
    (1)
        Fraction of the data to treat as training data

    do_manifoldplots: bool
    (1)
        Flag to make 3D plots in the POD state space or not

    Returns
    -------

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    x_true: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The true evolution of the temporal BOD modes

    x_sim: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The model evolution of the temporal BOD modes

    S2: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The singular value matrix

    """

    # Define a portion of the data for training and for testing
    M_train = int(len(time) * tfac)
    t_train = time[:M_train]
    t_test = time[M_train:]

    # Compute the dimensionalize POD to get the temporal POD modes in x
    x, feature_names, S2, Vh, = vector_POD(inner_prod, time, r)

    # Depending on poly_order, define a custom library of functions and
    # associated function names. This is necessary because we are excluding
    # the constant term, and because the order of the quadratic library
    # terms needs to be a_1a_2,...,a_{r-1}a_r, a_1^2, ..., a_r^2 for the
    # constraint indexing to work.
    if poly_order == 1:
        library_functions = [lambda x:x]
        library_function_names = [lambda x:x]
    if poly_order == 2:
        library_functions = [lambda x:x, lambda x, y:x * y, lambda x:x ** 2]
        library_function_names = [lambda x:x, lambda x, y:x + y, lambda x:x + x]
    if poly_order == 3:
        library_functions = [lambda x:x, lambda x, y:x * y, lambda x:x ** 2,
                             lambda x, y, z: x * y * z, lambda x, y: x ** 2 * y,
                             lambda x, y: x * y ** 2, lambda x: x ** 3]
        library_function_names = [lambda x:x, lambda x, y:x + y, lambda x:x + x,
                                  lambda x, y, z: x + y + z, lambda x, y: x + x + y,
                                  lambda x, y: x + y + y, lambda x: x + x + x]
    sindy_library = CustomLibrary(
                            library_functions=library_functions,
                            function_names=library_function_names)

    # Now set the constraint on the linear and/or quadratic model terms.
    N = r * (r ** 2 + 3 * r) // 2
    Nc = int(r * (r+1) // 2) + r + r * (r - 1) + int(r * (r - 1) * (r - 2) // 6.0 
    
    # constraint vector below is for a quadratic model constraint
    constraint_zeros = np.zeros(Nc)
    constraint_matrix = np.zeros((Nc, N))

    # Easy addition: the linear model constraint for models with
    #                poly_order > 2 is the same because the linear
    #                coefficients are still "in the same place". Just need
    #                to make constraint_zeros and constraint_matrix
    #                the correct size for the given r and poly_order.

    # Set the diagonal part of the linear coefficient matrix to be zero
    for i in range(r):
        constraint_matrix[i, i * (r + 1)] = 1.0
    q = r

    # Enforce anti-symmetry in the linear coefficient matrix
    for i in range(r):
        counter = 1
        for j in range(i + 1, r):
            constraint_matrix[q, i * r + j] = 1.0
            constraint_matrix[q, i * r + j + counter * (r - 1)] = 1.0
            counter = counter + 1
            q = q + 1

    # quadratic model constraint

    # Set coefficients adorning terms like a_i^3 to zero
    for i in range(r):
        constraint_matrix[q, r * (N - r) + i * (r + 1)] = 1.0
        q = q + 1

    # Set coefficients adorning terms like a_ia_j^2 to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[q, r * (r + j - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
            q = q + 1
    for i in range(r):
        for j in range(i):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[q, r * (r + i - 1) + j + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
            q = q + 1

    # Set coefficients adorning terms like a_ia_ja_k to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            for k in range(j + 1, r):
                constraint_matrix[q, r * (r + k - 1) + i + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
                constraint_matrix[q, r * (r + k - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
                constraint_matrix[q, r * (r + j - 1) + k + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
                q = q + 1

    # split the temporal POD modes into testing and training
    x_train = x[:M_train, :]
    x_test = x[M_train:, :]
    x0_train = x[0, :]
    x_true = x[M_train:, :]
    x0_test = x[M_train, :]

    # Set the SINDy optimizer parameters and define the SINDy model
    if r >= 3:
        linear_r4_mat = np.zeros((r, r + int(r * (r + 1) / 2)))
        linear_r4_mat[0, 1] = 0.091
        linear_r4_mat[1, 0] = -0.091
        linear_r4_mat[2, 3] = 0.182
        linear_r4_mat[3, 2] = -0.182
        linear_r4_mat[5, 4] = -3 * 0.091
        linear_r4_mat[4, 5] = 3 * 0.091
        
        # fully constrained run 1
        sindy_opt = ConstrainedSR3(threshold=threshold, nu=1, max_iter=600,
                                   constraint_lhs=constraint_matrix,
                                   constraint_rhs=constraint_zeros, tol=1e-5,
                                   thresholder='l0',
                                   initial_guess=linear_r4_mat)

        # Multiple-threshold constrained optimizer
        #sindy_opt = ConstrainedSR3(threshold=threshold, nu=10, max_iter=50000,
        #                           constraint_lhs=constraint_matrix,
        #                           constraint_rhs=constraint_zeros, tol=1e-5,
        #                           thresholder='weighted_l0',
        #                           initial_guess=linear_r4_mat,
        #                           thresholds=thresholds)

        # Single threshold constrained optimizer
        # sindy_opt = ConstrainedSR3(threshold=threshold, nu=10,
        #                            max_iter=50000,
        #                            constraint_lhs=constraint_matrix,
        #                            constraint_rhs=constraint_zeros,
        #                            tol=1e-5, thresholder='l0',
        #                            initial_guess=linear_r4_mat)

        # Unconstrained optimizer
        # sindy_opt = ConstrainedSR3(threshold=threshold, nu=10,
        #                           max_iter=50000,
        #                           tol=1e-5, thresholder='weighted_l0',
        #                           initial_guess=linear_r4_mat,
        #                           thresholds=thresholds)

        model = SINDy(optimizer=sindy_opt, feature_library=sindy_library,
                      differentiation_method=FiniteDifference(drop_endpoints=True),
                      feature_names=feature_names)
    else:
        sindy_opt = ConstrainedSR3(threshold=threshold, nu=10, max_iter=50000,
                                   tol=1e-5, thresholder='l0')
        model = SINDy(optimizer=sindy_opt, feature_library=sindy_library,
                      differentiation_method=FiniteDifference(
                                                drop_endpoints=True),
                      feature_names=feature_names)

    # fitting and plotting
    model.fit(x_train, t=t_train, unbias=False)
    print(model.coefficients())
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-20
    integrator_keywords['h0'] = 1e-5
    x_train_SINDy = model.simulate(x[0, :], t_train, integrator=odeint,
                                   stop_condition=None,integrator_kws=integrator_keywords)
    x_sim = model.simulate(x0_test, t_test, integrator=odeint,
                           stop_condition=None,integrator_kws=integrator_keywords)
    x_dot = model.differentiate(x, t=time)
    x_dot_train = model.predict(x_train)
    x_dot_sim = model.predict(x_true)
    print('Model score: %f' % model.score(x, t=time))

    # Plot the comparison between true and predicted POD mode evolutions
    make_evo_plots(x_dot, x_dot_train, x_dot_sim,
                   x_true, x_sim, time, t_train, t_test)

    # Makes 3D plots of the a_i, a_j, a_k state space
    if do_manifoldplots:
        make_3d_plots(x_true, x_sim, t_test, 'sim', 0, 1, 2)
        make_3d_plots(x_true, x_sim, t_test, 'sim', 0, 1, 3)
        make_3d_plots(x_true, x_sim, t_test, 'sim', 0, 1, 4)
        make_3d_plots(x_true, x_sim, t_test, 'sim', 0, 1, 5)
        make_3d_plots(x_true, x_sim, t_test, 'sim', 0, 1, 6)
        make_3d_plots(x_true, x_sim, t_test, 'sim', 3, 4, 5)
        make_3d_plots(x_true, x_sim, t_test, 'sim', 4, 5, 6)

    # Rescale back to original units
    for i in range(r):
        x_sim[:, i] = x_sim[:, i] * sum(np.amax(abs(Vh), axis=1)[0:r])
        x_true[:, i] = x_true[:, i] * sum(np.amax(abs(Vh), axis=1)[0:r])

    return t_test, x_true, x_sim, S2, x_train_SINDy


def vector_POD(inner_prod, t_train, r):
    """
    Performs the vector POD, and scales the resulting modes
    to lie on the unit ball. Also returns the names of the
    temporal modes which will be modeled.

    Parameters
    ----------
    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The scaled matrix of inner products X*X

    t_train: 1D numpy array of floats
    (M_train = number of time samples in the training data)
        The time samples for training

    r: int
    (1)
        The truncation number of the SVD

    Returns
    -------
    x: 2D numpy array of floats
    (M = number of time samples, r = truncation number of the SVD)
        The temporal BOD modes to be modeled, scaled to
        stay on the unit ball

    feature_names: numpy array of strings
    (r = truncation number of the SVD)
        Names of the temporal BOD modes to be modeled

    S2: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The matrix of singular values

    Vh: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The V* in the SVD, returned here because the SINDy modes
        will need to be rescaled off of the unit ball to compare
        with the original measurements

    """

    # Compute eigenvalue decomposition of <Q,Q> and sort by largest eigenvalues
    S2, v = eig(inner_prod)
    idx = S2.argsort()[::-1]
    S2 = S2[idx]
    v = v[:, idx]
    Vh = np.transpose(v)

    # Plot temporal eigenvectors and associated eigenvalues
    plot_pod_temporal_modes(v[:, 0:20], t_train)
    plot_BOD_Espectrum(S2)
    print("% field in first r modes = ",
          sum(np.sqrt(S2[0:r])) / sum(np.sqrt(S2)))
    print("% energy in first r modes = ", sum(S2[0:r]) / sum(S2))

    # Reshape some things and normalize the temporal modes for SINDy
    S2 = np.diag(S2)
    vh = np.zeros((r, np.shape(Vh)[1]))
    feature_names = []
    for i in range(r):
        vh[i, :] = Vh[i, :] / sum(np.amax(abs(Vh), axis=1)[0:r])
        feature_names.append(r'$\varphi_{:d}$'.format(i + 1))
    x = np.transpose(vh)

    return x, feature_names, S2, Vh
