import numpy as np
from numpy import pi
from matplotlib import pyplot as plt


def make_VandermondeT(omega, time):
    """
    Compute the transpose of the Vandermonde
    matrix from the times and frequencies

    Parameters
    ----------

    omega: 1D numpy array of complex values
    (r = truncation rank of the DMD)
        DMD frequencies

    time: 1D numpy array of floats
    (M = number of time samples)
        Time range for the DMD

    Returns
    ----------

    VandermondeT: 2D numpy array of complex values
    (r, M)
        Transpose of the Vandermonde matrix for DMD reconstructions

    """
    VandermondeT = np.exp(np.outer(time, omega))
    return VandermondeT


def dmd(data, r, time, M_train):
    """
    Compute the DMD of a set of time series

    Parameters
    ----------

    data: 2D numpy array of spacetime data
    (N = number of spatial locations, M = number of time samples)
        Data to use the DMD on

    r: integer
        Truncation rank for the DMD

    time: 1D numpy array of floats
    (M = number of time samples)
        Time range for the DMD

    M_train: integer
        The length of the time range for building the DMD model,
        the remainder is used for testing the model.

    Returns
    ----------

    Bfield: 2D numpy array of floats
    (N = number of spatial locations, M = number of time samples)
        DMD reconstruction of the data variable

    """

    tsize = len(time)
    time = time/1e6
    Qsize = int(np.shape(data)[0]/6)
    Bfield = np.zeros((np.shape(data)[0], tsize-M_train), dtype='complex')
    dt = 1e-6
    X = data[:, 0:M_train]
    Xprime = data[:, 1:M_train+1]
    Udmd, Sdmd, Vdmd = np.linalg.svd(X, full_matrices=False)
    Vdmd = np.transpose(Vdmd)
    Udmd = Udmd[:, 0:r]
    Sdmd = Sdmd[0:r]
    Vdmd = Vdmd[:, 0:r]
    S = np.diag(Sdmd)
    A = np.dot(np.dot(np.transpose(Udmd), Xprime), Vdmd/Sdmd)
    eigvals, Y = np.linalg.eig(A)
    Bt = np.dot(np.dot(Xprime, Vdmd/Sdmd), Y)
    omega = np.log(eigvals)/dt
    VandermondeT = make_VandermondeT(omega, time-time[0])
    Vandermonde = np.transpose(VandermondeT)
    q = np.conj(np.diag(np.dot(np.dot(np.dot(
                Vandermonde[:, :M_train], Vdmd), np.conj(S)), Y)))
    P = np.dot(np.conj(np.transpose(Y)), Y)*np.conj(
               np.dot(Vandermonde[:, :M_train], np.conj(
                   VandermondeT[:M_train, :])))
    b = np.dot(np.linalg.inv(P), q)
    c = 0.5*(b + np.conj(b)).real
    c = 500*c/np.max(c)
    energy_sort = np.flip(np.argsort(c))
    plt.figure()
    omega_sizes = c
    for i in range(len(c)):
        omega_sizes[i] = max(omega_sizes[i], 50)
    # plot the frequencies
    plt.scatter(omega.imag/(2*pi*1e3),
                omega.real/(2*pi*1e3), s=omega_sizes, c='k')
    plt.savefig('Pictures/dmd_freqs.pdf')
    for mode in range(r):
        Bfield += 0.5*b[mode]*np.outer(Bt[:, mode],
                                       Vandermonde[mode, M_train:])
    Bfield += np.conj(Bfield)
    return Bfield.real


def compare_SINDy_DMD(t_train, t_test, Q_train, sindy_Q_train,
                      dmd_Q_train, Q_test, sindy_Q_test, dmd_Q_test):
    """
    Plot (Bx,By,Bz,Bvx,Bvy,Bvz) for a random probe measurement,
    compare performance between the ground truth, SINDy model, and DMD model

    Parameters
    ----------

    t_train: 1D numpy array of floats
    (M = number of test data samples)
        Training time window

    t_test: 1D numpy array of floats
    (M = number of test data samples)
       Testing time window

    Q_train: 2D numpy array of floats
    (D = total number of probes x 6, M_train = number of train data samples)
        The ground truth training data

    """
    Qsize = int(np.shape(Q_train)[0]/6)
    plt.figure(figsize=(7, 9))
    plt.subplot(6, 2, 1)
    plt.plot(t_test, Q_test[0, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, sindy_Q_test[0, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6, 2, 3)
    plt.plot(t_test, Q_test[1, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, sindy_Q_test[1, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 5)
    plt.plot(t_test, Q_test[2, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, sindy_Q_test[2, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 7)
    plt.plot(t_test, Q_test[3, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, sindy_Q_test[3, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 9)
    plt.plot(t_test, Q_test[4, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, sindy_Q_test[4, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 11)
    plt.plot(t_test, Q_test[5, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, sindy_Q_test[5, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 2)
    plt.plot(t_test, Q_test[0, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, dmd_Q_test[0, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 4)
    plt.plot(t_test, Q_test[1, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, dmd_Q_test[1, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 6)
    plt.plot(t_test, Q_test[2, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, dmd_Q_test[2, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 8)
    plt.plot(t_test, Q_test[3, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, dmd_Q_test[3, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 10)
    plt.plot(t_test, Q_test[4, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, dmd_Q_test[4, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.subplot(6, 2, 12)
    plt.plot(t_test, Q_test[5, :], 'k', linewidth=2, label='True')
    plt.plot(t_test, dmd_Q_test[5, :], 'r', linewidth=2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig('Pictures/DMD_SINDy_comparison.jpg')
    plt.savefig('Pictures/DMD_SINDy_comparison.pdf')
