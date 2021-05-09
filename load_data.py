import numpy as np
from numpy import pi
# define some global constants
mu0 = 4 * pi * 1e-7
rho = 2.0e19 * 2 * 1.67 * 1e-27
mion = 2 * 1.67 * 1e-27


def load_data(start, end, skip, path_plus_prefix, is_incompressible):
    """
    Loads in the incompressible MHD simulations

    Parameters
    ----------

    start: int
    (1)
        Time index for t = 0

    end: int
    (1)
        Last time index to consider

    skip: int
    (1)
        Parameter used to skip through dump files at some
        desired frequency

    path_plus_prefix: string
    (1)
        A string with the file path + file prefix
    
    is_incompressible: bool
    (1)
        A flag to indicate if this is compressible MHD data or not

    Returns
    -------

    Bx_mat: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M = number of time samples)
        The mean-subtracted and dimensionalized Bx
        at every volume-sampled location

    By_mat: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M = number of time samples)
        The mean-subtracted and dimensionalized By
        at every volume-sampled location

    Bz_mat: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M = number of time samples)
        The mean-subtracted and dimensionalized Bz
        at every volume-sampled location

    Vx_mat: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M = number of time samples)
        The mean-subtracted Vx
        at every volume-sampled location

    Vy_mat: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M = number of time samples)
        The mean-subtracted Vy
        at every volume-sampled location

    Vz_mat: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M = number of time samples)
        The mean-subtracted Vz
        at every volume-sampled location
    
    dens_mat: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M = number of time samples)
        The density at every volume-sampled location,
        returned only if is_incompressible = False

    """
    Bx_mat = []
    By_mat = []
    Bz_mat = []
    Vx_mat = []
    Vy_mat = []
    Vz_mat = []
    # Sloppy but quick, correct code 
    if not is_incompressible:
        dens_mat = []
        cols = (3, 4, 5, 6, 7, 8, 9)
        for i in range(start, end, skip):

            # print progress
            if i % 10 == 0:
                print(i)

            if i < 10:
                Bx, By, Bz, Vx, Vy, Vz, dens = np.loadtxt(
                                               path_plus_prefix + '_0000' + str(i) + '.csv', 
                                               delimiter=',', usecols=cols, unpack=True)
            elif i < 100:
                Bx, By, Bz, Vx, Vy, Vz, dens = np.loadtxt(
                                               path_plus_prefix + '_000' + str(i) + '.csv', 
                                               delimiter=',', usecols=cols, unpack=True)
            elif i < 1000:
                Bx, By, Bz, Vx, Vy, Vz, dens = np.loadtxt(
                                               path_plus_prefix + '_00' + str(i) + '.csv', 
                                               delimiter=',', usecols=cols, unpack=True)
            elif i < 10000:
                Bx, By, Bz, Vx, Vy, Vz, dens = np.loadtxt(
                                               path_plus_prefix + '_0' + str(i) + '.csv',
                                               delimiter=',', usecols=cols, unpack=True)
            else:
                Bx, By, Bz, Vx, Vy, Vz, dens = np.loadtxt(
                                               path_plus_prefix + '_' + str(i) + '.csv', 
                                               delimiter=',', usecols=cols, unpack=True)
            Bx_mat.append(Bx)
            By_mat.append(By)
            Bz_mat.append(Bz)
            Vx_mat.append(Vx)
            Vy_mat.append(Vy)
            Vz_mat.append(Vz)
            dens_mat.append(dens)
    else:
        cols = (3, 4, 5, 15, 16, 17)
        for i in range(start, end, skip):

            # print progress
            if i % 10 == 0:
                print(i)

            if i < 10:
                Bx, By, Bz, Vx, Vy, Vz = np.loadtxt(
                                         path_plus_prefix + '_0000' + str(i) + '.csv', 
                                         delimiter=',', usecols=cols, unpack=True)
            elif i < 100:
                Bx, By, Bz, Vx, Vy, Vz = np.loadtxt(
                                         path_plus_prefix + '_000' + str(i) + '.csv', 
                                         delimiter=',', usecols=cols, unpack=True)
            elif i < 1000:
                Bx, By, Bz, Vx, Vy, Vz = np.loadtxt(
                                         path_plus_prefix + '_00' + str(i) + '.csv', 
                                         delimiter=',', usecols=cols, unpack=True)
            elif i < 10000:
                Bx, By, Bz, Vx, Vy, Vz = np.loadtxt(
                                         path_plus_prefix + '_0' + str(i) + '.csv',
                                         delimiter=',', usecols=cols, unpack=True)
            else:
                Bx, By, Bz, Vx, Vy, Vz = np.loadtxt(
                                         path_plus_prefix + '_' + str(i) + '.csv', 
                                         delimiter=',', usecols=cols, unpack=True)
            Bx_mat.append(Bx)
            By_mat.append(By)
            Bz_mat.append(Bz)
            Vx_mat.append(Vx)
            Vy_mat.append(Vy)
            Vz_mat.append(Vz)

    Bx_mat = np.array(Bx_mat)
    By_mat = np.array(By_mat)
    Bz_mat = np.array(Bz_mat)
    
    #scale to Bv
    if is_incompressible:
        Vx_mat = np.sqrt(mu0 * rho) * np.array(Vx_mat)
        Vy_mat = np.sqrt(mu0 * rho) * np.array(Vy_mat)
        Vz_mat = np.sqrt(mu0 * rho) * np.array(Vz_mat)
    else:
        dens_mat = np.asarray(dens_mat)
        Vx_mat = np.sqrt(mu0 * mion * dens_mat) * np.array(Vx_mat)
        Vy_mat = np.sqrt(mu0 * mion * dens_mat) * np.array(Vy_mat)
        Vz_mat = np.sqrt(mu0 * mion * dens_mat) * np.array(Vz_mat)
    
    # Subtract off temporal average
    bx_avg = np.mean(Bx_mat, 0)
    by_avg = np.mean(By_mat, 0)
    bz_avg = np.mean(Bz_mat, 0)
    vx_avg = np.mean(Vx_mat, 0)
    vy_avg = np.mean(Vy_mat, 0)
    vz_avg = np.mean(Vz_mat, 0)
    # for i in range(np.shape(Bx_mat)[0]):
    #     Bx_mat[i, :] = Bx_mat[i, :] - bx_avg
    #     By_mat[i, :] = By_mat[i, :] - by_avg
    #     Bz_mat[i, :] = Bz_mat[i, :] - bz_avg
    #     Vx_mat[i, :] = Vx_mat[i, :] - vx_avg
    #     Vy_mat[i, :] = Vy_mat[i, :] - vy_avg
    #     Vz_mat[i, :] = Vz_mat[i, :] - vz_avg
    # Bx_mat = np.transpose(Bx_mat)
    # By_mat = np.transpose(By_mat)
    # Bz_mat = np.transpose(Bz_mat)
    # Vx_mat = np.transpose(Vx_mat)
    # Vy_mat = np.transpose(Vy_mat)
    # Vz_mat = np.transpose(Vz_mat)
    Bx_mat = (Bx_mat - bx_avg).T
    By_mat = (By_mat - by_avg).T
    Bz_mat = (Bz_mat - bz_avg).T
    Vx_mat = (Vx_mat - vx_avg).T
    Vy_mat = (Vy_mat - vy_avg).T
    Vz_mat = (Vz_mat - vz_avg).T

    if is_incompressible:
        return Bx_mat, By_mat, Bz_mat, Vx_mat, Vy_mat, Vz_mat
    else:
        dens_mat = dens_mat.T
        return Bx_mat, By_mat, Bz_mat, Vx_mat, Vy_mat, Vz_mat, dens_mat
