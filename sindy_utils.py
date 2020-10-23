import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.integrate import simps
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from random import randint


def inner_product(Q, R):
    """
    Compute the MHD inner product in a uniformly
    sampled cylindrical geometry

    Parameters
    ----------

    Q: 2D numpy array of floats
    (6*n_samples = number of volume-sampled locations for each of the
    6 components of the two field vectors, M = number of time samples)
        Dimensionalized and normalized matrix of temporal BOD modes

    R: numpy array of floats
    (6*n_samples = number of volume-sampled locations for each of the
    6 components of the two field vectors, M = number of time samples)
        Radial coordinates of the volume-sampled locations

    Returns
    -------

    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The unscaled matrix of inner products X*X

    """
    Qr = np.zeros(np.shape(Q))
    for i in range(np.shape(Q)[1]):
        Qr[:, i] = Q[:, i] * np.sqrt(R)
    inner_prod = np.transpose(Qr) @ Qr
    return inner_prod


def plot_measurement(Qorig, Q_pod, Q_sim, t_test, r):
    """
    Plot (Bx, By, Bz, Bvx, Bvy, Bvz) for a random probe measurement,
    compare performanced between the ground truth and model,
    with option to also include the POD of the ground truth.

    Parameters
    ----------

    Qorig: 2D numpy array of floats
    (D = total number of probes x 6, M_test = number of test data samples)
        Ground truth probe data

    Q_pod: 2D numpy array of floats
    (D = total number of probes x 6, M_test = number of test data samples)
        Reconstruction of the ground truth testing data using the POD

    Q_sim: 2D numpy array of floats
    (D = total number of probes x 6, M_test = number of test data samples)
        Reconstruction of the ground truth testing
        data using the identified model

    t_test: 1D numpy array of floats
    (M_test = number of test data samples)
        Time samples for the test dataset

    r: int
    (1)
        Truncation number of the SVD

    """

    # Pick a random probe and plot the performance
    # of the model prediction for each of Bx, By, Bz, Bvx, Bvy, Bvz
    Qsize = int(np.shape(Qorig)[0]/6)
    plt.figure(figsize=(7, 9))
    rint = randint(0, Qsize-1)
    t_test = t_test/1.0e3

    # Loop through the field components
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(t_test, Qorig[rint + i*Qsize, :],
                 'k', linewidth=2, label='True')
        # plt.plot(t_test/1.0e3, Q_pod[rint, :],
        #           'k--', linewidth=2, label='True, r='+str(r))
        plt.plot(t_test, Q_sim[rint, :],
                 color='r', linewidth=2, label='Model, r='+str(r))
        plt.grid(True)
        plt.ylim(min(Q_sim[rint, :]), max(Q_sim[rint, :]))
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)

    # Save the results
    plt.savefig('Pictures/probe_measurement.pdf', dpi=100)


def make_table(sindy_model, feature_names):
    """
    Make color-coded table of coefficients. This is
    made for use only for r = 3 because otherwise
    the table is unreasonably large.

    Parameters
    ----------
    sindy_model: A PySINDy model
    (1)
        A SINDy model that has already been fit with coefficients.

    feature_names: numpy array of strings
    (r = truncation number of the SVD or number of POD modes to model)
        The names of the variables for which a time derivative is calculated

    """
    r = len(feature_names)
    output_names = sindy_model.get_feature_names()
    coefficients = np.transpose(sindy_model.coefficients())
    colors = np.zeros(np.shape(coefficients), dtype=str)
    for i in range(np.shape(coefficients)[0]):
        for j in range(np.shape(coefficients)[1]):
            coefficients[i, j] = '{0:.3f}'.format(coefficients[i, j])
            if np.shape(coefficients)[1] == 3:
                if abs(coefficients[i, j]) > 1e-3:
                    if j == 0:
                        colors[i, j] = 'b'
                    elif j == 1:
                        colors[i, j] = 'r'
                    elif j == 2:
                        colors[i, j] = 'g'
                else:
                    colors[i, j] = 'w'
                    coefficients[i, j] = 0
            else:
                colors[i, j] = 'w'

    # Fix the ^2 output names
    for i in range(len(output_names)):
        if '^2' in output_names[i]:
            print(output_names[i])
            temp = output_names[i][-3:]
            temp = temp[1:3]+temp[0]
            output_names[i] = output_names[i][:-3]+temp
            print(output_names[i])

    # Fix the feature names to add a dot on top
    for i in range(len(feature_names)):
        feature_names[i] = '$\dot{'+feature_names[i][1:-1]+'}$'
    df = pd.DataFrame(coefficients, columns=feature_names)
    fig, ax = plt.subplots(figsize=(6, 10))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    if r > 6:
        ytable = ax.table(
                 cellText=df.values[0:r, :], rowLabels=output_names[0:r],
                 cellColours=colors[0:r], colLabels=df.columns,
                 loc='center', colWidths=np.ones(12)*0.5/(12))
    else:
        ytable = ax.table(
                 cellText=df.values, rowLabels=output_names,
                 cellColours=colors, colLabels=df.columns,
                 loc='center', colWidths=np.ones(12)*0.5/(12))
    ytable.set_fontsize(18)
    ytable.scale(1, 2)
    plt.savefig('Pictures/SINDy_table.pdf')
    if r > 6:
        fig, ax = plt.subplots(figsize=(6, 30))
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ytable = ax.table(
                 cellText=df.values[r:, :], rowLabels=output_names[r:],
                 cellColours=colors[r:], colLabels=df.columns,
                 loc='center', colWidths=np.ones(12)*0.5/(12))
        ytable.set_fontsize(10)
        plt.savefig('Pictures/SINDy_table_quadratic.pdf')


def update_manifold_movie(frame, x_true, x_sim, t_test, i, j, k):
    """
    A function for the matplotlib.animation.FuncAnimation object
    to update the frame at each timestep. This makes 3D movies
    in the BOD state space.

    Parameters
    ----------
    frame: int
    (1)
        A particular frame in the animation

    x_true: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The true evolution of the temporal BOD modes

    x_sim: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The model evolution of the temporal BOD modes

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    i: int
    (1)
       Index of one of the POD modes

    j: int
    (1)
       Index of the second of the POD modes

    k: int
    (1)
       Index of the third of the POD modes

    """
    print(frame)
    r = np.shape(x_sim)[1]

    # Erase plot from previous movie frame
    plt.clf()

    # setup 3D plot, and three 2D projections on the "walls".
    fig = plt.figure(101, figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_true[0:frame, j], x_true[0:frame, k], zs=-0.4, zdir='x',
             color='gray', linewidth=3)
    ax1.plot(x_true[0:frame, i], x_true[0:frame, k], zs=-0.4, zdir='y',
             color='gray', linewidth=3)
    ax1.plot(x_true[0:frame, i], x_true[0:frame, j], zs=-0.4, zdir='z',
             color='gray', linewidth=3)
    ax1.plot(x_true[0:frame, i], x_true[0:frame, j], x_true[0:frame, k],
             'k', linewidth=5)
    ax1.scatter(x_true[frame-1, i], x_true[frame-1, j], x_true[frame-1, k],
                s=80, color='k', marker='o')
    ax1.azim = 25+0.5*frame/9.0
    ax1.elev = 5+0.5*frame/13.0
    ax1.set_xticks([-0.3, 0, 0.3])
    ax1.set_yticks([-0.3, 0, 0.3])
    ax1.set_zticks([-0.3, 0, 0.3])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xlim(-0.4, 0.4)
    ax1.set_ylim(-0.4, 0.4)
    ax1.set_zlim(-0.4, 0.4)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.tick_params(axis='both', which='minor', labelsize=18)

    # First remove fill
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Now set color to light color
    ax1.xaxis.pane.set_edgecolor('whitesmoke')
    ax1.yaxis.pane.set_edgecolor('whitesmoke')
    ax1.zaxis.pane.set_edgecolor('whitesmoke')

    # Repeat process for the model-predicted temporal POD modes
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x_sim[0:frame, j], x_sim[0:frame, k], zs=-0.4, zdir='x',
             color='lightsalmon', linewidth=3)
    ax2.plot(x_sim[0:frame, i], x_sim[0:frame, k], zs=-0.4, zdir='y',
             color='lightsalmon', linewidth=3)
    ax2.plot(x_sim[0:frame, i], x_sim[0:frame, j], zs=-0.4, zdir='z',
             color='lightsalmon', linewidth=3)
    ax2.plot(x_sim[0:frame, i], x_sim[0:frame, j], x_sim[0:frame, k],
             color='r', linewidth=5)
    ax2.scatter(x_sim[frame-1, i], x_sim[frame-1, j], x_sim[frame-1, k],
                s=80, color='r', marker='o')
    ax2.azim = 25+0.5*frame/9.0
    ax2.elev = 5+0.5*frame/13.0
    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_zlim(-0.4, 0.4)
    ax2.set_xticks([-0.3, 0, 0.3])
    ax2.set_yticks([-0.3, 0, 0.3])
    ax2.set_zticks([-0.3, 0, 0.3])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='minor', labelsize=18)

    # First remove fill
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # Now set color to light color
    ax2.xaxis.pane.set_edgecolor('whitesmoke')
    ax2.yaxis.pane.set_edgecolor('whitesmoke')
    ax2.zaxis.pane.set_edgecolor('whitesmoke')

    # Save a picture at frame 200 of the movie
    if frame == 200:
        plt.savefig('Pictures/' + str(i) + str(j) + str(k) +
                    'manifold' + str(frame) + '.pdf')


def update_toroidal_movie(frame, X, Y, Z, B_true,
                          B_pod, B_sim, t_test, prefix):
    """
    A function for the matplotlib.animation.FuncAnimation object
    to update the frame at each timestep. This makes a true vs. model
    movie at the Z=0 midplane of any of the field components.

    Parameters
    ----------

    frame: int
    (1)
        A particular frame in the animation

    X: numpy array of floats
    (n_samples = number of volume-sampled locations)
        X-coordinate locations of the volume-sampled locations

    Y: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Y-coordinate locations of the volume-sampled locations

    Z: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Z-coordinate locations of the volume-sampled locations

    B_true: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The true evolution of a particular field component
        at every volume-sampled location

    B_pod: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The POD-reconstructed evolution of a particular field component
        at every volume-sampled location

    B_sim: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The model evolution of a particular field component
        at every volume-sampled location

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    prefix: string
    (2)
        String of the field component being used. For instance,
        Bx, By, Bz, Vx, Vy, Vz are all appropriate choices.

    """
    print(frame)
    R = np.sqrt(X**2 + Y**2)

    # Find location where the probe Z location is approximately at Z=0
    Z0 = np.isclose(Z, np.ones(len(Z))*min(abs(Z)), rtol=1e-3, atol=1e-3)

    # Get the indices where Z=0
    ind_Z0 = [i for i, p in enumerate(Z0) if p]

    # Get the R and phi locations for the Z=0 probes
    ri = np.linspace(0, max(R[ind_Z0]), 40)
    phii = np.linspace(0, 2*np.pi, 100)
    ri, phii = np.meshgrid(ri, phii)

    # Convert to (x,y) coordinates
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)

    # Interpolate the true/POD-recon/predicted probe data onto the (xi,yi) mesh
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0, frame],
                  (xi, yi), method='cubic')
    Bi_pod = griddata((X[ind_Z0], Y[ind_Z0]), B_pod[ind_Z0, frame],
                      (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0, frame],
                      (xi, yi), method='cubic')

    # Erase figure from last movie frame
    plt.clf()

    # Plotting, with scaling depending on if this is B or V field.
    fig = plt.figure(102, figsize=(5, 20))
    plt.subplot(3, 1, 1)
    if prefix[0:2] == 'Bv':
        plt.pcolor(xi, yi, Bi*1.0e4, cmap='jet', vmin=-5e1, vmax=5e1)
    else:
        plt.pcolor(xi, yi, Bi*1.0e4, cmap='jet', vmin=-5e2, vmax=5e2)
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3, 1, 2)
    if prefix[0:2] == 'Bv':
        plt.pcolor(xi, yi, Bi_pod*1.0e4, cmap='jet', vmin=-5e1, vmax=5e1)
    else:
        plt.pcolor(xi, yi, Bi_pod*1.0e4, cmap='jet', vmin=-5e2, vmax=5e2)
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3, 1, 3)
    if prefix[0:2] == 'Bv':
        im = plt.pcolor(xi, yi, Bi_sim*1.0e4, cmap='jet', vmin=-5e1, vmax=5e1)
    else:
        im = plt.pcolor(xi, yi, Bi_sim*1.0e4, cmap='jet', vmin=-5e2, vmax=5e2)
    ax = plt.gca()
    ax.axis('off')
    fig.subplots_adjust(right=0.75)

    # Save picture of the contours on the first frame
    if frame == 0:
        plt.savefig('Pictures/'+prefix+'_contours.pdf')


def plot_BOD_Espectrum(S):
    """
    This function plots the energy spectrum of the data matrix.

    Parameters
    ----------

    S: numpy array of floats
    (r = truncation number of the SVD)
        Diagonal of the Sigma matrix in the SVD

    """
    fig = plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.plot(S[0:30]/S[0], 'ko')
    plt.yscale('log')
    plt.ylim(1e-4, 2)
    plt.box(on=None)
    ax = plt.gca()
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticklabels([r'$10^{-4}$', r'$10^{-3}$',
                        r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    plt.savefig('Pictures/BOD_spectrum.pdf')


def make_evo_plots(x_dot, x_dot_train, x_dot_sim,
                   x_true, x_sim, time, t_train, t_test):
    """
    Plots the true evolution of X and Xdot, along with
    the model evolution of X and Xdot, for both the
    training and test data.

    Parameters
    ----------

    x_dot: 2D numpy array of floats
    (M = number of time samples, r = truncation number of the SVD)
        True Xdot for the entire time range

    x_dot_train: 2D numpy array of floats
    (M_train = number of time samples in training data region,
    r = truncation number of the SVD)
        Model Xdot for the training data

    x_dot_test: 2D numpy array of floats
    (M_test = number of time samples in training data region,
    r = truncation number of the SVD)
        Model Xdot for the test data

    x_true: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The true evolution of the temporal BOD modes

    x_sim: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The model evolution of the temporal BOD modes

    time: numpy array of floats
    (M = number of time samples)
        Time in microseconds

    t_train: numpy array of floats
    (M_train = number of time samples in the test data region)
        Time in microseconds in the training data region

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    """
    r = x_true.shape[1]
    fig, axs = plt.subplots(r, 1, sharex=True, figsize=(7, 9))
    if r == 12 or r == 6:
        fig, axs = plt.subplots(3, int(r/3), figsize=(16, 9))
        axs = np.ravel(axs)

    # Loop over the r temporal Xdot modes that were fit
    for i in range(r):
        axs[i].plot(t_test/1.0e3, x_dot[t_train.shape[0]:, i], color='k',
                    linewidth=2, label='numerical derivative')
        # axs[i].plot(t_train/1.0e3, x_dot_train[:, i], color='red',
        #             linewidth=2, label='model prediction')
        axs[i].plot(t_test/1.0e3, x_dot_sim[:, i], color='r',
                    linewidth=2, label='model forecast')
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        axs[i].tick_params(axis='both', which='minor', labelsize=18)
        axs[i].grid(True)
    plt.savefig('Pictures/xdot.pdf')
    plt.savefig('Pictures/xdot.eps')

    # Repeat for X
    fig, axs = plt.subplots(r, 1, sharex=True, figsize=(7, 9))
    if r == 12 or r == 6:
        fig, axs = plt.subplots(3, int(r/3), figsize=(16, 9))
        axs = np.ravel(axs)
    for i in range(r):
        axs[i].plot(t_test/1.0e3, x_true[:, i], 'k',
                    linewidth=2, label='true simulation')
        axs[i].plot(t_test/1.0e3, x_sim[:, i], color='r',
                    linewidth=2, label='model forecast')
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        axs[i].tick_params(axis='both', which='minor', labelsize=18)
        axs[i].grid(True)
    plt.savefig('Pictures/x.pdf')
    plt.savefig('Pictures/x.eps')


def make_3d_plots(x_true, x_sim, t_test, prefix, i, j, k):
    """
    Plots in 3D the true evolution of X along with
    the model evolution of X for the test data.

    Parameters#a
    ----------

    x_true: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The true evolution of the temporal BOD modes

    x_sim: 2D numpy array of floats
    (M_test = number of time samples in the test data region,
    r = truncation number of the SVD)
        The model evolution of the temporal BOD modes

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    i: int
    (1)
       Index of one of the POD modes

    j: int
    (1)
       Index of the second of the POD modes

    k: int
    (1)
       Index of the third of the POD modes

    """

    # Setup plots in anticipation for animation-making
    r = np.shape(x_true)[1]
    fig = plt.figure(101, figsize=(18, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_true[0:2, i], x_true[0:2, j], x_true[0:2, k], 'k', linewidth=3)
    ax1.set_xlabel(r'$a_1$', fontsize=22)
    ax1.set_ylabel(r'$a_2$', fontsize=22)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel(r'$a_3$', fontsize=22)
    ax1.xaxis.labelpad = 10
    ax1.yaxis.labelpad = 12
    ax1.zaxis.labelpad = 22
    ax1.grid(True)
    ax1.axis('off')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x_sim[0:2, i], x_sim[0:2, j], x_sim[0:2, k], 'r', linewidth=3)
    ax2.set_xlabel(r'$a_1$', fontsize=22)
    ax2.set_ylabel(r'$a_2$', fontsize=22)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.zaxis.set_rotate_label(False)
    ax2.set_zlabel(r'$a_3$', fontsize=22)
    ax2.xaxis.labelpad = 10
    ax2.yaxis.labelpad = 12
    ax2.zaxis.labelpad = 22
    ax2.grid(True)
    ax2.axis('off')

    # Make animation object and loop over the time corresponding to test data
    ani = animation.FuncAnimation(fig, update_manifold_movie,
                                  range(2, len(t_test)),
                                  fargs=(x_true, x_sim,
                                         t_test, i, j, k), repeat=False,
                                  interval=100, blit=False)

    # Set the frames-per-second and save the animation
    FPS = 25
    ani.save('Pictures/'+prefix+'manifold'+str(i)+str(j)+str(k)+'.mp4',
             fps=FPS, dpi=100)


def plot_pod_temporal_modes(x, time):
    """
    Illustrate the temporal POD modes

    Parameters
    ----------

    x: 2D numpy array of floats
    (M = number of time samples, r = POD truncation number)
        The temporal POD modes

    time: numpy array of floats
    (M = number of time samples)
        Time range of interest

    """
    r = np.shape(x)[1]
    time = time/1.0e3
    plt.figure(figsize=(8, 5))

    # Use gridspec to make a nice gridded plot with no internal spacings
    gs1 = gridspec.GridSpec(2, 12)
    gs1.update(wspace=0.0, hspace=0.0)

    # Loop over the first 12 normalized temporal modes
    for i in range(12):
        plt.subplot(gs1[i])
        plt.plot(time, x[:, i]/np.max(abs(x[:, i])), 'k')
        ax = plt.gca()
        # ax.set_xticks([1.5, 2.75, 4.0])
        ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_yticks([-1, 0, 1])
        plt.ylim(-1.1, 1.1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.grid(True)

    # Save the figure
    plt.savefig('Pictures/temporal_modes.pdf')
    plt.savefig('Pictures/temporal_modes.eps')

    # Checking that the temporal POD modes approximately integrate to zero
    print('Simpson results: ')
    print(simps(x[:, 0], time)/(time[-1] - time[0]))
    print(simps(x[:, 1], time)/(time[-1] - time[0]))
    print(simps(x[:, 2], time)/(time[-1] - time[0]))
    print(simps(x[:, 3], time)/(time[-1] - time[0]))
    print(simps(x[:, 4], time)/(time[-1] - time[0]))
    print(simps(x[:, 5], time)/(time[-1] - time[0]))
    print(simps(x[:, 6], time)/(time[-1] - time[0]))

    # Now plot the fourier transforms OF THE MODES
    time_uniform = np.linspace(time[0], time[-1], len(time)*2)
    x_uniform = np.zeros((len(time)*2, x.shape[1]))

    # Interpolate onto a uniform time base for the DFT
    for i in range(x.shape[1]):
        x_uniform[:, i] = np.interp(time_uniform, time, x[:, i])
    fftx = np.fft.fft(x_uniform, axis=0)/len(time)
    freq = np.fft.fftfreq(len(time_uniform), time_uniform[1]-time_uniform[0])
    fftx = fftx[:len(time)-1, :]
    freq = freq[:len(time)-1]

    # Loop over the first 12 temporal mode FFTs
    for i in range(12):
        plt.subplot(gs1[12+i])
        plt.plot(freq, abs(fftx[:, i]), 'k', linewidth=3)
        ax = plt.gca()
        ax.set_xticks([0, 14.5, 14.5*2, 14.5*3, 14.5*4, 14.5*5])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        plt.xlim(0, 80)
        plt.grid(True)

    # Save the finished figure
    plt.savefig('Pictures/frequency_modes.pdf')
    plt.savefig('Pictures/frequency_modes.eps')

    # Plot the modes in each of their 2D state spaces
    plot_pairwise(x)

    # now save trajectories to a file
    np.savetxt('Pictures/trajectories_modes.txt', x)
    np.savetxt('Pictures/trajectories_time.txt', time)


def plot_pod_spatial_modes(X, Y, Z, U):
    """
    Makes midplane (Z=0) 2D contours of the spatial POD modes

    Parameters
    ----------

    X: numpy array of floats
    (n_samples = number of volume-sampled locations)
        X-coordinate locations of the volume-sampled locations

    Y: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Y-coordinate locations of the volume-sampled locations

    Z: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Z-coordinate locations of the volume-sampled locations

    U: 2D numpy array of floats
    (D = number of probe locations x 6, 12)
        The first 12 spatial modes from the SVD of the data matrix

    """
    R = np.sqrt(X**2 + Y**2)
    # Find location where the probe Z location is approximately at Z=0
    Z0 = np.isclose(Z, np.ones(len(Z))*min(abs(Z)), rtol=1e-3, atol=1e-3)

    # Get the indices where Z=0
    ind_Z0 = [i for i, p in enumerate(Z0) if p]

    # Get the R and phi locations for the Z=0 probes
    ri = np.linspace(0, max(R[ind_Z0]), 40)
    phii = np.linspace(0, 2*np.pi, 100)
    ri, phii = np.meshgrid(ri, phii)

    # Convert to (x,y) coordinates
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    n_sample = len(R)
    U = U.real
    fig = plt.figure(figsize=(12, 12))

    # prepare Gridspec object with no internal spacings
    gs1 = gridspec.GridSpec(12, 12)
    gs1.update(wspace=0.0, hspace=0.0)

    # Loop over Bx, By, Bz, Bvx, Bvy, Bvz
    for i in range(6):
        # Loop over the first 12 POD modes
        for j in range(12):
            U_sub = U[i*n_sample:(i+1)*n_sample, :]
            # Interpolate spatial data onto the (xi, yi) grid
            U_grid = griddata((X[ind_Z0], Y[ind_Z0]), U_sub[ind_Z0, j],
                              (xi, yi), method='cubic')
            plt.subplot(gs1[i+j*12])
            plt.pcolor(xi, yi, U_grid/np.nanmax(np.nanmax(U_grid)),
                       cmap='jet', vmin=-1e0, vmax=1e0)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])

    # Save figure
    plt.savefig('Pictures/spatial_modes.pdf', dpi=50)
    # np.savetxt("Pictures/compressible1_spatialmodes.csv", U, delimiter=",")


def plot_pairwise(x):
    """
    Makes pairwise feature space plots with the temporal
    POD modes

    Parameters
    ----------

    x: 2D numpy array of floats
    (M = number of time samples, r = POD truncation number)
        The temporal POD modes

    """
    r = np.shape(x)[1]
    plt.figure(figsize=(r, r))

    # Create GridSpec object with no internal spacings
    gs1 = gridspec.GridSpec(r, r)
    gs1.update(wspace=0.0, hspace=0.0)

    # Loop over all r temporal POD modes being modeled
    for i in range(r):
        # Loop over remaining POD modes
        for j in range(0, r-i):
            plt.subplot(gs1[i, j])
            ax = plt.gca()
            plt.plot(x[:, i], x[:, r-j-1], 'k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # Save figure
    plt.savefig('Pictures/pairwise_plots.pdf', dpi=100)


def plot_density(time, dens):
    """
    Makes density plots at a number of random
    locations to see how large the fluctuations are.

    Parameters
    ----------

    time: 1D numpy array of floats
    (M = number of time samples)
        Time samples

    dens: 2D numpy array of floats
    (n_samples = number of probe locations, M = number of time samples)
        Simulation density at every spatio-temporal location

    """

    # Rescale to ms and m^-3
    time = time/1.0e3
    dens = dens/1.0e19

    # Pick some random locations to see density fluctuation sizes
    plt.figure(figsize=(10, 14))
    for i in range(12):
        plt.subplot(6, 2, i+1)
        plt.plot(time, dens[randint(0, dens.shape[0]-1), :], 'k')
        plt.ylim(0.5, 3.5)
        ax = plt.gca()
        if i != 0 and i != 5:
            ax.set_yticklabels([])
        if i <= 9:
            ax.set_xticklabels([])

    # Save figure
    plt.savefig('Pictures/density_samples.pdf')


def make_toroidal_movie(X, Y, Z, B_true, B_pod,
                        B_sim, t_test, prefix):
    """
    Function to make a true vs. model movie at the Z=0 midplane of any
    of the field components.

    Parameters
    ----------

    X: numpy array of floats
    (n_samples = number of volume-sampled locations)
        X-coordinate locations of the volume-sampled locations

    Y: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Y-coordinate locations of the volume-sampled locations

    Z: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Z-coordinate locations of the volume-sampled locations

    B_true: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The true evolution of a particular field component
        at every volume-sampled location

    B_pod: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The POD-reconstructed evolution of a particular field component
        at every volume-sampled location

    B_sim: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The model evolution of a particular field component
        at every volume-sampled location

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    prefix: string
    (2)
        String of the field component being used. For instance,
        Bx, By, Bz, Vx, Vy, Vz are all appropriate choices.

    """
    R = np.sqrt(X**2 + Y**2)

    # Find location where the probe Z location is approximately at Z=0
    Z0 = np.isclose(Z, np.ones(len(Z))*min(abs(Z)), rtol=1e-3, atol=1e-3)

    # Get the indices where Z=0
    ind_Z0 = [i for i, p in enumerate(Z0) if p]

    # Get the R and phi locations for the Z=0 probes
    ri = np.linspace(0, max(R[ind_Z0]), 40)
    phii = np.linspace(0, 2*np.pi, 100)
    ri, phii = np.meshgrid(ri, phii)

    # Convert to (x,y) coordinates
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)

    # Interpolate measurements to the (xi,yi) mesh
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0, 0],
                  (xi, yi), method='cubic')
    Bi_pod = griddata((X[ind_Z0], Y[ind_Z0]), B_pod[ind_Z0, 0],
                      (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0, 0],
                      (xi, yi), method='cubic')

    # Setup figure for animation
    fig = plt.figure(102, figsize=(5, 20))
    plt.subplot(3, 1, 1)
    plt.contourf(xi, yi, Bi*1.0e4, cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.contourf(xi, yi, Bi_pod*1.0e4, cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3, 1, 3)
    plt.contourf(xi, yi, Bi_sim*1.0e4, cmap='jet')
    ax = plt.gca()
    ax.axis('off')

    # Setup animation object, looping over times corresponding to test data
    ani = animation.FuncAnimation(fig, update_toroidal_movie,
                                  range(0, len(t_test), 1),
                                  fargs=(X, Y, Z, B_true,
                                         B_pod, B_sim, t_test, prefix),
                                  repeat=False, interval=100,
                                  blit=False)

    # Set frames-per-second and save the animation
    FPS = 30
    ani.save('Pictures/'+prefix+'_toroidal_contour.mp4', fps=FPS, dpi=200)


def make_poloidal_movie(X, Y, Z, B_true, B_pod, B_sim, t_test, prefix):
    """
    Function to make a true vs. model movie at the Y=0 cross-section
    of any of the field components.

    Parameters
    ----------

    X: numpy array of floats
    (n_samples = number of volume-sampled locations)
        X-coordinate locations of the volume-sampled locations

    Y: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Y-coordinate locations of the volume-sampled locations

    Z: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Z-coordinate locations of the volume-sampled locations

    B_true: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The true evolution of a particular field component
        at every volume-sampled location

    B_pod: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The POD-reconstructed evolution of a particular field component
        at every volume-sampled location

    B_sim: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The model evolution of a particular field component
        at every volume-sampled location

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    prefix: string
    (2)
        String of the field component being used. For instance,
        Bx, By, Bz, Vx, Vy, Vz are all appropriate choices.

    """
    R = X**2 + Y**2

    # Find where X > 0 and Y = 0 (so a poloidal cross-section)
    X0 = np.ravel(np.where(np.array(X) > 0.0))
    Y0 = np.isclose(Y, np.ones(len(Y))*min(abs(Y)), rtol=1e-3, atol=1e-3)

    # Get indices with both X > 0 and Y = 0
    ind_Y0 = [i for i, p in enumerate(Y0) if p]
    ind_Y0 = np.intersect1d(X0, ind_Y0)

    # Make (X,Z) mesh
    xi = np.linspace(min(X[ind_Y0]), max(X[ind_Y0]))
    zi = np.linspace(min(Z[ind_Y0]), max(Z[ind_Y0]))
    xi, zi = np.meshgrid(xi, zi)

    # Interpolate measurements onto the (xi,zi) mesh
    Bi = griddata((X[ind_Y0], Z[ind_Y0]), B_true[ind_Y0, 0],
                  (xi, zi), method='cubic')
    Bi_pod = griddata((X[ind_Y0], Z[ind_Y0]), B_pod[ind_Y0, 0],
                      (xi, zi), method='cubic')
    Bi_sim = griddata((X[ind_Y0], Z[ind_Y0]), B_sim[ind_Y0, 0],
                      (xi, zi), method='cubic')

    # Setup figure for animation
    fig = plt.figure(103, figsize=(5, 20))
    plt.subplot(3, 1, 1)
    plt.contourf(xi, zi, Bi*1.0e4, cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.contourf(xi, zi, Bi_pod*1.0e4, cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3, 1, 3)
    plt.contourf(xi, zi, Bi_sim*1.0e4, cmap='jet')
    ax = plt.gca()
    ax.axis('off')

    # Setup animation object, looping over times corresponding to testing data
    ani = animation.FuncAnimation(fig, update_poloidal_movie,
                                  range(0, len(t_test), 1),
                                  fargs=(X, Y, Z, B_true, B_pod,
                                         B_sim, t_test, prefix),
                                  repeat=False, interval=100,
                                  blit=False)

    # Set frames-per-second and save animation
    FPS = 30
    ani.save('Pictures/'+prefix+'_poloidal_contour.mp4', fps=FPS, dpi=200)


def update_poloidal_movie(frame, X, Y, Z, B_true,
                          B_pod, B_sim, t_test, prefix):
    """
    A function for the matplotlib.animation.FuncAnimation object
    to update the frame at each timestep. This makes a true vs. model
    movie at the Z=0 midplane of any of the field components.

    Parameters
    ----------

    frame: int
    (1)
        A particular frame in the animation

    X: numpy array of floats
    (n_samples = number of volume-sampled locations)
        X-coordinate locations of the volume-sampled locations

    Y: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Y-coordinate locations of the volume-sampled locations

    Z: numpy array of floats
    (n_samples = number of volume-sampled locations)
        Z-coordinate locations of the volume-sampled locations

    B_true: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The true evolution of a particular field component
        at every volume-sampled location

    B_pod: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The POD-reconstructed evolution of a particular field component
        at every volume-sampled location

    B_sim: 2D numpy array of floats
    (n_samples = number of volume-sampled locations,
    M_test = number of time samples in the test data region)
        The model evolution of a particular field component
        at every volume-sampled location

    t_test: numpy array of floats
    (M_test = number of time samples in the test data region)
        Time in microseconds in the test data region

    prefix: string
    (2)
        String of the field component being used. For instance,
        Bx, By, Bz, Vx, Vy, Vz are all appropriate choices.

    """
    print(frame)
    R = np.sqrt(X**2 + Y**2)
    # Find where X > 0 and Y = 0 (so a poloidal cross-section)
    X0 = np.ravel(np.where(np.array(X) > 0.0))
    Y0 = np.isclose(Y, np.ones(len(Y))*min(abs(Y)), rtol=1e-3, atol=1e-3)

    # Get indices with both X > 0 and Y = 0
    ind_Y0 = [i for i, p in enumerate(Y0) if p]
    ind_Y0 = np.intersect1d(X0, ind_Y0)

    # Make (X,Z) mesh
    xi = np.linspace(min(X[ind_Y0]), max(X[ind_Y0]))
    zi = np.linspace(min(Z[ind_Y0]), max(Z[ind_Y0]))
    xi, zi = np.meshgrid(xi, zi)

    # Interpolate measurements onto the (xi,zi) mesh
    Bi = griddata((X[ind_Y0], Z[ind_Y0]), B_true[ind_Y0, frame],
                  (xi, zi), method='cubic')
    Bi_pod = griddata((X[ind_Y0], Z[ind_Y0]), B_pod[ind_Y0, frame],
                      (xi, zi), method='cubic')
    Bi_sim = griddata((X[ind_Y0], Z[ind_Y0]), B_sim[ind_Y0, frame],
                      (xi, zi), method='cubic')

    # Clear figure from previous frame
    plt.clf()

    # Plot data, scaling depending on whether B or V is plotted
    fig = plt.figure(103, figsize=(5, 20))
    plt.subplot(3, 1, 1)
    if prefix[0:2] == 'Bv':
        plt.pcolor(xi, zi, Bi*1.0e4, cmap='jet', vmin=-5e1, vmax=5e1)
    else:
        plt.pcolor(xi, zi, Bi*1.0e4, cmap='jet', vmin=-5e2, vmax=5e2)
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3, 1, 2)
    if prefix[0:2] == 'Bv':
        plt.pcolor(xi, zi, Bi_pod*1.0e4, cmap='jet', vmin=-5e1, vmax=5e1)
    else:
        plt.pcolor(xi, zi, Bi_pod*1.0e4, cmap='jet', vmin=-5e2, vmax=5e2)
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3, 1, 3)
    if prefix[0:2] == 'Bv':
        im = plt.pcolor(xi, zi, Bi_sim*1.0e4, cmap='jet', vmin=-5e1, vmax=5e1)
    else:
        im = plt.pcolor(xi, zi, Bi_sim*1.0e4, cmap='jet', vmin=-5e2, vmax=5e2)
    ax = plt.gca()
    ax.axis('off')
    fig.subplots_adjust(right=0.75)

    # Save picture at first frame
    if frame == 0:
        plt.savefig('Pictures/'+prefix+'_poloidal_contours.pdf')
