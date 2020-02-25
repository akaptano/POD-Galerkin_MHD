import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def inner_product(Q,R):
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
        Qr[:,i] = Q[:,i]*np.sqrt(R)
    inner_prod = np.transpose(Qr)@Qr
    return inner_prod

def plot_energy(time,inner_prod):
    """
    Plot the energy evolution as a function of time

    Parameters
    ----------
    time: numpy array of floats
    (M = number of time samples)
        Time range of interest

    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The scaled matrix of inner products X*X

    """
    plt.figure(1000)
    plt.plot(time,np.diag(inner_prod))

# def plot_Hc(time,Hc_mat,R):
#     """
#     Compute and plot the magnetic helicity evolution as a function of time
#
#     Parameters
#     ----------
#     time: numpy array of floats
#     (M = number of time samples)
#         Time range of interest
#
#     inner_prod: 2D numpy array of floats
#     (M = number of time samples, M = number of time samples)
#         The scaled matrix of inner products X*X
#
#     """
#     Hcr_mat = np.zeros(np.shape(Hc_mat))
#     for i in range(np.shape(Hc_mat)[1]):
#         Hcr_mat[:,i] = Hc_mat[:,i]*r
#     inner_prod = np.transpose(Hc_mat)@Hcr_mat
#     print('Hc: ',np.shape(inner_prod))
#     plt.figure(2000)
#     plt.plot(time,np.diag(inner_prod))

def plot_measurement(Qorig,Q,Q_true,Q_sim,t_test,r):
    """
    Plot a particular reconstructed Bx measurement

    Parameters
    ----------

    """
    Qorig = Qorig/1e3
    Q = Q/1e3
    Q_true = Q_true/1e3
    Q_sim = Q_sim/1e3
    print(np.shape(Q))
    plt.figure(324353400,figsize=(10,14))
    plt.suptitle('Probe measurement at \n(R,$\phi$,Z)=(0.034,0.81,-0.051)',fontsize=30)
    plt.subplot(6,1,1)
    plt.plot(t_test/1.0e3,Qorig[324,:],'k',linewidth=3,label='True')
    #plt.plot(t_test/1.0e3,Q[324,:],'m:')
    plt.plot(t_test/1.0e3,Q_true[324,:],'g--',linewidth=3,label='True, r='+str(r))
    plt.plot(t_test/1.0e3,Q_sim[324,:],'m:',linewidth=3,label='Model, r='+str(r))
    plt.grid(True)
    plt.ylim(-220,220)
    #plt.legend(fontsize=22,loc='upper right',framealpha=1.0)
    ax = plt.gca()
    ax.set_yticks([-200,0,200])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    plt.subplot(6,1,2)
    plt.plot(t_test/1.0e3,Qorig[324+1*28736,:],'k',linewidth=3,label=r'True $B_y$')
    #plt.plot(t_test/1.0e3,Q[327,:],'m:')
    plt.plot(t_test/1.0e3,Q_true[324+1*28736,:],'g--',linewidth=3,label=r'True $B_y$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+1*28736,:],'m:',linewidth=3,label=r'Model $B_y$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-290,290)
    ax = plt.gca()
    ax.set_yticks([-250,0,250])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    plt.subplot(6,1,3)
    plt.plot(t_test/1.0e3,Qorig[324+2*28736,:],'k',linewidth=3,label=r'True $B_z$')
    #plt.plot(t_test/1.0e3,Q[327,:],'m:')
    plt.plot(t_test/1.0e3,Q_true[324+2*28736,:],'g--',linewidth=3,label=r'True $B_z$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+2*28736,:],'m:',linewidth=3,label=r'Model $B_z$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-150,150)
    ax = plt.gca()
    ax.set_yticks([-120,0,120])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    plt.subplot(6,1,4)
    plt.plot(t_test/1.0e3,Qorig[324+3*28736,:],'k',linewidth=3,label=r'True $V_x$')
    #plt.plot(t_test/1.0e3,Q[327,:],'m:')
    plt.plot(t_test/1.0e3,Q_true[324+3*28736,:],'g--',linewidth=3,label=r'True $V_x$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+3*28736,:],'m:',linewidth=3,label=r'Model $V_x$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-19,19)
    ax = plt.gca()
    ax.set_yticks([-10,0,10])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    plt.subplot(6,1,5)
    plt.plot(t_test/1.0e3,Qorig[324+4*28736,:],'k',linewidth=3,label=r'True $V_y$')
    #plt.plot(t_test/1.0e3,Q[327,:],'m:')
    plt.plot(t_test/1.0e3,Q_true[324+4*28736,:],'g--',linewidth=3,label=r'True $V_y$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+4*28736,:],'m:',linewidth=3,label=r'Model $V_y$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-13,13)
    ax = plt.gca()
    ax.set_yticks([-10,0,10])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    plt.subplot(6,1,6)
    plt.plot(t_test/1.0e3,Qorig[324+5*28736,:],'k',linewidth=3,label=r'True $V_z$')
    #plt.plot(t_test/1.0e3,Q[327,:],'m:')
    plt.plot(t_test/1.0e3,Q_true[324+5*28736,:],'g--',linewidth=3,label=r'True $V_z$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+5*28736,:],'m:',linewidth=3,label=r'Model $V_z$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-9,9)
    ax = plt.gca()
    ax.set_yticks([-5,0,5])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    #ax.set_xticklabels([])
    plt.savefig('Pictures/Bx_fit.png',dpi=500)
    plt.savefig('Pictures/Bx_fit.pdf',dpi=500)
    plt.savefig('Pictures/Bx_fit.eps',dpi=500)
    plt.savefig('Pictures/Bx_fit.svg',dpi=500)

def make_table(sindy_model,feature_names):
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
    output_names = sindy_model.get_feature_names()
    coefficients = sindy_model.coefficients()
    colors = np.zeros(np.shape(coefficients),dtype=str)
    for i in range(np.shape(coefficients)[0]):
        for j in range(np.shape(coefficients)[1]):
            coefficients[i,j] = '{0:.3f}'.format(coefficients[i,j])
            if np.shape(coefficients)[1] == 3:
                if abs(coefficients[i,j]) > 1e-3:
                    if j == 0:
                        colors[i,j] = 'b'
                    elif j == 1:
                        colors[i,j] = 'r'
                    elif j == 2:
                        colors[i,j] = 'g'
                else:
                    colors[i,j] = 'w'
                    coefficients[i,j] = 0
            else:
                colors[i,j] = 'w'
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
        print(feature_names[i])
    df = pd.DataFrame(coefficients, columns=feature_names)
    fig, ax = plt.subplots(figsize=(14, 10))
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ytable = ax.table(cellText=df.values, rowLabels=output_names,cellColours=colors, \
        colLabels=df.columns, loc='center', colWidths=np.ones(12)*0.5/(12))
    ytable.set_fontsize(18)
    ytable.scale(1, 2)
    #fig.tight_layout()
    plt.savefig('Pictures/SINDy_table.pdf')

def update_manifold_movie(frame,x_true,x_sim,t_test):
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

    """
    print(frame)
    r = np.shape(x_sim)[1]
    plt.clf()
    fig = plt.figure(34300,figsize=(16,7))
    plt.suptitle('t = {:0.2f} (ms)'.format(t_test[frame]/1.0e3),fontsize=20)
    if r==3:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x_true[0:frame,0],x_true[0:frame,1],x_true[0:frame,2], 'k')
        ax1.scatter(x_true[frame-1,0],x_true[frame-1,1],x_true[frame-1,2], \
            color='k', marker='o')
        ax1.azim = 20+frame/9.0
        ax1.elev = frame/13.0
        ax1.set_xlabel(r'$\varphi_1(t)$',fontsize=16)
        ax1.set_ylabel(r'$\varphi_2(t)$',fontsize=16)
        ax1.set_xlim(-0.3,0.3)
        ax1.set_ylim(-0.3,0.3)
        ax1.set_zlim(-0.3,0.3)
        #ax1.set_xticklabels([])
        #ax1.set_yticklabels([])
        #ax1.set_zticklabels([])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$\varphi_3(t)$',fontsize=16) #,rotation=90)
        ax1.xaxis.labelpad=4
        ax1.yaxis.labelpad=6
        ax1.zaxis.labelpad=14
        ax1.grid(True)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(x_sim[0:frame,0],x_sim[0:frame,1],x_sim[0:frame,2], 'b--')
        ax2.scatter(x_sim[frame-1,0],x_sim[frame-1,1],x_sim[frame-1,2], \
            color='b', marker='o')
        ax2.azim = 20+frame/9.0
        ax2.elev = frame/13.0
        #ax2.plot(x_sim_predict[:,0],x_sim_predict[:,1],x_sim_predict[:,2], 'r--')
        ax2.set_xlabel(r'$\varphi_1(t)$',fontsize=16)
        ax2.set_ylabel(r'$\varphi_2(t)$',fontsize=16)
        ax2.set_xlim(-0.3,0.3)
        ax2.set_ylim(-0.3,0.3)
        ax2.set_zlim(-0.3,0.3)
        #ax2.set_xticklabels([])
        #ax2.set_yticklabels([])
        #ax2.set_zticklabels([])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$\varphi_3(t)$',fontsize=16) #,rotation=90)
        ax2.xaxis.labelpad=4
        ax2.yaxis.labelpad=6
        ax2.zaxis.labelpad=14
        ax2.grid(True)
    else:
        ax1 = fig.add_subplot(421, projection='3d')
        ax1.plot(x_true[0:frame,0],x_true[0:frame,1],x_true[0:frame,2], 'k')
        ax1.scatter(x_true[frame-1,0],x_true[frame-1,1],x_true[frame-1,2], \
            color='k', marker='o')
        ax1.azim = 20+frame/9.0
        ax1.elev = frame/13.0
        ax1.set_xlabel(r'$\varphi_1(t)$',fontsize=16)
        ax1.set_ylabel(r'$\varphi_2(t)$',fontsize=16)
        ax1.set_xlim(-0.08,0.08)
        ax1.set_ylim(-0.08,0.08)
        ax1.set_zlim(-0.08,0.08)
        ax1.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax1.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax1.set_zticks([-0.08,-0.04,0,0.04,0.08])
        #ax1.set_xticklabels([])
        #ax1.set_yticklabels([])
        #ax1.set_zticklabels([])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$\varphi_3(t)$',fontsize=16) #,rotation=90)
        ax1.xaxis.labelpad=16
        ax1.yaxis.labelpad=16
        ax1.zaxis.labelpad=14
        ax1.grid(True)
        ax2 = fig.add_subplot(422, projection='3d')
        ax2.plot(x_sim[0:frame,0],x_sim[0:frame,1],x_sim[0:frame,2], 'b--')
        ax2.scatter(x_sim[frame-1,0],x_sim[frame-1,1],x_sim[frame-1,2], \
            color='b', marker='o')
        ax2.azim = 20+frame/9.0
        ax2.elev = frame/13.0
        #ax2.plot(x_sim_predict[:,0],x_sim_predict[:,1],x_sim_predict[:,2], 'r--')
        ax2.set_xlabel(r'$\varphi_1(t)$',fontsize=16)
        ax2.set_ylabel(r'$\varphi_2(t)$',fontsize=16)
        ax2.set_xlim(-0.08,0.08)
        ax2.set_ylim(-0.08,0.08)
        ax2.set_zlim(-0.08,0.08)
        ax2.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax2.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax2.set_zticks([-0.08,-0.04,0,0.04,0.08])
        #ax2.set_xticklabels([])
        #ax2.set_yticklabels([])
        #ax2.set_zticklabels([])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$\varphi_3(t)$',fontsize=16) #,rotation=90)
        ax2.xaxis.labelpad=16
        ax2.yaxis.labelpad=16
        ax2.zaxis.labelpad=14
        ax2.grid(True)
        #
        ax3 = fig.add_subplot(423, projection='3d')
        ax3.plot(x_true[0:frame,3],x_true[0:frame,4],x_true[0:frame,5], 'k')
        ax3.scatter(x_true[frame-1,3],x_true[frame-1,4],x_true[frame-1,5], \
            color='k', marker='o')
        ax3.azim = 20+frame/9.0
        ax3.elev = frame/13.0
        ax3.set_xlabel(r'$\varphi_{4}(t)$',fontsize=16)
        ax3.set_ylabel(r'$\varphi_{5}(t)$',fontsize=16)
        ax3.set_xlim(-0.08,0.08)
        ax3.set_ylim(-0.08,0.08)
        ax3.set_zlim(-0.08,0.08)
        ax3.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax3.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax3.set_zticks([-0.08,-0.04,0,0.04,0.08])
        ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax3.set_zlabel(r'$\varphi_{6}(t)$',fontsize=16) #,rotation=90)
        ax3.xaxis.labelpad=16
        ax3.yaxis.labelpad=16
        ax3.zaxis.labelpad=14
        ax3.grid(True)
        ax4 = fig.add_subplot(424, projection='3d')
        ax4.plot(x_sim[0:frame,3],x_sim[0:frame,4],x_sim[0:frame,5], 'b--')
        ax4.scatter(x_sim[frame-1,3],x_sim[frame-1,4],x_sim[frame-1,5], \
            color='b', marker='o')
        ax4.azim = 20+frame/9.0
        ax4.elev = frame/13.0
        ax4.set_xlabel(r'$\varphi_{4}(t)$',fontsize=16)
        ax4.set_ylabel(r'$\varphi_{5}(t)$',fontsize=16)
        ax4.set_xlim(-0.08,0.08)
        ax4.set_ylim(-0.08,0.08)
        ax4.set_zlim(-0.08,0.08)
        ax4.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax4.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax4.set_zticks([-0.08,-0.04,0,0.04,0.08])
        ax4.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax4.set_zlabel(r'$\varphi_{6}(t)$',fontsize=16) #,rotation=90)
        ax4.xaxis.labelpad=16
        ax4.yaxis.labelpad=16
        ax4.zaxis.labelpad=14
        ax4.grid(True)
        #
        ax5 = fig.add_subplot(425, projection='3d')
        ax5.plot(x_true[0:frame,6],x_true[0:frame,7],x_true[0:frame,8], 'k')
        ax5.scatter(x_true[frame-1,6],x_true[frame-1,7],x_true[frame-1,8], \
            color='k', marker='o')
        ax5.azim = 20+frame/9.0
        ax5.elev = frame/13.0
        ax5.set_xlabel(r'$\varphi_{7}(t)$',fontsize=16)
        ax5.set_ylabel(r'$\varphi_{8}(t)$',fontsize=16)
        ax5.set_xlim(-0.08,0.08)
        ax5.set_ylim(-0.08,0.08)
        ax5.set_zlim(-0.08,0.08)
        ax5.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax5.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax5.set_zticks([-0.08,-0.04,0,0.04,0.08])
        ax5.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax5.set_zlabel(r'$\varphi_{9}(t)$',fontsize=16) #,rotation=90)
        ax5.xaxis.labelpad=16
        ax5.yaxis.labelpad=16
        ax5.zaxis.labelpad=14
        ax5.grid(True)
        ax6 = fig.add_subplot(426, projection='3d')
        ax6.plot(x_sim[0:frame,6],x_sim[0:frame,7],x_sim[0:frame,8], 'b--')
        ax6.scatter(x_sim[frame-1,6],x_sim[frame-1,7],x_sim[frame-1,8], \
            color='b', marker='o')
        ax6.azim = 20+frame/9.0
        ax6.elev = frame/13.0
        ax6.set_xlabel(r'$\varphi_{7}(t)$',fontsize=16)
        ax6.set_ylabel(r'$\varphi_{8}(t)$',fontsize=16)
        ax6.set_xlim(-0.08,0.08)
        ax6.set_ylim(-0.08,0.08)
        ax6.set_zlim(-0.08,0.08)
        ax6.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax6.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax6.set_zticks([-0.08,-0.04,0,0.04,0.08])
        ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax6.set_zlabel(r'$\varphi_{9}(t)$',fontsize=16) #,rotation=90)
        ax6.xaxis.labelpad=16
        ax6.yaxis.labelpad=16
        ax6.zaxis.labelpad=14
        ax6.grid(True)
        #
        ax7 = fig.add_subplot(427, projection='3d')
        ax7.plot(x_true[0:frame,9],x_true[0:frame,10],x_true[0:frame,11], 'k')
        ax7.scatter(x_true[frame-1,9],x_true[frame-1,10],x_true[frame-1,11], \
            color='k', marker='o')
        ax7.azim = 20+frame/9.0
        ax7.elev = frame/13.0
        ax7.set_xlabel(r'$\varphi_{10}(t)$',fontsize=16)
        ax7.set_ylabel(r'$\varphi_{11}(t)$',fontsize=16)
        ax7.set_xlim(-0.08,0.08)
        ax7.set_ylim(-0.08,0.08)
        ax7.set_zlim(-0.08,0.08)
        ax7.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax7.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax7.set_zticks([-0.08,-0.04,0,0.04,0.08])
        ax7.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax7.set_zlabel(r'$\varphi_{12}(t)$',fontsize=16) #,rotation=90)
        ax7.xaxis.labelpad=16
        ax7.yaxis.labelpad=16
        ax7.zaxis.labelpad=14
        ax7.grid(True)
        ax8 = fig.add_subplot(428, projection='3d')
        ax8.plot(x_sim[0:frame,9],x_sim[0:frame,10],x_sim[0:frame,11], 'b--')
        ax8.scatter(x_sim[frame-1,9],x_sim[frame-1,10],x_sim[frame-1,11], \
            color='b', marker='o')
        ax8.azim = 20+frame/9.0
        ax8.elev = frame/13.0
        ax8.set_xlabel(r'$\varphi_{10}(t)$',fontsize=16)
        ax8.set_ylabel(r'$\varphi_{11}(t)$',fontsize=16)
        ax8.set_xlim(-0.08,0.08)
        ax8.set_ylim(-0.08,0.08)
        ax8.set_zlim(-0.08,0.08)
        ax8.set_xticks([-0.08,-0.04,0,0.04,0.08])
        ax8.set_yticks([-0.08,-0.04,0,0.04,0.08])
        ax8.set_zticks([-0.08,-0.04,0,0.04,0.08])
        ax8.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax8.set_zlabel(r'$\varphi_{12}(t)$',fontsize=16) #,rotation=90)
        ax8.xaxis.labelpad=16
        ax8.yaxis.labelpad=16
        ax8.zaxis.labelpad=14
        ax8.grid(True)

def make_contour_movie(X,Y,Z,B_true,B_sim,t_test,prefix):
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
    R = X**2+Y**2
    Z0 = np.isclose(Z,np.ones(len(Z))*min(abs(Z)),rtol=1e-3,atol=1e-3)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(R[ind_Z0]),200)
    phii = np.linspace(0,2*np.pi,64)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    print(np.shape(X[ind_Z0]),np.shape(xi),np.shape(ri))
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0,0], (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0,0], (xi, yi), method='cubic')
    plt.clf()
    fig = plt.figure(6,figsize=(10,14))
    subprefix = 'r$' + prefix[0] + '_' + prefix[1] + '(R,\phi,0,t)$'
    plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[frame]/1.0e3),fontsize=20)
    plt.subplot(2,1,1)
    plt.contourf(xi,yi,Bi,cmap='plasma')
    ax = plt.gca()
    ax.axis('off')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.contourf(xi,yi,Bi_sim,cmap='plasma')
    ax = plt.gca()
    ax.axis('off')
    plt.colorbar()
    ani = animation.FuncAnimation( \
    fig, update_contour_movie, range(0,len(t_test),1), \
    fargs=(X,Y,Z,B_true,B_sim,t_test,prefix),repeat=False, \
    interval=100, blit=False)
    FPS = 30
    ani.save('Pictures/'+prefix+'_contour.mp4',fps=FPS,dpi=300)

def update_contour_movie(frame,X,Y,Z,B_true,B_sim,t_test,prefix):
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
    R = np.sqrt(X**2+Y**2)
    Z0 = np.isclose(Z,np.ones(len(Z))*min(abs(Z)),rtol=1e-3,atol=1e-3)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(R[ind_Z0]),200)
    phii = np.linspace(0,2*np.pi,64)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0,frame], (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0,frame], (xi, yi), method='cubic')
    print(frame)
    plt.clf()
    plt.figure(6,figsize=(10,7))
    subprefix = 'r$' + prefix[0] + '_' + prefix[1] + '(R,\phi,0,t)$'
    plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[frame]/1.0e3),fontsize=20)
    plt.subplot(2,1,1)
    plt.title('Simulation test data',fontsize=22)
    if prefix[0]=='B':
        plt.pcolor(xi,yi,Bi,cmap='plasma',vmin=-1e5,vmax=1e5)
        cbar = plt.colorbar(ticks=[-1e5,-5e4,0,5e4,1e5],extend='both')
        plt.clim(-1e5,1e5)
    else:
        plt.pcolor(xi,yi,Bi,cmap='plasma',vmin=-1e4,vmax=1e4)
        cbar = plt.colorbar(ticks=[-1e4,-5e3,0,5e3,1e4],extend='both')
        plt.clim(-1e4,1e4)
    # To plot the measurement locations
    #plt.scatter(X,Y,s=2,c='k')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(2,1,2)
    plt.title('Identified model',fontsize=22))
    if prefix[0]=='B':
        plt.pcolor(xi,yi,Bi_sim,cmap='plasma',vmin=-1e5,vmax=1e5)
        cbar = plt.colorbar(ticks=[-1e5,-5e4,0,5e4,1e5],extend='both')
        plt.clim(-1e5,1e5)
    else:
        plt.pcolor(xi,yi,Bi_sim,cmap='plasma',vmin=-1e4,vmax=1e4)
        cbar = plt.colorbar(ticks=[-1e4,-5e3,0,5e3,1e4],extend='both')
        plt.clim(-1e4,1e4)
    ax = plt.gca()
    ax.axis('off')

def plot_BOD_Espectrum(S):
    """
    This function plots the energy spectrum of the data matrix.

    Parameters
    ----------
    S: numpy array of floats
    (r = truncation number of the SVD)
        Diagonal of the Sigma matrix in the SVD

    """
    fig = plt.figure(1,figsize=(7,9))
    plt.subplot(2,2,4)
    plt.plot(S/S[0],'ro')
    plt.yscale('log')
    plt.ylim(1e-12,2)
    ax = plt.gca()
    ax.set_yticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0])
    #ax.set_yticklabels([r'$10^{-12}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    plt.grid(True)
    #ax.set_xticklabels([])
    plt.subplot(2,2,3)
    plt.plot(S[0:30]/S[0],'ro')
    plt.yscale('log')
    plt.ylim(1e-3,2)
    ax = plt.gca()
    ax.set_yticks([1e-3,1e-2,1e-1,1e0])
    ax.set_yticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    plt.grid(True)
    plt.savefig('Pictures/BOD_spectrum.pdf')
    plt.savefig('Pictures/BOD_spectrum.png')
    plt.savefig('Pictures/BOD_spectrum.eps')
    plt.savefig('Pictures/BOD_spectrum.svg')

def plot_BOD_Fspectrum(S):
    """
    This function plots the field spectrum of the data matrix.

    Parameters
    ----------
    S: numpy array of floats
    (r = truncation number of the SVD)
        Diagonal of the Sigma matrix in the SVD

    """
    plt.figure(1,figsize=(7,9))
    plt.subplot(2,2,2)
    plt.yscale('log')
    plt.plot(S/S[0],'ro')
    plt.ylim(1e-6,2)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticks([1e-6,1e-4,1e-2,1e0])
    plt.grid(True)
    plt.subplot(2,2,1)
    plt.plot(S[0:30]/S[0],'ro')
    plt.yscale('log')
    plt.ylim(1e-2,2)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.grid(True)
    plt.savefig('Pictures/field_spectrum.pdf')

def make_evo_plots(x_dot,x_dot_train, \
    x_dot_sim,x_true,x_sim,time,t_train,t_test):
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
    fig, axs = plt.subplots(x_true.shape[1], 1, sharex=True, figsize=(7,9))
    for i in range(x_true.shape[1]):
        axs[i].plot(time/1.0e3, x_dot[:,i], 'k', label='numerical derivative')
        axs[i].plot(t_train/1.0e3, x_dot_train[:,i], 'r--', label='model prediction')
        axs[i].plot(t_test/1.0e3, x_dot_sim[:,i], 'b--', label='model forecast')
        axs[i].set_yticklabels([])
        axs[i].grid(True)
    plt.savefig('Pictures/xdot.pdf')
    fig, axs = plt.subplots(x_true.shape[1], 1, sharex=True, figsize=(7,9))
    for i in range(x_true.shape[1]):
        axs[i].plot(t_test/1.0e3, x_true[:,i], 'k', label='true simulation')
        axs[i].plot(t_test/1.0e3, x_sim[:,i], 'b--', label='model forecast')
        axs[i].set_yticklabels([])
        axs[i].grid(True)
    plt.savefig('Pictures/x.pdf')

def make_3d_plots(x_true,x_sim,t_test):
    """
    Plots in 3D the true evolution of X along with
    the model evolution of X for the test data.

    Parameters
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

    """
    r = np.shape(x_true)[1]
    fig = plt.figure(34300,figsize=(18,10))
    if r!=3:
        ax1 = fig.add_subplot(421, projection='3d')
        ax1.plot(x_true[0:2,0],x_true[0:2,1],x_true[0:2,2], 'k')
        ax1.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$\varphi_3(t)$') #,rotation=90)
        ax1.xaxis.labelpad=4
        ax1.yaxis.labelpad=6
        ax1.zaxis.labelpad=14
        ax1.grid(True)
        ax2 = fig.add_subplot(422, projection='3d')
        ax2.plot(x_sim[0:2,0],x_sim[0:2,1],x_sim[0:2,2], 'r--')
        ax2.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$\varphi_3(t)$') #,rotation=90)
        ax2.xaxis.labelpad=4
        ax2.yaxis.labelpad=6
        ax2.zaxis.labelpad=14
        ax2.grid(True)
        ax3 = fig.add_subplot(423, projection='3d')
        ax3.plot(x_true[0:2,3],x_true[0:2,4],x_true[0:2,5], 'k')
        ax3.set(xlabel=r'$\varphi_4(t)$', ylabel=r'$\varphi_5(t)$')
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_zticklabels([])
        ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax3.set_zlabel(r'$\varphi_6(t)$') #,rotation=90)
        ax3.xaxis.labelpad=4
        ax3.yaxis.labelpad=6
        ax3.zaxis.labelpad=14
        ax3.grid(True)
        ax4 = fig.add_subplot(424, projection='3d')
        ax4.plot(x_sim[0:2,3],x_sim[0:2,4],x_sim[0:2,5], 'b--')
        #ax4.plot(x_sim_predict[:,3],x_sim_predict[:,4],x_sim_predict[:,5], 'r--')
        ax4.set(xlabel=r'$\varphi_4(t)$', ylabel=r'$\varphi_5(t)$')
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_zticklabels([])
        ax4.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax4.set_zlabel(r'$\varphi_6(t)$') #,rotation=90)
        ax4.xaxis.labelpad=4
        ax4.yaxis.labelpad=6
        ax4.zaxis.labelpad=14
        ax4.grid(True)
        ax5 = fig.add_subplot(425, projection='3d')
        ax5.plot(x_true[0:2,6],x_true[0:2,7],x_true[0:2,8], 'k')
        ax5.set(xlabel=r'$\varphi_7(t)$', ylabel=r'$\varphi_8(t)$')
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.set_zticklabels([])
        ax5.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax5.set_zlabel(r'$\varphi_9(t)$') #,rotation=90)
        ax5.xaxis.labelpad=4
        ax5.yaxis.labelpad=6
        ax5.zaxis.labelpad=14
        ax5.grid(True)
        ax6 = fig.add_subplot(426, projection='3d')
        ax6.plot(x_sim[0:2,6],x_sim[0:2,7],x_sim[0:2,8], 'b--')
        #ax4.plot(x_sim_predict[:,3],x_sim_predict[:,4],x_sim_predict[:,5], 'r--')
        ax6.set(xlabel=r'$\varphi_7(t)$', ylabel=r'$\varphi_8(t)$')
        ax6.set_xticklabels([])
        ax6.set_yticklabels([])
        ax6.set_zticklabels([])
        ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax6.set_zlabel(r'$\varphi_9(t)$') #,rotation=90)
        ax6.xaxis.labelpad=4
        ax6.yaxis.labelpad=6
        ax6.zaxis.labelpad=14
        ax6.grid(True)
        ax7 = fig.add_subplot(427, projection='3d')
        ax7.plot(x_true[0:2,9],x_true[0:2,10],x_true[0:2,11], 'k')
        ax7.set(xlabel=r'$\varphi_{10}(t)$', ylabel=r'$\varphi_{11}(t)$')
        ax7.set_xticklabels([])
        ax7.set_yticklabels([])
        ax7.set_zticklabels([])
        ax7.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax7.set_zlabel(r'$\varphi_{12}(t)$') #,rotation=90)
        ax7.xaxis.labelpad=4
        ax7.yaxis.labelpad=6
        ax7.zaxis.labelpad=14
        ax7.grid(True)
        ax8 = fig.add_subplot(428, projection='3d')
        ax8.plot(x_sim[0:2,9],x_sim[0:2,10],x_sim[0:2,11], 'b--')
        #ax4.plot(x_sim_predict[:,3],x_sim_predict[:,4],x_sim_predict[:,5], 'r--')
        ax8.set(xlabel=r'$\varphi_{10}(t)$', ylabel=r'$\varphi_{11}(t)$')
        ax8.set_xticklabels([])
        ax8.set_yticklabels([])
        ax8.set_zticklabels([])
        ax8.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax8.set_zlabel(r'$\varphi_{12}(t)$') #,rotation=90)
        ax8.xaxis.labelpad=4
        ax8.yaxis.labelpad=6
        ax8.zaxis.labelpad=14
        ax8.grid(True)
    else:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x_true[0:2,0],x_true[0:2,1],x_true[0:2,2], 'k')
        ax1.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$\varphi_3(t)$') #,rotation=90)
        ax1.xaxis.labelpad=4
        ax1.yaxis.labelpad=6
        ax1.zaxis.labelpad=14
        ax1.grid(True)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(x_sim[0:2,0],x_sim[0:2,1],x_sim[0:2,2], 'r--')
        ax2.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$\varphi_3(t)$') #,rotation=90)
        ax2.xaxis.labelpad=4
        ax2.yaxis.labelpad=6
        ax2.zaxis.labelpad=14
        ax2.grid(True)
    ani = animation.FuncAnimation( \
    fig, update_manifold_movie, range(2,len(t_test)), \
    fargs=(x_true,x_sim,t_test),repeat=False, \
    interval=100, blit=False)
    FPS = 25
    ani.save('Pictures/manifold.mp4',fps=FPS,dpi=300)
    plt.savefig('Pictures/manifold.pdf')
