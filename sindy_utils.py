import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from dmd import dmd
import matplotlib.gridspec as gridspec

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

def plot_energy(time,energy_mat,R):
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
    energyr_mat = np.zeros(np.shape(energy_mat))
    for i in range(np.shape(energy_mat)[1]):
        energyr_mat[:,i] = energy_mat[:,i]*np.sqrt(R)
    inner_prod = np.transpose(energyr_mat)@energyr_mat
    plt.figure(1000)
    plt.plot(time,np.diag(inner_prod))
    plt.savefig('Pictures/energy.pdf')

def plot_Hc(time,Hc_mat,R):
    """
    Compute and plot the magnetic helicity evolution as a function of time

    Parameters
    ----------
    time: numpy array of floats
    (M = number of time samples)
        Time range of interest

    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The scaled matrix of inner products X*X

    """
    Hcr_mat = np.zeros(np.shape(Hc_mat))
    indiv_size = int(np.shape(Hc_mat)[1]/6)
    for i in range(np.shape(Hc_mat)[1]):
        Hcr_mat[:,i] = Hc_mat[:,i]*np.sqrt(R)
    inner_prod = np.transpose(Hcr_mat[0:3*indiv_size,:])@Hcr_mat[3*indiv_size:6*indiv_size,:]
    print('Hc: ',np.shape(inner_prod))
    plt.figure(2000)
    plt.plot(time,np.diag(inner_prod))

def plot_measurement(Qorig,Q_true,Q_sim,t_test,r):
    """
    Plot a particular reconstructed Bx measurement

    Parameters
    ----------

    """
    Qsize = int(np.shape(Qorig)[0]/6)
    print(Qsize)
    Qorig = Qorig
    Q_true = Q_true
    Q_sim = Q_sim
    plt.figure(324353400,figsize=(7,9))
    #plt.suptitle('Probe measurement at \n(R,$\phi$,Z)=(0.034,0.81,-0.051)',fontsize=20)
    plt.subplot(6,1,1)
    plt.plot(t_test/1.0e3,Qorig[324,:],'k',linewidth=2,label='True')
    #plt.plot(t_test/1.0e3,Q[324,:],'b--')
    #plt.plot(t_test/1.0e3,Q_true[324,:],'k--',linewidth=2,label='True, r='+str(r))
    plt.plot(t_test/1.0e3,Q_sim[324,:],color='r',linewidth=2,label='Model, r='+str(r))
    plt.grid(True)
    plt.ylim(-350,350)
    #plt.legend(fontsize=18,loc='upper right',framealpha=1.0)
    ax = plt.gca()
    #ax.set_yticks([-200,0,200])
    #ax.set_xticks([2.4,2.6,2.8,3.0])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,2)
    plt.plot(t_test/1.0e3,Qorig[324+1*Qsize,:],'k',linewidth=2,label=r'True $B_y$')
    #plt.plot(t_test/1.0e3,Q[327,:],'b')
    #plt.plot(t_test/1.0e3,Q_true[324+1*Qsize,:],'k--',linewidth=2,label=r'True $B_y$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+1*Qsize,:],color='r',linewidth=2,label=r'Model $B_y$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-700,700)
    ax = plt.gca()
    #ax.set_yticks([-250,0,250])
    #ax.set_xticks([2.4,2.6,2.8,3.0])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,3)
    plt.plot(t_test/1.0e3,Qorig[324+2*Qsize,:],'k',linewidth=2,label=r'True $B_z$')
    #plt.plot(t_test/1.0e3,Q[327,:],'b')
    #plt.plot(t_test/1.0e3,Q_true[324+2*Qsize,:],'k--',linewidth=2,label=r'True $B_z$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+2*Qsize,:],color='r',linewidth=2,label=r'Model $B_z$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-200,200)
    ax = plt.gca()
    #ax.set_yticks([-120,0,120])
    #ax.set_xticks([2.4,2.6,2.8,3.0])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,4)
    plt.plot(t_test/1.0e3,Qorig[324+3*Qsize,:],'k',linewidth=2,label=r'True $V_x$')
    #plt.plot(t_test/1.0e3,Q[327,:],'b')
    #plt.plot(t_test/1.0e3,Q_true[324+3*Qsize,:],'k--',linewidth=2,label=r'True $V_x$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+3*Qsize,:],color='r',linewidth=2,label=r'Model $V_x$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-60,60)
    ax = plt.gca()
    #ax.set_yticks([-10,0,10])
    #ax.set_xticks([2.4,2.6,2.8,3.0])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,5)
    plt.plot(t_test/1.0e3,Qorig[324+4*Qsize,:],'k',linewidth=2,label=r'True $V_y$')
    #plt.plot(t_test/1.0e3,Q[327,:],'b')
    #plt.plot(t_test/1.0e3,Q_true[324+4*Qsize,:],'k--',linewidth=2,label=r'True $V_y$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+4*Qsize,:],color='r',linewidth=2,label=r'Model $V_y$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-40,40)
    ax = plt.gca()
    #ax.set_yticks([-10,0,10])
    #ax.set_xticks([2.4,2.6,2.8,3.0])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,6)
    plt.plot(t_test/1.0e3,Qorig[324+5*Qsize,:],'k',linewidth=2,label=r'True $V_z$')
    #plt.plot(t_test/1.0e3,Q[327,:],'b')
    #plt.plot(t_test/1.0e3,Q_true[324+5*Qsize,:],'k--',linewidth=2,label=r'True $V_z$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+5*Qsize,:],color='r',linewidth=2,label=r'Model $V_z$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-25,25)
    ax = plt.gca()
    #ax.set_yticks([-5,0,5])
    #ax.set_xticks([2.4,2.6,2.8,3.0])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    #ax.set_xticklabels([])
    plt.savefig('Pictures/Bx_fit.png',dpi=100)
    plt.savefig('Pictures/Bx_fit.pdf',dpi=100)
    plt.savefig('Pictures/Bx_fit.eps',dpi=100)
    plt.savefig('Pictures/Bx_fit.svg',dpi=100)

def make_table(sindy_model,feature_names,r):
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
    # Need to transpose when use SR3 method??
    coefficients = np.transpose(sindy_model.coefficients())
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
    print(coefficients,feature_names)
    df = pd.DataFrame(coefficients, columns=feature_names)
    fig, ax = plt.subplots(figsize=(6,10))
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    print(np.shape(df.values),np.shape(output_names),np.shape(colors),np.shape(df.columns))
    if r > 6:
        ytable = ax.table(cellText=df.values[0:r,:], rowLabels=output_names[0:r],cellColours=colors[0:r], \
            colLabels=df.columns, loc='center', colWidths=np.ones(12)*0.5/(12))
    else:
        ytable = ax.table(cellText=df.values, rowLabels=output_names,cellColours=colors, \
            colLabels=df.columns, loc='center', colWidths=np.ones(12)*0.5/(12))
    ytable.set_fontsize(18)
    ytable.scale(1, 2)
    #fig.tight_layout()
    plt.savefig('Pictures/SINDy_table.pdf')
    if r > 6:
        fig, ax = plt.subplots(figsize=(6,10))
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ytable = ax.table(cellText=df.values[r:,:], rowLabels=output_names[r:],cellColours=colors[r:], \
            colLabels=df.columns, loc='center', colWidths=np.ones(12)*0.5/(12))
        ytable.set_fontsize(18)
        ytable.scale(1, 2)
        plt.savefig('Pictures/SINDy_table_quadratic.pdf')

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
    if r < 13:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x_true[0:frame,1],x_true[0:frame,2],zs=-0.4,zdir='x',color='gray', linewidth=3)
        ax1.plot(x_true[0:frame,0],x_true[0:frame,2],zs=-0.4,zdir='y',color='gray', linewidth=3)
        ax1.plot(x_true[0:frame,0],x_true[0:frame,1],zs=-0.4,zdir='z',color='gray', linewidth=3) 
        ax1.plot(x_true[0:frame,0],x_true[0:frame,1],x_true[0:frame,2], 'k', linewidth=5)
        ax1.scatter(x_true[frame-1,0],x_true[frame-1,1],x_true[frame-1,2],s=80, \
            color='k', marker='o')
        ax1.azim = 25+0.5*frame/9.0
        ax1.elev = 5+0.5*frame/13.0
        #ax1.set_xlabel(r'$a_1$',fontsize=22)
        #ax1.set_ylabel(r'$a_2$',fontsize=22)
        ax1.set_xticks([-0.3,0,0.3])
        ax1.set_yticks([-0.3,0,0.3])
        ax1.set_zticks([-0.3,0,0.3])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        #ax1.set_xticks([-0.6,-0.3,0,0.3,0.6])
        #ax1.set_yticks([-0.6,-0.3,0,0.3,0.6])
        #ax1.set_zticks([-0.6,-0.3,0,0.3,0.6])
        ax1.set_xlim(-0.4,0.4)
        ax1.set_ylim(-0.4,0.4)
        ax1.set_zlim(-0.4,0.4) 
        #ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        #ax1.set_zlabel(r'$a_3$',fontsize=22) #,rotation=90)
        #ax1.xaxis.labelpad=10
        #ax1.yaxis.labelpad=12
        #ax1.zaxis.labelpad=22
        ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.tick_params(axis='both', which='minor', labelsize=18)
        # First remove fill
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        # Now set color to white (or whatever is "invisible")
        ax1.xaxis.pane.set_edgecolor('whitesmoke')
        ax1.yaxis.pane.set_edgecolor('whitesmoke')
        ax1.zaxis.pane.set_edgecolor('whitesmoke') 
        #ax1.axis('off')
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(x_sim[0:frame,1],x_sim[0:frame,2],zs=-0.4,zdir='x',color='lightsalmon', linewidth=3)
        ax2.plot(x_sim[0:frame,0],x_sim[0:frame,2],zs=-0.4,zdir='y',color='lightsalmon', linewidth=3)
        ax2.plot(x_sim[0:frame,0],x_sim[0:frame,1],zs=-0.4,zdir='z',color='lightsalmon', linewidth=3)
        ax2.plot(x_sim[0:frame,0],x_sim[0:frame,1],x_sim[0:frame,2], color='r',linewidth=5)
        ax2.scatter(x_sim[frame-1,0],x_sim[frame-1,1],x_sim[frame-1,2],s=80, \
            color='r', marker='o')
        ax2.azim = 25+0.5*frame/9.0
        ax2.elev = 5+0.5*frame/13.0
        #ax2.plot(x_sim_predict[:,0],x_sim_predict[:,1],x_sim_predict[:,2], 'r')
        #ax2.set_xlabel(r'$a_1$',fontsize=22)
        #ax2.set_ylabel(r'$a_2$',fontsize=22)
        ax2.set_xlim(-0.4,0.4)
        ax2.set_ylim(-0.4,0.4)
        ax2.set_zlim(-0.4,0.4)
        ax2.set_xticks([-0.3,0,0.3])
        ax2.set_yticks([-0.3,0,0.3])
        ax2.set_zticks([-0.3,0,0.3]) 
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        #ax2.set_xticks([-0.6,-0.3,0,0.3,0.6])
        #ax2.set_yticks([-0.6,-0.3,0,0.3,0.6])
        #ax2.set_zticks([-0.6,-0.3,0,0.3,0.6])
        #ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        #ax2.set_zlabel(r'$a_3$',fontsize=22) #,rotation=90)
        #ax2.xaxis.labelpad=10
        #ax2.yaxis.labelpad=12
        #ax2.zaxis.labelpad=22
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.tick_params(axis='both', which='minor', labelsize=18)
        # First remove fill
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        # Now set color to white (or whatever is "invisible")
        ax2.xaxis.pane.set_edgecolor('whitesmoke')
        ax2.yaxis.pane.set_edgecolor('whitesmoke')
        ax2.zaxis.pane.set_edgecolor('whitesmoke') 
        #ax2.axis('off')
    else:
        ax1 = fig.add_subplot(421, projection='3d')
        ax1.plot(x_true[0:frame,0],x_true[0:frame,1],x_true[0:frame,2], 'k', linewidth=3)
        ax1.scatter(x_true[frame-1,0],x_true[frame-1,1],x_true[frame-1,2],s=50, \
            color='k', marker='o')
        ax1.azim = 15+0.5*frame/9.0
        ax1.elev = 5+0.5*frame/13.0
        ax1.set_xlabel(r'$a_1$',fontsize=22)
        ax1.set_ylabel(r'$a_2$',fontsize=22)
        ax1.set_xlim(-0.08,0.08)
        ax1.set_ylim(-0.08,0.08)
        ax1.set_zlim(-0.08,0.08)
        ax1.set_xticks([-0.04,0,0.04])
        ax1.set_yticks([-0.04,0,0.04])
        ax1.set_zticks([-0.04,0,0.04])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$a_3$',fontsize=22) #,rotation=90)
        ax1.xaxis.labelpad=26
        ax1.yaxis.labelpad=26
        ax1.zaxis.labelpad=14
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.tick_params(axis='both', which='minor', labelsize=16)
        ax1.grid(True)
        ax2 = fig.add_subplot(422, projection='3d')
        ax2.plot(x_sim[0:frame,0],x_sim[0:frame,1],x_sim[0:frame,2], 'b', linewidth=3)
        ax2.scatter(x_sim[frame-1,0],x_sim[frame-1,1],x_sim[frame-1,2],s=50, \
            color='b', marker='o')
        ax2.azim = 15+0.5*frame/9.0
        ax2.elev = 5+0.5*frame/13.0
        #ax2.plot(x_sim_predict[:,0],x_sim_predict[:,1],x_sim_predict[:,2], 'r')
        ax2.set_xlabel(r'$a_1$',fontsize=22)
        ax2.set_ylabel(r'$a_2$',fontsize=22)
        ax2.set_xlim(-0.08,0.08)
        ax2.set_ylim(-0.08,0.08)
        ax2.set_zlim(-0.08,0.08)
        ax2.set_xticks([-0.04,0,0.04])
        ax2.set_yticks([-0.04,0,0.04])
        ax2.set_zticks([-0.04,0,0.04])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$a_3$',fontsize=22) #,rotation=90)
        ax2.xaxis.labelpad=26
        ax2.yaxis.labelpad=26
        ax2.zaxis.labelpad=14
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='both', which='minor', labelsize=16)
        #
        ax3 = fig.add_subplot(423, projection='3d')
        ax3.plot(x_true[0:frame,3],x_true[0:frame,4],x_true[0:frame,5], 'k', linewidth=3)
        ax3.scatter(x_true[frame-1,3],x_true[frame-1,4],x_true[frame-1,5],s=50, \
            color='k', marker='o')
        ax3.azim = 15+0.5*frame/9.0
        ax3.elev = 5+0.5*frame/13.0
        ax3.set_xlabel(r'$a_{4}(t)$',fontsize=22)
        ax3.set_ylabel(r'$a_{5}(t)$',fontsize=22)
        ax3.set_xlim(-0.08,0.08)
        ax3.set_ylim(-0.08,0.08)
        ax3.set_zlim(-0.08,0.08)
        ax3.set_xticks([-0.04,0,0.04])
        ax3.set_yticks([-0.04,0,0.04])
        ax3.set_zticks([-0.04,0,0.04])
        ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax3.set_zlabel(r'$a_{6}(t)$',fontsize=22) #,rotation=90)
        ax3.xaxis.labelpad=26
        ax3.yaxis.labelpad=26
        ax3.zaxis.labelpad=14
        ax3.grid(True)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.tick_params(axis='both', which='minor', labelsize=16)
        #
        ax4 = fig.add_subplot(424, projection='3d')
        ax4.plot(x_sim[0:frame,3],x_sim[0:frame,4],x_sim[0:frame,5], 'b', linewidth=3)
        ax4.scatter(x_sim[frame-1,3],x_sim[frame-1,4],x_sim[frame-1,5],s=50, \
            color='b', marker='o')
        ax4.azim = 15+0.5*frame/9.0
        ax4.elev = 5+0.5*frame/13.0
        ax4.set_xlabel(r'$a_{4}(t)$',fontsize=22)
        ax4.set_ylabel(r'$a_{5}(t)$',fontsize=22)
        ax4.set_xlim(-0.08,0.08)
        ax4.set_ylim(-0.08,0.08)
        ax4.set_zlim(-0.08,0.08)
        ax4.set_xticks([-0.04,0,0.04])
        ax4.set_yticks([-0.04,0,0.04])
        ax4.set_zticks([-0.04,0,0.04])
        ax4.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax4.set_zlabel(r'$a_{6}(t)$',fontsize=22) #,rotation=90)
        ax4.xaxis.labelpad=26
        ax4.yaxis.labelpad=26
        ax4.zaxis.labelpad=14
        ax4.grid(True)
        ax4.tick_params(axis='both', which='major', labelsize=16)
        ax4.tick_params(axis='both', which='minor', labelsize=16)
        #
        ax5 = fig.add_subplot(425, projection='3d')
        ax5.plot(x_true[0:frame,6],x_true[0:frame,7],x_true[0:frame,8], 'k', linewidth=3)
        ax5.scatter(x_true[frame-1,6],x_true[frame-1,7],x_true[frame-1,8],s=50, \
            color='k', marker='o')
        ax5.azim = 15+0.5*frame/9.0
        ax5.elev = 5+0.5*frame/13.0
        ax5.set_xlabel(r'$a_{7}(t)$',fontsize=22)
        ax5.set_ylabel(r'$a_{8}(t)$',fontsize=22)
        ax5.set_xlim(-0.08,0.08)
        ax5.set_ylim(-0.08,0.08)
        ax5.set_zlim(-0.08,0.08)
        ax5.set_xticks([-0.04,0,0.04])
        ax5.set_yticks([-0.04,0,0.04])
        ax5.set_zticks([-0.04,0,0.04])
        ax5.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax5.set_zlabel(r'$a_{9}(t)$',fontsize=22) #,rotation=90)
        ax5.xaxis.labelpad=26
        ax5.yaxis.labelpad=26
        ax5.zaxis.labelpad=14
        ax5.grid(True)
        ax5.tick_params(axis='both', which='major', labelsize=16)
        ax5.tick_params(axis='both', which='minor', labelsize=16)
        #
        ax6 = fig.add_subplot(426, projection='3d')
        ax6.plot(x_sim[0:frame,6],x_sim[0:frame,7],x_sim[0:frame,8], 'b', linewidth=3)
        ax6.scatter(x_sim[frame-1,6],x_sim[frame-1,7],x_sim[frame-1,8],s=50, \
            color='b', marker='o')
        ax6.azim = 15+0.5*frame/9.0
        ax6.elev = 5+0.5*frame/13.0
        ax6.set_xlabel(r'$a_{7}(t)$',fontsize=22)
        ax6.set_ylabel(r'$a_{8}(t)$',fontsize=22)
        ax6.set_xlim(-0.08,0.08)
        ax6.set_ylim(-0.08,0.08)
        ax6.set_zlim(-0.08,0.08)
        ax6.set_xticks([-0.04,0,0.04])
        ax6.set_yticks([-0.04,0,0.04])
        ax6.set_zticks([-0.04,0,0.04])
        ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax6.set_zlabel(r'$a_{9}(t)$',fontsize=22) #,rotation=90)
        ax6.xaxis.labelpad=26
        ax6.yaxis.labelpad=26
        ax6.zaxis.labelpad=14
        ax6.grid(True)
        ax6.tick_params(axis='both', which='major', labelsize=16)
        ax6.tick_params(axis='both', which='minor', labelsize=16)
        #
        ax7 = fig.add_subplot(427, projection='3d')
        ax7.plot(x_true[0:frame,9],x_true[0:frame,10],x_true[0:frame,11], 'k', linewidth=3)
        ax7.scatter(x_true[frame-1,9],x_true[frame-1,10],x_true[frame-1,11],s=50, \
            color='k', marker='o')
        ax7.azim = 15+0.5*frame/9.0
        ax7.elev = 5+0.5*frame/13.0
        ax7.set_xlabel(r'$a_{10}(t)$',fontsize=22)
        ax7.set_ylabel(r'$a_{11}(t)$',fontsize=22)
        ax7.set_xlim(-0.08,0.08)
        ax7.set_ylim(-0.08,0.08)
        ax7.set_zlim(-0.08,0.08)
        ax7.set_xticks([-0.04,0,0.04])
        ax7.set_yticks([-0.04,0,0.04])
        ax7.set_zticks([-0.04,0,0.04])
        ax7.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax7.set_zlabel(r'$a_{12}(t)$',fontsize=22) #,rotation=90)
        ax7.xaxis.labelpad=26
        ax7.yaxis.labelpad=26
        ax7.zaxis.labelpad=14
        ax7.grid(True)
        ax7.tick_params(axis='both', which='major', labelsize=16)
        ax7.tick_params(axis='both', which='minor', labelsize=16)
        #
        ax8 = fig.add_subplot(428, projection='3d')
        ax8.plot(x_sim[0:frame,9],x_sim[0:frame,10],x_sim[0:frame,11], 'b', linewidth=3)
        ax8.scatter(x_sim[frame-1,9],x_sim[frame-1,10],x_sim[frame-1,11],s=50, \
            color='b', marker='o')
        ax8.azim = 15+0.5*frame/9.0
        ax8.elev = 5+0.5*frame/13.0
        ax8.set_xlabel(r'$a_{10}(t)$',fontsize=22)
        ax8.set_ylabel(r'$a_{11}(t)$',fontsize=22)
        ax8.set_xlim(-0.08,0.08)
        ax8.set_ylim(-0.08,0.08)
        ax8.set_zlim(-0.08,0.08)
        ax8.set_xticks([-0.04,0,0.04])
        ax8.set_yticks([-0.04,0,0.04])
        ax8.set_zticks([-0.04,0,0.04])
        ax8.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax8.set_zlabel(r'$a_{12}(t)$',fontsize=22) #,rotation=90)
        ax8.xaxis.labelpad=26
        ax8.yaxis.labelpad=26
        ax8.zaxis.labelpad=14
        ax8.grid(True)
        ax8.tick_params(axis='both', which='major', labelsize=16)
        ax8.tick_params(axis='both', which='minor', labelsize=16)

def make_contour_movie(X,Y,Z,B_true,B_pod,B_sim,t_test,prefix):
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
    print(Z)
    #Z0 = np.isclose(Z,np.ones(len(Z))*25,rtol=0.81,atol=0.81)
    print(Z0)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(R[ind_Z0]),200)
    phii = np.linspace(0,2*np.pi,256)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    print(np.shape(X[ind_Z0]),np.shape(xi),np.shape(ri))
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0,0], (xi, yi), method='cubic')
    Bi_pod = griddata((X[ind_Z0], Y[ind_Z0]), B_pod[ind_Z0,0], (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0,0], (xi, yi), method='cubic')
    plt.clf()
    fig = plt.figure(6,figsize=(23,7))
    if prefix[1] != 'A':
        subprefix = r'$' + prefix[0] + r'_' + prefix[1] + r'(R,\phi,0,t)$'
    else:
        subprefix = r'$' + prefix[0] + r'_{' + prefix[1] + r',' + prefix[2] + r'}(R,\phi,0,t)$'
    plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[0]/1.0e3),fontsize=20)
    plt.subplot(1,3,1)
    plt.contourf(xi,yi,Bi*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.contourf(xi,yi,Bi_pod*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(1,3,2)
    plt.contourf(xi,yi,Bi_sim*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    #plt.colorbar()
    ani = animation.FuncAnimation( \
    fig, update_contour_movie, range(0,len(t_test),1), \
    fargs=(X,Y,Z,B_true,B_pod,B_sim,t_test,prefix),repeat=False, \
    interval=100, blit=False)
    FPS = 30
    ani.save('Pictures/'+prefix+'_contour.mp4',fps=FPS,dpi=200)

def update_contour_movie(frame,X,Y,Z,B_true,B_pod,B_sim,t_test,prefix):
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
    #Z0 = np.isclose(Z,np.ones(len(Z))*25,rtol=0.81,atol=0.81)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(R[ind_Z0]),100)
    phii = np.linspace(0,2*np.pi,256)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0,frame], (xi, yi), method='cubic')
    Bi_pod = griddata((X[ind_Z0], Y[ind_Z0]), B_pod[ind_Z0,frame], (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0,frame], (xi, yi), method='cubic')
    print(frame)
    plt.clf()
    fig=plt.figure(6,figsize=(23,7))
    if prefix[1] != 'v':
        subprefix = r'$' + prefix[0] + r'_' + prefix[1] + r'(R,\phi,0,t)$'
    else:
        subprefix = r'$' + prefix[0] + r'_{' + prefix[1] + r',' + prefix[2] + r'}(R,\phi,0,t)$'
    #plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[frame]/1.0e3),fontsize=30)
    plt.subplot(1,3,1)
    plt.title('Full simulation data',fontsize=30)
    if prefix[0:2]=='Bv':
        plt.pcolor(xi,yi,Bi*1.0e4,cmap='jet',vmin=-5e1,vmax=5e1)
        #cbar = plt.colorbar(ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both')
        #plt.clim(-5e1,5e1)
    else:
        plt.pcolor(xi,yi,Bi*1.0e4,cmap='jet',vmin=-5e2,vmax=5e2)
        #cbar = plt.colorbar(ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both')
        #plt.clim(-5e2,5e2)
    #cbar.ax.tick_params(labelsize=18)
    # To plot the measurement locations
    #plt.scatter(X,Y,s=2,c='k')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(1,3,2)
    plt.title('POD of simulation data',fontsize=30)
    if prefix[0:2]=='Bv':
        plt.pcolor(xi,yi,Bi_pod*1.0e4,cmap='jet',vmin=-5e1,vmax=5e1)
        #cbar = plt.colorbar(ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both')
        #plt.clim(-5e1,5e1)
        #cbar = plt.colorbar() #ticks=[-1e2,-5e1,0,5e1,1e2],extend='both')
        #plt.clim(-1e2,1e2)
    else:
        plt.pcolor(xi,yi,Bi_pod*1.0e4,cmap='jet',vmin=-5e2,vmax=5e2)
        #cbar = plt.colorbar(ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both')
        #plt.clim(-5e2,5e2)
        #cbar = plt.colorbar() #ticks=[-1e1,-5,0,5,1e1],extend='both')
        #plt.clim(-1e1,1e1)
    #cbar.ax.tick_params(labelsize=18)
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(1,3,3)
    plt.title('Identified model',fontsize=30)
    if prefix[0:2]=='Bv':
        im=plt.pcolor(xi,yi,Bi_sim*1.0e4,cmap='jet',vmin=-5e1,vmax=5e1)
        #cbar = plt.colorbar(ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both')
        #plt.clim(-5e1,5e1)
        #cbar = plt.colorbar() #ticks=[-1e2,-5e1,0,5e1,1e2],extend='both')
        #plt.clim(-1e2,1e2)
    else:
        im=plt.pcolor(xi,yi,Bi_sim*1.0e4,cmap='jet',vmin=-5e2,vmax=5e2)
        #cbar = plt.colorbar(ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both')
        #plt.clim(-5e2,5e2)
        #cbar = plt.colorbar() #ticks=[-1e1,-5,0,5,1e1],extend='both')
        #plt.clim(-1e1,1e1)
    ax = plt.gca()
    ax.axis('off')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
    if prefix[0:2]=='Bv':
        cbar = fig.colorbar(im,ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both',cax=cbar_ax)
        plt.clim(-5e1,5e1)
    else:
        cbar = fig.colorbar(im,ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both',cax=cbar_ax)
        plt.clim(-5e2,5e2)
    cbar.ax.tick_params(labelsize=18)
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
    fig = plt.figure(1,figsize=(16,7))
    #plt.subplot(1,2,2)
    #plt.plot(S/S[0],'ko')
    #plt.yscale('log')
    #plt.ylim(1e-12,2)
    #ax = plt.gca()
    #ax.set_yticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0])
    #ax.set_yticklabels([r'$10^{-12}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    #plt.grid(True)
    #ax.tick_params(axis='both', which='major', labelsize=22)
    #ax.tick_params(axis='both', which='minor', labelsize=22) 
    #ax.set_xticklabels([])
    plt.subplot(1,2,1)
    plt.plot(S[0:30]/S[0],'ko')
    plt.yscale('log')
    plt.ylim(1e-5,2)
    plt.box(on=None)
    ax = plt.gca()
    ax.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
    ax.set_yticklabels([r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    #plt.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22) 
    plt.savefig('Pictures/BOD_spectrum.pdf')
    plt.savefig('Pictures/BOD_spectrum.png')
    plt.savefig('Pictures/BOD_spectrum.eps')
    plt.savefig('Pictures/BOD_spectrum.svg')

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
    r = x_true.shape[1]
    fig, axs = plt.subplots(r, 1, sharex=True, figsize=(7,9))
    if r==12 or r==6:
        fig, axs = plt.subplots(3,int(r/3), figsize=(16,9))
        axs = np.ravel(axs)
    for i in range(r):
        axs[i].plot(t_test/1.0e3, x_dot[t_train.shape[0]:,i], color='k',linewidth=2, label='numerical derivative')
        #axs[i].plot(t_train/1.0e3, x_dot_train[:,i], color='red',linewidth=2, label='model prediction')
        axs[i].plot(t_test/1.0e3, x_dot_sim[:,i], color='r',linewidth=2, label='model forecast')
        axs[i].set_yticklabels([])
        #if i < r-2:
        #axs[i].set_xticks([2.4,2.6,2.8,3.0])
        axs[i].set_xticklabels([])
        #if i != r-1:
        #    axs[i].set_xticklabels([])
        #else:
        #    axs[i].set_xticklabels([2.4,2.6,2.8,3.0])
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        axs[i].tick_params(axis='both', which='minor', labelsize=18)
        axs[i].grid(True)
    plt.savefig('Pictures/xdot.pdf')
    plt.savefig('Pictures/xdot.eps')
    fig, axs = plt.subplots(r, 1, sharex=True, figsize=(7,9))
    if r==12 or r==6:
        fig, axs = plt.subplots(3,int(r/3), figsize=(16,9))
        axs = np.ravel(axs)
    for i in range(r):
        axs[i].plot(t_test/1.0e3, x_true[:,i], 'k',linewidth=2, label='true simulation')
        axs[i].plot(t_test/1.0e3, x_sim[:,i], color='r',linewidth=2, label='model forecast')
        axs[i].set_yticklabels([])
        #if i < r-2:
        #axs[i].set_xticks([2.4,2.6,2.8,3.0])
        axs[i].set_xticklabels([])
        #axs[i].set_xlim(t_test[0]/1.0e3,t_test[int(len(t_test)/2.0)]/1.0e3)
        #axs[i].set_ylim(-0.2,0.2)
        #axs[i].set_xlim(t_test[0]/1.0e3,t_test[int(len(t_test)/2.0)]/1.0e3)
        #axs[i].set_ylim(-1,1)
        #if i != r-1:
        #    axs[i].set_xticklabels([])
        #else:
        #    axs[i].set_xticklabels([2.4,2.6,2.8,3.0])
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        axs[i].tick_params(axis='both', which='minor', labelsize=18)
        axs[i].grid(True)
    plt.savefig('Pictures/x.pdf')
    plt.savefig('Pictures/x.eps')

def make_3d_plots(x_true,x_sim,t_test,prefix):
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
    if r > 12:
        ax1 = fig.add_subplot(421, projection='3d')
        ax1.plot(x_true[0:2,0],x_true[0:2,1],x_true[0:2,2], 'k', linewidth=3)
        ax1.set_xlabel(r'$a_1$',fontsize=22)
        ax1.set_ylabel(r'$a_2$',fontsize=22)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$a_3$',fontsize=22)
        ax1.xaxis.labelpad=10
        ax1.yaxis.labelpad=12
        ax1.zaxis.labelpad=22
        ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.tick_params(axis='both', which='minor', labelsize=18)
        ax2 = fig.add_subplot(422, projection='3d')
        ax2.plot(x_sim[0:2,0],x_sim[0:2,1],x_sim[0:2,2], 'r', linewidth=3)
        ax2.set_xlabel(r'$a_1$',fontsize=22)
        ax2.set_ylabel(r'$a_2$',fontsize=22)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$a_3$',fontsize=22)
        ax2.xaxis.labelpad=10
        ax2.yaxis.labelpad=12
        ax2.zaxis.labelpad=22
        ax2.grid(True)
        ax3 = fig.add_subplot(423, projection='3d')
        ax3.plot(x_true[0:2,3],x_true[0:2,4],x_true[0:2,5], 'k', linewidth=3)
        ax3.set_xlabel(r'$a_4(t)$',fontsize=22)
        ax3.set_ylabel(r'$a_5(t)$',fontsize=22)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_zticklabels([])
        ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax3.set_zlabel(r'$a_6(t)$',fontsize=22)
        ax3.xaxis.labelpad=10
        ax3.yaxis.labelpad=12
        ax3.zaxis.labelpad=14
        ax3.grid(True)
        ax4 = fig.add_subplot(424, projection='3d')
        ax4.plot(x_sim[0:2,3],x_sim[0:2,4],x_sim[0:2,5], 'b', linewidth=3)
        #ax4.plot(x_sim_predict[:,3],x_sim_predict[:,4],x_sim_predict[:,5], 'r')
        ax4.set_xlabel(r'$a_4(t)$',fontsize=22)
        ax4.set_ylabel(r'$a_5(t)$',fontsize=22)
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_zticklabels([])
        ax4.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax4.set_zlabel(r'$a_6(t)$',fontsize=22)
        ax4.xaxis.labelpad=10
        ax4.yaxis.labelpad=12
        ax4.zaxis.labelpad=14
        ax4.grid(True)
        ax5 = fig.add_subplot(425, projection='3d')
        ax5.plot(x_true[0:2,6],x_true[0:2,7],x_true[0:2,8], 'k', linewidth=3)
        ax5.set_xlabel(r'$a_7(t)$',fontsize=22)
        ax5.set_ylabel(r'$a_8(t)$',fontsize=22)
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.set_zticklabels([])
        ax5.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax5.set_zlabel(r'$a_9(t)$',fontsize=22)
        ax5.xaxis.labelpad=10
        ax5.yaxis.labelpad=12
        ax5.zaxis.labelpad=14
        ax5.grid(True)
        ax6 = fig.add_subplot(426, projection='3d')
        ax6.plot(x_sim[0:2,6],x_sim[0:2,7],x_sim[0:2,8], 'b', linewidth=3)
        #ax4.plot(x_sim_predict[:,3],x_sim_predict[:,4],x_sim_predict[:,5], 'r')
        ax6.set_xlabel(r'$a_7(t)$',fontsize=22)
        ax6.set_ylabel(r'$a_8(t)$',fontsize=22)
        ax6.set_xticklabels([])
        ax6.set_yticklabels([])
        ax6.set_zticklabels([])
        ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax6.set_zlabel(r'$a_9(t)$',fontsize=22)
        ax6.xaxis.labelpad=10
        ax6.yaxis.labelpad=12
        ax6.zaxis.labelpad=14
        ax6.grid(True)
        ax7 = fig.add_subplot(427, projection='3d')
        ax7.plot(x_true[0:2,9],x_true[0:2,10],x_true[0:2,11], 'k', linewidth=3)
        ax7.set_xlabel(r'$a_{10}(t)$',fontsize=22)
        ax7.set_ylabel(r'$a_{11}(t)$',fontsize=22)
        ax7.set_xticklabels([])
        ax7.set_yticklabels([])
        ax7.set_zticklabels([])
        ax7.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax7.set_zlabel(r'$a_{12}(t)$',fontsize=22)
        ax7.xaxis.labelpad=10
        ax7.yaxis.labelpad=12
        ax7.zaxis.labelpad=14
        ax7.grid(True)
        ax8 = fig.add_subplot(428, projection='3d')
        ax8.plot(x_sim[0:2,9],x_sim[0:2,10],x_sim[0:2,11], 'b', linewidth=3)
        #ax4.plot(x_sim_predict[:,3],x_sim_predict[:,4],x_sim_predict[:,5], 'r')
        ax8.set_xlabel(r'$a_{10}(t)$',fontsize=22)
        ax8.set_ylabel(r'$a_{11}(t)$',fontsize=22)
        ax8.set_xticklabels([])
        ax8.set_yticklabels([])
        ax8.set_zticklabels([])
        ax8.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax8.set_zlabel(r'$a_{12}(t)$',fontsize=22)
        ax8.xaxis.labelpad=10
        ax8.yaxis.labelpad=12
        ax8.zaxis.labelpad=14
        ax8.grid(True)
    else:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x_true[0:2,0],x_true[0:2,1],x_true[0:2,2], 'k', linewidth=3)
        ax1.set_xlabel(r'$a_1$',fontsize=22)
        ax1.set_ylabel(r'$a_2$',fontsize=22)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$a_3$',fontsize=22)
        ax1.xaxis.labelpad=10
        ax1.yaxis.labelpad=12
        ax1.zaxis.labelpad=22
        ax1.grid(True)
        ax1.axis('off')
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(x_sim[0:2,0],x_sim[0:2,1],x_sim[0:2,2], 'r', linewidth=3)
        ax2.set_xlabel(r'$a_1$',fontsize=22)
        ax2.set_ylabel(r'$a_2$',fontsize=22)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$a_3$',fontsize=22)
        ax2.xaxis.labelpad=10
        ax2.yaxis.labelpad=12
        ax2.zaxis.labelpad=22
        ax2.grid(True)
        ax2.axis('off')
    ani = animation.FuncAnimation( \
    fig, update_manifold_movie, range(2,len(t_test)), \
    fargs=(x_true,x_sim,t_test),repeat=False, \
    interval=100, blit=False)
    FPS = 25
    ani.save('Pictures/'+prefix+'manifold.mp4',fps=FPS,dpi=100)

def plot_pod_temporal_modes(x,time):
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

    time = time/1.0e3
    plt.figure(figsize=(16,10))
    plt.subplot(3,4,1)
    plt.plot(time,x[:,0]/np.max(abs(x[:,0])),'k')
    leg = plt.legend([r'$a_1$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    #ax.set_xticks([1.0,1.5,2.0])
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,2)
    plt.plot(time,x[:,1]/np.max(abs(x[:,1])),'k')
    leg = plt.legend([r'$a_2$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    #ax.set_xticks([1.0,1.5,2.0])
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,3)
    plt.plot(time,x[:,2]/np.max(abs(x[:,2])),'k')
    leg = plt.legend([r'$a_3$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    #ax.set_xticks([1.0,1.5,2.0])
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,4)
    plt.plot(time,x[:,3]/np.max(abs(x[:,3])),'k')
    leg = plt.legend([r'$a_4$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    #ax.set_xticks([1.0,1.5,2.0])
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,5)
    plt.plot(time,x[:,4]/np.max(abs(x[:,4])),'k')
    leg = plt.legend([r'$a_5$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    #ax.set_xticks([1.0,1.5,2.0])
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,6)
    plt.plot(time,x[:,5]/np.max(abs(x[:,5])),'k')
    leg = plt.legend([r'$a_6$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    #ax.set_xticks([1.0,1.5,2.0])
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,7)
    plt.plot(time,x[:,6]/np.max(abs(x[:,6])),'k')
    leg = plt.legend([r'$a_7$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    #ax.set_xticks([1.0,1.5,2.0])
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,8)
    plt.plot(time,x[:,7]/np.max(abs(x[:,7])),'k')
    leg = plt.legend([r'$a_8$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    #ax.set_xticks([1.5,2.25,3.0,3.75])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,9)
    plt.plot(time,x[:,8]/np.max(abs(x[:,8])),'k')
    leg = plt.legend([r'$a_9$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)  
    #ax.set_xticklabels([])
    #ax.set_xticks([1.0,1.5,2.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,10)
    plt.plot(time,x[:,9]/np.max(abs(x[:,9])),'k')
    leg = plt.legend([r'$a_{10}$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)  
    #ax.set_xticklabels([])
    #ax.set_xticks([1.0,1.5,2.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,11)
    plt.plot(time,x[:,10]/np.max(abs(x[:,10])),'k')
    leg = plt.legend([r'$a_{11}$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)  
    #ax.set_xticks([1.0,1.5,2.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    #ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(3,4,12)
    plt.plot(time,x[:,11]/np.max(abs(x[:,11])),'k')
    leg = plt.legend([r'$a_{12}$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    #ax.set_xticklabels([])
    #ax.set_xticks([1.0,1.5,2.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_yticklabels([])
    plt.grid(True)

    plt.savefig('Pictures/temporal_modes.pdf')
    plt.savefig('Pictures/temporal_modes.eps')
    # Now plot the fourier transforms
    print(np.shape(x))
    #time = time/1.0e3
    time_uniform = np.linspace(time[0],time[-1],len(time)*2)
    x_uniform = np.zeros((len(time)*2,x.shape[1]))
    for i in range(x.shape[1]):
        x_uniform[:,i] = np.interp(time_uniform,time,x[:,i])
    fftx = np.fft.fft(x_uniform,axis=0)/len(time)
    freq = np.fft.fftfreq(len(time_uniform),time_uniform[1]-time_uniform[0])
    fftx = fftx[:len(time)-1,:]
    freq = freq[:len(time)-1]
    plt.figure(figsize=(16,10))
    plt.subplot(3,4,1)
    plt.plot(freq,abs(fftx[:,0]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_1$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,2)
    plt.plot(freq,abs(fftx[:,1]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_2$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,3)
    plt.plot(freq,abs(fftx[:,2]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_3$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,4)
    plt.plot(freq,abs(fftx[:,3]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_4$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,5)
    plt.plot(freq,abs(fftx[:,4]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_5$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,6)
    plt.plot(freq,abs(fftx[:,5]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_6$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,7)
    plt.plot(freq,abs(fftx[:,6]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_7$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([]) #'0',r'$f_{inj}$',r'2$f_{inj}$',r'3$f_{inj}$',r'4$f_{inj}$',r'5$f_{inj}$',])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,8)
    plt.plot(freq,abs(fftx[:,7]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_8$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([]) #'0',r'$f_{inj}$',r'2$f_{inj}$',r'3$f_{inj}$',r'4$f_{inj}$',r'5$f_{inj}$',])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,9)
    plt.plot(freq,abs(fftx[:,8]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_9$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels(['0',r'$f_{inj}$',r'2$f_{inj}$',r'3$f_{inj}$',r'4$f_{inj}$',r'5$f_{inj}$'])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,10)
    plt.plot(freq,abs(fftx[:,9]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_{10}$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels(['0',r'$f_{inj}$',r'2$f_{inj}$',r'3$f_{inj}$',r'4$f_{inj}$',r'5$f_{inj}$',])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,11)
    plt.plot(freq,abs(fftx[:,10]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_{11}$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels(['0',r'$f_{inj}$',r'2$f_{inj}$',r'3$f_{inj}$',r'4$f_{inj}$',r'5$f_{inj}$',])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(3,4,12)
    plt.plot(freq,abs(fftx[:,11]),'k',linewidth=3) 
    leg = plt.legend([r'$\tilde{a}_{12}$'],fontsize=24,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels(['0',r'$f_{inj}$',r'2$f_{inj}$',r'3$f_{inj}$',r'4$f_{inj}$',r'5$f_{inj}$',])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    plt.xlim(0,80)
    plt.grid(True)
    plt.savefig('Pictures/frequency_modes.pdf')
    plt.savefig('Pictures/frequency_modes.eps')
    plot_pairwise(x)

def plot_pod_spatial_modes(X,Y,Z,Q,Vh,wr,U,time):
    R = np.sqrt(X**2+Y**2)
    Z0 = np.isclose(Z,np.ones(len(Z))*min(abs(Z)),rtol=1e-3,atol=1e-3)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    print('Number of points on midplane: ',ind_Z0,len(ind_Z0))
    ri = np.linspace(0,max(R[ind_Z0]),40)
    phii = np.linspace(0,2*np.pi,100)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    n_sample = len(R)
    #U = Q[ind_Z0,:]@np.transpose(Vh)@(np.linalg.inv(wr)[:,0:12])
    U = U.real
    fig = plt.figure(102930912,figsize=(12,12))
    gs1 = gridspec.GridSpec(12, 12)
    gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
    for i in range(6):
        for j in range(12):
            U_sub = U[i*n_sample:(i+1)*n_sample,:]
            U_grid = griddata((X[ind_Z0], Y[ind_Z0]), U_sub[ind_Z0,j], (xi, yi), method='cubic')
            plt.subplot(gs1[i+j*12])
            #plt.title(r'$\chi_1$',fontsize=18)
            plt.pcolor(xi,yi,U_grid/np.nanmax(np.nanmax(U_grid)),cmap='jet',vmin=-1e0,vmax=1e0)
            ax = plt.gca()
            #ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.25, 0.025, 0.5])
    #cbar = fig.colorbar(im,ticks=[-1.0,-0.5,0,0.5,1.0],extend='both',cax=cbar_ax)
    #plt.clim(-1e0,1e0)
    #cbar.ax.tick_params(labelsize=18)
    plt.savefig('Pictures/spatial_modes.pdf',dpi=50)
    plt.savefig('Pictures/spatial_modes.eps',dpi=50)
    # swapping in the DMD modes here
    #U = dmd(Q,24,time)
    #U = 0.5*(U[ind_Z0,::2]+np.conj(U[ind_Z0,::2]))
    #U = U.real
#    Ui1 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,0], (xi, yi), method='cubic')
#    Ui2 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,1], (xi, yi), method='cubic')
#    Ui3 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,2], (xi, yi), method='cubic')
#    Ui4 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,3], (xi, yi), method='cubic')
#    Ui5 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,4], (xi, yi), method='cubic')
#    Ui6 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,5], (xi, yi), method='cubic')
#    Ui7 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,6], (xi, yi), method='cubic')
#    Ui8 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,7], (xi, yi), method='cubic')
#    Ui9 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,8], (xi, yi), method='cubic')
#    Ui10 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,9], (xi, yi), method='cubic')
#    Ui11 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,10], (xi, yi), method='cubic')
#    Ui12 = griddata((X[ind_Z0], Y[ind_Z0]), U[:,11], (xi, yi), method='cubic')
#    print(Ui1,np.max(np.max(Ui1)))
#    fig = plt.figure(102930912,figsize=(10,16))
#    plt.subplot(3,4,1)
#    plt.pcolor(xi,yi,Ui1/np.nanmax(np.nanmax(Ui1)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,2)
#    plt.pcolor(xi,yi,Ui2/np.nanmax(np.nanmax(Ui2)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,3)
#    plt.pcolor(xi,yi,Ui3/np.nanmax(np.nanmax(Ui3)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,4)
#    plt.pcolor(xi,yi,Ui4/np.nanmax(np.nanmax(Ui4)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,5)
#    plt.pcolor(xi,yi,Ui5/np.nanmax(np.nanmax(Ui5)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,6)
#    plt.pcolor(xi,yi,Ui6/np.nanmax(np.nanmax(Ui6)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,7)
#    plt.pcolor(xi,yi,Ui7/np.nanmax(np.nanmax(Ui7)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,8)
#    plt.pcolor(xi,yi,Ui8/np.nanmax(np.nanmax(Ui8)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,9)
#    plt.pcolor(xi,yi,Ui9/np.nanmax(np.nanmax(Ui9)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,10)
#    plt.pcolor(xi,yi,Ui10/np.nanmax(np.nanmax(Ui10)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,11)
#    plt.pcolor(xi,yi,Ui11/np.nanmax(np.nanmax(Ui11)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
#    plt.subplot(3,4,12)
#    plt.pcolor(xi,yi,Ui12/np.nanmax(np.nanmax(Ui12)),cmap='jet',vmin=-1e0,vmax=1e0)
#    ax = plt.gca()
#    ax.axis('off')
# 
#    #fig.subplots_adjust(right=0.8)
#    #cbar_ax = fig.add_axes([0.85, 0.25, 0.025, 0.5])
#    #cbar = fig.colorbar(im,ticks=[-1.0,-0.5,0,0.5,1.0],extend='both',cax=cbar_ax)
#    #plt.clim(-1e0,1e0)
#    #cbar.ax.tick_params(labelsize=18)
#    plt.savefig('Pictures/spatial_dmd_modes.pdf')
#    plt.savefig('Pictures/spatial_dmd_modes.png')
# 
def plot_pairwise(x):
    plt.figure(figsize=(12,12))
    gs1 = gridspec.GridSpec(12, 12)
    gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
    for i in range(12):
        q = i
        for j in range(i+1,12):
            plt.subplot(gs1[i*12+q])
            ax = plt.gca()
            #if j == i+1:
            #    h = plt.ylabel(r'$a_{'+str(i+1)+r'}$',fontsize=12)
            #    h.set_rotation(0)
            #    ax.yaxis.labelpad = 12.0
            #plt.xlabel(r'$a_{'+str(j+1)+r'}$',fontsize=12)
            #ax.xaxis.labelpad = 0.25
            plt.plot(x[:,i],x[:,j],'k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            #plt.title(r'$(a_{'+str(i+1)+r'}$$,a_{'+str(j+1)+r'})$',fontsize=18)
            #leg = plt.legend([r'$(a_'+str(i+1)+r'$$,a_'+str(j+1)+r')$'],fontsize=18,loc='upper right',framealpha=1.0,handlelength=0,handletextpad=0,fancybox=True)
            #leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
            #for item in leg.legendHandles:
            #    item.set_visible(False)
            q = q + 1
    plt.savefig('Pictures/pairwise_plots.pdf',dpi=100)
    plt.savefig('Pictures/pairwise_plots.eps',dpi=100)

def plot_density(time,dens):
    time = time/1.0e3
    dens = dens/1.0e19
    plt.figure(324324123400,figsize=(10,14))
    plt.subplot(6,2,1)
    plt.plot(time,dens[123,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(6,2,2)
    plt.plot(time,dens[8912,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.subplot(6,2,3)
    plt.plot(time,dens[1,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(6,2,4)
    plt.plot(time,dens[23049,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.subplot(6,2,5)
    plt.plot(time,dens[35819,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(6,2,6)
    plt.plot(time,dens[40000,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.subplot(6,2,7)
    plt.plot(time,dens[15555,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(6,2,8)
    plt.plot(time,dens[29993,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.subplot(6,2,9)
    plt.plot(time,dens[31289,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(6,2,10)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.plot(time,dens[12122,:],'k')
    plt.ylim(0.5,3.5)
    ax.set_xticklabels([])
    plt.subplot(6,2,11)
    plt.plot(time,dens[3291,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    #ax.set_xticklabels([])
    plt.subplot(6,2,12)
    plt.plot(time,dens[43920,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_yticklabels([])
    #ax.set_xticklabels([])
    plt.savefig('Pictures/density_samples.pdf')
    plt.savefig('Pictures/density_samples.eps')
