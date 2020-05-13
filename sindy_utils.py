import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
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

def plot_measurement(Qorig,Q_pod,Q_sim,t_test,r):
    """
    Plot (Bx,By,Bz,Bvx,Bvy,Bvz) for a random probe measurement,
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
    Qsize = int(np.shape(Qorig)[0]/6)
    print(Qsize)
    Qorig = Qorig
    Q_pod = Q_pod
    Q_sim = Q_sim
    plt.figure(324353400,figsize=(7,9))
    plt.subplot(6,1,1)
    plt.plot(t_test/1.0e3,Qorig[324,:],'k',linewidth=2,label='True')
    #plt.plot(t_test/1.0e3,Q_pod[324,:],'k--',linewidth=2,label='True, r='+str(r))
    plt.plot(t_test/1.0e3,Q_sim[324,:],color='r',linewidth=2,label='Model, r='+str(r))
    plt.grid(True)
    plt.ylim(-350,350)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,2)
    plt.plot(t_test/1.0e3,Qorig[324+1*Qsize,:],'k',linewidth=2,label=r'True $B_y$')
    #plt.plot(t_test/1.0e3,Q_pod[324+1*Qsize,:],'k--',linewidth=2,label=r'True $B_y$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+1*Qsize,:],color='r',linewidth=2,label=r'Model $B_y$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-700,700)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,3)
    plt.plot(t_test/1.0e3,Qorig[324+2*Qsize,:],'k',linewidth=2,label=r'True $B_z$')
    #plt.plot(t_test/1.0e3,Q_pod[324+2*Qsize,:],'k--',linewidth=2,label=r'True $B_z$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+2*Qsize,:],color='r',linewidth=2,label=r'Model $B_z$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-200,200)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,4)
    plt.plot(t_test/1.0e3,Qorig[324+3*Qsize,:],'k',linewidth=2,label=r'True $V_x$')
    #plt.plot(t_test/1.0e3,Q_pod[324+3*Qsize,:],'k--',linewidth=2,label=r'True $V_x$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+3*Qsize,:],color='r',linewidth=2,label=r'Model $V_x$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-60,60)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,5)
    plt.plot(t_test/1.0e3,Qorig[324+4*Qsize,:],'k',linewidth=2,label=r'True $V_y$')
    #plt.plot(t_test/1.0e3,Q_pod[324+4*Qsize,:],'k--',linewidth=2,label=r'True $V_y$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+4*Qsize,:],color='r',linewidth=2,label=r'Model $V_y$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-40,40)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.subplot(6,1,6)
    plt.plot(t_test/1.0e3,Qorig[324+5*Qsize,:],'k',linewidth=2,label=r'True $V_z$')
    #plt.plot(t_test/1.0e3,Q_pod[324+5*Qsize,:],'k--',linewidth=2,label=r'True $V_z$ with r='+str(r)+' truncation')
    plt.plot(t_test/1.0e3,Q_sim[324+5*Qsize,:],color='r',linewidth=2,label=r'Model $V_z$ with r='+str(r)+' truncation')
    plt.grid(True)
    plt.ylim(-25,25)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig('Pictures/probe_measurement.png',dpi=100)
    plt.savefig('Pictures/probe_measurement.pdf',dpi=100)
    plt.savefig('Pictures/probe_measurement.eps',dpi=100)
    plt.savefig('Pictures/probe_measurement.svg',dpi=100)

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
    r = len(feature_names)
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
        fig, ax = plt.subplots(figsize=(6,30))
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ytable = ax.table(cellText=df.values[r:,:], rowLabels=output_names[r:],cellColours=colors[r:], \
            colLabels=df.columns, loc='center', colWidths=np.ones(12)*0.5/(12))
        ytable.set_fontsize(10)
        #ytable.scale(1, 2)
        plt.savefig('Pictures/SINDy_table_quadratic.pdf')

def update_manifold_movie(frame,x_true,x_sim,t_test,i,j,k):
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
    plt.clf()
    fig = plt.figure(34300,figsize=(16,7))
    #plt.suptitle('t = {:0.2f} (ms)'.format(t_test[frame]/1.0e3),fontsize=20)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_true[0:frame,j],x_true[0:frame,k],zs=-0.4,zdir='x',color='gray', linewidth=3)
    ax1.plot(x_true[0:frame,i],x_true[0:frame,k],zs=-0.4,zdir='y',color='gray', linewidth=3)
    ax1.plot(x_true[0:frame,i],x_true[0:frame,j],zs=-0.4,zdir='z',color='gray', linewidth=3) 
    ax1.plot(x_true[0:frame,i],x_true[0:frame,j],x_true[0:frame,k], 'k', linewidth=5)
    ax1.scatter(x_true[frame-1,i],x_true[frame-1,j],x_true[frame-1,k],s=80, \
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
    ax2.plot(x_sim[0:frame,j],x_sim[0:frame,k],zs=-0.4,zdir='x',color='lightsalmon', linewidth=3)
    ax2.plot(x_sim[0:frame,i],x_sim[0:frame,k],zs=-0.4,zdir='y',color='lightsalmon', linewidth=3)
    ax2.plot(x_sim[0:frame,i],x_sim[0:frame,j],zs=-0.4,zdir='z',color='lightsalmon', linewidth=3)
    ax2.plot(x_sim[0:frame,i],x_sim[0:frame,j],x_sim[0:frame,k], color='r',linewidth=5)
    ax2.scatter(x_sim[frame-1,i],x_sim[frame-1,j],x_sim[frame-1,k],s=80, \
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
    if frame == 200 or frame == 205 or frame == 208 or frame == 210:
        plt.savefig('Pictures/'+str(i)+str(j)+str(k)+'manifold'+str(frame)+'.pdf')

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
    R = np.sqrt(X**2+Y**2)
    Z0 = np.isclose(Z,np.ones(len(Z))*min(abs(Z)),rtol=1e-3,atol=1e-3)
    #Z0 = np.isclose(Z,np.ones(len(Z))*25,rtol=0.81,atol=0.81)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(R[ind_Z0]),40)
    phii = np.linspace(0,2*np.pi,100)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0,frame], (xi, yi), method='cubic')
    Bi_pod = griddata((X[ind_Z0], Y[ind_Z0]), B_pod[ind_Z0,frame], (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0,frame], (xi, yi), method='cubic')
    print(frame)
    plt.clf()
    fig=plt.figure(6888,figsize=(7,20))
    if prefix[1] != 'v':
        subprefix = r'$' + prefix[0] + r'_' + prefix[1] + r'(R,\phi,0,t)$'
    else:
        subprefix = r'$' + prefix[0] + r'_{' + prefix[1] + r',' + prefix[2] + r'}(R,\phi,0,t)$'
    #plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[frame]/1.0e3),fontsize=30)
    plt.subplot(3,1,1)
    #plt.title('Simulation',fontsize=30)
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
    plt.subplot(3,1,2)
    #plt.title('POD',fontsize=30)
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
    plt.subplot(3,1,3)
    #plt.title('Model',fontsize=30)
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
    fig.subplots_adjust(right=0.75)
    #cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    #if prefix[0:2]=='Bv':
    #    cbar = fig.colorbar(im,ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both',cax=cbar_ax)
    #    plt.clim(-5e1,5e1)
    #else:
    #    cbar = fig.colorbar(im,ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both',cax=cbar_ax)
    #    plt.clim(-5e2,5e2)
    #cbar.ax.tick_params(labelsize=18)
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
    plt.subplot(1,2,1)
    plt.plot(S[0:30]/S[0],'ko')
    plt.yscale('log')
    plt.ylim(1e-4,2)
    plt.box(on=None)
    ax = plt.gca()
    ax.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])
    ax.set_yticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
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
        axs[i].set_xticklabels([])
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
        axs[i].set_xticklabels([])
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        axs[i].tick_params(axis='both', which='minor', labelsize=18)
        axs[i].grid(True)
    plt.savefig('Pictures/x.pdf')
    plt.savefig('Pictures/x.eps')

def make_3d_plots(x_true,x_sim,t_test,prefix,i,j,k):
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
    r = np.shape(x_true)[1]
    fig = plt.figure(34300,figsize=(18,10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_true[0:2,i],x_true[0:2,j],x_true[0:2,k], 'k', linewidth=3)
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
    ax2.plot(x_sim[0:2,i],x_sim[0:2,j],x_sim[0:2,k], 'r', linewidth=3)
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
    fargs=(x_true,x_sim,t_test,i,j,k),repeat=False, \
    interval=100, blit=False)
    FPS = 25
    ani.save('Pictures/'+prefix+'manifold'+str(i)+str(j)+str(k)+'.mp4',fps=FPS,dpi=100)

def save_pod_temporal_modes(x,time,S2):
    """
    Save the temporal POD modes to a file to share with others,
    so that we don't have to reload the datasets every time

    Parameters
    ----------
    x: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The temporal POD modes

    time: numpy array of floats
    (M = number of time samples)
        Time range of interest

    S2: numpy array of floats
    (M = number of time samples)
        The singular values
    
    """
    myfile = open('trajectories.txt', 'w')
    for i in range(np.shape(x)[0]):
        for j in range(21):
            if j == 0:
              myfile.write("%2.6f " % time[i])
              myfile.write("%2.6f " % x[i,j])
            elif j+1 != 21:
              myfile.write("%2.6f " % x[i,j])
            else:
              myfile.write("%2.6f\n" % x[i,j])
    myfile.close()
    myfile = open('singular_values.txt', 'w')
    for i in range(len(S2)):
        myfile.write("%2.6f\n" % S2[i])
    myfile.close()

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
    r = np.shape(x)[1]
    time = time/1.0e3
    plt.figure(329,figsize=(85,5))
    gs1 = gridspec.GridSpec(2, 7)
    gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
    plt.subplot(gs1[0])
    plt.plot(time,x[:,0]/np.max(abs(x[:,0])),'k')
    ax = plt.gca()
    ax.set_xticks([1.5,2.75,4.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(gs1[1])
    plt.plot(time,x[:,1]/np.max(abs(x[:,1])),'k')
    ax = plt.gca()
    ax.set_xticks([1.5,2.75,4.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(gs1[2])
    plt.plot(time,x[:,2]/np.max(abs(x[:,2])),'k')
    ax = plt.gca()
    ax.set_xticks([1.5,2.75,4.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(gs1[3])
    plt.plot(time,x[:,3]/np.max(abs(x[:,3])),'k')
    ax = plt.gca()
    ax.set_xticks([1.5,2.75,4.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(gs1[4])
    plt.plot(time,x[:,4]/np.max(abs(x[:,4])),'k')
    ax = plt.gca()
    ax.set_xticks([1.5,2.75,4.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(gs1[5])
    plt.plot(time,x[:,5]/np.max(abs(x[:,5])),'k')
    ax = plt.gca()
    ax.set_xticks([1.5,2.75,4.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.subplot(gs1[6])
    plt.plot(time,x[:,6]/np.max(abs(x[:,6])),'k')
    ax = plt.gca()
    ax.set_xticks([1.5,2.75,4.0])
    ax.set_yticks([-1,0,1])
    plt.ylim(-1.1,1.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.savefig('Pictures/temporal_modes.pdf')
    plt.savefig('Pictures/temporal_modes.eps')
    # Now plot the fourier transforms
    time_uniform = np.linspace(time[0],time[-1],len(time)*2)
    x_uniform = np.zeros((len(time)*2,x.shape[1]))
    for i in range(x.shape[1]):
        x_uniform[:,i] = np.interp(time_uniform,time,x[:,i])
    fftx = np.fft.fft(x_uniform,axis=0)/len(time)
    freq = np.fft.fftfreq(len(time_uniform),time_uniform[1]-time_uniform[0])
    fftx = fftx[:len(time)-1,:]
    freq = freq[:len(time)-1]
    plt.subplot(gs1[7])
    plt.plot(freq,abs(fftx[:,0]),'k',linewidth=3) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(gs1[8])
    plt.plot(freq,abs(fftx[:,1]),'k',linewidth=3) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(gs1[9])
    plt.plot(freq,abs(fftx[:,2]),'k',linewidth=3) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(gs1[10])
    plt.plot(freq,abs(fftx[:,3]),'k',linewidth=3) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(gs1[11])
    plt.plot(freq,abs(fftx[:,4]),'k',linewidth=3) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(gs1[12])
    plt.plot(freq,abs(fftx[:,5]),'k',linewidth=3) 
    ax = plt.gca()
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.xlim(0,80)
    plt.grid(True)
    plt.subplot(gs1[13])
    plt.plot(freq,abs(fftx[:,6]),'k',linewidth=3) 
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([0,14.5,14.5*2,14.5*3,14.5*4,14.5*5])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18) 
    plt.xlim(0,80)
    plt.grid(True)
    plt.savefig('Pictures/frequency_modes.pdf')
    plt.savefig('Pictures/frequency_modes.eps')
    plot_pairwise(x)

def plot_pod_spatial_modes(X,Y,Z,U):
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
    U = U.real
    fig = plt.figure(102930912,figsize=(12,12))
    gs1 = gridspec.GridSpec(12, 12)
    gs1.update(wspace=0.0, hspace=0.0)
    for i in range(6):
        for j in range(12):
            U_sub = U[i*n_sample:(i+1)*n_sample,:]
            U_grid = griddata((X[ind_Z0], Y[ind_Z0]), U_sub[ind_Z0,j], (xi, yi), method='cubic')
            plt.subplot(gs1[i+j*12])
            plt.pcolor(xi,yi,U_grid/np.nanmax(np.nanmax(U_grid)),cmap='jet',vmin=-1e0,vmax=1e0)
            ax = plt.gca()
            #ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('Pictures/spatial_modes.pdf',dpi=50)
    plt.savefig('Pictures/spatial_modes.eps',dpi=50)
# 
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
    plt.figure(figsize=(r,r))
    gs1 = gridspec.GridSpec(r, r)
    gs1.update(wspace=0.0, hspace=0.0)
    for i in range(r):
        q = i
        for j in range(0,r-i):
            plt.subplot(gs1[i,j])
            ax = plt.gca()
            plt.plot(x[:,i],x[:,r-j-1],'k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            q = q + 1
    plt.savefig('Pictures/pairwise_plots.pdf',dpi=100)
    plt.savefig('Pictures/pairwise_plots.eps',dpi=100)

def plot_density(time,dens):
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
    plt.subplot(6,2,12)
    plt.plot(time,dens[43920,:],'k')
    plt.ylim(0.5,3.5)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.savefig('Pictures/density_samples.pdf')
    plt.savefig('Pictures/density_samples.eps')

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
    R = X**2+Y**2
    Z0 = np.isclose(Z,np.ones(len(Z))*min(abs(Z)),rtol=1e-3,atol=1e-3)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(R[ind_Z0]),40)
    phii = np.linspace(0,2*np.pi,100)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    Bi = griddata((X[ind_Z0], Y[ind_Z0]), B_true[ind_Z0,0], (xi, yi), method='cubic')
    Bi_pod = griddata((X[ind_Z0], Y[ind_Z0]), B_pod[ind_Z0,0], (xi, yi), method='cubic')
    Bi_sim = griddata((X[ind_Z0], Y[ind_Z0]), B_sim[ind_Z0,0], (xi, yi), method='cubic')
    plt.clf()
    fig = plt.figure(6888,figsize=(7,20))
    if prefix[1] != 'A':
        subprefix = r'$' + prefix[0] + r'_' + prefix[1] + r'(R,\phi,0,t)$'
    else:
        subprefix = r'$' + prefix[0] + r'_{' + prefix[1] + r',' + prefix[2] + r'}(R,\phi,0,t)$'
    #plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[0]/1.0e3),fontsize=20)
    plt.subplot(3,1,1)
    plt.contourf(xi,yi,Bi*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.contourf(xi,yi,Bi_pod*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3,1,3)
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

def make_poloidal_movie(X,Y,Z,B_true,B_pod,B_sim,t_test,prefix):
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
    R = X**2+Y**2
    X0 = np.ravel(np.where(np.array(X) > 0.0))
    Y0 = np.isclose(Y,np.ones(len(Y))*min(abs(Y)),rtol=1e-3,atol=1e-3)
    ind_Y0 = [i for i, p in enumerate(Y0) if p]
    ind_Y0 = np.intersect1d(X0,ind_Y0)
    #print(ind_Y0)
    xi = np.linspace(min(X[ind_Y0]),max(X[ind_Y0]))
    zi = np.linspace(min(Z[ind_Y0]),max(Z[ind_Y0]))
    xi,zi = np.meshgrid(xi,zi)
    Bi = griddata((X[ind_Y0], Z[ind_Y0]), B_true[ind_Y0,0], (xi, zi), method='cubic')
    Bi_pod = griddata((X[ind_Y0], Z[ind_Y0]), B_pod[ind_Y0,0], (xi, zi), method='cubic')
    Bi_sim = griddata((X[ind_Y0], Z[ind_Y0]), B_sim[ind_Y0,0], (xi, zi), method='cubic')
    plt.clf()
    fig = plt.figure(6889,figsize=(7,20))
    if prefix[1] != 'A':
        subprefix = r'$' + prefix[0] + r'_' + prefix[1] + r'(R,\phi,0,t)$'
    else:
        subprefix = r'$' + prefix[0] + r'_{' + prefix[1] + r',' + prefix[2] + r'}(R,\phi,0,t)$'
    #plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[0]/1.0e3),fontsize=20)
    plt.subplot(3,1,1)
    plt.contourf(xi,zi,Bi*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.contourf(xi,zi,Bi_pod*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3,1,3)
    plt.contourf(xi,zi,Bi_sim*1.0e4,cmap='jet')
    ax = plt.gca()
    ax.axis('off')
    #plt.colorbar()
    ani = animation.FuncAnimation( \
    fig, update_poloidal_movie, range(0,len(t_test),1), \
    fargs=(X,Y,Z,B_true,B_pod,B_sim,t_test,prefix),repeat=False, \
    interval=100, blit=False)
    FPS = 30
    ani.save('Pictures/'+prefix+'_poloidal_contour.mp4',fps=FPS,dpi=200)

def update_poloidal_movie(frame,X,Y,Z,B_true,B_pod,B_sim,t_test,prefix):
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
    R = np.sqrt(X**2+Y**2)
    X0 = np.ravel(np.where(np.array(X) > 0.0))
    Y0 = np.isclose(Y,np.ones(len(Y))*min(abs(Y)),rtol=1e-3,atol=1e-3)
    ind_Y0 = [i for i, p in enumerate(Y0) if p]
    ind_Y0 = np.intersect1d(X0,ind_Y0)
    xi = np.linspace(min(X[ind_Y0]),max(X[ind_Y0]),100)
    zi = np.linspace(min(Z[ind_Y0]),max(Z[ind_Y0]),100)
    xi,zi = np.meshgrid(xi,zi)
    Bi = griddata((X[ind_Y0], Z[ind_Y0]), B_true[ind_Y0,frame], (xi, zi), method='cubic')
    Bi_pod = griddata((X[ind_Y0], Z[ind_Y0]), B_pod[ind_Y0,frame], (xi, zi), method='cubic')
    Bi_sim = griddata((X[ind_Y0], Z[ind_Y0]), B_sim[ind_Y0,frame], (xi, zi), method='cubic')
    print(frame)
    plt.clf()
    fig=plt.figure(6889,figsize=(7,20))
    if prefix[1] != 'v':
        subprefix = r'$' + prefix[0] + r'_' + prefix[1] + r'(R,\phi,0,t)$'
    else:
        subprefix = r'$' + prefix[0] + r'_{' + prefix[1] + r',' + prefix[2] + r'}(R,\phi,0,t)$'
    #plt.suptitle(subprefix + '\n t = {:0.2f} (ms)'.format(t_test[frame]/1.0e3),fontsize=30)
    plt.subplot(3,1,1)
    #plt.title('Simulation',fontsize=30)
    if prefix[0:2]=='Bv':
        plt.pcolor(xi,zi,Bi*1.0e4,cmap='jet',vmin=-5e1,vmax=5e1)
        #cbar = plt.colorbar(ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both')
        #plt.clim(-5e1,5e1)
    else:
        plt.pcolor(xi,zi,Bi*1.0e4,cmap='jet',vmin=-5e2,vmax=5e2)
        #cbar = plt.colorbar(ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both')
        #plt.clim(-5e2,5e2)
    #cbar.ax.tick_params(labelsize=18)
    # To plot the measurement locations
    #plt.scatter(X,Y,s=2,c='k')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3,1,2)
    #plt.title('POD',fontsize=30)
    if prefix[0:2]=='Bv':
        plt.pcolor(xi,zi,Bi_pod*1.0e4,cmap='jet',vmin=-5e1,vmax=5e1)
        #cbar = plt.colorbar(ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both')
        #plt.clim(-5e1,5e1)
        #cbar = plt.colorbar() #ticks=[-1e2,-5e1,0,5e1,1e2],extend='both')
        #plt.clim(-1e2,1e2)
    else:
        plt.pcolor(xi,zi,Bi_pod*1.0e4,cmap='jet',vmin=-5e2,vmax=5e2)
        #cbar = plt.colorbar(ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both')
        #plt.clim(-5e2,5e2)
        #cbar = plt.colorbar() #ticks=[-1e1,-5,0,5,1e1],extend='both')
        #plt.clim(-1e1,1e1)
    #cbar.ax.tick_params(labelsize=18)
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(3,1,3)
    #plt.title('Model',fontsize=30)
    if prefix[0:2]=='Bv':
        im=plt.pcolor(xi,zi,Bi_sim*1.0e4,cmap='jet',vmin=-5e1,vmax=5e1)
        #cbar = plt.colorbar(ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both')
        #plt.clim(-5e1,5e1)
        #cbar = plt.colorbar() #ticks=[-1e2,-5e1,0,5e1,1e2],extend='both')
        #plt.clim(-1e2,1e2)
    else:
        im=plt.pcolor(xi,zi,Bi_sim*1.0e4,cmap='jet',vmin=-5e2,vmax=5e2)
        #cbar = plt.colorbar(ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both')
        #plt.clim(-5e2,5e2)
        #cbar = plt.colorbar() #ticks=[-1e1,-5,0,5,1e1],extend='both')
        #plt.clim(-1e1,1e1)
    ax = plt.gca()
    ax.axis('off')
    fig.subplots_adjust(right=0.75)
    #cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    #if prefix[0:2]=='Bv':
    #    cbar = fig.colorbar(im,ticks=[-5e1,-2.5e1,0,2.5e1,5e1],extend='both',cax=cbar_ax)
    #    plt.clim(-5e1,5e1)
    #else:
    #    cbar = fig.colorbar(im,ticks=[-5e2,-2.5e2,0,2.5e2,5e2],extend='both',cax=cbar_ax)
    #    plt.clim(-5e2,5e2)
    #cbar.ax.tick_params(labelsize=18)
    if frame == 0:
        plt.savefig('Pictures/'+prefix+'_poloidal_contours.pdf')

