import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.animation as animation

def inner_product(Q,r):
    Qr = np.zeros(np.shape(Q))
    for i in range(np.shape(Q)[1]):
        Qr[:,i] = Q[:,i]*r
    print('Qr ',np.shape(Qr))
    return np.transpose(Q)@Qr

def plot_energy(time,inner_mat):
    plt.figure(1000)
    plt.plot(time,np.diag(inner_mat))

def plot_Hc(time,Hc_mat,r):
    Hcr_mat = np.zeros(np.shape(Hc_mat))
    for i in range(np.shape(Hc_mat)[1]):
        Hcr_mat[:,i] = Hc_mat[:,i]*r
    inner_mat = np.transpose(Hc_mat)@Hcr_mat
    print('Hc: ',np.shape(inner_mat))
    plt.figure(2000)
    plt.plot(time,np.diag(inner_mat))

def plot_measurement_fits(Q,Qpred,Qfit,t_pred):
    # do nothing for now
    print(np.shape(Q))
    plt.figure(324353400)
    plt.subplot(4,1,1)
    plt.plot(t_pred/1.0e3,Q[324,:],'k')
    plt.plot(t_pred/1.0e3,Qpred[324,:],'b--')
    plt.plot(t_pred/1.0e3,Qfit[324,:],'r--')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(4,1,2)
    plt.plot(t_pred/1.0e3,Q[325,:],'k')
    plt.plot(t_pred/1.0e3,Qpred[325,:],'b--')
    plt.plot(t_pred/1.0e3,Qfit[325,:],'r--')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(4,1,3)
    plt.plot(t_pred/1.0e3,Q[326,:],'k')
    plt.plot(t_pred/1.0e3,Qpred[326,:],'b--')
    plt.plot(t_pred/1.0e3,Qfit[326,:],'r--')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(4,1,4)
    plt.plot(t_pred/1.0e3,Q[327,:],'k')
    plt.plot(t_pred/1.0e3,Qpred[327,:],'b--')
    plt.plot(t_pred/1.0e3,Qfit[327,:],'r--')
    plt.grid(True)
    ax = plt.gca()
    #ax.set_xticklabels([])
    plt.savefig('Pictures/Bx_fit.png')
    plt.savefig('Pictures/Bx_fit.pdf')
    plt.savefig('Pictures/Bx_fit.eps')
    plt.savefig('Pictures/Bx_fit.svg')

def make_table(sindy_model,feature_names):
    output_names = sindy_model.get_feature_names()
    coefficients = sindy_model.coefficients()
    #output_names = []
    #for i in range(len(feature_names)):
    #    for j in range(i,len(feature_names)):
    #        output_names.append(feature_names[i]+feature_names[j])
    #output_names = feature_names
    print(np.shape(coefficients),np.shape(output_names),np.shape(feature_names))
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
    ytable.set_fontsize(14)
    ytable.scale(1, 2)
    #fig.tight_layout()
    plt.savefig('SINDy_table.pdf')

def update_manifold_movie(frame,x_pred,x_test_sim,time,num_POD):
    print(frame)
    plt.clf()
    fig = plt.figure(34300,figsize=(16,7))
    plt.suptitle('t = {:0.2f} (ms)'.format(time[frame]/1.0e3),fontsize=20)
    if num_POD==3:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x_pred[0:frame,0],x_pred[0:frame,1],x_pred[0:frame,2], 'k')
        ax1.scatter(x_pred[frame-1,0],x_pred[frame-1,1],x_pred[frame-1,2], \
            color='k', marker='o')
        ax1.azim = frame/7.0
        ax1.elev = frame/11.0
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
        ax2.plot(x_test_sim[0:frame,0],x_test_sim[0:frame,1],x_test_sim[0:frame,2], 'b--')
        ax2.scatter(x_test_sim[frame-1,0],x_test_sim[frame-1,1],x_test_sim[frame-1,2], \
            color='b', marker='o')
        ax2.azim = frame/7.0
        ax2.elev = frame/11.0
        #ax2.plot(x_test_sim_predict[:,0],x_test_sim_predict[:,1],x_test_sim_predict[:,2], 'r--')
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
        ax1.plot(x_pred[0:frame,0],x_pred[0:frame,1],x_pred[0:frame,2], 'k')
        ax1.scatter(x_pred[frame-1,0],x_pred[frame-1,1],x_pred[frame-1,2], \
            color='k', marker='o')
        ax1.azim = frame/7.0
        ax1.elev = frame/11.0
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
        ax2 = fig.add_subplot(422, projection='3d')
        ax2.plot(x_test_sim[0:frame,0],x_test_sim[0:frame,1],x_test_sim[0:frame,2], 'b--')
        ax2.scatter(x_test_sim[frame-1,0],x_test_sim[frame-1,1],x_test_sim[frame-1,2], \
            color='b', marker='o')
        ax2.azim = frame/7.0
        ax2.elev = frame/11.0
        #ax2.plot(x_test_sim_predict[:,0],x_test_sim_predict[:,1],x_test_sim_predict[:,2], 'r--')
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
        #
        ax3 = fig.add_subplot(423, projection='3d')
        ax3.plot(x_pred[0:frame,3],x_pred[0:frame,4],x_pred[0:frame,5], 'k')
        ax3.scatter(x_pred[frame-1,3],x_pred[frame-1,4],x_pred[frame-1,5], \
            color='k', marker='o')
        ax3.azim = frame/7.0
        ax3.elev = frame/11.0
        ax3.set_xlabel(r'$\varphi_{4}(t)$',fontsize=16)
        ax3.set_ylabel(r'$\varphi_{5}(t)$',fontsize=16)
        ax3.set_xlim(-0.3,0.3)
        ax3.set_ylim(-0.3,0.3)
        ax3.set_zlim(-0.3,0.3)
        ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax3.set_zlabel(r'$\varphi_{6}(t)$',fontsize=16) #,rotation=90)
        ax3.xaxis.labelpad=4
        ax3.yaxis.labelpad=6
        ax3.zaxis.labelpad=14
        ax3.grid(True)
        ax4 = fig.add_subplot(424, projection='3d')
        ax4.plot(x_test_sim[0:frame,3],x_test_sim[0:frame,4],x_test_sim[0:frame,5], 'b--')
        ax4.scatter(x_test_sim[frame-1,3],x_test_sim[frame-1,4],x_test_sim[frame-1,5], \
            color='b', marker='o')
        ax4.azim = frame/7.0
        ax4.elev = frame/11.0
        ax4.set_xlabel(r'$\varphi_{4}(t)$',fontsize=16)
        ax4.set_ylabel(r'$\varphi_{5}(t)$',fontsize=16)
        ax4.set_xlim(-0.3,0.3)
        ax4.set_ylim(-0.3,0.3)
        ax4.set_zlim(-0.3,0.3)
        ax4.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax4.set_zlabel(r'$\varphi_{6}(t)$',fontsize=16) #,rotation=90)
        ax4.xaxis.labelpad=4
        ax4.yaxis.labelpad=6
        ax4.zaxis.labelpad=14
        ax4.grid(True)
        #
        ax5 = fig.add_subplot(425, projection='3d')
        ax5.plot(x_pred[0:frame,6],x_pred[0:frame,7],x_pred[0:frame,8], 'k')
        ax5.scatter(x_pred[frame-1,6],x_pred[frame-1,7],x_pred[frame-1,8], \
            color='k', marker='o')
        ax5.azim = frame/7.0
        ax5.elev = frame/11.0
        ax5.set_xlabel(r'$\varphi_{7}(t)$',fontsize=16)
        ax5.set_ylabel(r'$\varphi_{8}(t)$',fontsize=16)
        ax5.set_xlim(-0.3,0.3)
        ax5.set_ylim(-0.3,0.3)
        ax5.set_zlim(-0.3,0.3)
        ax5.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax5.set_zlabel(r'$\varphi_{9}(t)$',fontsize=16) #,rotation=90)
        ax5.xaxis.labelpad=4
        ax5.yaxis.labelpad=6
        ax5.zaxis.labelpad=14
        ax5.grid(True)
        ax6 = fig.add_subplot(426, projection='3d')
        ax6.plot(x_test_sim[0:frame,6],x_test_sim[0:frame,7],x_test_sim[0:frame,8], 'b--')
        ax6.scatter(x_test_sim[frame-1,6],x_test_sim[frame-1,7],x_test_sim[frame-1,8], \
            color='b', marker='o')
        ax6.azim = frame/7.0
        ax6.elev = frame/11.0
        ax6.set_xlabel(r'$\varphi_{7}(t)$',fontsize=16)
        ax6.set_ylabel(r'$\varphi_{8}(t)$',fontsize=16)
        ax6.set_xlim(-0.3,0.3)
        ax6.set_ylim(-0.3,0.3)
        ax6.set_zlim(-0.3,0.3)
        ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax6.set_zlabel(r'$\varphi_{9}(t)$',fontsize=16) #,rotation=90)
        ax6.xaxis.labelpad=4
        ax6.yaxis.labelpad=6
        ax6.zaxis.labelpad=14
        ax6.grid(True)
        #
        ax7 = fig.add_subplot(427, projection='3d')
        ax7.plot(x_pred[0:frame,9],x_pred[0:frame,10],x_pred[0:frame,11], 'k')
        ax7.scatter(x_pred[frame-1,9],x_pred[frame-1,10],x_pred[frame-1,11], \
            color='k', marker='o')
        ax7.azim = frame/7.0
        ax7.elev = frame/11.0
        ax7.set_xlabel(r'$\varphi_{10}(t)$',fontsize=16)
        ax7.set_ylabel(r'$\varphi_{11}(t)$',fontsize=16)
        ax7.set_xlim(-0.3,0.3)
        ax7.set_ylim(-0.3,0.3)
        ax7.set_zlim(-0.3,0.3)
        ax7.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax7.set_zlabel(r'$\varphi_{12}(t)$',fontsize=16) #,rotation=90)
        ax7.xaxis.labelpad=4
        ax7.yaxis.labelpad=6
        ax7.zaxis.labelpad=14
        ax7.grid(True)
        ax8 = fig.add_subplot(428, projection='3d')
        ax8.plot(x_test_sim[0:frame,9],x_test_sim[0:frame,10],x_test_sim[0:frame,11], 'b--')
        ax8.scatter(x_test_sim[frame-1,9],x_test_sim[frame-1,10],x_test_sim[frame-1,11], \
            color='b', marker='o')
        ax8.azim = frame/7.0
        ax8.elev = frame/11.0
        ax8.set_xlabel(r'$\varphi_{10}(t)$',fontsize=16)
        ax8.set_ylabel(r'$\varphi_{11}(t)$',fontsize=16)
        ax8.set_xlim(-0.3,0.3)
        ax8.set_ylim(-0.3,0.3)
        ax8.set_zlim(-0.3,0.3)
        ax8.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax8.set_zlabel(r'$\varphi_{12}(t)$',fontsize=16) #,rotation=90)
        ax8.xaxis.labelpad=4
        ax8.yaxis.labelpad=6
        ax8.zaxis.labelpad=14
        ax8.grid(True)

def make_contour_movie(x,y,z,B,B_sim,time,prefix):
    r = x**2+y**2
    Z0 = np.isclose(z,np.ones(len(z))*min(abs(z)),rtol=1e-3,atol=1e-3)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(r[ind_Z0]),200)
    phii = np.linspace(0,2*np.pi,64)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    Bi = griddata((x[ind_Z0], y[ind_Z0]), B[ind_Z0,0], (xi, yi), method='cubic')
    Bi_sim = griddata((x[ind_Z0], y[ind_Z0]), B_sim[ind_Z0,0], (xi, yi), method='cubic')
    plt.clf()
    fig = plt.figure(6,figsize=(10,7))
    plt.suptitle('t = {:0.2f} (ms)'.format(time[0]))
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
    fig, update_contour_movie, range(0,len(time),1), \
    fargs=(x,y,z,B,B_sim,time,prefix),repeat=False, \
    interval=100, blit=False)
    FPS = 30
    ani.save('Pictures/'+prefix+'_contour.mp4',fps=FPS,dpi=200)

def update_contour_movie(frame,x,y,z,B,B_sim,time,prefix):
    r = np.sqrt(x**2+y**2)
    Z0 = np.isclose(z,np.ones(len(z))*min(abs(z)),rtol=1e-3,atol=1e-3)
    ind_Z0 = [i for i, p in enumerate(Z0) if p]
    ri = np.linspace(0,max(r[ind_Z0]),200)
    phii = np.linspace(0,2*np.pi,64)
    ri,phii = np.meshgrid(ri,phii)
    xi = ri*np.cos(phii)
    yi = ri*np.sin(phii)
    Bi = griddata((x[ind_Z0], y[ind_Z0]), B[ind_Z0,frame], (xi, yi), method='cubic')
    Bi_sim = griddata((x[ind_Z0], y[ind_Z0]), B_sim[ind_Z0,frame], (xi, yi), method='cubic')
    print(frame)
    plt.clf()
    plt.figure(6,figsize=(10,7))
    plt.suptitle('t = {:0.2f} (ms)'.format(time[frame]/1.0e3))
    plt.subplot(2,1,1)
    plt.title('Simulation test data')
    if prefix[0]=='B':
        plt.pcolor(xi,yi,Bi,cmap='plasma',vmin=-1e5,vmax=1e5)
        cbar = plt.colorbar(ticks=[-1e5,-5e4,0,5e4,1e5],extend='both')
        plt.clim(-1e5,1e5)
    else:
        plt.pcolor(xi,yi,Bi,cmap='plasma',vmin=-1e4,vmax=1e4)
        cbar = plt.colorbar(ticks=[-1e4,-5e3,0,5e3,1e4],extend='both')
        plt.clim(-1e4,1e4)
    # To plot the measurement locations
    #plt.scatter(x,y,s=2,c='k')
    ax = plt.gca()
    ax.axis('off')
    plt.subplot(2,1,2)
    plt.title('Identified model')
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
