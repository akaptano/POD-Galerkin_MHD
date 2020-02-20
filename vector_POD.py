import numpy as np
from matplotlib import pyplot as plt
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference,SmoothedFiniteDifference
from sindy_utils import plot_energy, make_table, update_manifold_movie
from pysindy.utils.pareto import pareto_curve
from scipy.integrate import solve_ivp,odeint
import matplotlib.animation as animation

def vector_POD(svd_mat,time,poly_order,threshold,num_POD):
    plot_energy(time,svd_mat)
    print(np.shape(svd_mat))
    V,S,Vh = np.linalg.svd(svd_mat,full_matrices=False)
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
    plt.savefig('BOD_spectrum.pdf')
    plt.savefig('BOD_spectrum.png')
    plt.savefig('BOD_spectrum.eps')
    plt.savefig('BOD_spectrum.svg')
    print("% field in first num_POD modes = ",sum(np.sqrt(S[0:num_POD]))/sum(np.sqrt(S)))
    print("% energy in first num_POD modes = ",sum(S[0:num_POD])/sum(S))
    #ax = fig.add_subplot(122, projection='3d')
    vh = np.zeros((num_POD,np.shape(Vh)[1]))
    feature_names = []
    for i in range(num_POD):
        #vh[i,:] = Vh[i,:]*S[i]**2/sum(S[0:num_POD]**2*np.amax(abs(Vh),axis=1)[0:num_POD])
        vh[i,:] = Vh[i,:]/sum(np.amax(abs(Vh),axis=1)[0:num_POD])
        feature_names.append(r'$\varphi_{:d}$'.format(i+1))
    #ax.plot(vh[0,:],vh[1,:],vh[2,:],'k')

    x_test = np.transpose(vh)
    print(np.shape(x_test))
    model = SINDy(optimizer=STLSQ(threshold=threshold), \
        feature_library=PolynomialLibrary(degree=poly_order), \
        differentiation_method=SmoothedFiniteDifference(drop_endpoints=True), \
        feature_names=feature_names)
    print(np.shape(x_test))
    print(np.shape(time))
    tfac = 3.0/5.0
    t_fit = time[:int(len(time)*tfac)]
    x_fit = x_test[:int(len(time)*tfac),:]
    x0_fit = x_test[0,:]
    t_pred = time[int(len(time)*tfac):]
    x_pred = x_test[int(len(time)*tfac):,:]
    x0_pred = x_test[int(len(time)*tfac),:]
    #t_test = time[int(len(time)*tfac)-1:]
    t_test = time #np.linspace(time[0],time[len(time)-1],len(time)*2)
    model.fit(x_fit, t=t_fit)
    model.print()
    print(model.coefficients())
    x0_test = x_test[0,:]
    #x_test_sim = model.simulate(x0_test,t_span,integrator=solve_ivp)
    integrator_kws = {'full_output': True}
    x_test_recon,output = model.simulate(x0_fit,t_fit, \
        integrator=odeint,stop_condition=None,full_output=True, \
        rtol=1e-15,h0=1e-20,tcrit=[1702])
    #x0_pred = x_test_recon[-1,:]
    x_test_sim,output = model.simulate(x0_pred,t_pred, \
        integrator=odeint,stop_condition=None,full_output=True, \
        rtol=1e-15,h0=1e-20,tcrit=[1702]) #,hmax=1e-2) #,atol=1e-4)
    #x_test_solver = model.simulate_stiff(time[0],x0_test,time[len(time)-1])
    #q = 0
    #x_test_sim = []
    #t_test_sim = []
    #while x_test_solver.status != 'finished' and x_test_solver.status != 'failed':
    #    x_test_solver.step()
    #    x_test_sim.append(x_test_solver.y)
    #    t_test_sim.append(x_test_solver.t)
    #    q = q + 1
    #print(x_test_solver.status,x_test_sim,np.shape(x_test_sim))
    x_dot_test_computed = model.differentiate(x_test, t=time)
    x_dot_test_recon = model.predict(x_fit)
    x_dot_test_predicted = model.predict(x_pred)
    print('Model score: %f' % model.score(x_test, t=time))
    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7,9))
    for i in range(x_test.shape[1]):
        axs[i].plot(time/1.0e3, x_dot_test_computed[:,i], 'k', label='numerical derivative')
        axs[i].plot(t_fit/1.0e3, x_dot_test_recon[:,i], 'r--', label='model prediction')
        axs[i].plot(t_pred/1.0e3, x_dot_test_predicted[:,i], 'b--', label='model forecast')
        #axs[i].legend()
        axs[i].set_yticklabels([])
        #axs[i].set(xlabel='t', ylabel='$\dot x_{}$'.format(i))
        axs[i].grid(True)
    plt.savefig('xdot.pdf')
    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7,9))
    #x_test_sim = np.array(x_test_sim)
    for i in range(x_test.shape[1]):
        #axs[i].plot(time/1.0e3, x_test[:,i], 'k', label='true simulation')
        axs[i].plot(t_pred/1.0e3, x_pred[:,i], 'k', label='true simulation')
        #axs[i].plot(t_fit/1.0e3, x_test_recon[:,i], 'r--', label='model reconstruction')
        axs[i].plot(t_pred/1.0e3, x_test_sim[:,i], 'b--', label='model forecast')
        axs[i].set_yticklabels([])
        #axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))
        axs[i].grid(True)
    plt.savefig('x.pdf')
    if (num_POD == 3):
        fig = plt.figure(34300,figsize=(16,10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x_pred[0:2,0],x_pred[0:2,1],x_pred[0:2,2], 'k')
        ax1.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        #ax1.set_xlim(-0.08,0.08)
        #ax1.set_ylim(-0.08,0.08)
        #ax1.set_zlim(-0.08,0.08)
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
        ax2.plot(x_test_sim[0:2,0],x_test_sim[0:2,1],x_test_sim[0:2,2], 'r--')
        ax2.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        #ax2.set_xlim(-0.08,0.08)
        #ax2.set_ylim(-0.08,0.08)
        #ax2.set_zlim(-0.08,0.08)
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
        fig, update_manifold_movie, range(2,len(time)-int(len(time)*tfac)), \
        fargs=(x_pred,x_test_sim,t_pred),repeat=False, \
        interval=100, blit=False)
        FPS = 25
        ani.save('manifold.mp4',fps=FPS)

    elif (num_POD > 3):
        fig = plt.figure(figsize=(16,10))
        ax1 = fig.add_subplot(421, projection='3d')
        ax1.plot(x_test[:,0],x_test[:,1],x_test[:,2], 'k')
        ax1.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        ax1.set_xlim(-0.08,0.08)
        ax1.set_ylim(-0.08,0.08)
        ax1.set_zlim(-0.08,0.08)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax1.set_zlabel(r'$\varphi_3(t)$') #,rotation=90)
        ax1.xaxis.labelpad=-11
        ax1.yaxis.labelpad=-12
        ax1.zaxis.labelpad=-14
        ax1.grid(True)
        ax2 = fig.add_subplot(422, projection='3d')
        ax2.plot(x_test_recon[:,0],x_test_recon[:,1],x_test_recon[:,2], 'r--')
        ax2.plot(x_test_sim[:,0],x_test_sim[:,1],x_test_sim[:,2], 'b--')
        #ax2.plot(x_test_sim_predict[:,0],x_test_sim_predict[:,1],x_test_sim_predict[:,2], 'r--')
        ax2.set(xlabel=r'$\varphi_1(t)$', ylabel=r'$\varphi_2(t)$')
        ax2.set_xlim(-0.08,0.08)
        ax2.set_ylim(-0.08,0.08)
        ax2.set_zlim(-0.08,0.08)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel(r'$\varphi_3(t)$') #,rotation=90)
        ax2.xaxis.labelpad=-11
        ax2.yaxis.labelpad=-12
        ax2.zaxis.labelpad=-14
        ax2.grid(True)
    if (num_POD >= 6):
        ax3 = fig.add_subplot(423, projection='3d')
        ax3.plot(x_test[:,3],x_test[:,4],x_test[:,5], 'k')
        ax3.set(xlabel=r'$\varphi_4(t)$', ylabel=r'$\varphi_5(t)$')
        ax3.set_xlim(-0.08,0.08)
        ax3.set_ylim(-0.08,0.08)
        ax3.set_zlim(-0.08,0.08)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_zticklabels([])
        ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax3.set_zlabel(r'$\varphi_6(t)$') #,rotation=90)
        ax3.xaxis.labelpad=-11
        ax3.yaxis.labelpad=-12
        ax3.zaxis.labelpad=-14
        ax3.grid(True)
        ax4 = fig.add_subplot(424, projection='3d')
        ax4.plot(x_test_recon[:,3],x_test_recon[:,4],x_test_recon[:,5], 'r--')
        ax4.plot(x_test_sim[:,3],x_test_sim[:,4],x_test_sim[:,5], 'b--')
        #ax4.plot(x_test_sim_predict[:,3],x_test_sim_predict[:,4],x_test_sim_predict[:,5], 'r--')
        ax4.set(xlabel=r'$\varphi_4(t)$', ylabel=r'$\varphi_5(t)$')
        ax4.set_xlim(-0.08,0.08)
        ax4.set_ylim(-0.08,0.08)
        ax4.set_zlim(-0.08,0.08)
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_zticklabels([])
        ax4.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax4.set_zlabel(r'$\varphi_6(t)$') #,rotation=90)
        ax4.xaxis.labelpad=-11
        ax4.yaxis.labelpad=-12
        ax4.zaxis.labelpad=-14
        ax4.grid(True)
    if (num_POD >= 9):
        ax5 = fig.add_subplot(425, projection='3d')
        ax5.plot(x_test[:,6],x_test[:,7],x_test[:,8], 'k')
        ax5.set(xlabel=r'$\varphi_7(t)$', ylabel=r'$\varphi_8(t)$')
        ax5.set_xlim(-0.08,0.08)
        ax5.set_ylim(-0.08,0.08)
        ax5.set_zlim(-0.08,0.08)
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.set_zticklabels([])
        ax5.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax5.set_zlabel(r'$\varphi_9(t)$') #,rotation=90)
        ax5.xaxis.labelpad=-11
        ax5.yaxis.labelpad=-12
        ax5.zaxis.labelpad=-14
        ax5.grid(True)
        ax6 = fig.add_subplot(426, projection='3d')
        ax6.plot(x_test_recon[:,6],x_test_recon[:,7],x_test_recon[:,8], 'r--')
        ax6.plot(x_test_sim[:,6],x_test_sim[:,7],x_test_sim[:,8], 'b--')
        #ax4.plot(x_test_sim_predict[:,3],x_test_sim_predict[:,4],x_test_sim_predict[:,5], 'r--')
        ax6.set(xlabel=r'$\varphi_7(t)$', ylabel=r'$\varphi_8(t)$')
        ax6.set_xlim(-0.08,0.08)
        ax6.set_ylim(-0.08,0.08)
        ax6.set_zlim(-0.08,0.08)
        ax6.set_xticklabels([])
        ax6.set_yticklabels([])
        ax6.set_zticklabels([])
        ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax6.set_zlabel(r'$\varphi_9(t)$') #,rotation=90)
        ax6.xaxis.labelpad=-11
        ax6.yaxis.labelpad=-12
        ax6.zaxis.labelpad=-14
        ax6.grid(True)
    if (num_POD >= 9):
        ax7 = fig.add_subplot(427, projection='3d')
        ax7.plot(x_test[:,9],x_test[:,10],x_test[:,11], 'k')
        ax7.set(xlabel=r'$\varphi_{10}(t)$', ylabel=r'$\varphi_{11}(t)$')
        ax7.set_xlim(-0.08,0.08)
        ax7.set_ylim(-0.08,0.08)
        ax7.set_zlim(-0.08,0.08)
        ax7.set_xticklabels([])
        ax7.set_yticklabels([])
        ax7.set_zticklabels([])
        ax7.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax7.set_zlabel(r'$\varphi_{12}(t)$') #,rotation=90)
        ax7.xaxis.labelpad=-11
        ax7.yaxis.labelpad=-12
        ax7.zaxis.labelpad=-14
        ax7.grid(True)
        ax8 = fig.add_subplot(428, projection='3d')
        ax8.plot(x_test_recon[:,9],x_test_recon[:,10],x_test_recon[:,11], 'r--')
        ax8.plot(x_test_sim[:,9],x_test_sim[:,10],x_test_sim[:,11], 'b--')
        #ax4.plot(x_test_sim_predict[:,3],x_test_sim_predict[:,4],x_test_sim_predict[:,5], 'r--')
        ax8.set(xlabel=r'$\varphi_{10}(t)$', ylabel=r'$\varphi_{11}(t)$')
        ax8.set_xlim(-0.08,0.08)
        ax8.set_ylim(-0.08,0.08)
        ax8.set_zlim(-0.08,0.08)
        ax8.set_xticklabels([])
        ax8.set_yticklabels([])
        ax8.set_zticklabels([])
        ax8.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax8.set_zlabel(r'$\varphi_{12}(t)$') #,rotation=90)
        ax8.xaxis.labelpad=-11
        ax8.yaxis.labelpad=-12
        ax8.zaxis.labelpad=-14
        ax8.grid(True)
    plt.savefig('manifold.pdf')
    make_table(model,feature_names)
    # now attempt a pareto curve
    #print('performing Pareto analysis')
    poly_order = [poly_order]
    n_jobs = 1
    yscale = 'log'
    thresholds=np.linspace(0,0.4,20)
    #pareto_curve(STLSQ,PolynomialLibrary,FiniteDifference, \
    #    feature_names,False,n_jobs,thresholds,poly_order,x_test,time,yscale)
    for i in range(num_POD):
        x_test_sim[:,i] = x_test_sim[:,i]*sum(np.amax(abs(Vh),axis=1)[0:num_POD])
        x_pred[:,i] = x_pred[:,i]*sum(np.amax(abs(Vh),axis=1)[0:num_POD])
    return t_pred,x_pred,x_test_sim
