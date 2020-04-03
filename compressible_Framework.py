import numpy as np
from pysindy import SINDy
from pysindy.optimizers import STLSQ, SR3, SR3Enhanced
from pysindy.feature_library import PolynomialLibrary,CustomLibrary
from pysindy.differentiation import FiniteDifference,SmoothedFiniteDifference
from sindy_utils import make_table, \
    plot_BOD_Espectrum, make_evo_plots, \
    make_3d_plots, plot_pod_temporal_modes
from pysindy.utils.pareto import pareto_curve
from scipy.integrate import odeint
from scipy.linalg import eig
from matplotlib import pyplot as plt
from numpy.random import random

def compressible_Framework(Q,inner_prod,time,poly_order,threshold,r,tfac):
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

    """
    M_train = int(len(time)*tfac)
    t_train = time[:M_train]
    t_test = time[M_train:]
    #x,feature_names,S2,Vh, = vector_POD(Q,t_train,r)
    x,feature_names,S2,Vh, = vector_POD(inner_prod,t_train,r)
    x[:,6] = np.zeros(np.shape(x[:,6]))
    #for i in range(r):
    #    x[:,i] = x[:,i]*sum(np.amax(abs(Vh),axis=1)[0:r])
    #S = np.sqrt(S2[0:r,0:r])
    #U_true = Q@x@np.linalg.inv(S)
    #Q_true = U_true[:,0:r]@S@np.transpose(x)
    print('Now fitting SINDy model')
    if poly_order == 1:
        library_functions = [lambda x : x]
        library_function_names = [lambda x : x]
    if poly_order == 2:
        library_functions = [lambda x : x, lambda x,y : x*y, lambda x : x**2]
        library_function_names = [lambda x : x, lambda x,y : x+y, lambda x : x+x]
    if poly_order == 3:
        library_functions = [lambda x : x, lambda x,y : x*y, lambda x : x**2, lambda x,y,z: x*y*z, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x: x**3]
        library_function_names = [lambda x : x, lambda x,y : x+y, lambda x : x+x, lambda x,y,z: x+y+z, lambda x,y: x+x+y, lambda x,y: x+y+y, lambda x: x+x+x]
    if poly_order == 4:
        library_functions = [lambda x : x, lambda x,y : x*y, lambda x : x**2, lambda x,y,z: x*y*z, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x: x**3, lambda x,y,z,w: x*y*z*w, lambda x,y,z: x*y*z**2, lambda x,y: x**2*y**2, lambda x,y: x**3*y, lambda x: x**4]
        library_function_names = [lambda x : x, lambda x,y : x+y, lambda x : x+x, lambda x,y,z: x+y+z, lambda x,y: x+x+y, lambda x,y: x+y+y, lambda x: x+x+x, lambda x,y,z,w: x+y+z+w, lambda x,y,z: x+y+z+z, lambda x,y: x+x+y+y, lambda x,y: x+x+x+y, lambda x: x+x+x+x]
    sindy_library = CustomLibrary(library_functions=library_functions, \
        function_names=library_function_names)
    #print(library_function_names,feature_names)
    #sindy_library = PolynomialLibrary(degree=poly_order)
    #if r == 3:
    #    constraint_matrix = np.zeros((6,27))
        # constraints from vanishing diagonal
    #    constraint_matrix[0,0] = 1.0
    #    constraint_matrix[1,4] = 1.0
    #    constraint_matrix[2,8] = 1.0

        #constraint_matrix[0,1] = 1.0
        #constraint_matrix[0,3] = 1.0
        # constraints from anti-symmetry
    #    constraint_matrix[3,1] = 1.0
    #    constraint_matrix[3,3] = 1.0
    #    constraint_matrix[4,2] = 1.0
    #    constraint_matrix[4,6] = 1.0
    #    constraint_matrix[5,5] = 1.0
    #    constraint_matrix[5,7] = 1.0
    #    constraint_zeros = np.zeros(6)
    #    print(constraint_matrix)
    #else:
        # number of constraints is N*(N+1)/2
    #    constraint_matrix = np.zeros((6*13,int(r*(r**2+3*r)/2)))
    #    constraint_zeros = np.zeros(6*13)
    constraint_zeros = np.zeros(int(r*(r+1)/2))
    if poly_order == 1:
        constraint_matrix = np.zeros((int(r*(r+1)/2),r**2))
        for i in range(r):
            constraint_matrix[i,i*(r+1)] = 1.0
        q = r
        for i in range(r):
            counter = 1
            for j in range(i+1,r):
                constraint_matrix[q,i*r+j] = 1.0
                constraint_matrix[q,i*r+j+counter*(r-1)] = 1.0
                counter = counter + 1
                q = q + 1
    else:
        if poly_order == 2:
            #constraint_zeros = np.zeros(6+int(r*(r+1)/2))
            #constraint_matrix = np.zeros((6+int(r*(r+1)/2),int(r*(r**2+3*r)/2)))
            constraint_matrix = np.zeros((int(r*(r+1)/2),int(r*(r**2+3*r)/2)))
        if poly_order == 3:
            #constraint_matrix = np.zeros((int(r*(r+1)/2),int(r*(r**2+3*r)/2)+336))
            constraint_matrix = np.zeros((int(r*(r+1)/2),int(r*(r**2+3*r)/2)+30))
        if poly_order == 4:
            constraint_matrix = np.zeros((int(r*(r+1)/2),int(r*(r**2+3*r)/2)+60))
        for i in range(r):
            constraint_matrix[i,i*(r+1)] = 1.0
        q = r
        for i in range(r):
            counter = 1
            for j in range(i+1,r):
                constraint_matrix[q,i*r+j] = 1.0
                constraint_matrix[q,i*r+j+counter*(r-1)] = 1.0
                counter = counter + 1
                q = q + 1
        #constraint_matrix[-8,8*8+2] = 1.0
        #constraint_matrix[-8,9*8+5] = -2.0
        #constraint_matrix[-7,12*8+6] = 1.0
        #constraint_matrix[-7,12*8+7] = -1.0
        #constraint_matrix[-6,17*8+6] = 1.0
        #constraint_matrix[-6,17*8+7] = -1.0
        #constraint_matrix[-5,17*8+6] = 1.0
        #constraint_matrix[-5,18*8+7] = 1.0
        #constraint_matrix[-4,9*8+5] = 1.0
        #constraint_matrix[-4,10*8+4] = 1.0
        #constraint_matrix[-3,15*8+4] = 1.0
        #constraint_matrix[-3,16*8+5] = -1.0
        #constraint_matrix[-2,36*8+3] = 1.0
        #constraint_matrix[-2,37*8+3] = 1.0
        #constraint_matrix[-1,36*8+6] = 1.0
        #constraint_matrix[-1,37*8+6] = 1.0
        #
        #constraint_matrix[-6,9*8+6] = 1.0
        #constraint_matrix[-6,9*8+7] = 1.0
        #constraint_matrix[-5,40*8+7] = 1.0
        #constraint_matrix[-5,41*8+7] = -1.0
        #constraint_matrix[-4,36*8+3] = 1.0
        #constraint_matrix[-4,37*8+3] = 1.0
        #constraint_matrix[-3,12*8+6] = 1.0
        #constraint_matrix[-3,18*8+6] = 1.0
        #constraint_matrix[-2,18*8+6] = -1.0
        #constraint_matrix[-2,18*8+7] = 1.0
        #constraint_matrix[-1,12*8+6] = 2.0
        #constraint_matrix[-1,12*8+7] = -1.0
    print(constraint_matrix,np.shape(constraint_matrix))
    linear_r4_mat = np.zeros((r,r))
    linear_r4_mat[0,1] = 0.091
    linear_r4_mat[1,0] = -0.091
    linear_r4_mat[2,3] = 0.182
    linear_r4_mat[3,2] = -0.182
    linear_r4_mat[5,4] = -3*0.091
    linear_r4_mat[4,5] = 3*0.091
    #linear_r4_mat[6,7] = 4*0.091
    #linear_r4_mat[7,6] = -4*0.091
    linear_r12_mat = np.zeros((12,90))
    linear_r12_mat[0,1] = 0.089
    linear_r12_mat[1,0] = -0.089
    linear_r12_mat[2,3] = 0.172
    linear_r12_mat[3,2] = -0.172
    linear_r12_mat[2,5] = 0.03
    linear_r12_mat[5,2] = -0.03
    linear_r12_mat[2,6] = 0.022
    linear_r12_mat[6,2] = -0.022
    linear_r12_mat[6,4] = 0.022
    linear_r12_mat[4,6] = 0.023
    linear_r12_mat[7,5] = -0.023
    linear_r12_mat[5,7] = -0.123
    linear_r12_mat[7,5] = 0.123
    #sindy_opt = STLSQ(threshold=threshold)
    #sindy_opt = SR3(threshold=threshold, nu=1, max_iter=1000,tol=1e-8)
    sindy_opt = SR3Enhanced(threshold=threshold, nu=1, max_iter=10000, \
        constraint_lhs=constraint_matrix,constraint_rhs=constraint_zeros, \
        tol=1e-6,thresholder='l0',initial_guess=linear_r4_mat)
    model = SINDy(optimizer=sindy_opt, \
        feature_library=sindy_library, \
        differentiation_method=FiniteDifference(drop_endpoints=True), \
        feature_names=feature_names)
    x_train = x[:M_train,:]
    x0_train = x[0,:]
    x_true = x[M_train:,:]
    x0_test = x[M_train,:]
    model.fit(x_train, t=t_train, unbias=False)
    t_cycle = np.linspace(time[M_train],time[M_train]*1.3,int(len(time)/2.0))
    #model.print()
    print(model.coefficients())
    print('Sum = ',np.sum(np.sum(sindy_opt.coef_)))
    #print(np.ravel(model.coefficients()))
    #print(model.get_feature_names())
    #print(sindy_opt.history_)
    print(sindy_opt.coef_)
    print(sindy_opt.coef_full_)
    #print(sindy_opt.coef_/sindy_opt.coef_full_)
    integrator_kws = {'full_output': True}
    #x_train,output = model.simulate(x0_train,t_train, \
    #    integrator=odeint,stop_condition=None,full_output=True, \
    #    rtol=1e-15,h0=1e-20,tcrit=[2090])
    x_sim,output = model.simulate(x0_test,t_test, \
        integrator=odeint,stop_condition=None,full_output=True, \
        rtol=1e-20,h0=1e-5) #h0=1e-20
    x_sim1,output = model.simulate(-0.4*np.ones(r),t_cycle, \
        integrator=odeint,stop_condition=None,full_output=True, \
        rtol=1e-20,h0=1e-5)
    x_sim2,output = model.simulate(0.15*np.ones(r),t_cycle, \
        integrator=odeint,stop_condition=None,full_output=True, \
        rtol=1e-20,h0=1e-5)
    x_dot = model.differentiate(x, t=time)
    x_dot_train = model.predict(x_train)
    x_dot_sim = model.predict(x_true)
    print('Model score: %f' % model.score(x, t=time))
    make_evo_plots(x_dot,x_dot_train, \
        x_dot_sim,x_true,x_sim,time,t_train,t_test)
    make_table(model,feature_names,r)
    #make_3d_plots(x_true,x_sim,t_test,'sim')
    #make_3d_plots(x_sim1,x_sim2,t_cycle,'limitcycle')
    #plt.show()
    # now attempt a pareto curve
    #print('performing Pareto analysis')
    #poly_order = [poly_order]
    #n_jobs = 1
    #yscale = 'log'
    #thresholds=np.linspace(0,3.0,20)
    #pareto_curve(STLSQ,PolynomialLibrary,FiniteDifference, \
    #    feature_names,False,n_jobs,thresholds,poly_order,x_train,x_true,t_train,t_test,yscale)
    #print('x_tests size = ',np.shape(x_sim))
    for i in range(r):
        x_sim[:,i] = x_sim[:,i]*sum(np.amax(abs(Vh),axis=1)[0:r])
        x_true[:,i] = x_true[:,i]*sum(np.amax(abs(Vh),axis=1)[0:r])
    return t_test,x_true,x_sim,S2

def vector_POD(inner_prod,t_train,r):
    """
    Performs the vector BOD, and scales the resulting modes
    to lie on the unit ball. Also returns the names of the
    temporal modes which will be modeled.

    Parameters
    ----------
    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The scaled matrix of inner products X*X

    Returns
    -------
    x: 2D numpy array of floats
    (M = number of time samples, r = truncation number of the SVD)
        The temporal BOD modes to be modeled, scaled to
        stay on the unit ball

    feature_names: numpy array of strings
    (r = truncation number of the SVD)
        Names of the temporal BOD modes to be modeled

    Vh: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The V* in the SVD, returned here because the SINDy modes
        will need to be rescaled off of the unit ball to compare
        with the original measurements

    """
    #U,S2,Vh = np.linalg.svd(inner_prod,full_matrices=False)
    #v,S2,Vh = np.linalg.svd(inner_prod,full_matrices=False)
    S2,v = eig(inner_prod)
    idx = S2.argsort()[::-1]
    S2 = S2[idx]
    v = v[:,idx]
    Vh = np.transpose(v)
    #v = np.transpose(Vh)
    plot_pod_temporal_modes(v[:len(t_train),:],t_train)
    plot_BOD_Espectrum(S2)
    print("% field in first r modes = ",sum(np.sqrt(S2[0:r]))/sum(np.sqrt(S2)))
    print("% energy in first r modes = ",sum(S2[0:r])/sum(S2))
    S2 = np.diag(S2)
    vh = np.zeros((r,np.shape(Vh)[1]))
    feature_names = []
    # normalize the modes
    for i in range(r):
        #vh[i,:] = Vh[i,:]/sum(S[0:r]**2*np.amax(abs(Vh),axis=1)[0:r])
        vh[i,:] = Vh[i,:]/sum(np.amax(abs(Vh),axis=1)[0:r])
        feature_names.append(r'$\varphi_{:d}$'.format(i+1))
    x = np.transpose(vh)
    return x, feature_names, S2, Vh
