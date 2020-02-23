import numpy as np
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference,SmoothedFiniteDifference
from sindy_utils import plot_energy, make_table, \
    update_manifold_movie, plot_BOD_Espectrum, make_evo_plots, \
    make_3d_plots
from pysindy.utils.pareto import pareto_curve
from scipy.integrate import odeint

def compressible_Framework(inner_prod,time,poly_order,threshold,r):
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
    plot_energy(time,inner_prod)
    x,feature_names,Vh = vector_POD(inner_prod,r)
    model = SINDy(optimizer=STLSQ(threshold=threshold), \
        feature_library=PolynomialLibrary(degree=poly_order), \
        differentiation_method=SmoothedFiniteDifference(drop_endpoints=True), \
        feature_names=feature_names)
    tfac = 3.0/5.0
    t_train = time[:int(len(time)*tfac)]
    x_train = x[:int(len(time)*tfac),:]
    x0_train = x[0,:]
    t_test = time[int(len(time)*tfac):]
    x_true = x[int(len(time)*tfac):,:]
    x0_test = x[int(len(time)*tfac),:]
    model.fit(x_train, t=t_train)
    model.print()
    print(model.coefficients())
    integrator_kws = {'full_output': True}
    x_train,output = model.simulate(x0_train,t_train, \
        integrator=odeint,stop_condition=None,full_output=True, \
        rtol=1e-15,h0=1e-20,tcrit=[1702])
    x_sim,output = model.simulate(x0_test,t_test, \
        integrator=odeint,stop_condition=None,full_output=True, \
        rtol=1e-15,h0=1e-20,tcrit=[2090]) #,hmax=1e-2,atol=1e-15) #h0=1e-20
    x_dot = model.differentiate(x, t=time)
    x_dot_train = model.predict(x_train)
    x_dot_sim = model.predict(x_true)
    print('Model score: %f' % model.score(x, t=time))
    make_evo_plots(x_dot,x_dot_train, \
        x_dot_sim,x_true,x_sim,time,t_train,t_test)
    make_3d_plots(x_true,x_sim,t_test)
    make_table(model,feature_names)
    # now attempt a pareto curve
    #print('performing Pareto analysis')
    poly_order = [poly_order]
    n_jobs = 1
    yscale = 'log'
    thresholds=np.linspace(0,3.0,20)
    pareto_curve(STLSQ,PolynomialLibrary,FiniteDifference, \
        feature_names,False,n_jobs,thresholds,poly_order,x_train,x_true,t_train,t_test,yscale)
    print('x_tests size = ',np.shape(x_sim))
    for i in range(r):
        x_sim[:,i] = x_sim[:,i]*sum(np.amax(abs(Vh),axis=1)[0:r])
        x_true[:,i] = x_true[:,i]*sum(np.amax(abs(Vh),axis=1)[0:r])
    return t_test,x_true,x_sim

def vector_POD(inner_prod,r):
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
    (r = truncation number of the SVD, M = number of time samples)
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
    V,S,Vh = np.linalg.svd(inner_prod,full_matrices=False)
    plot_BOD_Espectrum(S)
    print("% field in first r modes = ",sum(np.sqrt(S[0:r]))/sum(np.sqrt(S)))
    print("% energy in first r modes = ",sum(S[0:r])/sum(S))
    vh = np.zeros((r,np.shape(Vh)[1]))
    feature_names = []
    # normalize the modes
    for i in range(r):
        #vh[i,:] = Vh[i,:]*S[i]**2/sum(S[0:r]**2*np.amax(abs(Vh),axis=1)[0:r])
        vh[i,:] = Vh[i,:]/sum(np.amax(abs(Vh),axis=1)[0:r])
        feature_names.append(r'$\varphi_{:d}$'.format(i+1))
    x = np.transpose(vh)
    return x, feature_names, Vh
