from pysindy import SINDy
from numpy import count_nonzero, zeros, ravel
from numpy.random import random
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.integrate import odeint

def pareto_curve(pareto_optimizer,pareto_feature_library, \
    pareto_differentiation_method,pareto_feature_names, \
    discrete_time,n_jobs,thresholds,poly_orders,x_fit,x_pred,t_fit,t_pred,yscale):
    """
    Function which sweeps out a Pareto Curve for the derivative of X

    Parameters
    ----------
    optimizer : optimizer object, optional
        Optimization method used to fit the SINDy model. This must be an object
        that extends the sindy.optimizers.BaseOptimizer class. Default is
        sequentially thresholded least squares with a threshold of 0.1.

    feature_library : feature library object, optional
        Default is polynomial features of degree 2.
        TODO: Implement better feature library class.

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be an object that
        extends the sindy.differentiation_methods.BaseDifferentiation class.
        Default is centered difference.

    feature_names : list of string, length n_input_features, optional
       Names for the input features. If None, will use ['x0','x1',...].

    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical system is assumed to be a flow
        (right hand side functions predict continuous time derivatives).

    n_jobs : int, optional (default 1)
        The number of parallel jobs to use when fitting, predicting with, and
        scoring the model.

    thresholds: array of floats
        The list of thresholds to change the number of terms available to the
        SINDy model, thus generating a Pareto curve

    x: array-like or list of array-like, shape
        (n_samples, n_input_features)
        Training data. If training data contains multiple trajectories,
        x should be a list containing data for each trajectory. Individual
        trajectories may contain different numbers of samples.

    t: float, numpy array of shape [n_samples], or list of numpy arrays,
        optional (default 1)
        If t is a float, it specifies the timestep between each sample.
        If array-like, it spoptimizer : optimizer object, optional
        Optimization method used to fit the SINDy model. This must be an object
        that extends the sindy.optimizers.BaseOptimizer class. Default is
        sequentially thresholded least squares with a threshold of 0.1.

    feature_library : feature library object, optional
        Default is polynomial features of degree 2.
        TODO: Implement better feature library class.

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be an object that
        extends the sindy.differentiation_methods.BaseDifferentiation class.
        Default is centered difference.

    feature_names : list of string, length n_input_features, optional
       Names for the input features. If None, will use ['x0','x1',...].

    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical systemecifies the time at which each sample was
        collected.
        In this case the values in t must be strictly increasing.
        In the case of multi-trajectory training data, t may also be a list
        of arrays containing the collection times for each individual
        trajectory.
        Default value is a timestep of 1 between samples.
    """
    model_scores = zeros((len(poly_orders),len(thresholds)))
    non_zeros_coeffs = zeros((len(poly_orders),len(thresholds)))
    x_err = zeros((len(poly_orders),len(thresholds)))
    plt.figure(30910236,figsize=(10,4.5))
    for i in range(len(poly_orders)):
        for j in range(len(thresholds)):
            model = SINDy(optimizer=pareto_optimizer(threshold=thresholds[j]), \
                feature_library=pareto_feature_library(degree=poly_orders[i]), \
                differentiation_method=pareto_differentiation_method(drop_endpoints=True), \
                feature_names=pareto_feature_names,discrete_time=discrete_time,n_jobs=n_jobs)
            model.fit(x_fit, t=t_fit)
            x0 = x_pred[0,:]
            integrator_kws = {'full_output': True}
            x_sim,output = model.simulate(x0,t_pred, \
                integrator=odeint,stop_condition=None,full_output=True, \
                rtol=1e-20,h0=1e-20,tcrit=[2270],hmax=1e-2)
            x_err[i,j] = norm(x_pred-x_sim)
            num_coeff = len(ravel(model.coefficients()))
            num_nonzero_coeff = count_nonzero(model.coefficients())
            non_zeros_coeffs[i,j] = num_nonzero_coeff/num_coeff*100
            model_scores[i,j] = min((1-min(model.score(x_pred, t=t_pred),1))*100,100)
        plt.figure(30910236)
        plt.subplot(1,2,1)
        plt.plot(non_zeros_coeffs[i,:],model_scores[i,:], \
            color='r',marker='o',label='Poly Degree = '+str(poly_orders[i]))
        plt.subplot(1,2,2)
        plt.plot(thresholds,model_scores[i,:], \
            color='r',marker='o',label='Poly Degree = '+str(poly_orders[i]))
        plt.figure(30910237,figsize=(10,4.5))
        plt.plot(thresholds,x_err[i,:], \
            color=random(3),marker='o',label='Poly Degree = '+str(poly_orders[i]))
    plt.figure(30910236)
    plt.subplot(1,2,1)
    plt.ylim(0,105)
    ax = plt.gca()
    ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.ylim(0,105)
    ax = plt.gca()
    ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
    ax.set_yticklabels([])
    plt.grid(True)
    plt.savefig('Pictures/xdot_pareto_curve.png',dpi=200)
    plt.savefig('Pictures/xdot_pareto_curve.svg',dpi=200)
    plt.savefig('Pictures/xdot_pareto_curve.pdf',dpi=200)
    plt.savefig('Pictures/xdot_pareto_curve.eps',dpi=200)
    plt.figure(30910237)
    plt.yscale(yscale)
    plt.savefig('Pictures/x_pareto_curve.png',dpi=200)
    plt.savefig('Pictures/x_pareto_curve.svg',dpi=200)
    plt.savefig('Pictures/x_pareto_curve.pdf',dpi=200)
    plt.savefig('Pictures/x_pareto_curve.eps',dpi=200)
