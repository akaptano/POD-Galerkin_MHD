import numpy as np
from numpy import pi
from matplotlib import pyplot as plt

def make_VandermondeT(omega,time):
    VandermondeT = np.exp(np.outer(time,omega))
    return VandermondeT

def dmd(data,r,tbase):
    tbase = tbase[:-1]
    dt = 1e-6
    X = data[:,0:-1]
    Xprime = data[:,1:]
    Udmd,Sdmd,Vdmd = np.linalg.svd(X,full_matrices=False)
    Vdmd = np.transpose(Vdmd)
    Udmd = Udmd[:,0:r]
    Sdmd = Sdmd[0:r]
    Vdmd = Vdmd[:,0:r]
    S = np.diag(Sdmd)
    A = np.dot(np.dot(np.transpose(Udmd),Xprime),Vdmd/Sdmd)
    eigvals,Y = np.linalg.eig(A)
    Bt = np.dot(np.dot(Xprime,Vdmd/Sdmd),Y)
    omega = np.log(eigvals)/dt
    VandermondeT = make_VandermondeT(omega,tbase-tbase[0])
    Vandermonde = np.transpose(VandermondeT)
    q = np.conj(np.diag(np.dot(np.dot(np.dot( \
        Vandermonde,Vdmd),np.conj(S)),Y)))
    P = np.dot(np.conj(np.transpose(Y)),Y)* \
        np.conj(np.dot(Vandermonde, \
        np.conj(VandermondeT)))
    b = np.dot(np.linalg.inv(P),q)
    b = 0.5*(b+np.conj(b)).real
    b = 500*b/np.max(b)
    energy_sort = np.flip(np.argsort(b))
    print(omega/(2*pi*1e3),b)
    plt.figure()
    omega_sizes = b
    for i in range(len(b)):
        omega_sizes[i] = max(omega_sizes[i],50)
    plt.scatter(omega.imag/(2*pi*1e3),omega.real/(2*pi*1e3),s=omega_sizes,c='k')
    plt.savefig('Pictures/dmd_freqs.pdf')
    Bt = Bt[:,energy_sort]
    return Bt
