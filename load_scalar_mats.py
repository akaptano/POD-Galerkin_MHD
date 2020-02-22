import numpy as np
from numpy import pi
mu0 = 4*pi*1e-7
rho = 2.0e19*2*1.67*1e-27

def load_mats(start,end,skip):
    Bx_mat = []
    By_mat = []
    Bz_mat = []
    Vx_mat = []
    Vy_mat = []
    Vz_mat = []
    for i in range(start,end,skip):
        if i % 10 == 0:
            print(i)
        if i < 10:
            Bx, By, Bz, Vx, Vy, Vz = \
                np.loadtxt('../HITSI_rMHD_HR_0000'+str(i)+'.csv',delimiter=',',usecols=(3,4,5,15,16,17),unpack=True)
        elif i < 100:
            Bx, By, Bz, Vx, Vy, Vz = \
                np.loadtxt('../HITSI_rMHD_HR_000'+str(i)+'.csv',delimiter=',',usecols=(3,4,5,15,16,17),unpack=True)
        elif i < 1000:
            Bx, By, Bz, Vx, Vy, Vz = \
                np.loadtxt('../HITSI_rMHD_HR_00'+str(i)+'.csv',delimiter=',',usecols=(3,4,5,15,16,17),unpack=True)
        elif i < 10000:
            Bx, By, Bz, Vx, Vy, Vz = \
                np.loadtxt('../HITSI_rMHD_HR_0'+str(i)+'.csv',delimiter=',',usecols=(3,4,5,15,16,17),unpack=True)
        else:
            Bx, By, Bz, Vx, Vy, Vz = \
                np.loadtxt('../HITSI_rMHD_HR_'+str(i)+'.csv',delimiter=',',usecols=(3,4,5,15,16,17),unpack=True)
        Bx_mat.append(Bx)
        By_mat.append(By)
        Bz_mat.append(Bz)
        Vx_mat.append(Vx)
        Vy_mat.append(Vy)
        Vz_mat.append(Vz)
    Bx_mat = np.array(Bx_mat)
    By_mat = np.array(By_mat)
    Bz_mat = np.array(Bz_mat)
    Vx_mat = np.array(Vx_mat)
    Vy_mat = np.array(Vy_mat)
    Vz_mat = np.array(Vz_mat)
    #scale to Va
    Bx_mat = Bx_mat/np.sqrt(mu0*rho)
    By_mat = By_mat/np.sqrt(mu0*rho)
    Bz_mat = Bz_mat/np.sqrt(mu0*rho)
    bx_avg = np.mean(Bx_mat,0)
    by_avg = np.mean(By_mat,0)
    bz_avg = np.mean(Bz_mat,0)
    vx_avg = np.mean(Vx_mat,0)
    vy_avg = np.mean(Vy_mat,0)
    vz_avg = np.mean(Vz_mat,0)
    for i in range(np.shape(Bx_mat)[0]):
        Bx_mat[i,:] = Bx_mat[i,:] - bx_avg
        By_mat[i,:] = By_mat[i,:] - by_avg
        Bz_mat[i,:] = Bz_mat[i,:] - bz_avg
        Vx_mat[i,:] = Vx_mat[i,:] - vx_avg
        Vy_mat[i,:] = Vy_mat[i,:] - vy_avg
        Vz_mat[i,:] = Vz_mat[i,:] - vz_avg
    Bx_mat = np.transpose(Bx_mat)
    By_mat = np.transpose(By_mat)
    Bz_mat = np.transpose(Bz_mat)
    Vx_mat = np.transpose(Vz_mat)
    Vy_mat = np.transpose(Vy_mat)
    Vz_mat = np.transpose(Vz_mat)
    return Bx_mat,By_mat,Bz_mat,Vx_mat,Vy_mat,Vz_mat
