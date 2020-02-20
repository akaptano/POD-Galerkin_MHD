import numpy as np
from matplotlib import pyplot as plt
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi
from sindy_utils import inner_product, plot_Hc, \
    plot_measurement_fits, make_contour_movie
from scalar_POD import scalar_POD
from load_scalar_mats import load_mats
from vector_POD import vector_POD
import matplotlib.animation as animation

#def run_sindy(num_POD,threshold,poly_order,start,end,skip,Bx_mat,By_mat,Bz_mat,Vx_mat,Vy_mat,Vz_mat):
mu0 = 4*pi*1e-7
mion = 2*1.67*1e-27
spacing = 10.0
num_POD = 3
threshold = 0.05 #0.05
poly_order = 2
start = 20000
end = 20500
skip = 10
time = np.loadtxt('time.csv')
print(np.shape(time))
# shorten and put into microseconds
time = time[int(start/spacing):int(end/spacing):int(skip/spacing)]*1e6
x, y, z, Bx, By, Bz, \
    dBxdx,dBxdy,dBxdz,dBydx,dBydy,dBydz,dBzdx,dBzdy,dBzdz, \
    Vx,Vy,Vz,dVxdx,dVxdy,dVxdz,dVydx,dVydy,dVydz,dVzdx,dVzdy,dVzdz \
    = np.loadtxt('HITSI_rMHD_HR_10000.csv',delimiter=',',unpack=True)
r = np.sqrt(x**2+y**2)
phi = np.arctan2(y,x)
dr = np.unique(r.round(decimals=4))[1]
dphi = np.unique(phi.round(decimals=4))[1]-np.unique(phi.round(decimals=4))[0]
dz = np.unique(z)[1]-np.unique(z)[0]
print(dr,dphi,dz,np.unique(r.round(decimals=4)),np.unique(phi.round(decimals=4)),np.unique(z))
# Probably need to integrate in (r,z,phi) because it is uniformly sampled in
# the cylindrical coordinates, not in cartesian
#print(x,y,np.unique(z),dx,dy,dz,np.shape(np.unique(x)),np.shape(np.unique(z)))
fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(np.sort(x),'ro')
ax = fig.add_subplot(132)
ax.plot(x,y,'ro')
ax = fig.add_subplot(133, projection='3d')
ax.plot(x,y,z,'ro')

#threshold = 1e-2
#poly_order = 2
#scalar_POD(Bx_mat,By_mat,Bz_mat,Vx_mat,Vy_mat,Vz_mat, \
#    time,threshold,poly_order)
Bx_mat,By_mat,Bz_mat,Vx_mat,Vy_mat,Vz_mat = load_mats(start,end,skip)
# Try animation
plt.figure(237898234)
ani = animation.FuncAnimation( \
fig, make_contour_movie, range(0,len(time),2), \
fargs=(x,y,z,Bx_mat,time),repeat=False, \
interval=100, blit=False)
FPS = 40
ani.save('Bx_contour.mp4',fps=FPS)
exit()
Q = np.vstack((Bx_mat,By_mat))
Q = np.vstack((Q,Bz_mat))
Q = np.vstack((Q,Vx_mat))
Q = np.vstack((Q,Vy_mat))
Q = np.vstack((Q,Vz_mat))
Hc = np.sqrt(mion/mu0)*(Bx_mat*Vx_mat + By_mat*Vy_mat + Bz_mat*Vz_mat)*dr*dphi*dz
plot_Hc(time,Hc,r)
#Q = np.transpose(Q)
print(np.shape(Q),dr,dphi,dz)
r = np.ravel([r,r,r,r,r,r])
svd_mat = inner_product(Q,r)*dr*dphi*dz
t_pred,x_pred,x_sim = vector_POD(svd_mat,time,poly_order,threshold,num_POD)
# need to get rid of the radial weights on x_sim somehow
# Can do this by x_sim*sqrt(r) presumably?
#Q = Q*np.sqrt(dr*dphi*dz)
#for i in range(np.shape(Q)[1]):
#    Q[:,i] = Q[:,i]*np.sqrt(r)
U,S,Vh = np.linalg.svd(Q,full_matrices=False)
print(np.shape(Q),np.shape(x_pred))
Qfit = ((U*S)[:,0:num_POD])@(np.transpose(x_pred))
print(np.shape(Qfit))
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
print('% Field = ',sum(S[0:num_POD])/sum(S))
plt.savefig('field_spectrum.pdf')
Q = ((U*S)[:,0:num_POD])@Vh[0:num_POD,:]
print(np.shape(Vh),np.shape(Q))
plot_measurement_fits(time,Q,Qfit,t_pred)
plt.show()
