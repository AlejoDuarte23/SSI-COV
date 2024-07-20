import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
import _ssi_cov_ad_vf as SSI
import _ssi_backend as SSIb
from scipy import signal
# import tabulate
# Load .mat file
mat = loadmat('BridgeData.mat')
t, rz, wn = mat['t'], mat['rz'], mat['wn']

def plot_Data(t, rz, wn):
    # Transform circular frequency (rad/s) into frequency (Hz)
    fnTarget = wn / (2 * np.pi)
    
    # Get time step
    dt = np.median(np.diff(t))
    
    # Get sampling frequency
    fs = 1 / dt
    
    # Get the number of sensors and the number of time steps
    Nyy, N = rz.shape
    
    # Visualization of the data
    plt.figure()
    
    # Displacement record from sensor no 2
    plt.subplot(221)
    plt.plot(t[0], rz[1, :])
    plt.xlim([0, 600])
    plt.xlabel('time (s)')
    plt.ylabel('r_z (m)')
    plt.title('Displ record from sensor no 2')
    
    # PSD estimate from sensor no 2
    plt.subplot(222)
    f, Pxx = welch(rz[1, :], fs=fs, nperseg=None, noverlap=None)
    plt.plot(f, Pxx)
    plt.xscale('log')
    plt.title('PSD estimate from sensor no 2')
    
    # Displacement record from sensor no 5
    plt.subplot(223)
    plt.plot(t[0], rz[4, :])
    plt.xlim([0, 600])
    plt.xlabel('time (s)')
    plt.ylabel('r_z (m)')
    plt.title('Displ record from sensor no 5')
    
    # PSD estimate from sensor no 5
    plt.subplot(224)
    f, Pxx = welch(rz[4, :], fs=fs, nperseg=None, noverlap=None)
    plt.plot(f, Pxx)
    plt.xscale('log')
    plt.title('PSD estimate from sensor no 5')
    
    # Set the figure background color to white
    plt.gcf().set_facecolor('w')
    
    # Show the plots
    plt.show()

fs = 15 

acc = rz.T
Nmin = 7
Nmax = 15
Nc = acc.shape[1]
Ts  = 40
fnS,zetaS, phiS,fn1_list,i_list,stablity_status,fn2  =  SSI.SSI_COV_AD(acc ,fs,Ts,Nc,Nmax,Nmin,10,0.01 )




num_clusters = 6
summary = SSIb.cluster_data_by_frequency(fnS, zetaS, phiS, num_clusters)
             
                 
SSIb.plotStabDiag(fn2, acc, fs, stablity_status, Nmin, Nmax, acc.shape[1], 0, 7.5)
