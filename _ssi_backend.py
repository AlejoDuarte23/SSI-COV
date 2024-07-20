import numpy as np
import matplotlib.pyplot as plt 

from scipy.cluster.hierarchy import linkage,fcluster
from collections import OrderedDict
from tabulate import tabulate
from scipy import signal
from sklearn.cluster import KMeans
from tabulate import tabulate
from numpy.typing import NDArray
from typing import Tuple

# --------------------------- 1. Load Data -----------------------------------#
def load_data(fn,delim):
    try:
        Acc= np.loadtxt(fn,delimiter = delim)
        Nc  = len(Acc[0,:]) 
        N   = len(Acc[:,0])
        return Acc,Nc,N
    except:
        print('error: could not load the txt file')        
        return [],[],[]    
# --------------------------- 2. pot_Acc -------------------------------------#
def plot_ACC(Acc,io,ie):
    plt.figure()
    for i in range(io,ie):
        plt.plot(Acc[:,i])
        plt.xlabel('Numner of data Points')
        plt.ylabel('Acceleration ')
        plt.title('Acceleration Record')
        plt.show()
# --------------------------- 3. NexT ----------------------------------------#
def NexT(acc: NDArray, fs:float, Ts:float, Nc:int) -> NDArray:
    dt = 1/fs
    M = round(Ts/dt)
    IRF = np.zeros((Nc,Nc,M-1),dtype = complex) 
    for oo in range(Nc):
        for jj in range(Nc):
            y1 = np.fft.fft(acc[:,oo])
            y2 = np.fft.fft(acc[:,jj])
            #cross-correlation: ifft[cross-power spectrum]
            h0 = np.fft.ifft(y1*y2.conj())
            #impulse response function
            IRF[oo,jj,:] = np.real(h0[0:M-1])

    if Nc ==1:
        IRF = np.squeeze(IRF)
        IRF = IRF/IRF[0] 

    return IRF 
# --------------------------- 4. blockToeplitz -------------------------------#
def blockToeplitz(IRF: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:   
    N1 = round(IRF.shape[2]/2)-1
    M = IRF.shape[1]
    T1= np.zeros(((N1)*M,(N1)*M),dtype = 'complex128')
    for oo in range(N1):
        for ll in range(N1):
            T1[(oo)*M:(oo+1)*M,(ll)*M:(ll+1)*M] = IRF[:,:,N1-1+oo-ll+1] 
    [U,S,Vt] = np.linalg.svd(T1)
    # this requires checking !!
    V = Vt.T

    return U,S,V,T1
# --------------------------- 5. Modal id ------------------------------------#
def modalID(U,S,Nmodes,Nyy,fs):
    S = np.diag(S)
    if Nmodes >= S.shape[0]:
        print("changing the number of modes to the maximum possible")
        Nmodes = S.shape[0]

    dt = 1/fs
    O = np.matmul(U[:,0:Nmodes],np.sqrt(S[0:Nmodes,0:Nmodes]))
    IndO = min(Nyy,len(O[:,0]))
    C = O[0:IndO,:]
    jb =O.shape[0]/IndO
    ao = int((IndO)*(jb-1))
    bo = int(len(O[:,0])-(IndO)*(jb-1))
    co = len(O[:,0])
    A =np.matmul( np.linalg.pinv(O[0:ao,:]),O[bo:co,:])
    [Vi,Di] = np.linalg.eig(A)
    mu = np.log(np.diag(np.diag(Vi)))/dt
    fno = np.abs(mu)/(2*np.pi)
    fn = fno[np.ix_(*[range(0,i,2) for i in fno.shape])]
    zetaoo = -np.real(mu)/np.abs(mu)
    zeta =  zetaoo[np.ix_(*[range(0,i,2) for i in zetaoo.shape])]
    phi0 = np.real(np.matmul(C[0:IndO,:],Di))
    #phi = phi0[:,1:-1:1]
    phi = phi0[:,1::2]
    print("shape phi",phi.shape)
    return fn,zeta,phi


##################################################

def stabilityCheck2(fn0, zeta0, phi0, fn1, zeta1, phi1):

    for mode in range(min(phi0.shape[1],phi1.shape[1])):
        print(f"Mode {mode + 1}")
        for shape0, shape1 in zip(phi0[:, mode], phi1[:, mode]):
            print(f"phi0: {shape0}, phi1: {shape1}")
    eps_freq = 2e-2 
    eps_zeta = 4e-2 
    eps_MAC = 5e-2
    stability_status = []
    fn = []
    zeta = []
    phi_list = []
    MAC = []

    # frequency stability
    N0 = len(fn0)
    N1 = len(fn1)

    for rr in range(N0-1):
        for jj in range(N1-1):
            stab_fn = errorcheck(fn0[rr], fn1[jj], eps_freq)
            stab_zeta = errorcheck(zeta0[rr], zeta1[jj], eps_zeta)
            stab_phi, dummyMAC = getMAC(phi0[:, rr], phi1[:, jj], eps_MAC)

            # get stability status
            if stab_fn == 0:
                stabStatus = 0  # new pole
            elif stab_fn == 1 and stab_phi == 1 and stab_zeta == 1:
                stabStatus = 1  # stable pole
            elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 1:
                stabStatus = 2  # pole with stable frequency and vector
            elif stab_fn == 1 and stab_zeta == 1 and stab_phi == 0:
                stabStatus = 3  # pole with stable frequency and damping
            elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 0:
                stabStatus = 4  # pole with stable frequency
            else:
                raise ValueError("Error: stability_status is undefined")

            fn.append(fn1[jj])
            zeta.append(zeta1[jj])
            phi_list.append(phi1[:, jj])
            MAC.append(dummyMAC)
            stability_status.append(stabStatus)

    ind = np.argsort(fn)
    fn = np.sort(fn)
    zeta = np.array(zeta)[ind]
    phi = np.column_stack(phi_list)[:, ind]
    MAC = np.array(MAC)[ind]
    stability_status = np.array(stability_status)[ind]

    return fn, zeta, phi, MAC, stability_status





def errCheck2(x0, x1, eps):
    if isinstance(x0, np.ndarray) or isinstance(x1, np.ndarray):
        raise ValueError('x0 and x1 must be a scalar')
    if abs(1 - x0 / x1) < eps:  # if frequency for mode i+1 is almost unchanged
        y = 1
    else:
        y = 0
    return y

def getMAC2(phi1, phi2, eps_MAC):
    N = len(phi1)
    MAC = np.abs(np.dot(phi1.conj().T, phi2)) ** 2 / np.dot(np.dot(phi1.conj().T, phi1), np.dot(phi2.conj().T, phi2))
    dummyMAC = MAC.copy()
    for ii in range(N):
        if np.isscalar(MAC):
            if MAC < eps_MAC:
                MAC = 0
        else:
            if MAC[ii, ii] < eps_MAC:
                MAC[ii, ii] = 0

    if np.isscalar(dummyMAC):
        stab_phi = 0  # mode shape is unstable
    else:
        if (np.abs(np.linalg.eigvals(dummyMAC)) > 1).any():
            stab_phi = 0  # mode shape is unstable
        else:
            stab_phi = 1  # mode shape is stable

    return stab_phi, dummyMAC

# --------------------------- 6. Stability Check  ----------------------------#
def  stabilityCheck(fn0,zeta0,phi0,fn1,zeta1,phi1):
    eps_freq = 1e-2 
    eps_zeta = 4e-2 
    eps_MAC = 5e-3
    stability_status = []
    fn = []
    zeta = []
    phi_list = []
    MAC = []

    # frequency stability
    N0 = len(fn0)
    N1 = len(fn1)

    for rr in range(N0):
        for jj in range(N1):
            stab_fn = errorcheck(fn0[rr], fn1[jj], eps_freq)
            stab_zeta = errorcheck(zeta0[rr], zeta1[jj], eps_zeta)
            stab_phi, dummyMAC = getMAC(phi0[:, rr], phi1[:, jj], eps_MAC)

            # get stability status
            if stab_fn == 0:
                stabStatus = 0  # new pole
            elif stab_fn == 1 and stab_phi == 1 and stab_zeta == 1:
                stabStatus = 1  # stable pole
            elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 1:
                stabStatus = 2  # pole with stable frequency and vector
            elif stab_fn == 1 and stab_zeta == 1 and stab_phi == 0:
                stabStatus = 3  # pole with stable frequency and damping
            elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 0:
                stabStatus = 4  # pole with stable frequency
            else:
                raise ValueError("Error: stability_status is undefined")

            fn.append(fn1[jj])
            zeta.append(zeta1[jj])
            phi_list.append(phi1[:, jj])
            MAC.append(dummyMAC)
            stability_status.append(stabStatus)

    ind = np.argsort(fn)
    fn = np.sort(fn)
    zeta = np.array(zeta)[ind]
    phi = np.column_stack(phi_list)[:, ind]
    MAC = np.array(MAC)[ind]
    stability_status = np.array(stability_status)[ind]

    return fn, zeta, phi, MAC, stability_status

# --------------------------- 7. error check ---------------------------------#
def errorcheck(xo,x1,eps):
    if abs(1-xo/x1)<eps:
        y = 1
    else:
        y = 0
    return y
# --------------------------- 8. Get Mac -------------------------------------#
# def getMAC(x0,x1,eps):
#     Num = np.abs(np.matmul(x0[:],(x1[:].reshape(-1,1))))**2
#     D1 = np.matmul(x0[:],(x0[:].reshape(-1,1)))
#     D2 = np.matmul(x1[:],(x1[:].reshape(-1,1)))
#     dummpyMac = Num/(D1*D2)
#     if dummpyMac >(1-eps):
#         y = 1
#     else:
#         y = 0 
#     return  y,dummpyMac

def getMAC(x0, x1, eps):
    Num = np.abs(np.dot(x0.flatten(), x1.flatten()))**2
    D1 = np.dot(x0.flatten(), x0.flatten())
    D2 = np.dot(x1.flatten(), x1.flatten())
    dummyMAC = Num / (D1 * D2)
    if dummyMAC > (1 - eps):
        y = 1
    else:
        y = 0
    return y, dummyMAC
# --------------------------- 9. flip dictionary -----------------------------#
def flip_dic(a):
    d = OrderedDict(a)
    dreversed = OrderedDict()
    for k in reversed(d):
        dreversed[k] = d[k]        
    return dreversed
# --------------------------- 9.Get stable Poles -----------------------------#
# def getStablePoles(fn,zeta,phi,MAC,stablity_status):
#     fnS = np.array([])
#     zetaS = np.array([])
#     phiS = []
#     MACS= np.array([])
#     for i in range(len(fn)):
#         for j in range(len(stablity_status[i])):
#             if stablity_status[i][j]==3 or stablity_status[i][j] == 1:

#                 fnS = np.append(fnS,fn[i][j])
#                 zetaS = np.append(zetaS,zeta[i][j])
#                 dummyyS= phi[i][:,j]
#                 phiS.append(dummyyS)
#                 MACS= np.append(MACS,MAC[i][j])           
#     phiS = np.asarray(phiS ) 
#     phiS = phiS.T
#     #  remove negative damping
#     fnS = np.delete(fnS,np.where(zetaS<0))
#     phiS= np.delete(phiS,np.where(zetaS<0),1)
#     MACS =  np.delete(MACS,np.where(zetaS<0))
#     zetaS = np.delete(zetaS,np.where(zetaS<0))
    
#     # for oo in range(phiS.shape[1]):
#     #     phiS[:,oo] = phiS[:,oo]/np.linalg.norm(phiS[:,oo])
#     #     if np.diff(phiS[0:1,oo]) < 0:
#     #         phiS[:,0] = -phiS[:,oo]
#     for oo in range(phiS.shape[1]):
#         phiS[:, oo] = phiS[:, oo] / np.max(np.abs(phiS[:, oo]))
#         if np.diff(phiS[0:2, oo]) < 0:
#             phiS[:, oo] = -phiS[:, oo]
            
#     return fnS,zetaS,phiS,MACS 

def getStablePoles(fn, zeta, phi, MAC, stablity_status):
    fnS = []
    zetaS = []
    phiS = []
    MACS = []
    
    for i in range(len(fn)):
        for j in range(len(stablity_status[i])):
            if stablity_status[i][j] == 1:
                fnS.append(fn[i][j])
                zetaS.append(zeta[i][j])
                phiS.append(phi[i][:, j])
                MACS.append(MAC[i][j])
    
    fnS = np.array(fnS)
    zetaS = np.array(zetaS)
    phiS = np.array(phiS).T
    MACS = np.array(MACS)
    
    # Remove negative damping
    valid_indices = zetaS > 0
    fnS = fnS[valid_indices]
    phiS = phiS[:, valid_indices]
    MACS = MACS[valid_indices]
    zetaS = zetaS[valid_indices]
    
    # Normalize mode shape
    for oo in range(phiS.shape[1]):
        phiS[:, oo] = phiS[:, oo] / np.max(np.abs(phiS[:, oo]))
        if np.diff(phiS[0:2, oo]) < 0:
            phiS[:, oo] = -phiS[:, oo]
    
    return fnS, zetaS, phiS, MACS
# --------------------------------- 11.Cluster  ------------------------------#


def ClusterFun4(fn0, zeta0, phi0, Ncl, Lk_dist):
    def distance_fn_zeta(fn_zeta_i, fn_zeta_j):
        fn_i, zeta_i = fn_zeta_i
        fn_j, zeta_j = fn_zeta_j
        
        dfn = np.abs(fn_i - fn_j) / fn_j
        dzeta = np.abs(zeta_i - zeta_j) / zeta_j
        
        return dfn + dzeta
    
    Nsamples = phi0.shape[1]
    pos = np.zeros((Nsamples,Nsamples))
    
    # Remove poles with low mode shapes
    mode_shape_threshold = 1e-3
    ind = []
    for i in range(Nsamples):
        mode_shape = phi0[:,i]
        if np.allclose(mode_shape, np.zeros_like(mode_shape)) or np.max(np.abs(mode_shape)) < mode_shape_threshold:
            ind.append(i)
    
    fn0 = np.delete(fn0, ind)
    phi0 = np.delete(phi0, ind, 1)
    zeta0 = np.delete(zeta0, ind)
    
    for i in range(fn0.size):
        for j in range(fn0.size):
            pos[i,j] = distance_fn_zeta((fn0[i], zeta0[i]), (fn0[j], zeta0[j]))
    
    Z = linkage(pos, 'ward', 'euclidean')
    myClus = fcluster(Z, Lk_dist, criterion='distance')

    Ncluster = max(myClus)
    
    ss = 0
    fn = {}
    zeta = {}
    phi = {}

    for rr in range(Ncluster):
        if len(myClus[np.where(myClus == rr)]) > Ncl:
            dummyZeta = zeta0[np.where(myClus == rr)]
            dummyFn = fn0[np.where(myClus == rr)]
            dummyPhi = phi0[:, np.where(myClus == rr)[0]]
            
            fn[ss] = dummyFn
            zeta[ss] = dummyZeta
            phi[ss] = dummyPhi
            ss += 1 

    return fn, zeta, phi, pos


def ClusterFun3(fn0, zeta0, phi0, Ncl, Ncluster = 5):
    def distance_fn_zeta(fn_zeta_i, fn_zeta_j):
        fn_i, zeta_i = fn_zeta_i
        fn_j, zeta_j = fn_zeta_j
        
        # Calculate the absolute differences in frequency and damping
        dfn = np.abs(fn_i - fn_j) / fn_j
        dzeta = np.abs(zeta_i - zeta_j) / zeta_j
        
        # Return the sum of the absolute differences
        return dfn + dzeta
    # Calculate the pairwise distance matrix
    pos = np.zeros((len(fn0), len(fn0)))
    for i in range(len(fn0)):
        for j in range(len(fn0)):
            pos[i,j] = distance_fn_zeta((fn0[i], zeta0[i]), (fn0[j], zeta0[j]))
            stab_phi, MAC0 = getMAC(phi0[:,i], phi0[:,j], 0.05)
            # pos[i,j] += 1 - MAC0

    
    # Cluster the data using K-means
    kmeans = KMeans(n_clusters=Ncluster).fit(pos)
    myClus = kmeans.labels_
    
    # Extract the clusters with more than Ncl elements
    fn = {}
    zeta = {}
    phi = {}
    ss = 0
    for rr in range(Ncluster):
        if len(myClus[np.where(myClus == rr)]) > Ncl:
            dummyZeta = zeta0[np.where(myClus == rr)]
            dummyFn = fn0[np.where(myClus == rr)]
            dummyPhi = phi0[:, np.where(myClus == rr)[0]]
            valMin = max(0, (np.quantile(dummyZeta, 0.25) - abs(np.quantile(dummyZeta, 0.75) - np.quantile(dummyZeta, 0.25)) * 1.5))
            valMax = np.quantile(dummyZeta, 0.75) + abs(np.quantile(dummyZeta, 0.75) - np.quantile(dummyZeta, 0.25)) * 1.5

            dummyFn = np.delete(dummyFn, np.where((dummyZeta > valMax) | (dummyZeta < valMin)))
            dummyPhi = np.delete(dummyPhi, np.where((dummyZeta > valMax) | (dummyZeta < valMin))[0], 1)
            dummyZeta = np.delete(dummyZeta, np.where((dummyZeta > valMax) | (dummyZeta < valMin)))

            fn[ss] = dummyFn
            zeta[ss] = dummyZeta
            phi[ss] = dummyPhi
            ss += 1 

    return fn, zeta, phi, pos



def ClusterFun2(fn0,zeta0,phi0,Ncl,Lk_dist): 
    Nsamples = phi0.shape[1]
    pos = np.zeros((Nsamples,Nsamples))
    for i in range(Nsamples):
        for j in range(Nsamples):
             stab_phi,MAC0 = getMAC(phi0[:,i],phi0[:,j],0.05)
             pos[i,j] = np.abs((fn0[i]-fn0[j])/(fn0[j]))+np.abs((zeta0[i]-zeta0[j])/(zeta0[j]))#+1-MAC0
             
    # Modify pos to achieve fewer clusters as Lk_dist increases
    thresh = np.exp(-Lk_dist/10) # threshold value, adjust the denominator to control the effect
    pos[pos > thresh] = thresh
    
    Z = linkage(pos,'single','euclidean')
    myClus = fcluster(Z,Lk_dist,criterion = 'distance')

    Ncluster = max(myClus)
    
    ss= 0
    fn = {}
    zeta = {}
    phi = {}

    for rr in range(Ncluster):
        if len(myClus[np.where(myClus == rr)])>Ncl:
         
            dummyZeta = zeta0[np.where(myClus==rr)]
            dummyFn = fn0[np.where(myClus==rr)]
            dummyPhi = phi0[:,np.where(myClus==rr)[0]]
            valMin = max(0,(np.quantile(dummyZeta,0.25)-abs(np.quantile(dummyZeta,0.75)-np.quantile(dummyZeta,0.25))*1.5))
            valMax = np.quantile(dummyZeta,0.75)+abs(np.quantile(dummyZeta,0.75)-np.quantile(dummyZeta,0.25))*1.5
            
            dummyFn = np.delete(dummyFn,np.where((dummyZeta>valMax)|(dummyZeta<valMin)))
            dummyPhi = np.delete(dummyPhi ,np.where((dummyZeta>valMax)|(dummyZeta<valMin))[0],1)
            dummyZeta = np.delete(dummyZeta,np.where((dummyZeta>valMax)|(dummyZeta<valMin)))
            
      
            fn[ss]= dummyFn
            zeta[ss] = dummyZeta
            
            phi[ss] = dummyPhi
            ss= ss +1 
   
    return fn,zeta,phi,pos


def ClusterFun(fn0,zeta0,phi0,Ncl,Lk_dist): 
    Nsamples = phi0.shape[1]
    pos = np.zeros((Nsamples,Nsamples))
    for i in range(Nsamples):
        for j in range(Nsamples):
             stab_phi,MAC0 = getMAC(phi0[:,i],phi0[:,j],0.05)
             pos[i,j] = np.abs((fn0[i]-fn0[j])/(fn0[j]))+np.abs((zeta0[i]-zeta0[j])/(zeta0[j]))#+1-MAC0
             
    
    Z = linkage(pos,'single','euclidean')
    myClus = fcluster(Z,Lk_dist,criterion = 'distance')

    Ncluster = max(myClus)
    
    ss= 0
    fn = {}
    zeta = {}
    phi = {}

    for rr in range(Ncluster):
        if len(myClus[np.where(myClus == rr)])>Ncl:
         
            dummyZeta = zeta0[np.where(myClus==rr)]
            dummyFn = fn0[np.where(myClus==rr)]
            dummyPhi = phi0[:,np.where(myClus==rr)[0]]
            valMin = max(0,(np.quantile(dummyZeta,0.25)-abs(np.quantile(dummyZeta,0.75)-np.quantile(dummyZeta,0.25))*1.5))
            valMax = np.quantile(dummyZeta,0.75)+abs(np.quantile(dummyZeta,0.75)-np.quantile(dummyZeta,0.25))*1.5
            
            dummyFn = np.delete(dummyFn,np.where((dummyZeta>valMax)|(dummyZeta<valMin)))
            dummyPhi = np.delete(dummyPhi ,np.where((dummyZeta>valMax)|(dummyZeta<valMin))[0],1)
            dummyZeta = np.delete(dummyZeta,np.where((dummyZeta>valMax)|(dummyZeta<valMin)))
            
      
            fn[ss]= dummyFn
            zeta[ss] = dummyZeta
            
            phi[ss] = dummyPhi
            ss= ss +1 
   
    return fn,zeta,phi,pos
# -------------------------------- 12. Results  ------------------------------#
def Cluster_Resuls(fn,zeta):
    std,mean,mean_d,std_d = np.array([]),np.array([]),np.array([]),np.array([])
    for i in range(len(fn)):
        std = np.append(std,np.std(fn[i])/len(fn[i]))
        mean = np.append(mean,np.mean(fn[i]))
        mean_d = np.append(mean_d,np.mean(zeta[i]))
        std_d = np.append(std_d,np.std(zeta[i]))
        
    idd = np.argsort(std)
    std = std[idd]
    mean = mean[idd]
    mean_d = mean_d[idd]
    std_d = std_d[idd]    
    return mean,mean_d


def organize_results(fn, zeta, phi):
    results = []
    for cluster in fn.keys():
        for i in range(len(fn[cluster])):
            pole_data = {
                "frequency": fn[cluster][i],
                "damping": zeta[cluster][i],
                "mode_shape": phi[cluster][:, i]
            }
            results.append(pole_data)
    return results

def organize_results2(fn, zeta, phi, mode_shape_threshold):
    results = []
    total_poles = sum(len(fn[cluster]) for cluster in fn.keys())
    print(total_poles)
    for cluster in fn.keys():
        if len(fn[cluster]) > 0.5*total_poles:
            continue
        for i in range(len(fn[cluster])):
            mode_shape = phi[cluster][:, i]
            if np.allclose(mode_shape, np.zeros_like(mode_shape)):
                continue
            if np.abs(mode_shape).max() < mode_shape_threshold:
                continue
            pole_data = {
                "frequency": fn[cluster][i],
                "damping": zeta[cluster][i],
                "mode_shape": mode_shape
            }
            results.append(pole_data)
    return results

def organize_results3(fn, zeta, phi, mode_shape_threshold):
    results = []
    total_poles = sum(len(fn[cluster]) for cluster in fn.keys())
    mean_cluster_size = total_poles / len(fn.keys())
    for cluster in fn.keys():
        if len(fn[cluster]) < mean_cluster_size:
            continue
        for i in range(len(fn[cluster])):
            mode_shape = phi[cluster][:, i]
            if np.allclose(mode_shape, np.zeros_like(mode_shape)):
                continue
            if np.abs(mode_shape).max() < mode_shape_threshold:
                continue
            pole_data = {
                "frequency": fn[cluster][i],
                "damping": zeta[cluster][i],
                "mode_shape": mode_shape
            }
            results.append(pole_data)
    return results



def print_organized_results(results):
    print(results)
    headers = ["Pole", "Frequency", "Damping", "Mode Shape"]
    table_data = []

    for i, pole_data in enumerate(results):
        row = [
            i + 1,
            pole_data["frequency"],
            pole_data["damping"],
            np.array2string(pole_data["mode_shape"], precision=2, separator=',')
        ]
        table_data.append(row)

    table = tabulate(table_data, headers=headers, tablefmt="pretty", floatfmt=".2f")
    print(table)



def cluster_summary(fn, zeta, phi):
    n_clusters = len(fn)
    summary = {}
    summary_reduce = {}

    for i in range(n_clusters):
        cluster_size = len(fn[i])
        freq_median = np.median(fn[i])
        freq_std = np.std(fn[i])
        zeta_median = np.median(zeta[i])
        zeta_std = np.std(zeta[i])
        phi_median = np.median(phi[i], axis=1)
        phi_std = np.std(phi[i], axis=1)

        if cluster_size > np.percentile([len(fn[c]) for c in fn], 75):
            summary[i + 1] = {
                "Number of elements": cluster_size,
                "Frequency Median": round(freq_median, 3),
                "Damping Median": round(zeta_median, 3),
                "Mode shapes Median": np.round(phi_median, 3)
            }

            summary_reduce[i + 1] = {
                "Number of elements": cluster_size,
                "Frequency Median": round(freq_median, 3),
                "Damping Median": round(zeta_median, 3),
                "Mode shapes Median": np.round(phi_median, 3)
            }

    short_headers = ["CLS", "# of Elements", "Freq.", "Damping", "Mode shapes"]
    short_table_data = [[i, summary[i]["Number of elements"], summary[i]["Frequency Median"],
                         summary[i]["Damping Median"], summary[i]["Mode shapes Median"]] for i in summary]

    print(tabulate(short_table_data, headers=short_headers))

    return summary, summary_reduce




def CPSD(Acc,fs,Nc,fo,fi):
    def nextpow2(Acc):
        N = Acc.shape[0]
        _ex = np.round(np.log2(N),0)
        Nfft = 2**(_ex+1)
        return int(Nfft)
    # Acc: Acceleration Matriz NcxN
    # fs:  Sampling Frequency
    # Nc:  Number of channels
    AN = nextpow2(Acc)
    # Memory alocation for the matrix
    PSD = np.zeros((Nc,Nc,int(AN/2)+1),dtype='complex128')
    freq= np.zeros((Nc,Nc,int(AN/2)+1),dtype='complex128')

    for i in range(Nc):
        for j in range(Nc):
            f, Pxy = signal.csd(Acc[:,i], Acc[:,j], fs, nfft=AN,nperseg=2**11,noverlap = None,window='hamming')
            freq[i,j]= f
            PSD[i,j]= Pxy
    TSx = np.trace(PSD)/len(f)      
    idd = (np.where((f>= fo) & (f <= fi)))
    freq_id= f[idd]
    TSxx= np.abs(TSx[idd])
    N = len(freq_id)
    
    return freq_id,TSxx,N




def plot_all_poles_clear(fn1_list,i_list,Acc,fs,fo,fi,summary2):
    # pxx,freqss = plt.psd(Acc[:,4],int(len(Acc[:,2])/2)+1,fs,color = 'blue')
    # plt.close()
    
    freq_id,TSxx,N = CPSD(Acc,fs,Acc.shape[1],fo,fi)
    fig, ax1 = plt.subplots()
    color = 'blue'

    ax1.set_xlabel('frequency [Hz]')
    ax1.set_ylabel('Power Spectral Density', color=color)

    # Check units of TSxx
    ax1.plot(freq_id,10*np.log10(TSxx/N),color,label = 'Trace')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('order', color=color)
    relevant_freqs = []
    for key in summary2:
        relevant_freqs.append(summary2[key]['Frequency Median'])
    # Define tolerance
    tolerance = 0.01
    filtered_fn1_list = []
    for fn1 in fn1_list:
        filtered_fn1 = [f for f in fn1 if any(abs(f - rf) <= tolerance for rf in relevant_freqs)]
        filtered_fn1_list.append(filtered_fn1)
        
    # Plot relevant frequencies
    for fn1, i in zip(filtered_fn1_list, i_list):
        ax2.plot(fn1,i*np.ones(len(fn1)),'x',color=color)
    ax2.tick_params(axis='y', labelcolor=color)



def plot_all_poles(fn1_list,i_list,Acc,fs,fo,fi):
    freq_id,TSxx,N = CPSD(Acc,fs,Acc.shape[1],fo,fi)
    fig, ax1 = plt.subplots()
    color = 'blue'

    ax1.set_xlabel('frequency [Hz]')
    ax1.set_ylabel('Power Spectral Density', color=color)

    # Check units of TSxx
    ax1.plot(freq_id,10*np.log10(TSxx/N),color,label = 'Trace')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('order', color=color)
    
    # Plot the fn1 values for each iteration
    for fn1, i in zip(fn1_list, i_list):
        ax2.plot(fn1,i*np.ones(len(fn1)),'x',color=color)
    ax2.tick_params(axis='y', labelcolor=color)




def plotStabDiag(fn, Acc, fs, stability_status, Nmin, Nmax, Nc, fo, fi):
    # CPSD function
    def CPSD(Acc, fs, Nc, fo, fi):
        def nextpow2(Acc):
            N = Acc.shape[0]
            _ex = np.round(np.log2(N), 0)
            Nfft = 2 ** (_ex + 1)
            return int(Nfft)

        AN = nextpow2(Acc)
        PSD = np.zeros((Nc, Nc, int(AN / 2) + 1), dtype='complex128')
        freq = np.zeros((Nc, Nc, int(AN / 2) + 1), dtype='complex128')

        for i in range(Nc):
            for j in range(Nc):
                f, Pxy = signal.csd(Acc[:, i], Acc[:, j], fs, nfft=AN, nperseg=2 ** 11, noverlap=None, window='hamming')
                freq[i, j] = f
                PSD[i, j] = Pxy
        TSx = np.trace(PSD) / len(f)
        idd = (np.where((f >= fo) & (f <= fi)))
        freq_id = f[idd]
        TSxx = np.abs(TSx[idd])
        N = len(freq_id)

        return freq_id, TSxx, N

    freq_id, TSxx, N = CPSD(Acc, fs, Nc, fo, fi)

    Npoles = np.arange(Nmin, Nmax + 1)
    # f, Saz = welch(Acc, fs=fs)
    fig, ax1 = plt.subplots()

    # Plot stability_status
    markers = ['k+', 'ro', 'bo', 'gs', 'gx']
    labels = ['new pole', 'stable pole', 'stable freq. & MAC', 'stable freq. & damp.', 'stable freq.']
    handles = []
    for jj in range(5):
        x = []
        y = []
        for ii in range(len(fn)):
            try:
                ind = np.where(stability_status[ii] == jj)
                x.extend(fn[ii][ind])
                y.extend([Npoles[ii]] * len(fn[ii][ind]))
            except:
                print(stability_status[ii] , "******************")
        h, = ax1.plot(x, y, markers[jj], label=labels[jj])
        handles.append(h)
    ax1.set_ylabel('number of poles')
    ax1.set_xlabel('f (Hz)')
    ax1.set_ylim(0, Nmax+2)
    ax2 = ax1.twinx()       

    max_fn2 = np.max([np.max(v) for v in fn.values()])
    
    # Plot CPSD
    color = 'blue'
    ax2.set_xlabel('frequency [Hz]')
    ax2.set_ylabel('Power Spectral Density', color=color)
    ax2.plot(freq_id, 10 * np.log10(TSxx / N), color, label='Trace')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim(0,max_fn2*1.1)
    # ax2.set_yscale('log')
    ax1.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.show() 

def cluster_data_by_frequency(fnS, zetaS, phiS, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(fnS.reshape(-1, 1))
    
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    summary = {}
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        num_elements = len(cluster_indices)
        freq_centroid = centroids[i, 0]

        cluster_damping = np.median(zetaS[cluster_indices])
        cluster_mode_shapes = np.median(phiS[:, cluster_indices], axis=1)

        summary[i + 1] = {
            "Number of elements": num_elements,
            "Frequency Median": round(freq_centroid, 3),
            "Damping Median": round(cluster_damping, 3),
            "Mode shapes Median": np.round(cluster_mode_shapes, 3)
        }

    short_headers = ["CLS", "# of Elements", "Freq.", "Damping", "Mode shapes"]
    short_table_data = [[i, summary[i]["Number of elements"], summary[i]["Frequency Median"],
                          summary[i]["Damping Median"], summary[i]["Mode shapes Median"]] for i in summary]

    print(tabulate(short_table_data, headers=short_headers))

    return summary