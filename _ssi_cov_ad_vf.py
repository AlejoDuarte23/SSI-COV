import _ssi_backend as SSI 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal
from tqdm import tqdm
from timeit import default_timer as timer
import pickle

def SSI_COV_AD(Acc,fs,Ts,Nc,Nmax,Nmin,Ncl,Lk_dist):
# --------------------------- 1. NexT ----------------------------------------#
    print('Doing Natural excitation')
    Star = timer()
    IRF = SSI.NexT(Acc,fs,Ts,Nc)
    print('Done Natural excitation')
    End = timer()
    print('Elapse time [s] :',End-Star)
# --------------------------- 2. Toeplitz -------------------------------------#
    
    print('Doing SVD BlokcToeplitz Matriz')    
    Star = timer()
    [U,S,V,T] = SSI.blockToeplitz(IRF)
    print('Done SVD BlokcToeplitz Matriz')    
    End = timer()
    print('Elapse time [s] :',End-Star)

    # plt.figure()
    # pxx,freqss = plt.psd(Acc[:,4],int(len(Acc[:,2])/2)+1,fs,color = 'blue');
    # plt.close()
    # fig, ax1 = plt.subplots()
    # color = 'blue'
    
    # ax1.set_xlabel('frequency [Hz]')
    # ax1.set_ylabel('Power Spectral Density', color=color)
    
    # ax1.plot(freqss, 10*np.log(pxx), color=color)
    # plt.xlim([min(freqss),max(freqss)])
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax2 = ax1.twinx() 
    # color = 'tab:red'
    # ax2.set_ylabel('order', color=color)  # we already handled the x-label with ax1

    fn1_list = []
    i_list = []
    kk=0
    print('Doing Stability Check')   
    Star = timer()
    fn2,zeta2,phi2,MAC,stablity_status = {},{},{},{},{}
    
    for i in range(Nmax,Nmin-1,-1):
        if kk == 0:
            fn0,zeta0,phi0 = SSI.modalID(U,S,i,Nc,fs)

    
        else:
            fn1,zeta1,phi1 = SSI.modalID(U,S,i,Nc,fs)
            fn1_list.append(fn1)
            i_list.append(i)
            # ax2.plot(fn1,i*np.ones(len(fn1)),'x',color=color)
            # ax2.tick_params(axis='y', labelcolor=color)        
            [a,b,c,d,e] = SSI.stabilityCheck2(fn0,zeta0,phi0,fn1,zeta1,phi1)
            
      
            fn2[kk-1]=a
            zeta2[kk-1]=b
            phi2[kk-1]=c
            MAC[kk-1]=d
            stablity_status[kk-1]=e
            # print(e)
            fn0=fn1
            zeta0=zeta1
 
            phi0=phi1  
            
            
        kk = kk +1
    End = timer()
    print('Done Stability Check')    
    print('Elapse time [s] :',End-Star)
    
    # --------------------------- 9. flip dictionary -----------------------------#
    fn2 = SSI.flip_dic(fn2)
    zeta2 = SSI.flip_dic(zeta2)
    phi2 = SSI.flip_dic(phi2)
  
    print('Doing Filter stable Poles')   
    Star = timer()
    fnS,zetaS,phiS,MACS = SSI.getStablePoles(fn2,zeta2,phi2,MAC,stablity_status)
    End = timer()
    print('Done Filter stable Poles')    
    print('Elapse time [s] :',End-Star)

    print('Doing Cluster Analysis')   
    Star = timer()   
    # fn,zeta,phi,pos = SSI.ClusterFun3(fnS, zetaS, phiS, Ncl, Ncluster = 5)
    # fn,zeta,phi,pos = SSI.ClusterFun4(fnS,zetaS,phiS,Ncl,Lk_dist)
    End = timer()
    print('Done Cluster Analysis')    
    print('Elapse time [s] :',End-Star)
    # summary,summary2 = SSI.cluster_summary(fn, zeta, phi)
    # organized_results = SSI.organize_results2(fn, zeta, phi,mode_shape_threshold = 0.0005)
    # SSI.print_organized_results(organized_results)
    # fopt,dampopt = SSI.Cluster_Resuls(fn,zeta)

    return fnS,zetaS, phiS,fn1_list,i_list,stablity_status,fn2#fopt,dampopt