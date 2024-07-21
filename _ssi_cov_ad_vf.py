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
     
            [a,b,c,d,e] = SSI.stabilityCheck2(fn0,zeta0,phi0,fn1,zeta1,phi1)
            
      
            fn2[kk-1]=a
            zeta2[kk-1]=b
            phi2[kk-1]=c
            MAC[kk-1]=d
            stablity_status[kk-1]=e
            
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
    End = timer()
    print('Done Cluster Analysis')    
    print('Elapse time [s] :',End-Star)


    return fnS,zetaS, phiS,fn1_list,i_list,stablity_status,fn2#fopt,dampopt