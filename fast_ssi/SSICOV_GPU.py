# import numpy as np
from collections import OrderedDict
from numba import jit
from typing import Tuple, Dict
from .utils import print_input_sizes, timeit
from numpy.typing import NDArray
from numba import jit, prange
import cupy as cp

import time
import numpy as np
import cupyx.linalg


def blockToeplitz_jit(
    IRF: cp.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    N1 = round(IRF.shape[2] / 2) - 1
    M = IRF.shape[1]
    IRF_cp = cp.array(IRF)  # Convert NumPy array to CuPy array

    T1 = cp.zeros((N1 * M, N1 * M), dtype=cp.complex128)

    # Replace prange with range if using 'numba' is not part of the requirement.
    for oo in prange(N1):
        for ll in prange(N1):
            # NumPy and CuPy indexing is similar. This should remain unchanged.
            T1[oo * M : (oo + 1) * M, ll * M : (ll + 1) * M] = IRF_cp[
                :, :, N1 - 1 + oo - ll
            ]

    # Singular Value Decomposition (SVD)
    start = time.time()
    _U, _S, Vt = cp.linalg.svd(T1)
    print(f"Elapse time {start - time.time()}")
    # Transpose Vt to get V
    V = Vt.T
    U = cp.asnumpy(_U)
    S = cp.asnumpy(_S)

    return U, S, V, T1


class SSICOV:
    def __init__(
        self, acc: NDArray, fs: float, Ts: float, Nc: int, Nmax: int, Nmin: int
    ) -> None:
        self.acc = cp.array(acc)
        self.fs = fs
        self.Ts = Ts
        self.Nc = Nc
        self.Nmax = Nmax
        self.Nmin = Nmin

    @timeit
    def NexT(self) -> cp.ndarray:
        dt = 1 / self.fs
        M = round(self.Ts / dt)
        IRF = cp.zeros(
            (self.Nc, self.Nc, M - 1), dtype=cp.complex64
        )  # Use cupy's complex64 dtype
        for oo in range(self.Nc):
            for jj in range(self.Nc):
                y1 = cp.fft.fft(self.acc[:, oo])
                y2 = cp.fft.fft(self.acc[:, jj])
                # cross-correlation: ifft of [cross-power spectrum]
                h0 = cp.fft.ifft(y1 * y2.conj())
                # impulse response function
                IRF[oo, jj, :] = cp.real(h0[: M - 1])  # Using cupy's real method

        if self.Nc == 1:
            IRF = IRF.squeeze()  # Use cupy's squeeze method
            IRF = IRF / IRF[0]  # Normalization with cupy's arithmetic
        return IRF

    @timeit
    def blockToeplitz(self, IRF: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        return blockToeplitz_jit(IRF)

    @timeit
    def modalID(self, U, S, Nmodes, Nyy, fs):
        S = np.diag(S)
        if Nmodes >= S.shape[0]:
            print("changing the number of modes to the maximum possible")
            Nmodes = S.shape[0]
        dt = 1 / self.fs
        O = np.matmul(U[:, 0:Nmodes], np.sqrt(S[0:Nmodes, 0:Nmodes]))
        IndO = min(Nyy, len(O[:, 0]))
        C = O[0:IndO, :]
        jb = O.shape[0] / IndO
        ao = int((IndO) * (jb - 1))
        bo = int(len(O[:, 0]) - (IndO) * (jb - 1))
        co = len(O[:, 0])
        A = np.matmul(np.linalg.pinv(O[0:ao, :]), O[bo:co, :])
        [Vi, Di] = np.linalg.eig(A)
        mu = np.log(np.diag(np.diag(Vi))) / dt
        fno = np.abs(mu) / (2 * np.pi)
        fn = fno[np.ix_(*[range(0, i, 2) for i in fno.shape])]
        zetaoo = -np.real(mu) / np.abs(mu)
        zeta = zetaoo[np.ix_(*[range(0, i, 2) for i in zetaoo.shape])]
        phi0 = np.real(np.matmul(C[0:IndO, :], Di))
        phi = phi0[:, 1::2]
        return fn, zeta, phi

    @timeit
    def stabilityCheck(self, fn0, zeta0, phi0, fn1, zeta1, phi1):
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

        for rr in range(N0 - 1):
            for jj in range(N1 - 1):
                stab_fn = self.errorcheck(fn0[rr], fn1[jj], eps_freq)
                stab_zeta = self.errorcheck(zeta0[rr], zeta1[jj], eps_zeta)
                stab_phi, dummyMAC = self.getMAC(phi0[:, rr], phi1[:, jj], eps_MAC)

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

    def errorcheck(self, xo, x1, eps):
        if abs(1 - xo / x1) < eps:
            y = 1
        else:
            y = 0
        return y

    def getMAC(self, x0, x1, eps):
        Num = np.abs(np.dot(x0.flatten(), x1.flatten())) ** 2
        D1 = np.dot(x0.flatten(), x0.flatten())
        D2 = np.dot(x1.flatten(), x1.flatten())
        dummyMAC = Num / (D1 * D2)
        if dummyMAC > (1 - eps):
            y = 1
        else:
            y = 0
        return y, dummyMAC

    def flip_dic(self, a) -> OrderedDict:
        d = OrderedDict(a)
        dreversed = OrderedDict()
        for k in reversed(d):
            dreversed[k] = d[k]
        return dreversed

    @timeit
    def getStablePoles(self, fn, zeta, phi, MAC, stablity_status):
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

    @timeit
    def run_stability(self, U, S):
        fn1_list = []
        i_list = []
        kk = 0
        fn2, zeta2, phi2, MAC, stability_status = {}, {}, {}, {}, {}

        for i in range(self.Nmax, self.Nmin - 1, -1):
            if kk == 0:
                fn0, zeta0, phi0 = self.modalID(U, S, i, self.Nc, self.fs)
            else:
                fn1, zeta1, phi1 = self.modalID(U, S, i, self.Nc, self.fs)
                fn1_list.append(fn1)
                i_list.append(i)

                [a, b, c, d, e] = self.stabilityCheck(
                    fn0, zeta0, phi0, fn1, zeta1, phi1
                )

                fn2[kk - 1] = a
                zeta2[kk - 1] = b
                phi2[kk - 1] = c
                MAC[kk - 1] = d
                stability_status[kk - 1] = e

                fn0 = fn1
                zeta0 = zeta1
                phi0 = phi1
        kk = kk + 1

        return fn2, zeta2, phi2, MAC, stability_status

    def run(self):
        IRF = self.NexT()
        [U, S, V, T] = self.blockToeplitz(IRF)
        # fn2, zeta2, phi2, MAC, stability_status = self.run_stability(U, S)
        fn1_list = []
        i_list = []
        kk = 0
        fn2, zeta2, phi2, MAC, stability_status = {}, {}, {}, {}, {}

        for i in range(self.Nmax, self.Nmin - 1, -1):
            if kk == 0:
                fn0, zeta0, phi0 = self.modalID(U, S, i, self.Nc, self.fs)
            else:
                fn1, zeta1, phi1 = self.modalID(U, S, i, self.Nc, self.fs)
                fn1_list.append(fn1)
                i_list.append(i)

                [a, b, c, d, e] = self.stabilityCheck(
                    fn0, zeta0, phi0, fn1, zeta1, phi1
                )

                fn2[kk - 1] = a
                zeta2[kk - 1] = b
                phi2[kk - 1] = c
                MAC[kk - 1] = d
                stability_status[kk - 1] = e

                fn0 = fn1
                zeta0 = zeta1
                phi0 = phi1
            kk = kk + 1

        fn2, zeta2, phi2 = self.flip_dic(fn2), self.flip_dic(zeta2), self.flip_dic(phi2)
        fnS, zetaS, phiS, MACS = self.getStablePoles(
            fn2, zeta2, phi2, MAC, stability_status
        )

        return fnS, zetaS, phiS, MACS, stability_status, fn2
