import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np
from numba import prange
from numpy.typing import NDArray

from .utils import timeit

Array = NDArray[Any]
T = TypeVar("T")


def blockToeplitz_jit(IRF: Array) -> tuple[Array, Array, Array, Array]:
    N1 = round(IRF.shape[2] / 2) - 1
    M = IRF.shape[1]
    T1 = np.zeros(((N1) * M, (N1) * M), dtype="complex128")

    for oo in prange(N1):
        for ll in prange(N1):
            T1[(oo) * M : (oo + 1) * M, (ll) * M : (ll + 1) * M] = IRF[
                :, :, N1 - 1 + oo - ll + 1
            ]

    start = time.time()
    U, S, Vt = np.linalg.svd(T1)
    print(f" Elapse time {time.time() - start}s")
    V = Vt.T

    return U, S, V, T1


class SSICOV:
    def __init__(
        self, acc: Array, fs: float, Ts: float, Nc: int, Nmax: int, Nmin: int
    ) -> None:
        self.acc = acc
        self.fs = fs
        self.Ts = Ts
        self.Nc = Nc
        self.Nmax = Nmax
        self.Nmin = Nmin

    @timeit
    def NexT(self) -> Array:
        dt = 1 / self.fs
        M = round(self.Ts / dt)
        IRF = np.zeros((self.Nc, self.Nc, M - 1), dtype=complex)
        for oo in range(self.Nc):
            for jj in range(self.Nc):
                y1 = np.fft.fft(self.acc[:, oo])
                y2 = np.fft.fft(self.acc[:, jj])
                # cross-correlation: ifft[cross-power spectrum]
                h0 = np.fft.ifft(y1 * y2.conj())
                # impulse response function
                IRF[oo, jj, :] = np.real(h0[0 : M - 1])

        if self.Nc == 1:
            IRF = np.squeeze(IRF)
            IRF = IRF / IRF[0]
        return IRF

    @timeit
    def blockToeplitz(self, IRF: Array) -> tuple[Array, Array, Array, Array]:
        return blockToeplitz_jit(IRF)

    @timeit
    def modalID(
        self, U: Array, S: Array, Nmodes: int, Nyy: int, fs: float
    ) -> tuple[Array, Array, Array]:
        S = np.diag(S)
        if Nmodes >= S.shape[0]:
            print("changing the number of modes to the maximum possible")
            Nmodes = S.shape[0]
        dt = 1 / self.fs
        # observability matrix
        obs = np.matmul(U[:, 0:Nmodes], np.sqrt(S[0:Nmodes, 0:Nmodes]))
        # time state-space matrices
        IndO = min(Nyy, len(obs[:, 0]))
        C = obs[0:IndO, :]
        jb = obs.shape[0] / IndO
        ao = int((IndO) * (jb - 1))
        bo = int(len(obs[:, 0]) - (IndO) * (jb - 1))
        co = len(obs[:, 0])
        A = np.matmul(np.linalg.pinv(obs[0:ao, :]), obs[bo:co, :])
        # eigen vals descop. of state matrix
        eigvals, eigvecs = np.linalg.eig(A)
        mu = np.log(eigvals) / dt
        fno = np.abs(mu) / (2 * np.pi)
        zetaoo = -np.real(mu) / np.abs(mu)
        idx = np.arange(0, len(mu), 2)
        fn = fno[idx]
        zeta = zetaoo[idx]
        phi0 = np.real(C @ eigvecs)
        phi = phi0[:, idx]
        return fn, zeta, phi

    @timeit
    def stabilityCheck(
        self,
        fn0: Array,
        zeta0: Array,
        phi0: Array,
        fn1: Array,
        zeta1: Array,
        phi1: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        eps_freq = 2e-2
        eps_zeta = 4e-2
        eps_MAC = 5e-2
        stability_list: list[int] = []
        fn_list: list[float] = []
        zeta_list: list[float] = []
        phi_list: list[Array] = []
        mac_list: list[float] = []

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

                fn_list.append(fn1[jj])
                zeta_list.append(zeta1[jj])
                phi_list.append(phi1[:, jj])
                mac_list.append(dummyMAC)
                stability_list.append(stabStatus)

        fn_arr = np.array(fn_list)
        ind = np.argsort(fn_arr)
        fn_sorted = fn_arr[ind]
        zeta_arr = np.array(zeta_list)[ind]
        phi_arr = np.column_stack(phi_list)[:, ind]
        mac_arr = np.array(mac_list)[ind]
        stability_arr = np.array(stability_list)[ind]

        return fn_sorted, zeta_arr, phi_arr, mac_arr, stability_arr

    def errorcheck(self, xo: float, x1: float, eps: float) -> int:
        return 1 if abs(1 - xo / x1) < eps else 0

    def getMAC(self, x0: Array, x1: Array, eps: float) -> tuple[int, float]:
        Num = np.abs(np.dot(x0.flatten(), x1.flatten())) ** 2
        D1 = np.dot(x0.flatten(), x0.flatten())
        D2 = np.dot(x1.flatten(), x1.flatten())
        dummyMAC = Num / (D1 * D2)
        y = 1 if dummyMAC > (1 - eps) else 0
        return y, dummyMAC

    def flip_dic(self, a: Mapping[int, T]) -> OrderedDict[int, T]:
        d = OrderedDict(a)
        dreversed = OrderedDict()
        for k in reversed(d):
            dreversed[k] = d[k]
        return dreversed

    @timeit
    def getStablePoles(
        self,
        fn: Mapping[int, Array],
        zeta: Mapping[int, Array],
        phi: Mapping[int, Array],
        MAC: Mapping[int, Array],
        stablity_status: Mapping[int, Array],
    ) -> tuple[Array, Array, Array, Array]:
        fnS_list: list[float] = []
        zetaS_list: list[float] = []
        phiS_list: list[Array] = []
        macs_list: list[float] = []

        for i in range(len(fn)):
            for j in range(len(stablity_status[i])):
                if stablity_status[i][j] == 1:
                    fnS_list.append(fn[i][j])
                    zetaS_list.append(zeta[i][j])
                    phiS_list.append(phi[i][:, j])
                    macs_list.append(MAC[i][j])

        fnS = np.array(fnS_list)
        zetaS = np.array(zetaS_list)
        phiS = np.array(phiS_list).T
        MACS = np.array(macs_list)

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
    def run_stability(
        self, U: Array, S: Array
    ) -> tuple[
        OrderedDict[int, Array],
        OrderedDict[int, Array],
        OrderedDict[int, Array],
        OrderedDict[int, Array],
        OrderedDict[int, Array],
    ]:
        fn1_list: list[Array] = []
        i_list: list[int] = []
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

        fn2 = self.flip_dic(fn2)
        zeta2 = self.flip_dic(zeta2)
        phi2 = self.flip_dic(phi2)
        MAC = self.flip_dic(MAC)
        stability_status = self.flip_dic(stability_status)

        return fn2, zeta2, phi2, MAC, stability_status

    def run(
        self,
    ) -> tuple[
        Array,
        Array,
        Array,
        Array,
        OrderedDict[int, Array],
        OrderedDict[int, Array],
    ]:
        IRF = self.NexT()
        [U, S, V, T] = self.blockToeplitz(IRF)
        # fn2, zeta2, phi2, MAC, stability_status = self.run_stability(U, S)
        fn1_list: list[Array] = []
        i_list: list[int] = []
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

        fn2 = self.flip_dic(fn2)
        zeta2 = self.flip_dic(zeta2)
        phi2 = self.flip_dic(phi2)
        MAC = self.flip_dic(MAC)
        stability_status = self.flip_dic(stability_status)

        fnS, zetaS, phiS, MACS = self.getStablePoles(
            fn2, zeta2, phi2, MAC, stability_status
        )

        return fnS, zetaS, phiS, MACS, stability_status, fn2
