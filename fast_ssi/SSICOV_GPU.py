import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Annotated, Any, TypeVar

import cupy as cp
import numpy as np
from numba import prange
from numpy.typing import NDArray

from .utils import timeit

type Array = NDArray[Any]
T = TypeVar("T")

type SamplesByChannels = Annotated[
    Array, "Acceleration time series, shape (n_samples, n_channels)"
]
type SamplingRateHz = Annotated[float, "Sampling frequency in Hz"]
type TimeLagSeconds = Annotated[
    float, "Time-lag window in seconds (Ts); M = round(Ts * fs)"
]
type ChannelCount = Annotated[int, "Number of channels/sensors (Nc)"]
type ModelOrder = Annotated[int, "Model order (state dimension)"]
type IRFArray = Annotated[
    cp.ndarray, "Impulse response function (CuPy), shape (Nc, Nc, n_lags)"
]
type ToeplitzMatrix = Annotated[
    cp.ndarray,
    "Block Toeplitz matrix (CuPy), shape (n_blocks * Nc, n_blocks * Nc) with n_blocks = round(n_lags/2) - 1",
]
type LeftSingularVectors = Annotated[
    Array, "Left singular vectors (NumPy), shape (n_rows, n_singular_vectors)"
]
type RightSingularVectors = Annotated[
    cp.ndarray, "Right singular vectors (CuPy), shape (n_cols, n_singular_vectors)"
]
type SingularValues = Annotated[
    Array, "Singular values (NumPy), shape (n_singular_values,)"
]
type NaturalFrequencies = Annotated[Array, "Natural frequencies (Hz), shape (n_modes,)"]
type DampingRatios = Annotated[Array, "Damping ratios, shape (n_modes,)"]
type ModeShapes = Annotated[Array, "Mode shapes, shape (n_channels, n_modes)"]
type MACValues = Annotated[Array, "MAC values, shape (n_pairs,)"]
type StabilityCodes = Annotated[
    Array, "Stability codes per pole pair, shape (n_pairs,)"
]


def blockToeplitz_jit(
    IRF: IRFArray,
) -> tuple[LeftSingularVectors, SingularValues, RightSingularVectors, ToeplitzMatrix]:
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
        self,
        acc: SamplesByChannels,
        fs: SamplingRateHz,
        Ts: TimeLagSeconds,
        Nc: ChannelCount,
        Nmax: ModelOrder,
        Nmin: ModelOrder,
    ) -> None:
        self.acc = cp.array(acc)
        self.fs = fs
        self.Ts = Ts
        self.Nc = Nc
        self.Nmax = Nmax
        self.Nmin = Nmin

    @timeit
    def NexT(self) -> IRFArray:
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
            raise ValueError(
                "Nc==1 is not supported; Toeplitz construction expects 3D IRF."
            )
        return IRF

    @timeit
    def blockToeplitz(
        self, IRF: IRFArray
    ) -> tuple[
        LeftSingularVectors, SingularValues, RightSingularVectors, ToeplitzMatrix
    ]:
        return blockToeplitz_jit(IRF)

    @timeit
    def modalID(
        self,
        U: LeftSingularVectors,
        S: SingularValues,
        Nmodes: ModelOrder,
        Nyy: ChannelCount,
        fs: SamplingRateHz,
    ) -> tuple[NaturalFrequencies, DampingRatios, ModeShapes]:
        S = np.diag(S)
        if Nmodes >= S.shape[0]:
            print("changing the number of modes to the maximum possible")
            Nmodes = S.shape[0]
        dt = 1 / self.fs
        obs = np.matmul(U[:, 0:Nmodes], np.sqrt(S[0:Nmodes, 0:Nmodes]))
        IndO = min(Nyy, len(obs[:, 0]))
        C = obs[0:IndO, :]
        jb = obs.shape[0] / IndO
        ao = int((IndO) * (jb - 1))
        bo = int(len(obs[:, 0]) - (IndO) * (jb - 1))
        co = len(obs[:, 0])
        A = np.matmul(np.linalg.pinv(obs[0:ao, :]), obs[bo:co, :])
        eigvals, eigvecs = np.linalg.eig(A)
        lam = np.log(eigvals) / dt
        fno = np.abs(lam) / (2 * np.pi)
        zetaoo = -np.real(lam) / np.abs(lam)
        keep = np.imag(lam) > 0
        if not np.any(keep):
            keep = np.ones_like(lam, dtype=bool)
        fn = fno[keep]
        zeta = zetaoo[keep]
        phi0 = C @ eigvecs
        phi = phi0[:, keep]
        return fn, zeta, phi

    @timeit
    def stabilityCheck(
        self,
        fn0: NaturalFrequencies,
        zeta0: DampingRatios,
        phi0: ModeShapes,
        fn1: NaturalFrequencies,
        zeta1: DampingRatios,
        phi1: ModeShapes,
    ) -> tuple[
        NaturalFrequencies,
        DampingRatios,
        ModeShapes,
        MACValues,
        StabilityCodes,
    ]:
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

        for rr in range(N0):
            for jj in range(N1):
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

    def errorcheck(
        self,
        xo: Annotated[float, "Reference value (scalar)"],
        x1: Annotated[float, "Compared value (scalar)"],
        eps: Annotated[float, "Relative tolerance threshold"],
    ) -> int:
        return 1 if abs(1 - xo / x1) < eps else 0

    def getMAC(
        self,
        x0: Annotated[Array, "Mode shape vector, shape (n_channels,)"],
        x1: Annotated[Array, "Mode shape vector, shape (n_channels,)"],
        eps: Annotated[float, "MAC tolerance threshold"],
    ) -> tuple[
        Annotated[int, "1 if MAC > (1 - eps), else 0"],
        Annotated[float, "Modal Assurance Criterion value"],
    ]:
        Num = np.abs(np.vdot(x0.flatten(), x1.flatten())) ** 2
        D1 = np.vdot(x0.flatten(), x0.flatten())
        D2 = np.vdot(x1.flatten(), x1.flatten())
        dummyMAC = Num / (D1 * D2)
        y = 1 if dummyMAC > (1 - eps) else 0
        return y, dummyMAC

    def flip_dic(
        self,
        a: Annotated[Mapping[int, T], "Mapping to reverse while preserving order"],
    ) -> OrderedDict[int, T]:
        d = OrderedDict(a)
        dreversed = OrderedDict()
        for k in reversed(d):
            dreversed[k] = d[k]
        return dreversed

    @timeit
    def getStablePoles(
        self,
        fn: Annotated[
            Mapping[int, Array],
            "Mapping: model order -> frequencies array, shape (n_i,)",
        ],
        zeta: Annotated[
            Mapping[int, Array],
            "Mapping: model order -> damping ratios array, shape (n_i,)",
        ],
        phi: Annotated[
            Mapping[int, Array],
            "Mapping: model order -> mode shapes, shape (n_channels, n_i)",
        ],
        MAC: Annotated[
            Mapping[int, Array],
            "Mapping: model order -> MAC values, shape (n_i,)",
        ],
        stablity_status: Annotated[
            Mapping[int, Array],
            "Mapping: model order -> stability codes, shape (n_i,)",
        ],
    ) -> tuple[NaturalFrequencies, DampingRatios, ModeShapes, MACValues]:
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
            ref = phiS[0, oo]
            if ref != 0:
                phiS[:, oo] *= np.exp(-1j * np.angle(ref))
            if np.real(phiS[0, oo]) < 0:
                phiS[:, oo] = -phiS[:, oo]

        return fnS, zetaS, phiS, MACS

    @timeit
    def run_stability(
        self, U: LeftSingularVectors, S: SingularValues
    ) -> tuple[
        Annotated[
            Mapping[int, Array],
            "Frequencies by model order, each shape (n_i,)",
        ],
        Annotated[
            Mapping[int, Array],
            "Damping ratios by model order, each shape (n_i,)",
        ],
        Annotated[
            Mapping[int, Array],
            "Mode shapes by model order, each shape (n_channels, n_i)",
        ],
        Annotated[
            Mapping[int, Array],
            "MAC values by model order, each shape (n_i,)",
        ],
        Annotated[
            Mapping[int, Array],
            "Stability codes by model order, each shape (n_i,)",
        ],
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

        return fn2, zeta2, phi2, MAC, stability_status

    def run(
        self,
    ) -> tuple[
        NaturalFrequencies,
        DampingRatios,
        ModeShapes,
        MACValues,
        Annotated[
            Mapping[int, Array],
            "Stability codes by model order, each shape (n_i,)",
        ],
        Annotated[
            Mapping[int, Array],
            "Frequencies by model order, each shape (n_i,)",
        ],
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
