import time
from typing import Annotated

import cupy as cp
from numba import prange

from .SSICOV import SSICOV as SSICOVBase
from .types import (
    ChannelCount,
    LeftSingularVectors,
    ModelOrder,
    SamplesByChannels,
    SamplingRateHz,
    SingularValues,
    TimeLagSeconds,
)
from .utils import timeit

type IRFArray = Annotated[
    cp.ndarray, "Impulse response function (CuPy), shape (Nc, Nc, n_lags)"
]
type ToeplitzMatrix = Annotated[
    cp.ndarray,
    "Block Toeplitz matrix (CuPy), shape (n_blocks * Nc, n_blocks * Nc) with n_blocks = round(n_lags/2) - 1",
]
type RightSingularVectors = Annotated[
    cp.ndarray, "Right singular vectors (CuPy), shape (n_cols, n_singular_vectors)"
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
                :, :, N1 - 1 + oo - ll + 1
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


class SSICOV(SSICOVBase):
    def __init__(
        self,
        acc: SamplesByChannels,
        fs: SamplingRateHz,
        Ts: TimeLagSeconds,
        Nc: ChannelCount,
        Nmax: ModelOrder,
        Nmin: ModelOrder,
    ) -> None:
        super().__init__(acc, fs, Ts, Nc, Nmax, Nmin)
        self.acc = cp.array(self.acc)

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
