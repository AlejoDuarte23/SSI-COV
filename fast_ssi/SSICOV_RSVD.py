import time
from typing import Annotated

import numpy as np

from .SSICOV import SSICOV as SSICOVBase
from .types import (
    IRFArray,
    LeftSingularVectors,
    RightSingularVectors,
    SingularValues,
    ToeplitzMatrix,
)
from .utils import timeit

type Rank = Annotated[int, "Target rank for truncated/randomized SVD"]
type Oversamples = Annotated[int, "Oversampling parameter for RSVD"]
type PowerIterations = Annotated[int, "Power iteration count for RSVD"]


def build_block_toeplitz(IRF: IRFArray) -> tuple[ToeplitzMatrix, Rank]:
    """
    Builds the block Toeplitz matrix T1 from IRF.
    IRF is assumed to have shape (X, M, 2*N1+2) or similar,
    where N1 = round(IRF.shape[2]/2) - 1.
    """
    N1 = round(IRF.shape[2] / 2) - 1
    M = IRF.shape[1]
    T1 = np.zeros((N1 * M, N1 * M), dtype=np.complex128)
    T = N1 * M  # The Toeplitz dimension
    k_percent = max(30 - 0.00156 * T, 25)
    rank_value = int((k_percent / 100.0) * T)

    for oo in range(N1):
        for ll in range(N1):
            T1[oo * M : (oo + 1) * M, ll * M : (ll + 1) * M] = IRF[
                :, :, N1 - 1 + oo - ll + 1
            ]

    return T1, rank_value


def randomized_svd(
    T1: ToeplitzMatrix,
    rank: Rank,
    num_oversamples: Oversamples = 10,
    n_iter: PowerIterations = 2,
) -> tuple[LeftSingularVectors, SingularValues, RightSingularVectors]:
    """
    Computes a randomized SVD of T1.
    - Random sampling creates low-dimensional sketch of input matrix
    - Construction of orthonormal basis for sampled subspace
    - Projection of original matrix onto constructed basis
    - SVD computation on smaller projected matrix

    """
    rows, cols = T1.shape
    # Random projection
    random_matrix = np.random.randn(cols, rank + num_oversamples)
    Y = T1 @ random_matrix

    # This improve the approximation :)
    for _ in range(n_iter):
        Y = T1 @ (T1.conj().T @ Y)

    # basis (Q) for Y
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # restrict T1 to the subspace spanned by Q
    B = Q.conj().T @ T1

    # SVD on  B
    start = time.time()
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    print(f" RSVD Elapse time {time.time() - start}s")

    # eeconstruct U and V
    U = Q @ U_tilde
    V = Vt.conj().T

    # keep only the top 'rank' components
    U = U[:, :rank]
    S = S[:rank]
    V = V[:, :rank]

    return U, S, V


def blockToeplitz_jit_randomSVD(
    IRF: IRFArray,
    rank: Rank,
    num_oversamples: Oversamples = 10,
    n_iter: PowerIterations = 2,
) -> tuple[LeftSingularVectors, SingularValues, RightSingularVectors, ToeplitzMatrix]:
    T1, rank = build_block_toeplitz(IRF)
    print(f"Selected Rank:{rank}")

    U, S, V = randomized_svd(T1, rank, num_oversamples, n_iter)

    return U, S, V, T1


class SSICOV(SSICOVBase):
    @timeit
    def blockToeplitz(
        self, IRF: IRFArray
    ) -> tuple[
        LeftSingularVectors, SingularValues, RightSingularVectors, ToeplitzMatrix
    ]:
        return blockToeplitz_jit_randomSVD(IRF, rank=50)
