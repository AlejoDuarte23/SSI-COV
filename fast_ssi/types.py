from typing import Annotated, Any

from numpy.typing import NDArray

type Array = NDArray[Any]

type SamplesByChannels = Annotated[
    Array, "Acceleration time series, shape (n_samples, n_channels)"
]
type SamplingRateHz = Annotated[float, "Sampling frequency in Hz"]
type TimeLagSeconds = Annotated[
    float, "Time-lag window in seconds (Ts); M = round(Ts * fs)"
]
type ChannelCount = Annotated[int, "Number of channels/sensors (Nc)"]
type ModelOrder = Annotated[int, "Model order (state dimension)"]

type IRFArray = Annotated[Array, "Impulse response function, shape (Nc, Nc, n_lags)"]
type ToeplitzMatrix = Annotated[
    Array,
    "Block Toeplitz matrix, shape (n_blocks * Nc, n_blocks * Nc) with n_blocks = round(n_lags/2) - 1",
]
type LeftSingularVectors = Annotated[
    Array, "Left singular vectors, shape (n_rows, n_singular_vectors)"
]
type RightSingularVectors = Annotated[
    Array, "Right singular vectors, shape (n_cols, n_singular_vectors)"
]
type SingularValues = Annotated[Array, "Singular values, shape (n_singular_values,)"]

type NaturalFrequencies = Annotated[Array, "Natural frequencies (Hz), shape (n_modes,)"]
type DampingRatios = Annotated[Array, "Damping ratios, shape (n_modes,)"]
type ModeShapes = Annotated[Array, "Mode shapes, shape (n_channels, n_modes)"]
type MACValues = Annotated[Array, "MAC values, shape (n_pairs,)"]
type StabilityCodes = Annotated[
    Array, "Stability codes per pole pair, shape (n_pairs,)"
]

type FrequencyHz = Annotated[float, "Frequency in Hz"]
type Frequencies = Annotated[Array, "Frequency vector, shape (n_freq,)"]
type SpectralDensity = Annotated[Array, "Spectral density vector, shape (n_freq,)"]
