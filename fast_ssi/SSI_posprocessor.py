from collections.abc import Mapping
from typing import Annotated, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from tabulate import tabulate

from .types import (
    Array,
    ChannelCount,
    DampingRatios,
    Frequencies,
    FrequencyHz,
    ModeShapes,
    NaturalFrequencies,
    SamplesByChannels,
    SamplingRateHz,
    SpectralDensity,
)


# CPSD function
def CPSD(
    Acc: SamplesByChannels,
    fs: SamplingRateHz,
    Nc: ChannelCount,
    fo: FrequencyHz,
    fi: FrequencyHz,
) -> tuple[
    Frequencies,
    SpectralDensity,
    Annotated[int, "Number of frequency bins in selected band"],
]:
    def nextpow2(Acc: SamplesByChannels) -> int:
        N = Acc.shape[0]
        _ex = np.round(np.log2(N), 0)
        Nfft = 2 ** (_ex + 1)
        return int(Nfft)

    AN = nextpow2(Acc)
    PSD = np.zeros((Nc, Nc, int(AN / 2) + 1), dtype="complex128")
    freq = np.zeros((Nc, Nc, int(AN / 2) + 1), dtype="complex128")

    for i in range(Nc):
        for j in range(Nc):
            f, Pxy = signal.csd(
                Acc[:, i],
                Acc[:, j],
                fs,
                nfft=AN,
                nperseg=2**11,
                noverlap=None,
                window="hamming",
            )
            freq[i, j] = f
            PSD[i, j] = Pxy

    TSx = np.trace(PSD) / len(f)
    idd = np.where((f >= fo) & (f <= fi))
    freq_id = f[idd]
    TSxx = np.abs(TSx[idd])
    N = len(freq_id)

    return freq_id, TSxx, N


def plotStabDiag(
    fn: Annotated[
        Mapping[int, Array],
        "Mapping: model order -> frequencies array, shape (n_i,)",
    ],
    Acc: SamplesByChannels,
    fs: SamplingRateHz,
    stability_status: Annotated[
        Mapping[int, Array],
        "Mapping: model order -> stability codes, shape (n_i,)",
    ],
    Nmin: Annotated[int, "Minimum model order"],
    Nmax: Annotated[int, "Maximum model order"],
    Nc: ChannelCount,
    fo: FrequencyHz,
    fi: FrequencyHz,
    stable_only: Annotated[bool, "Plot only stable poles (skip 'new pole')"] = False,
) -> None:
    freq_id, TSxx, N = CPSD(Acc, fs, Nc, fo, fi)
    nyquist = fs / 2
    f_max_data = min(fi, nyquist)

    Npoles = np.arange(Nmin, Nmax + 1)
    fig, ax1 = plt.subplots()

    # Plot stability_status
    markers = ["k+", "ro", "bo", "gs", "gx"]
    labels = [
        "new pole",
        "stable pole",
        "stable freq. & MAC",
        "stable freq. & damp.",
        "stable freq.",
    ]
    handles = []
    plot_classes = [1] if stable_only else range(5)
    for jj in plot_classes:
        x: list[float] = []
        y: list[float] = []
        for ii in range(len(fn)):
            try:
                ind = np.where(stability_status[ii] == jj)
                fvals = fn[ii][ind]
                # Low-rank RSVD can yield spurious, heavily damped poles with |λ| > Nyquist,
                # so we omit those from the stabilization diagram and enforce the fi window.
                fvals = fvals[(fvals >= fo) & (fvals <= f_max_data)]
                x.extend(fvals)
                y.extend([Npoles[ii]] * len(fvals))
            except Exception:
                print("Error !")
        (h,) = ax1.plot(x, y, markers[jj], label=labels[jj])
        handles.append(h)
    ax1.set_ylabel("number of poles")
    ax1.set_xlabel("f (Hz)")
    ax1.set_ylim(0, Nmax + 2)
    ax2 = ax1.twinx()

    # Plot CPSD
    color = "blue"
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel("Power Spectral Density", color=color)
    ax2.plot(freq_id, 10 * np.log10(TSxx / N), color, label="Trace")
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.set_xlim(fo, fi)
    ax2.set_xlim(fo, fi)
    # ax2.set_yscale('log')
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    plt.show()


def cluster_data_by_frequency(
    fnS: NaturalFrequencies,
    zetaS: DampingRatios,
    phiS: ModeShapes,
    num_clusters: Annotated[int, "Number of clusters to form"],
) -> Annotated[dict[int, dict[str, Any]], "Cluster summary keyed by cluster id"]:
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
            "Mode shapes Median": np.round(cluster_mode_shapes, 3),
        }

    short_headers = ["CLS", "# of Elements", "Freq.", "Damping", "Mode shapes"]
    short_table_data = [
        [
            i,
            summary[i]["Number of elements"],
            summary[i]["Frequency Median"],
            summary[i]["Damping Median"],
            summary[i]["Mode shapes Median"],
        ]
        for i in summary
    ]

    print(tabulate(short_table_data, headers=short_headers))

    return summary
