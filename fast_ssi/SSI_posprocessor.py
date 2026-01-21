import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from sklearn.cluster import KMeans
from tabulate import tabulate


# CPSD function
def CPSD(
    Acc: NDArray, fs: int, Nc: int, fo: float, fi: float
) -> tuple[NDArray, NDArray, int]:
    def nextpow2(Acc):
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


def plotStabDiag(fn, Acc, fs, stability_status, Nmin, Nmax, Nc, fo, fi):
    freq_id, TSxx, N = CPSD(Acc, fs, Nc, fo, fi)

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
    for jj in range(5):
        x = []
        y = []
        for ii in range(len(fn)):
            try:
                ind = np.where(stability_status[ii] == jj)
                x.extend(fn[ii][ind])
                y.extend([Npoles[ii]] * len(fn[ii][ind]))
            except Exception:
                print("Error !")
        (h,) = ax1.plot(x, y, markers[jj], label=labels[jj])
        handles.append(h)
    ax1.set_ylabel("number of poles")
    ax1.set_xlabel("f (Hz)")
    ax1.set_ylim(0, Nmax + 2)
    ax2 = ax1.twinx()

    max_fn2 = np.max([np.max(v) for v in fn.values()])

    # Plot CPSD
    color = "blue"
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel("Power Spectral Density", color=color)
    ax2.plot(freq_id, 10 * np.log10(TSxx / N), color, label="Trace")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_xlim(0, max_fn2 * 1.1)
    # ax2.set_yscale('log')
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
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
