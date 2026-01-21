from pathlib import Path
from scipy.io import loadmat
from fast_ssi import SSICOV

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_ssicov_runs():
    mat = loadmat(DATA_DIR / "BridgeData.mat")
    t, rz, wn = mat["t"], mat["rz"], mat["wn"]

    fs = 15
    acc = rz.T
    Nmin = 7
    Nmax = 50
    Nc = acc.shape[1]
    Ts = 10

    ssi_constructor = SSICOV(acc, fs, Ts, Nc, Nmax, Nmin)
    fnS, zetaS, phiS, MACS, stability_status, fn2 = ssi_constructor.run()

    assert fnS.size > 0
    assert zetaS.size > 0
    assert phiS.size > 0
    assert MACS.size > 0
    assert len(stability_status) > 0
    assert any(getattr(v, "size", 0) > 0 for v in stability_status.values())
    assert len(fn2) > 0
    assert any(getattr(v, "size", 0) > 0 for v in fn2.values())
