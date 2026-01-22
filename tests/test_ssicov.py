from pathlib import Path

import pytest
from scipy.io import loadmat

from fast_ssi import SSICOV
from fast_ssi.SSICOV_RSVD import SSICOV as SSICOV_RSVD

DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def bridge_data() -> tuple[object, int]:
    mat = loadmat(DATA_DIR / "BridgeData.mat")
    _t, rz, _wn = mat["t"], mat["rz"], mat["wn"]

    fs = 15
    acc = rz.T

    return acc, fs


def _assert_outputs(
    fnS: object,
    zetaS: object,
    phiS: object,
    MACS: object,
    stability_status: object,
    fn2: object,
) -> None:
    assert getattr(fnS, "size", 0) > 0
    assert getattr(zetaS, "size", 0) > 0
    assert getattr(phiS, "size", 0) > 0
    assert getattr(MACS, "size", 0) > 0
    assert len(stability_status) > 0
    assert any(getattr(v, "size", 0) > 0 for v in stability_status.values())
    assert len(fn2) > 0
    assert any(getattr(v, "size", 0) > 0 for v in fn2.values())


def test_ssicov_runs(bridge_data: tuple[object, int]) -> None:
    acc, fs = bridge_data
    Nmin = 7
    Nmax = 50
    Nc = acc.shape[1]
    Ts = 10

    ssi_constructor = SSICOV(acc, fs, Ts, Nc, Nmax, Nmin)
    fnS, zetaS, phiS, MACS, stability_status, fn2 = ssi_constructor.run()

    _assert_outputs(fnS, zetaS, phiS, MACS, stability_status, fn2)


def test_ssicov_rsvd_runs(bridge_data: tuple[object, int]) -> None:
    acc, fs = bridge_data
    Nmin = 4
    Nmax = 15
    Nc = acc.shape[1]
    Ts = 2

    ssi_constructor = SSICOV_RSVD(acc, fs, Ts, Nc, Nmax, Nmin)
    fnS, zetaS, phiS, MACS, stability_status, fn2 = ssi_constructor.run()

    _assert_outputs(fnS, zetaS, phiS, MACS, stability_status, fn2)


def test_ssicov_gpu_runs(bridge_data: tuple[object, int]) -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device available.")
    except Exception:
        pytest.skip("CUDA runtime unavailable.")

    from fast_ssi.SSICOV_GPU import SSICOV as SSICOV_GPU

    acc, fs = bridge_data
    Nmin = 4
    Nmax = 15
    Nc = acc.shape[1]
    Ts = 2

    ssi_constructor = SSICOV_GPU(acc, fs, Ts, Nc, Nmax, Nmin)
    fnS, zetaS, phiS, MACS, stability_status, fn2 = ssi_constructor.run()

    _assert_outputs(fnS, zetaS, phiS, MACS, stability_status, fn2)
