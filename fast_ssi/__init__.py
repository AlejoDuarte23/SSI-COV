"""Public package exports for fast-ssi."""

from .SSI_posprocessor import cluster_data_by_frequency, plotStabDiag  # noqa: F401
from .SSICOV import SSICOV  # noqa: F401

try:
    from .SSICOV_GPU import SSICOV_GPU  # noqa: F401
except Exception:
    # Optional GPU dependency (cupy) is not always available.
    SSICOV_GPU = None  # type: ignore
