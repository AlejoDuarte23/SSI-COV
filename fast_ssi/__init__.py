"""Public package exports for fast-ssi."""

from .SSICOV import SSICOV  # noqa: F401
from .SSI_posprocessor import plotStabDiag, cluster_data_by_frequency  # noqa: F401

try:
    from .SSICOV_GPU import SSICOV_GPU  # noqa: F401
except Exception:
    # Optional GPU dependency (cupy) is not always available.
    SSICOV_GPU = None  # type: ignore
