"""Adapters for wrapping causal ML libraries."""

from xcausal._parallel import INNER_WORKER_ENV as _INNER_WORKER_ENV
from xcausal.adapters.base import CausalEstimator
from xcausal.adapters.causalml import CausalMLAdapter, _is_causalml
from xcausal.adapters.doubleml import DoubleMLAdapter
from xcausal.adapters.econml import EconMLAdapter
from xcausal.adapters.generic import GenericAdapter, GenericATEAdapter, GenericCATEAdapter
from xcausal.adapters.stochtree import StochtreeAdapter

#: Sentinel env var set inside every XCausal parallel worker. Custom
#: components (:class:`~xcausal.adapters.GenericCATEAdapter`, arbitrary
#: callables) can check this to cooperate with XCausal's own-worker
#: parallelism guard -- pin your model to serial when it is set, the same
#: way EconMLAdapter/CausalMLAdapter/... do internally. See "Extending
#: XCausal" in the project README.
INNER_WORKER_ENV = _INNER_WORKER_ENV

__all__ = [
    "CausalEstimator",
    "CausalMLAdapter",
    "DoubleMLAdapter",
    "EconMLAdapter",
    "GenericATEAdapter",
    "GenericAdapter",        # backward-compatible alias for GenericATEAdapter
    "GenericCATEAdapter",
    "StochtreeAdapter",
    "INNER_WORKER_ENV",
]
