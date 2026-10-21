"""XCausal — Robust treatment effect estimation via ensembling."""

from importlib.metadata import PackageNotFoundError, version as _version

from xcausal.aggregation import (
    AggregationStrategy,
    AgreementStrategy,
    BootstrapResult,
    CBA,
    EnsembleWeights,
    Mean,
    Median,
    PointwiseStrategy,
    TrimmedMean,
)
from xcausal.adapters.causalml import CausalMLAdapter
from xcausal.adapters.doubleml import DoubleMLAdapter
from xcausal.adapters.econml import EconMLAdapter
from xcausal.adapters.generic import GenericAdapter, GenericATEAdapter, GenericCATEAdapter
from xcausal.adapters.stochtree import StochtreeAdapter
from xcausal.ensemble import CausalEnsemble
from xcausal.estimators import (
    AteEstimate,
    CausalEstimate,
    CateEstimate,
    ComponentAteEstimate,
    ComponentCateEstimate,
    EnsembleEstimate,
)
from xcausal.outcome_type import infer_outcome_type
from xcausal._warnings import (
    BootstrapWarning,
    ComponentExclusionWarning,
    ComponentFailureWarning,
    ComponentWarning,
    XCausalWarning,
)

# Read from installed package metadata (pyproject.toml is the single source
# of truth) rather than hardcoding a second copy that can drift out of sync.
try:
    __version__ = _version("xcausal")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CausalEnsemble",
    # Estimator result types
    "ComponentAteEstimate",
    "AteEstimate",
    "ComponentCateEstimate",
    "CateEstimate",
    # Aggregation
    "AggregationStrategy",
    "PointwiseStrategy",
    "AgreementStrategy",
    "Median",
    "Mean",
    "TrimmedMean",
    "CBA",
    "EnsembleWeights",
    "BootstrapResult",
    # Library adapters
    "CausalMLAdapter",
    "DoubleMLAdapter",
    "EconMLAdapter",
    "StochtreeAdapter",
    # Generic adapters
    "GenericATEAdapter",
    "GenericAdapter",        # backward-compatible alias
    "GenericCATEAdapter",
    # Outcome-type detection
    "infer_outcome_type",
    # Warning classes
    "XCausalWarning",
    "ComponentWarning",
    "ComponentFailureWarning",
    "ComponentExclusionWarning",
    "BootstrapWarning",
    # Backward-compatible aliases
    "CausalEstimate",
    "EnsembleEstimate",
]
