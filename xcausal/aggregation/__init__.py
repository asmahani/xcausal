"""Aggregation strategies for CausalEnsemble."""

from xcausal.aggregation.base import AggregationStrategy, AgreementStrategy, PointwiseStrategy, SupervisedStrategy
from xcausal.aggregation.causal_stacking import CausalStacking
from xcausal.aggregation.cba import CBA
from xcausal.aggregation.nuisance import NuisanceEstimates, dr_pseudo_outcome, fit_nuisance, robinson_residuals
from xcausal.aggregation.pointwise import Mean, Median, TrimmedMean, _STRING_FACTORIES as _POINTWISE_FACTORIES
from xcausal.aggregation.q_aggregation import QAggregation
from xcausal.aggregation.r_stacking import RStacking
from xcausal.aggregation.select import Select
from xcausal.aggregation.splitting import CrossFitSplit, FoldSpec, TrainAvgSplit
from xcausal.aggregation.weights import BootstrapResult, EnsembleWeights

# All string aliases, including agreement strategies.
# Values are classes (not instances); CausalEnsemble.__init__ calls them as constructors.
_STRING_FACTORIES = {
    **_POINTWISE_FACTORIES,
    "cba": CBA,
}

__all__ = [
    "AggregationStrategy",
    "PointwiseStrategy",
    "AgreementStrategy",
    "SupervisedStrategy",
    "Median",
    "Mean",
    "TrimmedMean",
    "CBA",
    "EnsembleWeights",
    "BootstrapResult",
    "FoldSpec",
    "CrossFitSplit",
    "TrainAvgSplit",
    "NuisanceEstimates",
    "fit_nuisance",
    "dr_pseudo_outcome",
    "robinson_residuals",
    "CausalStacking",
    "QAggregation",
    "RStacking",
    "Select",
]
