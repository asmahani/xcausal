"""Warning classes for xcausal-specific concerns.

Hierarchy::

    Warning
    └── XCausalWarning            (umbrella for all xcausal warnings)
        ├── ComponentWarning         (operating-pool concerns)
        │   ├── ComponentFailureWarning   (component fit/predict failures)
        │   └── ComponentExclusionWarning (configuration mismatches)
        ├── BootstrapWarning         (bootstrap inference reliability)
        └── RiskDifferenceWarning    (binary ensemble estimates outside [-1, 1])

The classes are re-exported from :mod:`xcausal`; import as
``from xcausal import XCausalWarning, ComponentWarning,
ComponentFailureWarning, ComponentExclusionWarning, BootstrapWarning,
RiskDifferenceWarning``.

The hierarchy lets users filter at the appropriate granularity::

    import warnings
    import xcausal as mc

    # Catch any xcausal-emitted warning (most strict)
    warnings.filterwarnings("error", category=mc.XCausalWarning)

    # Catch only operating-pool concerns (failure or exclusion)
    warnings.filterwarnings("error", category=mc.ComponentWarning)

    # Catch only fit/predict-time runtime failures
    warnings.filterwarnings("error", category=mc.ComponentFailureWarning)

    # Catch only configuration mismatches (outcome-type, ATE-only)
    warnings.filterwarnings("error", category=mc.ComponentExclusionWarning)

    # Catch only bootstrap inference reliability concerns
    warnings.filterwarnings("error", category=mc.BootstrapWarning)

    # Catch only binary ensemble point estimates outside [-1, 1]
    warnings.filterwarnings("error", category=mc.RiskDifferenceWarning)

Classes inherit from :class:`Warning` directly (not :class:`RuntimeWarning`)
so that xcausal-scoped policies are independent of broad
:class:`RuntimeWarning` behaviour applied by the user or other libraries
(numpy, scipy, sklearn).
"""

from __future__ import annotations


class XCausalWarning(Warning):
    """Umbrella class for all xcausal-emitted warnings.

    Currently subclassed into :class:`ComponentWarning` (operating-pool
    concerns: fit failures and configuration exclusions),
    :class:`BootstrapWarning` (bootstrap inference reliability concerns),
    and :class:`RiskDifferenceWarning` (binary ensemble point estimates
    outside the risk-difference bounds).
    Filtering at this level catches any xcausal-emitted warning
    regardless of subcategory.
    """


class ComponentWarning(XCausalWarning):
    """Base class for warnings indicating an operating-ensemble pool
    reduction.

    Subclassed into :class:`ComponentFailureWarning` (runtime fit/predict
    failures) and :class:`ComponentExclusionWarning` (configuration
    mismatches). Filtering at this level catches any partial-pool surprise
    regardless of cause, but does NOT catch :class:`BootstrapWarning`
    or :class:`RiskDifferenceWarning`.
    """


class ComponentFailureWarning(ComponentWarning):
    """A component raised an exception during fit, cross-fitting,
    prediction, or as part of nuisance/weight-fitting machinery, and
    was silently excluded from the operating ensemble.

    Data-dependent: the same configuration on different data may or may
    not trigger this warning.
    """


class ComponentExclusionWarning(ComponentWarning):
    """A component was excluded by configuration mismatch.

    Two cases:

    - The component's declared ``supported_outcome_types`` did not
      include the detected outcome type at fit time.
    - The component is ATE-only (``supports_cate=False``) and the
      active aggregation strategy requires CATE-capable adapters
      (i.e., :class:`SupervisedStrategy`).

    Deterministic given the user's configuration and the detected
    outcome type.
    """


class BootstrapWarning(XCausalWarning):
    """A bootstrap inference call hit a reliability concern.

    Currently emitted when the bootstrap replicate failure rate exceeds
    10% (CIs may be unreliable) or when all replicates failed (CIs are
    not computed). A bootstrap-replicate failure can arise from
    component issues *within* a replicate, but also from global causes
    (numerical issues on the resample, weight-optimisation failure,
    etc.), so this category is distinct from :class:`ComponentWarning`
    — it is about *inference reliability*, not pool composition.
    """


class RiskDifferenceWarning(XCausalWarning):
    """A binary-outcome ensemble point estimate is outside [-1, 1].

    Applies to every aggregation strategy when the fitted outcome type
    resolves to ``"binary"``. Checks ensemble ATE/CATE estimates only;
    component estimates, confidence intervals, and bootstrap replicates
    are not checked. Estimates are returned unchanged, without clipping.
    Absence of this warning does not imply a bounded estimator.

    ``cate()`` emits at most one warning, counting out-of-range entries.
    ``ate()`` also checks the ensemble CATE for agreement and supervised
    strategies, so it can emit a CATE warning and an ATE warning.
    ``bootstrap()`` checks the original-fit estimates at most once each
    for ATE and CATE, regardless of the number of replicates.
    """
