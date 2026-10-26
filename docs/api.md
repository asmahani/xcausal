# API reference

One page per public class and function, generated from docstrings via
`sphinx.ext.autosummary`. Each `##` section below is a real import
path (`xcausal`, `xcausal.aggregation`, ...); each item is
listed once, under the module it's *defined* in.

Many aggregation-strategy and adapter classes (e.g. `Median`,
`CausalMLAdapter`) are re-exported at the top level for convenience --
`from xcausal import Median` works even though it's documented
under `xcausal.aggregation` below. See the [Mixed-framework method
list](index.md#mixed-framework-method-list) or `xcausal.__all__`
for the exact top-level surface.

## `xcausal`

Top-level namespace: the ensemble class, its estimate/result types,
and outcome-type detection.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.CausalEnsemble
   xcausal.AteEstimate
   xcausal.CateEstimate
   xcausal.ComponentAteEstimate
   xcausal.ComponentCateEstimate
   xcausal.infer_outcome_type
```

### Warnings

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.XCausalWarning
   xcausal.ComponentWarning
   xcausal.ComponentFailureWarning
   xcausal.ComponentExclusionWarning
   xcausal.BootstrapWarning
   xcausal.RiskDifferenceWarning
```

## `xcausal.aggregation`

Aggregation strategies (combine component ATE/CATE predictions into an
ensemble estimate) and the result types they produce.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.aggregation.AggregationStrategy
   xcausal.aggregation.PointwiseStrategy
   xcausal.aggregation.AgreementStrategy
   xcausal.aggregation.SupervisedStrategy
   xcausal.aggregation.Mean
   xcausal.aggregation.Median
   xcausal.aggregation.TrimmedMean
   xcausal.aggregation.CBA
   xcausal.aggregation.CausalStacking
   xcausal.aggregation.QAggregation
   xcausal.aggregation.RStacking
   xcausal.aggregation.Select
   xcausal.aggregation.EnsembleWeights
   xcausal.aggregation.BootstrapResult
```

### Cross-fitting

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.aggregation.CrossFitSplit
   xcausal.aggregation.TrainAvgSplit
   xcausal.aggregation.FoldSpec
```

### Nuisance estimation

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.aggregation.NuisanceEstimates
   xcausal.aggregation.fit_nuisance
   xcausal.aggregation.dr_pseudo_outcome
   xcausal.aggregation.robinson_residuals
```

## `xcausal.adapters`

Wrappers presenting each supported library's estimators (EconML,
CausalML, DoubleML, stochtree) through one common interface, plus
generic adapters for wrapping arbitrary user-supplied estimators.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.adapters.CausalEstimator
   xcausal.adapters.EconMLAdapter
   xcausal.adapters.CausalMLAdapter
   xcausal.adapters.DoubleMLAdapter
   xcausal.adapters.StochtreeAdapter
   xcausal.adapters.GenericATEAdapter
   xcausal.adapters.GenericCATEAdapter
   xcausal.adapters.INNER_WORKER_ENV
```

## `xcausal.plots`

Requires the `plots` extra (`pip install "xcausal[plots]"`). Each
function is also available as a thin method on the class it plots --
see the corresponding class above (e.g. `BootstrapResult.forest()`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.plots.forest
   xcausal.plots.weights
   xcausal.plots.cate_profile
   xcausal.plots.disagreement
```

## `xcausal.datasets`

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   xcausal.datasets.load_lalonde
```
