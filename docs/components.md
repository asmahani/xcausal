# Default components and model configuration

`CausalEnsemble()` selects its component pool at `fit()` time from the resolved
outcome type. Continuous outcomes use regression models; binary outcomes target
the risk difference and use probabilities for outcome prediction. The tables
below ([generated source](https://github.com/asmahani/xcausal/blob/main/docs/_generated/components.md))
are generated from `xcausal.defaults.default_methods()` in the documented
dependency environment. They describe the shipped pool before fit-time
exclusions or failures. Inspect `ens.summary()` for the configured components
and `ens.ate().summary()` for the components that successfully returned ATE
estimates.

The configuration has two levels:

- **Member models** produce each component's treatment-effect estimate. Configure
  these through the estimators passed to `CausalEnsemble(methods=[...])`.
- **Aggregation nuisance models** produce the causal loss used to learn ensemble
  weights. A supervised strategy's `outcome_model`, `propensity_model`,
  `propensity_trim`, and `split` configure this separate calculation. They do
  not replace any member's models or internal folds.

## Construction notes

XCausal's explicit recipes live in
[`xcausal/defaults.py`](https://github.com/asmahani/xcausal/blob/main/xcausal/defaults.py).
The generated model recipes list all constructor parameters, including HGB's
early-stopping validation fraction and inherited tree and regularization
settings. XCausal adds no shared scaling, imputation, encoding, or polynomial
feature pipeline. Supply numeric covariates accepted by each backend, or put
preprocessing in the individual model's sklearn `Pipeline`. Learned transforms
should be fitted within each training fold.

### Doubly robust learner

EconML `DRLearner` receives HGB outcome and propensity models explicitly. Its
outcome regression includes treatment as a feature. Its **effect model is an
unregularized `StatsModelsLinearRegression`**, inherited from EconML, in both
outcome pools. With the inherited `featurizer=None`, this final regression uses
the original covariates, with an intercept and no added scaler or feature
expansion. Binary outcome prediction uses a classifier, but the final effect
stage remains a regression on the continuous risk-difference pseudo-outcome.

The member's `cv` and `min_propensity` are inherited from EconML; their resolved
values appear below. The propensity floor protects the DR pseudo-outcome's
denominators. It is separate from aggregation's `propensity_trim` and does not
bound the estimated treatment effect. XCausal does not harmonize member fold
counts or clipping thresholds across different estimation methods.

See the [DRLearner contract](https://www.pywhy.org/EconML/_autosummary/econml.dr.DRLearner.html)
and [versioned DR implementation](https://github.com/py-why/EconML/blob/v0.17.0/econml/dr/_drlearner.py).

### X-learners

For continuous outcomes, EconML `XLearner` receives an HGB outcome regressor and
HGB propensity classifier. XCausal leaves **`cate_models=None`**: EconML clones
the outcome models to construct the effect regressions and fits those fresh
clones to imputed treatment effects. The family is inherited; fitted outcome
models are not reused as fitted effect models. There is no member-level
cross-fitting or additional propensity clipping in this default construction.

For binary outcomes, CausalML `BaseXClassifier` instead receives explicit HGB
outcome classifiers and HGB **effect regressors**. The explicit `CausalMLAdapter`
fits an HGB propensity classifier on the member's training sample and supplies
its probabilities to both fitting and prediction. This bypasses CausalML's
internal propensity-model default. The adapter adds no clipping to those
probabilities. Both X constructions fit outcome and effect models separately
within the treatment arms.

See the [XLearner contract](https://www.pywhy.org/EconML/_autosummary/econml.metalearners.XLearner.html),
[EconML implementation](https://github.com/py-why/EconML/blob/v0.17.0/econml/metalearners/_metalearners.py),
and [CausalML X implementation](https://github.com/uber/causalml/blob/v0.18.0/causalml/inference/meta/xlearner.py).

### R-learners

`BaseRRegressor` and `BaseRClassifier` each receive separately constructed
`outcome_learner`, `propensity_learner`, and `effect_learner` objects. Matching
HGB families do not imply a shared fitted model. The outcome learner estimates
the prognostic mean given covariates, without treatment as a feature; the
effect learner fits the weighted R-loss. It must support `sample_weight`.
For binary outcomes the outcome model is a classifier and the effect model
remains a regressor.

The inherited `n_fold` controls out-of-fold outcome prediction. The HGB
propensity model is fitted on the member's full training sample, without a
separate propensity cross-fit or an added clipping step. CausalML's default
ElasticNet propensity wrapper's clipping settings do not apply to the supplied
HGB classifier. XCausal explicitly sets `cv_n_jobs=1` on its default R members;
this is a worker setting, independent of the internal fold count.

CausalML 0.18.0 resolves the fitted propensity from `propensity_learner` before
fitting, so this is the customization slot. See the
[versioned R implementation](https://github.com/uber/causalml/blob/v0.18.0/causalml/inference/meta/rlearner.py)
and [propensity helper](https://github.com/uber/causalml/blob/v0.18.0/causalml/propensity.py).

### T-learners

EconML `TLearner` fits separate HGB regressions for treated and control outcomes.
Binary outcomes use CausalML `BaseTClassifier` with separate HGB classifiers
and probability predictions. **The difference of outcome predictions defines
the effect**: there is no independent effect-model or propensity-model slot.
Neither default T construction adds internal cross-fitting or clipping.

See the [EconML meta-learner implementation](https://github.com/py-why/EconML/blob/v0.17.0/econml/metalearners/_metalearners.py)
and [CausalML T implementation](https://github.com/uber/causalml/blob/v0.18.0/causalml/inference/meta/tlearner.py).

### Native forest architectures

`CausalForestDML` uses HGB outcome and treatment nuisance models, with the
inherited internal `cv`. Its effect stage is EconML's native honest causal
forest. XCausal sets `n_estimators` and `n_jobs` explicitly; forest depth,
leaf size, honesty, sampling, and featurization settings otherwise come from
EconML, as shown below. There is no external effect-regressor slot analogous
to DR's `model_final`. There is no shared propensity clipping policy: EconML's
discrete-treatment doubly robust training-ATE calculation internally floors
the treatment variance at `1e-2`, which is distinct from clipping a propensity
for all forest predictions. See the
[forest implementation](https://github.com/py-why/EconML/blob/v0.17.0/econml/dml/causal_forest.py).

`StochtreeAdapter` uses BCF's native prognostic and treatment-effect forests.
Configure these through `prognostic_forest_params` and
`treatment_effect_forest_params`, and sampler behavior through `general_params`
and the adapter's iteration arguments. Empty/`None` forest dictionaries
delegate to stochtree. In the documented stochtree version those defaults
include 250 prognostic trees and 100 treatment-effect trees; outcome
standardization is enabled, adaptive treatment coding is disabled, and the
propensity is included as a prognostic covariate. These are resolved during
`BCFModel.sample()`, so they are source-reviewed rather than constructor
introspection results.

The adapter explicitly cross-fits its HGB propensity model using stratified
folds and clips computed scores to the interval defined by
`propensity_clip_eps`. Predictions on new data average the fitted fold models
and apply the same clipping. BCF performs its own covariate preprocessing and
outcome standardization; XCausal adds no scaler before it. See the
[versioned BCF implementation](https://github.com/StochasticTree/stochtree/blob/v0.4.5/stochtree/bcf.py)
and {py:class}`~xcausal.adapters.StochtreeAdapter`.

### ATE-only constructions

`DoubleMLIRM` uses outcome models by treatment arm (`ml_g`) and a propensity
classifier (`ml_m`). `DoubleMLPLR` uses a prognostic regression (`ml_l`) and a
treatment model (`ml_m`). Their effects are defined by their estimating
scores, without an independently fitted CATE regression in XCausal's adapter.
Internal folds, score choices, and IRM propensity truncation are inherited
from the upstream constructors and resolved below. PLR adds no propensity
clipping. See the versioned
[IRM](https://github.com/DoubleML/doubleml-for-py/blob/0.11.4/doubleml/irm/irm.py)
and [PLR](https://github.com/DoubleML/doubleml-for-py/blob/0.11.4/doubleml/plm/plr.py)
implementations.

`TMLELearner` uses an HGB regressor for outcome prediction in both pools, with
treatment appended to the covariates. The adapter fits its explicit HGB
propensity on the member's full training sample, without clipping. The
inherited `cv=None` means outcome predictions are also fitted in sample;
HGB's early-stopping validation split is not causal cross-fitting. Upstream
TMLE min-max scales the outcome and clips initial scaled outcome predictions
at `1e-4` before a logistic targeting update, then restores the outcome scale.
This clipping concerns outcome predictions, not propensity scores or final
effects. The adapter computes and caches the ATE during `fit()`. See the
[versioned TMLE implementation](https://github.com/uber/causalml/blob/v0.18.0/causalml/inference/meta/tmle.py).

## Configure effect models independently

This continuous-outcome example changes effect stages while leaving the
outcome and propensity recipes at XCausal's defaults. These are user choices,
not alternative shipped defaults.

```{testcode}
from causalml.inference.meta import BaseRRegressor
from econml.dr import DRLearner
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xcausal import CausalEnsemble
from xcausal.datasets import load_lalonde
from xcausal.defaults import default_outcome_model, default_propensity_model

dr = DRLearner(
    model_regression=default_outcome_model(),
    model_propensity=default_propensity_model(),
    model_final=Ridge(alpha=1.0),
)
x = XLearner(
    models=default_outcome_model(),
    propensity_model=default_propensity_model(),
    cate_models=RandomForestRegressor(n_estimators=20, n_jobs=1),
)
r = BaseRRegressor(
    outcome_learner=default_outcome_model(),
    propensity_learner=default_propensity_model(),
    effect_learner=RandomForestRegressor(n_estimators=20, n_jobs=1),
)
r.cv_n_jobs = 1

ens = CausalEnsemble(methods=[dr, x, r], aggregation="median")
X, T, Y = load_lalonde()
ens.fit(X, T, Y, random_state=42)
assert set(ens.ate().component_estimates) == {"DRLearner", "XLearner", "BaseRRegressor"}
```

For binary outcomes, use `default_outcome_model("binary")` in outcome slots,
`DRLearner(discrete_outcome=True, ...)`, and CausalML's `BaseXClassifier` or
`BaseRClassifier` with a regressor in the effect slot. An explicit
`CausalMLAdapter(BaseXClassifier(...), propensity_model=...)` supplies a
custom propensity. EconML T/X learners do not provide the binary probability
path used by the default binary pool.

An explicit EconML `cate_models` can be a single estimator or a list/tuple of
estimators, including sklearn pipelines. The adapter propagates the fit seed
into nested estimators and enforces serial inner workers when running inside
an XCausal worker; outside one, user-specified parallelism is preserved.

## Maintaining this reference

The roster, counts, capabilities, constructor settings, and model recipes in
the following tables are generated. After changing defaults, adapters, or
dependency ranges, run from the repository root:

```bash
python -m pip install -e '.[all,docs]' -c docs/constraints.txt
python scripts/generate_component_docs.py
python scripts/generate_component_docs.py --check
```

Review the generated diff with the code change. CI runs `--check` and fails on
stale output or an unmapped component. Changing the documentation baseline in
`docs/constraints.txt` also requires reviewing the versioned sources above and
the fit-time explanations, particularly native BCF defaults. Tables describe
resolved values; construction notes explain which settings XCausal supplies
and which the upstream libraries supply. The customization example is executed
by the existing Sphinx doctest job.

The manuscript's replication environment and reported results are historical
records. Its CausalML 0.17.0 pin predates the current R-propensity correction;
updating live documentation does not upgrade that environment or relabel its
results. A dependency change affecting those fits requires a separate decision
about regeneration.

```{include} _generated/components.md
```
