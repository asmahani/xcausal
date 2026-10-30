"""Binary ensemble bounds: diagnostics leave estimates and inference unchanged."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from xcausal import (
    BootstrapWarning,
    CausalEnsemble,
    GenericATEAdapter,
    GenericCATEAdapter,
    RiskDifferenceWarning,
)
from xcausal.aggregation import (
    CBA,
    CausalStacking,
    CrossFitSplit,
    Mean,
    Median,
    NuisanceEstimates,
    PointwiseStrategy,
    QAggregation,
    RStacking,
    Select,
    SupervisedStrategy,
    TrimmedMean,
)


STRATEGIES = [Median, Mean, TrimmedMean, CBA, CausalStacking, QAggregation, RStacking, Select]


def _data():
    # Balanced cells with Y = T * X. The oracle treatment effect is X,
    # allowing real supervised fits with a known prediction on a new grid.
    X = np.tile([0.0, 0.0, 1.0, 1.0], 20).reshape(-1, 1)
    T = np.tile([0, 1, 0, 1], 20)
    Y = T * X[:, 0]
    return X, T, Y


def _oracle_nuisance(X, T, Y, fold_spec, **kwargs):
    return NuisanceEstimates(
        e_hat=np.full(len(T), 0.5), mu0_hat=np.zeros(len(T)), mu1_hat=X[:, 0],
    )


def _strategy(cls):
    if issubclass(cls, SupervisedStrategy):
        return cls(split=CrossFitSplit(n_folds=2), fit_nuisance_fn=_oracle_nuisance)
    return cls()


def _adapter(offset=0.0):
    return GenericCATEAdapter(
        lambda X, T, Y, **kwargs: None,
        lambda state, X: X[:, 0] + offset,
        name=f"offset_{offset}", supported_outcome_types=("continuous", "binary"),
    )


def _ensemble(cls=Median, *, outcome_type="auto", continuous=False, methods=None):
    X, T, Y = _data()
    if continuous:
        Y = Y + np.arange(len(Y)) / 100
    if methods is None:
        offsets = [-1.0, 0.0, 1.0] if cls is TrimmedMean else [0.0]
        methods = [_adapter(offset) for offset in offsets]
    return CausalEnsemble(
        methods=methods,
        aggregation=_strategy(cls), outcome_type=outcome_type,
    ).fit(X, T, Y, random_state=0)


def _assert_warnings(caught, kinds, line):
    diagnostics = [w for w in caught if w.category is RiskDifferenceWarning]
    assert len(diagnostics) == len(kinds)
    assert sorted(str(w.message).split()[1] for w in diagnostics) == sorted(kinds)
    for warning in diagnostics:
        assert warning.filename == __file__
        assert warning.lineno == line
    return diagnostics


@pytest.mark.parametrize("cls", STRATEGIES)
def test_cate_all_strategies_count_both_tails_without_clipping(cls):
    ens = _ensemble(cls)
    X = np.array([[-2.0], [-0.5], [0.0], [0.5], [2.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        result = ens.cate(X)
    diagnostic, = _assert_warnings(caught, ["CATE"], line)
    assert "2 of 5 entries outside [-1, 1]" in str(diagnostic.message)
    np.testing.assert_allclose(result.cate, X[:, 0], atol=1e-14)
    np.testing.assert_array_equal(result.component_estimates["offset_0.0"].cate, X[:, 0])


@pytest.mark.parametrize("cls", STRATEGIES)
@pytest.mark.parametrize("sign", [-1, 1])
def test_ate_all_strategies_preserves_nested_diagnostics_and_weights(cls, sign):
    ens = _ensemble(cls)
    weights = ens.aggregation.ensemble_weights
    before = None if weights is None else (weights.weights.copy(), weights.intercept)
    X = sign * np.array([[1.5], [2.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        result = ens.ate(X)
    kinds = ["ATE"] if issubclass(cls, PointwiseStrategy) else ["CATE", "ATE"]
    _assert_warnings(caught, kinds, line)
    assert result.ate == pytest.approx(sign * 1.75)
    if before is not None:
        np.testing.assert_array_equal(weights.weights, before[0])
        assert weights.intercept == before[1]


def test_r_stacking_can_warn_even_with_bounded_components():
    ens = _ensemble(RStacking)
    weights = ens.aggregation.ensemble_weights
    # A feasible R-Stacking fit from the issue's example. The component
    # stays at 0.4, while the unconstrained intercept and conic weight
    # produce 1.3. Prediction must not change either fitted parameter.
    weights.weights[:] = 2.0
    weights.intercept = 0.5
    with pytest.warns(RiskDifferenceWarning, match="Ensemble CATE"):
        result = ens.cate(np.array([[0.4]]))
    assert result.cate[0] == pytest.approx(1.3)
    assert result.component_estimates["offset_0.0"].cate[0] == 0.4
    np.testing.assert_array_equal(weights.weights, [2.0])
    assert weights.intercept == 0.5


@pytest.mark.parametrize("cls", [Median, CBA, RStacking])
def test_out_of_range_cate_with_in_range_mean(cls):
    ens = _ensemble(cls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        result = ens.ate(np.array([[-2.0], [2.0]]))
    _assert_warnings(caught, [] if cls is Median else ["CATE"], line)
    assert result.ate == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("outcome_type", ["auto", "binary"])
@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_inclusive_boundaries_and_repeated_calls(outcome_type, value):
    ens = _ensemble(outcome_type=outcome_type)
    X = np.array([[value]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ens.ate(X).ate == value
        np.testing.assert_array_equal(ens.cate(X).cate, [value])
        assert not caught
        outside = np.array([[np.nextafter(1.0, np.inf)]])
        for _ in range(2):
            line = inspect.currentframe().f_lineno + 1
            ens.cate(outside)
        _assert_warnings(caught, ["CATE", "CATE"], line)


@pytest.mark.parametrize("outcome_type,continuous", [("continuous", False), ("auto", True)])
def test_continuous_outcome_or_override_is_silent(outcome_type, continuous):
    ens = _ensemble(outcome_type=outcome_type, continuous=continuous)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ens.ate(np.array([[2.0]])).ate == 2.0
        np.testing.assert_array_equal(ens.cate(np.array([[-2.0]])).cate, [-2.0])
        ens.bootstrap(np.array([[2.0]]), n_boot=3, random_state=0)
    assert not caught


def test_refit_updates_outcome_gate():
    ens = _ensemble()
    X, T, Y = _data()
    ens.fit(X, T, Y + np.arange(len(Y)) / 100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ens.cate(np.array([[2.0]]))
    assert not caught
    ens.fit(X, T, Y)
    with pytest.warns(RiskDifferenceWarning):
        ens.cate(np.array([[2.0]]))


def test_only_ensemble_values_are_checked_and_overrides_are_checked():
    ens = _ensemble(methods=[_adapter(-0.5), _adapter(0.0), _adapter(6.0)])
    X = np.zeros((2, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ens.ate(X).ate == 0.0
        np.testing.assert_array_equal(ens.cate(X).cate, [0.0, 0.0])
    assert not caught
    for method in (ens.ate, ens.cate):
        with pytest.warns(RiskDifferenceWarning):
            method(X, aggregation="mean")


@pytest.mark.parametrize("override", [Median, CBA, RStacking])
def test_supervised_fit_aggregation_overrides(override):
    ens = _ensemble(RStacking)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        result = ens.ate(np.array([[2.0]]), aggregation=_strategy(override))
    _assert_warnings(caught, ["ATE"] if override is Median else ["CATE", "ATE"], line)
    assert result.ate == pytest.approx(2.0)


@pytest.mark.parametrize("cls", [Median, CBA, RStacking])
@pytest.mark.parametrize("method", ["nonparametric", "subsample"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_bootstrap_warns_once_per_estimate_not_per_replicate(cls, method, n_jobs):
    ens = _ensemble(cls)
    X = np.array([[1.5], [2.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        result = ens.bootstrap(X, n_boot=5, method=method, n_jobs=n_jobs, random_state=0)
    _assert_warnings(caught, ["ATE", "CATE"], line)
    assert result.n_failed == 0
    assert result.ate == pytest.approx(1.75)
    np.testing.assert_allclose(result.cate, X[:, 0])
    np.testing.assert_allclose(result.boot_ates, np.full(5, 1.75))
    np.testing.assert_allclose(result.boot_cates, np.tile(X[:, 0], (5, 1)))
    assert result.ate_ci_lower == pytest.approx(1.75)
    assert result.ate_ci_upper == pytest.approx(1.75)
    np.testing.assert_allclose(result.cate_ci_lower, X[:, 0])
    np.testing.assert_allclose(result.cate_ci_upper, X[:, 0])


@pytest.mark.parametrize("cls", [Median, CBA, RStacking])
def test_replicate_computation_never_emits_bound_warning(cls):
    ens = _ensemble(cls)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RiskDifferenceWarning)
        result = ens._single_bootstrap(np.array([[2.0]]), seed=0)
    assert result is not None
    assert result[0] == pytest.approx(2.0)
    np.testing.assert_allclose(result[2], [2.0])


@pytest.mark.parametrize("cls", [Median, CBA])
def test_only_replicates_and_intervals_outside_bounds_are_silent(cls):
    adapter = GenericCATEAdapter(
        lambda X, T, Y, **kwargs: len(Y),
        lambda n, X: np.full(len(X), 0.0 if n == 80 else 2.0),
        supported_outcome_types=("binary",),
    )
    ens = _ensemble(cls, methods=[adapter])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RiskDifferenceWarning)
        warnings.filterwarnings("ignore", category=BootstrapWarning)
        result = ens.bootstrap(n_boot=5, method="subsample", random_state=0)
    assert result.ate == 0.0
    assert result.n_failed == 0
    np.testing.assert_array_equal(result.boot_ates, np.full(5, 2.0))
    assert result.ate_ci_lower > 1.0


def test_ate_only_pool_bootstrap_checks_only_ate():
    ens = _ensemble(methods=[GenericATEAdapter(
        lambda X, T, Y: -1.5, supported_outcome_types=("binary",),
    )])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        result = ens.bootstrap(n_boot=3, random_state=0)
    _assert_warnings(caught, ["ATE"], line)
    assert result.ate == -1.5
    assert result.cate is None


@pytest.mark.parametrize("n_boot", [0, 3])
def test_estimate_wrapper_warning_points_to_user(n_boot):
    ens = CausalEnsemble(methods=[_adapter(2.0)], aggregation="cba")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        ens.estimate(*_data(), n_boot=n_boot, random_state=0)
    _assert_warnings(caught, ["ATE", "CATE"], line)


def test_warning_as_error_does_not_leave_suppression_state():
    ens = _ensemble(CBA)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RiskDifferenceWarning)
        with pytest.raises(RiskDifferenceWarning):
            ens.bootstrap(np.array([[2.0]]), n_boot=3)
    with pytest.warns(RiskDifferenceWarning):
        ens.cate(np.array([[2.0]]))
