"""Tests for ``default_methods()``.

The smoke test that actually fits every default component on Lalonde is
marked ``slow`` because the stochtree BCF sampler adds noticeable
wall-clock time. Run with ``pytest -m slow`` to opt in, or
``pytest -m ''`` to run everything.
"""

from __future__ import annotations

import pytest

from xcausal import CausalEnsemble
from xcausal.datasets import load_lalonde
from xcausal.defaults import default_methods, default_outcome_model


def test_default_library_has_nine_components_spanning_four_frameworks():
    """Enumeration check: no fitting, just construction and name verification."""
    methods = default_methods()
    assert len(methods) == 9

    ens = CausalEnsemble(methods=methods)
    names = set(ens.method_names)

    expected = {
        # DoubleML
        "DoubleMLIRM", "DoubleMLPLR",
        # EconML
        "CausalForestDML", "DRLearner",
        "TLearner", "XLearner",
        # CausalML
        "BaseRRegressor", "TMLELearner",
        # stochtree
        "BCF",
    }
    assert names == expected, f"missing={expected - names}, unexpected={names - expected}"


class TestDefaultOutcomeModel:
    def test_continuous_returns_regressor(self):
        model = default_outcome_model("continuous")
        from sklearn.base import is_regressor
        assert is_regressor(model)
        assert not hasattr(model, "predict_proba")

    def test_default_is_continuous(self):
        # No-arg call preserves pre-Phase-1.4 behavior.
        from sklearn.base import is_regressor
        assert is_regressor(default_outcome_model())

    def test_binary_returns_classifier(self):
        model = default_outcome_model("binary")
        from sklearn.base import is_classifier
        assert is_classifier(model)
        assert hasattr(model, "predict_proba")

    def test_invalid_outcome_type_raises(self):
        with pytest.raises(ValueError, match="outcome_type"):
            default_outcome_model("survival")


class TestDefaultMethodsOutcomeType:
    def test_continuous_explicit_matches_default(self):
        names_default = {m.name if hasattr(m, "name") else type(m).__name__
                         for m in default_methods()}
        names_explicit = {m.name if hasattr(m, "name") else type(m).__name__
                          for m in default_methods("continuous")}
        assert names_default == names_explicit

    def test_binary_pool_composition(self):
        methods = default_methods("binary")
        assert len(methods) == 7

        ens = CausalEnsemble(methods=methods)
        names = set(ens.method_names)
        expected = {
            "DoubleMLIRM",
            "CausalForestDML", "DRLearner",
            "BaseTClassifier", "BaseXClassifier",
            "BaseRClassifier",
            "TMLELearner",
        }
        assert names == expected

    def test_binary_components_declare_binary_capability(self):
        ens = CausalEnsemble(methods=default_methods("binary"))
        for m in ens._wrapped_methods:
            assert "binary" in m.supported_outcome_types, (
                f"{m.name} does not support binary outcomes"
            )


class TestRLearnerSerialCrossFit:
    """CausalML's BaseRLearner family defaults ``cv_n_jobs=-1`` (unlike
    EconML's ``n_jobs``, this knob is on by default), which would otherwise
    nest a joblib pool inside XCausal's own outer parallelism on every
    fit (issue #47). The shipped default pools pin it to 1 explicitly.
    """

    def test_continuous_pool_r_learner_pinned(self):
        methods = default_methods()
        r_learner = next(m for m in methods if type(m).__name__ == "BaseRRegressor")
        assert r_learner.cv_n_jobs == 1

    def test_binary_pool_r_classifier_pinned(self):
        methods = default_methods("binary")
        r_classifier = next(m for m in methods if type(m).__name__ == "BaseRClassifier")
        assert r_classifier.cv_n_jobs == 1

    def test_invalid_outcome_type_raises(self):
        with pytest.raises(ValueError, match="outcome_type"):
            default_methods("survival")


def _synthetic_outcome(outcome_type, n=220, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * X[:, 0]))).astype(int)
    if outcome_type == "binary":
        p = 1.0 / (1.0 + np.exp(-(X[:, 0] + 0.8 * T)))
        Y = rng.binomial(1, p).astype(float)
    else:
        Y = X[:, 0] + 0.7 * T + rng.normal(scale=0.2, size=n)
    return X, T, Y


def _fit_r_learner(learner, X, T, Y, random_state=0):
    """Fit a default-pool or user-supplied R-learner through the adapter."""
    from xcausal.adapters.causalml import CausalMLAdapter

    adapter = CausalMLAdapter(learner)
    adapter.fit(X, T, Y, random_state=random_state)
    return adapter._fitted_model


class TestRLearnerPropensityOverride:
    """CausalML >= 0.18 fits ``propensity_learner`` (uber/causalml#937).

    ``fit()`` assigns ``model_p`` from ``propensity_learner`` before
    training it. The default pools therefore do not keep a second
    propensity instance on ``model_p``: that instance was what CausalML
    0.16/0.17 actually trained, so a later change to ``propensity_learner``
    was ignored.
    """

    def test_default_pools_do_not_preset_model_p(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        r_learner = next(
            m for m in default_methods() if type(m).__name__ == "BaseRRegressor"
        )
        r_classifier = next(
            m for m in default_methods("binary") if type(m).__name__ == "BaseRClassifier"
        )
        assert not hasattr(r_learner, "model_p")
        assert not hasattr(r_classifier, "model_p")
        assert isinstance(r_learner.propensity_learner, HistGradientBoostingClassifier)
        assert isinstance(r_classifier.propensity_learner, HistGradientBoostingClassifier)

    def test_default_regressor_fits_its_propensity_learner(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        X, T, Y = load_lalonde()
        r_learner = next(
            m for m in default_methods() if type(m).__name__ == "BaseRRegressor"
        )
        fitted = _fit_r_learner(r_learner, X, T, Y, random_state=42)

        assert isinstance(fitted.propensity_learner, HistGradientBoostingClassifier)
        assert fitted.propensity_model
        for group, model in fitted.propensity_model.items():
            assert model is fitted.propensity_learner, (
                f"group {group}: fitted {type(model).__name__}, "
                f"not the configured propensity_learner"
            )

    def test_default_classifier_fits_its_propensity_learner(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        X, T, Y = _synthetic_outcome("binary")
        r_classifier = next(
            m for m in default_methods("binary") if type(m).__name__ == "BaseRClassifier"
        )
        fitted = _fit_r_learner(r_classifier, X, T, Y, random_state=42)

        assert isinstance(fitted.propensity_learner, HistGradientBoostingClassifier)
        assert fitted.propensity_model
        for model in fitted.propensity_model.values():
            assert model is fitted.propensity_learner

    def test_replacing_default_propensity_learner_is_honored(self):
        from sklearn.linear_model import LogisticRegression

        for outcome_type, class_name in (
            ("continuous", "BaseRRegressor"),
            ("binary", "BaseRClassifier"),
        ):
            X, T, Y = _synthetic_outcome(outcome_type, seed=1)
            learner = next(
                m for m in default_methods(outcome_type)
                if type(m).__name__ == class_name
            )
            learner.propensity_learner = LogisticRegression(max_iter=400)
            fitted = _fit_r_learner(learner, X, T, Y, random_state=1)

            assert isinstance(fitted.propensity_learner, LogisticRegression)
            assert fitted.model_p is fitted.propensity_learner
            for model in fitted.propensity_model.values():
                assert model is fitted.propensity_learner


@pytest.mark.slow
def test_binary_default_ensemble_fits_on_synthetic_dgp():
    """Every binary-pool component fits on a synthetic binary DGP and
    produces a finite ATE within a sensible range of the true RD."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-0.3 * X[:, 0]))
    T = rng.binomial(1, e).astype(int)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] - 1.0 + 0.7 * T)))
    Y = rng.binomial(1, p)

    ens = CausalEnsemble(methods=default_methods("binary"))
    ens.fit(X, T, Y, random_state=0)
    result = ens.ate()

    assert np.isfinite(result.ate)
    # All components produced a finite ATE in a plausible RD range.
    assert len(result.component_estimates) == 7
    for name, est in result.component_estimates.items():
        assert np.isfinite(est.ate), f"{name} produced non-finite ATE"
        assert -1.0 <= est.ate <= 1.0, (
            f"{name} ATE={est.ate} is outside the [-1, 1] risk-difference range"
        )


@pytest.mark.slow
def test_default_ensemble_fits_on_lalonde():
    """Every default component fits on Lalonde and produces a finite ATE."""
    X, T, Y = load_lalonde()

    ens = CausalEnsemble()
    ens.fit(X, T, Y, random_state=42)

    result = ens.ate()

    # Ensemble ATE is finite
    assert result.ate == result.ate  # not NaN
    assert abs(result.ate) < 1e6  # not inf-ish

    # All nine components produced a finite ATE
    assert len(result.component_estimates) == 9
    for name, est in result.component_estimates.items():
        assert est.ate == est.ate, f"{name} returned NaN"
