"""Tests for the subsample bootstrap variant (``method='subsample'``).

The subsample bootstrap draws m-out-of-n without replacement (T-stratified),
eliminating duplicate units so any downstream cross-fit stays honest. CIs
use subsampling with finite-population correction.
"""

from __future__ import annotations

import warnings
from itertools import combinations, product

import numpy as np
import pytest

from xcausal import (
    BootstrapResult,
    BootstrapWarning,
    CausalEnsemble,
    ComponentAteEstimate,
    GenericCATEAdapter,
)
from xcausal.aggregation.weights import _bootstrap_scale
from xcausal.ensemble import _resolve_subsample_size, _stratified_subsample


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestResolveSubsampleSize:
    def test_float_fraction(self):
        assert _resolve_subsample_size(100, 0.5) == 50
        assert _resolve_subsample_size(200, 0.3) == 60

    def test_int_passthrough(self):
        assert _resolve_subsample_size(100, 30) == 30

    def test_float_zero_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            _resolve_subsample_size(100, 0.0)

    def test_float_one_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            _resolve_subsample_size(100, 1.0)

    def test_int_at_n_raises(self):
        with pytest.raises(ValueError, match="m=100"):
            _resolve_subsample_size(100, 100)

    def test_int_above_n_raises(self):
        with pytest.raises(ValueError):
            _resolve_subsample_size(100, 150)

    def test_int_zero_raises(self):
        with pytest.raises(ValueError):
            _resolve_subsample_size(100, 0)

    def test_bool_raises(self):
        with pytest.raises(TypeError, match="bool"):
            _resolve_subsample_size(100, True)

    def test_string_raises(self):
        with pytest.raises(TypeError):
            _resolve_subsample_size(100, "half")


class TestStratifiedSubsample:
    def test_no_duplicates(self):
        rng = np.random.default_rng(0)
        T = np.concatenate([np.zeros(100), np.ones(100)])
        idx = _stratified_subsample(rng, T, 50)
        assert len(idx) == 50
        assert len(np.unique(idx)) == 50

    def test_indices_in_range(self):
        rng = np.random.default_rng(0)
        n = 200
        T = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        idx = _stratified_subsample(rng, T, 80)
        assert idx.min() >= 0
        assert idx.max() < n

    def test_treatment_ratio_preserved(self):
        rng = np.random.default_rng(0)
        n_treated = 30
        n_control = 170
        T = np.concatenate([np.zeros(n_control), np.ones(n_treated)])
        m = 100
        idx = _stratified_subsample(rng, T, m)
        n_treated_sub = int(T[idx].sum())
        # Expected 100 * 30/200 = 15 treated; allow ±2 for rounding
        assert 13 <= n_treated_sub <= 17

    def test_extreme_imbalance_does_not_crash(self):
        rng = np.random.default_rng(0)
        n = 200
        T = np.zeros(n)
        T[:5] = 1.0
        idx = _stratified_subsample(rng, T, 50)
        assert len(idx) == 50
        assert len(np.unique(idx)) == 50

    def test_three_levels(self):
        rng = np.random.default_rng(0)
        T = np.tile([0, 1, 2], 30)  # 30 of each
        idx = _stratified_subsample(rng, T, 30)
        assert len(idx) == 30
        assert len(np.unique(idx)) == 30

    def test_reproducible_with_same_seed(self):
        T = np.concatenate([np.zeros(50), np.ones(50)])
        idx1 = _stratified_subsample(np.random.default_rng(42), T, 30)
        idx2 = _stratified_subsample(np.random.default_rng(42), T, 30)
        assert np.array_equal(idx1, idx2)


# ---------------------------------------------------------------------------
# Mocks for bootstrap-level tests (kept local so tests don't pull in fixtures
# that drag heavyweight upstream estimators into the import path)
# ---------------------------------------------------------------------------


class _MockATE:
    """Minimal ATE-only mock with optional fit-time noise."""

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def __init__(self, ate: float = 0.5, name: str = "mock", noise: float = 0.0):
        self._ate = ate
        self._name = name
        self._noise = noise
        self._ate_value = ate

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, random_state=None, **kwargs):
        rng = np.random.default_rng(random_state)
        self._ate_value = (
            self._ate + rng.normal(scale=self._noise) if self._noise else self._ate
        )

    def ate(self, X=None):
        return ComponentAteEstimate(ate=float(self._ate_value))

    def cate(self, X):
        raise NotImplementedError

    def validate_outcome_type(self, detected: str) -> None:
        pass


class _SizeBiasedATE:
    """ATE depends inversely on training size — used to provoke a
    distributional shift between full-sample θ̂ and subsample replicates,
    triggering the CI-exclusion warning.
    """

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def __init__(self, scale: float = 10.0, name: str = "biased"):
        self._scale = scale
        self._name = name
        self._ate_value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, random_state=None, **kwargs):
        rng = np.random.default_rng(random_state)
        self._ate_value = self._scale / len(Y) + float(rng.normal(scale=1e-4))

    def ate(self, X=None):
        return ComponentAteEstimate(ate=float(self._ate_value))

    def cate(self, X):
        raise NotImplementedError

    def validate_outcome_type(self, detected: str) -> None:
        pass


class _UniqueCountBiasedATE:
    """ATE depends inversely on the number of *distinct* training rows.

    A nonparametric (size-n, with-replacement) resample contains only ~63%
    distinct rows, so every replicate's ATE shifts above the full-sample θ̂
    (which sees all n distinct rows). This pushes the percentile CI
    entirely off θ̂ and triggers the CI-exclusion warning even though m == n,
    exercising the nonparametric branch of the diagnostic.
    """

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def __init__(self, scale: float = 10.0, name: str = "unique_biased"):
        self._scale = scale
        self._name = name
        self._ate_value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, random_state=None, **kwargs):
        n_unique = len(np.unique(np.asarray(X), axis=0))
        self._ate_value = self._scale / n_unique

    def ate(self, X=None):
        return ComponentAteEstimate(ate=float(self._ate_value))

    def cate(self, X):
        raise NotImplementedError

    def validate_outcome_type(self, detected: str) -> None:
        pass


def _data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = X[:, 0] + T * 0.5 + rng.normal(scale=0.1, size=n)
    return X, T, Y


def _fitted_cate_ensemble(n=81):
    """Data-dependent components for checking scalar and vector intervals."""
    methods = [
        GenericCATEAdapter(
            lambda X, T, Y, **kwargs: float(Y.mean()),
            lambda state, X: state + X[:, 0], name="a",
        ),
        GenericCATEAdapter(
            lambda X, T, Y, **kwargs: float(Y.mean()),
            lambda state, X: 0.5 * state + X[:, 1], name="b",
        ),
    ]
    X, T, Y = _data(n=n)
    return CausalEnsemble(methods).fit(X, T, Y, random_state=7), X[:7]


# ---------------------------------------------------------------------------
# Bootstrap-level tests
# ---------------------------------------------------------------------------


class TestSubsampleBootstrap:
    def test_returns_bootstrap_result(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data())
        result = ens.bootstrap(n_boot=20, random_state=0, method="subsample")
        assert isinstance(result, BootstrapResult)
        assert result.method == "subsample"
        assert result.subsample_m == 100  # n=200, default fraction 0.5

    def test_default_method_is_nonparametric(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data())
        result = ens.bootstrap(n_boot=10, random_state=0)
        assert result.method == "nonparametric"
        assert result.subsample_m is None

    def test_invalid_method_raises(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data())
        with pytest.raises(ValueError, match="method"):
            ens.bootstrap(n_boot=5, method="invalid")

    def test_int_subsample_size(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data(n=200))
        result = ens.bootstrap(
            n_boot=10, random_state=0, method="subsample", subsample_size=80
        )
        assert result.subsample_m == 80

    def test_invalid_subsample_size_raises(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data(n=100))
        with pytest.raises(ValueError):
            ens.bootstrap(n_boot=5, method="subsample", subsample_size=100)

    def test_boot_ates_shape(self):
        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1), _MockATE(0.6, "b", noise=0.1)])
        ens.fit(*_data())
        result = ens.bootstrap(
            n_boot=15, random_state=0, method="subsample"
        )
        assert result.boot_ates.shape == (15,)

    def test_ci_lower_le_upper_unbiased(self):
        """With unbiased components, percentile bounds should be ordered."""
        ens = CausalEnsemble([
            _MockATE(0.5, "a", noise=0.1),
            _MockATE(0.5, "b", noise=0.1),
        ])
        ens.fit(*_data(n=300), random_state=0)
        result = ens.bootstrap(
            n_boot=50, random_state=0, method="subsample", subsample_size=0.5
        )
        assert result.ate_ci_lower <= result.ate_ci_upper

    def test_ci_contains_point_unbiased(self):
        ens = CausalEnsemble([
            _MockATE(0.5, "a", noise=0.1),
            _MockATE(0.5, "b", noise=0.1),
        ])
        ens.fit(*_data(n=300), random_state=0)
        result = ens.bootstrap(
            n_boot=80, random_state=0, method="subsample", subsample_size=0.5
        )
        assert result.ate_ci_lower <= result.ate <= result.ate_ci_upper

    def test_ci_excluded_warning_fires(self):
        """When components are systematically biased at smaller n, the
        scaled-percentile CI ends up entirely on one side of the full-sample
        point estimate; the package must surface this with a warning.
        """
        ens = CausalEnsemble([_SizeBiasedATE(scale=10.0)])
        ens.fit(*_data(n=200))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            ens.bootstrap(
                n_boot=30,
                random_state=0,
                method="subsample",
                subsample_size=0.5,
            )
        boot_warnings = [
            str(w.message)
            for w in recorded
            if issubclass(w.category, BootstrapWarning)
        ]
        assert any(
            "does not contain the point estimate" in m for m in boot_warnings
        ), f"Expected CI-exclusion BootstrapWarning. Got: {boot_warnings}"

    def test_ci_excluded_warning_fires_nonparametric(self):
        """The exclusion warning must fire under the nonparametric scheme too,
        not only subsample. Containment is governed by the same condition for
        both schemes; here every replicate is biased by the reduced distinct-row
        count of with-replacement resampling.
        """
        ens = CausalEnsemble([_UniqueCountBiasedATE(scale=10.0)])
        ens.fit(*_data(n=200))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            ens.bootstrap(n_boot=30, random_state=0, method="nonparametric")
        boot_warnings = [
            str(w.message)
            for w in recorded
            if issubclass(w.category, BootstrapWarning)
        ]
        assert any(
            "does not contain the point estimate" in m for m in boot_warnings
        ), f"Expected nonparametric CI-exclusion warning. Got: {boot_warnings}"
        # The nonparametric remedy text, not the subsample one.
        assert any("with-replacement resampling" in m for m in boot_warnings)
        assert not any("subsample_size" in m for m in boot_warnings)

    def test_no_ci_excluded_warning_unbiased(self):
        """The CI-exclusion warning should NOT fire under unbiased setup."""
        ens = CausalEnsemble([
            _MockATE(0.5, "a", noise=0.1),
            _MockATE(0.5, "b", noise=0.1),
        ])
        ens.fit(*_data(n=300), random_state=0)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            ens.bootstrap(
                n_boot=80,
                random_state=0,
                method="subsample",
                subsample_size=0.5,
            )
        msgs = [
            str(w.message)
            for w in recorded
            if issubclass(w.category, BootstrapWarning)
        ]
        assert not any(
            "does not contain the point estimate" in m for m in msgs
        ), f"Did not expect CI-exclusion warning; got {msgs}"

    def test_reproducible_with_seed(self):
        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1), _MockATE(0.6, "b", noise=0.1)])
        ens.fit(*_data(), random_state=7)
        r1 = ens.bootstrap(n_boot=10, random_state=42, method="subsample")
        r2 = ens.bootstrap(n_boot=10, random_state=42, method="subsample")
        np.testing.assert_array_equal(r1.boot_ates, r2.boot_ates)
        assert r1.ate_ci_lower == r2.ate_ci_lower
        assert r1.ate_ci_upper == r2.ate_ci_upper

    def test_continuous_T_falls_back_to_unstratified(self):
        """When T has too many unique values to stratify, the package
        warns and falls back to non-stratified subsample (no replacement).
        Threshold mirrors aggregation.splitting._check_stratify (n_unique > 10).
        """
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 3))
        T = rng.normal(size=n)  # continuous T, ~all unique
        Y = X[:, 0] + T * 0.5 + rng.normal(scale=0.1, size=n)

        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1)])
        ens.fit(X, T, Y)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = ens.bootstrap(
                n_boot=10,
                random_state=0,
                method="subsample",
                subsample_size=0.5,
            )
        msgs = [str(w.message) for w in recorded if issubclass(w.category, UserWarning)]
        assert any("continuous treatment" in m for m in msgs), (
            f"Expected continuous-T fallback warning. Got: {msgs}"
        )
        # Replicates still completed (no duplicates, n_failed == 0).
        assert result.n_failed == 0
        assert result.boot_ates.shape == (10,)

    def test_discrete_T_does_not_warn(self):
        """Binary T must not trigger the continuous-T warning."""
        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1)])
        ens.fit(*_data(n=200))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            ens.bootstrap(
                n_boot=10,
                random_state=0,
                method="subsample",
                subsample_size=0.5,
            )
        msgs = [str(w.message) for w in recorded if issubclass(w.category, UserWarning)]
        assert not any("continuous treatment" in m for m in msgs), (
            f"Did not expect continuous-T warning on binary T. Got: {msgs}"
        )

    def test_subsample_replicates_have_no_duplicates(self):
        """Sanity check the actual resampling path inside the bootstrap.

        We patch the component's fit() to record the sample size of each
        replicate's training data, then assert it equals m exactly (with
        replacement, dups would still leave len(idx)==n; we instead check
        that all indices are unique by verifying T_b's stratification).
        """
        recorded_lens: list[int] = []
        recorded_unique: list[int] = []

        class _RecorderATE(_MockATE):
            def fit(self, X, T, Y, random_state=None, **kwargs):
                recorded_lens.append(len(Y))
                # We cannot recover the bootstrap idx itself from inside
                # the mock, but X being float-distinct lets us count
                # unique rows as a proxy for unique units.
                recorded_unique.append(
                    len(np.unique(X, axis=0))
                )
                super().fit(X, T, Y, random_state=random_state, **kwargs)

        ens = CausalEnsemble([_RecorderATE(0.5, "rec", noise=0.0)])
        ens.fit(*_data(n=200, seed=1))
        ens.bootstrap(
            n_boot=5, random_state=0, method="subsample", subsample_size=0.5
        )
        # 1 fit on full data + 5 fits on subsample replicates
        replicate_lens = recorded_lens[1:]
        replicate_unique = recorded_unique[1:]
        assert all(L == 100 for L in replicate_lens)
        assert all(u == 100 for u in replicate_unique)


# ---------------------------------------------------------------------------
# n_train / scale — needed to reconstruct the finite-population scale factor
# after the fact (BootstrapResult.ci_at, component_ate_summary).
# ---------------------------------------------------------------------------


class TestScale:
    def test_n_train_recorded_for_subsample(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data(n=200))
        result = ens.bootstrap(n_boot=10, random_state=0, method="subsample")
        assert result.n_train == 200

    def test_n_train_recorded_for_nonparametric(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data(n=200))
        result = ens.bootstrap(n_boot=10, random_state=0)
        assert result.n_train == 200

    @pytest.mark.parametrize("m,expected", [
        (1, np.sqrt(1 / 199)), (50, np.sqrt(1 / 3)), (100, 1.0),
        (150, np.sqrt(3)), (199, np.sqrt(199)),
    ])
    def test_finite_population_scale_across_subsample_sizes(self, m, expected):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data(n=200))
        result = ens.bootstrap(
            n_boot=5, random_state=0, method="subsample", subsample_size=m,
        )
        assert result.scale == pytest.approx(expected)

    @pytest.mark.parametrize("n,m", [(201, 100), (203, 102)])
    def test_half_fraction_uses_actual_integer_size_for_odd_n(self, n, m):
        ens = CausalEnsemble([_MockATE(0.5, "a")]).fit(*_data(n=n))
        result = ens.bootstrap(n_boot=3, random_state=0, method="subsample")
        assert result.subsample_m == m
        assert result.scale == pytest.approx(np.sqrt(m / (n - m)))
        assert result.scale != 1.0

    def test_numpy_integer_metadata(self):
        assert _bootstrap_scale("subsample", np.int64(3), np.uint64(4)) == pytest.approx(np.sqrt(3))

    def test_scale_is_one_for_nonparametric(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data(n=200))
        result = ens.bootstrap(n_boot=10, random_state=0)
        assert result.scale == 1.0

    @pytest.mark.parametrize("m,n", [
        (None, 10), (2, None), (2, 0), (0, 10), (-1, 10), (1, -10),
        (10, 10), (11, 10), (True, 10), (np.bool_(True), 10), (1, True),
        (1, np.bool_(True)), (2.0, 10), (2.5, 10), (2, 10.0), (2, 10.5),
        (np.nan, 10), (2, np.nan), (np.inf, 10), (2, np.inf), ("2", 10),
    ])
    def test_invalid_subsample_metadata_rejected_on_reconstruction(self, m, n):
        result = BootstrapResult(
            ate=0.5, ate_ci_lower=0.1, ate_ci_upper=0.9,
            boot_ates=np.array([0.4, 0.5, 0.6]),
            cate=None, cate_ci_lower=None, cate_ci_upper=None,
            component_boot_ates={"a": np.array([0.4, 0.5, 0.6])},
            method="subsample", subsample_m=m, n_train=n,
        )
        # Construction/loading still works; using the metadata for a CI
        # must fail clearly rather than returning an unscaled interval.
        for reconstruct in (
            lambda: result.scale, result.ci_at,
            result.component_ate_summary, result.summary,
        ):
            with pytest.raises(ValueError, match="integer subsample_m and n_train"):
                reconstruct()

    def test_missing_size_metadata_rejected(self):
        result = BootstrapResult(
            ate=0.5, ate_ci_lower=0.1, ate_ci_upper=0.9,
            boot_ates=np.array([0.4, 0.5, 0.6]),
            cate=None, cate_ci_lower=None, cate_ci_upper=None,
            method="subsample",
        )
        with pytest.raises(ValueError, match="1 <= subsample_m < n_train"):
            result.ci_at()

    @pytest.mark.parametrize("m,n", [(None, 0), (50, None), (np.nan, np.inf)])
    def test_nonparametric_needs_no_valid_size_metadata(self, m, n):
        result = BootstrapResult(
            ate=0.5, ate_ci_lower=0.1, ate_ci_upper=0.9,
            boot_ates=np.array([0.4, 0.5, 0.6]),
            cate=None, cate_ci_lower=None, cate_ci_upper=None,
            subsample_m=m, n_train=n,
        )
        assert result.scale == 1.0
        np.testing.assert_allclose(result.ci_at(), np.percentile(result.boot_ates, [2.5, 97.5]))

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="method must be"):
            _bootstrap_scale("unknown", 2, 4)


@pytest.mark.parametrize("retained_per_arm", [1, 2, 3])
def test_exact_subsample_variance_matches_difference_in_means_target(retained_per_arm):
    """Enumerate balanced draws; this check derives its target from data,
    independently of the CI scale formula.
    """
    control = np.array([-3.0, -1.0, 2.0, 4.0])
    treated = np.array([0.0, 2.0, 5.0, 9.0])
    point = treated.mean() - control.mean()
    target = treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)
    draws = np.array([
        np.mean(t) - np.mean(c)
        for t, c in product(
            combinations(treated, retained_per_arm),
            combinations(control, retained_per_arm),
        )
    ])
    scale = _bootstrap_scale("subsample", 2 * retained_per_arm, len(treated) + len(control))
    assert draws.mean() == pytest.approx(point)
    # All equally likely subsamples are present, so use population variance
    # for their distribution. Arm variances above use the unbiased ddof=1.
    assert np.var(scale * (draws - point), ddof=0) == pytest.approx(target)


# ---------------------------------------------------------------------------
# ci_at() — CI at an arbitrary level/distribution, reusing an already-
# computed bootstrap run.
# ---------------------------------------------------------------------------


class TestCiAt:
    def test_default_matches_ate_ci_at_same_alpha(self):
        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1), _MockATE(0.6, "b", noise=0.1)])
        ens.fit(*_data(), random_state=7)
        result = ens.bootstrap(n_boot=30, random_state=0, method="subsample")
        lo, hi = result.ci_at(1.0 - result.alpha)
        assert lo == pytest.approx(result.ate_ci_lower)
        assert hi == pytest.approx(result.ate_ci_upper)

    def test_default_level_matches_construction_alpha(self):
        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1), _MockATE(0.6, "b", noise=0.1)])
        ens.fit(*_data(), random_state=7)
        result = ens.bootstrap(n_boot=30, random_state=0, alpha=0.10)
        lo, hi = result.ci_at()  # level defaults to 1 - alpha
        assert lo == pytest.approx(result.ate_ci_lower)
        assert hi == pytest.approx(result.ate_ci_upper)

    def test_wider_level_gives_narrower_interval(self):
        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1), _MockATE(0.6, "b", noise=0.1)])
        ens.fit(*_data(), random_state=7)
        result = ens.bootstrap(n_boot=50, random_state=0, method="subsample")
        lo95, hi95 = result.ci_at(0.95)
        lo70, hi70 = result.ci_at(0.70)
        assert (hi70 - lo70) <= (hi95 - lo95)

    def test_component_dist_matches_centered_percentiles(self):
        """ci_at() on a per-component distribution reproduces the same
        formula the ensemble's own ate_ci uses, applied to that component."""
        ens = CausalEnsemble([_MockATE(0.5, "a", noise=0.1), _MockATE(0.6, "b", noise=0.1)])
        ens.fit(*_data(), random_state=7)
        result = ens.bootstrap(n_boot=40, random_state=0, method="subsample")

        dist = result.component_boot_ates["a"]
        point = result.component_ate_estimates["a"].ate
        lo, hi = result.ci_at(0.90, dist=dist, point_estimate=point)

        alpha = 0.10
        centered = dist - point
        expected_lo = point + result.scale * np.percentile(centered, 100 * alpha / 2)
        expected_hi = point + result.scale * np.percentile(centered, 100 * (1 - alpha / 2))
        assert lo == pytest.approx(expected_lo)
        assert hi == pytest.approx(expected_hi)

    @pytest.mark.parametrize("method", ["nonparametric", "subsample"])
    @pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
    def test_ate_cate_component_intervals_share_training_size_scale(self, method, fraction):
        ens, grid = _fitted_cate_ensemble()
        result = ens.bootstrap(
            grid, n_boot=20, random_state=0, method=method,
            subsample_size=fraction, alpha=0.10,
        )
        assert result.n_train == 81
        assert len(grid) == 7
        assert result.n_failed == 0
        m = round(fraction * 81)
        scale = np.sqrt(m / (81 - m)) if method == "subsample" else 1.0
        levels = [5, 95]
        expected_ate = result.ate + scale * (np.percentile(result.boot_ates, levels) - result.ate)
        np.testing.assert_allclose([result.ate_ci_lower, result.ate_ci_upper], expected_ate)
        np.testing.assert_allclose(result.ci_at(), expected_ate)
        expected_cate = result.cate + scale * (
            np.percentile(result.boot_cates, levels, axis=0) - result.cate
        )
        np.testing.assert_allclose([result.cate_ci_lower, result.cate_ci_upper], expected_cate)
        for i in range(len(grid)):
            np.testing.assert_allclose(
                result.ci_at(dist=result.boot_cates[:, i], point_estimate=result.cate[i]),
                expected_cate[:, i],
            )
        summary = result.component_ate_summary().set_index("name")
        for name, dist in result.component_boot_ates.items():
            point = result.component_ate_estimates[name].ate
            expected = point + scale * (np.percentile(dist, levels) - point)
            np.testing.assert_allclose(summary.loc[name, ["lo", "hi"]], expected)
            assert summary.loc[name, "mean"] == pytest.approx(dist.mean())

    def test_dist_without_point_estimate_raises(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data())
        result = ens.bootstrap(n_boot=10, random_state=0)
        with pytest.raises(ValueError, match="together"):
            result.ci_at(0.90, dist=result.component_boot_ates["a"])

    def test_point_estimate_without_dist_raises(self):
        ens = CausalEnsemble([_MockATE(0.5, "a"), _MockATE(0.6, "b")])
        ens.fit(*_data())
        result = ens.bootstrap(n_boot=10, random_state=0)
        with pytest.raises(ValueError, match="together"):
            result.ci_at(0.90, point_estimate=0.5)
