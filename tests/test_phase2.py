"""Design expectation, group denominators, posterior enumeration and CF isolation."""

from itertools import combinations, product
from unittest.mock import patch

import numpy as np
import pytest

from consensuscensor.censoring.rules import censor
from consensuscensor.data.views import Audit, Observed, Oracle
from consensuscensor.estimators.assisted import (
    cross_fitted,
    posterior_missing_prediction,
    prompt_folds,
    simple_missing_prediction,
)
from consensuscensor.estimators.residual import (
    AuditDesign,
    contributions,
    post_stratified_m2,
    residual_estimate,
    srs,
)
from consensuscensor.simulation.phase2 import generate_phase2


def finite_population() -> Oracle:
    return Oracle(
        np.array(
            [[0, 0, 0, 2], [1, 1, 1, 1], [0, 2, 1, 2], [2, 2, 0, 0], [2, 1, 2, 2]], dtype=np.int64
        )
    )


def test_b4_all_fixed_size_audit_subsets_design_expectation() -> None:
    oracle = finite_population()
    target = contributions(oracle.labels).mean(axis=0)
    for mechanism in ["M0", "M1", "M2"]:
        observed, _ = censor(oracle, mechanism, 23)
        estimates = []
        for ids in combinations(range(5), 2):
            design = AuditDesign(oracle.release_audit(np.array(ids)), np.full(5, 2 / 5))
            estimates.append(residual_estimate(observed, design).groups)
        np.testing.assert_allclose(np.mean(estimates, axis=0), target, atol=1e-14)


def test_unequal_probabilities_correct_ht_weights() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "M1", 11)
    pi = np.array([0.10, 0.15, 0.20, 0.25, 0.30])  # choose exactly one with these probabilities
    expectation = np.zeros((2, 3))
    for j, probability in enumerate(pi):
        design = AuditDesign(oracle.release_audit(np.array([j])), pi, "unequal-size-one")
        expectation += probability * residual_estimate(obs, design).groups
    np.testing.assert_allclose(expectation, contributions(oracle.labels).mean(axis=0), atol=1e-14)


def test_full_visibility_requires_no_audit_b4_b5() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "NONE", 11)
    design = srs(oracle, 0, 23)
    expected = contributions(oracle.labels).mean(axis=0)
    np.testing.assert_allclose(residual_estimate(obs, design).groups, expected)
    for method in ["B5a", "B5b"]:
        result = cross_fitted(obs, design, np.array([0, 0, 1, 1, 1]), method)
        np.testing.assert_allclose(result.estimate.groups, expected)


def test_m1_visible_total_contribution_exact_three_quarters() -> None:
    obs, _ = censor(finite_population(), "M1", 37)
    visible = contributions(obs.labels)
    np.testing.assert_allclose(visible.mean(axis=1).sum(axis=1), 0.75)


def test_group_denominators_and_weights() -> None:
    labels = np.array([[0, -1, 2, 2]])
    visible = contributions(labels)
    np.testing.assert_array_equal(visible[0, 0], [0.5, 0, 0])
    np.testing.assert_array_equal(visible[0, 1], [0, 0, 1.0])
    np.testing.assert_array_equal(visible.mean(axis=1)[0], [0.25, 0, 0.5])


def test_m2_known_retained_no_extra_audit() -> None:
    oracle = Oracle(np.tile([0, 0, 0, 2], (8, 1)))
    obs, _ = censor(oracle, "M2", 0)
    design = srs(oracle, 0, 1)
    result = post_stratified_m2(obs, design)
    np.testing.assert_allclose(result.groups, contributions(oracle.labels).mean(axis=0))
    assert result.status == "ok"


def test_zero_probability_unknown_marked_unavailable() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "M1", 11)
    design = srs(oracle, 0, 11)
    assert residual_estimate(obs, design).status == "not_estimable"
    assert (
        cross_fitted(obs, design, np.array([0, 0, 1, 1, 1]), "B5a").estimate.status
        == "not_estimable"
    )


def test_zero_pi_only_fully_known_units_allowed() -> None:
    oracle = Oracle(np.array([[0, 0, 0, 2], [0, 1, 2, 0]]))
    obs, _ = censor(oracle, "M2", 0)
    design = AuditDesign(oracle.release_audit(np.array([1])), np.array([0.0, 1.0]))
    np.testing.assert_allclose(
        residual_estimate(obs, design).groups, contributions(oracle.labels).mean(axis=0)
    )


def test_empty_deleted_stratum_audits_unavailable() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "M2", 0)
    design = AuditDesign(oracle.release_audit(np.array([0])), np.full(5, 0.2))
    assert post_stratified_m2(obs, design).status == "not_estimable"


def test_poststrat_m2_equals_known_plus_deleted_audit_mean() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "M2", 0)
    design = AuditDesign(oracle.release_audit(np.array([0, 2])), np.full(5, 0.4))
    result = post_stratified_m2(obs, design)
    expected = contributions(obs.labels).mean(axis=0)
    expected += (~obs.retained).mean() * contributions(oracle.labels[[2]])[0]
    np.testing.assert_allclose(result.groups, expected)


def test_b4_raw_mass_not_silently_normalized() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "M2", 0)
    design = AuditDesign(oracle.release_audit(np.array([0])), np.full(5, 0.2))
    result = residual_estimate(obs, design)
    assert not np.isclose(result.overall.sum(), 1)
    assert (result.groups >= 0).all()


def test_model_assisted_negative_estimate_preserved() -> None:
    oracle = Oracle(np.array([[0, 0, 0, 0], [2, 2, 2, 2]]))
    obs = Observed(np.full((2, 4), -1), np.zeros(2, dtype=bool), "M2")
    prediction = np.array([[[0.0, 0, 1.0], [0, 0, 1.0]], [[1.0, 0, 0], [1.0, 0, 0]]])
    result = residual_estimate(obs, srs(oracle, 1, 11), prediction)
    assert result.groups.min() < 0


def test_prediction_mass_matches_missing_group_slots() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "M1", 37)
    pred = simple_missing_prediction(obs, oracle.release_audit(np.arange(5)), obs)
    expected = (obs.labels.reshape(-1, 2, 2) < 0).sum(axis=2) / 2
    np.testing.assert_allclose(pred.sum(axis=2), expected)
    bad = pred.copy() + 0.1
    with pytest.raises(ValueError):
        residual_estimate(obs, srs(oracle, 2, 11), bad)


def test_posterior_independent_bruteforce_m1_m2() -> None:
    # Independent conditional enumeration, no use of production selection kernel.
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.25, 0.55]])
    target_rows = {"M1": (-1, 0, 1, 0), "M2": (-1, -1, -1, -1)}
    for mechanism, event in target_rows.items():
        expected = np.zeros((2, 3))
        denominator = 0.0
        for y in product(range(3), repeat=4):
            probability = np.prod([p[j // 2, y[j]] for j in range(4)])
            if any(v >= 0 and y[j] != v for j, v in enumerate(event)):
                continue
            if mechanism == "M2":
                if max(y.count(c) for c in range(3)) >= 3:
                    continue
                selection = 1.0
            else:
                costs = []
                for deleted in range(4):
                    kept = [y[j] for j in range(4) if j != deleted]
                    costs.append(
                        sum(abs(kept[a] - kept[b]) for a in range(3) for b in range(a + 1, 3))
                    )
                deleted = event.index(-1)
                selection = 1 / costs.count(min(costs)) if costs[deleted] == min(costs) else 0
            weight = probability * selection
            denominator += weight
            missing = np.array([[-1 if v >= 0 else y[j] for j, v in enumerate(event)]])
            expected += weight * contributions(missing)[0]
        obs = Observed(np.array([event]), np.array([any(v >= 0 for v in event)]), mechanism)
        actual = posterior_missing_prediction(obs, p)[0]
        np.testing.assert_allclose(actual, expected / denominator, atol=1e-14)


def test_b3_posterior_missing_mass_and_no_oracle_access() -> None:
    oracle = finite_population()
    p = np.full((2, 3), 1 / 3)
    for mechanism in ["M0", "M1", "M2"]:
        obs, _ = censor(oracle, mechanism, 11)
        with patch.object(Oracle, "release_audit", side_effect=AssertionError("leak")):
            pred = posterior_missing_prediction(obs, p)
        mass = (obs.labels.reshape(-1, 2, 2) < 0).sum(axis=2) / 2
        np.testing.assert_allclose(pred.sum(axis=2), mass, atol=1e-14)


def test_prompt_fold_integrity_and_reproducibility() -> None:
    prompts = ["p1", "p1", "p2", "p3", "p3", "p4"]
    a = prompt_folds(prompts)
    np.testing.assert_array_equal(a, prompt_folds(prompts))
    assert a[0] == a[1] and a[3] == a[4]
    assert set(a) == {0, 1}


def test_crossfit_own_audit_not_in_predictor_training() -> None:
    oracle, _ = generate_phase2(80, "S2", 11)
    obs, _ = censor(oracle, "M1", 23)
    design = srs(oracle, 24, 37)
    folds = prompt_folds([str(i // 2) for i in range(80)])
    first = cross_fitted(obs, design, folds, "B5a")
    changed = design.audit.labels.copy()
    target = folds[design.audit.indices] == 0
    hidden = obs.labels[design.audit.indices] < 0
    changed[target[:, None] & hidden] = (changed[target[:, None] & hidden] + 1) % 3
    second = cross_fitted(
        obs, AuditDesign(Audit(design.audit.indices, changed), design.pi), folds, "B5a"
    )
    np.testing.assert_array_equal(first.prediction[folds == 0], second.prediction[folds == 0])
    for record in first.trace:
        assert not set(record["target_indices"]) & set(record["train_audit_indices"])
        assert not set(record["target_indices"]) & set(record["train_indices"])


def test_b5b_training_isolation_and_failure_explicit() -> None:
    oracle, _ = generate_phase2(160, "S1", 11)
    obs, _ = censor(oracle, "M1", 23)
    design = srs(oracle, 32, 37)
    folds = prompt_folds([str(i) for i in range(160)])
    first = cross_fitted(obs, design, folds, "B5b")
    assert first.estimate.status == "ok"
    changed = design.audit.labels.copy()
    mask = (folds[design.audit.indices] == 0)[:, None] & (obs.labels[design.audit.indices] < 0)
    changed[mask] = (changed[mask] + 1) % 3
    second = cross_fitted(
        obs, AuditDesign(Audit(design.audit.indices, changed), design.pi), folds, "B5b"
    )
    np.testing.assert_allclose(first.prediction[folds == 0], second.prediction[folds == 0])


def test_conditional_pi_uses_realized_fold_counts() -> None:
    oracle, _ = generate_phase2(20, "S2", 11)
    obs, _ = censor(oracle, "M1", 23)
    folds = np.repeat([0, 1], 10)
    ids = np.array([0, 10, 11, 12])
    design = AuditDesign(oracle.release_audit(ids), np.full(20, 0.2))
    result = cross_fitted(obs, design, folds, "B5a")
    np.testing.assert_allclose(result.conditional_pi[:10], 0.1)
    np.testing.assert_allclose(result.conditional_pi[10:], 0.3)


def test_b5_conditional_all_subset_expectation() -> None:
    oracle = Oracle(finite_population().labels[:4])
    folds = np.array([0, 0, 1, 1])
    obs, _ = censor(oracle, "M1", 23)
    values = []
    for left in [0, 1]:
        for right in [2, 3]:
            ids = np.array([left, right])
            design = AuditDesign(oracle.release_audit(ids), np.full(4, 0.5))
            values.append(cross_fitted(obs, design, folds, "B5a").estimate.groups)
    np.testing.assert_allclose(
        np.mean(values, axis=0), contributions(oracle.labels).mean(axis=0), atol=1e-14
    )


def test_empty_target_fold_audits_unavailable() -> None:
    oracle = finite_population()
    obs, _ = censor(oracle, "M1", 11)
    ids = np.array([0])
    design = AuditDesign(oracle.release_audit(ids), np.full(5, 0.2))
    result = cross_fitted(obs, design, np.array([0, 0, 1, 1, 1]), "B5a")
    assert result.estimate.status == "not_estimable"


def test_srs_actual_inclusion_probability_and_same_audit_for_methods() -> None:
    oracle = finite_population()
    a, b = srs(oracle, 2, 11), srs(oracle, 2, 11)
    np.testing.assert_array_equal(a.audit.indices, b.audit.indices)
    np.testing.assert_allclose(a.pi, 0.4)
    obs, _ = censor(oracle, "M1", 11)
    before = a.audit.indices.copy()
    residual_estimate(obs, a)
    cross_fitted(obs, a, np.array([0, 0, 1, 1, 1]), "B5a")
    np.testing.assert_array_equal(a.audit.indices, before)


def test_s3_heterogeneity_marginals_and_reproducibility() -> None:
    a, truth = generate_phase2(10000, "S3", 11)
    b, _ = generate_phase2(10000, "S3", 11)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(contributions(a.labels).mean(axis=0), truth, atol=0.025)
    observed, _ = censor(a, "M2", 23)
    assert 0 < observed.retained.mean() < 1


def test_b4_does_not_universally_dominate_audit_only_variance() -> None:
    oracle = Oracle(np.tile([0, 0, 2, 2], (4, 1)))
    labels = oracle.labels.copy()
    labels[[0, 2], 0] = -1
    labels[[1, 3], 2] = -1
    obs = Observed(labels, np.ones(4, dtype=bool), "M0")
    estimates = []
    for j in range(4):
        design = AuditDesign(oracle.release_audit(np.array([j])), np.full(4, 0.25))
        estimates.append(residual_estimate(obs, design).overall)
    # Every full comparison has the SAME class proportions: audit-only has zero variance.
    assert np.var(np.array(estimates)[:, 0]) > 0


def test_b5b_nonconvergent_training_is_not_silently_replaced() -> None:
    from types import SimpleNamespace

    oracle, _ = generate_phase2(40, "S1", 11)
    obs, _ = censor(oracle, "M1", 23)
    design = srs(oracle, 12, 37)
    failed = SimpleNamespace(
        success=False,
        boundary=False,
        message="forced failure",
        failed_starts=3,
        probabilities=np.full((2, 3), 1 / 3),
    )
    with patch("consensuscensor.estimators.assisted.fit", return_value=failed):
        result = cross_fitted(obs, design, np.repeat([0, 1], 20), "B5b")
    assert result.estimate.status == "numeric_failure"
    assert np.isnan(result.estimate.groups).all()
