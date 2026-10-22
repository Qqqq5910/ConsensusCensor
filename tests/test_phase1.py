"""Substantive selection, likelihood, isolation, and synthetic checks."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from consensuscensor.censoring.rules import SUBSETS, censor, cost, outcomes
from consensuscensor.data.multipref import MAPPING, convert, split_prompts
from consensuscensor.data.views import Audit, Observed, Oracle
from consensuscensor.estimands.distribution import distribution, errors
from consensuscensor.estimators.baselines import baseline
from consensuscensor.estimators.likelihood import (
    design,
    fit,
    kernel,
    masses,
    objective,
)
from consensuscensor.simulation.generate import P, generate


def empty_audit() -> Audit:
    return Audit(np.array([], dtype=np.int64), np.empty((0, 4), dtype=np.int64))


def test_five_levels_preserved_mapping() -> None:
    assert len(MAPPING) == 5
    assert MAPPING["Tie"] == 1
    assert MAPPING["A-is-slightly-better"] == MAPPING["A-is-clearly-better"] == 0
    assert MAPPING["B-is-slightly-better"] == MAPPING["B-is-clearly-better"] == 2


def test_unknown_not_tie(tmp_path: Path) -> None:
    row = {
        "comparison_id": "c",
        "prompt_id": "p",
        "normal_worker_annotations": [{"overall_pref": "unknown"}, {"overall_pref": "Tie"}],
        "expert_worker_annotations": [{"overall_pref": "Tie"}] * 2,
    }
    oracle, _, audit = convert([row], tmp_path)
    assert len(oracle.labels) == 0
    assert any(x["problem"] == "unknown label" for x in audit["anomalies"])
    table = pd.read_parquet(tmp_path / "annotations.parquet")
    assert table.preference_class.isna().sum() == 1
    assert table.preference_raw.iloc[0] == "unknown"


def test_invalid_oracle_rejected() -> None:
    with pytest.raises(ValueError):
        Oracle(np.array([[0, 1, -1, 2]]))


def test_m0_kernel_is_label_independent() -> None:
    for y in product(range(3), repeat=4):
        result = outcomes(y, "M0")
        assert len(result) == 4
        assert all(p == 0.25 and row.count(-1) == 1 for row, p in result)


def test_m0_uniform_empirical() -> None:
    obs, _ = censor(Oracle(np.zeros((12000, 4), dtype=np.int64)), "M0", 11)
    deleted = (obs.labels < 0).mean(axis=0)
    assert np.max(abs(deleted - 0.25)) < 0.02


def test_m1_exhaustive_minimum() -> None:
    for y in product(range(3), repeat=4):
        minimum = min(cost(y, s) for s in SUBSETS)
        for row, _ in outcomes(y, "M1"):
            assert cost(y, tuple(j for j, v in enumerate(row) if v >= 0)) == minimum


def test_m1_tie_kernel_uniform() -> None:
    for y in product(range(3), repeat=4):
        result = outcomes(y, "M1")
        assert all(p == 1 / len(result) for _, p in result)


def test_m1_ties_empirical() -> None:
    obs, ledger = censor(Oracle(np.tile([0, 0, 0, 2], (9000, 1))), "M1", 23)
    assert (obs.labels[:, 3] == -1).all()
    obs, ledger = censor(Oracle(np.ones((12000, 4), dtype=np.int64)), "M1", 23)
    assert all(x["tie"] for x in ledger)
    assert np.max(abs((obs.labels < 0).mean(axis=0) - 0.25)) < 0.02


def test_m2_exhaustive_rule() -> None:
    for y in product(range(3), repeat=4):
        row, p = outcomes(y, "M2")[0]
        assert (row == y) == (max(y.count(c) for c in range(3)) >= 3)
        assert p == 1


def test_all_kernel_columns_normalize() -> None:
    for mechanism in ("NONE", "M0", "M1", "M2"):
        _, matrix = kernel(mechanism)
        np.testing.assert_allclose(matrix.sum(axis=0), 1)
        assert (matrix >= 0).all()


def test_joint_probabilities_normalize() -> None:
    w, dw = masses(np.array([0.7, -0.3, 0.2, 1.0]))
    assert abs(w.sum() - 1) < 1e-12
    np.testing.assert_allclose(dw.sum(axis=0), 0, atol=1e-12)


def test_independent_bruteforce_matches_likelihood() -> None:
    # Independent mechanism implementation; never calls outcomes/cost/kernel.
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.4, 0.4]])
    z = np.log(p[:, :2] / p[:, 2, None]).reshape(-1)
    for mechanism in ("M0", "M1", "M2"):
        events, matrix = kernel(mechanism)
        expected = dict.fromkeys(events, 0.0)
        for y in product(range(3), repeat=4):
            mass = np.prod([p[j // 2, y[j]] for j in range(4)])
            if mechanism == "M2":
                event = y if any(sum(v == c for v in y) >= 3 for c in range(3)) else (-1,) * 4
                expected[event] += mass
                continue
            distances = []
            for deleted in range(4):
                kept = [y[j] for j in range(4) if j != deleted]
                distances.append(
                    sum(abs(kept[a] - kept[b]) for a in range(3) for b in range(a + 1, 3))
                )
            choices = [j for j in range(4) if mechanism == "M0" or distances[j] == min(distances)]
            for deleted in choices:
                event = tuple(-1 if j == deleted else y[j] for j in range(4))
                expected[event] += mass / len(choices)
        np.testing.assert_allclose(matrix @ masses(z)[0], list(expected.values()), atol=1e-14)


def test_analytic_gradient() -> None:
    oracle, _ = generate(80, "S1", 11)
    obs, _ = censor(oracle, "M1", 23)
    matrix, counts = design(obs, oracle.release_audit(np.arange(8)))
    z = np.array([0.2, -0.3, 0.4, -0.5])
    numerical = approx_derivative(lambda x: np.array([objective(x, matrix, counts)[0]]), z)
    np.testing.assert_allclose(objective(z, matrix, counts)[1], numerical.ravel(), rtol=1e-6)


def test_audits_count_exactly_once() -> None:
    oracle, _ = generate(50, "S1", 11)
    obs, _ = censor(oracle, "M1", 23)
    _, counts = design(obs, oracle.release_audit(np.arange(20)))
    assert counts.sum() == 50


def test_full_audit_equals_uncensored_likelihood() -> None:
    oracle, _ = generate(80, "S1", 11)
    full, _ = censor(oracle, "NONE", 0)
    selected, _ = censor(oracle, "M1", 23)
    a, ac = design(full, empty_audit())
    b, bc = design(selected, oracle.release_audit(np.arange(80)))
    z = np.array([0.3, 0.4, -0.2, 0.0])
    assert abs(objective(z, a, ac)[0] - objective(z, b, bc)[0]) < 1e-10


def test_rejected_m2_event_contributes() -> None:
    obs = Observed(np.full((10, 4), -1), np.zeros(10, dtype=bool), "M2")
    matrix, counts = design(obs, empty_audit())
    assert counts.sum() == 10
    assert 0 < (matrix @ masses(np.zeros(4))[0])[0] < 1


def test_hidden_labels_cannot_affect_estimator() -> None:
    a = Oracle(np.array([[0, 0, 0, 1]] * 30))
    b = Oracle(np.array([[0, 0, 0, 2]] * 30))
    oa, _ = censor(a, "M1", 11)
    ob, _ = censor(b, "M1", 11)
    np.testing.assert_array_equal(oa.labels, ob.labels)
    assert not hasattr(oa, "oracle")
    np.testing.assert_allclose(
        fit(oa, empty_audit()).probabilities, fit(ob, empty_audit()).probabilities
    )


def test_observed_readonly_independent_copy() -> None:
    labels = np.array([[0, -1, 1, 2]])
    obs = Observed(labels, np.array([True]), "M0")
    labels[:] = 2
    assert obs.labels[0, 1] == -1
    with pytest.raises(ValueError):
        obs.labels[0, 1] = 0


def test_duplicate_audit_rejected() -> None:
    oracle, _ = generate(20, "S1", 11)
    obs, _ = censor(oracle, "M0", 0)
    with pytest.raises(ValueError):
        design(obs, oracle.release_audit(np.array([1, 1])))


def test_zero_audit_baselines_unavailable() -> None:
    oracle, _ = generate(20, "S1", 11)
    obs, _ = censor(oracle, "M0", 0)
    for method in ("B1", "B2"):
        assert np.isnan(baseline(obs, empty_audit(), method)[0]).all()


def test_b1_b2_equality_balanced_audit() -> None:
    oracle, _ = generate(60, "S1", 11)
    obs, _ = censor(oracle, "M1", 0)
    audit = oracle.release_audit(np.arange(10))
    np.testing.assert_allclose(baseline(obs, audit, "B1")[1], baseline(obs, audit, "B2")[1])


def test_no_filter_mle_matches_empirical() -> None:
    oracle, _ = generate(1000, "S1", 23)
    obs, _ = censor(oracle, "NONE", 0)
    result = fit(obs, empty_audit())
    assert result.success
    np.testing.assert_allclose(result.probabilities, distribution(oracle.labels), atol=1e-5)


def test_correct_model_large_sample_recovery() -> None:
    oracle, _ = generate(12000, "S1", 37)
    for mechanism in ("M0", "M1", "M2"):
        obs, _ = censor(oracle, mechanism, 41)
        result = fit(obs, oracle.release_audit(np.arange(600)))
        assert result.success and result.local_rank == 4
        assert np.max(abs(result.probabilities - P)) < 0.035
        assert (result.probabilities >= 0).all()
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1)


def test_shared_latent_increases_agreement() -> None:
    s1, _ = generate(15000, "S1", 11)
    s2, _ = generate(15000, "S2", 11)
    assert (s2.labels[:, 0] == s2.labels[:, 1]).mean() > (
        s1.labels[:, 0] == s1.labels[:, 1]
    ).mean() + 0.1
    np.testing.assert_allclose(distribution(s2.labels), P, atol=0.025)


def test_seed_reproducibility() -> None:
    a, _ = generate(100, "S2", 23)
    b, _ = generate(100, "S2", 23)
    np.testing.assert_array_equal(a.labels, b.labels)
    oa, la = censor(a, "M1", 11)
    ob, lb = censor(b, "M1", 11)
    np.testing.assert_array_equal(oa.labels, ob.labels)
    assert la == lb


def test_prompt_split_no_overlap() -> None:
    table = pd.DataFrame({"prompt_id": np.repeat(np.arange(100), 2)})
    splits = split_prompts(table)
    ids = [set(table.iloc[i].prompt_id) for i in splits.values()]
    assert len(splits["train"]) == 120
    assert all(not ids[a] & ids[b] for a in range(3) for b in range(a + 1, 3))
    assert len(set(np.concatenate(list(splits.values())))) == len(table)


def test_group_gap_metric_and_tie_score() -> None:
    p = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    q = np.array([[0.0, 1, 0], [1.0, 0, 0]])
    result = errors(p, q)
    assert result["tv"] == 0
    assert result["group_gap_error"] == 1


def test_extreme_sparse_empty_and_boundary() -> None:
    empty = Observed(np.empty((0, 4), dtype=np.int64), np.array([], dtype=bool), "M0")
    assert not fit(empty, empty_audit()).success
    obs, _ = censor(Oracle(np.zeros((30, 4), dtype=np.int64)), "M1", 11)
    result = fit(obs, empty_audit())
    assert result.boundary or result.minimum_singular < 1e-6
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1)


def test_report_synthetic_only_does_not_claim_real_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/report_results.py"
    spec = importlib.util.spec_from_file_location("cc_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "experiments"
    out.mkdir()
    (tmp_path / "docs").mkdir()
    (out / "pytest.txt").write_text("synthetic fixture test record")
    # A stale downloaded-data audit must not imply that a real-data experiment ran.
    (out / "data_audit.json").write_text("{}")
    records = []
    for scenario in ["S1", "S2"]:
        for method in ["B0", "B1", "B2", "B3"]:
            records.append(
                {
                    "dataset": scenario,
                    "split": "full",
                    "n": 20,
                    "seed": 11,
                    "mechanism": "M2",
                    "audit_rate": 0.2,
                    "method": method,
                    "tv": 0.01,
                    "truth_tv": 0.02,
                    "seconds": 0.001,
                    "success": True,
                    "boundary": False,
                    "failed_starts": 0,
                    "local_rank": 4,
                    "min_singular": 0.1,
                    "message": "ok",
                }
            )
    pd.DataFrame(records).to_csv(out / "results.csv", index=False)
    pd.DataFrame(columns=["dataset", "split"]).to_csv(out / "selection_impact.csv", index=False)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    module.main()
    report = (tmp_path / "docs/phase1_results.md").read_text()
    assert "真实数据实验未执行" in report
    assert "真实 Pilot 已执行" not in report
