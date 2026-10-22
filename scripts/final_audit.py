"""Read-only audit of frozen results, with failure-aware sensitivity summaries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from verify_phase2_design import verify

ROOT = Path(__file__).resolve().parents[1]
STRATA = ["experiment", "dataset", "split", "mechanism", "audit_rate"]


def probability_validity(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = frame[[f"{name}_{c}" for c in "ATB"]].to_numpy(dtype=float)
    return (
        np.isfinite(values).all(axis=1)
        & (values >= -1e-10).all(axis=1)
        & (values <= 1 + 1e-10).all(axis=1)
        & (np.abs(values.sum(axis=1) - 1) <= 1e-10)
    )


def failure_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(STRATA + ["method"], sort=False):
        record = dict(zip(STRATA + ["method"], key, strict=True))
        ok = group[group.status == "ok"]
        failures = group[group.status == "numeric_failure"]
        eligible_n = len(ok) + len(failures)
        finite = group[group.status.isin(["ok", "numeric_failure"]) & np.isfinite(group.tv)]
        record.update(
            {
                "recorded_n": len(group),
                "success_n": len(ok),
                "numeric_failure_n": len(failures),
                "not_estimable_n": int((group.status == "not_estimable").sum()),
                "not_applicable_n": int((group.status == "not_applicable").sum()),
                "eligible_n": eligible_n,
                "failure_rate_eligible": len(failures) / eligible_n if eligible_n else np.nan,
                "success_conditional_half_l1": ok.tv.mean(),
                "all_finite_raw_half_l1": finite.tv.mean(),
                "finite_raw_n": len(finite),
                "failure_loss_0_lower": np.nan,
                "failure_loss_1_upper": np.nan,
            }
        )
        # This is a LOSS-ASSIGNMENT sensitivity envelope, not an estimated missing result.
        # Apply only where successful outputs are genuine overall probabilities.
        if eligible_n and len(failures) and len(ok) and probability_validity(ok, "overall").all():
            record["failure_loss_0_lower"] = ok.tv.sum() / eligible_n
            record["failure_loss_1_upper"] = (ok.tv.sum() + len(failures)) / eligible_n
        rows.append(record)
    return pd.DataFrame(rows)


def paired_complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    index = STRATA + ["population_seed", "audit_seed", "audit_hash"]
    comparisons = [
        ("B4", "B1"),
        ("B4-M2", "B4"),
        ("B5a", "B4"),
        ("B5b", "B4"),
        ("B5b", "B3"),
        ("B5b", "B4-M2"),
    ]
    loss = frame.pivot(index=index, columns="method", values="tv")
    status = frame.pivot(index=index, columns="method", values="status")
    rows = []
    for lhs, rhs in comparisons:
        if lhs not in loss or rhs not in loss:
            continue
        mask = (status[lhs] == "ok") & (status[rhs] == "ok")
        values = loss.loc[mask, [lhs, rhs]].reset_index()
        for key, group in values.groupby(STRATA):
            diff = group[lhs] - group[rhs]
            row = dict(zip(STRATA, key, strict=True))
            row.update(
                {
                    "comparison": lhs + " minus " + rhs,
                    "paired_n": len(group),
                    "lhs_same_pairs_mean": group[lhs].mean(),
                    "rhs_same_pairs_mean": group[rhs].mean(),
                    "paired_mean_difference": diff.mean(),
                    "paired_sd_difference": diff.std(),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/finalization/audit")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Audit output must be empty; no archived files are overwritten")
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "experiments/phase2/results.csv"
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    frame = pd.read_csv(raw)
    keys = STRATA + ["population_seed", "audit_seed"]
    assert len(frame) == 28800
    assert (frame.groupby(keys).size() == 8).all()
    assert (frame.groupby(keys).audit_hash.nunique() == 1).all()
    assert (frame.groupby(keys).extra_labels.nunique() == 1).all()
    counts = frame[frame.status == "numeric_failure"].groupby("method").size().to_dict()
    assert counts == {"B3": 3, "B5b": 12}
    for name in ["overall", "normal", "expert"]:
        frame[f"{name}_valid"] = probability_validity(frame, name)
    diagnostics = []
    for key, group in frame[frame.status == "ok"].groupby(
        ["experiment", "dataset", "split", "method"]
    ):
        row = dict(zip(["experiment", "dataset", "split", "method"], key, strict=True))
        row["n"] = len(group)
        row.update(
            {
                f"{name}_invalid_n": int((~group[f"{name}_valid"]).sum())
                for name in ["overall", "normal", "expert"]
            }
        )
        diagnostics.append(row)
    pd.DataFrame(diagnostics).to_csv(out / "raw_vector_validity.csv", index=False)
    failure_sensitivity(frame).to_csv(out / "failure_sensitivity.csv", index=False)
    paired_complete_cases(frame).to_csv(out / "paired_complete_cases.csv", index=False)
    frame[frame.status == "numeric_failure"].to_csv(
        out / "numeric_failure_records.csv", index=False
    )
    enumeration = verify()
    pd.DataFrame(enumeration).to_csv(out / "design_enumeration.csv", index=False)
    # Confirm sampling records, rather than merely trusting a per-method hash column.
    audits = json.loads((ROOT / "experiments/phase2/audit_manifest.json").read_text())
    for audit in audits:
        idx = np.asarray(audit["indices"], dtype=np.int64)
        assert len(idx) == len(np.unique(idx)) == audit["k"]
        assert audit["k"] == int(np.floor(audit["N"] * audit["audit_rate"]))
        assert np.isclose(audit["unconditional_pi"], audit["k"] / audit["N"])
        assert hashlib.sha256(idx.tobytes()).hexdigest() == audit["audit_hash"]
    grouping = frame.groupby(["experiment", "dataset", "split"])
    units = []
    for key, group in grouping:
        experiment, dataset, split = key
        audits_n = group.audit_seed.nunique()
        populations_n = group.population_seed.nunique()
        if experiment == "fixed_audit":
            assert audits_n == 50 and populations_n == 1
        elif experiment == "regenerated":
            assert populations_n == 20 and audits_n == 1
        elif experiment == "real_audit":
            assert audits_n == 30 and populations_n == 1
        units.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "split": split,
                "audit_seeds": audits_n,
                "population_seeds": populations_n,
            }
        )
    pd.DataFrame(units).to_csv(out / "randomization_units.csv", index=False)
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == sha
    report = {
        "archived_rows": len(frame),
        "archived_results_sha256": sha,
        "numerical_failures": counts,
        "audit_records_checked": len(audits),
        "same_audit_indices_and_allocated_budget": True,
        "design_enumeration_max_error": max(x["max_design_expectation_error"] for x in enumeration),
        "archived_file_unchanged": True,
        "success_only_mean_is_conditional": True,
        "failure_sensitivity_is_not_a_confidence_interval": True,
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
