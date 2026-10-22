"""Frozen small pilot. Run tests first; synthetic before real. No model tuning."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from consensuscensor.censoring.rules import censor
from consensuscensor.data.multipref import convert, split_prompts
from consensuscensor.data.views import Oracle
from consensuscensor.estimands.distribution import distribution, errors, tv
from consensuscensor.estimators.baselines import baseline
from consensuscensor.estimators.likelihood import fit
from consensuscensor.simulation.generate import DELTA, P, generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase1"
SEEDS = [11, 23, 37, 41, 59]
RATES = [0.0, 0.05, 0.10, 0.20]


def run_target(
    oracle: Oracle, dataset: str, split: str, truth: np.ndarray | None = None
) -> tuple[list[dict], list[dict]]:
    reference = distribution(oracle.labels)
    rows: list[dict] = []
    impacts: list[dict] = []
    n = len(oracle.labels)
    for seed in SEEDS:
        # Shared random audit ordering for every method AND every censoring mechanism.
        ordering = np.random.default_rng(seed + 100000).permutation(n)
        for mechanism in ["M0", "M1", "M2"]:
            observed, ledger = censor(oracle, mechanism, seed)
            ledger_dir = OUT / "selection_ledgers"
            ledger_dir.mkdir(exist_ok=True)
            (ledger_dir / f"{dataset}_{split}_{mechanism}_{seed}.json").write_text(
                json.dumps(ledger, separators=(",", ":"))
            )
            before = np.bincount(oracle.labels.ravel(), minlength=3)
            after = np.bincount(observed.labels[observed.labels >= 0], minlength=3)
            retained = distribution(observed.labels)
            pooled = after / after.sum() if after.sum() else np.full(3, np.nan)
            impacts.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "seed": seed,
                    "mechanism": mechanism,
                    "n": n,
                    "annotation_retention": float((observed.labels >= 0).mean()),
                    "sample_retention": float(observed.retained.mean()),
                    "tv": tv(pooled, reference.mean(axis=0)),
                    "normal_tv": tv(retained[0], reference[0]),
                    "expert_tv": tv(retained[1], reference[1]),
                    **{
                        f"deletion_{c}": float(1 - after[j] / before[j]) if before[j] else np.nan
                        for j, c in enumerate("ATB")
                    },
                }
            )
            for rate in RATES:
                audit = oracle.release_audit(np.sort(ordering[: int(np.floor(rate * n))]))
                for method in ["B0", "B1", "B2", "B3"]:
                    start = perf_counter()
                    diagnostics: dict = {}
                    if method == "B3":
                        result = fit(observed, audit)
                        groups = result.probabilities
                        total = groups.mean(axis=0)
                        diagnostics = {
                            "success": result.success,
                            "nll": result.nll,
                            "local_rank": result.local_rank,
                            "min_singular": result.minimum_singular,
                            "multistart_spread": result.multistart_spread,
                            "boundary": result.boundary,
                            "failed_starts": result.failed_starts,
                            "message": result.message,
                        }
                    else:
                        groups, total = baseline(observed, audit, method)
                        diagnostics["success"] = bool(np.isfinite(total).all())
                    row = {
                        "dataset": dataset,
                        "split": split,
                        "seed": seed,
                        "mechanism": mechanism,
                        "audit_rate": rate,
                        "method": method,
                        "audit_n": len(audit.indices),
                        "n": n,
                        "seconds": perf_counter() - start,
                        **diagnostics,
                        **errors(groups, reference, total),
                    }
                    if truth is not None:
                        row.update(
                            {"truth_" + k: v for k, v in errors(groups, truth, total).items()}
                        )
                    for g, probs in zip(
                        ["normal", "expert", "overall"], [*groups, total], strict=True
                    ):
                        row.update({f"{g}_{c}": float(probs[j]) for j, c in enumerate("ATB")})
                    rows.append(row)
        print(f"{dataset}/{split} seed={seed} completed", flush=True)
    return rows, impacts


def agreement_diagnostics(oracle: Oracle) -> list[dict]:
    rows = []
    for a in range(4):
        for b in range(a + 1, 4):
            pa = np.bincount(oracle.labels[:, a], minlength=3) / len(oracle.labels)
            pb = np.bincount(oracle.labels[:, b], minlength=3) / len(oracle.labels)
            observed = float((oracle.labels[:, a] == oracle.labels[:, b]).mean())
            rows.append(
                {
                    "slot_a": a,
                    "slot_b": b,
                    "observed_agreement": observed,
                    "independent_marginal_agreement": float(pa @ pb),
                    "excess_agreement": observed - float(pa @ pb),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-n", type=int, default=1500)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    check = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True
    )
    (OUT / "pytest.txt").write_text(check.stdout + check.stderr)
    check.check_returncode()
    protocol = {
        "seeds": SEEDS,
        "audit_rates": RATES,
        "synthetic_n": args.synthetic_n,
        "S1_probabilities": P.tolist(),
        "S2_delta": DELTA.tolist(),
        "S2_mixture_weights": [0.5, 0.5],
        "split_seed": 20261022,
        "model": "group-only independent categorical, no tuned hyperparameters",
        "optimization": "3 fixed starts, L-BFGS-B logits [-20,20], maxiter=500",
        "evaluation": "fit permitted observations within each target partition; test is "
        "transductive finite-target estimation, not held-out label prediction",
        "uncertainty": "five paired randomizations: mean and SD; no significance claims",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (OUT / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2))
    (OUT / "protocol.sha256").write_text(
        hashlib.sha256((OUT / "frozen_protocol.json").read_bytes()).hexdigest()
    )
    all_rows, all_impacts = [], []
    for scenario in ["S1", "S2"]:
        # Regenerate full synthetic target per fixed seed? Fixed population here; selection
        # randomizations are repeated. Independent population check is in test suite.
        oracle, truth = generate(args.synthetic_n, scenario, 20261022)
        np.savez(OUT / f"{scenario}_reference.npz", labels=oracle.labels, truth=truth)
        rows, impacts = run_target(oracle, scenario, "full", truth)
        all_rows += rows
        all_impacts += impacts
        pd.DataFrame(all_rows).to_csv(OUT / "results.csv", index=False)
        pd.DataFrame(all_impacts).to_csv(OUT / "selection_impact.csv", index=False)
    if not args.synthetic_only and (ROOT / "data/raw/multipref.json").exists():
        oracle, samples, _ = convert(
            json.loads((ROOT / "data/raw/multipref.json").read_text()), ROOT / "data/processed"
        )
        splits = split_prompts(samples)
        manifest = []
        refs = {}
        for split, indices in splits.items():
            subset = Oracle(oracle.labels[indices])
            refs[split] = {
                "n": len(indices),
                "prompts": int(samples.iloc[indices].prompt_id.nunique()),
                "normal_expert": distribution(subset.labels).tolist(),
            }
            manifest.extend(
                {
                    "comparison_id": str(samples.iloc[i].comparison_id),
                    "prompt_id": str(samples.iloc[i].prompt_id),
                    "split": split,
                }
                for i in indices
            )
        pd.DataFrame(manifest).to_csv(OUT / "split_manifest.csv", index=False)
        (OUT / "real_references.json").write_text(json.dumps(refs, indent=2))
        (OUT / "real_agreement_diagnostics.json").write_text(
            json.dumps(agreement_diagnostics(oracle), indent=2)
        )
        for split, indices in splits.items():
            rows, impacts = run_target(Oracle(oracle.labels[indices]), "MultiPref", split)
            all_rows += rows
            all_impacts += impacts
            pd.DataFrame(all_rows).to_csv(OUT / "results.csv", index=False)
            pd.DataFrame(all_impacts).to_csv(OUT / "selection_impact.csv", index=False)
    grouping = ["dataset", "split", "mechanism", "audit_rate", "method"]
    frame = pd.DataFrame(all_rows)
    metrics = [
        "tv",
        "score_error",
        "normal_tv",
        "expert_tv",
        "group_gap_error",
        "group_tv_gap_error",
        "seconds",
    ]
    summary = frame.groupby(grouping)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary.to_csv(OUT / "summary.csv")
    subprocess.run([sys.executable, str(ROOT / "scripts/report_results.py")], check=True)


if __name__ == "__main__":
    main()
