# First-stage experiment outputs

From the repository root, `python scripts/prepare_data.py` and `python scripts/run_phase1.py` reproduce the pipeline. `scripts/run_phase1.py` is the complete experiment entry point; `scripts/report_results.py` regenerates both numerical reports from CSV/JSON. No tuning on test labels.

- results.csv: every method/seed/rate/partition, full probability estimates, metrics, time and optimization diagnostics. Zero-audit B1/B2 are explicitly unavailable.
- summary.csv: mean/SD/count by target, mechanism, rate, method.
- selection_impact.csv: retention, per-class deletion, group and overall distortion.
- data_audit.json: original nested schema, source, label counts, license and input SHA256.
- frozen_protocol.json / protocol.sha256: settings fixed before experiment execution.
- split_manifest.csv / real_references.json: prompt-group split and finite reference distributions.
- selection_ledgers/: kept/deleted slots, M1 tie indicator, version and seed. Estimators never load these ledgers; the tie indicator may contain information about hidden labels.
- S1_reference.npz / S2_reference.npz: public synthetic labels and known generating parameters (not real data).
- real_agreement_diagnostics.json: full-input descriptive independence warning; not a significance test or tuning input.
- pytest.txt / run.log / ruff.txt: validation records.

Five seeds repeat audit/selection on each fixed population, not population sampling. Timing is CPU/platform dependent. Raw and transformed MultiPref labels remain under ignored data/ and are not in the deliverable archive. Upstream row ordering can change; compare the SHA256 before treating a later download as exact replication.
