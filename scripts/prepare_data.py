"""Read at most 2000 comparisons, persist provenance and schema."""

import json
from pathlib import Path

from consensuscensor.data.multipref import convert, download
from consensuscensor.estimands.distribution import distribution


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "data/raw/multipref.json"
    try:
        provenance = download(dest)
        oracle, samples, audit = convert(json.loads(dest.read_text()), root / "data/processed")
        audit["distribution_normal_expert"] = distribution(oracle.labels).tolist()
        audit["unique_prompts"] = int(samples.prompt_id.nunique())
        (root / "experiments/phase1/data_audit.json").write_text(
            json.dumps({**provenance, **audit}, indent=2)
        )
    except Exception as error:
        (root / "experiments/phase1/data_failure.json").write_text(
            json.dumps({"error": repr(error), "real_experiment_executed": False}, indent=2)
        )
        raise


if __name__ == "__main__":
    main()
