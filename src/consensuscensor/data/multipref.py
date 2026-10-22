"""Bounded HF rows API retrieval and lossless five-level overall-label preservation."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

from consensuscensor.data.views import Oracle

MAPPING = {
    "A-is-clearly-better": 0,
    "A-is-slightly-better": 0,
    "Tie": 1,
    "B-is-slightly-better": 2,
    "B-is-clearly-better": 2,
}


def download(destination: Path, limit: int = 2000) -> dict[str, Any]:
    if not 1 <= limit <= 2000:
        raise ValueError("Pilot capped at 2000 comparisons")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    features: list[dict] = []
    for offset in range(0, limit, 100):
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=allenai/multipref"
            f"&config=default&split=train&offset={offset}&length={min(100, limit - offset)}"
        )
        with urlopen(url, timeout=30) as response:
            payload = json.load(response)
        features = payload["features"]
        for entry in payload["rows"]:
            if entry.get("truncated_cells"):
                raise ValueError("Truncated API data; do not silently use")
            rows.append(entry["row"])
        print(f"Read {len(rows)} comparisons", flush=True)
    content = json.dumps(rows, ensure_ascii=False).encode()
    destination.write_bytes(content)
    return {
        "rows_read": len(rows),
        "sha256": hashlib.sha256(content).hexdigest(),
        "features": features,
        "source": "allenai/multipref",
        "config": "default",
        "upstream_split": "train",
        "license": "ODC-BY",
        "sampling": "first n rows",
    }


def convert(rows: list[dict[str, Any]], output: Path) -> tuple[Oracle, pd.DataFrame, dict]:
    output.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    annotations: list[dict] = []
    usable: list[list[int]] = []
    anomalies: list[dict] = []
    seen: set[str] = set()
    raw_counts: Counter = Counter()
    for row in rows:
        cid = row.get("comparison_id")
        labels: list[int] = []
        valid = bool(cid and row.get("prompt_id") and cid not in seen)
        seen.add(cid)
        for group in ("normal", "expert"):
            values = row.get(group + "_worker_annotations") or []
            if len(values) != 2:
                valid = False
                anomalies.append({"comparison_id": cid, "problem": group + " slot count"})
            for j, annotation in enumerate(values):
                raw = annotation.get("overall_pref")
                cls = MAPPING.get(raw)
                raw_counts[str(raw)] += 1
                if cls is None:
                    valid = False
                    anomalies.append({"comparison_id": cid, "problem": "unknown label", "raw": raw})
                if not annotation.get("evaluator"):
                    anomalies.append({"comparison_id": cid, "problem": "missing evaluator"})
                labels.append(-1 if cls is None else cls)
                annotations.append(
                    {
                        "comparison_id": cid,
                        "annotator_slot": group + str(j),
                        "annotator_group": group,
                        "preference_raw": raw,
                        "preference_class": None if cls is None else "ATB"[cls],
                        "evaluator": annotation.get("evaluator"),
                    }
                )
        sample = {
            k: row.get(k)
            for k in (
                "comparison_id",
                "prompt_id",
                "category",
                "model_a",
                "model_b",
                "source",
                "subject_study",
            )
        }
        sample["usable"] = valid and len(labels) == 4
        samples.append(sample)
        if sample["usable"]:
            usable.append(labels)
        else:
            anomalies.append({"comparison_id": cid, "problem": "excluded comparison"})
    table = pd.DataFrame(samples)
    table.to_parquet(output / "samples.parquet", index=False)
    pd.DataFrame(annotations).to_parquet(output / "annotations.parquet", index=False)
    return (
        Oracle(np.array(usable, dtype=np.int64).reshape(-1, 4)),
        table[table.usable].copy(),
        {
            "rows_read": len(rows),
            "usable": len(usable),
            "anomalies": anomalies,
            "raw_label_counts": dict(raw_counts),
        },
    )


def split_prompts(samples: pd.DataFrame, seed: int = 20261022) -> dict[str, np.ndarray]:
    prompts = np.array(sorted(samples.prompt_id.unique()))
    np.random.default_rng(seed).shuffle(prompts)
    a, b = int(0.6 * len(prompts)), int(0.8 * len(prompts))
    return {
        name: np.flatnonzero(samples.prompt_id.isin(ids).to_numpy())
        for name, ids in zip(
            ("train", "validation", "test"), (prompts[:a], prompts[a:b], prompts[b:]), strict=True
        )
    }
