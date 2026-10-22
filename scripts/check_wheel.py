"""Check packaging boundary: include the source data module, exclude real datasets."""

import sys
import zipfile
from pathlib import Path


def main() -> None:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = list(directory.glob("consensuscensor-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one ConsensusCensor wheel")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        for path in [
            "consensuscensor/data/views.py",
            "consensuscensor/data/multipref.py",
            "consensuscensor/estimators/assisted.py",
        ]:
            if path not in names:
                raise ValueError(f"Missing package source: {path}")
        if any("multipref.json" in path or path.endswith(".parquet") for path in names):
            raise ValueError("Real raw data unexpectedly present in wheel")
    print("Wheel includes required source modules and excludes real data")


if __name__ == "__main__":
    main()
