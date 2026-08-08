"""Subsample the extracted ncvoter CSV to a workable experiment set.

``ncvoter_records.csv`` has about 9.2M records, which is far too large to embed
and index for routine experiments. This script produces a deterministic uniform
reservoir sample of ``--count`` records into ``--output``, preserving the source
columns. The sampled CSV is then consumed by the ncvoter experiment scripts.

Usage::

    python scripts/ncvoter/prepare_sample.py \\
        --input datasets/ncvoter/ncvoter_records.csv \\
        --output datasets/ncvoter/sample_5000.csv --count 5000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def subsample(input_path: Path, output_path: Path, count: int, seed: int) -> int:
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    seen = 0
    with input_path.open(encoding="utf-8", newline="") as source, \
         output_path.open("w", encoding="utf-8", newline="") as sink:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(sink, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            seen += 1
            if len(reservoir) < count:
                reservoir.append(dict(row))
            else:
                index = rng.randint(0, seen - 1)
                if index < count:
                    reservoir[index] = dict(row)
        for row in reservoir:
            writer.writerow(row)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically subsample the ncvoter CSV")
    parser.add_argument("--input", type=Path, default="datasets/ncvoter/ncvoter_records.csv")
    parser.add_argument("--output", type=Path, default="datasets/ncvoter/sample_5000.csv")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reservoir sampling {args.count:,} records from {args.input} -> {args.output} ...")
    seen = subsample(args.input, args.output, args.count, args.seed)
    print(f"Done: sampled {args.count:,} of {seen:,} records")


if __name__ == "__main__":
    main()
