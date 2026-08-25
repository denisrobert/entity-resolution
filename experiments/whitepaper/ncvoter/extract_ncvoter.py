"""Extract a standard person CSV from the NC voter registration export.

Reads ``ncvoter_Statewide.txt`` (tab-delimited, quoted) and writes a standard
CSV containing only the columns relevant to entity resolution. Because the
source export does not carry a full date of birth or an email address, the birth
year is kept instead of the DOB and no email column is emitted.

Output columns: ``voter_reg_num``, ``first_name``, ``last_name``, ``birth_year``,
``street_address``, ``city``, ``state``, ``zip_code``.

Rows are kept only where the identity exists (non-empty first and last name);
address and birth year may be empty for a retained row (real missingness).

Usage::

    python scripts/extract_ncvoter.py \\
        --input datasets/ncvoter/ncvoter_Statewide.txt \\
        --output datasets/ncvoter/ncvoter_records.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Expose the repo root, this script's directory, and the shared whitepaper
# experiment dir so entity_resolution, experiments.common, and the sibling
# experiment imports (e.g. experiment_duplicate_benchmark) resolve regardless
# of how this script is invoked.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
while not (_REPO_ROOT / "pyproject.toml").is_file() and _REPO_ROOT != _REPO_ROOT.parent:
    _REPO_ROOT = _REPO_ROOT.parent
for _IMPORT_DIR in (_SCRIPT_DIR, _REPO_ROOT / "experiments" / "whitepaper",
                    _REPO_ROOT / "experiments", _REPO_ROOT):
    _IMPORT_DIR_S = str(_IMPORT_DIR)
    if _IMPORT_DIR_S not in sys.path:
        sys.path.insert(0, _IMPORT_DIR_S)

import argparse
import csv
from pathlib import Path

# Source columns of interest (kept only when present in the header).
SOURCE_COLUMNS = {
    "voter_reg_num": "voter_reg_num",
    "first_name": "first_name",
    "last_name": "last_name",
    "birth_year": "birth_year",
    "res_street_address": "street_address",
    "res_city_desc": "city",
    "state_cd": "state",
    "zip_code": "zip_code",
}


def _strip(value: str) -> str:
    return value.strip(" \t\r\n\"")


def extract(input_path: Path, output_path: Path) -> dict[str, int]:
    seen = 0
    kept = 0
    with input_path.open(encoding="latin-1", newline="") as source, \
         output_path.open("w", encoding="utf-8", newline="") as sink:
        reader = csv.reader(source, delimiter="\t", quotechar='"')
        header = next(reader, None)
        if header is None:
            raise RuntimeError("input file is empty")
        header = [_strip(col) for col in header]
        indices = {name: header.index(name) for name in SOURCE_COLUMNS if name in header}
        missing = [name for name in SOURCE_COLUMNS if name not in indices]
        if missing:
            raise RuntimeError(f"missing expected columns: {missing}")

        out_columns = [SOURCE_COLUMNS[name] for name in header if name in SOURCE_COLUMNS]
        writer = csv.DictWriter(sink, fieldnames=out_columns)
        writer.writeheader()

        for row in reader:
            seen += 1
            first = _strip(row[indices["first_name"]])
            last = _strip(row[indices["last_name"]])
            if not first or not last:
                continue  # identity does not exist for this row
            record = {}
            for name in header:
                if name in SOURCE_COLUMNS:
                    record[SOURCE_COLUMNS[name]] = _strip(row[indices[name]])
            writer.writerow(record)
            kept += 1
            if seen % 500_000 == 0:
                print(f"  processed {seen:,} rows, kept {kept:,}")

    return {"seen": seen, "kept": kept}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a person CSV from the NC voter registration export")
    parser.add_argument("--input", type=Path, default="datasets/ncvoter/ncvoter_Statewide.txt")
    parser.add_argument("--output", type=Path, default="datasets/ncvoter/ncvoter_records.csv")
    args = parser.parse_args()

    print(f"Extracting from {args.input} -> {args.output} ...")
    stats = extract(args.input, args.output)
    print(f"Done: {stats['seen']:,} rows processed, {stats['kept']:,} records kept")


if __name__ == "__main__":
    main()