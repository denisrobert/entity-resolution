"""Prepare NC voter snapshot data for the temporal (age-gap) experiment.

The downloaded ``datasets/ncvoter_historical/VR_Snapshot_{year}0101.zip`` files are
the full per-voter registration snapshots published by the NC State Board of
Elections. They are (a) UTF-16-encoded and (b) very large (9-17 GB uncompressed),
so this script provides two reproducible, memory-safe steps:

1. ``extract`` -- stream one year's snapshot, keep only the fields needed for
   entity resolution, optionally restrict to a county, and write a compact
   per-snapshot CSV to ``datasets/ncvoter_snapshots/``.

2. ``join`` -- take two extracted snapshots (an older and a newer year) that
   share the stable ``voter_id`` key and emit *the same voter at two points in
   time*: both addresses, both capture years, the age gap, whether the residence
   changed, and a short/long cohort. This is the input the gap-weighted
   experiment (``experiment_temporal_gap.py``) needs, built on real addresses
   and real capture dates instead of a synthesizer.

Reproduction::

    # 1. extract one county from the 2012 and 2022 snapshots (optionally --limit)
    python experiments/whitepaper/ncvoter/ncvoter_snapshot_prep.py extract \\
        --source datasets/ncvoter_historical/VR_Snapshot_20120101.zip \\
        --county WAKE --output datasets/ncvoter_snapshots/wake_2012.csv
    python experiments/whitepaper/ncvoter/ncvoter_snapshot_prep.py extract \\
        --source datasets/ncvoter_historical/VR_Snapshot_20220101.zip \\
        --county WAKE --output datasets/ncvoter_snapshots/wake_2022.csv

    # 2. join the two snapshots into temporal duplicate/stable pairs
    python experiments/whitepaper/ncvoter/ncvoter_snapshot_prep.py join \\
        --older datasets/ncvoter_snapshots/wake_2012.csv \\
        --newer datasets/ncvoter_snapshots/wake_2022.csv

The ``extract`` step filters on county; ``join`` keys on ``voter_id`` and keeps
voters present in both snapshots.
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
import io
import zipfile
from pathlib import Path

# internal name -> snapshot column name (stable across vintages).
SOURCE_COLS = {
    "voter_id": "voter_reg_num",
    "first_name": "first_name",
    "last_name": "last_name",
    "age": "age",
    "birth_place": "birth_place",
    "house_num": "house_num",
    "street_dir": "street_dir",
    "street_name": "street_name",
    "street_type": "street_type_cd",
    "street_sufx": "street_sufx_cd",
    "unit_designator": "unit_designator",
    "unit_num": "unit_num",
    "city": "res_city_desc",
    "state": "state_cd",
    "zip": "zip_code",
    "county": "county_desc",
    "registr_dt": "registr_dt",
    "snapshot_dt": "snapshot_dt",
}

OUT_FIELDS = [
    "voter_id", "first_name", "last_name", "age", "birth_place",
    "street", "city", "state", "zip", "county",
    "registr_dt", "snapshot_dt", "year",
]

PAIR_FIELDS = [
    "voter_id", "age", "birth_place",
    "year_older", "year_newer", "gap_years",
    "address_older", "address_newer", "moved", "cohort",
]


def _nonempty(vals: list[str]) -> list[str]:
    return [v.strip() for v in vals if v and v.strip()]


def _unit(row: dict[str, str]) -> str:
    d = row.get("unit_designator", "").strip()
    u = row.get("unit_num", "").strip()
    if not d and not u:
        return ""
    return f"{d} {u}".strip()


def build_street(row: dict[str, str]) -> str:
    """Compose a plain one-line street address from the split NC fields."""
    street = " ".join(_nonempty([
        row.get("house_num", ""),
        row.get("street_dir", ""),
        row.get("street_name", ""),
        row.get("street_type", ""),
        row.get("street_sufx", ""),
        _unit(row),
    ]))
    return ", ".join(_nonempty([
        street, row.get("city", ""), row.get("state", ""), row.get("zip", ""),
    ]))


# ---------------------------------------------------------------------------
# Reading a snapshot
# ---------------------------------------------------------------------------


def _detect(source: Path):
    """Return ``(inner_member, encoding)`` read from the snapshot zip header."""
    with zipfile.ZipFile(source) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f:
            head = f.read(4096)
    for enc in ("utf-16-le", "utf-16", "utf-8"):
        try:
            line = head.decode(enc, errors="ignore").split("\n", 1)[0]
            cols = [c.strip("\x00 \ufeff\r\n").strip() for c in line.split("\t")]
            cols = [c for c in cols if c]
            if cols and any(src in cols for src in SOURCE_COLS.values()):
                return inner, enc
        except Exception:
            continue
    raise RuntimeError(f"could not read header of {source}")


def _iter_rows(source: Path, inner: str, encoding: str):
    """Yield stdout-clean text lines from a snapshot's inner member, streamed."""
    with zipfile.ZipFile(source) as zf, zf.open(inner) as fd:
        txt = io.TextIOWrapper(fd, encoding=encoding, newline="")
        for line in txt:
            yield line.rstrip("\n").rstrip("\r")


def extract(source: Path, output: Path, county: str | None = None,
            limit: int | None = None) -> dict[str, int]:
    """Stream a snapshot into a compact per-row CSV, optionally by county."""
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    inner, enc = _detect(source)

    # header
    ri = iter(_iter_rows(source, inner, enc))
    header = [c.strip().strip("\ufeff") for c in next(ri).split("\t")]
    header = [c for c in header if c]
    idx = {name: header.index(src) for name, src in SOURCE_COLS.items() if src in header}

    seen = kept = 0
    with open(output, "w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for raw in ri:
            seen += 1
            parts = [c.strip() for c in raw.split("\t")]
            row = {name: (parts[i] if i < len(parts) else "") for name, i in idx.items()}
            if county and row["county"].upper() != county.upper():
                continue
            if limit is not None and kept >= limit:
                break
            year = row.get("snapshot_dt", "")[:4]
            rec = {
                "voter_id": row.get("voter_id", ""),
                "first_name": row.get("first_name", ""),
                "last_name": row.get("last_name", ""),
                "age": row.get("age", ""),
                "birth_place": row.get("birth_place", ""),
                **dict.fromkeys(["city", "state", "zip"], ""),
                "county": row.get("county", ""),
                "registr_dt": row.get("registr_dt", ""),
                "snapshot_dt": row.get("snapshot_dt", ""),
                "year": year,
            }
            rec["city"] = row.get("city", "")
            rec["state"] = row.get("state", "")
            rec["zip"] = row.get("zip", "")
            rec["street"] = build_street(row)
            writer.writerow({k: rec.get(k, "") for k in OUT_FIELDS})
            kept += 1
    return {"rows_seen": seen, "rows_kept": kept}


# ---------------------------------------------------------------------------
# Joining two snapshots
# ---------------------------------------------------------------------------


def _load_index(path: Path):
    """Load extracted CSV into {voter_id: row} (memory: OK for a single county)."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {r["voter_id"]: r for r in reader if r["voter_id"]}


def join(older: Path, newer: Path, output: Path | None = None,
         residency_years: int = 10) -> dict[str, int]:
    """Join two extracted snapshots into temporal duplicate/stable pairs."""
    old = _load_index(older)
    new = _load_index(newer)
    if output is None:
        output = older.parent / f"{older.stem}__{newer.stem}__pairs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    moved = 0
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for vid, r in old.items():
            n = new.get(vid)
            if n is None:
                continue
            y_old = int(r["year"] or 0)
            y_new = int(n["year"] or 0)
            gap = y_new - y_old
            cohort = "short" if gap <= residency_years else "long"
            addr_old = r.get("street", "")
            addr_new = n.get("street", "")
            moved_flag = bool(addr_old and addr_new and addr_old != addr_new)
            moved += int(moved_flag)
            total += 1
            writer.writerow({
                "voter_id": vid,
                "age": r.get("age", ""),
                "birth_place": r.get("birth_place", ""),
                "year_older": y_old, "year_newer": y_new, "gap_years": gap,
                "address_older": addr_old, "address_newer": addr_new,
                "moved": moved_flag, "cohort": cohort,
            })
    return {"pairs": total, "moved": moved}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep NC voter snapshots for the experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="stream a snapshot zip into a per-row CSV")
    p.add_argument("--source", required=True)
    p.add_argument("--county", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=lambda a: print(extract(Path(a.source), Path(a.output),
                                              county=a.county, limit=a.limit)))

    p = sub.add_parser("join", help="join two extracted snapshots on voter_id")
    p.add_argument("--older", required=True)
    p.add_argument("--newer", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--residency-years", type=int, default=10)
    p.set_defaults(fn=lambda a: print(join(Path(a.older), Path(a.newer),
                                           Path(a.output) if a.output else None,
                                           residency_years=a.residency_years)))

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()