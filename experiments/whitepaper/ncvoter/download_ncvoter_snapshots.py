"""Download NC voter registration snapshots into datasets/ncvoter_historical.

Pulls ``VR_Snapshot_{year}0101.zip`` files from the NC State Board of Elections
public download area (``https://dl.ncsbe.gov/data/Snapshots/``) for the years
2010-2026 into ``datasets/ncvoter_historical``.

These are the full per-voter registration extracts containing one row per
``voter_reg_num`` with ``res_street_address``, ``res_city_desc``, ``state_cd``,
``zip_code``, ``birth_year`` and ``dt_registration`` --- the input the two-snapshot
temporal-aware experiment needs.

Usage::

    python scripts/download_ncvoter_snapshots.py [--years 2010 2012 2022] [--output datasets/ncvoter_historical]
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
import sys
import urllib.request
from pathlib import Path

# NC publishes snapshots at the S3-backed ``dl.ncsbe.gov`` host; the direct
# object URL is under s3.amazonaws.com and redirects are followed automatically.
BASE_URL = "https://s3.amazonaws.com/dl.ncsbe.gov/data/Snapshots/VR_Snapshot_{year}0101.zip"
_DEFAULT_DIR = Path("datasets/ncvoter_historical")


def download_one(url: str, dest: Path, timeout: int = 600) -> tuple[str, int]:
    """Download ``url`` to ``dest`` with resume support; return (status, bytes)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(
        url, headers={"User-Agent": "kilo-entity/1.0", "Range": f"bytes={existing}-"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("ab") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.rename(dest)
    return "ok", dest.stat().st_size


def download_snapshots(
    output_dir: Path = _DEFAULT_DIR,
    years: list[int] | None = None,
    overwrite: bool = False,
    timeout: int = 600,
) -> list[dict[str, str]]:
    """Download ``VR_Snapshot_{year}0101.zip`` for each year into ``output_dir``.

    Returns a list of per-file results.
    """
    if years is None:
        years = list(range(2010, 2027))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []

    for year in years:
        url = BASE_URL.format(year=year)
        dest = output_dir / f"VR_Snapshot_{year}0101.zip"
        if dest.exists() and not overwrite:
            results.append({"year": str(year), "url": url, "status": "exists",
                            "path": str(dest), "bytes": dest.stat().st_size})
            print(f"[{year}] already present, skipping ({dest.stat().st_size:,} bytes)", flush=True)
            continue
        try:
            print(f"[{year}] downloading {url} ...", flush=True)
            status, size = download_one(url, dest, timeout)
            results.append({"year": str(year), "url": url, "status": status,
                            "path": str(dest), "bytes": size})
            print(f"[{year}] done ({size:,} bytes)", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"year": str(year), "url": url, "status": f"error: {exc}",
                            "path": str(dest), "bytes": 0})
            print(f"[{year}] FAILED: {exc}", flush=True)
    return results


def download_snapshots_wrapper(years, output_dir):
    """Argument-compatible wrapper so the function name matches module use."""
    return download_snapshots(Path(output_dir), years=years)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NC voter snapshot registration files")
    parser.add_argument("--year", type=int, nargs="*", default=list(range(2010, 2027)),
                        help="Specific years (default 2010..2026)")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    results = download_snapshots(
        output_dir=args.output_dir,
        years=args.year,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nFinished: {ok}/{len(results)} downloaded.")


if __name__ == "__main__":
    main()