"""Cross-check the numeric claims in the whitepaper against the result artifacts.

Each claim in ``CLAIMS`` names a source-of-truth JSON file (the script output)
and a field path into it. The verifier:

* computes the expected value from the artifact,
* parses the numbers out of ``entity_resolution_whitepaper.tex``,
* passes if a parsed number lies within ``tol`` of the expected value,
* marks a claim ``NO_ARTIFACT`` when its output file does not exist (so you
  know which numbers cannot be verified from persisted data and must be
  re-run), and
* marks ``STALE`` when a required text anchor has disappeared from the paper.

Usage (from the repository root):

    python scripts/verify_claims.py
    python scripts/verify_claims.py --manifest-out source_papers/claims_manifest.md

Every entry is also rendered into ``claims_manifest.md`` so the reviewer-facing
map of claim -> script -> artifact -> field stays in sync with the checks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX_PATH = ROOT / ".docs" / "entity_resolution_whitepaper.tex"

# domain -> (scaling factor, number regex)
_DOMAIN = {
    "pct": (100.0, r"\d+\.?\d*"),          # stored as fraction -> shown as percent
    "frac": (1.0, r"0\.\d+|\d+\.\d*"),
    "ms": (1.0, r"\d+\.?\d*"),
    "int": (1.0, r"\d[\d,]*"),
}


def get_path(data, path):
    for key in path:
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif isinstance(data, list) and isinstance(key, int) and key < len(data):
            data = data[key]
        else:
            return None
    return data


def select_value(data, match, value_path):
    """Return value_path within the first list item matching all key=value pairs."""
    if not isinstance(data, list):
        return None
    for item in data:
        if all(item.get(k) == v for k, v in match.items()):
            return get_path(item, value_path)
    return None


def tex_numbers(text: str, domain: str):
    pat = _DOMAIN[domain][1]
    out = []
    for m in re.finditer(pat, text):
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def load_claim_value(claim):
    """Return (value, ok, reason) from the artifact."""
    artifact = claim.get("artifact")
    if not artifact:
        return None, False, "no artifact declared (re-run to persist)"
    path = ROOT / artifact
    if not path.is_file():
        return None, False, f"artifact missing: {artifact}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return None, False, f"artifact unreadable: {exc}"
    if "match" in claim:
        value = select_value(data.get(claim["json_key"]), claim["match"], claim["sub"])
    else:
        value = get_path(data, claim["json_path"])
    if value is None:
        return None, False, f"field {claim.get('json_path') or claim.get('json_key')} not found"
    return value, True, ""


def check(claim, text):
    anchor = claim.get("tex_anchor")
    if anchor and anchor.lower() not in text.lower():
        return "STALE", f"anchor {anchor!r} absent from paper"
    value, ok, reason = load_claim_value(claim)
    if not ok:
        return "NO_ARTIFACT", reason
    scale, _ = _DOMAIN[claim["domain"]]
    expected = value * scale
    tol = claim["tol"]
    nums = tex_numbers(text, claim["domain"])
    if any(abs(n - expected) <= tol for n in nums):
        return "PASS", f"{expected:.4g} found in paper (tol {tol})"
    nearby = sorted(nums, key=lambda n: abs(n - expected))[:3]
    return "FAIL", f"expected ~{expected:.4g} not found; nearest {nearby}"


CLAIMS = [
    # ---- Duplicate-bearing 100K benchmark (results from experiment_duplicate_benchmark.py) ----
    dict(id="bk-unt-f1", section="8.2", desc="100K benchmark: untrained F1",
         artifact="results/erwhitepaper/training_results.json", json_path=["variants", "untrained", "metrics", "f1"],
         domain="pct", tol=0.1, tex_anchor="duplicate-bearing benchmark"),
    dict(id="bk-sup-f1", section="8.2", desc="100K benchmark: supervised F1",
         artifact="results/erwhitepaper/training_results.json", json_path=["variants", "supervised", "metrics", "f1"],
         domain="pct", tol=0.1, tex_anchor="duplicate-bearing benchmark"),
    dict(id="bk-em-f1", section="8.2", desc="100K benchmark: EM F1",
         artifact="results/erwhitepaper/training_results.json", json_path=["variants", "em", "metrics", "f1"],
         domain="pct", tol=0.1, tex_anchor="duplicate-bearing benchmark"),

    # ---- Calibration paradox: joint (tau, prior) surface (§8.1) ----
    dict(id="pr-unt-opt", section="8.1", desc="paradox: untrained joint optimum F1",
         artifact="results/erwhitepaper/mu_prior_tau_surface.json", json_path=["summary", "untrained", "best", "f1"],
         domain="pct", tol=2.0, tex_anchor="calibration paradox"),
    dict(id="pr-em-opt", section="8.1", desc="paradox: EM joint optimum F1",
         artifact="results/erwhitepaper/mu_prior_tau_surface.json", json_path=["summary", "em", "best", "f1"],
         domain="pct", tol=2.0, tex_anchor="calibration paradox"),

    # ---- Threshold / address-weight sweep (§8.3) ----
    dict(id="f1-tau095", section="8.3", desc="F1 at tau=0.95 (full address weight)",
         artifact="results/erwhitepaper/f1_sweep_results.json", json_key="grid",
         match={"address_strength": 1.0, "threshold": 0.95}, sub=["metrics", "f1"],
         domain="pct", tol=0.05, tex_anchor="Decision-threshold"),
    dict(id="wkn-addr", section="8.3", desc="weakened-address F1 (0.8)",
         artifact="results/erwhitepaper/confusion_matrix_results_address08.json", json_path=["metrics", "f1"],
         domain="pct", tol=0.2, tex_anchor="weakening the address"),

    # ---- NC-voter replication (§9) ----
    dict(id="nc-f1", section="9", desc="NC-voter mutated resolution F1",
         artifact="results/erwhitepaper/ncvoter/results_resolution.json", json_path=["metrics", "f1"],
         domain="pct", tol=0.1, tex_anchor="Non-Synthetic Replication"),
    dict(id="nc-recall", section="9", desc="NC-voter mutated recall",
         artifact="results/erwhitepaper/ncvoter/results_resolution.json", json_path=["metrics", "recall"],
         domain="pct", tol=0.2, tex_anchor="Non-Synthetic Replication"),
    dict(id="nc-br20", section="9", desc="NC-voter blocking recall @k=20",
         artifact="results/erwhitepaper/ncvoter/results_blocking_recall.json", json_path=["recall_at_k", "20"],
         domain="pct", tol=0.2, tex_anchor="NC Voter Data"),
    dict(id="nc-br100", section="9", desc="NC-voter blocking recall @k=100",
         artifact="results/erwhitepaper/ncvoter/results_blocking_recall.json", json_path=["recall_at_k", "100"],
         domain="pct", tol=0.2, tex_anchor="NC Voter Data"),
    dict(id="nc-f1-k50", section="9", desc="NC-voter F1 at k=50",
         artifact="results/erwhitepaper/ncvoter/results_resolution_k50.json", json_path=["metrics", "f1"],
         domain="pct", tol=0.1, tex_anchor="NC Voter Data"),
    dict(id="nc-f1-k100", section="9", desc="NC-voter F1 at k=100",
         artifact="results/erwhitepaper/ncvoter/results_resolution_k100.json", json_path=["metrics", "f1"],
         domain="pct", tol=0.1, tex_anchor="NC Voter Data"),

    # ---- Claims now backed by persisted artifacts (re-run to persist) ----
    dict(id="cm-baseline", section="8", desc="baseline confusion-matrix F1",
         artifact="results/erwhitepaper/confusion_matrix_results.json", json_path=["metrics", "f1"],
         domain="pct", tol=0.02, tex_anchor="confusion matrix"),
    dict(id="cm-recall", section="8", desc="baseline confusion-matrix recall",
         artifact="results/erwhitepaper/confusion_matrix_results.json", json_path=["metrics", "recall"],
         domain="pct", tol=0.02, tex_anchor="confusion matrix"),
    dict(id="s7-br-def", section="7", desc="Section 7 default top-20 blocking recall",
         artifact="results/erwhitepaper/section7_results.json", json_path=["strategies", "default", "blocking_recall", "20", "recall"],
         domain="pct", tol=0.02, tex_anchor="blocking recall"),
    dict(id="s7-f1-def", section="7", desc="Section 7 default F1 at tau=0.85",
         artifact="results/erwhitepaper/section7_results.json", json_path=["strategies", "default", "threshold_metrics", "0.85", "f1"],
         domain="pct", tol=0.02, tex_anchor="Section 7"),
    dict(id="s7-f1-comp", section="7", desc="Section 7 compact F1 at tau=0.85",
         artifact="results/erwhitepaper/section7_results.json", json_path=["strategies", "compact", "threshold_metrics", "0.85", "f1"],
         domain="pct", tol=0.02, tex_anchor="compact serialization"),
    dict(id="mu-unt-f1", section="8.1", desc="Section 8.1 untrained F1 (50k index)",
         artifact="results/erwhitepaper/mu_calibration_results.json", json_path=["variants", "untrained", "metrics", "f1"],
         domain="pct", tol=0.05, tex_anchor="Trained versus untrained"),
    dict(id="mu-trn-f1", section="8.1", desc="Section 8.1 supervised-trained F1 (50k index)",
         artifact="results/erwhitepaper/mu_calibration_results.json", json_path=["variants", "trained", "metrics", "f1"],
         domain="pct", tol=0.05, tex_anchor="Trained versus untrained"),
]


def render_manifest() -> str:
    lines = ["# Claims Manifest", "",
             "Each claim links a number printed in the paper to the script that produced it and the exact "
             "JSON field in the result artifact. `NO_ARTIFACT` rows have no persisted output and must be "
             "re-run before they can be verified.", "",
             "| ID | Sec | Claim | Script / Artifact | JSON field | Verified value |", "|---|---|---|---|---|---|"]
    for c in CLAIMS:
        if c.get("artifact"):
            path = c["artifact"]
            field = ".".join(c.get("json_path", [])) or (c["json_key"] + repr(c.get("match")))
            script = " ".join(path.split("/")[-1:])
        else:
            path = c.get("script", "")
            field = "(no artifact)"
            script = path
        ver = "re-run to persist" if not c.get("artifact") else f"`{path}`"
        lines.append(f"| {c['id']} | {c['section']} | {c['desc']} | `{script}` | `{field}` | {ver} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(TEX_PATH))
    ap.add_argument("--manifest-out", default=None, help="write the claims manifest to this path and exit")
    args = ap.parse_args()

    if args.manifest_out:
        out = Path(args.manifest_out)
        out.write_text(render_manifest(), encoding="utf-8")
        print(f"wrote manifest -> {out}")
        return

    text = Path(args.tex).read_text(encoding="utf-8")
    print(f"Checking {len(CLAIMS)} claims against {args.tex}\n")
    failed = no_artifact = stale = passed = 0
    for c in CLAIMS:
        status, reason = check(c, text)
        flag = {"PASS": "PASS", "FAIL": "FAIL", "NO_ARTIFACT": "  -", "STALE": "STALE"}[status]
        print(f"[{flag}] {c['id']:14} {c['desc'][:58]:58} {reason}")
        passed += status == "PASS"
        failed += status == "FAIL"
        stale += status == "STALE"
        no_artifact += status == "NO_ARTIFACT"
    print(f"\nPASS={passed}  FAIL={failed}  NO_ARTIFACT={no_artifact}  STALE={stale}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()