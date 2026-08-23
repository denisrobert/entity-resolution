"""Quick probe: can all-MiniLM-L6-v2 run through OpenVINO on GPU (or CPU)?

Purpose
-------
This project embeds every person row with ``sentence-transformers/all-MiniLM-L6-v2``
through PyTorch. sentence-transformers ships an **OpenVINO backend** that can run
the same model on Intel CPU/GPU without PyTorch inference. This script answers,
quickly:

1. is the OpenVINO stack installed? (``openvino`` + ``optimum-intel``)
2. which OpenVINO devices are visible (CPU, GPU, AUTO, NPU, ...)?
3. does the OpenVINO backend actually load and embed with the pinned model,
   producing the expected 384-d vectors with sane cosine neighbourhoods, and
   how fast is it on the requested device?

KNOWN-ISSUE / WORKAROUND
------------------------
On sentence-transformers 5.6.1 the ``backend="openvino", device="GPU"`` path
fails with ``RuntimeError: Expected one of cpu, cuda, ... at start of device
string: GPU``. The cause is sentence-transformers' container calling
``nn.Module.to(device)`` with the OpenVINO device string, which torch rejects;
it is an integration bug, NOT an unsupported device. The workaround (already
in this script) is to construct with ``device="cpu"`` (a torch-parseable device)
and then move the OpenVINO sub-model to the device with
``model[0].auto_model.to(device)``. This engages the OpenVINO device
(e.g. ``GPU``) for inference while keeping the container's torch call valid.

It intentionally uses the exact model id used by the project
(``sentence-transformers/all-MiniLM-L6-v2``), so the result transfers directly to
the experiment pipeline (``entity_pipeline.HuggingFaceEmbeddingModel`` and the
stores). OpenVINO is a *replacement for the torch backend*, not a different
model.

Install (once)::

    python -m pip install -e ".[openvino]"

Run::

    python test_openvino_gpu.py               # tries GPU, then CPU, then AUTO
    python test_openvino_gpu.py --device GPU
    python test_openvino_gpu.py --device CPU --rows 8
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Optional

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def openvino_available() -> tuple[Optional[str], list[str]]:
    """Return ``(openvino_version, available_devices)``; ``None``/``[]`` if absent."""
    try:
        import openvino
    except ImportError:
        return None, []
    version = getattr(openvino, "__version__", "unknown")
    try:
        from openvino import Core

        devices = list(Core().available_devices)
    except Exception:
        devices = []
    return version, devices


def build_sample_rows(n: int) -> list[dict[str, str]]:
    """Minimal stand-ins for Person rows (same serialization shape as the project).

    Rows 0 and 1 are a near-duplicate pair (identical identity; the address
    number differs by one) so the sanity check can assert the near-duplicate
    distances correctly. Row 2 is a dissimilar person.
    """
    if n < 3:
        n = 3
    rows = [
        {
            "First Name": "Grace",
            "Last Name": "Smith",
            "Date of Birth": "1985-06-15",
            "Address": "100 Main Street, Toronto, ON M5V 1A1",
            "Email": "grace.smith@example.com",
        },
        {
            "First Name": "Grace",
            "Last Name": "Smith",
            "Date of Birth": "1985-06-15",
            "Address": "101 Main Street, Toronto, ON M5V 1A1",
            "Email": "grace.smith@example.org",
        },
        {
            "First Name": "Helen",
            "Last Name": "Zhang",
            "Date of Birth": "1999-11-02",
            "Address": "456 Lakeview Ave, Calgary, AB T2P 2Z2",
            "Email": "helen.zhang@example.com",
        },
    ]
    # extend deterministically for timing volumes
    i = 3
    while len(rows) < n:
        rows.append({
            "First Name": "Liam" if i % 2 else "Noah",
            "Last Name": f"Park{i % 7}",
            "Date of Birth": f"19{80 + (i % 15)}-0{(i % 9) + 1}-15",
            "Address": f"{200 + i} Oak Street, Montreal, QC H2X 1Y4",
            "Email": f"liam.park{i % 7}@example{i}.com",
        })
        i += 1
    return rows


def row_text(row: dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in row.items())


def embed_texts(
    device: str,
    texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Load MODEL_ID with the OpenVINO backend and embed ``texts`` on ``device``.

    Returns ``(embeddings, encode_seconds)``.

    Workaround for sentence-transformers 5.6: the top-level container calls
    ``nn.Module.to(device)`` with the OpenVINO device string (e.g. "GPU", "CPU"),
    which torch rejects for any device string it doesn't recognise (it only
    accepts lowercase torch devices like "cpu", "cuda", ...). We therefore load
    with the torch-parseable device ``"cpu"`` and then move the OpenVINO
    sub-model to the requested device (``model[0].auto_model.to(device)``).
    ``encode`` then runs through OpenVINO on that device; the container's own
    ``.to()`` is only ever called with the torch device ("cpu"), which works.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, backend="openvino", device="cpu")
    if device.upper() != "CPU":
        model[0].auto_model.to(device)
    start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    elapsed = time.perf_counter() - start
    return np.asarray(embeddings, dtype="float32"), elapsed


def cosine_neighbourhood(embeddings: np.ndarray, texts: list[str]) -> Optional[str]:
    """Ordering check: row0 vs near-duplicate row1 should be closer than vs row2.

    Rows 0 and 1 share the identity (Grace Smith, DOB 1985-06-15) and differ only
    in the address number / email domain; row 2 is a dissimilar person.
    """
    if len(texts) < 3 or embeddings.ndim != 2 or embeddings.shape[1] != 384:
        return None
    v0, v1, v2 = embeddings[0], embeddings[1], embeddings[2]
    n0, n1, n2 = (np.linalg.norm(x) for x in (v0, v1, v2))
    s01 = float(v0 @ v1 / (n0 * n1))
    s02 = float(v0 @ v2 / (n0 * n2))
    return (
        f"cos(q,row0)={s01:.4f} cos(q,row2)={s02:.4f} "
        f"-> near-duplicate {'closer' if s01 >= s02 else 'FARTHER'} than dissimilar"
    )


def probe(device: str, texts: list[str], batch_size: int) -> dict[str, Any]:
    entry: dict[str, Any] = {"device": device}
    try:
        embeddings, elapsed = embed_texts(device, texts, batch_size)
        entry.update(
            {
                "n": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]),
                "encode_seconds": round(elapsed, 3),
                "ms_per_row": round(1000.0 * elapsed / max(1, len(texts)), 2),
                "sanity": cosine_neighbourhood(embeddings, texts),
            }
        )
        entry["ok"] = embeddings.shape == (len(texts), 384)
    except Exception as exc:  # noqa: BLE001
        entry["ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe OpenVINO on-device embedding for all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--device",
        nargs="+",
        default=None,
        help="OpenVINO device ids to try (default: GPU, CPU, AUTO if present).",
    )
    parser.add_argument("--rows", type=int, default=32,
                        help="Number of sample rows to embed for timing")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    version, devices = openvino_available()
    print("OpenVINO probe: all-MiniLM-L6-v2 via sentence-transformers backend=openvino")
    print("=" * 78)
    if version is None:
        print(
            f"FATAL: the `openvino` package is not installed in this interpreter "
            f"({sys.executable}).\n"
            'Install it with:\n'
            '  python -m pip install "optimum-intel[openvino]"\n'
            "Then re-run this probe."
        )
        sys.exit(1)
    print(f"openvino version : {version}")
    print(f"visible devices  : {', '.join(devices) if devices else '(none reported)'}")

    wanted = args.device or [d for d in ("GPU", "CPU", "AUTO") if d in devices]
    wanted = wanted or devices
    if not wanted:
        print("No OpenVINO devices to test.")
        sys.exit(1)

    rows = build_sample_rows(args.rows)
    texts = [row_text(r) for r in rows]
    print(f"sample rows      : {len(texts)} (person-like, 384-d expected)")

    results = []
    for device in dict.fromkeys(wanted):  # preserve order, de-dup
        print(f"\n-- device {device!r} --")
        result = probe(device, texts, args.batch_size)
        results.append(result)
        for key, value in result.items():
            if key == "sanity":
                print(f"    sanity       : {value}")
            elif key == "error":
                print(f"    ERROR        : {value}")
            elif key == "ok":
                print(f"    VERDICT      : {'OK' if value else 'FAILED'}")
            else:
                print(f"    {key:<13}: {value}")

    passed = any(r.get("ok") for r in results)
    print("\n" + "=" * 78)
    if passed:
        print("RESULT: OpenVINO is usable for these tests on device(s): "
              f"{', '.join(r['device'] for r in results if r.get('ok'))}.")
        print("The pipeline's embedding construction can be pointed at "
              "backend='openvino' (entity_pipeline.HuggingFaceEmbeddingModel) to run "
              "the experiments without PyTorch inference.")
    else:
        print("RESULT: no OpenVINO device produced a valid 384-d embedding.")
        print("Check device availability (openvino.Core().available_devices), the "
              "installation, and the error messages above.")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()