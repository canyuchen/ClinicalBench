"""Download the released result files from the Hugging Face Hub.

The results are not in git -- they are 295 MB of per-patient model output, which
would make every clone expensive -- so they live in a gated dataset repository
and land in ``results/`` on demand.

::

    clinicalbench-fetch-results
    python -m clinicalbench.eval.fetch --dest results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_REPO = "canyuchen/clinicalbench-results"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}"


def fetch(dest: Path, summary_only: bool = False, token: str | None = None) -> Path:
    """Download the dataset into ``dest``'s parent, so files land in ``dest``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "  pip install huggingface_hub\n"
            f"or download by hand from {DATASET_URL}"
        )

    # The repo stores files under `results/`, so downloading into dest's parent
    # puts them exactly at dest.
    local_dir = dest.parent if dest.name == "results" else dest
    patterns = ["summary.csv"] if summary_only else ["results/*", "summary.csv"]

    print(f"downloading {DATASET_REPO} -> {local_dir}/")
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=token,
    )
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path("results"),
                        help="where the result files should end up (default: results/)")
    parser.add_argument("--summary-only", action="store_true",
                        help="fetch just summary.csv, the precomputed metrics")
    parser.add_argument("--token", default=None,
                        help="Hugging Face token; falls back to a cached login")
    args = parser.parse_args()

    try:
        local_dir = fetch(args.dest, args.summary_only, args.token)
    except Exception as exc:  # noqa: BLE001 -- the message matters more than the type
        print(
            f"download failed: {exc}\n\n"
            f"The dataset is gated. Accept the terms once at\n"
            f"  {DATASET_URL}\n"
            f"then log in with `hf auth login` (or pass --token).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    n = len(list((local_dir / "results").rglob("*.csv"))) if not args.summary_only else 0
    if n:
        print(f"done: {n} result files under {local_dir / 'results'}/")
    else:
        print(f"done: {local_dir}/summary.csv")


if __name__ == "__main__":
    main()
