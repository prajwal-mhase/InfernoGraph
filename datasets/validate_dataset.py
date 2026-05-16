from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "manifest.json"
TARGET = 2_000_000


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(
            "datasets/manifest.json is missing. Run: "
            "python datasets/build_open_access_pmc_dataset.py --target-tokens 2000000"
        )
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    token_count = int(data.get("token_count", 0))
    document_count = int(data.get("document_count", 0))
    target_met = token_count >= TARGET
    print(f"Dataset source: {data.get('source')}")
    print(f"Documents: {document_count:,}")
    print(f"Tokens: {token_count:,}")
    print(f"Target met: {target_met}")
    if not target_met:
        raise SystemExit("Dataset token target is not met.")


if __name__ == "__main__":
    main()
