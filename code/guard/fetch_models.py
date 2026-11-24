"""Explicitly download the pinned model snapshots into this bundle's models directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = Path(__file__).with_name("runtime_provenance.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir: Path, expected: dict[str, str]) -> None:
    for name, expected_digest in expected.items():
        path = model_dir / name
        actual = sha256_file(path)
        if actual != expected_digest:
            raise ValueError(f"SHA-256 mismatch: {path.name}")
        print(f"Verified {path.relative_to(BUNDLE_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify existing bundle weights without network access.")
    args = parser.parse_args()
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    models = BUNDLE_ROOT / "models"
    guard_dir = models / "jev-guard-v3.2-2b"
    base_dir = models / "qwen3.5-2b"
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        print("Downloading pinned model snapshots into the bundle; base weights are about 4.55 GB.")
        snapshot_download(
            repo_id=provenance["guard_model"]["repo_id"],
            revision=provenance["guard_model"]["revision"],
            local_dir=guard_dir,
            cache_dir=models / ".download-cache",
            allow_patterns=["checkpoint/**", "jev_inference.py", "serve.py", "requirements.txt",
                            "README.md", "README.zh-CN.md", "LICENSE-NOTICE.md", "SHA256SUMS"],
        )
        snapshot_download(
            repo_id=provenance["base_model"]["repo_id"],
            revision=provenance["base_model"]["revision"],
            local_dir=base_dir,
            cache_dir=models / ".download-cache",
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE", "README.md"],
        )
    guard_meta = provenance["guard_model"]
    verify(guard_dir, {
        "checkpoint/adapter/adapter_model.safetensors": guard_meta["adapter_sha256"],
        "checkpoint/heads.pt": guard_meta["heads_sha256"],
        "checkpoint/score-head.pt": guard_meta["score_head_sha256"],
        "jev_inference.py": "b03d650506926feff6a90e985f9480e0f62a1c18b9514d022fb2c032f3c54b3c",
        "serve.py": "e3e2f70c44f6c14bfbbdbdefbaf4414e0a8d75a770c749901ebbaac25574ca11",
    })
    verify(base_dir, {provenance["base_model"]["weights_file"]:
                      provenance["base_model"]["weights_sha256"]})
    print("Models verified. Start code/guard/serve.py using the command in code/guard/README.md.")


if __name__ == "__main__":
    main()
