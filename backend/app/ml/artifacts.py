"""Versioned model artifacts (PC-F14) with a hash check before unpickling (PC-N12).

Layout in MODEL_DIR:
    models-<version>.pkl         the pickled bundle
    models-<version>.pkl.sha256  its sha256
    metrics.json                 metrics of that run (carries version + sha256)
    current.json                 manifest: which version the API should load

A pickle runs code when loaded, so the API only unpickles a file whose sha256
matches the manifest. Optional S3 sync is behind MODEL_S3_URI and is skipped when
boto3 or credentials are missing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST = "current.json"


class ArtifactError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(model_dir: Path, version: str, bundle: dict, metrics: dict) -> dict:
    """Write models-<version>.pkl, its .sha256, metrics.json and the manifest."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(bundle)
    digest = sha256_bytes(data)
    name = f"models-{version}.pkl"
    (model_dir / name).write_bytes(data)
    (model_dir / f"{name}.sha256").write_text(f"{digest}  {name}\n")
    metrics = {**metrics, "artifact": {"file": name, "sha256": digest, "version": version}}
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (model_dir / f"metrics-{version}.json").write_text(json.dumps(metrics, indent=2, default=str))
    manifest = {"version": version, "file": name, "sha256": digest, "metrics": f"metrics-{version}.json"}
    (model_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(model_dir: Path, version: str | None = None) -> dict | None:
    model_dir = Path(model_dir)
    if version:
        name = f"models-{version}.pkl"
        sidecar = model_dir / f"{name}.sha256"
        if not sidecar.exists():
            return None
        return {"version": version, "file": name, "sha256": sidecar.read_text().split()[0],
                "metrics": f"metrics-{version}.json"}
    path = model_dir / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


# What it does: loads a model pickle only after its sha256 matches the manifest and the sidecar.
# Catch it: "pickle.load runs code. How do you make sure the API only loads the model you trained?"
# Card: docs/cards/verified-artifact.md
def load_bundle(model_dir: Path, version: str | None = None) -> tuple[dict, dict, dict]:
    """Return (bundle, metrics, manifest). Raise ArtifactError on any mismatch."""
    model_dir = Path(model_dir)
    # STEP: the manifest names the version, file and expected hash
    manifest = read_manifest(model_dir, version)
    # DECISION: no manifest or no file means no model, never a guess
    if not manifest:
        raise ArtifactError(f"no model manifest in {model_dir}")
    path = model_dir / manifest["file"]
    if not path.exists():
        raise ArtifactError(f"model file missing: {manifest['file']}")
    # STEP: hash the bytes we are about to load, not a path we might re-read
    data = path.read_bytes()
    actual = sha256_bytes(data)
    sidecar = model_dir / f"{manifest['file']}.sha256"
    expected = {manifest["sha256"]}
    if sidecar.exists():
        expected.add(sidecar.read_text().split()[0])
    # DECISION: manifest and sidecar must both agree with the bytes
    if expected != {actual}:
        raise ArtifactError(f"sha256 mismatch for {manifest['file']}: refusing to load")
    # STEP: only now unpickle those same bytes
    bundle = pickle.loads(data)  # noqa: S301 - hash verified above
    metrics_path = model_dir / manifest.get("metrics", "metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    # RETURN: bundle plus the metrics and manifest it was trained with
    return bundle, metrics, manifest


# --- optional S3 sync ---

def _s3():
    uri = os.getenv("MODEL_S3_URI", "")
    if not uri.startswith("s3://"):
        return None, None, None
    try:
        import boto3  # lazy: only needed when MODEL_S3_URI is set
    except ImportError:
        logger.info("MODEL_S3_URI set but boto3 not installed; skipping S3")
        return None, None, None
    bucket, _, prefix = uri[5:].partition("/")
    return boto3.client("s3"), bucket, prefix.strip("/")


def upload(model_dir: Path, manifest: dict) -> bool:
    client, bucket, prefix = _s3()
    if not client:
        return False
    files = [manifest["file"], f"{manifest['file']}.sha256", manifest["metrics"], MANIFEST]
    try:
        for name in files:
            client.upload_file(str(Path(model_dir) / name), bucket, f"{prefix}/{name}" if prefix else name)
    except Exception as e:  # missing credentials, denied, network
        logger.warning("S3 upload skipped (%s)", type(e).__name__)
        return False
    logger.info("Uploaded %s to s3://%s/%s", manifest["file"], bucket, prefix)
    return True


def download(model_dir: Path) -> bool:
    """Fetch the manifest and its files from S3. The hash is still checked on load."""
    client, bucket, prefix = _s3()
    if not client:
        return False
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def key(name):
        return f"{prefix}/{name}" if prefix else name

    try:
        client.download_file(bucket, key(MANIFEST), str(model_dir / MANIFEST))
        manifest = json.loads((model_dir / MANIFEST).read_text())
        for name in (manifest["file"], f"{manifest['file']}.sha256", manifest["metrics"]):
            client.download_file(bucket, key(name), str(model_dir / name))
    except Exception as e:
        logger.warning("S3 download skipped (%s)", type(e).__name__)
        return False
    return True
