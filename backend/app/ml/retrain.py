"""Background retraining with a verified hot-swap (POST /ops/retrain).

One run at a time per process: `start()` returns False while a run is going, which
the route turns into a 409. A run trains with the existing pipeline (the same code as
`python -m app.ml.train`), then loads the new artifact back through the hash check.
Only a bundle that verifies, and whose hash equals the one training computed in
memory, replaces the loaded models. Otherwise the previous manifest and metrics.json
are restored, so a restart still loads the old model.

The result of every run is kept in memory and in retrain-status.json in MODEL_DIR
(the Railway volume), so GET /ops/retrain survives a restart. A run cut off by a
restart is reported as failed.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from app.ml import artifacts

logger = logging.getLogger(__name__)

STATUS_FILE = "retrain-status.json"
RESTORE = (artifacts.MANIFEST, "metrics.json")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Retrainer:
    def __init__(self, model_dir: Path | str, train_fn: Callable[[], dict],
                 apply_fn: Callable[[dict, dict, dict], None], current_version: Callable[[], str | None],
                 pinned_version: str | None = None):
        self.model_dir = Path(model_dir)
        self.train_fn = train_fn
        self.apply_fn = apply_fn
        self.current_version = current_version
        self.pinned_version = pinned_version or None
        self._lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self._status = self._load_status()

    def _load_status(self) -> dict:
        try:
            status = json.loads((self.model_dir / STATUS_FILE).read_text())
        except (OSError, ValueError):
            return {"state": "idle"}
        if status.get("state") == "running":
            status.update(state="failed", error="interrupted by a restart", finished_at=None)
        return status

    def _save_status(self):
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            (self.model_dir / STATUS_FILE).write_text(json.dumps(self._status, indent=2))
        except OSError as e:
            logger.warning("Could not write %s (%s)", STATUS_FILE, type(e).__name__)

    def status(self) -> dict:
        return {**self._status, "running": self._lock.locked()}

    def start(self, trigger: str = "api") -> dict | None:
        """Start a run in a background thread, or return None if one is running."""
        if not self._lock.acquire(blocking=False):
            return None
        self._status = {"state": "running", "run_id": uuid.uuid4().hex[:12], "trigger": trigger,
                        "started_at": _now(), "finished_at": None, "previous_version": self.current_version(),
                        "new_version": None, "verified": None, "swapped": False, "error": None}
        self._save_status()
        self.thread = threading.Thread(target=self._run, name="retrain", daemon=True)
        self.thread.start()
        return dict(self._status)

    def _snapshot(self) -> dict[str, bytes]:
        return {n: (self.model_dir / n).read_bytes() for n in RESTORE if (self.model_dir / n).exists()}

    def _restore(self, saved: dict[str, bytes]):
        for name in RESTORE:
            path = self.model_dir / name
            if name in saved:
                path.write_bytes(saved[name])
            elif path.exists():
                path.unlink()

    def _verify(self, metrics: dict) -> tuple[dict, dict, dict]:
        """Load the new artifact the way the API does; refuse anything that does not match."""
        expected = metrics.get("artifact") or {}
        bundle, loaded_metrics, manifest = artifacts.load_bundle(self.model_dir)
        if manifest.get("version") != expected.get("version") or manifest.get("sha256") != expected.get("sha256"):
            raise artifacts.ArtifactError("manifest does not name the artifact this run trained")
        if not bundle.get("models"):
            raise artifacts.ArtifactError("the new bundle has no trained targets")
        return bundle, loaded_metrics, manifest

    def _run(self):
        saved = self._snapshot()
        result: dict = {}
        try:
            metrics = self.train_fn()
            result["new_version"] = (metrics.get("artifact") or {}).get("version")
            try:
                bundle, loaded_metrics, manifest = self._verify(metrics)
            except artifacts.ArtifactError as e:
                self._restore(saved)
                result.update(state="failed", verified=False, error=f"verification failed: {e}")
            else:
                result.update(verified=True, sha256=manifest["sha256"],
                              override_feedback=metrics.get("override_feedback"))
                if self.pinned_version:
                    result.update(state="succeeded", swapped=False,
                                  note=f"MODEL_VERSION pins {self.pinned_version}; not swapped")
                else:
                    self.apply_fn(bundle, loaded_metrics, manifest)
                    result.update(state="succeeded", swapped=True)
                    artifacts.upload(self.model_dir, manifest)  # no-op unless MODEL_S3_URI is set
        except Exception as e:  # training itself failed
            logger.exception("Retrain failed")
            self._restore(saved)
            result.update(state="failed", verified=False, error=f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            self._status = {**self._status, **result, "finished_at": _now()}
            self._save_status()
            logger.info("Retrain %s: %s (%s -> %s)", self._status["run_id"], self._status["state"],
                        self._status["previous_version"], self._status.get("new_version"))
            self._lock.release()
