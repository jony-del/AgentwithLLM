from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from agent_core.file_lock import FileLock

MODEL_BUNDLE_SCHEMA = 1
MAX_BUNDLE_FILES = 256
MAX_BUNDLE_FILE_BYTES = 3 * 1024 * 1024 * 1024
MAX_BUNDLE_TOTAL_BYTES = 6 * 1024 * 1024 * 1024
_VERIFICATION_LOCK = threading.Lock()
_VERIFIED_INSTALLATIONS: dict[str, tuple[Any, ...]] = {}


class ModelInstallError(RuntimeError):
    pass


@dataclass(slots=True)
class MemoryModelStatus:
    valid: bool
    installed: bool
    root: str
    bundle_id: str = ""
    embedding_fingerprint: str = ""
    reranker_fingerprint: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_models_root() -> Path:
    override = os.getenv("POLARIS_MEMORY_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    polaris_home = os.getenv("POLARIS_HOME")
    if polaris_home:
        return Path(polaris_home).expanduser() / "models" / "memory"
    return Path.home() / ".polaris" / "models" / "memory"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ModelInstallError(f"unsafe model-bundle path: {value!r}")
    return Path(*pure.parts)


class MemoryModelManager:
    """Checksummed, atomic installation of an offline CPU ONNX model bundle."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        distribution_manifest: str | Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_models_root()
        self.active_path = self.root / "active.json"
        self.skip_marker = self.root / ".explicitly-skipped"
        self.distribution_manifest = (
            Path(distribution_manifest)
            if distribution_manifest is not None
            else Path(__file__).with_name("model_bundle.json")
        )

    def _distribution(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.distribution_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelInstallError("memory model distribution manifest is unavailable") from exc
        if not isinstance(loaded, dict):
            raise ModelInstallError("memory model distribution manifest is malformed")
        return loaded

    def active_manifest(self, *, verify: bool = True) -> tuple[Path, dict[str, Any]] | None:
        try:
            pointer = json.loads(self.active_path.read_text(encoding="utf-8"))
            bundle_id = str(pointer["bundle_id"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None
        root = self.root / bundle_id
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None
        if verify:
            self._validate_installation_cached(root, manifest)
        return root, manifest

    def status(self) -> MemoryModelStatus:
        try:
            active = self.active_manifest(verify=True)
        except ModelInstallError as exc:
            return MemoryModelStatus(
                False,
                True,
                str(self.root),
                detail=str(exc),
            )
        if active is None:
            return MemoryModelStatus(
                False,
                False,
                str(self.root),
                detail=(
                    "models are not installed; run `polaris memory models install` "
                    "or use `--model-bundle PATH`"
                ),
            )
        root, manifest = active
        models = manifest.get("models", {})
        embedding = models.get("embedding", {}) if isinstance(models, dict) else {}
        reranker = models.get("reranker", {}) if isinstance(models, dict) else {}
        return MemoryModelStatus(
            True,
            True,
            str(root),
            bundle_id=str(manifest.get("bundle_id", root.name)),
            embedding_fingerprint=str(embedding.get("fingerprint", "")),
            reranker_fingerprint=str(reranker.get("fingerprint", "")),
            detail="checksums verified",
        )

    @property
    def explicitly_skipped(self) -> bool:
        return self.skip_marker.is_file()

    def mark_explicitly_skipped(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / ".explicitly-skipped.tmp"
        temporary.write_text(
            json.dumps({"schema": 1, "reason": "installer flag"}) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.skip_marker)

    def install(self, *, bundle: str | Path | None = None) -> MemoryModelStatus:
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(self.root / ".install.lock"):
            return self._install_locked(bundle=bundle)

    def _install_locked(self, *, bundle: str | Path | None = None) -> MemoryModelStatus:
        distribution = self._distribution()
        environment_url = os.getenv("POLARIS_MEMORY_MODEL_BUNDLE_URL")
        if bundle is None and not environment_url and distribution.get("sources"):
            return self._install_locked_sources(distribution)
        expected_bundle_sha = ""
        if bundle is None:
            url = environment_url or str(distribution.get("url", ""))
            expected_bundle_sha = os.getenv("POLARIS_MEMORY_MODEL_BUNDLE_SHA256") or str(
                distribution.get("sha256", "")
            )
            if not url or len(expected_bundle_sha) != 64:
                raise ModelInstallError(
                    "online memory model bundle metadata is incomplete; provide "
                    "`--model-bundle PATH` for an offline install"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            cache = self.root / ".downloads" / Path(urllib.parse.urlparse(url).path).name
            source = self._download_resumable(url, cache)
        else:
            source = Path(bundle).expanduser()
            if not source.is_file():
                raise ModelInstallError(f"model bundle does not exist: {source}")
            expected_bundle_sha = str(distribution.get("sha256", "")) if bool(
                distribution.get("verify_offline_bundle_against_distribution", False)
            ) else ""
        if expected_bundle_sha and _sha256(source) != expected_bundle_sha.casefold():
            if bundle is None:
                source.unlink(missing_ok=True)
            raise ModelInstallError("SHA-256 mismatch for memory model bundle")

        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".memory-models-", dir=self.root))
        try:
            self._extract(source, staging)
            self._activate_staging(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self.status()

    def _install_locked_sources(self, distribution: dict[str, Any]) -> MemoryModelStatus:
        sources = distribution.get("sources")
        manifest_template = distribution.get("manifest")
        golden = distribution.get("golden")
        if not isinstance(sources, list) or not sources:
            raise ModelInstallError("memory model distribution has no locked sources")
        if not isinstance(manifest_template, dict) or not isinstance(golden, dict):
            raise ModelInstallError("memory model distribution manifest is malformed")
        if len(sources) > MAX_BUNDLE_FILES:
            raise ModelInstallError("memory model distribution has too many files")
        expected_bundle_id = str(manifest_template.get("bundle_id", ""))
        active = self.active_manifest(verify=True)
        if active is not None and str(active[1].get("bundle_id", "")) == expected_bundle_id:
            return self.status()

        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".memory-models-", dir=self.root))
        caches: list[Path] = []
        activated = False
        try:
            checksums: dict[str, str] = {}
            total_size = 0
            for source_spec in sources:
                if not isinstance(source_spec, dict):
                    raise ModelInstallError("memory model source entry is malformed")
                relative = _safe_relative(str(source_spec.get("path", "")))
                url = str(source_spec.get("url", ""))
                expected_sha = str(source_spec.get("sha256", "")).casefold()
                try:
                    expected_size = int(source_spec.get("size", 0))
                except (TypeError, ValueError) as exc:
                    raise ModelInstallError("memory model source size is malformed") from exc
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme not in {"https", "file"}:
                    raise ModelInstallError("memory model sources must use HTTPS")
                if (
                    len(expected_sha) != 64
                    or any(character not in "0123456789abcdef" for character in expected_sha)
                ):
                    raise ModelInstallError("memory model source has an invalid SHA-256")
                if expected_size <= 0 or expected_size > MAX_BUNDLE_FILE_BYTES:
                    raise ModelInstallError("memory model source has an unsafe size")
                total_size += expected_size
                if total_size > MAX_BUNDLE_TOTAL_BYTES:
                    raise ModelInstallError("memory model distribution is too large")

                cache = self.root / ".downloads" / f"{expected_sha}.blob"
                caches.append(cache)
                if cache.is_file() and (
                    cache.stat().st_size != expected_size or _sha256(cache) != expected_sha
                ):
                    cache.unlink()
                downloaded = self._download_resumable(url, cache)
                if (
                    downloaded.stat().st_size != expected_size
                    or _sha256(downloaded) != expected_sha
                ):
                    downloaded.unlink(missing_ok=True)
                    raise ModelInstallError(
                        f"checksum failed for downloaded model file: {relative.as_posix()}"
                    )
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, target)
                checksums[relative.as_posix()] = expected_sha

            manifest = json.loads(json.dumps(manifest_template))
            golden_relative = _safe_relative(str(manifest.get("golden_vectors", "")))
            golden_bytes = (
                json.dumps(golden, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            golden_path = staging / golden_relative
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_bytes(golden_bytes)
            checksums[golden_relative.as_posix()] = hashlib.sha256(golden_bytes).hexdigest()
            manifest["files"] = checksums
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._activate_staging(staging)
            activated = True
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if activated:
                for cache in caches:
                    cache.unlink(missing_ok=True)
                try:
                    (self.root / ".downloads").rmdir()
                except OSError:
                    pass
        return self.status()

    def _activate_staging(self, staging: Path) -> None:
        manifest_path = staging / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelInstallError("model bundle is missing a valid manifest.json") from exc
        if not isinstance(manifest, dict):
            raise ModelInstallError("model bundle manifest must be an object")
        self._validate_installation(staging, manifest)
        bundle_id = str(manifest.get("bundle_id", "")).strip()
        if (
            not bundle_id
            or "/" in bundle_id
            or "\\" in bundle_id
            or _safe_relative(bundle_id).as_posix() != bundle_id
        ):
            raise ModelInstallError("model bundle has an unsafe bundle_id")
        destination = self.root / bundle_id
        if destination.exists():
            try:
                existing = json.loads(
                    (destination / "manifest.json").read_text(encoding="utf-8")
                )
                self._validate_installation(destination, existing)
            except Exception:
                raise ModelInstallError(
                    f"refusing to overwrite invalid existing model directory: {destination}"
                ) from None
            if existing != manifest:
                raise ModelInstallError(
                    f"model bundle_id collision with existing installation: {bundle_id}"
                )
            shutil.rmtree(staging)
        else:
            os.replace(staging, destination)
        (destination / ".polaris-owned").write_text(
            json.dumps({"schema": 1, "bundle_id": bundle_id}) + "\n",
            encoding="utf-8",
        )
        pointer_tmp = self.root / ".active.tmp"
        pointer_tmp.write_text(
            json.dumps({"schema": 1, "bundle_id": bundle_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(pointer_tmp, self.active_path)
        self.skip_marker.unlink(missing_ok=True)
        signature = self._installation_signature(destination, manifest)
        if signature is not None:
            with _VERIFICATION_LOCK:
                _VERIFIED_INSTALLATIONS[str(destination.resolve(strict=False))] = signature

    @staticmethod
    def _installation_signature(
        root: Path,
        manifest: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            return None
        try:
            states: list[tuple[str, int, int, str]] = []
            for raw_path, expected in sorted(files.items()):
                relative = _safe_relative(str(raw_path))
                stat = (root / relative).stat()
                states.append(
                    (
                        relative.as_posix(),
                        stat.st_size,
                        stat.st_mtime_ns,
                        str(expected).casefold(),
                    )
                )
        except (OSError, ModelInstallError):
            return None
        manifest_digest = hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return (manifest_digest, *states)

    def _validate_installation_cached(self, root: Path, manifest: dict[str, Any]) -> None:
        signature = self._installation_signature(root, manifest)
        cache_key = str(root.resolve(strict=False))
        if signature is not None:
            with _VERIFICATION_LOCK:
                if _VERIFIED_INSTALLATIONS.get(cache_key) == signature:
                    return
        self._validate_installation(root, manifest)
        signature = self._installation_signature(root, manifest)
        if signature is not None:
            with _VERIFICATION_LOCK:
                _VERIFIED_INSTALLATIONS[cache_key] = signature

    def _validate_installation(self, root: Path, manifest: dict[str, Any]) -> None:
        if int(manifest.get("schema", 0)) != MODEL_BUNDLE_SCHEMA:
            raise ModelInstallError("unsupported memory model bundle schema")
        if manifest.get("trust_remote_code") not in {False, None}:
            raise ModelInstallError("memory model bundles may not enable trust_remote_code")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ModelInstallError("model bundle manifest has no checksummed files")
        for raw_path, expected in files.items():
            relative = _safe_relative(str(raw_path))
            target = (root / relative).resolve(strict=False)
            try:
                target.relative_to(root.resolve(strict=False))
            except ValueError as exc:
                raise ModelInstallError(f"model file escapes bundle root: {raw_path}") from exc
            if not target.is_file():
                raise ModelInstallError(f"model bundle file is missing: {raw_path}")
            expected_sha = str(expected).casefold()
            if len(expected_sha) != 64 or _sha256(target) != expected_sha:
                raise ModelInstallError(f"checksum failed for model file: {raw_path}")
        checksummed = {_safe_relative(str(path)).as_posix() for path in files}
        models = manifest.get("models")
        if not isinstance(models, dict):
            raise ModelInstallError("model bundle manifest has no models section")
        for role in ("embedding", "reranker"):
            model = models.get(role)
            if not isinstance(model, dict):
                raise ModelInstallError(f"model bundle is missing {role}")
            for key in ("fingerprint", "model", "tokenizer", "upstream_commit", "license"):
                if not str(model.get(key, "")).strip():
                    raise ModelInstallError(f"{role} manifest is missing {key}")
            for key in ("model", "tokenizer"):
                relative = _safe_relative(str(model[key]))
                if not (root / relative).is_file():
                    raise ModelInstallError(f"{role} {key} is missing")
                if relative.as_posix() not in checksummed:
                    raise ModelInstallError(f"{role} {key} is not checksummed")
        if not isinstance(manifest.get("quantization"), dict):
            raise ModelInstallError("model bundle is missing quantization metadata")
        golden = manifest.get("golden_vectors")
        if not golden or not (root / _safe_relative(str(golden))).is_file():
            raise ModelInstallError("model bundle is missing golden vectors")
        if _safe_relative(str(golden)).as_posix() not in checksummed:
            raise ModelInstallError("golden vectors are not checksummed")

    def _download_resumable(self, url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            return destination
        partial = destination.with_suffix(destination.suffix + ".part")
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "polaris-memory-model-installer"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = existing > 0 and getattr(response, "status", 200) == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream)
        except (OSError, urllib.error.URLError) as exc:
            raise ModelInstallError(
                f"memory model download failed; rerun to resume or use --model-bundle PATH: {exc}"
            ) from exc
        os.replace(partial, destination)
        return destination

    def _extract(self, source: Path, destination: Path) -> None:
        if zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_BUNDLE_FILES:
                    raise ModelInstallError("model bundle contains too many files")
                total_size = sum(info.file_size for info in infos)
                if (
                    total_size > MAX_BUNDLE_TOTAL_BYTES
                    or any(info.file_size > MAX_BUNDLE_FILE_BYTES for info in infos)
                ):
                    raise ModelInstallError("model bundle is too large")
                for info in infos:
                    relative = _safe_relative(info.filename)
                    target = destination / relative
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as incoming, target.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
            return
        try:
            tar_archive = tarfile.open(source, mode="r:*")
        except tarfile.TarError as exc:
            raise ModelInstallError("unsupported memory model bundle format") from exc
        with tar_archive:
            members = tar_archive.getmembers()
            if len(members) > MAX_BUNDLE_FILES:
                raise ModelInstallError("model bundle contains too many files")
            total_size = sum(member.size for member in members)
            if (
                total_size > MAX_BUNDLE_TOTAL_BYTES
                or any(member.size > MAX_BUNDLE_FILE_BYTES for member in members)
            ):
                raise ModelInstallError("model bundle is too large")
            for member in members:
                if not member.isfile() and not member.isdir():
                    raise ModelInstallError("model bundle contains a link or special file")
                relative = _safe_relative(member.name)
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tar_incoming = tar_archive.extractfile(member)
                if tar_incoming is None:
                    raise ModelInstallError(f"could not extract model file: {member.name}")
                with tar_incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(tar_incoming, outgoing)
