from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from agent_core.memory.models_manager import MemoryModelManager, ModelInstallError


def _bundle(tmp_path: Path, *, corrupt_checksum: bool = False) -> Path:
    source = tmp_path / "source"
    (source / "embedding").mkdir(parents=True)
    (source / "reranker").mkdir(parents=True)
    files = {
        "embedding/model.onnx": b"embedding-model",
        "embedding/tokenizer.json": b"{}",
        "reranker/model.onnx": b"reranker-model",
        "reranker/tokenizer.json": b"{}",
        "golden.json": json.dumps(
            {
                "embedding": {"texts": ["one"]},
                "reranker": {"query": "one", "passages": ["two"]},
            }
        ).encode(),
    }
    for relative, content in files.items():
        path = source / relative
        path.write_bytes(content)
    checksums = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in files.items()
    }
    if corrupt_checksum:
        checksums["embedding/model.onnx"] = "0" * 64
    manifest = {
        "schema": 1,
        "bundle_id": "test-bge-int8-v1",
        "trust_remote_code": False,
        "files": checksums,
        "models": {
            "embedding": {
                "fingerprint": "embedding-test-v1",
                "model": "embedding/model.onnx",
                "tokenizer": "embedding/tokenizer.json",
                "upstream_commit": "a" * 40,
                "license": "MIT",
                "dimension": 2,
            },
            "reranker": {
                "fingerprint": "reranker-test-v1",
                "model": "reranker/model.onnx",
                "tokenizer": "reranker/tokenizer.json",
                "upstream_commit": "b" * 40,
                "license": "Apache-2.0",
            },
        },
        "quantization": {"format": "INT8", "tool": "onnxruntime"},
        "golden_vectors": "golden.json",
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = tmp_path / "models.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    return bundle


def _locked_distribution(tmp_path: Path) -> Path:
    sources = tmp_path / "locked-sources"
    sources.mkdir()
    payloads = {
        "embedding/model.onnx": b"embedding-model",
        "embedding/tokenizer.json": b"embedding-tokenizer",
        "reranker/model.onnx": b"reranker-model",
        "reranker/tokenizer.json": b"reranker-tokenizer",
    }
    entries = []
    for relative, payload in payloads.items():
        source = sources / relative.replace("/", "-")
        source.write_bytes(payload)
        entries.append(
            {
                "path": relative,
                "url": source.as_uri(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    manifest = {
        "schema": 1,
        "bundle_id": "locked-test-v1",
        "sources": entries,
        "manifest": {
            "schema": 1,
            "bundle_id": "locked-test-v1",
            "trust_remote_code": False,
            "models": {
                "embedding": {
                    "fingerprint": "embedding-locked-v1",
                    "model": "embedding/model.onnx",
                    "tokenizer": "embedding/tokenizer.json",
                    "upstream_commit": "a" * 40,
                    "license": "MIT",
                    "dimension": 2,
                },
                "reranker": {
                    "fingerprint": "reranker-locked-v1",
                    "model": "reranker/model.onnx",
                    "tokenizer": "reranker/tokenizer.json",
                    "upstream_commit": "b" * 40,
                    "license": "Apache-2.0",
                },
            },
            "quantization": {"format": "INT8", "method": "dynamic"},
            "golden_vectors": "golden.json",
        },
        "golden": {
            "embedding": {"texts": ["one"], "dimension": 2},
            "reranker": {"query": "one", "passages": ["two"]},
        },
    }
    path = tmp_path / "distribution.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_offline_model_bundle_is_verified_and_atomically_activated(tmp_path: Path) -> None:
    manager = MemoryModelManager(tmp_path / "models")
    status = manager.install(bundle=_bundle(tmp_path))
    assert status.valid
    assert status.embedding_fingerprint == "embedding-test-v1"
    active = manager.active_manifest()
    assert active is not None
    root, _ = active
    assert (root / ".polaris-owned").is_file()

    (root / "embedding/model.onnx").write_bytes(b"tampered")
    damaged = manager.status()
    assert not damaged.valid
    assert "checksum failed" in damaged.detail


def test_model_bundle_checksum_failure_is_non_destructive(tmp_path: Path) -> None:
    manager = MemoryModelManager(tmp_path / "models")
    with pytest.raises(ModelInstallError, match="checksum failed"):
        manager.install(bundle=_bundle(tmp_path, corrupt_checksum=True))
    assert not manager.active_path.exists()


def test_model_bundle_rejects_path_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "escape.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escape", "bad")
    manager = MemoryModelManager(tmp_path / "models")
    with pytest.raises(ModelInstallError, match="unsafe model-bundle path"):
        manager.install(bundle=bundle)
    assert not (tmp_path / "escape").exists()


def test_standard_install_fetches_locked_sources_and_activates(tmp_path: Path) -> None:
    manager = MemoryModelManager(
        tmp_path / "models",
        distribution_manifest=_locked_distribution(tmp_path),
    )
    status = manager.install()
    assert status.valid
    assert status.bundle_id == "locked-test-v1"
    active = manager.active_manifest()
    assert active is not None
    root, manifest = active
    assert (root / "golden.json").is_file()
    assert manifest["files"]["embedding/model.onnx"] == hashlib.sha256(
        b"embedding-model"
    ).hexdigest()
    assert not (manager.root / ".downloads").exists()
    assert manager.install().valid


def test_bundled_distribution_uses_pinned_https_sources() -> None:
    manager = MemoryModelManager()
    distribution = manager._distribution()
    assert distribution["sources"]
    for source in distribution["sources"]:
        assert source["url"].startswith("https://")
        assert "/resolve/main/" not in source["url"]
        assert len(source["sha256"]) == 64
        assert source["size"] > 1_000_000
    assert distribution["manifest"]["trust_remote_code"] is False


def test_golden_inference_checks_dimension_norm_order_and_margin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_core.memory import runtime

    golden = {
        "embedding": {
            "texts": ["English", "ä¸­æ–‡"],
            "dimension": 2,
            "unit_norm": True,
            "tolerance": 0.0001,
        },
        "reranker": {
            "query": "query",
            "passages": ["relevant", "irrelevant"],
            "expected_order": [0, 1],
            "minimum_margin": 0.1,
        },
    }
    (tmp_path / "golden.json").write_text(json.dumps(golden), encoding="utf-8")

    class Manager:
        def active_manifest(self, *, verify=True):
            return tmp_path, {"golden_vectors": "golden.json"}

    class Embedding:
        fingerprint = "embedding"
        dimension = 2

        def embed(self, texts, *, deadline=None):
            return [[1.0, 0.0] for _ in texts]

    class Reranker:
        fingerprint = "reranker"

        def rerank(self, query, passages, *, deadline=None):
            return [2.0, -1.0]

    monkeypatch.setattr(
        runtime,
        "load_installed_backends",
        lambda **kwargs: (Embedding(), Reranker()),
    )
    assert runtime.golden_inference_check(manager=Manager()) == (
        True,
        "checksums and golden inference passed",
    )
