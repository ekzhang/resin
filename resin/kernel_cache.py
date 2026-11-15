"""Content-validated cache records for runtime-built kernel libraries.

The cache never consults mtimes.  A record is usable only when its ABI,
logical identity, relative path, byte count, and BLAKE3 digest match the
artifact currently on disk.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

from blake3 import blake3

_SCHEMA_VERSION = 2
_DEFAULT_RECORD_SUFFIX = ".resin-cache.json"
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """The durable inputs used to decide whether an artifact is reusable."""

    schema: int
    abi: str
    identity: str
    artifact: str
    size: int
    blake3: str


class ArtifactCache:
    """A filesystem namespace for content-validated compiled artifacts.

    ``root`` is a storage location chosen by the host runtime. ``abi`` is one
    safe path component describing every compatibility input the caller cares
    about, such as the GPU architecture and compiler/library versions.  Resin
    does not prescribe either value or know how the storage is mounted.

    A compiled artifact can either remain at the backend's build path or be
    atomically copied from local scratch into this namespace.  In both cases a
    small readiness record is published next to it only after the finished
    bytes have been hashed.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        abi: str,
        *,
        record_suffix: str = _DEFAULT_RECORD_SUFFIX,
    ) -> None:
        if not abi or Path(abi).parts != (abi,) or abi in {".", ".."}:
            raise ValueError("abi must be one non-empty path component")
        if not record_suffix or "/" in record_suffix or os.sep in record_suffix:
            raise ValueError("record_suffix must not contain a path separator")

        self._storage_root = Path(root).expanduser().resolve(strict=False)
        self._abi = abi
        self._root = (self._storage_root / abi).resolve(strict=False)
        self._record_suffix = record_suffix
        self._hits = 0
        self._misses = 0
        self._stats_lock = threading.Lock()

    @property
    def abi(self) -> str:
        return self._abi

    @property
    def root(self) -> Path:
        """The ABI-scoped directory kernel backends should write beneath."""

        return self._root

    @property
    def hits(self) -> int:
        """Number of validated loads performed through load/build methods."""

        with self._stats_lock:
            return self._hits

    @property
    def misses(self) -> int:
        """Number of build fallbacks performed through load/build methods."""

        with self._stats_lock:
            return self._misses

    def record_path(self, artifact: str | os.PathLike[str]) -> Path:
        """Return the sidecar path for an artifact inside this ABI namespace."""

        artifact_path, _ = self._artifact_path(artifact)
        return artifact_path.with_name(f"{artifact_path.name}{self._record_suffix}")

    def validate(self, artifact: str | os.PathLike[str], *, identity: str) -> bool:
        """Return whether a sidecar exactly describes the artifact's bytes."""

        self._check_identity(identity)
        try:
            artifact_path, _ = self._artifact_path(artifact)
            record_path = artifact_path.with_name(
                f"{artifact_path.name}{self._record_suffix}"
            )
            if not artifact_path.is_file() or not record_path.is_file():
                return False
            raw_record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(raw_record, dict):
                return False
            size, digest = _fingerprint(artifact_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False

        return raw_record == asdict(self._record(artifact_path, identity, size, digest))

    def publish(
        self, artifact: str | os.PathLike[str], *, identity: str
    ) -> ArtifactRecord:
        """Hash an artifact and atomically publish its validation record."""

        self._check_identity(identity)
        artifact_path, _ = self._artifact_path(artifact)
        return self._publish_record(
            artifact_path, identity, _fingerprint(artifact_path)
        )

    def publish_from(
        self,
        source: str | os.PathLike[str],
        artifact: str | os.PathLike[str],
        *,
        identity: str,
    ) -> ArtifactRecord:
        """Atomically copy a finished build into the cache and mark it ready.

        ``source`` may live on a compiler-friendly local filesystem while
        ``artifact`` lives in the durable ABI namespace.  The destination is
        replaced only after the complete source has been copied, flushed, and
        fingerprinted.  Its validation record is published last, so a reader
        can observe a miss during publication but can never accept a partial
        artifact.

        If source and destination resolve to the same path, no copy is made and
        this method has the same behavior as :meth:`publish`.
        """

        self._check_identity(identity)
        artifact_path, _ = self._artifact_path(artifact)
        source_path = Path(source).expanduser().resolve(strict=False)
        fingerprint = (
            _fingerprint(artifact_path)
            if source_path == artifact_path
            else _atomic_copy_artifact(source_path, artifact_path)
        )
        return self._publish_record(artifact_path, identity, fingerprint)

    def _record(
        self, artifact: Path, identity: str, size: int, digest: str
    ) -> ArtifactRecord:
        return ArtifactRecord(
            schema=_SCHEMA_VERSION,
            abi=self._abi,
            identity=identity,
            artifact=artifact.relative_to(self._root).as_posix(),
            size=size,
            blake3=digest,
        )

    def _publish_record(
        self, artifact: Path, identity: str, fingerprint: tuple[int, str]
    ) -> ArtifactRecord:
        record = self._record(artifact, identity, *fingerprint)
        _atomic_write_json(
            artifact.with_name(f"{artifact.name}{self._record_suffix}"), asdict(record)
        )
        return record

    def load_or_build(
        self,
        artifact: str | os.PathLike[str],
        *,
        identity: str,
        load: Callable[[Path], _T],
        build: Callable[[], _T],
    ) -> _T:
        """Load a validated artifact, or build and record it on any miss."""

        artifact_path, _ = self._artifact_path(artifact)
        return self.load_or_build_from(
            artifact_path,
            source=artifact_path,
            identity=identity,
            load=load,
            build=build,
        )

    def load_or_build_from(
        self,
        artifact: str | os.PathLike[str],
        *,
        source: str | os.PathLike[str],
        identity: str,
        load: Callable[[Path], _T],
        build: Callable[[], _T],
    ) -> _T:
        """Load a durable artifact, or build locally and publish from source.

        ``build`` must finish ``source`` before returning.  A failed build or
        publication is propagated and never creates a new readiness record.
        Miss accounting includes failed build and publication attempts, which
        matches :meth:`load_or_build`'s historical behavior.
        """

        artifact_path, _ = self._artifact_path(artifact)
        if self.validate(artifact_path, identity=identity):
            with self._stats_lock:
                self._hits += 1
            return load(artifact_path)

        with self._stats_lock:
            self._misses += 1
        result = build()
        self.publish_from(source, artifact_path, identity=identity)
        return result

    def _artifact_path(self, artifact: str | os.PathLike[str]) -> tuple[Path, Path]:
        candidate = Path(artifact).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(self._root)
        except ValueError as error:
            raise ValueError(
                f"artifact escaped ABI cache namespace {self._root}: {resolved}"
            ) from error
        if relative == Path("."):
            raise ValueError("artifact must name a file below the ABI cache namespace")
        return resolved, relative

    @staticmethod
    def _check_identity(identity: str) -> None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("identity must be a non-empty string")


def wrap_flashinfer_build_and_load(
    original: Callable[..., _T],
    cache: ArtifactCache,
    *,
    durable_artifact_resolver: Callable[[Any], str | os.PathLike[str]] | None = None,
) -> Callable[..., _T]:
    """Wrap ``FlashInfer JitSpec.build_and_load`` with hash-based reuse.

    The returned function has no import-time dependency on FlashInfer.  The
    caller installs it on the desired ``JitSpec`` class, keeping version and
    process-wide monkey-patching policy outside the generic cache layer.

    By default, FlashInfer's ``jit_library_path`` is both the build output and
    durable artifact, preserving the original API.  A runtime can instead pass
    ``durable_artifact_resolver`` to map a spec to a cache-relative or
    cache-contained absolute path.  FlashInfer then builds and initially loads
    from its local path on a miss; Resin atomically publishes the finished
    library into the durable cache for subsequent direct loads.
    """

    if (
        getattr(original, "_resin_artifact_cache", None) is cache
        and getattr(original, "_resin_durable_artifact_resolver", None)
        is durable_artifact_resolver
    ):
        return original

    @wraps(original)
    def build_and_load(spec: Any, *args: Any, **kwargs: Any) -> _T:
        if spec.is_aot:
            return original(spec, *args, **kwargs)
        build_artifact = spec.jit_library_path
        if durable_artifact_resolver is None:
            return cache.load_or_build(
                build_artifact,
                identity=f"flashinfer:{spec.name}",
                load=spec.load,
                build=lambda: original(spec, *args, **kwargs),
            )
        durable_artifact = durable_artifact_resolver(spec)
        return cache.load_or_build_from(
            durable_artifact,
            source=build_artifact,
            identity=f"flashinfer:{spec.name}",
            load=spec.load,
            build=lambda: original(spec, *args, **kwargs),
        )

    setattr(build_and_load, "_resin_artifact_cache", cache)
    setattr(
        build_and_load, "_resin_durable_artifact_resolver", durable_artifact_resolver
    )
    return build_and_load


def _fingerprint(path: Path) -> tuple[int, str]:
    digest = blake3()
    size = 0
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _atomic_copy_artifact(source: Path, destination: Path) -> tuple[int, str]:
    """Copy and fingerprint ``source``, then atomically replace destination."""

    # Open the source before creating anything at the destination.  A missing,
    # unreadable, or non-file build output therefore cannot disturb an existing
    # durable artifact or readiness record.
    with source.open("rb") as source_file:
        source_mode = stat.S_IMODE(os.fstat(source_file.fileno()).st_mode)
        digest = blake3()
        size = 0
        with _atomic_file(destination) as temporary:
            os.fchmod(temporary.fileno(), source_mode)
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                temporary.write(chunk)
                size += len(chunk)
                digest.update(chunk)

    return size, digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with _atomic_file(path) as temporary:
        temporary.write(payload)


@contextmanager
def _atomic_file(path: Path) -> Generator[BinaryIO, None, None]:
    """Flush a complete temporary file before atomically publishing its name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            yield temporary
            temporary.flush()
            _fsync_if_supported(temporary.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_FSYNC_ERRNOS:
            return
        raise
    try:
        _fsync_if_supported(descriptor)
    finally:
        os.close(descriptor)


_UNSUPPORTED_FSYNC_ERRNOS = {
    errno.EBADF,
    errno.EINVAL,
    errno.ENOSYS,
    errno.ENOTSUP,
    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
}


def _fsync_if_supported(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
