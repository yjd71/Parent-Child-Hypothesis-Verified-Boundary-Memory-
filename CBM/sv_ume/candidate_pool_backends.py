from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Protocol, runtime_checkable

import torch

from CBM.sv_ume.unlabeled_dense_memory import UnlabeledMemoryToken


CandidateIdentity = Hashable
CandidateRank = tuple[float, int]


@dataclass(frozen=True)
class CandidatePoolStats:
    """Immutable snapshot of candidate-pool storage and replacement counters."""

    backend: str
    active_counts: Mapping[str, int]
    records_written: Mapping[str, int]
    replacement_counts: Mapping[str, int]
    ignored_duplicate_counts: Mapping[str, int]
    buffered_counts: Mapping[str, int]
    shard_counts: Mapping[str, int]
    cached_shards: int
    max_cached_shards: int
    storage_dir: Optional[str]
    closed: bool

    def __post_init__(self) -> None:
        for name in (
            "active_counts",
            "records_written",
            "replacement_counts",
            "ignored_duplicate_counts",
            "buffered_counts",
            "shard_counts",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, MappingProxyType(dict(value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "active_counts": dict(self.active_counts),
            "records_written": dict(self.records_written),
            "replacement_counts": dict(self.replacement_counts),
            "ignored_duplicate_counts": dict(self.ignored_duplicate_counts),
            "buffered_counts": dict(self.buffered_counts),
            "shard_counts": dict(self.shard_counts),
            "cached_shards": int(self.cached_shards),
            "max_cached_shards": int(self.max_cached_shards),
            "storage_dir": self.storage_dir,
            "closed": bool(self.closed),
        }


@runtime_checkable
class CandidatePoolProtocol(Protocol):
    def append_many(
        self,
        region: str,
        candidates: Sequence[UnlabeledMemoryToken],
    ) -> None: ...

    def iter_region(
        self,
        region: str,
        order: str = "canonical",
    ) -> Iterator[UnlabeledMemoryToken]: ...

    def iter_region_by_image(
        self,
        region: str,
    ) -> Iterator[tuple[str, list[UnlabeledMemoryToken]]]: ...

    def get_candidate(
        self,
        region: str,
        canonical_index: int,
        *,
        shard_id: Optional[int] = None,
        shard_offset: Optional[int] = None,
    ) -> UnlabeledMemoryToken: ...

    def counts(self) -> dict[str, int]: ...

    def clear(self) -> None: ...

    def flush_all(self) -> None: ...

    def close(self) -> None: ...

    def stats(self) -> CandidatePoolStats: ...


def _validated_regions(regions: Sequence[str]) -> tuple[str, ...]:
    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        raise TypeError("regions must be a sequence of region names")
    normalized = tuple(str(region) for region in regions)
    if not normalized:
        raise ValueError("regions must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("regions must not contain duplicates")
    for region in normalized:
        if not region or region in {".", ".."} or Path(region).name != region:
            raise ValueError(f"unsafe candidate-pool region name: {region!r}")
    return normalized


def _validated_rank(value: Any) -> CandidateRank:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError("candidate rank must be (reliability, step_added)")
    reliability = float(value[0])
    if not math.isfinite(reliability):
        raise ValueError("candidate rank reliability must be finite")
    raw_step = value[1]
    step_added = int(raw_step)
    if isinstance(raw_step, float) and not raw_step.is_integer():
        raise ValueError("candidate rank step_added must be an integer")
    if step_added < 0:
        raise ValueError("candidate rank step_added must be non-negative")
    return reliability, step_added


def _validated_identity(value: Any) -> CandidateIdentity:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("candidate identity must be hashable") from exc
    return value


def _validated_candidates(candidates: Sequence[UnlabeledMemoryToken]) -> None:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")


def _candidate_image_id(candidate: UnlabeledMemoryToken) -> str:
    meta = getattr(candidate, "meta", None)
    if not isinstance(meta, Mapping):
        raise TypeError("candidate.meta must be a mapping")
    if "image_id" not in meta:
        raise KeyError("candidate.meta is missing 'image_id'")
    return str(meta["image_id"])


def _candidate_token_kind(candidate: UnlabeledMemoryToken) -> int:
    if not isinstance(candidate, UnlabeledMemoryToken):
        raise TypeError("candidate must be TokenCandidate or UnlabeledMemoryToken")
    from CBM.sv_ume.sam_refined_candidate_builder import TokenCandidate

    return 1 if isinstance(candidate, TokenCandidate) else 0


def _candidate_with_location(
    candidate: UnlabeledMemoryToken,
    *,
    canonical_index: int,
    shard_id: int,
    shard_offset: int,
) -> UnlabeledMemoryToken:
    view = copy.copy(candidate)
    object.__setattr__(view, "canonical_index", int(canonical_index))
    object.__setattr__(view, "shard_id", int(shard_id))
    object.__setattr__(view, "shard_offset", int(shard_offset))
    return view


class RAMCandidatePool:
    """Exact wrapper around the existing region lists and identity replacement."""

    def __init__(
        self,
        *,
        regions: Sequence[str],
        rank_fn: Callable[[UnlabeledMemoryToken], Any],
        identity_fn: Callable[[UnlabeledMemoryToken, str], Any],
    ) -> None:
        if not callable(rank_fn) or not callable(identity_fn):
            raise TypeError("rank_fn and identity_fn must be callable")
        self.regions = _validated_regions(regions)
        self.rank_fn = rank_fn
        self.identity_fn = identity_fn
        self._closed = False
        self._pools: dict[str, list[UnlabeledMemoryToken]] = {}
        self._identity_index: dict[str, dict[CandidateIdentity, tuple[int, CandidateRank]]] = {}
        self._records_written: dict[str, int] = {}
        self._replacement_counts: dict[str, int] = {}
        self._ignored_duplicate_counts: dict[str, int] = {}
        self._reset()

    def append_many(
        self,
        region: str,
        candidates: Sequence[UnlabeledMemoryToken],
    ) -> None:
        self._ensure_open()
        self._ensure_region(region)
        _validated_candidates(candidates)
        for candidate in candidates:
            _candidate_token_kind(candidate)
            identity = _validated_identity(self.identity_fn(candidate, region))
            rank = _validated_rank(self.rank_fn(candidate))
            previous = self._identity_index[region].get(identity)
            if previous is None:
                index = len(self._pools[region])
                self._pools[region].append(candidate)
                self._identity_index[region][identity] = (index, rank)
                self._records_written[region] += 1
            elif rank > previous[1]:
                self._pools[region][previous[0]] = candidate
                self._identity_index[region][identity] = (previous[0], rank)
                self._records_written[region] += 1
                self._replacement_counts[region] += 1
            else:
                self._ignored_duplicate_counts[region] += 1

    def iter_region(
        self,
        region: str,
        order: str = "canonical",
    ) -> Iterator[UnlabeledMemoryToken]:
        self._ensure_open()
        self._ensure_region(region)
        self._ensure_canonical_order(order)
        return iter(self._pools[region])

    def iter_region_by_image(
        self,
        region: str,
    ) -> Iterator[tuple[str, list[UnlabeledMemoryToken]]]:
        self._ensure_open()
        self._ensure_region(region)
        grouped: OrderedDict[str, list[UnlabeledMemoryToken]] = OrderedDict()
        for canonical_index, candidate in enumerate(self._pools[region]):
            grouped.setdefault(_candidate_image_id(candidate), []).append(
                _candidate_with_location(
                    candidate,
                    canonical_index=canonical_index,
                    shard_id=-1,
                    shard_offset=canonical_index,
                )
            )
        return iter(grouped.items())

    def get_candidate(
        self,
        region: str,
        canonical_index: int,
        *,
        shard_id: Optional[int] = None,
        shard_offset: Optional[int] = None,
    ) -> UnlabeledMemoryToken:
        self._ensure_open()
        self._ensure_region(region)
        index = int(canonical_index)
        if index < 0 or index >= len(self._pools[region]):
            raise KeyError(
                f"unknown canonical candidate index {index} for region {region!r}"
            )
        if shard_id is not None and int(shard_id) != -1:
            raise ValueError("RAM candidate shard_id must be -1")
        if shard_offset is not None and int(shard_offset) != index:
            raise ValueError("RAM candidate shard_offset must equal canonical_index")
        return _candidate_with_location(
            self._pools[region][index],
            canonical_index=index,
            shard_id=-1,
            shard_offset=index,
        )

    def counts(self) -> dict[str, int]:
        return {region: len(self._pools[region]) for region in self.regions}

    def clear(self) -> None:
        self._ensure_open()
        self._reset()

    def flush_all(self) -> None:
        self._ensure_open()

    def close(self) -> None:
        if self._closed:
            return
        self.flush_all()
        self._closed = True

    def stats(self) -> CandidatePoolStats:
        zeros = {region: 0 for region in self.regions}
        return CandidatePoolStats(
            backend="ram",
            active_counts=self.counts(),
            records_written=self._records_written,
            replacement_counts=self._replacement_counts,
            ignored_duplicate_counts=self._ignored_duplicate_counts,
            buffered_counts=zeros,
            shard_counts=zeros,
            cached_shards=0,
            max_cached_shards=0,
            storage_dir=None,
            closed=self._closed,
        )

    def _reset(self) -> None:
        self._pools = {region: [] for region in self.regions}
        self._identity_index = {region: {} for region in self.regions}
        self._records_written = {region: 0 for region in self.regions}
        self._replacement_counts = {region: 0 for region in self.regions}
        self._ignored_duplicate_counts = {region: 0 for region in self.regions}

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("candidate pool is closed")

    def _ensure_region(self, region: str) -> None:
        if region not in self._pools:
            raise KeyError(f"unknown candidate-pool region: {region!r}")

    @staticmethod
    def _ensure_canonical_order(order: str) -> None:
        if order != "canonical":
            raise ValueError(f"unsupported candidate iteration order: {order!r}")


@dataclass(frozen=True)
class _CandidatePointer:
    canonical_index: int
    rank: CandidateRank
    image_id: str
    shard_id: int
    offset: int


@dataclass(frozen=True)
class _PendingRecord:
    key: torch.Tensor
    value: torch.Tensor
    global_key: torch.Tensor
    reliability: float
    diversity: float
    rank: CandidateRank
    canonical_index: int
    token_kind: int
    meta: dict[str, Any]
    global_meta: Optional[dict[str, Any]]


class DiskCandidatePool:
    """Lossless torch-shard candidate pool with exact canonical replacement."""

    SHARD_VERSION = 1

    def __init__(
        self,
        *,
        root_dir: Optional[os.PathLike[str] | str],
        regions: Sequence[str],
        shard_size: int = 4096,
        max_open_shards: int = 2,
        rank_fn: Callable[[UnlabeledMemoryToken], Any],
        identity_fn: Callable[[UnlabeledMemoryToken, str], Any],
        epoch: Optional[int] = None,
        cfg: Any = None,
        logger: Any = None,
    ) -> None:
        del cfg  # Reserved for a future manager integration without changing this API.
        if not callable(rank_fn) or not callable(identity_fn):
            raise TypeError("rank_fn and identity_fn must be callable")
        self.regions = _validated_regions(regions)
        self.rank_fn = rank_fn
        self.identity_fn = identity_fn
        self.shard_size = int(shard_size)
        self.max_open_shards = int(max_open_shards)
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        if epoch is not None and int(epoch) < 0:
            raise ValueError("epoch must be non-negative or None")
        self.epoch = None if epoch is None else int(epoch)
        self.logger = logger
        self._closed = False
        self._storage_dir = self._new_storage_dir(root_dir)
        self._loaded_shards: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._identity_index: dict[str, dict[CandidateIdentity, _CandidatePointer]] = {}
        self._canonical_index: dict[str, dict[int, _CandidatePointer]] = {}
        self._buffers: dict[str, list[_PendingRecord]] = {}
        self._next_canonical_index: dict[str, int] = {}
        self._next_shard_id: dict[str, int] = {}
        self._records_written: dict[str, int] = {}
        self._replacement_counts: dict[str, int] = {}
        self._ignored_duplicate_counts: dict[str, int] = {}
        self._reset_state()
        self._create_region_dirs()

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def append_many(
        self,
        region: str,
        candidates: Sequence[UnlabeledMemoryToken],
    ) -> None:
        self._ensure_open()
        self._ensure_region(region)
        _validated_candidates(candidates)
        for candidate in candidates:
            identity = _validated_identity(self.identity_fn(candidate, region))
            rank = _validated_rank(self.rank_fn(candidate))
            previous = self._identity_index[region].get(identity)
            if previous is not None and rank <= previous.rank:
                self._ignored_duplicate_counts[region] += 1
                continue

            canonical_index = (
                self._next_canonical_index[region]
                if previous is None
                else previous.canonical_index
            )
            record = self._pack_candidate(candidate, rank, canonical_index)
            shard_id = self._next_shard_id[region]
            offset = len(self._buffers[region])
            pointer = _CandidatePointer(
                canonical_index=canonical_index,
                rank=rank,
                image_id=str(record.meta["image_id"]),
                shard_id=shard_id,
                offset=offset,
            )
            self._buffers[region].append(record)
            self._identity_index[region][identity] = pointer
            self._canonical_index[region][canonical_index] = pointer
            self._records_written[region] += 1
            if previous is None:
                self._next_canonical_index[region] += 1
            else:
                self._replacement_counts[region] += 1
            if len(self._buffers[region]) >= self.shard_size:
                self._flush_region(region)

    def iter_region(
        self,
        region: str,
        order: str = "canonical",
    ) -> Iterator[UnlabeledMemoryToken]:
        self._ensure_open()
        self._ensure_region(region)
        RAMCandidatePool._ensure_canonical_order(order)
        self.flush_all()
        pointers = sorted(
            self._identity_index[region].values(),
            key=lambda item: item.canonical_index,
        )
        return (self._candidate_from_pointer(region, pointer) for pointer in pointers)

    def iter_region_by_image(
        self,
        region: str,
    ) -> Iterator[tuple[str, list[UnlabeledMemoryToken]]]:
        self._ensure_open()
        self._ensure_region(region)
        self.flush_all()
        pointers = sorted(
            self._identity_index[region].values(),
            key=lambda item: item.canonical_index,
        )
        grouped: OrderedDict[str, list[_CandidatePointer]] = OrderedDict()
        for pointer in pointers:
            grouped.setdefault(pointer.image_id, []).append(pointer)

        def generate() -> Iterator[tuple[str, list[UnlabeledMemoryToken]]]:
            for image_id, image_pointers in grouped.items():
                yield image_id, [
                    self._candidate_from_pointer(region, pointer)
                    for pointer in image_pointers
                ]

        return generate()

    def get_candidate(
        self,
        region: str,
        canonical_index: int,
        *,
        shard_id: Optional[int] = None,
        shard_offset: Optional[int] = None,
    ) -> UnlabeledMemoryToken:
        self._ensure_open()
        self._ensure_region(region)
        self.flush_all()
        index = int(canonical_index)
        pointer = self._canonical_index[region].get(index)
        if pointer is None:
            raise KeyError(
                f"unknown canonical candidate index {index} for region {region!r}"
            )
        if shard_id is not None and int(shard_id) != pointer.shard_id:
            raise RuntimeError("scored candidate shard_id no longer matches active index")
        if shard_offset is not None and int(shard_offset) != pointer.offset:
            raise RuntimeError(
                "scored candidate shard_offset no longer matches active index"
            )
        return self._candidate_from_pointer(region, pointer)

    def counts(self) -> dict[str, int]:
        return {
            region: len(self._identity_index[region])
            for region in self.regions
        }

    def clear(self) -> None:
        self._ensure_open()
        self._loaded_shards.clear()
        self._buffers = {region: [] for region in self.regions}
        if self._storage_dir.exists():
            shutil.rmtree(self._storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=False)
        self._reset_state()
        self._create_region_dirs()

    def flush_all(self) -> None:
        self._ensure_open()
        for region in self.regions:
            self._flush_region(region)

    def close(self) -> None:
        if self._closed:
            return
        self.flush_all()
        self._loaded_shards.clear()
        self._buffers = {region: [] for region in self.regions}
        self._closed = True

    def stats(self) -> CandidatePoolStats:
        return CandidatePoolStats(
            backend="disk",
            active_counts=self.counts(),
            records_written=self._records_written,
            replacement_counts=self._replacement_counts,
            ignored_duplicate_counts=self._ignored_duplicate_counts,
            buffered_counts={
                region: len(self._buffers[region]) for region in self.regions
            },
            shard_counts=dict(self._next_shard_id),
            cached_shards=len(self._loaded_shards),
            max_cached_shards=self.max_open_shards,
            storage_dir=str(self._storage_dir),
            closed=self._closed,
        )

    def __enter__(self) -> DiskCandidatePool:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _pack_candidate(
        self,
        candidate: UnlabeledMemoryToken,
        rank: CandidateRank,
        canonical_index: int,
    ) -> _PendingRecord:
        token_kind = _candidate_token_kind(candidate)
        meta = getattr(candidate, "meta", None)
        if not isinstance(meta, Mapping):
            raise TypeError("candidate.meta must be a mapping")
        if "image_id" not in meta:
            raise KeyError("candidate.meta is missing 'image_id'")
        global_meta = getattr(candidate, "global_meta", None)
        if global_meta is not None and not isinstance(global_meta, Mapping):
            raise TypeError("candidate.global_meta must be a mapping or None")
        return _PendingRecord(
            key=self._cpu_tensor(getattr(candidate, "key", None), "candidate.key"),
            value=self._cpu_tensor(getattr(candidate, "value", None), "candidate.value"),
            global_key=self._cpu_tensor(
                getattr(candidate, "global_key", None),
                "candidate.global_key",
            ),
            reliability=float(getattr(candidate, "reliability")),
            diversity=float(getattr(candidate, "diversity", 0.0)),
            rank=rank,
            canonical_index=int(canonical_index),
            token_kind=token_kind,
            meta=copy.deepcopy(dict(meta)),
            global_meta=(
                copy.deepcopy(dict(global_meta)) if global_meta is not None else None
            ),
        )

    def _flush_region(self, region: str) -> None:
        records = self._buffers[region]
        if not records:
            return
        shard_id = self._next_shard_id[region]
        payload = {
            "version": self.SHARD_VERSION,
            "region": region,
            "records": {
                "keys": self._stack_records(records, "key"),
                "values": self._stack_records(records, "value"),
                "global_keys": self._stack_records(records, "global_key"),
                "reliability": torch.tensor(
                    [record.reliability for record in records],
                    dtype=torch.float64,
                ),
                "diversity": torch.tensor(
                    [record.diversity for record in records],
                    dtype=torch.float64,
                ),
                "rank": {
                    "reliability": torch.tensor(
                        [record.rank[0] for record in records],
                        dtype=torch.float64,
                    ),
                    "step_added": torch.tensor(
                        [record.rank[1] for record in records],
                        dtype=torch.int64,
                    ),
                },
                "canonical_index": torch.tensor(
                    [record.canonical_index for record in records],
                    dtype=torch.int64,
                ),
                "token_kind": torch.tensor(
                    [record.token_kind for record in records],
                    dtype=torch.uint8,
                ),
            },
            "meta": [record.meta for record in records],
            "global_meta": [record.global_meta for record in records],
        }
        path = self._shard_path(region, shard_id)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._buffers[region] = []
        self._next_shard_id[region] += 1

    def _candidate_from_pointer(
        self,
        region: str,
        pointer: _CandidatePointer,
    ) -> UnlabeledMemoryToken:
        payload = self._load_shard(region, pointer.shard_id)
        records = payload["records"]
        offset = pointer.offset
        canonical_index = int(records["canonical_index"][offset].item())
        if canonical_index != pointer.canonical_index:
            raise RuntimeError(
                "candidate shard canonical index does not match the active index"
            )
        kwargs = {
            "key": records["keys"][offset].detach().cpu().clone(),
            "value": records["values"][offset].detach().cpu().clone(),
            "global_key": records["global_keys"][offset].detach().cpu().clone(),
            "meta": copy.deepcopy(payload["meta"][offset]),
            "reliability": float(records["reliability"][offset].item()),
            "diversity": float(records["diversity"][offset].item()),
            "global_meta": copy.deepcopy(payload["global_meta"][offset]),
        }
        token_kind = int(records["token_kind"][offset].item())
        if token_kind == 0:
            token_type = UnlabeledMemoryToken
        elif token_kind == 1:
            from CBM.sv_ume.sam_refined_candidate_builder import TokenCandidate

            token_type = TokenCandidate
        else:
            raise RuntimeError(f"unsupported candidate token kind: {token_kind}")
        token = token_type(**kwargs)
        return _candidate_with_location(
            token,
            canonical_index=pointer.canonical_index,
            shard_id=pointer.shard_id,
            shard_offset=pointer.offset,
        )

    def _load_shard(self, region: str, shard_id: int) -> dict[str, Any]:
        cache_key = (region, shard_id)
        cached = self._loaded_shards.pop(cache_key, None)
        if cached is not None:
            self._loaded_shards[cache_key] = cached
            return cached
        path = self._shard_path(region, shard_id)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        self._validate_shard(payload, region)
        self._loaded_shards[cache_key] = payload
        while len(self._loaded_shards) > self.max_open_shards:
            self._loaded_shards.popitem(last=False)
        return payload

    def _validate_shard(self, payload: Any, region: str) -> None:
        if not isinstance(payload, Mapping):
            raise RuntimeError("candidate shard payload must be a mapping")
        if int(payload.get("version", -1)) != self.SHARD_VERSION:
            raise RuntimeError("unsupported candidate shard version")
        if str(payload.get("region")) != region:
            raise RuntimeError("candidate shard region mismatch")
        records = payload.get("records")
        if not isinstance(records, Mapping):
            raise RuntimeError("candidate shard records must be a mapping")
        required = {
            "keys",
            "values",
            "global_keys",
            "reliability",
            "diversity",
            "rank",
            "canonical_index",
            "token_kind",
        }
        missing = required - set(records)
        if missing:
            raise RuntimeError(f"candidate shard is missing fields: {sorted(missing)}")
        count = int(records["canonical_index"].numel())
        for name in (
            "keys",
            "values",
            "global_keys",
            "reliability",
            "diversity",
            "token_kind",
        ):
            value = records[name]
            if not torch.is_tensor(value) or value.ndim == 0 or value.size(0) != count:
                raise RuntimeError(f"candidate shard field {name!r} is not aligned")
        rank = records["rank"]
        if not isinstance(rank, Mapping):
            raise RuntimeError("candidate shard rank must be a tensor mapping")
        for name in ("reliability", "step_added"):
            value = rank.get(name)
            if not torch.is_tensor(value) or value.numel() != count:
                raise RuntimeError(f"candidate shard rank field {name!r} is not aligned")
        meta = payload.get("meta")
        global_meta = payload.get("global_meta")
        if not isinstance(meta, list) or len(meta) != count:
            raise RuntimeError("candidate shard meta is not aligned")
        if not isinstance(global_meta, list) or len(global_meta) != count:
            raise RuntimeError("candidate shard global_meta is not aligned")

    def _reset_state(self) -> None:
        self._identity_index = {region: {} for region in self.regions}
        self._canonical_index = {region: {} for region in self.regions}
        self._buffers = {region: [] for region in self.regions}
        self._next_canonical_index = {region: 0 for region in self.regions}
        self._next_shard_id = {region: 0 for region in self.regions}
        self._records_written = {region: 0 for region in self.regions}
        self._replacement_counts = {region: 0 for region in self.regions}
        self._ignored_duplicate_counts = {region: 0 for region in self.regions}

    def _create_region_dirs(self) -> None:
        for region in self.regions:
            (self._storage_dir / region).mkdir(parents=True, exist_ok=False)

    def _new_storage_dir(self, root_dir: Optional[os.PathLike[str] | str]) -> Path:
        epoch_label = "none" if self.epoch is None else str(self.epoch)
        if root_dir is None:
            return Path(tempfile.mkdtemp(prefix=f"sv_ume_pool_e{epoch_label}_"))
        root = Path(root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        storage_dir = root / f"pool_e{epoch_label}_{uuid.uuid4().hex}"
        storage_dir.mkdir(parents=False, exist_ok=False)
        return storage_dir

    def _shard_path(self, region: str, shard_id: int) -> Path:
        return self._storage_dir / region / f"shard_{shard_id:08d}.pt"

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("candidate pool is closed")

    def _ensure_region(self, region: str) -> None:
        if region not in self._identity_index:
            raise KeyError(f"unknown candidate-pool region: {region!r}")

    @staticmethod
    def _cpu_tensor(value: Any, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        return value.detach().cpu().clone()

    @staticmethod
    def _stack_records(records: Sequence[_PendingRecord], name: str) -> torch.Tensor:
        tensors = [getattr(record, name) for record in records]
        expected_shape = tensors[0].shape
        expected_dtype = tensors[0].dtype
        if any(
            tensor.shape != expected_shape or tensor.dtype != expected_dtype
            for tensor in tensors[1:]
        ):
            raise ValueError(
                f"candidate shard field {name!r} has inconsistent shapes or dtypes"
            )
        try:
            return torch.stack(tensors, dim=0)
        except RuntimeError as exc:
            raise ValueError(
                f"candidate shard field {name!r} has inconsistent shapes or dtypes"
            ) from exc


__all__ = [
    "CandidatePoolProtocol",
    "CandidatePoolStats",
    "RAMCandidatePool",
    "DiskCandidatePool",
]
