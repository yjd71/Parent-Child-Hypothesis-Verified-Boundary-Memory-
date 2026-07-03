from __future__ import annotations

import math
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence


@dataclass(frozen=True)
class ScoredCandidateRecord:
    region: str
    gate_mode: str
    image_id: str
    input_index: int
    selection_score: float
    shard_id: int
    shard_offset: int
    canonical_index: int
    d_img: float
    d_region: float
    raw_diversity: float
    diversity_score: float


class ScoredCandidateStore:
    """SQLite-backed exact ordering index for scored SV-UME candidates."""

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        *,
        delete_on_close: bool = True,
    ) -> None:
        self.delete_on_close = bool(delete_on_close)
        self._closed = False
        self._storage_dir = self._new_storage_dir(root_dir)
        self._db_path = self._storage_dir / "scored_candidates.sqlite3"
        self._connection = sqlite3.connect(str(self._db_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute("PRAGMA cache_size=-2048")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    @property
    def db_path(self) -> Path:
        return self._db_path

    def add_many(
        self,
        region: str,
        records: Sequence[ScoredCandidateRecord],
    ) -> None:
        self._ensure_open()
        if not records:
            return
        rows = []
        for record in records:
            self._validate_record(record, region)
            rows.append(
                (
                    record.region,
                    record.gate_mode,
                    record.image_id,
                    int(record.input_index),
                    float(record.selection_score),
                    int(record.shard_id),
                    int(record.shard_offset),
                    int(record.canonical_index),
                    float(record.d_img),
                    float(record.d_region),
                    float(record.raw_diversity),
                    float(record.diversity_score),
                )
            )
        try:
            self._connection.executemany(
                """
                INSERT INTO scored_candidates (
                    region, gate_mode, image_id, input_index, selection_score,
                    shard_id, shard_offset, canonical_index,
                    d_img, d_region, raw_diversity, diversity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def iter_ranked(
        self,
        region: str,
        gate_mode: Optional[str] = None,
    ) -> Iterator[ScoredCandidateRecord]:
        self._ensure_open()
        if gate_mode is not None and gate_mode not in {"strict", "relaxed"}:
            raise ValueError(f"unsupported gate_mode: {gate_mode!r}")
        if gate_mode is None:
            cursor = self._connection.execute(
                """
                SELECT * FROM scored_candidates
                WHERE region = ?
                ORDER BY selection_score DESC, input_index ASC
                """,
                (str(region),),
            )
        else:
            cursor = self._connection.execute(
                """
                SELECT * FROM scored_candidates
                WHERE region = ? AND gate_mode = ?
                ORDER BY selection_score DESC, input_index ASC
                """,
                (str(region), str(gate_mode)),
            )
        try:
            for row in cursor:
                yield self._row_to_record(row)
        finally:
            cursor.close()

    def count(self, region: str, gate_mode: Optional[str] = None) -> int:
        self._ensure_open()
        if gate_mode is None:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM scored_candidates WHERE region = ?",
                (str(region),),
            ).fetchone()
        else:
            if gate_mode not in {"strict", "relaxed"}:
                raise ValueError(f"unsupported gate_mode: {gate_mode!r}")
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM scored_candidates
                WHERE region = ? AND gate_mode = ?
                """,
                (str(region), str(gate_mode)),
            ).fetchone()
        return int(row["count"])

    def gate_counts(self, region: str) -> dict[str, int]:
        self._ensure_open()
        rows = self._connection.execute(
            """
            SELECT gate_mode, COUNT(*) AS count
            FROM scored_candidates WHERE region = ? GROUP BY gate_mode
            """,
            (str(region),),
        ).fetchall()
        return {str(row["gate_mode"]): int(row["count"]) for row in rows}

    def image_counts(self, region: str) -> dict[str, int]:
        self._ensure_open()
        rows = self._connection.execute(
            """
            SELECT image_id, COUNT(*) AS count
            FROM scored_candidates WHERE region = ? GROUP BY image_id
            """,
            (str(region),),
        ).fetchall()
        return {str(row["image_id"]): int(row["count"]) for row in rows}

    def close(self) -> None:
        if self._closed:
            return
        self._connection.commit()
        self._connection.close()
        self._closed = True
        if self.delete_on_close and self._storage_dir.exists():
            shutil.rmtree(self._storage_dir)

    def __enter__(self) -> ScoredCandidateStore:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE scored_candidates (
                region TEXT NOT NULL,
                gate_mode TEXT NOT NULL,
                image_id TEXT NOT NULL,
                input_index INTEGER NOT NULL,
                selection_score REAL NOT NULL,
                shard_id INTEGER NOT NULL,
                shard_offset INTEGER NOT NULL,
                canonical_index INTEGER NOT NULL,
                d_img REAL NOT NULL,
                d_region REAL NOT NULL,
                raw_diversity REAL NOT NULL,
                diversity_score REAL NOT NULL,
                PRIMARY KEY (region, input_index)
            );
            CREATE INDEX idx_scored_rank_all
                ON scored_candidates(region, selection_score DESC, input_index ASC);
            CREATE INDEX idx_scored_rank_gate
                ON scored_candidates(
                    region, gate_mode, selection_score DESC, input_index ASC
                );
            CREATE INDEX idx_scored_image
                ON scored_candidates(region, image_id);
            """
        )
        self._connection.commit()

    @staticmethod
    def _validate_record(record: ScoredCandidateRecord, region: str) -> None:
        if record.region != str(region):
            raise ValueError("scored candidate region does not match insertion region")
        if record.gate_mode not in {"strict", "relaxed"}:
            raise ValueError(f"unsupported gate_mode: {record.gate_mode!r}")
        for name in (
            "selection_score",
            "d_img",
            "d_region",
            "raw_diversity",
            "diversity_score",
        ):
            if not math.isfinite(float(getattr(record, name))):
                raise ValueError(f"scored candidate {name} must be finite")
        if int(record.input_index) < 0 or int(record.canonical_index) < 0:
            raise ValueError("candidate indices must be non-negative")

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ScoredCandidateRecord:
        return ScoredCandidateRecord(
            region=str(row["region"]),
            gate_mode=str(row["gate_mode"]),
            image_id=str(row["image_id"]),
            input_index=int(row["input_index"]),
            selection_score=float(row["selection_score"]),
            shard_id=int(row["shard_id"]),
            shard_offset=int(row["shard_offset"]),
            canonical_index=int(row["canonical_index"]),
            d_img=float(row["d_img"]),
            d_region=float(row["d_region"]),
            raw_diversity=float(row["raw_diversity"]),
            diversity_score=float(row["diversity_score"]),
        )

    @staticmethod
    def _new_storage_dir(root_dir: Optional[str | Path]) -> Path:
        if root_dir is None:
            return Path(tempfile.mkdtemp(prefix="sv_ume_scored_"))
        root = Path(root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        storage_dir = root / f"scored_{uuid.uuid4().hex}"
        storage_dir.mkdir(parents=False, exist_ok=False)
        return storage_dir

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scored candidate store is closed")


__all__ = ["ScoredCandidateRecord", "ScoredCandidateStore"]
