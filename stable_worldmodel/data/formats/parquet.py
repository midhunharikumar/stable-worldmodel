"""Parquet format: single .parquet file with episode-contiguous flat rows.

Each step is one row. Two writer-managed index columns — ``episode_idx`` and
``step_idx`` — let the reader recover episode boundaries by scanning a single
column (same convention as the Lance format).

Columns are stored losslessly and in a tool-readable way: numeric/image arrays
are flattened into ``large_list`` columns of their native Arrow dtype, so
external readers (pandas, DuckDB, parquet-tools) see real numbers rather than
opaque blobs. ``large_list`` uses int64 offsets — the regular ``list`` type's
int32 offsets overflow on large image columns. The per-step shape + dtype are
recorded in the schema's key/value metadata so the reader reshapes each row
back exactly. String columns are stored as ``large_string``. The roundtrip is
exact (no JPEG re-encode), unlike the Lance backend.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

import pyarrow as pa
import pyarrow.parquet as pq

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import (
    Format,
    register_format,
    validate_write_mode,
)
from stable_worldmodel.data.utils import get_cache_dir


_INDEX_COLUMNS = ('episode_idx', 'step_idx')
_META_KEY = b'swm_meta'
_DATA_FILE = 'data.parquet'


def _resolve_parquet_file(p: Path) -> Path:
    """Resolve a dataset location to the single ``.parquet`` file inside it.

    The on-disk layout is a directory (``<path>/data.parquet``), mirroring the
    Lance backend, so it drops into directory-oriented tooling (``copytree``,
    autodetection). A direct path to a ``.parquet`` file is also accepted.
    """
    if p.is_dir():
        files = sorted(p.glob('*.parquet')) + sorted(p.glob('*.pq'))
        if not files:
            raise FileNotFoundError(f'No .parquet file in {p}')
        if len(files) > 1:
            raise ValueError(
                f'Ambiguous dataset: multiple Parquet files in {p}. '
                'Pass the file directly.'
            )
        return files[0]
    return p


class ParquetDataset(Dataset):
    """Reader for a Parquet file written by :class:`ParquetWriter`.

    Columns are kept in their on-disk Arrow form and decoded per access:
    ``__getitem__`` slices only the rows it needs, flattens the Arrow list to
    NumPy, and reshapes with the schema metadata
    (``arrow_array.flatten().to_numpy().reshape(shape)``) — NumPy is just the
    bridge to ``torch.from_numpy``. The Arrow table is memory-mapped and
    reopened lazily per worker, so DataLoader spawn doesn't pickle the data.
    Derived columns from :meth:`merge_col` are cached as NumPy. Restrict what
    is loaded with ``keys_to_load`` for wide tables.
    """

    def __init__(
        self,
        name: str | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        if path is not None:
            loc = Path(path)
        else:
            if name is None:
                raise TypeError('ParquetDataset requires either `name` or `path`')
            datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
            loc = Path(datasets_dir, f'{name}.parquet')

        self.path = _resolve_parquet_file(loc)
        self._specs = _read_specs(pq.read_schema(self.path))

        available = list(self._specs)
        self._keys = keys_to_load or available
        missing = [k for k in self._keys if k not in available]
        if missing:
            raise KeyError(f"Columns {missing} missing from '{self.path}'")

        # Only read the columns we need, plus the index column used to recover
        # episode boundaries.
        self._read_cols = list(dict.fromkeys([*self._keys, 'episode_idx']))
        self._columns: dict | None = None  # Arrow arrays, opened lazily
        self._cache: dict[str, np.ndarray] = {}  # merged/derived columns
        self._open()

        lengths, offsets = self._episode_structure(self._table)

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def __getstate__(self) -> dict:
        # Don't pickle the Arrow table into DataLoader workers; each reopens
        # the memory-mapped file lazily on first access.
        state = self.__dict__.copy()
        state['_table'] = None
        state['_columns'] = None
        return state

    def _open(self) -> None:
        if self._columns is None:
            self._table = pq.read_table(
                self.path, columns=self._read_cols, memory_map=True
            )
            self._columns = {
                col: _single_array(self._table.column(col))
                for col in self._keys
            }

    @staticmethod
    def _episode_structure(table) -> tuple[np.ndarray, np.ndarray]:
        ep_ids = table.column('episode_idx').to_numpy()
        if len(ep_ids) == 0:
            empty = np.array([], dtype=np.int64)
            return empty, empty
        if len(ep_ids) > 1 and (np.diff(ep_ids) < 0).any():
            raise ValueError(
                f"Parquet file '{table}' is not episode-contiguous "
                '(episode_idx decreases). Rebuild it.'
            )
        change = np.flatnonzero(np.diff(ep_ids) != 0) + 1
        offsets = np.concatenate([[0], change]).astype(np.int64)
        lengths = np.diff(
            np.concatenate([offsets, [len(ep_ids)]])
        ).astype(np.int64)
        return lengths, offsets

    def _decode(self, arrow_array, spec: dict) -> np.ndarray:
        """Decode an Arrow (large_list / large_string) slice into NumPy."""
        if spec['kind'] == 'str':
            return np.asarray(arrow_array.to_pylist(), dtype=object)
        dtype = np.dtype(spec['dtype'])
        shape = tuple(spec['shape'])
        n = len(arrow_array)
        if n == 0:
            return np.empty((0, *shape), dtype=dtype)
        flat = arrow_array.flatten().to_numpy(zero_copy_only=False)
        return flat.astype(dtype, copy=False).reshape((n, *shape))

    def _column_slice(self, col: str, g_start: int, g_end: int) -> np.ndarray:
        if col in self._cache:
            return self._cache[col][g_start:g_end]
        self._open()
        arrow_array = self._columns[col].slice(g_start, g_end - g_start)
        return self._decode(arrow_array, self._specs[col])

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            data = self._column_slice(col, g_start, g_end)
            if col != 'action':
                data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                # np.array(...) yields a writable, contiguous copy; the Arrow
                # buffer behind `data` is read-only and small per slice.
                steps[col] = torch.from_numpy(np.array(data))
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self._decode(self._columns[col], self._specs[col])

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {col: self.get_col_data(col)[row_idx] for col in self._keys}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        if isinstance(source, str):
            source = [k for k in self._keys if re.match(source, k)]
        merged = np.concatenate(
            [self.get_col_data(s) for s in source], axis=dim
        )
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(f"Merged columns {source} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1


class ParquetWriter:
    """Append episodes to a single Parquet file. Schema is inferred from the
    first episode and locked thereafter.

    Parquet files are written whole on close (Parquet has no in-place append),
    so episodes are buffered in memory and flushed in ``__exit__``. In
    ``'append'`` mode an existing file is read back into the buffer first.

    The on-disk layout is a directory holding a single ``data.parquet``
    (mirroring the Lance backend's ``<table>.lance/`` directory), so it drops
    into directory-oriented tooling like ``shutil.copytree``.

    Args:
        path: target dataset directory (e.g. ``foo`` or ``foo.parquet``); the
            table is written to ``<path>/data.parquet``.
        mode: ``'append'`` (default — extend if the dataset exists),
            ``'overwrite'`` (replace it), or ``'error'`` (raise if it already
            exists). See :data:`stable_worldmodel.data.format.WRITE_MODES`.
    """

    def __init__(self, path, *, mode: str = 'append'):
        validate_write_mode(mode)
        self.dir = Path(path)
        self.path = self.dir / _DATA_FILE
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._existing_file: Path | None = None
        self._entered = False
        self._initialized = False
        self._appending_existing = False
        self._validated = False
        self._cols: dict[str, list[np.ndarray]] = {}
        self._ordered_cols: list[str] = []
        self._specs: dict[str, dict] = {}
        self._lengths: list[int] = []

    def __enter__(self):
        self._entered = True
        existing = sorted(self.dir.glob('*.parquet')) + sorted(
            self.dir.glob('*.pq')
        )
        if existing:
            if self.mode == 'error':
                raise FileExistsError(
                    f"ParquetWriter: '{self.dir}' already contains a dataset. "
                    "Pass mode='overwrite' to replace it or mode='append' to "
                    'extend it.'
                )
            if self.mode == 'overwrite':
                for f in existing:
                    f.unlink()
            else:
                self._existing_file = existing[0]
                self._load_existing_state()
        return self

    def __exit__(self, *exc):
        self._entered = False
        if exc[0] is None:
            self._flush()

    def write_episode(self, ep_data: dict) -> None:
        if not self._entered:
            raise RuntimeError('ParquetWriter used outside of a `with` block')
        if not self._initialized:
            self._init_schema(ep_data)
            self._initialized = True
        elif self._appending_existing and not self._validated:
            self._validate_episode_against_existing(ep_data)
            self._validated = True

        ep_len = len(next(iter(ep_data.values())))
        for col in self._ordered_cols:
            self._cols[col].append(self._stack_column(col, ep_data[col]))
        self._lengths.append(ep_len)

    def write_episodes(self, episodes) -> None:
        for ep in episodes:
            self.write_episode(ep)

    def _stack_column(self, col: str, vals) -> np.ndarray:
        if self._specs[col]['kind'] == 'str':
            return np.asarray([_as_str(v) for v in vals], dtype=object)
        spec = self._specs[col]
        return np.asarray(vals, dtype=np.dtype(spec['dtype']))

    def _load_existing_state(self) -> None:
        table = pq.read_table(self._existing_file)
        self._specs = _read_specs(table.schema)
        self._ordered_cols = [
            c for c in table.schema.names if c not in _INDEX_COLUMNS
        ]
        for col in self._ordered_cols:
            self._cols[col] = [_column_to_numpy(table, col, self._specs[col])]
        ep_ids = table.column('episode_idx').to_numpy()
        if len(ep_ids):
            _, counts = np.unique(ep_ids, return_counts=True)
            self._lengths = counts.astype(int).tolist()
        # The whole table is buffered now; drop the source so `_flush` can
        # rewrite a single consolidated `data.parquet` without leaving a
        # second (ambiguous) file in the directory.
        if self._existing_file != self.path:
            self._existing_file.unlink()
        self._initialized = True
        self._appending_existing = True

    def _validate_episode_against_existing(self, ep_data: dict) -> None:
        existing = set(self._ordered_cols)
        incoming = {c for c in ep_data if c not in _INDEX_COLUMNS}
        missing = existing - incoming
        extra = incoming - existing
        if missing or extra:
            raise ValueError(
                f"ParquetWriter: append failed — schema mismatch on "
                f"'{self.path}'. Missing columns: {sorted(missing)}; "
                f'unexpected columns: {sorted(extra)}.'
            )
        for col in self._ordered_cols:
            spec = self._specs[col]
            if spec['kind'] == 'str':
                continue
            sample = np.asarray(ep_data[col][0])
            if list(sample.shape) != spec['shape']:
                raise ValueError(
                    f"ParquetWriter: append failed — column '{col}' shape "
                    f"mismatch: existing per-step={tuple(spec['shape'])}, "
                    f'incoming per-step={sample.shape}.'
                )

    def _init_schema(self, sample_ep: dict) -> None:
        dropped = [c for c in sample_ep if c in _INDEX_COLUMNS]
        if dropped:
            logging.warning(
                'ParquetWriter: dropping incoming columns %s — names are '
                'reserved for the writer-managed index columns.',
                dropped,
            )
        for col, vals in sample_ep.items():
            if col in _INDEX_COLUMNS:
                continue
            first = vals[0]
            if isinstance(first, (str, bytes)):
                self._specs[col] = {'kind': 'str'}
            else:
                sample = np.asarray(first)
                self._specs[col] = {
                    'kind': 'array',
                    'shape': list(sample.shape),
                    'dtype': sample.dtype.str,
                }
            self._ordered_cols.append(col)
            self._cols[col] = []

    def _flush(self) -> None:
        if not self._initialized:
            # Nothing written — emit an empty, well-formed table so the file
            # still round-trips through the reader.
            self._write_table(0)
            return
        total = int(sum(self._lengths))
        self._write_table(total)

    def _write_table(self, total: int) -> None:
        episode_idx = np.concatenate(
            [np.full(n, i, dtype=np.int32) for i, n in enumerate(self._lengths)]
        ) if self._lengths else np.zeros(0, dtype=np.int32)
        step_idx = np.concatenate(
            [np.arange(n, dtype=np.int32) for n in self._lengths]
        ) if self._lengths else np.zeros(0, dtype=np.int32)

        arrays = [pa.array(episode_idx), pa.array(step_idx)]
        fields = [
            pa.field('episode_idx', pa.int32()),
            pa.field('step_idx', pa.int32()),
        ]
        for col in self._ordered_cols:
            spec = self._specs[col]
            parts = self._cols[col]
            if spec['kind'] == 'str':
                flat = (
                    np.concatenate(parts)
                    if parts
                    else np.empty(0, dtype=object)
                )
                arrays.append(pa.array(list(flat), type=pa.large_string()))
                fields.append(pa.field(col, pa.large_string()))
            else:
                # Store each step's array, flattened, in a large_list column
                # so external tools (pandas, DuckDB, ...) see real numbers.
                # large_list uses int64 offsets; the regular `list` type's
                # int32 offsets overflow on big image columns. The per-step
                # shape is recovered from the schema metadata on read.
                shape = tuple(spec['shape'])
                dtype = np.dtype(spec['dtype'])
                dim = int(np.prod(shape)) if shape else 1
                value_type = pa.from_numpy_dtype(dtype)
                if parts:
                    flat = np.ascontiguousarray(
                        np.concatenate(parts).reshape((total, dim)),
                        dtype=dtype,
                    ).reshape(-1)
                else:
                    flat = np.empty(0, dtype=dtype)
                values = pa.array(flat, type=value_type)
                offsets = pa.array(
                    np.arange(total + 1, dtype=np.int64) * dim,
                    type=pa.int64(),
                )
                arrays.append(
                    pa.LargeListArray.from_arrays(offsets, values)
                )
                fields.append(pa.field(col, pa.large_list(value_type)))

        schema = pa.schema(fields).with_metadata(
            {_META_KEY: json.dumps(self._specs).encode()}
        )
        table = pa.table(arrays, schema=schema)
        pq.write_table(table, str(self.path))


def _as_str(val) -> str:
    return val.decode() if isinstance(val, bytes) else str(val)


def _read_specs(schema: pa.Schema) -> dict[str, dict]:
    meta = schema.metadata or {}
    raw = meta.get(_META_KEY)
    if raw is None:
        raise ValueError(
            'Parquet file is missing stable-worldmodel column metadata — '
            'was it written by ParquetWriter?'
        )
    return json.loads(raw)


def _single_array(chunked) -> pa.Array:
    """Collapse a ChunkedArray into one contiguous Array for O(1) slicing."""
    if chunked.num_chunks == 0:
        return pa.array([], type=chunked.type)
    if chunked.num_chunks == 1:
        return chunked.chunk(0)
    return pa.concat_arrays(chunked.chunks)


def _column_to_numpy(table, col: str, spec: dict) -> np.ndarray:
    column = _single_array(table.column(col))
    if spec['kind'] == 'str':
        return np.asarray(column.to_pylist(), dtype=object)
    dtype = np.dtype(spec['dtype'])
    shape = tuple(spec['shape'])
    nrows = len(column)
    if nrows == 0:
        return np.empty((0, *shape), dtype=dtype)
    flat = column.flatten().to_numpy(zero_copy_only=False)
    return flat.astype(dtype, copy=False).reshape((nrows, *shape))


@register_format
class Parquet(Format):
    name = 'parquet'

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        if p.suffix in ('.parquet', '.pq'):
            return True
        if p.is_dir():
            return any(p.glob('*.parquet')) or any(p.glob('*.pq'))
        return False

    @classmethod
    def open_reader(cls, path, **kwargs) -> ParquetDataset:
        return ParquetDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> ParquetWriter:
        return ParquetWriter(path, **kwargs)


__all__ = ['Parquet', 'ParquetDataset', 'ParquetWriter']
