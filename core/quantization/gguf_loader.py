"""GGUF checkpoint loader for TorchaVerse models (v1.0.0).

Purpose
-------
:mod:`core.quantization.gguf_loader` provides a minimal, dependency-free
reader for the `GGUF <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>`_
binary checkpoint format.  GGUF is the unified container used by the
``ggml-org`` ecosystem (llama.cpp, whisper.cpp, stable-diffusion.cpp...)
to ship quantized model weights and their metadata in a single file.

The format we implement here is a strict subset that is sufficient for
round-tripping inference-ready weights in TorchaVerse:

* ``F32`` / ``F16`` / ``BF16`` tensors are decoded losslessly.
* ``Q4_0`` / ``Q4_1`` / ``Q5_0`` / ``Q5_1`` / ``Q8_0`` quantized tensors
  are dequantized back to ``float32`` so they can be consumed by
  :class:`torch.nn.Module` instances whose forward pass has not been
  re-implemented with a quantization-aware matmul.

The intent of the loader is *not* to be a production-grade re-implementation
of every GGUF feature - for that, the upstream ``gguf-py`` package is
the canonical source.  Instead, this module is a teaching / integration
reference that demonstrates how a TorchaVerse
:class:`models.base.ModelMixin` can be hydrated from a GGUF file via
:meth:`models.base.ModelMixin.from_pretrained`.

Integration with :class:`ModelMixin`
------------------------------------
:meth:`GGUFLoader.to_state_dict_for` is the entry point that ties GGUF
into the rest of the project: it loads the tensor data block, runs the
caller-supplied ``key_map`` (mirroring the diffusers ``key_renames``
mechanism) and then delegates the actual weight copy to
:func:`models.base.load_state_dict_with_renames`.  This keeps dtype
preservation behaviour consistent with the safetensors loader.

References
----------
* `ggml-org/gguf <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>`_
  -- official GGUF specification.
* `gguf-py <https://github.com/ggml-org/ggml/tree/master/gguf-py>`_ --
  the reference Python implementation this loader is patterned after.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

# The re-export at the bottom of this module is the only ``import``
# outside the stdlib + torch we need.  We do it lazily inside the
# method that uses it so this module remains importable even in
# minimal environments that do not have ``models.base`` on the path.

__all__ = [
    "GGUFLoader",
    "GGUFTensorInfo",
    "GGUFMetadataValueType",
    "GGMLQuantType",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class GGUFMetadataValueType:
    """Subset of the GGUF metadata value-type tags (matching the spec)."""

    ARRAY = 8
    STRING = 8  # 8 in GGUF v3
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    FLOAT32 = 5
    FLOAT64 = 6
    BOOL = 7
    STRING_ARRAY = 9
    INT32_ARRAY = 10
    INT64_ARRAY = 11
    FLOAT32_ARRAY = 12
    BOOL_ARRAY = 13


class GGMLQuantType:
    """Subset of the GGML quantization tags we dequantize.

    Values match the GGUF / ggml enum so we can index into a lookup
    table without an extra translation step.
    """

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    BF16 = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GGUFTensorInfo:
    """Tensor descriptor extracted from the GGUF tensor-infos block."""

    name: str
    n_dims: int
    dims: Tuple[int, ...]
    ggml_type: int
    # Byte offset into the tensor data block.  Stored for callers that
    # want to peek at raw bytes without re-walking the whole file.
    offset: int = 0

    @property
    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= int(d)
        return n

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(d) for d in self.dims)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class GGUFLoader:
    """Streaming reader for GGUF v3 checkpoint files.

    The class is intentionally *stateful*: :meth:`__init__` opens the
    file and reads the header, while :meth:`metadata`,
    :meth:`tensor_info` and :meth:`load_state_dict` walk the
    well-ordered sections of the format in turn.  This mirrors how the
    upstream ``gguf-py`` reader works and keeps the public surface
    small.

    Args:
        path: Path to a ``.gguf`` file.
    """

    #: Maximum number of bytes we allow for a single string metadata
    #: entry.  This is a safety belt against accidentally- or
    #: maliciously-huge string values that would otherwise allocate
    #: gigabytes of memory.
    _MAX_STRING_BYTES = 64 * 1024 * 1024  # 64 MiB

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")
        # ``open(..., 'rb')`` is OK here - the file is small enough
        # (most GGUF weights are loaded by the kernel via mmap, not by
        # this Python reader) and we want a single, predictable
        # resource lifetime.
        self._fh: BinaryIO = self.path.open("rb")
        try:
            self._read_header()
        except Exception:
            # Make sure we don't leak a dangling file handle if the
            # header is malformed.
            self._fh.close()
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the underlying file handle (idempotent)."""
        fh = getattr(self, "_fh", None)
        if fh is not None and not fh.closed:
            fh.close()

    def __enter__(self) -> "GGUFLoader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _read_header(self) -> None:
        """Read the fixed-size GGUF header.

        Layout::

            char     magic[4];        // "GGUF"
            uint32_t version;         // 3 for the format we support
            uint64_t tensor_count;    // number of tensor-info entries
            uint64_t metadata_kv_count; // number of metadata entries
        """
        magic = self._fh.read(4)
        if magic != b"GGUF":
            raise ValueError(
                f"bad GGUF magic in {self.path!r}: got {magic!r}, "
                f"expected b'GGUF'",
            )
        (version,) = struct.unpack("<I", self._fh.read(4))
        if version != 3:
            # We only implement v3 of the format.  Earlier versions
            # have a different metadata value-type table; later
            # versions may add fields we don't model.
            raise ValueError(
                f"unsupported GGUF version: {version} "
                f"(this loader only supports version 3)",
            )
        (self._tensor_count, self._metadata_kv_count) = struct.unpack(
            "<QQ", self._fh.read(16),
        )
        # The current file cursor points at the start of the metadata
        # KV block.  Cache the offset so :meth:`metadata` and the
        # other walkers don't need to know the header size.
        self._metadata_offset = self._fh.tell()
        # The tensor data block follows the tensor-infos block; we
        # don't know its offset yet because the metadata block size is
        # not stored up-front.  Compute it lazily inside
        # :meth:`tensor_info`.

    def _read_string(self) -> str:
        """Read a length-prefixed UTF-8 string."""
        (length,) = struct.unpack("<Q", self._fh.read(8))
        if length > self._MAX_STRING_BYTES:
            raise ValueError(
                f"refusing to read GGUF string of {length} bytes "
                f"(limit is {self._MAX_STRING_BYTES})",
            )
        raw = self._fh.read(length)
        if len(raw) != length:
            raise EOFError("unexpected EOF while reading GGUF string")
        return raw.decode("utf-8", errors="replace")

    def _read_kv(self) -> Tuple[str, Any]:
        """Read a single ``(key, value)`` metadata entry.

        Returns a 2-tuple ``(key, value)``; ``value`` is the decoded
        Python object (``str`` / ``int`` / ``float`` / ``bool`` /
        ``list``).
        """
        key = self._read_string()
        (value_type,) = struct.unpack("<I", self._fh.read(4))
        if value_type == GGUFMetadataValueType.STRING:
            value: Any = self._read_string()
        elif value_type == GGUFMetadataValueType.BOOL:
            (value,) = struct.unpack("<?", self._fh.read(1))
        elif value_type in (
            GGUFMetadataValueType.INT8,
            GGUFMetadataValueType.INT16,
            GGUFMetadataValueType.INT32,
        ):
            fmt = {0: "<b", 1: "<h", 2: "<i"}[value_type]
            (value,) = struct.unpack(fmt, self._fh.read(struct.calcsize(fmt)))
        elif value_type == GGUFMetadataValueType.INT64:
            (value,) = struct.unpack("<q", self._fh.read(8))
        elif value_type == GGUFMetadataValueType.FLOAT32:
            (value,) = struct.unpack("<f", self._fh.read(4))
        elif value_type == GGUFMetadataValueType.FLOAT64:
            (value,) = struct.unpack("<d", self._fh.read(8))
        elif value_type == GGUFMetadataValueType.ARRAY:
            (elem_type,) = struct.unpack("<I", self._fh.read(4))
            (length,) = struct.unpack("<Q", self._fh.read(8))
            value = self._read_array(elem_type, length)
        else:
            # Unknown value type -- skip the rest of the entry by
            # raising.  A more lenient reader would log and ``return
            # None`` here.
            raise ValueError(
                f"unsupported GGUF metadata value type: {value_type!r}",
            )
        return key, value

    def _read_array(self, elem_type: int, length: int) -> List[Any]:
        """Decode a GGUF metadata array value.

        Only the array element types we actually use in TorchaVerse
        metadata are implemented; the rest raise.  This keeps the
        reader tight while still being useful for checkpoint
        introspection.
        """
        if elem_type == GGUFMetadataValueType.STRING:
            return [self._read_string() for _ in range(length)]
        if elem_type == GGUFMetadataValueType.BOOL:
            return list(struct.unpack(f"<{length}?", self._fh.read(length)))
        if elem_type == GGUFMetadataValueType.INT32:
            return list(struct.unpack(f"<{length}i", self._fh.read(4 * length)))
        if elem_type == GGUFMetadataValueType.INT64:
            return list(struct.unpack(f"<{length}q", self._fh.read(8 * length)))
        if elem_type == GGUFMetadataValueType.FLOAT32:
            return list(struct.unpack(f"<{length}f", self._fh.read(4 * length)))
        raise ValueError(
            f"unsupported GGUF metadata array element type: {elem_type!r}",
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    @property
    def version(self) -> int:
        return 3

    @property
    def tensor_count(self) -> int:
        return int(self._tensor_count)

    @property
    def metadata_kv_count(self) -> int:
        return int(self._metadata_kv_count)

    def metadata(self) -> Dict[str, Any]:
        """Walk the metadata KV block and return a ``{key: value}`` dict.

        The cursor is rewound to the start of the metadata block
        first, so this method is safe to call multiple times.
        """
        self._fh.seek(self._metadata_offset)
        out: Dict[str, Any] = {}
        for _ in range(self._metadata_kv_count):
            key, value = self._read_kv()
            out[key] = value
        return out

    def _read_tensor_info(self) -> GGUFTensorInfo:
        """Read a single tensor-info entry from the current cursor."""
        name = self._read_string()
        (n_dims,) = struct.unpack("<I", self._fh.read(4))
        dims = struct.unpack(f"<{n_dims}Q", self._fh.read(8 * n_dims))
        (ggml_type, offset) = struct.unpack("<iQ", self._fh.read(12))
        return GGUFTensorInfo(
            name=name,
            n_dims=n_dims,
            dims=tuple(int(d) for d in dims),
            ggml_type=int(ggml_type),
            offset=int(offset),
        )

    def tensor_info(self) -> List[GGUFTensorInfo]:
        """Read the tensor-infos block.

        The block follows the metadata KV block, so we walk metadata
        first to advance the cursor to the right place, then read
        ``tensor_count`` tensor-info entries.  Returns a list of
        :class:`GGUFTensorInfo` objects in file order.
        """
        # Force a metadata walk to advance the cursor.
        self.metadata()
        infos: List[GGUFTensorInfo] = []
        for _ in range(self._tensor_count):
            infos.append(self._read_tensor_info())
        # Cache the data-block offset so :meth:`load_state_dict` does
        # not need to redo the same bookkeeping.
        self._data_block_offset = self._fh.tell()
        return infos

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------
    def _block_size(self, ggml_type: int) -> int:
        """Number of elements in a single GGML quantization block."""
        return {
            GGMLQuantType.F32: 1,
            GGMLQuantType.F16: 1,
            GGMLQuantType.BF16: 1,
            GGMLQuantType.Q4_0: 32,
            GGMLQuantType.Q4_1: 32,
            GGMLQuantType.Q5_0: 32,
            GGMLQuantType.Q5_1: 32,
            GGMLQuantType.Q8_0: 32,
        }.get(ggml_type, 1)

    def _block_bytes(self, ggml_type: int) -> int:
        """Size in bytes of a single GGML quantization block."""
        return {
            GGMLQuantType.F32: 4,
            GGMLQuantType.F16: 2,
            GGMLQuantType.BF16: 2,
            GGMLQuantType.Q4_0: 18,  # 2-byte scale + 16 bytes of 4-bit quants
            GGMLQuantType.Q4_1: 20,  # 2-byte scale + 2-byte min + 16 bytes
            GGMLQuantType.Q5_0: 22,  # 2-byte scale + 4-byte high bit + 16 bytes
            GGMLQuantType.Q5_1: 24,  # + 2-byte min on top of Q5_0
            GGMLQuantType.Q8_0: 34,  # 2-byte scale + 32 bytes of int8 quants
        }.get(ggml_type, 0)

    def _dequantize_block(self, raw: bytes, ggml_type: int) -> torch.Tensor:
        """Dequantize a single GGML block to a ``float32`` torch tensor.

        Returns a 1-D tensor whose length matches the block's element
        count.  All quantization schemes below follow the ggml-py
        reference closely; the per-block scale (and, for asymmetric
        schemes, the per-block zero point) is multiplied back into the
        stored nibbles / bytes to recover the original float32 value.
        """
        if ggml_type == GGMLQuantType.F32:
            return torch.frombuffer(raw, dtype=torch.float32).clone()
        if ggml_type == GGMLQuantType.F16:
            return torch.frombuffer(raw, dtype=torch.float16).float()
        if ggml_type == GGMLQuantType.BF16:
            return torch.frombuffer(raw, dtype=torch.bfloat16).float()
        if ggml_type == GGMLQuantType.Q4_0:
            (scale,) = struct.unpack("<f", raw[:4])
            # The lower 4 bits of each byte are kept; we shift the
            # upper 4 bits back into place for the second half of the
            # 32-element block.  Nibbles are stored as ``x - 8`` to
            # exploit the unsigned range; we re-center them.
            qs = raw[4:18]
            out = torch.empty(32, dtype=torch.float32)
            for i in range(16):
                lo = (qs[i] & 0x0F) - 8
                hi = (qs[i] >> 4) - 8
                out[i] = lo * scale
                out[i + 16] = hi * scale
            return out
        if ggml_type == GGMLQuantType.Q4_1:
            (scale, mn) = struct.unpack("<ff", raw[:8])
            qs = raw[8:20]
            out = torch.empty(32, dtype=torch.float32)
            for i in range(16):
                lo = qs[i] & 0x0F
                hi = qs[i] >> 4
                out[i] = lo * scale + mn
                out[i + 16] = hi * scale + mn
            return out
        if ggml_type == GGMLQuantType.Q5_0:
            (scale,) = struct.unpack("<f", raw[:4])
            high_bits = struct.unpack("<I", raw[4:8])[0]
            qs = raw[8:20]
            out = torch.empty(32, dtype=torch.float32)
            for i in range(16):
                lo = ((qs[i] & 0x0F) | ((high_bits >> i) & 1) << 4) - 16
                hi = ((qs[i] >> 4) | ((high_bits >> (i + 16)) & 1) << 4) - 16
                out[i] = lo * scale
                out[i + 16] = hi * scale
            return out
        if ggml_type == GGMLQuantType.Q5_1:
            (scale, mn) = struct.unpack("<ff", raw[:8])
            high_bits = struct.unpack("<I", raw[8:12])[0]
            qs = raw[12:24]
            out = torch.empty(32, dtype=torch.float32)
            for i in range(16):
                lo = (qs[i] & 0x0F) | ((high_bits >> i) & 1) << 4
                hi = (qs[i] >> 4) | ((high_bits >> (i + 16)) & 1) << 4
                out[i] = lo * scale + mn
                out[i + 16] = hi * scale + mn
            return out
        if ggml_type == GGMLQuantType.Q8_0:
            (scale,) = struct.unpack("<f", raw[:4])
            qs = raw[4:36]
            out = torch.empty(32, dtype=torch.float32)
            for i in range(32):
                out[i] = (qs[i] - 128) * scale
            return out
        raise ValueError(f"unsupported ggml_type: {ggml_type!r}")

    def to_torch_tensor(
        self,
        raw: bytes,
        ggml_type: int,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Decode ``raw`` as a tensor with the given ``ggml_type`` and ``shape``.

        The decoded tensor is returned in ``float32`` for the quantized
        types (since the dequantized values are inherently fp32) and in
        the natural precision for ``F32`` / ``F16`` / ``BF16``.
        ``shape`` is used as-is - callers are expected to pass the
        dims from :attr:`GGUFTensorInfo.dims`.
        """
        block = self._block_size(ggml_type)
        block_bytes = self._block_bytes(ggml_type)
        expected_bytes = (len(shape) and 1) * 0
        # Compute the expected raw-byte count for a contiguous tensor
        # of the given shape.
        n_elems = 1
        for d in shape:
            n_elems *= int(d)
        n_blocks = n_elems // block
        expected_bytes = n_blocks * block_bytes
        if len(raw) < expected_bytes:
            raise ValueError(
                f"truncated GGUF tensor: have {len(raw)} bytes, "
                f"need {expected_bytes} for shape={shape!r} type={ggml_type!r}",
            )
        # Dequantize one block at a time and concatenate.  For F32 /
        # F16 / BF16 the per-block size is 1 element so this is
        # effectively a single ``torch.frombuffer`` call.
        if ggml_type in (GGMLQuantType.F32, GGMLQuantType.F16, GGMLQuantType.BF16):
            tensor = self._dequantize_block(raw[:expected_bytes], ggml_type)
        else:
            chunks = [
                self._dequantize_block(raw[i : i + block_bytes], ggml_type)
                for i in range(0, expected_bytes, block_bytes)
            ]
            tensor = torch.cat(chunks, dim=0) if chunks else torch.empty(0)
        return tensor.reshape(tuple(int(d) for d in shape))

    # ------------------------------------------------------------------
    # High-level loading
    # ------------------------------------------------------------------
    def load_state_dict(
        self,
        *,
        target_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode every tensor in the file into a ``{name: tensor}`` dict.

        Args:
            target_dtype: Optional dtype to cast every tensor to after
                decoding.  The default (``None``) preserves each
                tensor's natural precision (``float32`` for the
                quantized types).

        Returns:
            A ``state_dict``-style mapping ready to be fed into
            :func:`models.base.load_state_dict_with_renames`.
        """
        infos = self.tensor_info()
        data_offset = self._data_block_offset
        out: Dict[str, torch.Tensor] = {}
        for info in infos:
            self._fh.seek(data_offset + info.offset)
            n_elems = 1
            for d in info.dims:
                n_elems *= int(d)
            n_blocks = n_elems // self._block_size(info.ggml_type)
            raw = self._fh.read(n_blocks * self._block_bytes(info.ggml_type))
            tensor = self.to_torch_tensor(raw, info.ggml_type, info.shape)
            if target_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=target_dtype)
            out[info.name] = tensor
        return out

    def to_state_dict_for(
        self,
        model: nn.Module,
        *,
        key_map: Optional[Mapping[str, str]] = None,
        target_dtype: Optional[torch.dtype] = None,
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Load this file's tensors into ``model``.

        This is the convenience method that ties GGUF into the
        diffusers-style :class:`ModelMixin` flow.  Internally it walks
        the tensor-infos block, decodes every tensor, optionally
        applies ``key_map`` to rename GGUF names to the model's own
        parameter names, and finally delegates the weight copy to
        :func:`models.base.load_state_dict_with_renames`.

        Args:
            model: The target :class:`nn.Module`.
            key_map: Optional ``{gguf_name: model_name}`` rewrite table.
            target_dtype: Optional dtype to cast tensors to before
                the weight copy.
            strict: Forwarded to
                :func:`models.base.load_state_dict_with_renames`.

        Returns:
            ``(missing_keys, unexpected_keys)`` from the underlying
            load call.
        """
        state_dict = self.load_state_dict(target_dtype=target_dtype)
        # Lazy import: ``models.base`` may not be importable in every
        # environment this module ends up in.
        from models.base import load_state_dict_with_renames
        return load_state_dict_with_renames(
            model, state_dict, key_map, strict=strict,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import tempfile
    import os

    def _build_fake_gguf(path: str) -> None:
        """Write a minimal but spec-conformant empty GGUF v3 file."""
        with open(path, "wb") as fh:
            # Header: magic + version + tensor_count=0 + metadata_kv_count=0
            fh.write(b"GGUF")
            fh.write(struct.pack("<I", 3))        # version
            fh.write(struct.pack("<Q", 0))        # tensor_count
            fh.write(struct.pack("<Q", 0))        # metadata_kv_count
            # Tensor-infos block (empty) + data block (empty) follow;
            # both are zero-byte because the counts above are zero.

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tf:
        tmp_path = tf.name
    try:
        _build_fake_gguf(tmp_path)
        loader = GGUFLoader(tmp_path)
        try:
            assert loader.version == 3
            assert loader.tensor_count == 0
            assert loader.metadata_kv_count == 0
            assert loader.metadata() == {}
            assert loader.tensor_info() == []
            assert loader.load_state_dict() == {}
        finally:
            loader.close()
        print("[gguf_loader] smoke OK")
    finally:
        os.unlink(tmp_path)
