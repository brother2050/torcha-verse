"""Image-caption (image-text) dataset for the v0.6.x training stack.

The :class:`ImageTextDataset` loads image paths and their
associated captions from a JSONL/CSV/Parquet file.  Each row
contains a path to the image and a caption (or generic text
field); the dataset tokenises the caption and returns a
dictionary that includes the tokenised text plus the image
path.

When ``load_images=True`` the image pixels are also decoded
on access (via :mod:`PIL`, then converted to a ``torch.Tensor``
in ``(channels, height, width)`` layout with values in
``[0, 1]``).  The image-load path is opt-in because
:class:`PIL` is an optional dependency and the dataset is
often used in a streaming / path-only fashion.

File extensions:

* ``.jsonl`` -- one JSON object per line.
* ``.csv`` -- a CSV file with ``image_field`` / ``caption_field`` columns.
* ``.parquet`` / ``.pq`` -- a Parquet table with the same schema.

This module depends on :mod:`._base` for :class:`BaseDataset`
and on :mod:`._readers` for the format-agnostic row readers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ._base import BaseDataset, BaseTokenizer, PathLike
from ._readers import read_csv_rows, read_parquet_rows

__all__ = ["ImageTextDataset"]


class ImageTextDataset(BaseDataset):
    """Image-caption (image-text pair) dataset.

    Loads image paths and their associated captions from a
    JSONL/CSV/Parquet file.  Each line should be a JSON object
    with ``"image"`` (path) and ``"caption"`` (or ``"text"``)
    keys.

    Images are loaded lazily on access via :mod:`PIL` when
    available; otherwise the image path is returned and the
    caller is responsible for decoding.

    Args:
        file_path: Path to the metadata file.
        image_dir: Base directory for resolving relative image
            paths.
        tokenizer: Text tokenizer.
        max_length: Maximum caption length.
        caption_field: Name of the caption field in the JSON.
        image_field: Name of the image-path field in the JSON.
        load_images: When ``True`` load the image pixels on
            access (requires Pillow).
    """

    def __init__(
        self,
        file_path: PathLike,
        image_dir: Optional[PathLike] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        max_length: int = 512,
        caption_field: str = "caption",
        image_field: str = "image",
        load_images: bool = False,
    ) -> None:
        super().__init__(tokenizer=tokenizer, max_length=max_length)
        self.file_path: Path = Path(file_path).expanduser().resolve()
        self.image_dir: Optional[Path] = (
            Path(image_dir).expanduser().resolve() if image_dir else None
        )
        self.caption_field: str = caption_field
        self.image_field: str = image_field
        self.load_images: bool = load_images
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load image-caption pairs from the configured file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix == ".jsonl":
            raw_objects: List[Dict[str, Any]] = []
            with open(self.file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_objects.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        elif suffix == ".csv":
            raw_objects = read_csv_rows(self.file_path)
        elif suffix in (".parquet", ".pq"):
            raw_objects = read_parquet_rows(self.file_path)
        else:
            raise ValueError(
                f"ImageTextDataset does not support .{suffix} files; "
                "use .jsonl, .csv or .parquet."
            )

        for obj in raw_objects:
            image_path = obj.get(self.image_field, "")
            caption = obj.get(self.caption_field, obj.get("text", ""))
            if image_path and caption:
                self._examples.append(
                    {"image": image_path, "caption": caption}
                )

        self._logger.info(
            "Loaded %d image-caption pairs from %s.",
            len(self._examples),
            self.file_path,
        )

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve an image path relative to ``image_dir`` when needed.

        Args:
            image_path: The image path from the metadata.

        Returns:
            The resolved absolute :class:`~pathlib.Path`.
        """
        p = Path(image_path)
        if p.is_absolute() or self.image_dir is None:
            return p
        return (self.image_dir / image_path).resolve()

    def _load_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Load an image as a tensor (requires Pillow).

        Args:
            image_path: Path to the image file.

        Returns:
            A ``torch.Tensor`` of shape
            ``(channels, height, width)`` or ``None`` if Pillow
            is unavailable.
        """
        try:
            from PIL import Image
        except ImportError:
            return None

        img = Image.open(image_path).convert("RGB")
        import numpy as np

        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return arr

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an image-caption example.

        Returns a dictionary with tokenised ``input_ids``,
        ``attention_mask``, ``labels``, and the image ``path``
        (and ``image`` tensor when ``load_images`` is ``True``).
        """
        pair = self._examples[index]
        caption = pair["caption"]
        input_ids = self._encode(caption)
        attention_mask = self._make_attention_mask(input_ids)
        labels = list(input_ids)

        result: Dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "path": pair["image"],
        }

        if self.load_images:
            image_path = self._resolve_image_path(pair["image"])
            image = self._load_image(image_path)
            if image is not None:
                result["image"] = image

        return result
