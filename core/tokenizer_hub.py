"""Unified tokenizer management for TorchaVerse.

This module provides :class:`TokenizerHub`, a central registry that manages
tokenizers for every modality supported by the framework: text, audio,
image, and video.  All tokenizers share a uniform ``encode`` / ``decode``
interface defined by :class:`BaseTokenizer`, allowing downstream
components to treat different modalities polymorphically.

The built-in tokenizer implementations are lightweight, self-contained
PyTorch modules that do not depend on external model files.  They are
designed to be fully functional (round-trippable) while remaining simple
enough to serve as drop-in placeholders or testing fixtures.
"""

from __future__ import annotations

import abc
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "BaseTokenizer",
    "TextTokenizer",
    "AudioTokenizer",
    "ImageTokenizer",
    "VideoTokenizer",
    "TokenizerHub",
]


# ---------------------------------------------------------------------------
# BaseTokenizer
# ---------------------------------------------------------------------------
class BaseTokenizer(nn.Module, abc.ABC):
    """Abstract base class for all modality tokenizers.

    Every tokenizer must implement :meth:`encode` and :meth:`decode`.
    The ``encode`` method converts raw modality input into discrete or
    continuous token tensors, while ``decode`` reconstructs the original
    modality output from tokens.

    Args:
        vocab_size: Size of the discrete codebook / vocabulary.
        device: Device on which the tokenizer parameters reside.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
        self.vocab_size: int = vocab_size
        self._device_manager: DeviceManager = DeviceManager()
        self._target_device: torch.device = (
            torch.device(device) if isinstance(device, str) else device or self._device_manager.get_device()
        )

    @abc.abstractmethod
    def encode(self, inputs: Any, **kwargs: Any) -> torch.Tensor:
        """Encode raw modality input into tokens.

        Args:
            inputs: The raw input (text string, waveform tensor, image
                tensor, etc.).
            **kwargs: Modality-specific encoding options.

        Returns:
            A tensor of tokens (typically ``LongTensor`` for discrete
            tokenizers).
        """
        ...

    @abc.abstractmethod
    def decode(self, tokens: torch.Tensor, **kwargs: Any) -> Any:
        """Decode tokens back into the original modality output.

        Args:
            tokens: Token tensor produced by :meth:`encode`.
            **kwargs: Modality-specific decoding options.

        Returns:
            The reconstructed output (text string, waveform tensor, etc.).
        """
        ...

    @property
    def device(self) -> torch.device:
        """The device on which the tokenizer parameters reside."""
        return self._target_device

    def to_device(self, device: Union[str, torch.device]) -> "BaseTokenizer":
        """Move the tokenizer to ``device``."""
        self._target_device = torch.device(device) if isinstance(device, str) else device
        self.to(self._target_device)
        return self


# ---------------------------------------------------------------------------
# TextTokenizer
# ---------------------------------------------------------------------------
class TextTokenizer(BaseTokenizer):
    """Text tokenizer supporting BPE / SentencePiece / Unigram styles.

    This is a self-contained byte-level tokenizer that maps UTF-8 bytes
    to token ids.  It supports optional byte-pair merging via a simple
    merge table, special tokens, and padding/truncation.

    Args:
        vocab_size: Maximum vocabulary size (number of byte + merge + special
            tokens).
        max_length: Maximum sequence length.  Longer inputs are truncated.
        pad_token_id: Token id used for padding.
        bos_token_id: Beginning-of-sequence token id.
        eos_token_id: End-of-sequence token id.
        device: Device for any internal tensors.
    """

    #: Special token strings.
    PAD_TOKEN: str = "<pad>"
    BOS_TOKEN: str = "<bos>"
    EOS_TOKEN: str = "<eos>"
    UNK_TOKEN: str = "<unk>"

    def __init__(
        self,
        vocab_size: int = 256,
        max_length: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size, device=device)
        self.max_length: int = max_length
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id
        self.eos_token_id: int = eos_token_id
        self.unk_token_id: int = unk_token_id

        # Byte-level base vocabulary starts after the special tokens.
        self._byte_offset: int = 4  # 4 special tokens
        self._num_base_bytes: int = 256

        # Simple merge table (byte-pair merges).  Empty by default; can be
        # populated via ``train_merges``.
        self._merges: Dict[Tuple[int, int], int] = {}

        # Inverse vocab for decoding.
        self._id_to_token: Dict[int, str] = {
            pad_token_id: self.PAD_TOKEN,
            bos_token_id: self.BOS_TOKEN,
            eos_token_id: self.EOS_TOKEN,
            unk_token_id: self.UNK_TOKEN,
        }

    # ------------------------------------------------------------------
    def train_merges(self, texts: List[str], num_merges: int = 100) -> None:
        """Learn byte-pair merges from ``texts``.

        This is a simplified BPE training routine that counts adjacent
        byte-pair frequencies and greedily merges the most frequent pairs.

        Args:
            texts: Corpus of text strings.
            num_merges: Number of merge operations to perform.
        """
        # Convert corpus to byte sequences.
        sequences: List[List[int]] = []
        for text in texts:
            byte_ids = list(text.encode("utf-8"))
            sequences.append([b + self._byte_offset for b in byte_ids])

        next_id = self._byte_offset + self._num_base_bytes
        for _ in range(num_merges):
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            self._merges[best_pair] = next_id
            # Apply the merge to all sequences.
            new_sequences: List[List[int]] = []
            for seq in sequences:
                new_seq: List[int] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(next_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences
            next_id += 1

        self.vocab_size = next_id

    # ------------------------------------------------------------------
    def encode(
        self,
        inputs: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: bool = True,
    ) -> Union[torch.Tensor, List[List[int]]]:
        """Encode text into token ids.

        Args:
            inputs: A single string or a list of strings.
            add_special_tokens: Prepend BOS and append EOS tokens.
            padding: Pad sequences to the same length (when batching).
            truncation: Truncate sequences longer than ``max_length``.
            max_length: Override the tokenizer's default ``max_length``.
            return_tensors: When ``True`` return a ``LongTensor``;
                otherwise return a list of lists.

        Returns:
            Token ids as a tensor or list of lists.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        effective_max = max_length or self.max_length
        all_ids: List[List[int]] = []

        for text in inputs:
            byte_ids = list(text.encode("utf-8"))
            ids = [b + self._byte_offset for b in byte_ids]

            # Apply learned merges.
            ids = self._apply_merges(ids)

            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]

            if truncation and len(ids) > effective_max:
                ids = ids[:effective_max]
                if add_special_tokens:
                    ids[-1] = self.eos_token_id

            all_ids.append(ids)

        if padding:
            max_len = max(len(seq) for seq in all_ids)
            for seq in all_ids:
                seq.extend([self.pad_token_id] * (max_len - len(seq)))

        if return_tensors:
            if padding:
                return torch.tensor(all_ids, dtype=torch.long, device=self._target_device)
            # Unpadded: return as a padded tensor anyway for consistency.
            max_len = max(len(seq) for seq in all_ids)
            padded = [
                seq + [self.pad_token_id] * (max_len - len(seq)) for seq in all_ids
            ]
            return torch.tensor(padded, dtype=torch.long, device=self._target_device)

        return all_ids

    def _apply_merges(self, ids: List[int]) -> List[int]:
        """Apply learned BPE merges to a list of byte ids."""
        if not self._merges:
            return ids
        changed = True
        while changed:
            changed = False
            new_ids: List[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) in self._merges:
                    new_ids.append(self._merges[(ids[i], ids[i + 1])])
                    i += 2
                    changed = True
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    # ------------------------------------------------------------------
    def decode(
        self,
        tokens: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """Decode token ids back into text strings.

        Args:
            tokens: Token ids as a tensor or list.
            skip_special_tokens: Omit special tokens from the output.

        Returns:
            A single string (single input) or a list of strings (batch).
        """
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 1:
                tokens = [tokens.tolist()]
            else:
                tokens = tokens.tolist()

        if tokens and isinstance(tokens[0], int):
            tokens = [tokens]  # type: ignore[list-item]

        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
        results: List[str] = []

        for seq in tokens:  # type: ignore[union-attr]
            byte_vals: List[int] = []
            for tid in seq:
                if skip_special_tokens and tid in special_ids:
                    continue
                # Reverse-lookup merge tokens by expanding them.
                byte_vals.extend(self._id_to_bytes(tid))
            try:
                text = bytes(byte_vals).decode("utf-8", errors="replace")
            except Exception:
                text = ""
            results.append(text)

        return results[0] if len(results) == 1 else results

    def _id_to_bytes(self, token_id: int) -> List[int]:
        """Expand a token id back into its constituent byte values."""
        # Check if it's a merge token.
        for (a, b), merged_id in self._merges.items():
            if merged_id == token_id:
                return self._id_to_bytes(a) + self._id_to_bytes(b)
        # Base byte token.
        if token_id >= self._byte_offset and token_id < self._byte_offset + self._num_base_bytes:
            return [token_id - self._byte_offset]
        return []


# ---------------------------------------------------------------------------
# AudioTokenizer
# ---------------------------------------------------------------------------
class AudioTokenizer(BaseTokenizer):
    """Audio tokenizer using Residual Vector Quantization (RVQ).

    Inspired by EnCodec, this tokenizer encodes a waveform into discrete
    codes using a multi-layer vector quantizer.  Each quantizer layer
    captures the residual from the previous layer, enabling high-fidelity
    reconstruction with a small codebook.

    Args:
        vocab_size: Codebook size per quantizer layer.
        num_quantizers: Number of RVQ layers (depth).
        sample_rate: Expected sample rate of the input waveform.
        device: Device for internal tensors.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        num_quantizers: int = 4,
        sample_rate: int = 24000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size, device=device)
        self.num_quantizers: int = num_quantizers
        self.sample_rate: int = sample_rate

        # Learnable codebooks: (num_quantizers, vocab_size, 1)
        # We use 1-D codes for simplicity; real EnCodec uses higher dims.
        self.codebooks = nn.ParameterDict()
        for i in range(num_quantizers):
            self.codebooks[f"layer_{i}"] = nn.Parameter(
                torch.randn(vocab_size, 1) * 0.1
            )

    # ------------------------------------------------------------------
    def encode(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        frame_length: int = 320,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode a waveform into discrete RVQ tokens.

        Args:
            inputs: Waveform tensor of shape ``(batch, samples)`` or
                ``(samples,)``.
            frame_length: Number of samples per frame (hop size).

        Returns:
            Token tensor of shape ``(batch, num_quantizers, num_frames)``
            containing discrete codebook indices.
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected tensor input, got {type(inputs).__name__}.")

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        batch_size, num_samples = inputs.shape
        inputs = inputs.to(self._target_device)

        # Pad to a multiple of frame_length.
        pad_len = (frame_length - num_samples % frame_length) % frame_length
        if pad_len > 0:
            inputs = F.pad(inputs, (0, pad_len))

        # Frame the waveform: (batch, num_frames, frame_length)
        num_frames = inputs.shape[1] // frame_length
        frames = inputs.view(batch_size, num_frames, frame_length)

        # Compute frame energy as the feature to quantize.
        features = frames.mean(dim=-1, keepdim=True)  # (batch, num_frames, 1)

        tokens = torch.zeros(
            batch_size, self.num_quantizers, num_frames,
            dtype=torch.long, device=self._target_device,
        )
        residual = features

        for layer_idx in range(self.num_quantizers):
            codebook = self.codebooks[f"layer_{layer_idx}"]  # (vocab_size, 1)
            # Distance: (batch, num_frames, vocab_size)
            dist = torch.cdist(residual, codebook.unsqueeze(0))
            indices = dist.argmin(dim=-1)  # (batch, num_frames)
            tokens[:, layer_idx] = indices
            # Quantized values.
            quantized = codebook[indices]  # (batch, num_frames, 1)
            residual = residual - quantized

        return tokens

    # ------------------------------------------------------------------
    def decode(
        self,
        tokens: torch.Tensor,
        frame_length: int = 320,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode RVQ tokens back into a waveform.

        Args:
            tokens: Token tensor of shape ``(batch, num_quantizers, num_frames)``.
            frame_length: Number of samples per frame (must match encoding).

        Returns:
            Reconstructed waveform of shape ``(batch, samples)``.
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected tokens of shape (batch, num_quantizers, num_frames), "
                f"got {tuple(tokens.shape)}."
            )

        tokens = tokens.to(self._target_device)
        batch_size, num_q, num_frames = tokens.shape

        # Reconstruct features by summing quantized values from each layer.
        features = torch.zeros(
            batch_size, num_frames, 1, device=self._target_device
        )
        for layer_idx in range(min(num_q, self.num_quantizers)):
            codebook = self.codebooks[f"layer_{layer_idx}"]
            indices = tokens[:, layer_idx]
            features = features + codebook[indices]

        # Expand features to waveform frames.
        waveform = features.expand(-1, -1, frame_length).reshape(
            batch_size, -1
        )
        return waveform


# ---------------------------------------------------------------------------
# ImageTokenizer
# ---------------------------------------------------------------------------
class ImageTokenizer(BaseTokenizer):
    """Image tokenizer using VQ-VAE style codebook quantization.

    Encodes an image into a grid of discrete tokens by patchifying the
    image, projecting each patch to a feature vector, and quantizing via
    a learnable codebook (nearest-neighbor lookup).

    Args:
        vocab_size: Codebook size.
        patch_size: Spatial size of each patch (patch_size x patch_size).
        channels: Number of image channels (3 for RGB).
        device: Device for internal tensors.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        patch_size: int = 16,
        channels: int = 3,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size, device=device)
        self.patch_size: int = patch_size
        self.channels: int = channels

        # Patch embedding: flatten patch pixels to a 1-D feature for simplicity.
        self.patch_embed = nn.Linear(patch_size * patch_size * channels, 1)
        # Codebook: (vocab_size, 1)
        self.codebook = nn.Parameter(torch.randn(vocab_size, 1) * 0.1)

    # ------------------------------------------------------------------
    def encode(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode an image into a grid of discrete tokens.

        Args:
            inputs: Image tensor of shape ``(batch, channels, height, width)``
                or ``(channels, height, width)``.

        Returns:
            Token tensor of shape ``(batch, grid_h, grid_w)`` containing
            discrete codebook indices.
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected tensor input, got {type(inputs).__name__}.")

        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        inputs = inputs.to(self._target_device).float()
        batch_size, channels, height, width = inputs.shape

        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}."
            )

        # Pad to a multiple of patch_size.
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h))

        height_p = inputs.shape[2]
        width_p = inputs.shape[3]
        grid_h = height_p // self.patch_size
        grid_w = width_p // self.patch_size

        # Patchify: (batch, grid_h, grid_w, patch_size*patch_size*channels)
        patches = inputs.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        # (batch, channels, grid_h, grid_w, patch_size, patch_size)
        patches = patches.contiguous().view(
            batch_size, grid_h, grid_w, self.patch_size * self.patch_size * channels
        )

        # Project patches to 1-D features.
        features = self.patch_embed(patches)  # (batch, grid_h, grid_w, 1)

        # Quantize via nearest-neighbor in the codebook.
        flat = features.reshape(-1, 1)  # (batch*grid_h*grid_w, 1)
        dist = torch.cdist(flat, self.codebook.unsqueeze(0))
        indices = dist.argmin(dim=-1)  # (batch*grid_h*grid_w,)
        tokens = indices.reshape(batch_size, grid_h, grid_w)

        return tokens

    # ------------------------------------------------------------------
    def decode(
        self,
        tokens: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode a token grid back into an image.

        Args:
            tokens: Token tensor of shape ``(batch, grid_h, grid_w)``.
            target_size: Optional ``(height, width)`` to crop/pad the output.

        Returns:
            Reconstructed image tensor of shape
            ``(batch, channels, height, width)``.
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(self._target_device)

        batch_size, grid_h, grid_w = tokens.shape
        # Look up codebook values for each token.
        values = self.codebook[tokens.reshape(-1)]  # (batch*grid_h*grid_w, 1)
        values = values.reshape(batch_size, grid_h, grid_w, 1)

        # Expand each token to a uniform patch.
        patch = values.expand(
            -1, -1, -1, self.patch_size * self.patch_size * self.channels
        )
        # Reshape to image.
        image = patch.reshape(
            batch_size, grid_h, grid_w, self.channels, self.patch_size, self.patch_size
        )
        image = image.permute(0, 3, 1, 4, 2, 5).contiguous()
        image = image.reshape(
            batch_size, self.channels, grid_h * self.patch_size, grid_w * self.patch_size
        )

        if target_size is not None:
            th, tw = target_size
            image = image[:, :, :th, :tw]

        return image


# ---------------------------------------------------------------------------
# VideoTokenizer
# ---------------------------------------------------------------------------
class VideoTokenizer(BaseTokenizer):
    """Video tokenizer using spatio-temporal patch embedding.

    Encodes a video clip into discrete tokens by dividing the video into
    3D spatio-temporal patches and quantizing each patch via a codebook.

    Args:
        vocab_size: Codebook size.
        temporal_patch: Number of frames per temporal patch.
        spatial_patch: Spatial patch size (square).
        channels: Number of image channels.
        device: Device for internal tensors.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        temporal_patch: int = 2,
        spatial_patch: int = 16,
        channels: int = 3,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size, device=device)
        self.temporal_patch: int = temporal_patch
        self.spatial_patch: int = spatial_patch
        self.channels: int = channels

        # 3D patch embedding: flatten to 1-D feature.
        feat_dim = temporal_patch * spatial_patch * spatial_patch * channels
        self.patch_embed = nn.Linear(feat_dim, 1)
        self.codebook = nn.Parameter(torch.randn(vocab_size, 1) * 0.1)

    # ------------------------------------------------------------------
    def encode(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode a video clip into a grid of discrete tokens.

        Args:
            inputs: Video tensor of shape
                ``(batch, channels, frames, height, width)``.

        Returns:
            Token tensor of shape ``(batch, t_grid, h_grid, w_grid)``.
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected tensor input, got {type(inputs).__name__}.")

        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self._target_device).float()

        batch_size, channels, frames, height, width = inputs.shape

        # Pad temporal and spatial dimensions.
        pad_t = (self.temporal_patch - frames % self.temporal_patch) % self.temporal_patch
        pad_h = (self.spatial_patch - height % self.spatial_patch) % self.spatial_patch
        pad_w = (self.spatial_patch - width % self.spatial_patch) % self.spatial_patch
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h, 0, pad_t))

        frames_p = inputs.shape[2]
        height_p = inputs.shape[3]
        width_p = inputs.shape[4]
        t_grid = frames_p // self.temporal_patch
        h_grid = height_p // self.spatial_patch
        w_grid = width_p // self.spatial_patch

        feat_dim = self.temporal_patch * self.spatial_patch * self.spatial_patch * channels

        # Patchify into 3D blocks.
        patches = []
        for bt in range(t_grid):
            for bh in range(h_grid):
                for bw in range(w_grid):
                    patch = inputs[
                        :,
                        :,
                        bt * self.temporal_patch : (bt + 1) * self.temporal_patch,
                        bh * self.spatial_patch : (bh + 1) * self.spatial_patch,
                        bw * self.spatial_patch : (bw + 1) * self.spatial_patch,
                    ]
                    patches.append(patch.reshape(batch_size, -1))

        patches_tensor = torch.stack(patches, dim=1)  # (batch, t_grid*h_grid*w_grid, feat_dim)
        features = self.patch_embed(patches_tensor)  # (batch, num_patches, 1)

        # Quantize.
        flat = features.reshape(-1, 1)
        dist = torch.cdist(flat, self.codebook.unsqueeze(0))
        indices = dist.argmin(dim=-1)
        tokens = indices.reshape(batch_size, t_grid, h_grid, w_grid)

        return tokens

    # ------------------------------------------------------------------
    def decode(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode a token grid back into a video clip.

        Args:
            tokens: Token tensor of shape ``(batch, t_grid, h_grid, w_grid)``.

        Returns:
            Reconstructed video tensor of shape
            ``(batch, channels, frames, height, width)``.
        """
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(self._target_device)

        batch_size, t_grid, h_grid, w_grid = tokens.shape
        values = self.codebook[tokens.reshape(-1)]  # (num_tokens, 1)
        feat_dim = self.temporal_patch * self.spatial_patch * self.spatial_patch * self.channels

        # Expand each token to a full patch.
        patches = values.expand(-1, feat_dim)  # (num_tokens, feat_dim)
        patches = patches.reshape(
            batch_size, t_grid, h_grid, w_grid,
            self.channels, self.temporal_patch, self.spatial_patch, self.spatial_patch,
        )
        # Permute to (batch, channels, t_grid, temporal_patch, h_grid, spatial_patch, w_grid, spatial_patch)
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        video = patches.reshape(
            batch_size,
            self.channels,
            t_grid * self.temporal_patch,
            h_grid * self.spatial_patch,
            w_grid * self.spatial_patch,
        )
        return video


# ---------------------------------------------------------------------------
# TokenizerHub
# ---------------------------------------------------------------------------
class TokenizerHub:
    """Central registry for all modality tokenizers.

    Provides a unified API to register, retrieve, and manage tokenizers
    across text, audio, image, and video modalities.  Built-in tokenizer
    classes are registered automatically on first instantiation.

    Example:
        >>> hub = TokenizerHub()
        >>> tok = hub.get_tokenizer("text", vocab_size=256)
        >>> ids = tok.encode("Hello, world!")
        >>> text = tok.decode(ids)
    """

    _instance: Optional["TokenizerHub"] = None
    _initialized: bool = False

    #: Built-in tokenizer name to class mapping.
    _BUILTIN: Dict[str, Type[BaseTokenizer]] = {
        "text": TextTokenizer,
        "audio": AudioTokenizer,
        "image": ImageTokenizer,
        "video": VideoTokenizer,
    }

    def __new__(cls, *args: Any, **kwargs: Any) -> "TokenizerHub":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._registry: Dict[str, Type[BaseTokenizer]] = {}
        self._instances: Dict[str, BaseTokenizer] = {}
        self._logger = get_logger(self.__class__.__name__)
        self._device_manager: DeviceManager = DeviceManager()

        # Register built-in tokenizers.
        for name, cls in self._BUILTIN.items():
            self.register_tokenizer(name, cls)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_tokenizer(
        self,
        name: str,
        tokenizer_class: Type[BaseTokenizer],
    ) -> None:
        """Register a tokenizer class under ``name``.

        Args:
            name: Unique identifier (e.g. ``"text"``, ``"audio"``).
            tokenizer_class: A subclass of :class:`BaseTokenizer`.

        Raises:
            TypeError: If ``tokenizer_class`` is not a subclass of
                :class:`BaseTokenizer`.
        """
        if not (isinstance(tokenizer_class, type) and issubclass(tokenizer_class, BaseTokenizer)):
            raise TypeError(
                f"tokenizer_class must be a subclass of BaseTokenizer, got "
                f"{tokenizer_class!r}."
            )
        key = name.strip().lower()
        self._registry[key] = tokenizer_class
        # Invalidate any cached instance.
        self._instances.pop(key, None)
        self._logger.debug("Registered tokenizer '%s' -> %s", key, tokenizer_class.__name__)

    def unregister_tokenizer(self, name: str) -> bool:
        """Remove a tokenizer from the registry."""
        key = name.strip().lower()
        removed = self._registry.pop(key, None) is not None
        self._instances.pop(key, None)
        return removed

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_tokenizer(self, name: str, **kwargs: Any) -> BaseTokenizer:
        """Retrieve (or create) a tokenizer instance.

        When ``kwargs`` are provided a new instance is always created.
        Otherwise a cached instance is returned if available.

        Args:
            name: Registered tokenizer name.
            **kwargs: Constructor arguments forwarded to the tokenizer.

        Returns:
            A :class:`BaseTokenizer` instance.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        key = name.strip().lower()
        if key not in self._registry:
            raise KeyError(
                f"Tokenizer '{name}' is not registered. "
                f"Available: {', '.join(self.list_available()) or '(none)'}."
            )

        # If kwargs are given, always create a fresh instance.
        if kwargs:
            tokenizer = self._registry[key](**kwargs)
            self._instances[key] = tokenizer
            return tokenizer

        # Return cached instance or create a default one.
        if key not in self._instances:
            self._instances[key] = self._registry[key]()
        return self._instances[key]

    def list_available(self) -> List[str]:
        """Return a sorted list of registered tokenizer names."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Return ``True`` if ``name`` is a registered tokenizer."""
        return name.strip().lower() in self._registry

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False
