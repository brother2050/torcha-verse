"""Basic text generation example.

This script demonstrates how to use TorchaVerse to:
1. Create a small TransformerDecoder model from scratch.
2. Generate text with top-k / top-p sampling.
3. Compute text embeddings.
4. Stream tokens one by one.

Run with::

    python examples/basic_text_gen.py
"""

from __future__ import annotations

import sys
import os

# Add project root to path so we can import torcha_verse modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.text.transformer import TransformerDecoder


def main() -> None:
    """Run the basic text generation demo."""

    print("=" * 60)
    print("TorchaVerse — Basic Text Generation Demo")
    print("=" * 60)

    # --- 1. Create a small Transformer model ---
    print("\n[1] Creating a small TransformerDecoder model...")
    model = TransformerDecoder(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,           # GQA: fewer KV heads
        intermediate_size=512,
        max_seq_len=512,
        norm_type="rmsnorm",
        activation="swiglu",
    )
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # --- 2. Generate tokens ---
    print("\n[2] Generating tokens...")
    prompt_ids = torch.randint(0, 1000, (1, 10), device=device)
    print(f"    Prompt shape: {prompt_ids.shape}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            max_tokens=32,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
    print(f"    Output shape: {output_ids.shape}")
    print(f"    Generated {output_ids.shape[1] - prompt_ids.shape[1]} new tokens")

    # --- 3. Compute embeddings ---
    print("\n[3] Computing text embeddings...")
    input_ids = torch.randint(0, 1000, (2, 20), device=device)
    with torch.no_grad():
        logits = model(input_ids)
    # Use mean pooling of the last hidden state as embedding.
    # logits shape: (batch, seq_len, vocab_size)
    # For embedding, we use the hidden state before the LM head.
    # In this simplified example, we use the logits directly.
    embeddings = logits.mean(dim=1)  # (batch, vocab_size)
    print(f"    Input shape: {input_ids.shape}")
    print(f"    Embedding shape: {embeddings.shape}")

    # --- 4. Forward pass with attention mask ---
    print("\n[4] Forward pass with attention mask...")
    input_ids = torch.randint(0, 1000, (4, 32), device=device)
    attention_mask = torch.ones(4, 32, device=device)
    attention_mask[0, 16:] = 0  # Mask out second half of first sequence

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    print(f"    Batch logits shape: {logits.shape}")

    # --- 5. KV Cache generation ---
    print("\n[5] Generation with KV Cache...")
    prompt = torch.randint(0, 1000, (1, 8), device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids=prompt,
            max_tokens=16,
            temperature=0.7,
            top_k=20,
        )
    print(f"    Prompt length: {prompt.shape[1]}")
    print(f"    Output length: {output.shape[1]}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
