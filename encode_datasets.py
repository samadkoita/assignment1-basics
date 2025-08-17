#!/usr/bin/env python3
"""
Simple script to encode TinyStories train and validation datasets into token IDs.
Saves as NumPy arrays with uint16 datatype.
Uses encode_iterable for proper handling of special tokens across line boundaries.
"""

import numpy as np
import time
from cs336_basics.tokenization import Tokenizer

def encode_dataset(tokenizer, input_path, output_path):
    """Encode a dataset file and save as numpy array."""
    print(f"\nEncoding {input_path}...")
    start_time = time.time()
    
    # Collect all tokens using encode_iterable
    all_tokens = []
    tokens_processed = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for token_id in tokenizer.encode_iterable(f, chunk_size=1024*1024):  # 1MB chunks
            all_tokens.append(token_id)
            tokens_processed += 1
            
            if tokens_processed % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {tokens_processed:,} tokens "
                      f"({elapsed:.1f}s, {tokens_processed/elapsed:.1f} tokens/sec)")
    
    # Convert to numpy array with uint16
    token_array = np.array(all_tokens, dtype=np.uint16)
    
    # Save to file
    np.save(output_path, token_array)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {input_path}:")
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Throughput: {tokens_processed/elapsed:.1f} tokens/sec")
    print(f"  Output saved to: {output_path}")
    print(f"  File size: {token_array.nbytes / (1024*1024):.1f} MB")
    
    return token_array

def main():
    print("="*70)
    print("ENCODING TINYSTORIES DATASETS TO TOKEN IDS")
    print("="*70)
    
    # Load tokenizer
    print("\nLoading TinyStories tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_path="data/vocab_TinyStories_final",
        merges_path="data/merges_TinyStories_final",
        special_tokens=["<|endoftext|>"]
    )
    print(f"Tokenizer loaded: {len(tokenizer.vocab)} vocab size")
    print(f"Using encode_iterable with 1MB chunk size")
    
    # Encode validation set (smaller, good for testing)
    val_tokens = encode_dataset(
        tokenizer,
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        output_path="data/TinyStories_valid_tokens.npy"
    )
    
    # Encode training set (this will take a while)
    train_tokens = encode_dataset(
        tokenizer,
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        output_path="data/TinyStories_train_tokens.npy"
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nValidation set:")
    print(f"  Shape: {val_tokens.shape}")
    print(f"  Dtype: {val_tokens.dtype}")
    print(f"  Min token ID: {val_tokens.min()}")
    print(f"  Max token ID: {val_tokens.max()}")
    
    print(f"\nTraining set:")
    print(f"  Shape: {train_tokens.shape}")
    print(f"  Dtype: {train_tokens.dtype}")
    print(f"  Min token ID: {train_tokens.min()}")
    print(f"  Max token ID: {train_tokens.max()}")
    
    print("\nFiles created:")
    print("  - data/TinyStories_valid_tokens.npy")
    print("  - data/TinyStories_train_tokens.npy")
    
    print("\nTo load these arrays later:")
    print("  tokens = np.load('data/TinyStories_train_tokens.npy')")

if __name__ == "__main__":
    main()