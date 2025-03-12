#!/usr/bin/env python
"""
Optimized tokenizer implementation for llama3.np

This script tests the hypothesis that replacing the list.index() lookup
in the tokenizer with a dictionary-based lookup will improve performance.
"""

import numpy as np
import time
import sys
from tokenizer import Tokenizer as OriginalTokenizer
import json

class OptimizedTokenizer:
    def __init__(self, tokenizer_path):
        """
        Initialize the tokenizer with a mapping from tokens to indices.
        The original tokenizer uses a list and searches with .index().
        This optimized version uses a dictionary for O(1) lookups.
        """
        # Load the original tokenizer first
        self.original_tokenizer = OriginalTokenizer(tokenizer_path)
        
        # Copy key attributes from the original tokenizer
        self.eos_id = self.original_tokenizer.eos_id
        self.bos_id = self.original_tokenizer.bos_id
        self.vocab = self.original_tokenizer.vocab
        self.scores = self.original_tokenizer.scores
        
        # Create a dictionary mapping from tokens to their first occurrence index
        # This is necessary because the vocabulary contains duplicates
        # and the original implementation uses .index() which finds the first occurrence
        self.token_to_id = {}
        for i, token in enumerate(self.vocab):
            # Only add if not already in the dictionary (keep first occurrence)
            if token not in self.token_to_id:
                self.token_to_id[token] = i
    
    def str_lookup(self, s):
        """
        Optimized token lookup using a dictionary instead of list.index()
        """
        return self.token_to_id.get(s, -1)
    
    def encode(self, text, add_bos=True, add_eos=False):
        """
        Encode a string into tokens, using the optimized lookup.
        This implementation follows the original algorithm but uses
        the optimized str_lookup method.
        """
        tokens = []
        for pos, char in enumerate(text):
            id = self.str_lookup(char)
            if id >= 0:
                tokens.append(id)
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for i in range(len(tokens) - 1):
                # Check if we can merge the pair (tokens[i], tokens[i+1])
                string = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                id = self.str_lookup(string)
                if id != -1 and self.scores[id] > best_score:
                    best_score = self.scores[id]
                    best_id = id
                    best_idx = i

            if best_idx == -1:
                break

            # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id
            # Delete token at position best_idx+1, shift the entire sequence back 1
            tokens = tokens[0: best_idx + 1] + tokens[best_idx + 2:]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens
    
    def decode(self, ids):
        """
        Decode a list of tokens into a string.
        This is unchanged from the original tokenizer since it's already optimal.
        """
        res = []
        for i in ids:
            token = self.vocab[i]
            res.append(token)
        text = "".join(res)
        text = text.strip("<s>").strip("</s>")
        return text

def benchmark_tokenizers(text, iterations=100, debug=False):
    """Benchmark the original and optimized tokenizers."""
    # Initialize tokenizers
    tokenizer_path = "tokenizer.model.np"
    original_tokenizer = OriginalTokenizer(tokenizer_path)
    optimized_tokenizer = OptimizedTokenizer(tokenizer_path)
    
    # Verify dictionary creation
    if debug:
        print(f"Original vocab size: {len(original_tokenizer.vocab)}")
        print(f"Optimized token_to_id size: {len(optimized_tokenizer.token_to_id)}")
        
        # Check if we're missing entries or have duplicates
        if len(original_tokenizer.vocab) != len(optimized_tokenizer.token_to_id):
            num_unique_tokens = len(set(original_tokenizer.vocab))
            print(f"Original vocab has {len(original_tokenizer.vocab)} entries, but only {num_unique_tokens} unique tokens")
            
            # Find duplicates in the original vocab
            token_counts = {}
            for i, token in enumerate(original_tokenizer.vocab):
                if token in token_counts:
                    token_counts[token].append(i)
                else:
                    token_counts[token] = [i]
            
            # Print duplicates
            print("Duplicate tokens:")
            for token, indices in token_counts.items():
                if len(indices) > 1:
                    print(f"  '{token}' appears at indices: {indices}")
        
        # Sample a few tokens to verify they map correctly
        sample_tokens = ["the", "and", "a", "to", "in"]
        for token in sample_tokens:
            orig_id = original_tokenizer.str_lookup(token)
            opt_id = optimized_tokenizer.str_lookup(token)
            print(f"Token '{token}': Original ID = {orig_id}, Optimized ID = {opt_id}, Match: {orig_id == opt_id}")
    
    # Test simple lookup functionality
    if debug:
        test_words = ["hello", "world", "the", "a", "once", "upon"]
        print("\nToken lookup test:")
        for word in test_words:
            orig_id = original_tokenizer.str_lookup(word)
            opt_id = optimized_tokenizer.str_lookup(word)
            match = orig_id == opt_id
            print(f"'{word}': Original ID = {orig_id}, Optimized ID = {opt_id}, Match: {match}")
    
    # Warm up and compare
    original_tokens = original_tokenizer.encode(text)
    optimized_tokens = optimized_tokenizer.encode(text)
    
    # Convert to simple lists for comparison
    original_list = list(original_tokens)
    optimized_list = list(optimized_tokens)
    
    # Make sure they produce the same tokens
    tokens_match = original_list == optimized_list
    if not tokens_match and debug:
        print("Warning: Tokenizers produced different results!")
        print(f"Original ({len(original_list)}): {original_list}")
        print(f"Optimized ({len(optimized_list)}): {optimized_list}")
        
        # Find where they differ
        min_len = min(len(original_list), len(optimized_list))
        for i in range(min_len):
            if original_list[i] != optimized_list[i]:
                print(f"First difference at position {i}: Original = {original_list[i]}, Optimized = {optimized_list[i]}")
                original_token = original_tokenizer.vocab[original_list[i]]
                optimized_token = optimized_tokenizer.vocab[optimized_list[i]]
                print(f"Tokens: Original = '{original_token}', Optimized = '{optimized_token}'")
                break
    
    # Benchmark original implementation (encode)
    original_encode_times = []
    for _ in range(iterations):
        start = time.time()
        original_tokenizer.encode(text)
        original_encode_times.append(time.time() - start)
    
    # Benchmark optimized implementation (encode)
    optimized_encode_times = []
    for _ in range(iterations):
        start = time.time()
        optimized_tokenizer.encode(text)
        optimized_encode_times.append(time.time() - start)
    
    # Benchmark original implementation (decode)
    original_decode_times = []
    for _ in range(iterations):
        start = time.time()
        original_tokenizer.decode(original_tokens)
        original_decode_times.append(time.time() - start)
    
    # Benchmark optimized implementation (decode)
    optimized_decode_times = []
    for _ in range(iterations):
        start = time.time()
        optimized_tokenizer.decode(optimized_tokens)
        optimized_decode_times.append(time.time() - start)
    
    # Calculate statistics
    avg_original_encode = np.mean(original_encode_times) * 1000  # ms
    avg_optimized_encode = np.mean(optimized_encode_times) * 1000  # ms
    encode_speedup = avg_original_encode / avg_optimized_encode if avg_optimized_encode > 0 else 0
    
    avg_original_decode = np.mean(original_decode_times) * 1000  # ms
    avg_optimized_decode = np.mean(optimized_decode_times) * 1000  # ms
    decode_speedup = avg_original_decode / avg_optimized_decode if avg_optimized_decode > 0 else 0
    
    # Print results
    print(f"Tokenizer Benchmark Results (text length: {len(text)}):")
    print(f"Original encode: {avg_original_encode:.3f} ms")
    print(f"Optimized encode: {avg_optimized_encode:.3f} ms")
    print(f"Encode speedup: {encode_speedup:.2f}x")
    print(f"Original decode: {avg_original_decode:.3f} ms")
    print(f"Optimized decode: {avg_optimized_decode:.3f} ms")
    print(f"Decode speedup: {decode_speedup:.2f}x")
    print(f"Tokens match: {'Yes' if tokens_match else 'No'}")
    
    return {
        "original_encode_ms": avg_original_encode,
        "optimized_encode_ms": avg_optimized_encode,
        "encode_speedup": encode_speedup,
        "original_decode_ms": avg_original_decode,
        "optimized_decode_ms": avg_optimized_decode,
        "decode_speedup": decode_speedup,
        "tokens_match": tokens_match
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark tokenizer implementations")
    parser.add_argument("--text", type=str, default="Once upon a time in a land far, far away, there lived a brave knight who dreamed of adventure.", 
                        help="Text to tokenize")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--long", action="store_true", help="Use a longer text sample for benchmarking")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    if args.long:
        # Use a longer text sample for more realistic benchmarking
        text = """Once upon a time in a land far, far away, there lived a brave knight who dreamed of adventure.
        Every day, he would polish his armor and sharpen his sword, preparing for the moment when he would 
        be called to fulfill his destiny. The villagers admired him for his dedication and courage, even though
        some whispered that he was merely chasing fairy tales. One morning, as the sun painted the sky with hues
        of orange and pink, a messenger arrived at the knight's door. "The king summons you," the messenger said,
        handing him a scroll bearing the royal seal. The knight's heart raced with excitement as he unrolled the
        parchment. This was it—the adventure he had been waiting for."""
    else:
        text = args.text
    
    benchmark_tokenizers(text, iterations=args.iter, debug=args.debug)