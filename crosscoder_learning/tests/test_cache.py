import os
import tempfile
import torch as th
import pytest
from dictionary_learning.cache import ActivationShard, save_shard
from datasets import Dataset
from nnsight import LanguageModel
from dictionary_learning.cache import ActivationCache
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_activation_shard_float16(temp_dir):
    # Create random activations
    dtype = th.float16
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.equal(activations, loaded_activations)


def test_activation_shard_bfloat16(temp_dir):
    # Create random activations
    dtype = th.bfloat16
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.equal(activations, loaded_activations)


def test_activation_shard_float32(temp_dir):
    # Create random activations
    dtype = th.float32
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.equal(activations, loaded_activations)


def test_activation_shard_int8(temp_dir):
    # Create random activations
    dtype = th.int8
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randint(-128, 127, shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.all(activations == loaded_activations)


def test_activation_shard_indexing(temp_dir):
    # Create random activations
    dtype = th.float32
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Test different indexing patterns
    # Single index
    assert th.equal(activations[5], shard[5])

    # Slice
    assert th.equal(activations[10:20], shard[10:20])

    # List of indices
    indices = [5, 10, 15, 20]
    assert th.equal(activations[indices], shard[indices])


def test_activation_shard_multiple_shards(temp_dir):
    # Create and save multiple shards
    dtype = th.float32
    shape = (100, 128)  # 100 tokens, 128 dimensions

    # Create and save shard 0
    activations0 = th.randn(shape, dtype=dtype)
    save_shard(activations0, temp_dir, 0, "test", "out")

    # Create and save shard 1
    activations1 = th.randn(shape, dtype=dtype)
    save_shard(activations1, temp_dir, 1, "test", "out")

    # Load shards
    shard0 = ActivationShard(temp_dir, 0)
    shard1 = ActivationShard(temp_dir, 1)

    # Verify contents
    assert th.equal(activations0, shard0[:])
    assert th.equal(activations1, shard1[:])


def test_activation_cache_with_normalizer(temp_dir):
    """Test ActivationCache collection and normalizer against direct model activations."""
    # Set flag to handle meta tensors properly
    th.fx.experimental._config.meta_nonzero_assume_all_nonzero = True

    # Skip test if CUDA not available to avoid device mapping issues
    if not th.cuda.is_available():
        pytest.skip("CUDA not available, skipping test to avoid device mapping issues")

    # Test strings
    test_strings = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological brain structures.",
        "Deep learning has revolutionized computer vision and natural language processing.",
    ]

    # Use the list directly - it already implements __len__ and __getitem__
    dataset = test_strings

    # Load GPT-2 model - use auto device mapping but force concrete tensors
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", device_map="auto", torch_dtype=th.float32
    )
    model = LanguageModel(model, torch_dtype=th.float32, tokenizer=tokenizer)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Get a transformer block to extract activations from
    target_layer = model.transformer.h[6]  # Middle layer of GPT-2
    submodule_name = "transformer_h_6"

    # Parameters for activation collection
    batch_size = 2
    context_len = 64
    d_model = 768  # GPT-2 hidden size

    # Collect activations using ActivationCache
    ActivationCache.collect(
        data=dataset,
        submodules=(target_layer,),
        submodule_names=(submodule_name,),
        model=model,
        store_dir=temp_dir,
        batch_size=batch_size,
        context_len=context_len,
        shard_size=1000,  # Small shard size for testing
        d_model=d_model,
        io="out",
        max_total_tokens=10000,
        store_tokens=True,
    )

    # Load the cached activations
    cache = ActivationCache(temp_dir, submodule_name + "_out")

    # Collect activations directly from model for comparison
    direct_activations = []
    direct_tokens = []

    for i in range(0, len(test_strings), batch_size):
        batch_texts = test_strings[i : i + batch_size]

        # Tokenize
        tokens = model.tokenizer(
            batch_texts,
            max_length=context_len,
            truncation=True,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )

        # Get activations directly
        with model.trace(tokens):
            layer_output = target_layer.output[0].save()

        # Extract valid tokens (non-padding)
        attention_mask = tokens["attention_mask"]
        valid_activations = layer_output.reshape(-1, d_model)[
            attention_mask.reshape(-1).bool()
        ]
        valid_tokens = tokens["input_ids"].reshape(-1)[
            attention_mask.reshape(-1).bool()
        ]

        direct_activations.append(valid_activations.cpu())
        direct_tokens.append(valid_tokens.cpu())

    # Concatenate direct activations
    direct_activations = th.cat(direct_activations, dim=0)
    direct_tokens = th.cat(direct_tokens, dim=0)

    # Test that we have the same number of activations
    assert (
        len(cache) == direct_activations.shape[0]
    ), f"Cache length {len(cache)} != direct activations length {direct_activations.shape[0]}"

    # Test that tokens match
    assert th.equal(
        cache.tokens, direct_tokens
    ), "Cached tokens don't match direct tokens"

    # Test that activations match (within tolerance for numerical precision)
    cached_activations = th.stack([cache[i] for i in range(len(cache))], dim=0)
    assert th.allclose(
        cached_activations, direct_activations, atol=1e-5, rtol=1e-5
    ), "Cached activations don't match direct activations"

    # Test mean and std computation
    computed_mean = direct_activations.mean(dim=0)
    computed_std = direct_activations.std(dim=0, unbiased=True)

    assert th.allclose(
        cache.mean, computed_mean, atol=1e-5, rtol=1e-5
    ), "Cached mean doesn't match computed mean"
    assert th.allclose(
        cache.std, computed_std, atol=1e-5, rtol=1e-5
    ), "Cached std doesn't match computed std"

    print(f"✓ Successfully tested ActivationCache with {len(cache)} activations")
    print(f"✓ Mean shape: {cache.mean.shape}, Std shape: {cache.std.shape}")


def test_sequence_ranges_no_bos_token(temp_dir):
    """Test that sequence ranges are stored when model has no BOS token."""
    # Set flag to handle meta tensors properly
    if hasattr(th.fx, "experimental"):
        th.fx.experimental._config.meta_nonzero_assume_all_nonzero = True

    # Skip test if CUDA not available
    if not th.cuda.is_available():
        pytest.skip("CUDA not available, skipping test")

    # Test strings of different lengths
    test_strings = [
        "Hello world",
        "This is a longer sentence with more tokens",
        "Short",
        "Medium length text here",
    ]

    # Load GPT-2 model and modify tokenizer to simulate no BOS token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", device_map="auto", torch_dtype=th.float32
    )
    model = LanguageModel(model, torch_dtype=th.float32, tokenizer=tokenizer)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Simulate model without BOS token
    original_bos_token_id = model.tokenizer.bos_token_id
    model.tokenizer.bos_token_id = None

    tokens = model.tokenizer(
        test_strings,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    lengths = tokens["attention_mask"].sum(dim=1).tolist()
    ranges = np.cumsum([0] + lengths)
    try:
        # Get a transformer block
        target_layer = model.transformer.h[6]
        submodule_name = "transformer_h_6"

        # Parameters for activation collection
        batch_size = 2
        context_len = 32
        d_model = 768

        # Collect activations with sequence start tracking
        ActivationCache.collect(
            data=test_strings,
            submodules=(target_layer,),
            submodule_names=(submodule_name,),
            model=model,
            store_dir=temp_dir,
            batch_size=batch_size,
            context_len=context_len,
            shard_size=1000,
            d_model=d_model,
            io="out",
            store_tokens=True,
            shuffle_shards=False,  # Required for sequence ranges
        )

        # Load the cached activations
        cache = ActivationCache(temp_dir, submodule_name + "_out")

        # Verify sequence ranges were stored
        sequence_ranges = cache.sequence_ranges
        assert (
            sequence_ranges is not None
        ), "sequence ranges should be stored for model without BOS token"

        # Should have one sequence start per input string plus one for the last sequence
        assert (
            len(sequence_ranges) == len(test_strings) + 1
        ), f"Expected {len(test_strings)} sequence ranges, got {len(sequence_ranges)}"

        # First sequence should start at position 0
        assert (
            sequence_ranges[0].item() == 0
        ), "First sequence should start at position 0"

        # sequence ranges should be the same as the ranges computed from the tokens
        assert np.allclose(
            sequence_ranges, ranges
        ), "sequence ranges should be the same as the ranges computed from the tokens"

        # sequence ranges should be in ascending order
        for i in range(1, len(sequence_ranges)):
            assert (
                sequence_ranges[i] > sequence_ranges[i - 1]
            ), f"sequence ranges should be ascending: {sequence_ranges}"

        # Verify sequence ranges align with token boundaries
        tokens = cache.tokens
        total_tokens = len(tokens)

        # All sequence ranges should be valid indices
        for start_idx in sequence_ranges:
            assert (
                0 <= start_idx <= total_tokens
            ), f"Invalid sequence start index: {start_idx}"

    finally:
        # Restore original BOS token
        model.tokenizer.bos_token_id = original_bos_token_id


def test_sequence_ranges_with_bos_token(temp_dir):
    """Test that sequence ranges are NOT stored when model has BOS token."""
    # Set flag to handle meta tensors properly
    if hasattr(th.fx, "experimental"):
        th.fx.experimental._config.meta_nonzero_assume_all_nonzero = True

    # Skip test if CUDA not available
    if not th.cuda.is_available():
        pytest.skip("CUDA not available, skipping test")

    test_strings = ["Hello world", "Another test sentence"]

    # Load GPT-2 model with BOS token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", device_map="auto", torch_dtype=th.float32
    )
    model = LanguageModel(model, torch_dtype=th.float32, tokenizer=tokenizer)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Ensure model has BOS token (set it explicitly)
    model.tokenizer.bos_token_id = model.tokenizer.eos_token_id

    # Get a transformer block
    target_layer = model.transformer.h[6]
    submodule_name = "transformer_h_6"

    # Collect activations
    ActivationCache.collect(
        data=test_strings,
        submodules=(target_layer,),
        submodule_names=(submodule_name,),
        model=model,
        store_dir=temp_dir,
        batch_size=2,
        context_len=32,
        shard_size=1000,
        d_model=768,
        io="out",
        store_tokens=True,
        shuffle_shards=False,
    )

    # Load the cached activations
    cache = ActivationCache(temp_dir, submodule_name + "_out")

    # Verify sequence ranges were NOT stored
    sequence_ranges = cache.sequence_ranges
    assert (
        sequence_ranges is None
    ), "sequence ranges should not be stored for model with BOS token"


def test_activation_cache_slice_indexing_cross_shard(temp_dir):
    """Test ActivationCache slice indexing that crosses shard boundaries."""
    # Set flag to handle meta tensors properly
    th.fx.experimental._config.meta_nonzero_assume_all_nonzero = True

    # Skip test if CUDA not available to avoid device mapping issues
    if not th.cuda.is_available():
        pytest.skip("CUDA not available, skipping test to avoid device mapping issues")

    # Create test strings with sufficient data to span multiple shards
    test_strings = [
        f"This is test sentence number {i} with some content to fill up the cache."
        for i in range(20)  # Create more samples to ensure multiple shards
    ]

    # Use the list directly
    dataset = test_strings

    # Load GPT-2 model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", device_map="auto", torch_dtype=th.float32
    )
    model = LanguageModel(model, torch_dtype=th.float32, tokenizer=tokenizer)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Get a transformer block to extract activations from
    target_layer = model.transformer.h[6]  # Middle layer of GPT-2
    submodule_name = "transformer_h_6"

    # Parameters for activation collection - use small shard size to ensure multiple shards
    batch_size = 3
    context_len = 32
    d_model = 768  # GPT-2 hidden size
    shard_size = 50  # Small shard size to force multiple shards

    # Collect activations using ActivationCache
    ActivationCache.collect(
        data=dataset,
        submodules=(target_layer,),
        submodule_names=(submodule_name,),
        model=model,
        store_dir=temp_dir,
        batch_size=batch_size,
        context_len=context_len,
        shard_size=shard_size,  # Small shard size for testing cross-shard slicing
        d_model=d_model,
        io="out",
        max_total_tokens=5000,
        store_tokens=True,
        shuffle_shards=False,  # Important: don't shuffle so we can predict shard boundaries
    )

    # Load the cached activations
    cache = ActivationCache(temp_dir, submodule_name + "_out")

    # Verify we have multiple shards
    assert (
        len(cache.shards) >= 2
    ), f"Expected at least 2 shards, got {len(cache.shards)}"

    total_size = len(cache)
    print(f"Cache has {len(cache.shards)} shards with total size {total_size}")

    # Print shard boundaries for debugging
    shard_boundaries = cache._range_to_shard_idx
    print(f"Shard boundaries: {shard_boundaries}")

    # Test 1: Slice that crosses exactly one shard boundary
    if len(cache.shards) >= 2:
        # Find a slice that starts in first shard and ends in second shard
        first_shard_end = shard_boundaries[1]
        start_idx = max(0, first_shard_end - 10)
        end_idx = min(total_size, first_shard_end + 10)

        # Get slice result
        slice_result = cache[start_idx:end_idx]

        # Get individual results for comparison
        individual_results = th.stack(
            [cache[i] for i in range(start_idx, end_idx)], dim=0
        )

        # Verify they match
        assert th.allclose(
            slice_result, individual_results, atol=1e-5, rtol=1e-5
        ), f"Slice result doesn't match individual indexing for indices {start_idx}:{end_idx}"

        # Verify correct shape
        expected_length = end_idx - start_idx
        assert (
            slice_result.shape[0] == expected_length
        ), f"Expected slice length {expected_length}, got {slice_result.shape[0]}"

        print(f"✓ Cross-shard slice test 1 passed: indices {start_idx}:{end_idx}")

    # Test 2: Slice that spans multiple shards
    if len(cache.shards) >= 3:
        # Find a slice that starts in first shard and ends in third shard
        second_shard_end = shard_boundaries[2]
        start_idx = max(0, shard_boundaries[1] - 5)  # Start near end of first shard
        end_idx = min(total_size, second_shard_end + 5)  # End in third shard

        slice_result = cache[start_idx:end_idx]
        individual_results = th.stack(
            [cache[i] for i in range(start_idx, end_idx)], dim=0
        )

        assert th.allclose(
            slice_result, individual_results, atol=1e-5, rtol=1e-5
        ), f"Multi-shard slice result doesn't match individual indexing for indices {start_idx}:{end_idx}"

        expected_length = end_idx - start_idx
        assert (
            slice_result.shape[0] == expected_length
        ), f"Expected multi-shard slice length {expected_length}, got {slice_result.shape[0]}"

        print(f"✓ Multi-shard slice test passed: indices {start_idx}:{end_idx}")

    # Test 3: Slice with step parameter across shards
    if total_size >= 50:
        start_idx = 5
        end_idx = min(total_size, 45)
        step = 3

        slice_result = cache[start_idx:end_idx:step]
        individual_results = th.stack(
            [cache[i] for i in range(start_idx, end_idx, step)], dim=0
        )

        assert th.allclose(
            slice_result, individual_results, atol=1e-5, rtol=1e-5
        ), f"Stepped slice result doesn't match individual indexing for indices {start_idx}:{end_idx}:{step}"

        expected_length = len(range(start_idx, end_idx, step))
        assert (
            slice_result.shape[0] == expected_length
        ), f"Expected stepped slice length {expected_length}, got {slice_result.shape[0]}"

        print(f"✓ Stepped slice test passed: indices {start_idx}:{end_idx}:{step}")

    # Test 4: Edge cases - slice at boundaries
    if len(cache.shards) >= 2:
        # Test slice starting exactly at shard boundary
        boundary_idx = shard_boundaries[1]
        if boundary_idx < total_size - 5:
            slice_result = cache[boundary_idx : boundary_idx + 5]
            individual_results = th.stack(
                [cache[i] for i in range(boundary_idx, boundary_idx + 5)], dim=0
            )

            assert th.allclose(
                slice_result, individual_results, atol=1e-5, rtol=1e-5
            ), f"Boundary slice result doesn't match individual indexing"

            print(
                f"✓ Boundary slice test passed: starting at shard boundary {boundary_idx}"
            )

    # Test 5: Empty slice
    empty_slice = cache[10:10]
    assert (
        empty_slice.shape[0] == 0
    ), f"Expected empty slice, got shape {empty_slice.shape}"
    print("✓ Empty slice test passed")

    print(
        f"✓ All slice indexing tests passed for cache with {len(cache.shards)} shards"
    )
