import math
import random
import torch
import pytest

from dictionary_learning.cache import RunningStatWelford


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("D", [5, 127])  # feature dimensionalities
def test_streaming_matches_reference(dtype, D):
    """
    Stream random data through RunningStatWelford in random-sized batches
    and check that mean / std match the ground-truth computed in one shot.
    """
    torch.manual_seed(0)
    N_total = 50_000

    # Create full data for reference computation
    full = torch.randn(N_total, D, dtype=dtype)

    # Stream through accumulator in random batches
    acc = RunningStatWelford(shape=(D,))
    idx = 0
    while idx < N_total:
        batch_size = random.randint(1, 2048)
        x = full[idx : idx + batch_size]
        acc.update(x)
        idx += batch_size

    # Ground-truth (double precision to remove numeric noise)
    ref_mean = full.double().mean(dim=0)
    ref_std = full.double().std(dim=0, unbiased=True)

    # Compare
    torch.testing.assert_close(acc.mean, ref_mean, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(acc.std(), ref_std, rtol=1e-6, atol=1e-7)
    assert acc.n == N_total


def test_merge_two_accumulators():
    """
    Splits the dataset in two, accumulates statistics separately, then
    merges and checks against reference.
    """
    torch.manual_seed(123)
    N_total, D = 32_768, 64
    data = torch.randn(N_total, D)

    split = N_total // 2
    part1, part2 = data[:split], data[split:]

    acc1 = RunningStatWelford(shape=(D,))
    acc2 = RunningStatWelford(shape=(D,))
    acc1.update(part1)
    acc2.update(part2)

    # Merge acc2 into acc1
    acc1.merge(acc2)

    # Reference
    ref_mean = data.double().mean(dim=0)
    ref_std = data.double().std(dim=0, unbiased=True)

    torch.testing.assert_close(acc1.mean, ref_mean, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(acc1.std(), ref_std, rtol=1e-6, atol=1e-7)
    assert acc1.n == N_total


def test_edge_cases():
    """
    Edge-case behaviour: empty updates, very small sample counts, etc.
    """
    dtype = torch.float32
    acc = RunningStatWelford(shape=(3,), dtype=dtype)
    # Empty update should be a no-op
    acc.update(torch.empty(0, 3))
    assert acc.n == 0
    assert math.isnan(acc.var()[0])

    # Single example → variance undefined (n-1 == 0)
    acc.update(torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype))
    assert acc.n == 1
    assert math.isnan(acc.var()[0])

    # Two examples → variance defined
    acc.update(torch.tensor([[2.0, 4.0, 6.0]], dtype=dtype))
    assert acc.n == 2
    torch.testing.assert_close(acc.mean, torch.tensor([1.5, 3.0, 4.5], dtype=dtype))
    torch.testing.assert_close(
        acc.std(), torch.tensor([0.70710678, 1.41421356, 2.12132034], dtype=dtype)
    )
