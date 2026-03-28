"""Balanced batch sampler for domain-adaptive training.

Ensures every mini-batch contains an equal number of source and target
samples, which is required by DANN-style adversarial domain alignment.
The sampler is designed to work with a ``ConcatDataset([source, target])``
where target indices are offset by the source dataset length.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class DomainBatchSampler(Sampler[list[int]]):
    """Yield balanced batches of source + target indices.

    Each batch contains ``batch_size // 2`` source indices followed by
    ``batch_size // 2`` target indices.  Both index pools are independently
    shuffled at the start of every epoch (every call to :meth:`__iter__`).

    Target indices are **offset** by ``source_dataset_size`` so they map
    to the correct positions in a ``ConcatDataset([source_ds, target_ds])``.

    Parameters
    ----------
    source_dataset_size : int
        Number of samples in the source dataset.
    target_dataset_size : int
        Number of samples in the target dataset.
    batch_size : int
        Total batch size.  Must be even (split equally between domains).
    drop_last : bool
        If ``True`` (default), the last incomplete batch is dropped so
        every batch has exactly ``batch_size`` elements.
    """

    def __init__(
        self,
        source_dataset_size: int,
        target_dataset_size: int,
        batch_size: int,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError(
                f"batch_size must be a positive even integer, got {batch_size}"
            )
        if source_dataset_size < 1:
            raise ValueError(f"source_dataset_size must be >= 1, got {source_dataset_size}")
        if target_dataset_size < 1:
            raise ValueError(f"target_dataset_size must be >= 1, got {target_dataset_size}")

        self.source_size = source_dataset_size
        self.target_size = target_dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.half_batch = batch_size // 2

        # The number of complete batches is limited by the smaller domain.
        # When one domain is exhausted, we restart its index pool (cyclic
        # sampling) so training is not bottlenecked by the smaller set.
        self._n_batches = self._compute_n_batches()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_n_batches(self) -> int:
        """Number of full batches per epoch.

        We base the epoch length on the *larger* domain so the model sees
        all available data.  The smaller domain is cycled as needed.
        """
        max_domain_samples = max(self.source_size, self.target_size)
        n = max_domain_samples / self.half_batch
        return math.floor(n) if self.drop_last else math.ceil(n)

    @staticmethod
    def _shuffled_indices(n: int) -> list[int]:
        """Return a randomly shuffled list of indices ``[0, n)``."""
        return torch.randperm(n).tolist()

    @staticmethod
    def _cyclic_extend(indices: list[int], required_length: int) -> list[int]:
        """Repeat *indices* (with re-shuffle each cycle) until we have enough."""
        result = list(indices)
        while len(result) < required_length:
            extra = torch.randperm(len(indices)).tolist()
            # Map back to original index values
            extra = [indices[i] for i in extra]
            result.extend(extra)
        return result[:required_length]

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one list of indices per batch."""
        needed = self._n_batches * self.half_batch

        # Shuffle source indices [0, source_size).
        src_indices = self._shuffled_indices(self.source_size)
        if needed > self.source_size:
            src_indices = self._cyclic_extend(src_indices, needed)
        else:
            src_indices = src_indices[:needed]

        # Shuffle target indices [0, target_size), then offset.
        tgt_indices_raw = self._shuffled_indices(self.target_size)
        if needed > self.target_size:
            tgt_indices_raw = self._cyclic_extend(tgt_indices_raw, needed)
        else:
            tgt_indices_raw = tgt_indices_raw[:needed]

        # Offset target indices for ConcatDataset layout.
        tgt_indices = [idx + self.source_size for idx in tgt_indices_raw]

        # Emit batches.
        for b in range(self._n_batches):
            start = b * self.half_batch
            end = start + self.half_batch
            batch = src_indices[start:end] + tgt_indices[start:end]
            yield batch

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self._n_batches

    def __repr__(self) -> str:
        return (
            f"DomainBatchSampler(source={self.source_size}, target={self.target_size}, "
            f"batch_size={self.batch_size}, n_batches={self._n_batches})"
        )
