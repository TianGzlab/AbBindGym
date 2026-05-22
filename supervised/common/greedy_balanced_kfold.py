#!/usr/bin/env python3
"""
Greedy Balanced K-Fold Splitter

This module provides cluster-aware k-fold splitting that:
1. Keeps all samples from the same cluster together (no leakage)
2. Balances sample counts across folds using greedy assignment
3. Creates train/valid/test splits for each fold

Used by multiple prepare scripts: SKEMPI, SabDab, AbELA, PPB, AbCoV, AbDesign
"""

from typing import List, Dict, Tuple, Set
import random
import numpy as np
import pandas as pd


def split_folds_sample_balanced(
    clusters: List[str],
    cluster_sizes: Dict[str, int],
    k: int,
    seed: int,
) -> List[List[str]]:
    """
    Assign clusters to k folds to balance total sample counts per fold (no cluster split).

    Greedy algorithm:
    1. Sort clusters by size (largest first)
    2. Assign each cluster to the fold with smallest current total

    Args:
        clusters: List of cluster IDs
        cluster_sizes: Dict mapping cluster ID to number of samples
        k: Number of folds
        seed: Random seed for tie-breaking

    Returns:
        List of k lists, each containing cluster IDs for that fold
    """
    rng = random.Random(seed)
    cs = list(clusters)
    rng.shuffle(cs)  # Random shuffle for tie-breaking
    cs.sort(key=lambda c: cluster_sizes.get(c, 0), reverse=True)

    folds: List[List[str]] = [[] for _ in range(k)]
    fold_counts = [0] * k

    for c in cs:
        # Assign to fold with smallest current count
        j = int(np.argmin(fold_counts))
        folds[j].append(c)
        fold_counts[j] += cluster_sizes.get(c, 0)

    return folds


def choose_valid_no_big_overshoot(
    train_pool: List[str],
    cluster_sizes: Dict[str, int],
    target_valid_samples: int,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    """
    Pick validation clusters from train_pool to be close to target_valid_samples.

    Strategy:
    - First pass: only add clusters if they don't exceed target
    - If still short, add one extra cluster that minimizes absolute error
    - Ensures we don't create empty sets

    Args:
        train_pool: Available cluster IDs
        cluster_sizes: Dict mapping cluster ID to number of samples
        target_valid_samples: Target number of validation samples
        seed: Random seed

    Returns:
        (train_clusters, valid_clusters) tuple of sets
    """
    rng = random.Random(seed)
    cs = list(train_pool)
    rng.shuffle(cs)
    cs.sort(key=lambda c: cluster_sizes.get(c, 0), reverse=True)

    valid = set()
    valid_count = 0
    remaining = []

    # First pass: add without exceeding target
    for c in cs:
        s = cluster_sizes.get(c, 0)
        if valid_count + s <= target_valid_samples:
            valid.add(c)
            valid_count += s
        else:
            remaining.append(c)

    # If still short, add one cluster that minimizes error
    if valid_count < target_valid_samples and remaining:
        best = min(
            remaining,
            key=lambda c: abs((valid_count + cluster_sizes.get(c, 0)) - target_valid_samples)
        )
        valid.add(best)

    # Safety: avoid empty or all-in-one sets
    if len(valid) == 0 and len(cs) > 0:
        valid.add(cs[0])
    if len(valid) == len(cs) and len(cs) > 1:
        valid.remove(cs[-1])

    train = set(cs) - valid
    return train, valid


def greedy_balanced_kfold(
    df: pd.DataFrame,
    cluster_col: str,
    n_folds: int,
    valid_frac: float = 0.10,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    """
    Generate k-fold splits with cluster-aware sample balancing.

    For each fold:
    - Test set: clusters assigned to this fold
    - Train+Valid pool: all other clusters
    - Valid set: selected from train pool to match valid_frac
    - Train set: remaining clusters from train pool

    Args:
        df: DataFrame with data
        cluster_col: Column name containing cluster IDs
        n_folds: Number of folds (k)
        valid_frac: Fraction of train pool to use for validation (default 0.10)
        seed: Random seed

    Returns:
        List of n_folds dictionaries, each with:
        {
            'train_idx': List[int],  # Row indices for training
            'valid_idx': List[int],  # Row indices for validation
            'test_idx': List[int],   # Row indices for testing
        }
    """
    # Treat missing cluster labels as one explicit group instead of silently
    # dropping those rows from every fold.
    cluster_values = df[cluster_col].fillna("__missing_cluster__").astype(str)
    cluster_sizes = cluster_values.value_counts().to_dict()
    clusters = cluster_values.unique().tolist()

    # Create sample-balanced test folds
    fold_clusters = split_folds_sample_balanced(
        clusters=clusters,
        cluster_sizes=cluster_sizes,
        k=n_folds,
        seed=seed
    )

    # Generate splits for each fold
    folds = []
    for fold_idx in range(n_folds):
        test_clusters = set(fold_clusters[fold_idx])

        # Collect train pool from all other folds
        train_pool = []
        for i in range(n_folds):
            if i != fold_idx:
                train_pool.extend(fold_clusters[i])

        # Split train pool into train and valid
        train_pool_samples = sum(cluster_sizes.get(c, 0) for c in train_pool)
        target_valid_samples = int(round(train_pool_samples * valid_frac))

        train_clusters, valid_clusters = choose_valid_no_big_overshoot(
            train_pool=train_pool,
            cluster_sizes=cluster_sizes,
            target_valid_samples=target_valid_samples,
            seed=seed + 1000 + fold_idx,
        )

        # Convert clusters to row indices
        train_idx = df[cluster_values.isin(train_clusters)].index.tolist()
        valid_idx = df[cluster_values.isin(valid_clusters)].index.tolist()
        test_idx = df[cluster_values.isin(test_clusters)].index.tolist()

        folds.append({
            'train_idx': train_idx,
            'valid_idx': valid_idx,
            'test_idx': test_idx,
        })

    return folds
