from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NumericTable:
    columns: tuple[str, ...]
    values: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.columns):
            raise ValueError('The matrix and column names must have matching widths')
        if len(set(self.columns)) != len(self.columns):
            raise ValueError('Column names must be unique')
        if not torch.isfinite(self.values).all():
            raise ValueError('Numeric values must be finite')


def read_numeric_csv(path: str | Path, columns: Sequence[str]) -> NumericTable:
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Select at least one unique column')
    rows = []
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(col not in reader.fieldnames for col in columns):
            raise ValueError('A requested column is absent')
        for row in reader:
            rows.append([float(row[col]) for col in columns])
    if not rows:
        raise ValueError('The table is empty')
    return NumericTable(tuple(columns), torch.tensor(rows, dtype=torch.float64))


def clip_and_average(gradients: Sequence[Tensor], bound: float) -> tuple[Tensor, ...]:
    if not gradients or not math.isfinite(bound) or bound <= 0:
        raise ValueError('A positive finite clipping bound and gradients are required')
    batch = gradients[0].shape[0]
    if batch == 0 or any(g.ndim == 0 or g.shape[0] != batch for g in gradients):
        raise ValueError('Gradient tensors must share a nonempty batch dimension')
    if any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError('Gradients must be finite')
    norms = torch.sqrt(sum(g.reshape(batch, -1).square().sum(dim=1) for g in gradients))
    factors = torch.clamp(bound / norms.clamp_min(torch.finfo(norms.dtype).tiny), max=1)
    return tuple((g * factors.reshape((batch,) + (1,) * (g.ndim - 1))).mean(dim=0) for g in gradients)


def perturb_average(gradients: Sequence[Tensor], noise_std: float, generator: torch.Generator) -> tuple[Tensor, ...]:
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('Noise standard deviation must be finite and nonnegative')
    return tuple(g + torch.randn(g.shape, dtype=g.dtype, device=g.device, generator=generator) * noise_std for g in gradients)


def apply_gradients(model: nn.Module, gradients: Sequence[Tensor], optimizer: torch.optim.Optimizer) -> None:
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if len(parameters) != len(gradients) or any(p.shape != g.shape for p, g in zip(parameters, gradients)):
        raise ValueError('Gradient shapes do not match the model parameters')
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.detach().clone()
    optimizer.step()


def top_fraction(values: Tensor, fraction: float) -> Tensor:
    if values.numel() == 0 or not 0 < fraction <= 1 or not torch.isfinite(values).all():
        raise ValueError('A finite nonempty tensor and fraction in (0, 1] are required')
    count = math.ceil(fraction * values.numel())
    indices = torch.argsort(values.abs().reshape(-1), descending=True, stable=True)[:count]
    result = torch.zeros_like(values).reshape(-1)
    source = values.reshape(-1)
    result[indices] = source[indices]
    return result.reshape(values.shape)


def auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels))
    s = np.asarray(list(scores), dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not np.isfinite(s).all():
        raise ValueError('Labels and scores must be finite vectors of equal length')
    if not np.isin(y, [0, 1]).all() or not (np.any(y == 0) and np.any(y == 1)):
        raise ValueError('Both binary classes are required')
    order = np.argsort(s, kind='stable')
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positives = y == 1
    npos = int(positives.sum())
    nneg = len(y) - npos
    return float((ranks[positives].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ks_distance(first: Iterable[float], second: Iterable[float]) -> float:
    a = np.sort(np.asarray(list(first), dtype=float))
    b = np.sort(np.asarray(list(second), dtype=float))
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Both samples must be nonempty finite vectors')
    values = np.concatenate((a, b))
    return float(np.max(np.abs(np.searchsorted(a, values, side='right') / len(a) - np.searchsorted(b, values, side='right') / len(b))))


def net_benefit(labels: Iterable[int], scores: Iterable[float], threshold: float) -> float:
    y = np.asarray(list(labels))
    s = np.asarray(list(scores), dtype=float)
    if not 0 < threshold < 1 or y.ndim != 1 or not len(y) or y.shape != s.shape:
        raise ValueError('Valid vectors and a threshold strictly between zero and one are required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Labels must be binary and scores must be probabilities')
    predicted = s >= threshold
    tp = np.count_nonzero(predicted & (y == 1))
    fp = np.count_nonzero(predicted & (y == 0))
    return float((tp - fp * threshold / (1 - threshold)) / len(y))


def save_state(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, seed: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix='.state-', suffix='.tmp')
    os.close(descriptor)
    try:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'seed': seed, 'rng': torch.get_rng_state()}, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(Path(path), map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    return int(state['seed'])
