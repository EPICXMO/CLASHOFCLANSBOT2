from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class EpsilonGreedy:
    k: int
    epsilon: float = 0.3
    counts: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * self.k
        if not self.values:
            self.values = [0.0] * self.k

    def select(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.k)
        return max(range(self.k), key=lambda i: self.values[i])

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # incremental mean update
        self.values[arm] = value + (reward - value) / n

    def ensure_arms(self, k: int) -> None:
        if k <= self.k:
            return
        add = k - self.k
        self.counts.extend([0] * add)
        self.values.extend([0.0] * add)
        self.k = k

    def decay(self, episode: int, total: int = 100, min_epsilon: float = 0.05) -> None:
        if total <= 0:
            return
        frac = max(0.0, min(1.0, episode / float(total)))
        start = 0.3
        end = min_epsilon
        self.epsilon = max(end, start - (start - end) * frac)
