"""
Defines TabularPolicy (TabularPolicy) -- necessary to avoid circular imports.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, obs, deterministic=True):
        raise NotImplementedError()


class TabularPolicy(BasePolicy):
    def __init__(self, policy: np.ndarray):
        self.matrix = np.copy(policy)

    def get_action(self, state, deterministic=True):
        if deterministic:
            return np.argmax(self.matrix[state, :])
        else:
            return np.random.choice(range(self.matrix.shape[1]), p=self.matrix[state, :])

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)
