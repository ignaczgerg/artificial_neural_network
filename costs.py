import numpy as np
from abc import ABC, abstractmethod


class AbstractCost(ABC):
  @abstractmethod
  def loss(self, predict: np.array, target: np.array) -> float:
    raise NotImplementedError
  
  @abstractmethod
  def gradient(self, predict: np.array, target: np.array) -> np.array:
    raise NotImplementedError

class BinaryCrossEntropyCost(AbstractCost):
  def __init__(self, limit=1e-15):
    self.limit = limit

  def loss(self, predict: np.array, target: np.array) -> np.array:
    clipped_predict = np.clip(predict, self.limit, 1 - self.limit)
    return -1 * np.sum(target * np.log(clipped_predict) + (1 - target) * np.log(1 - clipped_predict)) / len(target)
  
  def gradient(self, predict: np.array, target: np.array) -> np.array:
    clipped_predict = np.clip(predict, self.limit, 1 - self.limit)
    return -1 * ((target / clipped_predict) - ((1 - target) / (1 - clipped_predict)))
