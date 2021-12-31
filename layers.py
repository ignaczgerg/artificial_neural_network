from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class AbstractLayer(ABC):
  def __init__(self):
    self.input_shape = None
    self.output_shape = None
    self.weights = None
    self.weights_gradient = None
    self.bias = None
    self.bias_gradient = None
    self.prev_input = None
  
  @abstractproperty
  def has_weights(self) -> bool:
    pass
  
  @property
  def weights_shape(self) -> tuple:
    return ()
  
  @property
  def weights_size(self) -> int:
    return np.prod(self.weights_shape) if self.weights_shape else 0
  
  def update_weights(self, weights: np.array):
    self.weights = weights
  
  @abstractproperty
  def has_bias(self) -> bool:
    pass

  @property
  def bias_size(self) -> int:
    return 0
  
  def update_bias(self, bias: np.array):
    self.bias = bias
  
  @property
  def input_size(self) -> int:
    return np.prod(self.input_shape)
  
  @property
  def output_size(self) -> int:
    return np.prod(self.output_shape)

  def initialize(self):
    pass

  @abstractmethod
  def forward(self, x: np.array) -> np.array:
    raise NotImplementedError

  @abstractmethod
  def backward(self, dout: np.array) -> np.array:
    raise NotImplementedError


class InputLayer(AbstractLayer):
  def __init__(self, shape: tuple):
    super().__init__()
    self.input_shape = shape
    self.output_shape = shape
  
  @property
  def has_weights(self):
    return False

  @property
  def has_bias(self):
    return False

  def forward(self, x: np.array) -> np.array:
    return x
  
  def backward(self, dout: np.array) -> np.array:
    return dout


class DenseLayer(AbstractLayer):
  def __init__(self, units: int):
    super().__init__()
    self.units = units
    self.output_shape = (units, )
  
  @property
  def has_weights(self):
    return True

  @property
  def has_bias(self):
    return True
  
  @property
  def weights_shape(self) -> tuple:
    return (self.input_size, self.output_size)
  
  @property
  def bias_size(self) -> int:
    return self.units

  def initialize(self):
    if self.weights is None:
      self.weights = np.random.randn(*self.weights_shape) * 0.1
    if self.bias is None:
      self.bias = np.zeros(self.bias_size)

  def forward(self, x: np.array) -> np.array:
    self.prev_input = x.copy()
    return (x @ self.weights) + self.bias

  def backward(self, dout: np.array) -> np.array:
    self.weights_gradient = self.prev_input.T @ dout
    self.bias_gradient = dout.sum(axis=0)
    return dout @ self.weights.T


class ActivationLayer(AbstractLayer):
  def __init__(self, function):
    super().__init__()
    self.function = function
  
  @property
  def has_weights(self):
    return False

  @property
  def has_bias(self):
    return False

  @property
  def output_shape(self) -> tuple:
    return self.input_shape
  
  @output_shape.setter
  def output_shape(self, value):
    pass
  
  def forward(self, x: np.array) -> np.array:
    self.prev_input = x.copy()
    return self.function(x)
  
  def backward(self, dout: np.array) -> np.array:
    return dout * self.function(self.prev_input, derivative=True)
