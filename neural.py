from typing import List, Tuple
import numpy as np
from time import time
from datetime import timedelta, datetime
import pickle
import gzip

from layers import *
from optimizers import AbstractOptimizer
from costs import AbstractCost
from metrics import AbstractMetric
from utils import generate_batches

class Neural:
  def __init__(self, optimizer: AbstractOptimizer, cost: AbstractCost, metric: AbstractMetric):
    self.layers : List[AbstractLayer] = []
    self.config_file = None

    self.optimizer = optimizer
    self.cost = cost
    self.metric = metric

    self.train_loss_per_iter = []
    self.train_accu_per_iter = []
    self.valid_loss_per_epoch = []
    self.valid_accu_per_epoch = []

  def add_layer(self, layer: AbstractLayer):
    if len(self.layers) == 0:
      if not isinstance(layer, InputLayer):
        raise TypeError(f'first layer must be InputLayer(AbstractLayer), not \'{type(layer).__name__}\'')
    else:
      layer.input_shape = self.layers[-1].output_shape

    if not isinstance(layer, AbstractLayer):
      raise TypeError(f'layer must be Layer(AbstractLayer), not \'{type(layer).__name__}\'')

    self.layers.append(layer)
  
  def compile(self):
    for layer in self.layers:
      layer.initialize()

  def train(
    self, 
    train_data: Tuple[np.array, np.array], 
    epochs: int, 
    batch_size: int, 
    validation_data: Tuple[np.array, np.array]=None,
    test_data: Tuple[np.array, np.array]=None):
    '''
    Train is slow
    '''
    xtrain, ytrain = train_data
    try:
      print('Training')
      for epoch in range(epochs):
        now = datetime.now()
        print(f'\nEpoch: {epoch + 1}/{epochs}\t{now}')
        t0 = time()
        for i, (x, y) in enumerate(generate_batches(train_data, batch_size)):

          predict = self.feedforward(x, training=True)

          train_loss = self.cost.loss(predict, y)
          train_accuracy = self.metric.compare(predict, y)

          output_gradient = self.cost.gradient(predict, y)
          self.backpropagation(output_gradient)
          self.update_weights()
          
          self.train_loss_per_iter.append(train_loss)
          self.train_accu_per_iter.append(train_accuracy)

        print(f'Train:\tAccu: {train_accuracy:.3f}\tLoss: {train_loss:.3f}')

        if validation_data:
          xvalid, yvalid = validation_data
          valid_predict = self.predict(xvalid)
          valid_loss = self.cost.loss(valid_predict, yvalid)
          valid_accuracy = self.metric.compare(valid_predict, yvalid)
          
          self.valid_loss_per_epoch.append(valid_loss)
          self.valid_accu_per_epoch.append(valid_accuracy)

          print(f'Valid:\tAccu: {valid_accuracy:.3f}\tLoss: {valid_loss:.3f}')
        
        t1 = time()
        time_passed = timedelta(seconds=t1 - t0)
        print(f'Time: {time_passed}')
        
      if test_data:
        xtest, ytest = test_data
        test_predict = self.predict(xtest)
        test_loss = self.cost.loss(test_predict, ytest)
        test_accuracy = self.metric.compare(test_predict, ytest)
        print(f'\nTest:\tAccu: {test_accuracy:.3f}\tLoss: {test_loss:.3f}')
    except KeyboardInterrupt:
      print('\nStoped!\n')
    else:
      print('\nDone\n')

  def feedforward(self, input_data: np.array, training: bool) -> np.array:
    activation = input_data
    for layer in self.layers:
      if not training and isinstance(layer, InputLayer):
        continue
      activation = layer.forward(activation)
    return activation
  
  def backpropagation(self, output_gradient: np.array) -> np.array:
    gradient = output_gradient
    for layer in reversed(self.layers):
      gradient = layer.backward(gradient)
    return gradient
  
  def update_weights(self):
    self.optimizer.optimize(self.layers)

  def predict(self, input_data: np.array) -> np.array:
    return self.feedforward(input_data, training=False)

  def test(self, test_data: Tuple[np.array, np.array]) -> np.array:
    input_data, target = test_data
    predict = self.predict(input_data)
    accuracy = self.metric.compare(predict, target)
    loss = self.cost.loss(predict, target)
    return accuracy, loss
  
  def save_config(self):
    configs = []
    for i, layer in enumerate(self.layers):
      configs.append({
        'type': type(layer).__name__,
        'weights_size': layer.weights_size,
        'weights': layer.weights,
        'bias_size': layer.bias_size,
        'bias': layer.bias
      })
    with gzip.open(self.config_file, 'wb') as file:
      pickle.dump(configs, file)
  
  def load_config(self, config_file):
    try:
      with gzip.open(config_file, 'rb') as file:
        configs = pickle.load(file)
    except FileNotFoundError:
      pass
    else:
      if len(configs) != len(self.layers):
        raise Exception('Invalid config file: number of layers does not match')

      for i, layer in enumerate(self.layers):
        config = configs[i]

        if config['type'] != type(layer).__name__:
          raise Exception(f'Invalid config file: {i}° layer type does not match')
        
        if layer.weights_size != config['weights_size'] or layer.bias_size != config['bias_size']:
          raise Exception(f'Invalid config file: weights or bias shapes does not match')
        
        layer.weights = config['weights']
        layer.bias = config['bias']
    finally:
      self.config_file = config_file
