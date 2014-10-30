import numpy as np
import numpy, pdb
import os,sys
import matplotlib.pyplot as plt

import convert_data_to_numpy as convert

class online_learner(object):
  def __init__(self):
    pass

  def predict(self, x):
    pass

  def fit(self, x, y, t):
    pass

  def compute_single_loss(self, y_gt, y_p):
    pass

  def evaluate(self, X, Y):
    n_samples = X.shape[0]

    Y_p = numpy.zeros_like(Y)
    losses = numpy.zeros_like(Y)
    total_loss = 0

    for (xi, x) in enumerate(X):
      if xi % 1000 == 0:
        print "round ", xi, total_loss

      y_p = self.predict(x)
      Y_p[xi] = y_p

      loss = self.compute_single_loss(Y[xi], y_p)
      total_loss += loss
      losses[xi] = loss
      
      self.fit(x, Y[xi])

    return Y_p, losses

class online_logistic(online_learner):
    def __init__(self, l2_lam):
        pass

    def predict(x):
        pass

    def fit(x, y):
        pass


class online_svm(online_learner):
  def __init__(self, lam, feature_size, margin = 1):
    self.lam = float(lam)
    self.w = numpy.zeros((feature_size,), dtype = numpy.float64)

    self.t = 1
    self.batch_size = 20
    self.margin = margin
    self.sqrt_lam = numpy.sqrt(self.lam)
    
    self.feature_size = feature_size
    self.cur_grad = numpy.zeros((self.feature_size,))
    self.batch_grads = numpy.zeros((feature_size, self.batch_size))
    
  def predict(self, x):
    y_p = numpy.dot(self.w.T, x)
    return y_p
    
  def compute_single_loss(self, y_gt, y_p):
    prod = y_gt * y_p
    loss = max(0, self.margin - prod) + self.lam  * numpy.dot(self.w.T, self.w) / 2.0
    return loss
    
  def fit(self, x, y_gt, do_batch = True):
    alpha = 1.0 / (self.lam * self.t)
    y_p = self.predict(x)
    if y_gt * y_p < self.margin:
      grad = - alpha * self.lam * self.w + alpha * y_gt * x
    else:
      grad = - alpha * self.lam * self.w

    if not do_batch:
      self.w += grad
      
      wtw = numpy.dot(self.w.T, self.w)
      if wtw > 1.0 / self.lam:
        self.w = self.w / (numpy.sqrt(wtw) * (self.sqrt_lam))    
    else:
      self.cur_grad += grad
      if self.t % self.batch_size == 0:
          self.w += self.cur_grad / float(self.batch_size)
          self.cur_grad = numpy.zeros((self.feature_size,))

    self.t += 1
    return self.w

  def evaluate(self, X, Y):
    inds = range(Y.shape[0])
    numpy.random.shuffle(inds)
    
    X = X[inds, :]
    Y = Y[inds, :]

    X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
    return super(online_svm, self).evaluate(X, Y)

def main():
  do_svm = True
  
  data_o = convert.dataset_oakland(numpy_fn = 'data/oakland_part3_am_rf.node_features.npz')

  lam = 1e-4
  osvm = online_svm(lam = lam, feature_size = data_o.features.shape[1])
  
  

  wall_and_ground_inds = numpy.where(numpy.logical_or(data_o.facade_inds, data_o.ground_inds))[0]

  Y = data_o.labels[wall_and_ground_inds]
  X = data_o.features[wall_and_ground_inds, :]

  Y[Y == data_o.label_map['facade']] = -1
  Y[Y == data_o.label_map['ground']] = 1

  ypred, losses = osvm.evaluate(X, Y)
  
  classification = ypred > 0
  n_right = (classification == (Y > 0)).sum()
  accuracy = n_right / float(Y.shape[0])

  cum_losses = numpy.cumsum(losses - osvm.lam * numpy.dot(osvm.w.T, osvm.w) / 2.)

  print "right, accuracy: {}, {}".format(n_right, accuracy)
  print "w: {}".format(osvm.w)
  plt.figure()
  plt.plot(range(losses.shape[0]), (cum_losses) / osvm.t)
  plt.show(block = False)

  pdb.set_trace()
if __name__ == '__main__':
  main()
