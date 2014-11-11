#!/usr/bin/env python
import numpy as np
import numpy, pdb
import os,sys
import matplotlib.pyplot as plt

import convert_data_to_numpy as convert
import sklearn.metrics

import gtkutils.ml_util as mlu
import gtkutils.pdbwrap as pdbw

class online_learner(object):
  def __init__(self):
    pass

  def predict(self, x):
    return RuntimeError("not overridden")

  def fit(self, x, y, t):
    return RuntimeError("not overridden")

  def compute_single_loss(self, y_gt, y_p, x=None):
    return RuntimeError("not overridden")

  def evaluate(self, X, Y, fit = True):
    n_samples = X.shape[0]

    Y_p = numpy.zeros_like(Y)
    losses = numpy.zeros_like(Y)
    total_loss = 0

    for (xi, x) in enumerate(X):
      if xi % 1000 == 0:
        print "round ", xi, total_loss

      y_p = self.predict(x)
      Y_p[xi] = y_p

      loss = self.compute_single_loss(Y[xi], y_p, x)
      total_loss += loss
      losses[xi] = loss
      
      if fit:
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
    
  def compute_single_loss(self, y_gt, y_p, x=None):
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
    else:
      self.cur_grad += grad
      if self.t % self.batch_size == 0:
          self.w += self.cur_grad / float(self.batch_size)
          self.cur_grad = numpy.zeros((self.feature_size,))
    
    # Note from echo: renormalization was not done to batch version... 
    wtw = numpy.dot(self.w.T, self.w)
    if wtw > 1.0 / self.lam:
      self.w = self.w / (numpy.sqrt(wtw) * (self.sqrt_lam))    

    self.t += 1
    return self.w

  def evaluate(self, X, Y, **kwargs):
    inds = range(Y.shape[0])
    numpy.random.shuffle(inds)
    
    X = X[inds, :]
    Y = Y[inds, :]

    X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
    return super(online_svm, self).evaluate(X_norm, Y, **kwargs)

class online_exponentiated_sq_loss(online_learner):
    def __init__(self, nbr_classes, feature_size, lam=1e-3, grad_scale=1.0):
        self.t = 1.0
        self.lam=lam
        self.lam_inv = 1.0 / lam
        self.eta = grad_scale
        self.K = nbr_classes
        self.D = feature_size
        self.w = np.ones((nbr_classes, feature_size), dtype=np.float64) / self.D

    def compute_single_loss(self, y_gt, y_p, x):
        return 0
    
    def predict(self, x):
        return np.argmax(np.dot(self.w, x))

    def fit(self, x, y_gt, do_batch = True): 
        score = np.dot(self.w, x)
        y_gt_one_hot = self.one_hot(y_gt)
        grad = (self.w + np.outer(score - y_gt_one_hot, x) * self.lam_inv) / self.t
        self.w *= np.exp(-grad * self.eta) 
        
        w_sum = np.sum(self.w, axis=1)[:, np.newaxis]
        w_sum += w_sum == 0
        self.w /= w_sum

        if np.any(np.isnan(self.w)):
            pdb.set_trace()
        self.t += 1

    def one_hot(self, y):
        y_one_hot = np.zeros((self.K,), dtype=np.float64)
        y_one_hot[y] = 1
        return y_one_hot

    def evaluate(self, X,Y, **kwargs):
        inds = range(Y.shape[0])
        np.random.shuffle(inds)

        X = X[inds, :]
        Y = Y[inds, :]

        # normalize so that all feature are in [0,1].
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        x_max += x_max == x_min

        X_norm = (X - x_min) / (x_max - x_min)
        return super(online_exponentiated_sq_loss, self).evaluate(X_norm,Y, **kwargs)
    
    

class online_multi_svm(online_learner):
    # K: nbr of classes
    # D: nbr of feature dimension
    def __init__(self, lam, nbr_classes, feature_size, margin=1):
        self.lam = float(lam)
        self.sqrt_lam = numpy.sqrt(self.lam)
        self.K = nbr_classes
        self.D = feature_size
        self.w = numpy.zeros((self.K, self.D), dtype = numpy.float64)

        self.margin=1
        self.t = 1
        self.batch_size = 20
        self.cur_grad = numpy.zeros_like(self.w)
        
    def predict(self, x):
        y_p = np.argmax(numpy.dot(self.w, x))
        return y_p

    def compute_margin(self, y_gt, x, compete=False):
        score = self.w.dot(x)
        mask = np.ones((self.K,), dtype=bool)
        mask[y_gt] = False
        if not compete:
            return -max(score[mask]) + score[y_gt]

        y_c = np.argmax(score[mask])
        y_c += y_c >= y_gt
        score_c = score[y_c]

        #assert(score[y_c] == score_c)
        #for i in range(self.K):
        #    assert(score_c >= score[i] or i == y_gt)
        return -score_c + score[y_gt], y_c 

        
    def compute_single_loss(self, y_gt, y_p, x):
        return max(0, 1 - self.compute_margin(y_gt, x)) + self.lam * np.sum(self.w * self.w) / 2.0
        
      

    def fit(self, x, y_gt, do_batch = True):
        alpha = 1.0 / (self.lam * self.t)
        margin, y_c = self.compute_margin(y_gt, x, compete=True)
        grad = - alpha * self.lam * self.w
        if margin < self.margin:
            alpha_x = alpha * x
            grad[y_gt, :] += alpha_x
            grad[y_c, :] -= alpha_x

        if not do_batch:
            self.w += grad
            w_norm = np.linalg.norm(self.w)
            if w_norm > 1.0 / self.lam:
                self.w = self.w / (w_norm * (self.sqrt_lam))        
        else:
            self.cur_grad += grad
            if self.t % self.batch_size == 0:
                self.w += self.cur_grad / float(self.batch_size)
                self.cur_grad = numpy.zeros_like(self.w)
                w_norm = np.linalg.norm(self.w)
                if w_norm > 1.0 / self.lam:
                    self.w = self.w / (w_norm * (self.sqrt_lam))        

        self.t += 1
        return self.w

    def evaluate(self, X, Y, **kwargs):
        inds = range(Y.shape[0])
        numpy.random.shuffle(inds)
        
        X = X[inds, :]
        Y = Y[inds, :]


        assert((numpy.unique(Y) == numpy.array([-1, 1])).all())

        X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
        return super(online_multi_svm, self).evaluate(X_norm, Y, **kwargs)


def poly_kernel_func(x0, x1, kernel_params):
    return numpy.power((numpy.dot(x0, x1) + kernel_params['c']), kernel_params['d'])

class online_kernel_svm(online_learner):
  def __init__(self, feature_size, lam, kernel_type = 'poly', kernel_params = {'c' : 1, 'd': 2}):
    self.kernel_type = kernel_type
    self.kernel_params = kernel_params

    self.sample_weights = numpy.zeros((1,1))
    self.samples = numpy.zeros((1, feature_size))

    self.feature_size = feature_size

    self.lam = lam
    self.t = float(1)



    if self.kernel_type == 'poly':
      self.kernel_func = poly_kernel_func
    else:
      raise RuntimeError("unknown kernel: {}".format(kernel_type))
      
  def compute_single_loss(self, y_gt, y_p, x=None):
    return 0

  def predict(self, x):
    s = 0
    for (sample_weight, sample) in zip(self.sample_weights, self.samples):
      s += sample_weight * self.kernel_func(sample, x, self.kernel_params)
    return s

  def fit(self, x, y_gt, do_batch = True):
    f_x = self.predict(x)

    eta = 1. / (self.lam * self.t)
    # self.sample_weights *= (1 - self.lam * eta)
    self.sample_weights *= ( 1 - 1. /self.t)

    if 1 - y_gt * f_x > 0:
      self.samples = numpy.vstack((self.samples, x))
      self.sample_weights = numpy.vstack((self.sample_weights, y_gt * eta))

    self.t += 1

    #TODO shrink????

  def evaluate(self, X, Y, **kwargs):
    inds = range(Y.shape[0])
    numpy.random.shuffle(inds)
    
    X = X[inds, :]
    Y = Y[inds, :]

    # X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
    return super(self.__class__, self).evaluate(X, Y, **kwargs)

class bayesian_linear_regression(object):
  def __init__(self, feature_size):
    self.sigma_sq = 2.0
    self.P = numpy.eye(feature_size)

    self.mu = numpy.zeros((feature_size, 1))
    self.J = numpy.dot(self.P, self.mu_t)

  def predict(self, x):
    f_x = numpy.dot(self.mu_t, x)
    if f_x >= 0:
      return 1
    else:
      return -1

  def fit(self, x_t, y_t):
    self.J += y_t * x_t / self.sigma_sq
    self.P += numpy.outer(x_t, x_t) / self.sigma_sq

    self.mu = numpy.dot(numpy.linalg.pinv(self.P), self.J)

  def evaluate(self, X, Y, **kwargs):
    inds = range(Y.shape[0])
    numpy.random.shuffle(inds)
    
    X = X[inds, :]
    Y = Y[inds, :]

    return super(self.__class__, self).evaluate(X, Y, **kwargs)
    
def main():
    data_o = convert.dataset_oakland(numpy_fn = 'data/oakland_part3_am_rf.node_features.npz',
                                     fn = 'data/oakland_part3_am_rf.node_features')

    test_data_o = convert.dataset_oakland(numpy_fn = 'data/oakland_part3_an_rf.node_features.npz',
                                          fn = 'data/oakland_part3_an_rf.node_features')
    method = 'multi_svm'

    if method == 'svm':

        lam = 1e-4
        learner = online_svm(lam = lam, feature_size = data_o.features.shape[1])

        wall_and_ground_inds = numpy.where(numpy.logical_or(data_o.facade_inds, data_o.ground_inds))[0]
        test_wall_and_ground_inds = numpy.where(numpy.logical_or(test_data_o.facade_inds, 
                                                                 test_data_o.ground_inds))[0]

        Y = data_o.labels[wall_and_ground_inds]
        X = data_o.features[wall_and_ground_inds, :]

        Y[Y == data_o.label_map['facade']] = -1
        Y[Y == data_o.label_map['ground']] = 1

        Y_test = test_data_o.labels[test_wall_and_ground_inds]
        X_test = test_data_o.features[test_wall_and_ground_inds, :]

        Y_test[Y_test == test_data_o.label_map['facade']] = -1
        Y_test[Y_test == test_data_o.label_map['ground']] = 1

        ypred, losses = learner.evaluate(X, Y)

        classification = ypred > 0
        n_right = (classification == (Y > 0)).sum()
        accuracy = n_right / float(Y.shape[0])

    elif method == 'multi_svm' or method == 'EG':
        
        lam = 4e-4
        if method == 'multi_svm':
            learner = online_multi_svm(lam=lam, nbr_classes=5, feature_size = data_o.features.shape[1])
        else:
            learner = online_exponentiated_sq_loss(nbr_classes=5, feature_size = data_o.features.shape[1], lam=lam, grad_scale=1e-3)

        X = data_o.features
        Y = np.array([ [data_o.label2ind[l[0]]] for l in data_o.labels ])

        X_test = test_data_o.features
        Y_test = np.array([ [test_data_o.label2ind[l[0]]] for l in test_data_o.labels ])

        ypred, losses = learner.evaluate(X,Y)

        n_right = np.sum(ypred == Y)
        accuracy = np.sum(ypred == Y) / float(Y.shape[0])

    elif method == 'kernel_svm':
        
        lam = 4e-4
        learner = online_kernel_svm(lam=lam, feature_size = data_o.features.shape[1])

        X = data_o.features
        Y = np.array([ [data_o.label2ind[l[0]]] for l in data_o.labels ])

        X_test = test_data_o.features
        Y_test = np.array([ [test_data_o.label2ind[l[0]]] for l in test_data_o.labels ])


        ypred, losses = learner.evaluate(X,Y)

        n_right = np.sum(ypred == Y)
        accuracy = np.sum(ypred == Y) / float(Y.shape[0])
    elif method == 'blr':
      # blr = bayesian_linear_regression(feature_size = data_o.features.shape[0])
      
    else:
      raise RuntimeError("method not understood")

      
    cum_losses = numpy.cumsum(losses - learner.lam * np.sum(learner.w * learner.w) / 2.)

    print "right, accuracy: {}, {}".format(n_right, accuracy)
    
    y_test_pred, y_test_losses = learner.evaluate(X_test, 
                                                  Y_test,
                                                  fit = False)
    if method in ['svm']:
      ypred = ypred > 0
      ypred[ypred == 0] = -1

      y_test_pred = y_test_pred > 0
      y_test_pred[y_test_pred == 0] = -1

    # print "w: {}".format(learner.w)
    plt.figure()
    plt.plot(range(losses.shape[0]), (cum_losses) / learner.t)
    plt.show(block = False)

    cm_train = mlu.confusion_matrix(Y, ypred)
    cm_test = mlu.confusion_matrix(Y_test, y_test_pred)
    print "{} training accuracy: {}".format(method, cm_train.overall_accuracy)
    print "{} test accuracy: {}".format(method, cm_test.overall_accuracy)


    pdb.set_trace()

if __name__ == '__main__':
  pdbw.pdbwrap(main)()
