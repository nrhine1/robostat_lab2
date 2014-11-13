#!/usr/bin/env python
import collections
import copy
import numpy as np
import numpy, pdb, warnings
import os,sys
import matplotlib.pyplot as plt
import argh

import convert_data_to_numpy as convert
import sklearn.metrics

import gtkutils.ml_util as mlu
import gtkutils.pdbwrap as pdbw

class online_learner(object):
  def __init__(self):
    pass

  def predict(self, x):
    return RuntimeError("not overridden")

  def fit(self, x, y):
    return RuntimeError("not overridden")

  def compute_single_loss(self, y_gt, y_p, x=None):
    return RuntimeError("not overridden")

  def evaluate(self, X, Y, fit = True, class_weights=None):
    n_samples = X.shape[0]

    Y_p = numpy.zeros_like(Y)
    losses = numpy.zeros_like(Y)
    total_loss = 0

    for (xi, x) in enumerate(X):
      if xi % 1000 == 0:
        print "round ", xi, total_loss

      y_p = self.predict(x)
      if class_weights is None:
          #TODO change all predictions (none kernelized versions esp.) to return vector of weights instead 
          # of classification
          Y_p[xi] = y_p
      else:
          Y_p[xi] = np.argmax(y_p)

      loss = self.compute_single_loss(Y[xi], y_p, x)
      total_loss += loss
      losses[xi] = loss
      
      if fit:
        if class_weights is None:
          self.fit(x, Y[xi])
        else:
          self.fit(x, Y[xi], weight=class_weights[Y[xi]], y_p=y_p)

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
    y_p, losses =  super(online_svm, self).evaluate(X_norm, Y, **kwargs)
    ind_inv = numpy.zeros_like(inds)
    ind_inv[inds] = numpy.arange(0, Y.shape[0], 1)
    y_p = y_p[ind_inv]
    return y_p, losses


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
        y_p, losses =  super(self.__class__, self).evaluate(X_norm,Y, **kwargs)
        ind_inv = numpy.zeros_like(inds)
        ind_inv[inds] = numpy.arange(0, Y.shape[0], 1)
        y_p = y_p[ind_inv]
        return y_p, losses

    

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

        X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
        y_p, losses = super(self.__class__, self).evaluate(X_norm, Y, **kwargs)
        
        ind_inv = numpy.zeros_like(inds)
        ind_inv[inds] = numpy.arange(0, Y.shape[0], 1)
        y_p = y_p[ind_inv]
        return y_p, losses

def get_kernel_func(kernel_type=None, H=1, params=None):
    if kernel_type == 'poly':
        if params is None:
            params = {'c':1, 'd':2}
        return lambda a,b,params=params: (np.dot(a,b) + params['c'])**params['d']
    elif kernel_type == 'unif':
        return lambda a,b,H=H: 0.5*( np.linalg.norm(a-b) <= H) / H
    elif kernel_type == 'epane':
        return lambda a,b,H=H: max(0.75*(1 - (np.linalg.norm(a-b)/H)**2), 0)
    elif kernel_type == 'dot':
        return np.dot
    else:
        print "WARNING. not valid kernel; using dot product instead"
        return np.dot

class online_kernel_svm(online_learner):
  def __init__(self, feature_size, lam, max_nbr_pts=2000, kernel_func = np.dot): 
    self.kernel_func = kernel_func

    #self.sample_weights = collections.deque([], max_nbr_pts)
    self.sw_f = 0
    self.sw_r = 0
    self.sw_max_len = max_nbr_pts
    self.sample_weights = np.zeros(max_nbr_pts, dtype=np.float64)
    self.samples = collections.deque([], max_nbr_pts)

    self.feature_size = feature_size

    self.lam = lam
    self.t = float(1)

  def compute_single_loss(self, y_gt, y_p, x=None):
    return 0

  def predict(self, x):
    s = 0
    #for (sample_weight, sample) in zip(self.sample_weights, self.samples):
    #  s += sample_weight * self.kernel_func(sample, x)
    for (si, sample) in enumerate(self.samples):
        s += self.sample_weights[ (self.sw_f+si) % self.sw_max_len ] * self.kernel_func(sample, x)
    return s

  def fit(self, x, y_gt, do_batch = True, weight=1, y_p=None):
    if y_p is None:
        y_p = self.predict(x)
    eta = 1. / (self.lam * self.t)
    # self.sample_weights *= (1 - self.lam * eta)
    weight_multiplier = 1 - 1. / self.t
#    sample_weights = copy.deepcopy(self.sample_weights)
#    for sw in sample_weights:
#        self.sample_weights.append(sw * weight_multiplier)
    self.sample_weights *= weight_multiplier

    if 1 - y_gt * y_p > 0:
      self.samples.append(x)
#      self.sample_weights.append(y_gt * eta * weight)
      self.sample_weights[self.sw_r] = y_gt * eta * weight
      self.sw_r = (self.sw_r + 1) % self.sw_max_len
      if self.sw_r == self.sw_f:
          self.sw_f = (self.sw_f + 1) % self.sw_max_len
      

    self.t += 1

    #TODO shrink????

  def evaluate(self, X, Y, **kwargs):
    inds = range(Y.shape[0])
    numpy.random.shuffle(inds)
    
    X = X[inds, :]
    Y = Y[inds, :]
    
    y_p, losses = super(self.__class__, self).evaluate(X, Y, **kwargs)
    ind_inv = numpy.zeros_like(inds)
    ind_inv[inds] = numpy.arange(0, Y.shape[0], 1)
    y_p = y_p[ind_inv]
    return y_p, losses

    # X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]

class online_multi_kernel_svm(online_learner):
    def __init__(self, nbr_classes, feature_size, lam, max_nbr_pts, kernel_func = np.dot):
        self.svms = [online_kernel_svm(feature_size, lam, max_nbr_pts, kernel_func)  for ci in range(nbr_classes)]

    def compute_single_loss(self, y_gt, y_p, x=None):
        return 0

    def predict(self, x):
        return [ svm.predict(x) for svm in self.svms ]

    def fit(self, x, y_gt, do_batch=True,weight=1.0, y_p=None):
        if y_p is None:
            y_p = [ svm.predict(x) for svm in self.svms ]
        for si, svm in enumerate(self.svms):
            if si != y_gt:
                svm.fit(x, -1, do_batch=do_batch, weight=weight, y_p=y_p[si])
            else:
                svm.fit(x, 1,do_batch=do_batch, weight=weight, y_p=y_p[si])
        return 0

    def evaluate(self, X, Y, **kwargs):
        inds = range(Y.shape[0])
        numpy.random.shuffle(inds)
        
        X = X[inds, :]
        Y = Y[inds, :]

        X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
        y_p, losses = super(self.__class__, self).evaluate(X_norm, Y, **kwargs)
        
        ind_inv = numpy.zeros_like(inds)
        ind_inv[inds] = numpy.arange(0, Y.shape[0], 1)
        y_p = y_p[ind_inv]
        return y_p, losses


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
    
def duplicate_data(X, Y, copy_list):
  def copy_data(X, Y, n_copies, label_idx):
    data_inds = list(numpy.arange(0, Y.shape[0], 1)[(Y == label_idx).ravel()])
    data_inds = numpy.hstack(n_copies * data_inds)
    X_new = X[data_inds, :]
    Y_new = Y[data_inds, :]
    return X_new, Y_new

  X_l = [X]
  Y_l = [Y]

  for (i, c) in enumerate(copy_list):
    if c <= 0:
      continue

    xn, yn = copy_data(X, Y, c, i)
    X_l.append(xn)
    Y_l.append(yn)

  X = numpy.vstack(X_l)
  Y = numpy.vstack(Y_l)
  return X, Y

def add_random_features(X, dim = 13, scaling = .1):
  rfs = scaling * numpy.random.randn(int(X.shape[0]), dim)
  X = numpy.hstack((X, rfs))
  return X

def add_corrupted_features(X, sigma_scaling =.1):  
  mean_feat = numpy.mean(X, axis = 0)
  std_feat = numpy.std(X, axis = 0)
  
  noisy_X = numpy.zeros_like(X)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for (xi, x) in enumerate(X):
      noisy_X[xi, :] = numpy.random.multivariate_normal(x, sigma_scaling * numpy.eye(x.shape[0], dtype = numpy.int))

  X = numpy.hstack((X, noisy_X))
  return X
                                       
@argh.arg('--method', choices = ['EG', 'multi_svm', 'multi_ksvm'], required = True)
@argh.arg('--kernel-type', choices = ['unif', 'poly', 'epane', 'dot'])
@argh.set_all_toggleable()
def main(method = 'EG',
         do_copy = True,
         random_features_dim = 0,
         random_features_scaling = 1.0,
         do_add_corrupted_features = False,
         corruption_sigma_scaling = 1.0,
         kernel_type = 'unif',
         max_nbr_points = 10000):
    data_o = convert.dataset_oakland(numpy_fn = 'data/oakland_part3_am_rf.node_features.npz',
                                     fn = 'data/oakland_part3_am_rf.node_features')

    test_data_o = convert.dataset_oakland(numpy_fn = 'data/oakland_part3_an_rf.node_features.npz',
                                          fn = 'data/oakland_part3_an_rf.node_features')
    copy_list = [0, 0, 0, 0, 0]
    compute_kernel_width = True
    class_weights = None

    #noise
    noise_dim = random_features_dim + (do_add_corrupted_features) * 13

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

    elif method == 'multi_svm' or method == 'EG' or method == 'multi_ksvm':
        X = data_o.features
        Y = np.array([ [data_o.label2ind[l[0]]] for l in data_o.labels ])
        X_test = test_data_o.features

        # if 0 it doesn't do anything
        X = add_random_features(X, random_features_dim)
        X_test = add_random_features(X_test, random_features_dim)

        if do_add_corrupted_features:
          X = add_corrupted_features(X, corruption_sigma_scaling)
          X_test = add_corrupted_features(X_test, corruption_sigma_scaling)

        Y_test = np.array([ [test_data_o.label2ind[l[0]]] for l in test_data_o.labels ])

        # compute kernel width
        H=1.02723450351
        if compute_kernel_width:
            n_samples = 200000
            dists = np.zeros((n_samples,))
            for rand_pi in range(n_samples):
                rand_p = np.asarray(np.random.uniform(0, 1-1e-9, (2,)) * X.shape[0], dtype=int)
                dists[rand_pi] = np.linalg.norm(X[rand_p[0],:] - X[rand_p[1],:])

            H = np.sort(dists)[int(n_samples / 2)]

        
        # classifier set up
        svm_lam = 4e-4
        eg_lam = 4e-4
        ksvm_lam = 4e-4
        kwargs = {}

        if method == 'multi_svm':
          if do_copy:
            copy_list = [7, 10, 10, 0, 2]
          learner = online_multi_svm(lam=svm_lam, nbr_classes=5, 
                                     feature_size = data_o.features.shape[1] + noise_dim)
        elif method == 'EG':
          if do_copy:
              copy_list = [5, 50, 12, 0, 2]
          learner = online_exponentiated_sq_loss(nbr_classes=5, 
                                                 feature_size = data_o.features.shape[1] + noise_dim,
                                                 lam=eg_lam, 
                                                 grad_scale=1e-3)
        elif method == 'multi_ksvm':
          class_weights = np.array([5, 10, 10, 0, 2]) + 1.0
          kwargs = {'class_weights' : class_weights}
          learner = online_multi_kernel_svm(nbr_classes=5, 
                                            feature_size = data_o.features.shape[1] + noise_dim, 
                                            lam=ksvm_lam,
                                            max_nbr_pts = max_nbr_points,
                                            kernel_func = get_kernel_func(kernel_type=kernel_type, H=H)) 

        # data duplication
        X,Y = duplicate_data(X, Y, copy_list)

        # prediction/learning 
        ypred, losses = learner.evaluate(X,Y,**kwargs)
          
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
      raise NotImplementedError("no blr")
    else:
      raise RuntimeError("method not understood")

      
    # cum_losses = numpy.cumsum(losses - learner.lam * np.sum(learner.w * learner.w) / 2.)

    # print "right, accuracy: {}, {}".format(n_right, accuracy)
    
    y_test_pred, y_test_losses = learner.evaluate(X_test, 
                                                  Y_test,
                                                  fit = False,
                                                  class_weights = class_weights)
    if method in ['svm']:
      ypred = ypred > 0
      ypred[ypred == 0] = -1

      y_test_pred = y_test_pred > 0
      y_test_pred[y_test_pred == 0] = -1

    # print "w: {}".format(learner.w)
    # plt.figure()
    # plt.plot(range(losses.shape[0]), (cum_losses) / learner.t)
    # plt.show(block = False)

    cm_train = mlu.confusion_matrix(Y, ypred)
    cm_test = mlu.confusion_matrix(Y_test, y_test_pred)
    
    print cm_train.summary_table(data_o.ind2class)
    print cm_test.summary_table(test_data_o.ind2class)

    write_classification(test_data_o.points, y_test_pred, 'test_classification.txt')
    write_classification(data_o.points, ypred, 'training_classification.txt')
    print "{} training accuracy: {}".format(method, cm_train.overall_accuracy)
    print "{} test accuracy: {}".format(method, cm_test.overall_accuracy)


    pdb.set_trace()

def write_classification(X_points, Y_pred, fn = 'blah.txt'):
  with open(fn, 'w') as f:
    for (point, y_pred) in zip(X_points, Y_pred):
      f.write('{:.3f} {:.3f} {:.3f} {}\n'.format(point[0], point[1], point[2], y_pred[0]))
    

if __name__ == '__main__':
  pdbw.pdbwrap(argh.dispatch_command)(main)
