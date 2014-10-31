import numpy as np
import numpy, pdb
import os,sys
import matplotlib.pyplot as plt

import convert_data_to_numpy as convert

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

      loss = self.compute_single_loss(Y[xi], y_p, x)
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

  def evaluate(self, X, Y):
    inds = range(Y.shape[0])
    numpy.random.shuffle(inds)
    
    X = X[inds, :]
    Y = Y[inds, :]

    X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
    return super(online_svm, self).evaluate(X_norm, Y)


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

    def evaluate(self, X, Y):
        inds = range(Y.shape[0])
        numpy.random.shuffle(inds)
        
        X = X[inds, :]
        Y = Y[inds, :]

        X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
        return super(online_multi_svm, self).evaluate(X_norm, Y)


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

  def evaluate(self, X, Y):
    inds = range(Y.shape[0])
    numpy.random.shuffle(inds)
    
    X = X[inds, :]
    Y = Y[inds, :]

    # X_norm = X / numpy.linalg.norm(X, axis = 1)[:, numpy.newaxis]
    return super(self.__class__, self).evaluate(X, Y)


def main():
    data_o = convert.dataset_oakland(numpy_fn = 'data/oakland_part3_am_rf.node_features.npz')
    method = 'kernel_svm'

    if method == 'svm':

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

    elif method == 'multi_svm':
        
        lam = 4e-4
        osvm = online_multi_svm(lam=lam, nbr_classes=5, feature_size = data_o.features.shape[1])

        X = data_o.features
        Y = np.array([ [data_o.label2ind[l[0]]] for l in data_o.labels ])

        ypred, losses = osvm.evaluate(X,Y)

        n_right = np.sum(ypred == Y)
        accuracy = np.sum(ypred == Y) / float(Y.shape[0])

    elif method == 'kernel_svm':
        
        lam = 4e-4
        osvm = online_kernel_svm(lam=lam, feature_size = data_o.features.shape[1])

        X = data_o.features
        Y = np.array([ [data_o.label2ind[l[0]]] for l in data_o.labels ])

        ypred, losses = osvm.evaluate(X,Y)

        n_right = np.sum(ypred == Y)
        accuracy = np.sum(ypred == Y) / float(Y.shape[0])
    else:
      raise RuntimeError("method not understood")


    cum_losses = numpy.cumsum(losses - osvm.lam * np.sum(osvm.w * osvm.w) / 2.)

    print "right, accuracy: {}, {}".format(n_right, accuracy)
    print "w: {}".format(osvm.w)
    plt.figure()
    plt.plot(range(losses.shape[0]), (cum_losses) / osvm.t)
    plt.show(block = False)



    pdb.set_trace()

if __name__ == '__main__':
  pdbw.pdbwrap(main)()
