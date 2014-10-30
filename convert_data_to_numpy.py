#!/usr/bin/env python
import os, sys, argh, pdb, numpy
import sklearn.linear_model

def test_scikit_learner():
    do = dataset_oakland(numpy_fn = 'data/oakland_part3_am_rf.node_features.npz')
    
    C = 1
    learner = sklearn.linear_model.LogisticRegression(penalty = 'l2', C = C)

    split_size = .8

    inds = range(do.labels.shape[0])
    train_inds = sorted(numpy.random.choice(inds,
                                            size = int(split_size * do.labels.shape[0]),
                                            replace = False))
    print "here"
    test_inds = [i for i in inds if i not in train_inds]
    print "therer"


class dataset_oakland(object):
    def __init__(self,  fn = 'data/oakland_part3_am_rf.node_features', numpy_fn = None):
        if numpy_fn is None or not os.path.isfile(numpy_fn):
            if not os.path.isfile(fn):
                raise RuntimeError("numpy dataset file not valid: {}".format(fn))

            print "generating data from text"
            self.data = numpy.genfromtxt(open(fn, 'r'), skip_header = 3)
            save_fn = fn + '.npz'
            numpy.savez_compressed(save_fn, self.data)
            print "saved to {}".format(save_fn)
        else:
            print "generating data from numpy file"
            if not os.path.isfile(numpy_fn):
                raise RuntimeError("dataset file not valid: {}".format(numpy_fn))
            self.data = numpy.load(numpy_fn)['arr_0']
           
        self.points = self.data[:, :3]
        self.labels = self.data[:, 4]

        if self.labels.ndim == 1:
            self.labels = self.labels[:, numpy.newaxis]

        self.features = self.data[:, 5:]

        self.points_normalized = self.points / numpy.linalg.norm(self.points, axis = 1)[:, numpy.newaxis]
        self.features = numpy.hstack((self.points_normalized, self.features))

        self.label_map  = {'veg' : 1004, 'wire' : 1100, 'pole':  1103, 'ground' : 1200, 'facade' : 1400}

        for (label_type, val) in self.label_map.items():
            setattr(self, '{}_inds'.format(label_type), self.labels == val)

def main():
    line_skip = 3

    do = dataset_oakland(numpy_fn = 'data/oakland_part3_am_rf.node_features.npz')
    test_scikit_learner()
    pdb.set_trace()

if __name__ == '__main__':
    argh.dispatch_command(main)
