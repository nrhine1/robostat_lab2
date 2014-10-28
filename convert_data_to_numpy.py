#!/usr/bin/env python
import os, sys, argh, pdb, numpy

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
        self.features = self.data[:, 5:]

def main():
    line_skip = 3

    do = dataset_oakland(numpy_fn = 'data/oakland_part3_am_rf.node_features.npz')

if __name__ == '__main__':
    argh.dispatch_command(main)
