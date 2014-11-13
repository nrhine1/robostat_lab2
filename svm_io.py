

import numpy as np
import gtkutils.file_util as fu
from online_learn import get_kernel_func


def save_ksvm(ksvm):
    for svm in ksvm.svms:
        svm.kernel_func = None

    fu.save_pkl(ksvm, 'ksvm_learner.pkl')

def load_ksvm():
    ksvm = fu.load_pkl('ksvm_learner.pkl')
    
    for svm in ksvm.svms:
        svm.kernel_func = get_kernel_func(kernel_type='unif', H=1.02723450351)
    return ksvm
    
