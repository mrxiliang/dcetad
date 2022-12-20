import numpy as np
import scipy.io

from dcsae_processing import preprocess_small_data

# arrhythmia
en_input1 = 274
en_output1 = 200
en_input2 = 200
en_output2 = 400
de_input1 = 400
de_output1 = 200
de_input2 = 200
de_output2 = 274
epochs = 200
matfile = scipy.io.loadmat("data/arrhythmia.mat")
bs = 452
lr = 0.001


def process():
    res = preprocess_small_data(en_input1, en_output1,
                                en_input2, en_output2,
                                de_input1, de_output1,
                                de_input2, de_output2,
                                epochs, matfile, bs, lr)
    return res


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


res = process()
