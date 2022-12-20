import numpy as np
from sklearn import preprocessing

from dcsae_processing import preprocess_small_data1


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def HandOutlines_process():
    X_train, y_train, X_test, y_test = HandOutlines_loader('data', 'HandOutlines')
    X_data = np.concatenate([X_train, X_test])
    y_data = np.concatenate([y_train, y_test])
    X_res = preprocess_small_data1(2709, 4000, 4000, 6000, 6000, 4000, 4000, 2709, 200, X_data,
                                   1370, 0.0001)

    return X_res, y_data


def HandOutlines_loader(dataset_path, dataset_name):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test


HandOutlines_X_res, HandOutlines_y_data = HandOutlines_process()
