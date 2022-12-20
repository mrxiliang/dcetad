import argparse

import numpy as np
import torch

import classifier as tc
from data_loader import Data_Loader


def setup_seed(seed):
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def load_trans_data(args):
    setup_seed(args.seed)
    dl = Data_Loader(args)
    train_real, val_real, val_fake = dl.get_dataset(args.dataset)
    y_test_fscore = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])
    ratio = 100.0 * len(val_real) / (len(val_real) + len(val_fake))

    n_train, n_dims = train_real.shape
    rots = np.random.randn(args.n_rots, n_dims, args.d_out)

    print('Calculating transforms')
    x_train = np.stack([train_real.dot(rot) for rot in rots], 2)
    val_real_xs = np.stack([val_real.dot(rot) for rot in rots], 2)
    val_fake_xs = np.stack([val_fake.dot(rot) for rot in rots], 2)
    x_test = np.concatenate([val_real_xs, val_fake_xs])
    return x_train, x_test, y_test_fscore, ratio


def train_anomaly_detector(args):
    x_train, x_test, y_test, ratio = load_trans_data(
        args)
    tc_obj = tc.TransClassifier(args)
    f_score, auc, ap = tc_obj.fit_trans_classifier(x_train, x_test, y_test, ratio)
    return (f_score, auc, ap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_rots', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epoch', default=1, type=int)
    parser.add_argument('--d_out', default=32, type=int)
    parser.add_argument('--dataset', default='arrhythmia', type=str)
    parser.add_argument('--exp', default='affine', type=str)
    parser.add_argument('--c_pr', default=0, type=int)
    parser.add_argument('--true_label', default=1, type=int)
    parser.add_argument('--ndf', default=8, type=int)
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--seed', default=2, type=int)

    args = parser.parse_args()
    print("Dataset: ", args.dataset)

    if args.dataset == 'thyroid' or args.dataset == 'arrhythmia' or args.dataset == 'satimage-2' or args.dataset == "handoutlines":
        n_iters = args.n_iters
        f_scores = np.zeros(n_iters)
        aucs = np.zeros(n_iters)
        aps = np.zeros(n_iters)
        times = np.zeros(n_iters)

        for i in range(n_iters):
            f_scores[i], aucs[i], aps[i] = train_anomaly_detector(args)
        print("AVG f1 Score: ", f_scores.mean(),
              " AVG AUC: ", aucs.mean(),
              " AVG AP: ", aps.mean())
    else:
        train_anomaly_detector(args)
