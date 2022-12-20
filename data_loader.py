import numpy as np
import pandas as pd
import scipy.io

from dcsae_params import res
from dcsae_processing import preprocess_big_data


class Data_Loader:

    def __init__(self, n_trains=None):
        self.n_train = n_trains
        self.urls = [
            "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
            "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
        ]

    def norm_kdd_data(self, train_real, val_real, val_fake, cont_indices):
        symb_indices = np.delete(np.arange(train_real.shape[1]), cont_indices)
        mus = train_real[:, cont_indices].mean(0)
        sds = train_real[:, cont_indices].std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            bin_cols = xs[:, symb_indices]
            cont_cols = xs[:, cont_indices]
            cont_cols = np.array([(x - mu) / sd for x in cont_cols])
            return np.concatenate([bin_cols, cont_cols], 1)

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        return train_real, val_real, val_fake

    def norm_data(self, train_real, val_real, val_fake):
        mus = train_real.mean(0)
        sds = train_real.std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            return np.array([(x - mu) / sd for x in xs])

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        return train_real, val_real, val_fake

    def norm(self, data, mu=1):
        return 2 * (data / 255.) - mu

    def get_dataset(self, dataset_name):
        if dataset_name == 'kdd':
            return self.KDD99_train_valid_data()
        if dataset_name == 'kddrev':
            return self.KDD99Rev_train_valid_data()
        if dataset_name == 'thyroid':
            return self.Thyroid_train_valid_data(res)
        if dataset_name == 'arrhythmia':
            return self.Arrhythmia_train_valid_data(res)
        if dataset_name == 'handoutlines':
            from handoutlines_params import HandOutlines_X_res, HandOutlines_y_data
            return self.HandOutlines_train_valid_data(HandOutlines_X_res, HandOutlines_y_data)
        if dataset_name == 'satimage-2':
            return self.Satimage_train_valid_data(res)

    def Arrhythmia_train_valid_data(self, Arrhythmia_res):
        data = scipy.io.loadmat("data/arrhythmia.mat")
        samples = Arrhythmia_res
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]  # 386 norm
        anom_samples = samples[labels == 1]  # 66 anom

        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]  # 193 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples
        return self.norm_data(x_train, val_real, val_fake)

    def Thyroid_train_valid_data(self, Thyroid_res):
        data = scipy.io.loadmat("data/thyroid.mat")
        samples = Thyroid_res
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]
        anom_samples = samples[labels == 1]

        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]

        val_real = norm_samples[n_train:]
        val_fake = anom_samples
        return self.norm_data(x_train, val_real, val_fake)

    def HandOutlines_train_valid_data(self, HandOutlines_X_res, HandOutlines_y_data):
        samples = HandOutlines_X_res
        labels = HandOutlines_y_data

        norm_samples = []  # 0 is abnormal sample, 1 is normal sample
        anom_samples = []

        for i in range(len(labels)):
            if labels[i] == 1:
                norm_samples.append(samples[i])
            else:
                anom_samples.append(samples[i])

        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]

        val_real = norm_samples[n_train:]
        val_fake = anom_samples

        x_train = np.array(x_train)
        val_real = np.array(val_real)
        val_fake = np.array(val_fake)

        return self.norm_data(x_train, val_real, val_fake)

    def Satimage_train_valid_data(self, Satimage_res):
        data = scipy.io.loadmat("data/satimage-2.mat")
        samples = Satimage_res
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]
        anom_samples = samples[labels == 1]

        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]

        val_real = norm_samples[n_train:]

        val_fake = anom_samples
        return self.norm_data(x_train, val_real, val_fake)

    def KDD99_preprocessing(self):
        df_colnames = pd.read_csv(self.urls[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
        df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
        df = pd.read_csv(self.urls[0], header=None, names=df_colnames['f_names'].values)
        df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
        df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
        samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])

        smp_keys = samples.keys()
        cont_indices = []
        for cont in df_continuous['f_names']:
            cont_indices.append(smp_keys.get_loc(cont))

        labels = np.where(df['status'] == 'normal.', 1, 0)
        return np.array(samples), np.array(labels), cont_indices

    def KDD99_train_valid_data(self):
        samples, labels, cont_indices = self.KDD99_preprocessing()

        samples = preprocess_big_data(121, 400, 400, 175, 175, 400, 400, 121, 245, samples, 44911, 0.001)

        anom_samples = samples[labels == 1]  # norm: 97278
        norm_samples = samples[labels == 0]  # attack: 396743

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples
        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices)

    def KDD99Rev_train_valid_data(self):
        samples, labels, cont_indices = self.KDD99_preprocessing()

        samples = preprocess_big_data(121, 400, 400, 175, 175, 400, 400, 121, 235, samples, 44911, 0.001)

        norm_samples = samples[labels == 1]  # norm: 97278

        # Randomly draw samples labeled as 'attack'
        # so that the ratio btw norm:attack will be 4:1
        # len(anom) = 24,319
        anom_samples = samples[labels == 0]  # attack: 396743

        rp = np.random.permutation(len(anom_samples))
        rp_cut = rp[:24319]
        anom_samples = anom_samples[rp_cut]  # attack:24319

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples
        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices)
