import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import dcsae


def preprocess_small_data(en_input1, en_output1, en_input2, en_output2, de_input1, de_output1, de_input2, de_output2,
                          epochs,
                          matfile, batch_size, lr):
    dcsae.en_input1 = en_input1
    dcsae.en_output1 = en_output1
    dcsae.en_input2 = en_input2
    dcsae.en_output2 = en_output2
    dcsae.de_input1 = de_input1
    dcsae.de_output1 = de_output1
    dcsae.de_input2 = de_input2
    dcsae.de_output2 = de_output2
    dcsae.epochs = epochs

    # the weight of the expected mean activation value and KL divergence
    expect_tho = 0.05
    tho_tensor = torch.FloatTensor([expect_tho for _ in range(en_output2)])
    if torch.cuda.is_available():
        tho_tensor = tho_tensor.cuda()
    beta = 1

    X_data = matfile['X']
    X_data = torch.FloatTensor(X_data)

    X_dataset = TensorDataset(X_data, X_data)
    X_DataLoader = DataLoader(dataset=X_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        autoencoder = dcsae.DCSparseAutoEncoder().cuda()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        loss_func = nn.MSELoss().cuda()
        loss_train = np.zeros((dcsae.epochs, 1))
    else:
        autoencoder = dcsae.DCSparseAutoEncoder()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        loss_func = nn.MSELoss()
        loss_train = np.zeros((dcsae.epochs, 1))

    # train
    for epoch in range(dcsae.epochs):
        for batchidx, (x, _) in enumerate(X_DataLoader):
            if torch.cuda.is_available():
                x = x.cuda()

            encoded, decoded = autoencoder(x)
            loss = loss_func(decoded, x)
            kl = KL_divergence(tho_tensor, encoded)
            loss += beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train[epoch, 0] = loss.item()
        print('Epoch: %04d, Training loss=%.8f' %
              (epoch + 1, loss.item()))

    encoded = encoded.cpu()
    res = encoded.detach().numpy()
    return res

def preprocess_small_data1(en_input1, en_output1, en_input2, en_output2, de_input1, de_output1, de_input2, de_output2,
                          epochs,
                          matfile, batch_size, lr):
    dcsae.en_input1 = en_input1
    dcsae.en_output1 = en_output1
    dcsae.en_input2 = en_input2
    dcsae.en_output2 = en_output2
    dcsae.de_input1 = de_input1
    dcsae.de_output1 = de_output1
    dcsae.de_input2 = de_input2
    dcsae.de_output2 = de_output2
    dcsae.epochs = epochs

    # the weight of the expected mean activation value and KL divergence
    expect_tho = 0.05
    tho_tensor = torch.FloatTensor([expect_tho for _ in range(en_output2)])
    if torch.cuda.is_available():
        tho_tensor = tho_tensor.cuda()
    beta = 1

    X_data = matfile
    X_data = torch.FloatTensor(X_data)

    X_dataset = TensorDataset(X_data, X_data)
    X_DataLoader = DataLoader(dataset=X_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        autoencoder = dcsae.DCSparseAutoEncoder().cuda()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        loss_func = nn.MSELoss().cuda()
        loss_train = np.zeros((dcsae.epochs, 1))
    else:
        autoencoder = dcsae.DCSparseAutoEncoder()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        loss_func = nn.MSELoss()
        loss_train = np.zeros((dcsae.epochs, 1))

    # train
    for epoch in range(dcsae.epochs):
        for batchidx, (x, _) in enumerate(X_DataLoader):
            if torch.cuda.is_available():
                x = x.cuda()

            encoded, decoded = autoencoder(x)
            loss = loss_func(decoded, x)
            kl = KL_divergence(tho_tensor, encoded)
            loss += beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train[epoch, 0] = loss.item()
        print('Epoch: %04d, Training loss=%.8f' %
              (epoch + 1, loss.item()))

    encoded = encoded.cpu()
    res = encoded.detach().numpy()
    return res


def KL_divergence(p, q):
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0) / len(q)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def preprocess_big_data(en_input1, en_output1, en_input2, en_output2, de_input1, de_output1, de_input2,
                        de_output2,
                        epochs,
                        samples, batch_size, lr):
    dcsae.en_input1 = en_input1
    dcsae.en_output1 = en_output1
    dcsae.en_input2 = en_input2
    dcsae.en_output2 = en_output2
    dcsae.de_input1 = de_input1
    dcsae.de_output1 = de_output1
    dcsae.de_input2 = de_input2
    dcsae.de_output2 = de_output2
    dcsae.epochs = epochs

    expect_tho = 0.05
    tho_tensor = torch.FloatTensor([expect_tho for _ in range(en_output2)])
    if torch.cuda.is_available():
        tho_tensor = tho_tensor.cuda()
    beta = 0

    X_train = samples
    trainData = torch.FloatTensor(X_train)
    train_dataset = TensorDataset(trainData, trainData)
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        autoencoder = dcsae.DCSparseAutoEncoder().cuda()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        loss_func = nn.MSELoss().cuda()
        loss_train = np.zeros((dcsae.epochs, 1))
    else:
        autoencoder = dcsae.DCSparseAutoEncoder()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        loss_func = nn.MSELoss()
        loss_train = np.zeros((dcsae.epochs, 1))

    res = []

    # train
    for epoch in range(dcsae.epochs):
        for batchidx, (x, _) in enumerate(trainDataLoader):
            if torch.cuda.is_available():
                x = x.cuda()

            encoded, decoded = autoencoder(x)
            loss = loss_func(decoded, x)

            kl = KL_divergence(tho_tensor, encoded)
            loss += beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == dcsae.epochs - 1:
                for tmp in range(trainDataLoader.batch_size):
                    encoded = encoded.cpu()
                    res.append(encoded.detach().numpy()[tmp])

        loss_train[epoch, 0] = loss.item()
        print('Epoch: %04d, Training loss=%.8f' %
              (epoch + 1, loss.item()))

    res = np.array(res)
    return res
