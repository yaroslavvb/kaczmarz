import os

import numpy as np
import torch

import kaczmarz.kaczmarz_util as kac

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def one_hot_decode(Y):
    newY = torch.zeros((Y.shape[0], 10))
    newY.scatter_(1, Y.unsqueeze(1), 1)
    return newY

def main():
    dataset = kac.CustomMNIST(train=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    X, Y = next(iter(loader))
    Y = one_hot_decode(Y)
    X0, Y0 = kac.to_numpy(X), kac.to_numpy(Y)

    np.save(root + '/data/mnistTrain.npy', X0)
    np.save(root + '/data/mnistTrain-labels.npy', Y0)

    dataset = kac.CustomMNIST(train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    X, Y = next(iter(loader))
    Y = one_hot_decode(Y)
    X0, Y0 = kac.to_numpy(X), kac.to_numpy(Y)

    np.save(root + '/data/mnistTest.npy', X0)
    np.save(root + '/data/mnistTest-labels.npy', Y0)

if __name__ == '__main__':
    main()


