import math
import os
import sys

# import torch
import pytest
import scipy
from scipy import linalg
import torch

import numpy as np
import kac as u


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
# import wandb
from PIL import Image
from torch.utils import tensorboard

import torch.nn.functional as F

import inspect
import time

import kac
u = kac

def test_kron():
    """Test kron, vec and vecr identities"""
    a = torch.tensor([1, 2, 3, 4]).reshape(2, 2)
    b = torch.tensor([5, 6, 7, 8]).reshape(2, 2)
    u.check_close(u.kron(a, b).diag().sum(), 65)

    a = torch.tensor([[2., 7, 9], [1, 9, 8], [2, 7, 5]])
    b = torch.tensor([[6., 6, 1], [10, 7, 7], [7, 10, 10]])
    Ck = u.kron(a, b)
    u.check_close(a.flatten().norm() * b.flatten().norm(), Ck.flatten().norm())

    u.check_close(Ck.norm(), 4 * math.sqrt(11635.))

def test_checkequal():
    with pytest.raises(AssertionError):
        u.check_equal([1], [[1]])

    u.check_equal(1, 1)
    u.check_equal(1, 1.)

def test_toy_dataset():
    dataset = u.ToyDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    train_iter = u.infinite_iter(train_loader)

    data, targets = None, None
    for i in range(4):
        data, targets = next(train_iter)
    u.check_equal(data, u.from_numpy([[1., 1.]]))
    u.check_equal(targets, u.from_numpy([[3., 5.]]))

def test_least_squares_loss():
    y = torch.Tensor(u.to_numpy([[0, 0], [0, 0]]))
    y0 = torch.Tensor(u.to_numpy([[1, 2], [3, 5]]))
    u.check_equal(u.least_squares_loss(y, y0), 39 / 4)

def test_simple_fully_connected():
    net = u.SimpleFullyConnected([28**2, 2, 10], nonlin=True, bias=True, init_scale=1)

    # two linear layers, two relu layers, itself
    assert len(list(net.named_modules())) == 5

    # just layers with parameters
    assert len(net.layers) == 2

    # all layers
    assert len(net.all_layers) == 4

    # dimensions
    assert net.d == [784, 2, 10]

    image = torch.ones((28, 28))
    assert net(image).shape == (1, 10)
    assert torch.allclose(net(image), torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]))

def test_tiny_mnist():
    dataset1 = u.TinyMNIST(train=True, )
    dataset2 = u.TinyMNIST(train=False)

    dataset1 = u.TinyMNIST(data_width=28, dataset_size=100, train=True, loss_type='CrossEntropy')
    dataset2 = u.TinyMNIST(data_width=28, dataset_size=100, train=False, loss_type='CrossEntropy')

    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False)
    data1, targets1 = next(iter(loader1))
    data2, targets2 = next(iter(loader2))

    assert data1.shape == (1, 1, 28, 28)   # B x C x H x W
    assert torch.allclose(targets1, torch.tensor([5], dtype=torch.int64))
    assert torch.allclose(targets2, torch.tensor([7], dtype=torch.int64))

    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=2, shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=3, shuffle=False)
    data1, _unused_targets1 = next(iter(loader1))
    data2, _unused_targets2 = next(iter(loader2))
    assert data1.shape == (2, 1, 28, 28)   # B x C x H x W
    assert data2.shape == (3, 1, 28, 28)   # B x C x H x W

    dataset1 = u.TinyMNIST(train=True, loss_type='LeastSquares')
    dataset2 = u.TinyMNIST(train=False, loss_type='LeastSquares')

    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False)
    data1, targets1 = next(iter(loader1))
    data2, targets2 = next(iter(loader2))

    assert torch.allclose(targets1, torch.tensor([[0., 0, 0, 0, 0, 1, 0, 0, 0, 0]]))
    assert torch.allclose(targets2, torch.tensor([[0., 0, 0, 0, 0, 0, 0, 1, 0, 0]]))


def test_isymsqrtStable():
    X = np.array([[1, 2], [3, 4]])
    cov = X.T @ X
    whiten = kac.isymsqrtStable(cov)

    Y = X @ whiten
    covW = Y.T @ Y
    np.testing.assert_allclose(covW, torch.eye(2), atol=1e-10)


def test_mnist_whitening():
    """Following values from "Whiten MNIST" of linear-estimation.nb, and tolga-mnist-whitening"""

    dataset = kac.TinyMNIST()
    loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    X, Y = next(iter(loader))
    X = X.double().reshape(-1, 28 * 28)

    cov = kac.getCov(X).cpu().numpy()
    np.testing.assert_allclose(kac.effRank(cov), 5.044191669053714)

    whiten = kac.isymsqrtStable(cov)
    cov = kac.getCov(X @ whiten)
    print(kac.effRank(cov))
    np.testing.assert_allclose(kac.effRank(cov), 712)

    # after centering each example
    X = X - X.mean(dim=1, keepdim=True)
    cov = kac.getCov(X).cpu().numpy()
    np.testing.assert_allclose(kac.effRank(cov),  8.43133453309047)

    whiten = kac.isymsqrtStable(cov)
    cov = kac.getCov(X @ whiten)
    print(kac.effRank(cov))
    np.testing.assert_allclose(kac.effRank(cov), 712)


def test_tiny_mnist_whiten():
    train = kac.TinyMNIST(whiten_and_center=True)

    dataset = kac.TinyMNIST(train=True, whiten_and_center=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    X, Y = next(iter(loader))
    X = X.double().reshape(-1, 28 * 28)
    assert(kac.effRank(kac.getCov(X)) > 711)   # actual rank is slightly smaller due to float32 downcast

    assert 60000 <= np.trace(X.T @ X) <= 60040

    dataset = kac.TinyMNIST(train=False, whiten_and_center=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    X, Y = next(iter(loader))
    X = X.double().reshape(-1, 28 * 28)
    assert(int(kac.effRank(kac.getCov(X))) == 33)


if __name__ == '__main__':
    # test_kron()
    u.run_all_tests(sys.modules[__name__])
