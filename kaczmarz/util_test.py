import math
import os
import sys

# import torch
import pytest
import scipy
from scipy import linalg
import torch

import numpy as np
import util as u


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

import util as u

def test_kron():
    """Test kron, vec and vecr identities"""
    torch.set_default_dtype(torch.float64)
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
    u.check_equal(u.least_squares_loss(y, y0), 39/4)

if __name__ == '__main__':
    # test_kron()
    u.run_all_tests(sys.modules[__name__])
