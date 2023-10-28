import numpy as np

import util as u
from torch import nn

import sys

from contextlib import contextmanager
from typing import Callable, Tuple

import torch
import torch.nn as nn

import numpy as np

import inspect
import math
import os
import random
import sys
import time
from typing import Any, Dict, Callable, Optional, Tuple, Union
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
# import wandb
from PIL import Image
from torch.utils import tensorboard

from contextlib import contextmanager
from typing import Callable, Tuple

import torch
import torch.nn as nn

import numpy as np

from torchvision import datasets, transforms

def numpy_kron(a, b):
    result = u.kron(u.from_numpy(a), u.from_numpy(b))
    return u.to_numpy(result)


def test_toy_multiclass_example():
    # Numbers taken from "Unit test for toy multiclass example" of
    # https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/linear-estimation-headers.nb
    #
    # Data gen "Exporting  toy multiclass example to numpy array" of
    # https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/linear-estimation.nb

    A = np.load('data/toy-A.npy')
    Y = np.load('data/toy-Y.npy')
    print(A)
    print(Y)

    numSteps = 5
    (m, n) = A.shape
    (m0, c) = Y.shape
    assert m0 == m

    assert (m, n) == (2, 2)
    W0 = np.zeros((n, c))

    def getLoss(W):
        return np.linalg.norm(A @ W - Y) ** 2 / (2 * m)

    def run(use_kac: bool, step=1.):
        W = W0
        losses = [getLoss(W)]

        for i in range(numSteps):
            idx = i % m
            a = A[idx:idx + 1, :]
            y = a @ W
            r = y - Y[idx:idx + 1]
            g = numpy_kron(a.T, r)
            factor = (a * a).sum() if use_kac else 1
            W = W - step * g / factor
            losses.extend([getLoss(W)])
        return losses

    losses_kac = run(True)
    losses_sgd = run(False)

    golden_losses_kac = [39 / 4., 13 / 4., 13 / 16., 13 / 16., 13 / 64., 13 / 64.]
    golden_losses_sgd = [39 / 4, 13 / 4, 13 / 2, 0, 0, 0]
    u.check_close(losses_kac, golden_losses_kac)
    u.check_close(losses_sgd, golden_losses_sgd)


def test_toy_multiclass_with_bias():
    # toy multiclass with bias: https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/linear-estimation.nb
    # toy multiclass with bias: https://colab.research.google.com/drive/1dGeCen7ikIXcWBbsTtrdQ7NKyJ-iHyUw#scrollTo=wHeqLIn3bcNl

    A = np.array([[1, 0, 1], [1, 1, 1]])
    Y = np.array([[1, 2], [3, 5]])

    print(A)
    print(Y)

    numSteps = 5
    (m, n) = A.shape
    (m0, c) = Y.shape

    W0 = np.zeros((n, c))

    def getLoss(W):
        return np.linalg.norm(A @ W - Y) ** 2 / (2 * m)

    def run(use_kac: bool, step=1.):
        W = W0
        losses = [getLoss(W)]

        for i in range(numSteps):
            idx = i % m
            a = A[idx:idx + 1, :]
            y = a @ W
            r = y - Y[idx:idx + 1]
            g = numpy_kron(a.T, r)
            factor = (a * a).sum() if use_kac else 1
            W = W - step * g / factor
            losses.extend([getLoss(W)])
        return losses

    losses_kac = run(True)
    losses_sgd = run(False)

    golden_losses_kac = [39 / 4, 13 / 4, 13 / 9, 13 / 9, 52 / 81, 52 / 81]
    golden_losses_sgd = [39 / 4, 7 / 4, 33 / 4, 77 / 4, 297 / 4, 109 / 4]

    print("Losses Kaczmarz: ", losses_kac[:5])
    print("Losses SGD: ", losses_sgd[:5])

    np.testing.assert_allclose(losses_kac, golden_losses_kac, rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(losses_sgd, golden_losses_sgd, rtol=1e-10, atol=1e-20)


def test_d10_example():
    A = np.load('data/d10-A.npy')
    Y = np.load('data/d10-Y.npy')

    (m, n) = A.shape
    (m0, c) = Y.shape
    assert m0 == m

    W0 = np.zeros((n, c))

    def getLoss(): return np.linalg.norm(A @ W - Y) ** 2 / (2 * m)

    W = W0
    losses = [getLoss()]

    golden_losses = [0.0460727, 0.048437, 0.0360559, 0.0472155, 0.046455, 0.0443264]
    numSteps = len(golden_losses) - 1
    for i in range(numSteps):
        idx = i % m
        a = A[idx:idx + 1, :]
        y = a @ W
        r = y - Y[idx:idx + 1]
        g = numpy_kron(a.T, r)
        W = W - g / (a * a).sum()
        losses.extend([getLoss()])

    u.check_close(golden_losses, losses)
    print(losses)


def test_d1000_example():
    A = np.load('data/d1000-A.npy')
    Y = np.load('data/d1000-Y.npy')

    (m, n) = A.shape
    (m0, c) = Y.shape
    assert m0 == m

    W0 = np.zeros((n, c))

    def getLoss(W): return np.linalg.norm(A @ W - Y) ** 2 / (2 * m)

    golden_losses_kac = [0.00533897, 0.00522692, 0.00513617, 0.00494244, 0.00460737, 0.00421404, 0.00406255, 0.00393202, 0.0039428,
                         0.00394316]
    golden_losses_sgd = [0.00533897, 0.00524321, 0.0051401, 0.00494911, 0.00469713, 0.00431367, 0.00419808, 0.00408195, 0.00408793,
                         0.00406008]
    numSteps = len(golden_losses_kac) - 1
    assert len(golden_losses_kac) == len(golden_losses_sgd)

    def run(use_kac):
        W = W0
        losses = [getLoss(W)]

        for i in range(numSteps):
            idx = i % m
            a = A[idx:idx + 1, :]
            y = a @ W
            r = y - Y[idx:idx + 1]
            g = numpy_kron(a.T, r)
            factor = (a * a).sum() if use_kac else 1
            W = W - g / factor
            losses.extend([getLoss(W)])
        return losses

    kac_losses = run(True)
    sgd_losses = run(False)

    u.check_close(golden_losses_kac, kac_losses)
    u.check_close(golden_losses_sgd, sgd_losses)


def test_d1000c_example():
    A = np.load('data/d1000c-A.npy')
    Y = np.load('data/d1000c-Y.npy')

    (m, n) = A.shape
    (m0, c) = Y.shape
    assert m0 == m

    W0 = np.zeros((n, c))

    def getLoss(W): return np.linalg.norm(A @ W - Y) ** 2 / (2 * m)

    golden_losses_kac = [8202.34, 8202.99, 8164.45, 8165.31, 8166.43, 8128.15]
    golden_losses_sgd = [8202.34, 3.05277e7, 1.89181e8, 1.91024e8, 2.34391e8, 2.36751e11]
    numSteps = len(golden_losses_kac) - 1
    assert len(golden_losses_kac) == len(golden_losses_sgd)

    def run(use_kac):
        W = W0
        losses = [getLoss(W)]

        for i in range(numSteps):
            idx = i % m
            a = A[idx:idx + 1, :]
            y = a @ W
            r = y - Y[idx:idx + 1]
            g = numpy_kron(a.T, r)
            factor = (a * a).sum() if use_kac else 1
            W = W - g / factor
            losses.extend([getLoss(W)])
        return losses

    kac_losses = run(True)
    sgd_losses = run(False)

    u.check_close(golden_losses_kac, kac_losses)
    u.check_close(golden_losses_sgd, sgd_losses)


import torch


def test_toy_multiclass_pytorch():
    """Test toy multiclass example using PyTorch API"""

    # debugged in kaczmarz-scratch: https://colab.research.google.com/drive/1dGeCen7ikIXcWBbsTtrdQ7NKyJ-iHyUw#scrollTo=W64pkgEvSg01
    dataset = u.ToyDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    train_iter = u.infinite_iter(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    def getLoss(model):
        losses = []
        for data, targets in test_loader:
            output = model(data)
            losses.append(loss_fn(output, targets).item())
        return np.mean(losses)

    model = u.SimpleFullyConnected([2, 2])
    loss_fn = u.least_squares_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0)

    num_steps = 5

    losses = [getLoss(model)]
    for step in range(num_steps):
        optimizer.zero_grad()
        model.zero_grad()

        data, targets = next(train_iter)
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        losses.append(getLoss(model))

    u.check_equal([39 / 4, 13 / 4, 13 / 2, 0, 0, 0], losses)


def test_d1000_pytorch():
    A = np.load('data/d1000-A.npy')
    Y = np.load('data/d1000-Y.npy')
    (m, n) = A.shape
    (m0, c) = Y.shape
    assert m0 == m

    dataset = u.NumpyDataset(A, Y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    train_iter = u.infinite_iter(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    loss_fn = u.least_squares_loss

    def getLoss(model):
        losses = []
        for data, targets in test_loader:
            output = model(data)
            losses.append(loss_fn(output, targets).item())
        return np.mean(losses)

    golden_losses_kac = [0.00533897, 0.00522692, 0.00513617, 0.00494244, 0.00460737, 0.00421404, 0.00406255, 0.00393202, 0.0039428,
                         0.00394316]
    golden_losses_sgd = [0.00533897, 0.00524321, 0.0051401, 0.00494911, 0.00469713, 0.00431367, 0.00419808, 0.00408195, 0.00408793,
                         0.00406008]
    num_steps = len(golden_losses_kac) - 1
    assert len(golden_losses_kac) == len(golden_losses_sgd)

    def run(use_kac):
        model = u.SimpleFullyConnected([n, c])
        if use_kac:
            assert False, "Kaczmarz not implemented"
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0)

        losses = [getLoss(model)]

        for step in range(num_steps):
            optimizer.zero_grad()
            model.zero_grad()

            data, targets = next(train_iter)
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            losses.append(getLoss(model))
        return losses

    # kac_losses = run(True)
    sgd_losses = run(False)

    # u.check_close(golden_losses_kac, kac_losses)
    u.check_close(golden_losses_sgd, sgd_losses)


def test_manual_optimizer():
    """Use SGD optimizer, but substitute manual gradient computation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = u.ToyDataset()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = u.least_squares_loss

    def getLoss(model):
        losses = []
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            losses.append(loss_fn(output, targets).item())
        return np.mean(losses)

    # forward hook with self-removing backward hook
    handles = []
    def manual_grad_linear(layer: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
        # skip over all non-leaf modules, like top-level nn.Sequential
        if not u.is_leaf_module(layer):
            return

        # which one to use?
        assert u._layer_type(layer) == 'Linear', f"Only linear layers supported but got {u._layer_type(layer)}"
        assert layer.__class__ == nn.Linear

        assert len(inputs) == 1, "multi-input layer??"
        A = inputs[0].detach()

        has_bias = hasattr(layer, 'bias') and layer.bias is not None
        idx = len(handles)  # idx is used for closure trick to autoremove hook

        def tensor_backwards(B):
            # use notation of "Kaczmarz step-size"/Multiclass Layout
            # https://notability.com/n/2TQJ3NYAK7If1~xRfL26Ap
            (m, n) = A.shape
            ones = torch.ones((m)).to(device)
            update = torch.einsum('mn,mc,m->nc', A, B, ones)
            layer.weight.manual_grad = update.T  # B.T @ A

            if has_bias:
                # B is (m, c) residual matrix
                update = torch.einsum('mc,m->c', B, ones)
                layer.bias.manual_grad = update.T / m  # B.T.sum(axis=1)

            handles[idx].remove()

        handles.append(output.register_hook(tensor_backwards))

    def optimize(bias):
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        train_iter = u.infinite_iter(train_loader)

        d = 2
        layer = nn.Linear(d, d, bias=bias)
        layer.weight.data.copy_(0 * torch.eye(d))
        if bias:
            layer.bias.data.zero_()
        model = torch.nn.Sequential(layer).to(device)
        print(f'step {-1}: test loss = {getLoss(model)}')

        optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0)
        num_steps = 5
        print(f'step {-1}: test loss = {getLoss(model)}')

        losses = [getLoss(model)]
        for step in range(num_steps):
            optimizer.zero_grad()
            data, targets = next(train_iter)
            data = data.to(device)
            targets = targets.to(device)

            with u.module_hook(manual_grad_linear):
                output = model(data)

            loss = loss_fn(output, targets)
            loss.backward()

            # TODO(y): update bias as well
            model[0].weight.grad = model[0].weight.manual_grad
            optimizer.step()
            print(f'step {step}: test loss = {getLoss(model)}')
            losses.append(getLoss(model))
        return losses

    losses_nobias = optimize(bias=False)
    u.check_equal([39 / 4, 13 / 4, 13 / 2, 0, 0, 0], losses_nobias)

    losses_bias = optimize(bias=True)
    u.check_equal([39 / 4, 7 / 4, 33 / 4, 77 / 4, 297 / 4, 109 / 4], losses_bias)


def test_kaczmarz_optimizer():
    """Use SGD optimizer, but substitute manual gradient computation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = u.ToyDataset()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = u.least_squares_loss

    def getLoss(model):
        losses = []
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            losses.append(loss_fn(output, targets).item())
        return np.mean(losses)

    # forward hook with self-removing backward hook
    handles = []
    def kaczmarz_grad_linear(layer: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
        # skip over all non-leaf modules, like top-level nn.Sequential
        if not u.is_leaf_module(layer):
            return

        # which one to use?
        assert u._layer_type(layer) == 'Linear', f"Only linear layers supported but got {u._layer_type(layer)}"
        assert layer.__class__ == nn.Linear

        assert len(inputs) == 1, "multi-input layer??"
        A = inputs[0].detach()

        has_bias = hasattr(layer, 'bias') and layer.bias is not None
        idx = len(handles)  # idx is used for closure trick to autoremove hook

        def tensor_backwards(B):
            # use notation of "Kaczmarz step-size"/Multiclass Layout
            # https://notability.com/n/2TQJ3NYAK7If1~xRfL26Ap
            (m, n) = A.shape

            norms2 = (A * A).sum(axis=1)
            if has_bias:
                norms2 += 1

            ones = torch.ones((m,)).to(device)
            update = torch.einsum('mn,mc,m->nc', A, B, ones/norms2)
            layer.weight.kaczmarz_grad = update.T  # B.T @ A

            if has_bias:
                # B is (m, c) residual matrix

                update = torch.einsum('mc,m->c', B, ones/norms2)
                layer.bias.kaczmarz_grad = update.T / m  # B.T.sum(axis=1)

            handles[idx].remove()

        handles.append(output.register_hook(tensor_backwards))

    def optimize(bias):
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        train_iter = u.infinite_iter(train_loader)

        d = 2
        layer = nn.Linear(d, d, bias=bias)
        layer.weight.data.copy_(0 * torch.eye(d))
        if bias:
            layer.bias.data.zero_()
        model = torch.nn.Sequential(layer).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0)
        num_steps = 5

        losses = [getLoss(model)]
        for step in range(num_steps):
            optimizer.zero_grad()
            data, targets = next(train_iter)
            data = data.to(device)
            targets = targets.to(device)

            with u.module_hook(kaczmarz_grad_linear):
                output = model(data)

            loss = loss_fn(output, targets)
            loss.backward()

            model[0].weight.grad = model[0].weight.kaczmarz_grad
            if bias:
                model[0].bias.grad = model[0].bias.kaczmarz_grad

            optimizer.step()
            losses.append(getLoss(model))
        return losses

    losses_nobias = optimize(bias=False)
    u.check_equal( [39/4., 13/4., 13/16., 13/16., 13/64., 13/64.], losses_nobias)

    losses_bias = optimize(bias=True)
    u.check_close([39/4, 13/4, 13/9, 13/9, 52/81, 52/81], losses_bias)

def test_linear_mnist(bias=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, download=True, transform=transform)

    model = u.SimpleFullyConnected([28**2, 10], bias=bias)
    loss_fn = u.least_squares_loss

    def getLoss(dataset, max_eval_samples = 10):
        test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False)
        losses = []
        for (i, (data, targets)) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            losses.append(loss_fn(output, targets).item())
            if i >= max_eval_samples:
                break
        return np.mean(losses)

    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0)
    num_steps = 15

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False)
    train_iter = u.infinite_iter(train_loader)
    losses = [getLoss(model)]

    # forward hook with self-removing backward hook
    handles = []
    def kaczmarz_grad_linear(layer: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
        # skip over all non-leaf modules, like top-level nn.Sequential
        if not u.is_leaf_module(layer):
            return

        # which one to use?
        assert u._layer_type(layer) == 'Linear', f"Only linear layers supported but got {u._layer_type(layer)}"
        assert layer.__class__ == nn.Linear

        assert len(inputs) == 1, "multi-input layer??"
        A = inputs[0].detach()

        has_bias = hasattr(layer, 'bias') and layer.bias is not None
        idx = len(handles)  # idx is used for closure trick to autoremove hook

        def tensor_backwards(B):
            # use notation of "Kaczmarz step-size"/Multiclass Layout
            # https://notability.com/n/2TQJ3NYAK7If1~xRfL26Ap
            (m, n) = A.shape

            norms2 = (A * A).sum(axis=1)
            if has_bias:
                norms2 += 1

            ones = torch.ones((m,)).to(device)
            update = torch.einsum('mn,mc,m->nc', A, B, ones/norms2)
            layer.weight.kaczmarz_grad = update.T  # B.T @ A

            if has_bias:
                # B is (m, c) residual matrix

                update = torch.einsum('mc,m->c', B, ones/norms2)
                layer.bias.kaczmarz_grad = update.T / m  # B.T.sum(axis=1)

            handles[idx].remove()

        handles.append(output.register_hook(tensor_backwards))

    for step in range(num_steps):
        optimizer.zero_grad()
        data, targets = next(train_iter)
        data = data.to(device)
        targets = targets.to(device)

        with u.module_hook(kaczmarz_grad_linear):
            output = model(data)

        loss = loss_fn(output, targets)
        loss.backward()

        model.layers[0].weight.grad = model.layers[0].weight.kaczmarz_grad
        if bias:
            model.layers[0].bias.grad = model.layers[0].bias.kaczmarz_grad

        optimizer.step()
        losses.append(getLoss(model))

    # [124.54545454545455, 83.3533919971775, 94.94022894752297, 71.47339670631018, 70.7996980888261, 29.232307191255543, 35.90959778157148, 46.41351659731431, 54.23867620820899, 55.18098633710972, 44.854006835682824, 39.05859527533705, 34.10272614522414, 33.8443342582746, 35.933635773983866, 36.40805794434114]
    golden_losses = [124.54545454545455, 83.3533919971775, 94.94022894752297, 71.47339670631018, 70.7996980888261, 29.232307191255543, 35.90959778157148, 46.41351659731431, 54.23867620820899, 55.18098633710972, 44.854006835682824, 39.05859527533705, 34.10272614522414, 33.8443342582746, 35.933635773983866, 36.40805794434114]
    np.testing.assert_allclose(losses, golden_losses)

if __name__ == '__main__':
    # test_d10_example()
    test_d1000c_example()
    #    u.run_all_tests(sys.modules[__name__])
