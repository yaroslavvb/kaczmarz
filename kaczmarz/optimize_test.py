from typing import Tuple

import numpy as np
import pytest
from torch.utils import tensorboard

import kaczmarz_util

u = kaczmarz_util

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def numpy_kron(a, b):
    result = u.kron(u.from_numpy(a), u.from_numpy(b))
    return u.to_numpy(result)


# for line profiling
try:
    # noinspection PyUnboundLocalVariable
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.


def test_toy_multiclass_example():
    # Numbers taken from "Unit test for toy multiclass example" of
    # https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/linear-estimation-headers.nb
    #
    # Data gen "Exporting  toy multiclass example to numpy array" of
    # https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/linear-estimation.nb

    A = np.load(root + '/data/toy-A.npy')
    Y = np.load(root + '/data/toy-Y.npy')

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

    np.testing.assert_allclose(losses_kac, golden_losses_kac, rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(losses_sgd, golden_losses_sgd, rtol=1e-10, atol=1e-20)


def test_d10_example():
    A = np.load(root + '/data/d10-A.npy')
    Y = np.load(root + '/data/d10-Y.npy')

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


def test_d1000_example():
    A = np.load(root + '/data/d1000-A.npy')
    Y = np.load(root + '/data/d1000-Y.npy')

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
    A = np.load(root + '/data/d1000c-A.npy')
    Y = np.load(root + '/data/d1000c-Y.npy')

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
    loss_fn = u.least_squares_loss_add_classes
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
    A = np.load(root + '/data/d1000-A.npy')
    Y = np.load(root + '/data/d1000-Y.npy')
    (m, n) = A.shape
    (m0, c) = Y.shape
    assert m0 == m

    dataset = u.NumpyDataset(A, Y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    train_iter = u.infinite_iter(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    loss_fn = u.least_squares_loss_add_classes

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
    loss_fn = u.least_squares_loss_add_classes

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
            ones = torch.ones((m,)).to(device)
            update = torch.einsum('mn,mc,m->nc', A, B, ones)
            layer.weight.manual_grad = update.T  # B.T @ A

            if has_bias:
                # B is (m, c) residual matrix
                update = torch.einsum('mc,m->c', B, ones)
                layer.bias.manual_grad = update.unsqueeze(0).T / m  # B.T.sum(axis=1)

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

            with u.module_hook(manual_grad_linear):
                output = model(data)

            loss = loss_fn(output, targets)
            loss.backward()

            # TODO(y): update bias as well
            model[0].weight.grad = model[0].weight.manual_grad
            optimizer.step()
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
    loss_fn = u.least_squares_loss_add_classes

    def getLoss(model):
        losses = []
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            losses.append(loss_fn(output, targets).item())
        return np.mean(losses)

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

            with u.module_hook(u.kaczmarz_grad_linear):
                output = model(data)

            loss = loss_fn(output, targets)
            loss.backward()

            model[0].weight.grad = model[0].weight.custom_grad
            if bias:
                model[0].bias.grad = model[0].bias.custom_grad

            optimizer.step()
            losses.append(getLoss(model))
        return losses

    losses_nobias = optimize(bias=False)
    u.check_equal([39 / 4., 13 / 4., 13 / 16., 13 / 16., 13 / 64., 13 / 64.], losses_nobias)

    losses_bias = optimize(bias=True)
    u.check_close([39 / 4, 13 / 4, 13 / 9, 13 / 9, 52 / 81, 52 / 81], losses_bias)


def test_linear_mnist(bias=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, download=True, transform=transform)

    model = u.SimpleFullyConnected([28 ** 2, 10], bias=bias)
    loss_fn = u.combined_nll_loss

    def getLoss(model, max_eval_samples=10):
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

    for step in range(num_steps):
        model.zero_grad()
        u.zero_custom_grad(model)
        data, targets = next(train_iter)
        data = data.to(device)
        targets = targets.to(device)

        with u.module_hook(u.kaczmarz_grad_linear):
            output = model(data)

        loss = loss_fn(output, targets)
        loss.backward()
        u.copy_custom_grad_to_grad(model)

        optimizer.step()
        losses.append(getLoss(model))

    # [124.54545454545455, 83.3533919971775, 94.94022894752297, 71.47339670631018, 70.7996980888261, 29.232307191255543, 35.90959778157148, 46.41351659731431, 54.23867620820899, 55.18098633710972, 44.854006835682824, 39.05859527533705, 34.10272614522414, 33.8443342582746, 35.933635773983866, 36.40805794434114]
    # golden_losses = [124.54545454545455, 83.3533919971775, 94.94022894752297, 71.47339670631018, 70.7996980888261, 29.232307191255543,
    #                 35.90959778157148, 46.41351659731431, 54.23867620820899, 55.18098633710972, 44.854006835682824, 39.05859527533705,
    #                 34.10272614522414, 33.8443342582746, 35.933635773983866, 36.40805794434114]
    golden_losses = [2.302585, 2.295276, 2.204854, 2.163034, 2.097804, 2.037777, 2.050594, 2.029167, 2.055976, 2.030072, 1.998083, 2.033249,
                     2.042463, 2.067483, 2.089148, 2.056059]

    np.testing.assert_allclose(losses, golden_losses, rtol=1e-6, atol=1e-6)


# End-to-end testing of toy mnist dataset
def test_toy_mnist():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)

    dataset_size = 10
    train_kwargs = {'batch_size': dataset_size, 'num_workers': 0, 'shuffle': False}
    test_kwargs = dict(train_kwargs)
    if device == 'cuda':
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = u.CustomMNIST(data_width=28, dataset_size=dataset_size, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    test_dataset = u.CustomMNIST(data_width=28, dataset_size=dataset_size, train=False)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 10):
        model.eval()
        test_loss, correct = 0, 0

        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

    assert correct == 5


def test_toy_mnist_sqloss():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0)
            self.dropout2 = nn.Dropout(0)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)

    dataset_size = 10
    num_epochs = 10
    train_kwargs = {'batch_size': 100, 'num_workers': 0, 'shuffle': False}
    test_kwargs = {'batch_size': 1000}

    do_squared_loss = False
    do_squared_loss = True
    loss_fn = u.least_squares_loss if do_squared_loss else u.combined_nll_loss
    loss_type = 'LeastSquares' if do_squared_loss else 'CrossEntropy'

    train_dataset = u.CustomMNIST(data_width=28, dataset_size=dataset_size, train=True, loss_type=loss_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    test_dataset = u.CustomMNIST(data_width=28, dataset_size=dataset_size, train=False, loss_type=loss_type)
    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0)

    for epoch in range(1, num_epochs + 1):
        model.eval()
        test_loss, correct = 0, 0

        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
                if do_squared_loss:
                    actual = target.argmax(dim=1, keepdim=True)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(actual).sum().item()
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            output = model(data)
            # new_loss = F.nll_loss(output, target)
            new_loss = loss_fn(output, target)
        # print(f"correct: {correct}, test_loss: {test_loss:.2f}, old train loss: {loss.item():.2f}, new train loss: {new_loss.item():.2f}")


def test_prewhitened_mnist():
    A = np.load(root + '/data/mnistTrainW.npy')
    norms2 = (A * A).sum(axis=1)
    means = A.sum(axis=1)

    # mean 1, median 0.62    print(f"Average norm squared: {norms2.mean()}, median: {np.quantile(norms2,.5)}")
    np.testing.assert_allclose(norms2.mean(), 1)
    np.testing.assert_allclose((means * means).sum(), 0, atol=1e-6)


@pytest.mark.slow
def test_whitening():
    """Following values from "Whiten MNIST" of linear-estimation.nb, and tolga-mnist-whitening"""

    dataset = u.CustomMNIST()
    # loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    # X, Y = next(iter(loader))
    X, Y = dataset.data, dataset.targets

    X = X.double().reshape(-1, 28 * 28)

    cov = u.getCov(X).cpu().numpy()
    np.testing.assert_allclose(u.getIntrinsicDim(cov), 5.044191669053714)

    whiten = kaczmarz_util.isymsqrtStable(cov)
    X2 = X @ whiten
    cov = kaczmarz_util.getCov(X2)
    np.testing.assert_allclose(u.getIntrinsicDim(cov), 712)

    # after centering each example
    X = X - X.mean(dim=1, keepdim=True)
    cov = kaczmarz_util.getCov(X).cpu().numpy()
    np.testing.assert_allclose(kaczmarz_util.getIntrinsicDim(cov), 8.43133453309047)

    whiten = kaczmarz_util.isymsqrtStable(cov)
    cov = u.getCov(X @ whiten)
    np.testing.assert_allclose(kaczmarz_util.getIntrinsicDim(cov), 712)


@pytest.mark.slow
def test_whitened_mnist():
    """Sanity check for whitened MNIST, see https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/whitened-mnist.nb"""
    dataset = u.CustomMNIST(train=True, whiten_and_center=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    #    X, Y = next(iter(loader))
    X, Y = dataset.data, dataset.targets

    X = X.double().reshape(-1, 28 * 28)

    assert (u.getIntrinsicDim(u.getCov(X)) > 711)  # actual rank is slightly smaller than 712, maybe due to float32 downcast

    # average norm squared is 1
    np.testing.assert_allclose(np.trace(u.getCov(X)), 1, atol=1e-3, rtol=1e-3)

    # full batch convergence happens in 1 step using learning rate 712
    (m, n) = X.shape
    w0 = kaczmarz_util.v2c(torch.zeros(n, dtype=torch.double))
    yopt = kaczmarz_util.v2c(torch.ones(m))

    lr = 712

    w = w0
    y = X @ w
    r = y - yopt
    loss = 0.5 * kaczmarz_util.norm2(r) / m
    np.testing.assert_allclose(loss, 0.5)

    g = X.T @ r / m
    w = w - lr * g
    y = X @ w
    r = y - yopt
    loss = 0.5 * kaczmarz_util.norm2(r) / m
    np.testing.assert_allclose(loss, 0.0265, atol=1e-5, rtol=1e-5)

    # check the test dataset
    dataset = kaczmarz_util.CustomMNIST(train=False, whiten_and_center=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    X, Y = next(iter(loader))
    X = X.double().reshape(-1, 28 * 28)
    assert (int(kaczmarz_util.getIntrinsicDim(kaczmarz_util.getCov(X))) == 58)

def test_mnist_numpy_optimize():
    dataset_size = 10
    do_squared_loss = True
    loss_fn = u.least_squares_loss if do_squared_loss else u.combined_nll_loss
    loss_type = 'LeastSquares' if do_squared_loss else 'CrossEntropy'

    train_dataset = u.CustomMNIST(dataset_size=dataset_size, train=True, whiten_and_center=True, loss_type=loss_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    train_loader_eval = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_size, shuffle=False)

    test_dataset = u.CustomMNIST(dataset_size=dataset_size, train=False, whiten_and_center=True, loss_type=loss_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=dataset_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = u.SimpleFullyConnected([28 ** 2, 10])

    # largest convergent LR for MNIST, see https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/whitened-mnist.nb
    # max_lr = 0.4867711
    # lr = max_lr / 2

    lr = 1000
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)

    pytorch_losses = []

    def norm2(p): return (p*p).sum()

    for epoch in range(1, 2):
        test_loss, test_acc = kaczmarz_util.evaluate_mnist(test_loader, model, do_squared_loss)
        train_loss, train_acc = kaczmarz_util.evaluate_mnist(train_loader_eval, model, do_squared_loss)
        # print(f"accuracy: {train_acc:0.2f}/{test_acc:0.2f}, loss: {train_loss:.2f}/{test_loss:.2f}")

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            pytorch_losses.extend([loss.item()])
            loss.backward()
            optimizer.step()
            loss = loss_fn(output, target)

    # print("Pytorch losses: ", pytorch_losses)
    # print("πyTorch gradnorms: ", pytorch_gradnorms)

    # run the training using numpy
    model = kaczmarz_util.SimpleFullyConnected([28 ** 2, 10])

    # using "Multiclass layout", classes is the second dimension"
    # https://notability.com/n/2TQJ3NYAK7If1~xRfL26Ap
    W = model.layers[0].weight.data.T

    X, Y = train_dataset.data, train_dataset.targets
    X = X.reshape(-1, 28*28)
    m = X.shape[0]

    numpy_losses = []
    numpy_gradnorms = []
    bs = 1
    c = 10  # number of classes
    for i in range(dataset_size):
        idx = i % m
        a = X[idx:idx + bs, :]
        y = a @ W
        r = y - Y[idx:idx + bs]
        loss = 0.5 * (r**2).sum()/(bs * c)
        g = a.T @ r / (bs * c)
        W = W - lr * g
        numpy_losses.extend([loss.item()])

    # print("Numpy losses: ", numpy_losses)
    # print("Numpy gradnorms: ", numpy_gradnorms)
    np.testing.assert_allclose(pytorch_losses, numpy_losses, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    # test_d10_example()
    # test_d1000c_example()
    # test_mnist_numpy_optimize()
    #  test_whitening()
    test_whitened_mnist()
    #    u.run_all_tests(sys.modules[__name__])
