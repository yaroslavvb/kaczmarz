import numpy as np

import util as u

import sys


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

    def run(use_kac: bool, step = 1.):
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
    golden_losses_sgd = [39/4, 13/4, 13/2, 0, 0, 0]
    u.check_close(losses_kac, golden_losses_kac)
    u.check_close(losses_sgd, golden_losses_sgd)


def test_toy_multiclass_bias():
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

    def run(use_kac: bool, step = 1.):
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

    u.check_equal([39/4, 13/4, 13/2, 0, 0, 0], losses)

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
            assert False, "Kacmzarz not implemented"
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


if __name__ == '__main__':
    # test_d10_example()
    test_d1000c_example()
    #    u.run_all_tests(sys.modules[__name__])
