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

    def getLoss():
        return np.linalg.norm(A@W-Y)**2 / (2*m)
    W = W0
    losses = [getLoss()]

    for i in range(numSteps):
        idx = i % m
        a = A[idx:idx+1, :]
        y = a @ W
        r = y - Y[idx:idx+1]
        g = numpy_kron(a.T, r)
        W = W - g / (a*a).sum()
        losses.extend([getLoss()])

    golden_losses = [39/4., 13/4., 13/16., 13/16., 13/64., 13/64.]
    u.check_close(losses, golden_losses)


if __name__ == '__main__':
    u.run_all_tests(sys.modules[__name__])
