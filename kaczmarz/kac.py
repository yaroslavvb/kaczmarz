import gc
import inspect
import math
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
# import wandb
from PIL import Image

import scipy # for gesvd

mnistTrainWhiteningMatrix = None

# to enable referring to functions in its own module as u.func
# u = sys.modules[__name__]


global_timeit_dict = {}

# for line profiling
try:
    # noinspection PyUnboundLocalVariable
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.


# Loads file from Github
# import urllib
# from io import BytesIO

# def githubLoad(fn):
#    srcRoot = 'https://github.com/yaroslavvb/kaczmarz/raw/main/kaczmarz'
#    url = srcRoot + '/' + fn
#
#    with urllib.request.urlopen(url) as response:
#        buffer = BytesIO(response.read())
#        arr = np.load(buffer, allow_pickle=True)
#    return arr


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
    it to TensorBoard. This function is
    """

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        global last_time
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        torch.cuda.synchronize()
        print(f"{interval_ms:8.2f}   {self.tag}")
        last_time = interval_ms


@contextmanager
def measure_memory(label):
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.memory_allocated()
    yield
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    memory_after = torch.cuda.memory_allocated()
    memory_delta = memory_after - memory_before
    memory_peak = torch.cuda.max_memory_allocated()
    print(f"{label:20s}: {memory_before / 1e6:8.0f} {memory_after / 1e6:8.0f} {memory_delta / 1e6:8.0f} {memory_peak / 1e6:8.0f}")


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
    it to TensorBoard."""

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        print(f"{interval_ms:8.2f}   {self.tag}")
        # log_scalars({'time/' + self.tag: interval_ms})

        # global last_time
        # self.end = time.perf_counter()
        # interval_ms = 1000 * (self.end - self.start)
        # torch.cuda.synchronize()
        # print(f"{interval_ms:8.2f}   {self.tag}")
        # last_time = interval_ms


_pytorch_floating_point_types = (torch.float16, torch.float32, torch.float64)

_numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def from_numpy(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.tensor(x)


def pytorch_dtype_to_floating_numpy_dtype(dtype):
    """Converts PyTorch dtype to numpy floating point dtype, defaulting to np.float32 for non-floating point types."""
    if dtype == torch.float64:
        dtype = np.float64
    elif dtype == torch.float32:
        dtype = np.float32
    elif dtype == torch.float16:
        dtype = np.float16
    else:
        dtype = np.float32
    return dtype


def to_numpy(x, dtype: np.dtype = None) -> np.ndarray:
    """
    Convert numeric object to floating point numpy array. If dtype is not specified, use PyTorch default dtype.

    Args:
        x: numeric object
        dtype: numpy dtype, must be floating point

    Returns:
        floating point numpy array
    """

    assert np.issubdtype(dtype, np.floating), "dtype must be real-valued floating point"

    # Convert to normal_form expression from a special form (https://reference.wolfram.com/language/ref/Normal.html)
    if hasattr(x, 'normal_form'):
        x = x.normal_form()

    if type(x) == np.ndarray:
        assert np.issubdtype(x.dtype, np.floating), f"numpy type promotion not implemented for {x.dtype}"

    if hasattr(x, "detach"):
        dtype = pytorch_dtype_to_floating_numpy_dtype(x.dtype)
        return x.detach().cpu().numpy().astype(dtype)

    # list or tuple, iterate inside to convert PyTorch arrrays
    if type(x) in [list, tuple]:
        x = [to_numpy(r) for r in x]

    # Some Python type, use numpy conversion
    result = np.array(x, dtype=dtype)
    assert np.issubdtype(result.dtype, np.number), f"Provided object ({result}) is not numeric, has type {result.dtype}"
    if dtype is None:
        return result.astype(pytorch_dtype_to_floating_numpy_dtype(torch.get_default_dtype()))
    return result


def to_numpys(*xs, dtype=np.float32):
    return (to_numpy(x, dtype) for x in xs)


def check_close_reshape(a0, b0, *args, **kwargs) -> None:
    check_close(a0, b0.reshape(a0.shape), *args, **kwargs)


# TODO(y): rename args to observed, truth
def check_close(a0, b0, rtol=1e-5, atol=1e-8, label: str = '') -> None:
    """Convenience method for check_equal with tolerances defaulting to typical errors observed in neural network
    ops in float32 precision."""
    return check_equal(a0, b0, rtol=rtol, atol=atol, label=label)


def check_equal(observed, truth, rtol=1e-9, atol=1e-12, label: str = '') -> None:
    """
    Assert fail any entries in two arrays are not close to each to desired tolerance. See np.allclose for meaning of rtol, atol

    """

    # special handling for lists, which could contain
    # if type(observed) == List and type(truth) == List:
    #    for a, b in zip(observed, truth):
    #        check_equal(a, b)

    truth = to_numpy(truth)
    observed = to_numpy(observed)

    assert observed.shape == truth.shape, "Shapes don't match"
    # broadcast to match shapes if necessary
    #  if observed.shape != truth.shape:
    #
    #        common_shape = (np.zeros_like(observed) + np.zeros_like(truth)).shape
    # truth = truth + np.zeros_like(observed)
    # observed = observed + np.zeros_like(truth)

    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    # run np.testing.assert_allclose for extra info on discrepancies
    if not np.allclose(observed, truth, rtol=rtol, atol=atol, equal_nan=True):
        print(f'Numerical testing failed for {label}')
        np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol, equal_nan=True)


import torch
import torch.utils.data as data


def is_matrix(dd) -> bool:
    shape = dd.shape
    return len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1


def least_squares_loss_add_classes(data: torch.Tensor, targets=None, reduction='mean'):
    return least_squares_loss(data, targets, reduction, class_reduction='sum')

def least_squares_loss(data: torch.Tensor, targets=None, reduction='mean', class_reduction='mean'):
    """Least squares loss (like MSELoss), but an extra 1/2 factor.

    reduction and class_reduction determine whether to average or add batch elements/class elements respectively.
    """

    assert is_matrix(data), f"Expected matrix, got {data.shape}"
    assert reduction in ('mean', 'sum')
    if targets is None:
        targets = torch.zeros_like(data)
    # err = data - targets.view(-1, data.shape[1])
    err = data - targets
    normalizer = len(data) if reduction == 'mean' else 1
    assert len(targets.shape) == 2, f"Expected rank-2 target shape, instead see shape {targets.shape}"
    num_classes = targets.shape[1]
    normalizer = normalizer * (num_classes if class_reduction == 'mean' else 1)
    return torch.sum(err * err) / 2 / normalizer


def combined_nll_loss(output, target, reduction='mean'):
    # output = F.log_softmax(x, dim=1)
    #             loss = F.nll_loss(output, target)
    # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    output = F.log_softmax(output, dim=1)
    return F.nll_loss(output, target, reduction=reduction)


class NumpyDataset(data.Dataset):
    def __init__(self, A, Y, dataset_size=0):
        super().__init__()

        def maybe_load_from_numpy(arr):
            if isinstance(arr, str):
                return np.load(arr)
            return arr

        self.data = torch.tensor(maybe_load_from_numpy(A)).type(torch.get_default_dtype())
        self.targets = torch.tensor(maybe_load_from_numpy(Y)).type(torch.get_default_dtype())

        if dataset_size > 0:
            self.data = self.data[:dataset_size]
            self.targets = self.targets[:dataset_size]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class ToyDataset(NumpyDataset):
    def __init__(self):
        A = [[1, 0], [1, 1]]
        Y = [[1, 2], [3, 5]]
        super().__init__(A, Y)

    def __len__(self):
        return len(self.data)


class CustomMNIST(datasets.MNIST):
    """Custom-size MNIST autoencoder dataset with extra features. Generates data/target images with reduced resolution and 0
    channels.

    Use original_targets kwarg to get original MNIST labels instead of autoencoder targets.


    """

    mnistTrainWhiteningMatrix = None

    def __init__(self, dataset_root='../data', data_width=28, dataset_size=0, train=True, loss_type='CrossEntropy', device=None,
                 whiten_and_center=False):
        """

        Args:
            data_width: dimension of input images
            dataset_size: number of examples, use for smaller subsets and running locally
            loss_type: if LeastSquares, then convert classes to one-hot format
            whiten_and_center: if True, centers and whitens dataset
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(dataset_root, download=True, train=train)
        assert loss_type in [None, 'LeastSquares', 'CrossEntropy']

        # compute whitening matrix. Can only be done on training data MNIST
        if whiten_and_center:
            assert train or CustomMNIST.mnistTrainWhiteningMatrix is not None

            root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            if CustomMNIST.mnistTrainWhiteningMatrix is None:
                # load whitening matrix from file
                cached_whiten_fn = root + '/data/mnistWhiten.npy'

                if os.path.exists(cached_whiten_fn):
                    W = np.load(cached_whiten_fn)
                    print('total entries2: ', W.sum())
                    np.testing.assert_allclose(W.sum(), -461.2285)
                # compute it
                else:
                    data = self.data
                    data = data.reshape(data.shape[0], -1)
                    A = data - torch.mean(data.float(), dim=1, keepdim=True)
                    #  cov = A.T @ A
                    cov = getCov(A)
                    W = isymsqrtStable(cov)

                    # 712 non-zero eigs, 60000 examples, normalize to have examples with unit norm on average
                    W = W / np.sqrt(712)

                    # (optional)
                    # normalize examples to have E[x^2]/E[x^4]=1, see "Adjustment for fourth-moment normalization" in linear-estimation.nb
                    # W = W * 0.0184887

                    np.save(cached_whiten_fn, W)
                    # 0.4867711938302488

                CustomMNIST.mnistTrainWhiteningMatrix = torch.tensor(W, device='cpu')  # do on CPU because GPU numerics weren't tested

        if dataset_size > 0:
            self.data = self.data[:dataset_size, :, :]
            self.targets = self.targets[:dataset_size]

        if data_width != 28:
            assert not whiten_and_center, "Whitening only supported for default size images"
            new_data = np.zeros((self.data.shape[0], data_width, data_width))
            for i in range(self.data.shape[0]):
                arr = self.data[i, :].numpy().astype(np.uint8)
                im = Image.fromarray(arr)
                im.thumbnail((data_width, data_width), Image.ANTIALIAS)
                new_data[i, :, :] = np.array(im) / 255
            self.data = torch.from_numpy(new_data).type(torch.get_default_dtype())

        if whiten_and_center:
            data = self.data.cpu().double()
            data = data.reshape(data.shape[0], -1)
            data = data - torch.mean(data, dim=1, keepdim=True)
            data = data @ CustomMNIST.mnistTrainWhiteningMatrix
            data = data.reshape(-1, 28, 28)
            self.data = data.cuda() if torch.cuda.is_available() else data

        # insert channel dimension
        self.data = self.data.type(torch.get_default_dtype()).unsqueeze(1)

        if loss_type == 'LeastSquares':  # convert to one-hot format
            new_targets = torch.zeros((self.targets.shape[0], 10))
            new_targets.scatter_(1, self.targets.unsqueeze(1), 1)
            self.targets = new_targets

        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def infinite_iter(obj):
    """Wraps iterable object to restart on last iteration."""

    while True:
        for result in iter(obj):
            yield result


from typing import Callable, Optional, Tuple, Union
from typing import List

import torch
import torch.nn as nn


class SimpleFullyConnected(nn.Module):
    """Simple feedforward network that works on images. """

    def __init__(self, d: List[int], nonlin=False, bias=False, init_scale=0., last_layer_linear=False, hadamard_init=False):
        """
        Feedfoward network of linear layers with optional ReLU nonlinearity. Stores linear layers in "layers" attr, ie
        model.layers[0] refers to first linear layer.

        Initializes to multiple of identity.

        Args:
            d: list of layer dimensions, ie [768, 20, 10] for MNIST 10-output with hidden layer of 20
            nonlin: whether to include ReLU nonlinearity
            last_layer_linear: if True, don't apply nonlinearity to loast layer
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d

        for i in range(len(d) - 1):
            linear = nn.Linear(d[i], d[i + 1], bias=bias)

            self.layers.append(linear)
            self.all_layers.append(linear)

            if hadamard_init:
                # get num_vecs orthogonal vectors corresponding to the smaller dimension, embedding dimension is the larger
                # since vectors need to be powers of two, pad remainder with 0s
                orig_dim = max(d[i + 1], d[i])
                num_vecs = min(d[i + 1], d[i])
                dim = 2 ** math.floor(math.log2(orig_dim))
                hadmat = scipy.linalg.hadamard(dim) / np.sqrt(dim)
                hadmat = hadmat[:num_vecs]
                hadmat = np.pad(hadmat, ((0, 0), (0, orig_dim - dim)), 'constant', constant_values=0)
                hadmat = hadmat if d[i+1] <= d[i] else hadmat.T
                hadmat = torch.from_numpy(hadmat)
                linear.weight.data = hadmat.type(torch.get_default_dtype())
            else:
                # Initialize with identity
                bigger = max(d[i + 1], d[i])
                smaller = min(d[i + 1], d[i])
                # linear.weight.data = torch.zeros((d[i + 1], d[i]))
                eye = torch.eye(bigger)
                linear.weight.data = init_scale * eye[:d[i + 1], :d[i]]

            if bias:
                linear.bias.data = torch.zeros(d[i + 1])
            setattr(self, f'linear_{i:03d}', linear)

            if nonlin:
                if not last_layer_linear or i < len(d) - 2:
                    relu = nn.ReLU()
                    self.all_layers.append(relu)
                    setattr(self, f'relu_{i:03d}', relu)

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        for layer in self.all_layers:
            x = layer(x)
        return x


# def run_all_tests(module: nn.Module):
#     class local_timeit:
#         """Decorator to measure length of time spent in the block in millis and log
#         it to TensorBoard."""
#
#         def __init__(self, tag=""):
#             self.tag = tag
#
#         def __enter__(self):
#             self.start = time.perf_counter()
#             return self
#
#         def __exit__(self, *args):
#             self.end = time.perf_counter()
#             interval_ms = 1000 * (self.end - self.start)
#             global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
#             print(f"{interval_ms:8.2f}   {self.tag}")
#
#     all_functions = inspect.getmembers(module, inspect.isfunction)
#     for name, func in all_functions:
#         if name.startswith("test_"):
#             with local_timeit(name):
#                 func()
#     print(module.__name__ + " tests passed.")


def is_vector(dd) -> bool:
    shape = dd.shape
    return len(shape) == 1 and shape[0] >= 1


def kron(a: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], b: Optional[torch.Tensor] = None):
    """Kronecker product a otimes b."""

    if isinstance(a, Tuple):
        assert b is None
        a, b = a

    if is_vector(a) and is_vector(b):
        return torch.einsum('i,j->ij', a, b).flatten()

    # print('inside a', a)
    # print('inside b', b)
    result = torch.einsum("ab,cd->acbd", a, b)
    # print('kron', result)
    # TODO: use tensor.continuous

    if result.is_contiguous():
        return result.view(a.size(0) * b.size(0), a.size(1) * b.size(1))
    else:
        # print("Warning kronecker product not contiguous, using reshape")
        return result.reshape(a.size(0) * b.size(0), a.size(1) * b.size(1))


def nan_check(mat):
    nan_mask = torch.isnan(mat).float()
    nans = torch.sum(nan_mask).item()
    not_nans = torch.sum(torch.tensor(1) - nan_mask).item()

    assert nans == 0, f"matrix of shape {mat.shape} has {nans}/{nans + not_nans} nans"


def has_nan(mat):
    return torch.sum(torch.isnan(mat)) > 0


def isymsqrt(mat, *args):
    return symsqrt(mat, inverse=True, *args)


def symsqrt(mat, cond=None, return_rank=False, inverse=False):
    """Computes the symmetric square root of a symmetric matrix. Throws away small and negative eigenvalues."""

    nan_check(mat)
    s, u = torch.symeig(mat, eigenvectors=True)

    # todo(y): dedupe with getcond
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    if cond in [None, -1]:
        cond = cond_dict[mat.dtype]

    # Note, this can include negative values, see https://github.com/pytorch/pytorch/issues/25972
    above_cutoff = (s > cond * torch.max(abs(s)))

    if torch.sum(above_cutoff) == 0:
        return torch.zeros_like(mat)

    sigma_diag = torch.sqrt(s[above_cutoff])
    if inverse:
        sigma_diag = 1 / sigma_diag
    u = u[:, above_cutoff]

    B = u @ torch.diag(sigma_diag) @ u.t()

    if torch.sum(torch.isnan(B)) > 0:
        assert False, "Got nans"

    if return_rank:
        return B, len(sigma_diag)
    else:
        return B


def isymsqrtStable(cov: np.ndarray):
    """Stable inverse square root.

    Use float64 precision
    Don't double tiny dimensions

    Use GESVD algorithm, GESDD is known to have problems with ill-conditioned matrices, see
    https://github.com/tensorflow/tensorflow/issues/9234
    https://github.com/pytorch/pytorch/pull/11194#issuecomment-418421118
    https://github.com/tensorflow/tensorflow/issues/26842
    https://discourse.julialang.org/t/svd-better-default-to-gesvd-instead-of-gesdd/20603

    """

    cov = cov.cpu().numpy() if hasattr(cov, 'cpu') else cov
    cov = cov.astype(np.float64)

    U, s, Vh = scipy.linalg.svd(cov, lapack_driver='gesvd')
    eps = np.finfo(cov.dtype).eps

    # For cutoff discussion, see https://github.com/scipy/scipy/issues/10879
    cutoff = np.max(s) * max(cov.shape) * eps

    indices = np.where(s < cutoff)[0]   # indices of svals below cutoff, could be empty
    rank = min(indices) if len(indices) else len(s)
    return U[:, :rank] @ np.diag(1/np.sqrt(s[:rank])) @ Vh[:rank]


def getIntrinsicDim(mat):
    mat = np.array(mat.cpu() if hasattr(mat, 'cpu') else mat)
    s = scipy.linalg.svd(mat, compute_uv=False)
    return (s.sum()**2)/(s*s).sum()


def getCov(mat):
    m = mat.shape[0]
    return mat.T @ mat / m

def getMoment2(X):
    vals = (X*X).sum(axis=1)
    return vals.mean()

def getMoment4(X):
    vals = (X*X*X*X).sum(axis=1)
    return vals.mean()

@contextmanager
def module_hook(hook: Callable):
    handle = nn.modules.module.register_module_forward_hook(hook, always_call=True)
    yield
    handle.remove()


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def zero_custom_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def copy_custom_grad_to_grad(model: nn.Module) -> None:
    """for all layers in the model, replaces .grad with .custom_grad"""
    for p in model.parameters():
        if hasattr(p, 'custom_grad'):
            p.grad = p.custom_grad
        else:
            assert p.grad is None, "existing grad found, but not replacing it with custom_grad, consider model.zero_grad()"


def kaczmarz_grad_linear(layer: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
    """Linear hook for Kaczmarz update"""
    # skip over all non-leaf modules, like top-level nn.Sequential
    if not is_leaf_module(layer):
        return

    # which one to use?
    assert _layer_type(layer) == 'Linear', f"Only linear layers supported but got {_layer_type(layer)}"
    assert layer.__class__ == nn.Linear

    assert len(inputs) == 1, "multi-input layer??"
    A = inputs[0].detach()

    has_bias = hasattr(layer, 'bias') and layer.bias is not None

    def tensor_backwards(B: torch.Tensor):
        # use notation of "Kaczmarz step-size"/Multiclass Layout
        # https://notability.com/n/2TQJ3NYAK7If1~xRfL26Ap
        (m, n) = A.shape

        norms2 = (A * A).sum(axis=1)
        if has_bias:
            norms2 += 1

        ones = torch.ones((m,)).to(B.device)
        # ones = torch.ones_like()
        update = torch.einsum('mn,mc,m->nc', A, B, ones / norms2)
        layer.weight.custom_grad = update.T  # B.T @ A

        if has_bias:
            # B is (m, c) residual matrix
            update = torch.einsum('mc,m->c', B, ones / norms2)
            layer.bias.custom_grad = update / m

    existing_hooks = output._backward_hooks
    assert existing_hooks is None, f"Tensor already has backward hooks, {existing_hooks}, potential bug if we forgot to remove previous hooks"
    output.register_hook(tensor_backwards)


def evaluate_mnist(test_loader, model, do_squared_loss):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_loss = 0
    num_correct = 0
    loss_fn = least_squares_loss if do_squared_loss else combined_nll_loss

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            if do_squared_loss:
                actual = target.argmax(dim=1, keepdim=True)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                num_correct += pred.eq(actual).sum().item()
            else:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                num_correct += pred.eq(target.view_as(pred)).sum().item()

    m = len(test_loader.dataset)
    return total_loss / m, num_correct / m


def run_all_tests(module: nn.Module):
    class local_timeit:
        """Decorator to measure length of time spent in the block in millis and log
        it to TensorBoard."""

        def __init__(self, tag=""):
            self.tag = tag

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            interval_ms = 1000 * (self.end - self.start)
            global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
            print(f"{interval_ms:8.2f}   {self.tag}")

    all_functions = inspect.getmembers(module, inspect.isfunction)
    for name, func in all_functions:
        if name.startswith("test_"):
            with local_timeit(name):
                func()
    print(module.__name__ + " tests passed.")


if __name__ == '__main__':
    run_all_tests(sys.modules[__name__])
