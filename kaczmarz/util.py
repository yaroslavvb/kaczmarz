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

import globals as gl

# to enable referring to functions in its own module as u.func
u = sys.modules[__name__]

def log_scalars(metrics: Dict[str, Any]) -> None:
    assert gl.event_writer is not None, "initialize event_writer as gl.event_writer = SummaryWriter(logdir)"
    for tag in metrics:
        gl.event_writer.add_scalar(tag=tag, scalar_value=metrics[tag], global_step=gl.get_global_step())
        # gl.event_writer.add_s
    if 'epoch' in metrics:
        print('logging at ', gl.get_global_step(), metrics.get('epoch', -1))


global_timeit_dict = {}
class timeit:
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
        # print(f"{interval_ms:8.2f}   {self.tag}")
        log_scalars({'time/' + self.tag: interval_ms})


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

    # broadcast to match shapes if necessary
    if observed.shape != truth.shape:
        #        common_shape = (np.zeros_like(observed) + np.zeros_like(truth)).shape
        truth = truth + np.zeros_like(observed)
        observed = observed + np.zeros_like(truth)

    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    # run np.testing.assert_allclose for extra info on discrepancies
    if not np.allclose(observed, truth, rtol=rtol, atol=atol, equal_nan=True):
        print(f'Numerical testing failed for {label}')
        np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol, equal_nan=True)


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
