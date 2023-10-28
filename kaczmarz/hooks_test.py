from contextlib import contextmanager
from typing import Callable, Tuple

import torch
import torch.nn as nn

import numpy as np

import util as u

def test_global_forward_hook():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.tensor([[1., 0.], [1., 1.]]).to(device)
    bs = data.shape[0]  # batch size

    def simple_model(d, num_layers):
        """Creates simple linear neural network initialized to 2*identity"""
        layers = []
        for i in range(num_layers):
            layer = nn.Linear(d, d, bias=False)
            layer.weight.data.copy_(2 * torch.eye(d))
            layers.append(layer)
        return torch.nn.Sequential(*layers)

    norms = [torch.zeros(bs).to(device)]

    def compute_norms(layer: nn.Module, inputs: Tuple[torch.Tensor], _output: torch.Tensor):
        assert len(inputs) == 1, "multi-input layer??"
        A = inputs[0].detach()
        layer.norms2 = (A * A).sum(dim=1)

    model = simple_model(2, 3).to(device)

    with u.module_hook(compute_norms):
        outputs = model(data)

    np.testing.assert_allclose(model[0].norms2.cpu(), [1, 2])
    np.testing.assert_allclose(model[1].norms2.cpu(), [4, 8])
    np.testing.assert_allclose(model[2].norms2.cpu(), [16, 32])

    # print("layer", "norms squared")
    # for name, layer in model.named_modules():
    #     if not name:
    #         continue
    #    print(name, layer.norms2)

# prototyped in https://colab.research.google.com/drive/1dGeCen7ikIXcWBbsTtrdQ7NKyJ-iHyUw#scrollTo=LDlLyY18-eFa&line=1&uniqifier=1
# revision: manual grad computation
def test_manual_grad_computation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.tensor([[1., 0.], [1., 1.]]).to(device)
    bs = data.shape[0]  # batch size

    def simple_model(d, num_layers):
        """Creates simple linear neural network initialized to 2*identity"""
        layers = []
        for i in range(num_layers):
            layer = nn.Linear(d, d, bias=True)
            layer.weight.data.copy_(2 * torch.eye(d))
            layers.append(layer)
        return torch.nn.Sequential(*layers)

    @contextmanager
    def module_hook(hook: Callable):
        handle = nn.modules.module.register_module_forward_hook(hook, always_call=True)
        yield
        handle.remove()

    def _layer_type(layer: nn.Module) -> str:
        return layer.__class__.__name__

    def is_leaf_module(module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    handles = []

    def manual_grad_linear(layer: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
        # skip over all non-leaf modules, like top-level nn.Sequential
        if not is_leaf_module(layer):
            return

        # which one to use?
        assert _layer_type(layer) == 'Linear', f"Only linear layers supported but got {_layer_type(layer)}"
        assert layer.__class__ == nn.Linear

        assert len(inputs) == 1, "multi-input layer??"
        A = inputs[0].detach()

        idx = len(handles)  # idx is used for closure trick to autoremove hook

        def tensor_backwards(B):
            layer.weight.manual_grad = B.T @ A

            # use notation of "Kaczmarz step-size"/Multiclass Layout
            # https://notability.com/n/2TQJ3NYAK7If1~xRfL26Ap
            if hasattr(layer, 'bias'):
                # B is (m, c) residual matrix
                layer.bias.manual_grad = (B.T).sum(axis=1)

            handles[idx].remove()

        handles.append(output.register_hook(tensor_backwards))

    model = simple_model(2, 3).to(device)

    with module_hook(manual_grad_linear):
        outputs = model(data)

    def least_squares_loss(data, targets=None, aggregation='mean'):
        """Least squares loss (like MSELoss, but an extra 1/2 factor."""
        assert aggregation in ('mean', 'sum')
        if targets is None:
            targets = torch.zeros_like(data)
        err = data - targets.view(-1, data.shape[1])
        normalizer = len(data) if aggregation == 'mean' else 1
        return torch.sum(err * err) / 2 / normalizer

    loss = least_squares_loss(outputs)
    loss.backward()

    for i in range(len(model)):
        assert torch.allclose(model[i].weight.manual_grad, model[i].weight.grad)
        assert torch.allclose(model[i].bias.manual_grad, model[i].bias.grad)