from contextlib import contextmanager
from typing import Callable, Tuple

import torch
import torch.nn as nn

import numpy as np

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
        print("computing norms")
        A = inputs[0].detach()
        layer.norms2 = (A * A).sum(dim=1)

    model = simple_model(2, 3).to(device)

    @contextmanager
    def module_hook(hook: Callable):
        handle = nn.modules.module.register_module_forward_hook(hook, always_call=True)
        yield
        handle.remove()

    with module_hook(compute_norms):
        outputs = model(data)

    np.testing.assert_allclose(model[0].norms2, [1, 2])
    np.testing.assert_allclose(model[1].norms2, [4, 8])
    np.testing.assert_allclose(model[2].norms2, [16, 32])

    print("layer", "norms squared")
    for name, layer in model.named_modules():
        if not name:
            continue
        print(name, layer.norms2)
