import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import kaczmarz.kac as kac

u = kac

class Net(nn.Module):
    def __init__(self, d0):
        super(Net, self).__init__()
        self.d0 = d0
        self.fc1 = nn.Linear(d0, 10, bias=False)

    def forward(self, x):
        x = x.reshape((-1, self.d0))
        x = self.fc1(x)
        return x


root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def train_pytorch():
    do_kaczmarz = False
    do_kaczmarz = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)

    dataset_size = 1000
    train_kwargs = {'batch_size': 1, 'num_workers': 0, 'shuffle': False}
    test_kwargs = {'batch_size': 1}

    # do_squared_loss = False
    do_squared_loss = True
    loss_type = 'LeastSquares' if do_squared_loss else 'CrossEntropy'
    loss_fn = kac.least_squares_loss if do_squared_loss else kac.combined_nll_loss

    train_dataset = kac.CustomMNIST(train=True, loss_type=loss_type, whiten_and_center=True, dataset_size=dataset_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    test_dataset = kac.CustomMNIST(train=False, loss_type=loss_type, whiten_and_center=True, dataset_size=dataset_size)

    model = Net(d0=28 * 28).to(device)

    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.)

    for epoch in range(1, 10):
        model.eval()
        test_loss, correct, total = 0, 0, 0

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
                total += data.shape[0]
        test_loss /= len(test_loader.dataset)

        model.train()
        loss, new_loss = None, None

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if do_kaczmarz:
                with kac.module_hook(kac.kaczmarz_grad_linear):
                    output = model(data)
            else:
                output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

            if do_kaczmarz:
                kac.copy_custom_grad_to_grad(model)

            optimizer.step()

            new_loss = loss_fn(output, target)
        print(
            f"accuracy: {correct / total:0.2f}, test_loss: {test_loss:.2f}, old train loss: {loss.item():.2f}, new train loss: {new_loss.item():.2f}")


def train_numpy(use_kaczmarz=False):
    model = u.SimpleFullyConnected([28 ** 2, 10])
    W = model.layers[0].weight.data.T

    # largest convergent LR for MNIST SGD, see https://www.wolframcloud.com/obj/yaroslavvb/nn-linear/whitened-mnist.nb
    max_lr = 0.4867711
    lr = 1 if use_kaczmarz else max_lr

    # using "Multiclass layout", classes is the second dimension"
    # https://notability.com/n/2TQJ3NYAK7If1~xRfL26Ap

    dataset_size = 10000
    loss_fn = u.least_squares_loss
    loss_type = 'LeastSquares'
    bs = 1   # batch size
    c = 10   # number of classes

    train_dataset = u.CustomMNIST(dataset_size=dataset_size, train=True, whiten_and_center=True, loss_type=loss_type)
    train_loader_eval = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_size)

    test_dataset = u.CustomMNIST(dataset_size=dataset_size, train=False, whiten_and_center=True, loss_type=loss_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=dataset_size)

    X, Y = train_dataset.data, train_dataset.targets
    X = X.reshape(-1, 28*28)
    m, n = X.shape

    losses = []
    num_steps = 10000
    eval_interval = 1000

    print(f"loss: train/test")

    for i in range(num_steps):
        idx = i % m
        a = X[idx:idx + bs, :]
        y = a @ W
        r = y - Y[idx:idx + bs]
        loss = 0.5 * (r**2).sum()/(bs * c)
        g = a.T @ r / (bs * c)
        normalizer = 1/u.norm2(a) if use_kaczmarz else 1
        W = W - lr * g * normalizer
        losses.extend([loss.item()])

        if i % eval_interval == 0:
            model.layers[0].weight.data = W.T
            train_loss, _ = kac.evaluate_mnist(train_loader_eval, model, True)
            test_loss, _ = kac.evaluate_mnist(test_loader, model, True)
            print(f"loss: {train_loss:.4f}/{test_loss:.4f}")


if __name__ == '__main__':
    print("SGD")
    train_numpy(use_kaczmarz=False)
    print("\nKaczmarz")
    train_numpy(use_kaczmarz=True)

