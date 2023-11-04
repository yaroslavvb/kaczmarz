import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import kaczmarz.kac as kac


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


def main():
    do_kaczmarz = True
    do_kaczmarz = False

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
    optimizer = optim.SGD(model.parameters(), lr=2/10, momentum=0.)

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


if __name__ == '__main__':
    main()
