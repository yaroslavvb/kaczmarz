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
        self.fc1 = nn.Linear(d0, 10)

    def forward(self, x):
        x = x.reshape((-1, self.d0))
        x = self.fc1(x)
        return x


root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)

    dataset_size = 1000
    train_kwargs = {'batch_size': 10, 'num_workers': 0, 'shuffle': False}
    test_kwargs = {'batch_size': 10}

    # do_squared_loss = False
    do_squared_loss = True
    loss_type = 'LeastSquares' if do_squared_loss else 'CrossEntropy'
    loss_fn = kac.least_squares_loss if do_squared_loss else kac.combined_nll_loss

    train_dataset = kac.NumpyDataset(root + '/data/mnistTrainW.npy',
                                     root + '/data/mnistTrain-labels.npy',
                                     dataset_size=dataset_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    test_dataset = kac.NumpyDataset(root + '/data/mnistTestW.npy',
                                    root + '/data/mnistTest-labels.npy',
                                    dataset_size=dataset_size)

    # train_dataset = kac.TinyMNIST(train=True, whiten=True)
    # test_dataset = kac.TinyMNIST(train=False, whiten=True, whiteningMatrix=train_dataset.whiteningMatix)

    model = Net(d0=28 * 28).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1/10, momentum=0.)

    print("dataset size: ", len(train_dataset))
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
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            output = model(data)
            # new_loss = F.nll_loss(output, target)
            new_loss = loss_fn(output, target)
        print(
            f"accuracy: {correct / total:0.2f}, test_loss: {test_loss:.2f}, old train loss: {loss.item():.2f}, new train loss: {new_loss.item():.2f}")


if __name__ == '__main__':
    main()
