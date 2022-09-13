import torch
from torch import nn, optim
from torch.utils.data import (Dataset, 
                              DataLoader,
                              TensorDataset)
from src.model import FizzBuzz
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def fizzbuzz(number: int) -> str:
    if number % 15 == 0:
        return "fizzbuzz"
    elif number % 3 == 0:
        return "fizz"
    elif number % 5 == 0:
        return "buzz"
    return str(number)

def binary_encode(i: int, num_digits: int) -> List[int]:
    return np.array([i >> d & 1 for d in range(num_digits)])

assert fizzbuzz(1) == "1"
assert fizzbuzz(3) == "fizz"
assert fizzbuzz(5) == "buzz"
assert fizzbuzz(15) == "fizzbuzz"


def fizz_buzz_encode(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else: return 0

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


NUM_DIGITS = 10
NUM_HIDDEN = 100
BATCH_SIZE = 32
X = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
y = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

X_train = X[100:]
y_train = y[100:]
X_valid = X[:100]
y_valid = y[:100]

assert X_train.shape == torch.Size([823, 10])
assert y_train.shape == torch.Size([823])
assert X_valid.shape == torch.Size([100, 10])
assert y_valid.shape == torch.Size([100])


X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = torch.tensor(y_valid, dtype=torch.int64)


net = FizzBuzz()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.03)

ds = TensorDataset(X_train, y_train)
loader = DataLoader(ds, batch_size=32, shuffle=True)


def testing():
    numbers = np.arange(1, 101)
    X_test = np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
    X_test = torch.tensor(X_test, dtype=torch.float32)

    net.eval()
    _, y_pred = torch.max(net(X_test), 1)

    output = np.vectorize(fizz_buzz)(numbers, y_pred)
    print(output)


if __name__ == '__main__':

    train_losses = []
    test_losses = []

    for _ in range(1000):
        running_loss = 0.0

        net.train()
        
        for i, (xx, yy) in enumerate(loader):
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / i)
        
        net.eval()
        y_pred = net(X_valid)
        test_loss = loss_fn(y_pred, y_valid)
        print('Validation loss: ', test_loss, 'Training loss: ', loss)
        test_losses.append(test_loss.item())

        testing()

