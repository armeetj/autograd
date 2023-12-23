import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
sys.path.append("/Users/armeetjatyani/Developer/autograd/")
import numpy as np

# load dataset
ds = np.load("mnist.npz")
X_train = np.array(ds["train"].T, float)
y_train = np.array(ds["train_labels"][0], int)
X_test = np.array(ds["test"].T, float)
y_test = np.array(ds["test_labels"][0], int)

def one_hot_encode(x):
    enc = np.zeros(10, float)
    enc[x] = 1
    return enc

def one_hot_decode(enc):
    x = np.argmax(enc)
    return x

X_train /= 255.0
X_test /= 255.0
y_train = np.array([one_hot_encode(y) for y in y_train])
y_test = np.array([one_hot_encode(y) for y in y_test])

print([one_hot_decode(y) for y in y_test])

# construct model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(28 * 28, 100)
        self.l2 = nn.Linear(100, 50)
        self.l3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        return x
net = Net()
device = torch.device("mps")
net.to(device)

lr = 0.05
optimizer = optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# train model
epochs = 10
# net.train()
for epoch in tqdm(range(epochs), desc="[epoch]"):
    # for i in tqdm(range(len(X_train[:10000])), leave=True, position=1):
    for i in range(len(X_train[:10000])):
        x = torch.tensor(X_train[i], dtype=torch.float32).to(device)
        y_pred = net(x).to(device)
        y_actual = torch.tensor(y_train[i], dtype=torch.float32).to(device)

        loss = loss_fn(y_pred, y_actual)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            count = 0
            for j in range(1000):
                x = torch.tensor(X_test[j], dtype=torch.float32).to(device)
                y_pred = one_hot_decode(net(x).cpu().detach().numpy())
                y_actual = one_hot_decode(y_test[j])
                print(f"y_pred: {y_pred}, y_actual: {y_actual}")
                if y_pred == y_actual:
                    count += 1
            print(count)
            print(f"loss: {loss}, acc: {count / 1000.}")
        