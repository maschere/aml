# %% imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.random.manual_seed(124)

# auto grad
a = torch.tensor([5.0]).requires_grad_(True)
(4*a + 3).backward()
a.grad
a.grad.data.zero_() # zero grad



# image data
img1 = torch.rand((100, 3, 128, 128))  # N,C,H,W
img2 = torch.rand((50, 3, 256, 128))

# classification target
num_classes = 7


# %% MLP sequential

mlp = nn.Sequential(
    nn.Flatten(), #flatten image to 1d vector
    nn.Linear(in_features=3 * 128 * 128, out_features=128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(64, 7),  # final layer for classification
    nn.Softmax(1)
)
mlp.forward(img1).shape

# %% MLP class
class MLP_Model(nn.Module):
    def __init__(self): #define all layers here (order does not matter)
        super().__init__()
        self.fully1 = nn.Linear(in_features=3 * 128 * 128, out_features=128)
        self.bn = nn.BatchNorm1d(128)
        self.fully2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(64, 7)  # final layer for classification

    def forward(self, x): #define forward direction
        x = torch.flatten(x,1)
        x = F.relu(self.bn(self.fully1(x)))
        x = F.relu(self.fully2(x))
        x = self.dropout(x)  # deactivate with eval() mode
        x = F.softmax(self.head(x), 1)
        return x


mlp2 = MLP_Model()
[{n: (p.shape, np.product(p.shape))} for n, p in list(mlp2.named_parameters())]
mlp2.forward(img1).shape

#eval mode
mlp2.forward(img1).std()
mlp2.eval()
mlp2.forward(img1).std()


# %% CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2)
        self.bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2)
        self.head = nn.Linear(30*30*12, num_classes) # how to calculate this?
        # W' =  (W-K+2*P)/S+1
        # W'' = int((int((128-5)/2+1)-3)/2+1)
        # n = W'' * W'' * C

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1)
        x = F.softmax(self.head(x), 1)
        return x



cnn = CNN()
[{n: (p.shape, np.product(p.shape))} for n, p in list(cnn.named_parameters())]
cnn.forward(img1).shape

# %% fully convolutional
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2)
        self.head = nn.Conv2d(12, num_classes, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.softmax(F.adaptive_avg_pool2d(self.head(x), 1), 1)
        return x.squeeze(-1).squeeze(-1) #get rid of dim 2,3 as they are 1


fcnn = FCNN()
[{n: (p.shape, np.product(p.shape))} for n, p in list(fcnn.named_parameters())]
fcnn.forward(img1).shape
# %%
