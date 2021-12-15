
# Create an virtual environments with Anaconda

Go to https://www.anaconda.com/ to download the right version for the machine you are using.

Install it following the instructions provided by anaconda. You could check its version to make sure you installed successfully: 

$ conda -V

Now, create a new environment using conda:

$ conda create -n yourenvname python=x.x

(for example: $conda create -n MyEnv python=3.6)

Activate your created virtual environment:

$ source activate yourenvname

Once you typed, your terminal should look sth like:

(yourenvname) xxxxxx@xxxxxxxx: $ 

Install all your needed packages under 'yourenvname' by typing (generally speaking):

$ conda install -n NAME package=VERSION 

(PLEASE GOOGLE how to install the package in anaconda before your installing.)

The packages that you will need for your project includes: numpy, scipy, matplotlib, pytorch, opencv, jupyter notebook, etc... 
    

!! IF YOU USE GOOGLE CLOUD:
You can SKIP all the above procedures.  !!

# Start with jupyter notebook

1. IF YOU USE A LOCAL MACHINE

    Open a terminal, activate your virtual environment. 

    Then open your jupyter notebook:
    
    $ jupyter notebook

    This will automaticly open up a browser. 


2. IF YOU USE A REMOTE MACHINE

    Remotely log in to the machine, you could also do this:
    
    $ jupyter notebook --no-browser --port=xxxx

    Open another terminal, do:
    
    $ ssh -N -L localhost:yyyy:localhost:xxxx xxxxx@129.10.xx.xx -p 44
    
    (e.g. 'xxxx' = '8889', 'yyyy' = '8880')
    
   
3. IF YOU USE GOOGLE CLOUD
    
    Skip the above steps. 

# Now let's start with Pytorch

Before you start, you should roughly know:

1. Your goal is to build a network, which includes one(or more) layers, and each layer (might)has one or more weights(tensors).

2. Your network will do the following jobs:
   read in data; forward data through the network; yield some loss; back propagate the gradients through the network.
   
3. Your real tasks and objectives:
   What will be your input data? (Tensors? Images? Videos?)
   What will be your output from the network? (Class labels? Images? Features?)
   What loss function will you use? (MSE? L2? )

# Let's start coding! 

Our very first coding example is really simple.

Input Data - some random generated 2D tensors of shape 64 by 1000

Output Data - 2D tensors of shape 64 by 100

Loss Function - MSE

Network(Model) - 2 layers model that apply linear transformations to the incoming data.
    

# First, import the packages you need


```python
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
```

# Second, your data!


```python
#### Input data generation ####
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
```

# Third, your model!


```python
#### Build your model ####
class TwoLayerNet(nn.Module):
    def __init__(self,D_in,H,D_out):

        super(TwoLayerNet, self).__init__()

        
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
```

# Lastly, train your model for multiple times(epochs).


```python
EPOCH = 500
LR = 1e-4
#gpu_id = 1

model = TwoLayerNet(D_in, H, D_out)
#model.cuda(gpu_id)

loss_fn = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    y_pred = model(x) #(x.cuda(gpu_id))

    loss = loss_fn(y_pred,y) #y.cuda(gpu_id)
    print(epoch, loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
```

# Other ways of coding your model...


```python
#### Build your model ####
class L1(nn.Module):
    def __init__(self,D_in,H):
        super(L1,self).__init__()
        self.linear1 = nn.Linear(D_in,H)
    def forward(self,x):
        o1 = self.linear1(x).clamp(min=0)
        return o1
        
        
class L2(nn.Module):
    def __init__(self,H,D_out):
        super(L2,self).__init__()
        self.linear2 = nn.Linear(H,D_out)
    def forward(self,x):
        o2 = self.linear2(x)
        return o2
    
class TwoLayerNet2(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet2,self).__init__()
        self.l1 = L1(D_in,H)
        self.l2 = L2(H,D_out)

    def forward(self,x):
        return self.l2(self.l1(x))
 
```

# A More Complicated Example!


```python
#### Package import ####
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
```


```python
#### Data Loader ####

batch_size=64
mnist = datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_loader = torch.utils.data.DataLoader(mnist,batch_size=batch_size, shuffle=True, num_workers = 1)

# For more information, please refer to https://pytorch.org/docs/stable/torchvision/datasets.html
```


```python
#### Build model ####
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```


```python
#### Train #####

epochs=5
Lr=0.01
Momentum=0.5
log_interval=10
save_model=True

model = Net().cuda()
optimizer = optim.SGD(model.parameters(), lr=Lr, momentum=Momentum)

for epoch in range(0, epochs):
    #model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(),target.cuda()        
        output = model(data)
        loss = F.nll_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    
    #test(args, model, device, test_loader)

if (save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
```

# What about testing?


```python
#### Test Data ####
test_batch_size=1000
mnist = datasets.MNIST('data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_loader = torch.utils.data.DataLoader(mnist,batch_size=test_batch_size, shuffle=True, num_workers = 1)


```


```python
#### Test #####
loadedcheckpoint = torch.load('mnist_cnn.pt')

model = Net().cuda()

for epoch in range(0, epochs):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

```
