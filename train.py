from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv1 block: 1 -> 8 channels
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # (3*3*1*8) + 8 = 80 params
        self.bn1 = nn.BatchNorm2d(8)                # 16 params
        self.dropout1 = nn.Dropout(0.01)

        # Conv2 block: 8 -> 12 channels (reduced from 16)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1) # (3*3*8*12) + 12 = 876 params
        self.bn2 = nn.BatchNorm2d(12)               # 24 params
        self.dropout2 = nn.Dropout(0.01)
        self.pool1 = nn.MaxPool2d(2, 2)             # Output: 12x14x14

        # Conv3 block: 12 -> 16 channels (reduced from 32)
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1) # (3*3*12*16) + 16 = 1,744 params
        self.bn3 = nn.BatchNorm2d(16)                # 32 params
        self.dropout3 = nn.Dropout(0.01)
        self.pool2 = nn.MaxPool2d(2, 2)              # Output: 16x7x7

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 10)         # (784*10) + 10 = 7,850 params
        
    def forward(self, x):
        # Conv1 block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Conv2 block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        # Conv3 block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.pool2(x)

        # Flatten and FC
        x = x.view(-1, 16 * 7 * 7)  # Flatten the 16x7x7 feature maps
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)