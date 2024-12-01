from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from torchvision.transforms import RandomAffine
from torch.optim.lr_scheduler import ReduceLROnPlateau
from test_model import run_all_tests
from model import Net

ROTATION_DEGREES = 7.0  # Max rotation degrees (random between -7 to +7)
SHEAR_DEGREES = 5.0    # Max shear degrees (random between -5 to +5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
if not run_all_tests(model):
    raise ValueError("Model does not meet requirements")

summary(model, input_size=(1, 28, 28))

torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        RandomAffine(
                            degrees=ROTATION_DEGREES,
                            shear=SHEAR_DEGREES
                        ),
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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return accuracy  # Return accuracy for scheduler

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Increased initial learning rate
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Increase number of epochs for better convergence
for epoch in range(1, 20):  # Changed to 20 epochs
    print(f'\nEpoch: {epoch}')
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
    scheduler.step(accuracy)  # Update learning rate based on accuracy