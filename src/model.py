import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv1 block: 1 -> 8 channels
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  
        self.bn1 = nn.BatchNorm2d(8)                
        self.dropout1 = nn.Dropout(0.01)

        # Conv2 block: 8 -> 12 channels
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(12)               
        self.dropout2 = nn.Dropout(0.01)
        self.pool1 = nn.MaxPool2d(2, 2)             

        # Conv3 block: 12 -> 16 channels
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)               
        self.dropout3 = nn.Dropout(0.01)
        self.pool2 = nn.MaxPool2d(2, 2)             

        # FC layers
        self.fc1 = nn.Linear(16 * 7 * 7, 16)        
        self.fc_dropout = nn.Dropout(0.1)            
        self.fc2 = nn.Linear(16, 10)                
        
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

        # FC forward pass
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1) 