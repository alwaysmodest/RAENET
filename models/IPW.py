import torch
import torch.nn as nn
import torch.optim as optim


class AIPW(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AIPW, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, treatment):
        x = torch.cat((x, treatment), dim=1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

class IPW(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IPW, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


class CBPSNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CBPSNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
