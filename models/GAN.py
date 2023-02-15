import torch
import torch.nn as nn
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dims, hid_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_dims, out_dims)
        )

    def forward(self, x):
        return self.model(x)
