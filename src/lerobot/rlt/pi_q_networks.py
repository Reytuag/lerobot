import torch.nn as nn
import torch

class Policy_network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x, action_ref):
        x = torch.cat([x, action_ref.reshape(action_ref.shape[0], -1)], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class Q_network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x,action_ref):
        x = torch.cat([x, action_ref.reshape(action_ref.shape[0], -1)], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)