import torch
import torch.nn as nn

"""
The state is represented by a 200-element boolean tensor:
- 40 elements for the player's hand (multihot)
- 40 elements for the cards taken by the player (multihot)
- 40 elements for the cards taken by the opponent (multihot)
- 40 elements for the briscola card (onehot)
- 40 elements for the card on the table (onehot)
"""
INPUT_SIZE = 40 + 40 + 40 + 40 + 40 
HIDDEN_SIZE = 512
OUTPUT_SIZE = 40

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(INPUT_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, OUTPUT_SIZE)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = x.masked_fill(~mask, float('-inf'))
        x = self.softmax(x)
        return x