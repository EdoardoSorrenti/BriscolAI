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
HIDDEN_SIZE = 128
OUTPUT_SIZE = 40

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, mask, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = x.masked_fill(~mask, float('-inf'))
        return self.softmax(x)