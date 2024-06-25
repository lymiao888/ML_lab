
import torch
from torch import nn

class fully_connected_model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(in_features=input_shape, out_features=hidden_units),  # in_features = number of features in the data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)  # out_features = number of classes for classification (10)
        )
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)