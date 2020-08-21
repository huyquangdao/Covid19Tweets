import torch
import torch.nn as nn


class MNISTModel(nn.Module):

    def __init__(self, n_classes, image_size, hidden_size):
        
        super().__init__()
        self.linear = nn.Linear(image_size ** 2, hidden_size)
        self.linear_out = nn.Linear(hidden_size, n_classes)

    def forward(self, images):

        bs = images.shape[0]
        output = torch.relu(self.linear(images.view(bs, -1)))
        output = self.linear_out(output)
        return output
