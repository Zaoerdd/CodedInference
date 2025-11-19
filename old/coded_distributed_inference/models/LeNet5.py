import time
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.layers = nn.Sequential( # 0,3,6,8,9,10
            nn.Conv2d(1, 6, 3),  #1
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  #3
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),  #6
            nn.Flatten(start_dim=1), # keep batch
            nn.Linear(400, 120),  #8
            nn.Linear(120, 84),  #9
            nn.Linear(84, 10)  #10
        )  # 5x5 image dimension
        self.input_shape = (1, 28, 28)
        self.depth = len(self.layers)
        print(f'depth of model: {self.depth}')

    def forward(self, x):
        x = self.layers(x)
        return x

    def forward_part(self, x, start_idx, end_idx):
        assert 0 <= start_idx <= end_idx <= len(self.layers)
        for i in range(start_idx, end_idx):
            x = self.layers[i](x)
        return x

    def forward_with_shapes(self, x, start_idx, end_idx):
        assert 0 <= start_idx <= end_idx <= len(self.layers)
        shapes = []
        for i in range(start_idx, end_idx):
            shape = tuple(x.shape)
            shapes.append(shape)
            print(f'input shape {shape} of layer {i + 1}')
            x = self.layers[i](x)
        shape = tuple(x.shape)
        print(f'output shape {shape} of layer {i + 1}')

        return x

    def delay_of_layers(self, x):
        delays = []
        for i in self.layers:
            t1 = time.time()
            x = i(x)
            t2 = time.time()
            delays.append(t2 - t1)
        return delays
