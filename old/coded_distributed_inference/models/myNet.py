import time
from torch import nn

from googlenet import BasicConv2d
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True, return_indices=False)
        self.conv2 = BasicConv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3 = BasicConv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True, return_indices=False)
        self.BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        self.BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.layers = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(4, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(6 * 5 * 5, 10)
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
