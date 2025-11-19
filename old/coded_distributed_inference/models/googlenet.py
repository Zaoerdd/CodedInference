# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import time
from collections import namedtuple
from typing import Optional, Tuple, Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "GoogLeNetOutputs",
    "GoogLeNet",
    "BasicConv2d", "Inception", "InceptionAux",
    "googlenet",
]

# According to the writing of the official library of Torchvision
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

def cal_forward_time(layer, x):
    start = time.time()
    out = layer(x)
    consumption = time.time() - start
    return out, consumption

class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
            self,
            num_classes: int = 1000,
            aux_logits: bool = True,
            transform_input: bool = False,
            dropout: float = 0.2,
            dropout_aux: float = 0.7,
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True, return_indices=False)
        self.conv2 = BasicConv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3 = BasicConv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True, return_indices=False)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True, return_indices=False)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True, return_indices=False)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes, dropout_aux)
            self.aux2 = InceptionAux(528, num_classes, dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout, True)
        self.fc = nn.Linear(1024, num_classes)

        self.input_shape = (1, 3, 224, 224)
        self.output_shapes = None
        self.layers = [
            self.conv1,  # 0
            self.maxpool1,  # 1
            self.conv2,  # 2
            self.conv3,  # 3
            self.maxpool2,  # 4
            self.inception3a.branch1,  # 5
            *self.inception3a.branch2,  # 67
            *self.inception3a.branch3,  # 89
            *self.inception3a.branch4,  # 1011
            # concat 70
            self.inception3b.branch1,  # 12
            *self.inception3b.branch2,  # 1314
            *self.inception3b.branch3,  # 1516
            *self.inception3b.branch4,  # 1718
            # concat 71
            self.maxpool3,  # 19
            self.inception4a.branch1,  # 20
            *self.inception4a.branch2,  # 2122
            *self.inception4a.branch3,  # 2324
            *self.inception4a.branch4,  # 2526
            # concat 72
            self.inception4b.branch1,  # 27
            *self.inception4b.branch2,  # 2829
            *self.inception4b.branch3,  # 3031
            *self.inception4b.branch4,  # 3233
            # concat 73
            self.inception4c.branch1,  # 34
            *self.inception4c.branch2,  # 3536
            *self.inception4c.branch3,  # 3738
            *self.inception4c.branch4,  # 3940
            # concat 74
            self.inception4d.branch1,  # 41
            *self.inception4d.branch2,  # 4243
            *self.inception4d.branch3,  # 4445
            *self.inception4d.branch4,  # 4647
            # concat 75
            self.inception4e.branch1,  # 48
            *self.inception4e.branch2,  # 4950
            *self.inception4e.branch3,  # 5152
            *self.inception4e.branch4,  # 5354
            # concat 76
            self.maxpool4,  # 55
            self.inception5a.branch1,  # 56
            *self.inception5a.branch2,  # 5758
            *self.inception5a.branch3,  # 5960
            *self.inception5a.branch4,  # 6162
            # concat 77
            self.inception5b.branch1,  # 63
            *self.inception5b.branch2,  # 6465
            *self.inception5b.branch3,  # 6667
            *self.inception5b.branch4,  # 6869
            # concat 78
            # model.avgpool,  # 70  avgpool with output_shape (1, 1) means kernel_size = input_shape
            # model.dropout,  # 71
            # model.fc,  # 72
            'concat',  # 73
            'concat',  # 74
            'concat',  # 75
            'concat',  # 76
            'concat',  # 77
            'concat',  # 78
            'concat',  # 79
            'concat',  # 80
            'concat',  # 81
        ]
        self.next = [1, 2, 3, 4, [5, 6, 8, 10], 70, 7, 70, 9, 70, 11, 70, 71, 14, 71, 16, 71, 18, 71, [20, 21, 23, 25],
                     72, 22, 72,
                     24, 72, 26, 72, 73, 29, 73, 31, 73, 33,
                     73, 74, 36, 74, 38, 74, 40, 74, 75, 43, 75, 45, 75, 47, 75, 76, 50, 76, 52, 76, 54, 76,
                     [56, 57, 59, 61], 77,
                     58, 77, 60, 77, 62, 77, 78, 65, 78,
                     67, 78, 69, 78, [12, 13, 15, 17], 19, [27, 28, 30, 32], [34, 35, 37, 39], [41, 42, 44, 46],
                     # remove the last_array several layers like adaptive average pool and fully connected layers
                     [48, 49, 51, 53], 55, [63, 64, 66, 68], []]
        self.depth = len(self.next)
        # Initialize neural network weights
        self._initialize_weights()

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        if self.training and self.aux_logits:
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        out = self._forward_impl(x)

        return out

    def forward_feature(self, x: Tensor):
        x = self._transform_input(x)

        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)
        out = self.inception4a(out)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(out)

        out = self.inception4e(out)
        out = self.maxpool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)

        return aux1, aux2, out

    def forward_classifier(self, aux1, aux2, x: Tensor):
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        aux3 = self.fc(out)

        if torch.jit.is_scripting():
            return GoogLeNetOutputs(aux3, aux2, aux1)
        else:
            return self.eager_outputs(aux3, aux2, aux1)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)

        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)
        out = self.inception4a(out)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(out)

        out = self.inception4e(out)
        out = self.maxpool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)

        return out

        # out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        # out = self.dropout(out)
        # aux3 = self.fc(out)
        #
        # if torch.jit.is_scripting():
        #     return GoogLeNetOutputs(aux3, aux2, aux1)
        # else:
        #     return self.eager_outputs(aux3, aux2, aux1)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def layer_latency(self):
        latency = []
        x = torch.randn(self.input_shape)
        layers = [self.conv1,
                  self.maxpool1,
                  self.conv2,
                  self.conv3,
                  self.maxpool2,
                  self.inception3a,
                  self.inception3b,
                  self.maxpool3,
                  self.inception4a,
                  self.inception4b,
                  self.inception4c,
                  self.inception4d,
                  self.inception4e,
                  self.maxpool4,
                  self.inception5a,
                  self.inception5b,
                  self.avgpool,
                  self.fc,
                  ]
        for layer in layers:
            if layer == self.fc:
                x = x.view(x.size(0), -1)
            x, delay = cal_forward_time(layer, x)
            latency.append(delay)
        print(latency)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        start = time.time()
        out = self.conv(x)
        end = time.time()
        print(end - start)
        out = self.bn(out)
        out = self.relu(out)

        return out


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
    ) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = [branch1, branch2, branch3, branch4]

        out = torch.cat(out, 1)

        return out


class InceptionAux(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            dropout: float = 0.7,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout, True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.avgpool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def googlenet(**kwargs: Any) -> GoogLeNet:
    model = GoogLeNet(**kwargs)

    return model


if __name__ == '__main__':
    model = GoogLeNet()
    model.layer_latency()