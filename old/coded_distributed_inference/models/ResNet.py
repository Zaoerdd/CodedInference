import time
# from functions import next_to_last, translate_next_array
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # two continuous conv layer in self.left
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # if the shape of feature map has changed, the shortcut output should be processed
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        # add and relu
        self.block_relu = nn.ReLU()
        self.layers = [*self.left, *self.shortcut, 'add', self.block_relu]

    def forward(self, x):
        out = self.left(x)
        # add result from left and shortcut
        out = out + self.shortcut(x)
        out = self.block_relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.features = [*self.conv1,
                       *self.layer1[0].layers,
                       *self.layer1[1].layers,
                       *self.layer2[0].layers,
                       *self.layer2[1].layers,
                       *self.layer3[0].layers,
                       *self.layer3[1].layers,
                       *self.layer4[0].layers,
                       *self.layer4[1].layers,
                    ]
        self.layers = self.features + [self.avg_pool, self.fc]
        self.input_shape = 3, 224, 224
        self.output_shapes = [(1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 64, 224, 224), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 128, 112, 112), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 56, 56), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28), (1, 512, 28, 28)]
        # 表示所有layers的topological order的next数组
        self.next = [1,2,[3,8],4,5,6,7,8,9,[10,15],11,12,13,14,15,16,[17,22],18,19,20,21,24,23,24,25,[26,31],27,28,29,30
            ,31,32,[33,38],34,35,36,37,40,39,40,41,[42,47],43,44,45,46,47,48,[49,54],50,51,52,53,56,55,56,57,[58,63],59,
            60,61,62,63,64,[]]


    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        # 1st block with stride=stride, and the rest blocks with stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# class ResNet18(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet18, self).__init__()
#
#         # 还有一个+的操作没有在这里面体现出来
#
#         self.conv1 = BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         # layer1 block0
#         self.layer1_block0_conv1 = BasicConv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
#         self.layer1_block0_conv2 = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))
#         self.layer1_block0_bn2 = nn.BatchNorm2d(64)
#         # layer1 block0 shortcut
#         self.layer1_block0_relu = nn.ReLU()
#         # layer1 block1
#         self.layer1_block1_conv1 = BasicConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.layer1_block1_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.layer1_block1_bn2 = nn.BatchNorm2d(64)
#         # layer1 block1 shortcut
#         self.layer1_block1_relu = nn.ReLU()
#         # layer2 block0
#         self.layer2_block0_conv1 = BasicConv2d(64, 128, kernel_size=(3,3), stride=(2, 2), padding=(1,1))
#         self.layer2_block0_conv2 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
#         self.layer2_block0_bn = nn.BatchNorm2d(128)
#         # layer2 block0 shortcut
#         self.layer2_block0_shortcut_conv = BasicConv2d(64, 128, kernel_size=(1,1), stride=(2, 2))
#         self.layer2_block0_shortcut_bn = nn.BatchNorm2d(128)
#         self.layer2_block0_relu = nn.ReLU()
#         # layer2 block1
#         self.layer2_block1_conv1 = BasicConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.layer2_block1_conv2 = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))
#         self.layer2_block1_bn2 = nn.BatchNorm2d(128)
#         # layer2 block1 shortcut
#         self.layer2_block1_relu = nn.ReLU()
#         # layer3 block0
#         self.layer3_block0_conv1 = BasicConv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.layer3_block0_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.layer3_block0_bn = nn.BatchNorm2d(256)
#         # layer3 block0 shortcut
#         self.layer3_block0_shortcut_conv = BasicConv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
#         self.layer3_block0_shortcut_bn = nn.BatchNorm2d(256)
#         self.layer3_block0_relu = nn.ReLU()
#         # layer3 block1
#         self.layer3_block1_conv1 = BasicConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.layer3_block1_conv2 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
#         self.layer3_block1_bn2 = nn.BatchNorm2d(256)
#         # layer3 block1 shortcut
#         self.layer3_block1_relu = nn.ReLU()
#         # layer4 block0
#         self.layer4_block0_conv1 = BasicConv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.layer4_block0_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.layer4_block0_bn = nn.BatchNorm2d(512)
#         # layer4 block0 shortcut
#         self.layer4_block0_shortcut_conv = BasicConv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
#         self.layer4_block0_shortcut_bn = nn.BatchNorm2d(512)
#         self.layer4_block0_relu = nn.ReLU()
#         # layer4 block1
#         self.layer4_block1_conv1 = BasicConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.layer4_block1_conv2 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))
#         self.layer4_block1_bn2 = nn.BatchNorm2d(512)
#         # layer4 block1 shortcut
#         self.layer4_block1_relu = nn.ReLU()
#
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
#         self.fc = nn.Linear(512, num_classes)


if __name__ == '__main__':
    model = ResNet(ResBlock)
    # for i, layer in enumerate(model.layers):
    #     print(i, layer
    #     )
    # # next array转成last array
    # translate_next_array(model.next)
    # print(next_to_last(model.next))

    test_latency = True
    if test_latency:
        print(f'{len(model.layers)} layers in total')
        x = torch.randn(1, 3, 224, 224)
        start = time.time()
        y = model(x)
        consumption = time.time() - start
        print(consumption)