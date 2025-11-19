import torch
import torch.nn as nn
from models.googlenet import BasicConv2d


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
            # nn.ReLU(inplace=True),
            BasicConv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d((3, 3), (2, 2)),  # output[48, 27, 27] kernel_num为原论文一半
            # nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            # nn.ReLU(inplace=True),
            BasicConv2d(48, 128, kernel_size=5, padding=2),
            nn.MaxPool2d((3, 3), (2, 2)),  # output[128, 13, 13]
            # nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            # nn.ReLU(inplace=True),
            BasicConv2d(128, 192, kernel_size=3, padding=1),
            # nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            # nn.ReLU(inplace=True),
            BasicConv2d(192, 192, kernel_size=3, padding=1),
            # nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            # nn.ReLU(inplace=True),
            BasicConv2d(192, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.input_shape = (1, 3, 224, 224)
        self.output_shapes = [(1, 48, 55, 55), (1, 48, 27, 27), (1, 128, 27, 27), (1, 128, 13, 13), (1, 192, 13, 13),
                              (1, 192, 13, 13), (1, 128, 13, 13), (1, 128, 6, 6)]
        self.len_features = len(self.features)
        self.len_classifier = len(self.classifier)
        self.depth = self.len_features + self.len_classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_part(self, x, start_idx, end_idx):
        assert 0 <= start_idx <= end_idx <= self.len_features + self.len_classifier
        for i in range(start_idx, end_idx):
            if i < self.len_features:
                x = self.features[i](x)
            else:
                if i == self.len_features:
                    x = x.view(x.size(0), -1)
                x = self.classifier[i - self.len_features](x)
        return x

    def get_layer(self, idx):
        if idx < self.len_features:
            return self.features[idx]
        return self.classifier[idx - self.len_features]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)


class AlexNetbig(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNetbig, self).__init__()

        self.features = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            # nn.ReLU(True),
            BasicConv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.MaxPool2d((3, 3), (2, 2)),

            # nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            # nn.ReLU(True),
            BasicConv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.MaxPool2d((3, 3), (2, 2)),

            # nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(True),
            BasicConv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(True),
            BasicConv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(True),
            BasicConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self.input_shape = (1, 3, 224, 224)
        self.output_shapes = [(1, 64, 55, 55), (1, 64, 27, 27), (1, 192, 27, 27), (1, 192, 13, 13), (1, 384, 13, 13), (1, 256, 13, 13), (1, 256, 13, 13), (1, 256, 6, 6)]
        self.len_features = len(self.features)
        self.len_classifier = len(self.classifier)
        self.depth = self.len_features + self.len_classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_part(self, x, start_idx, end_idx):
        assert 0 <= start_idx <= end_idx <= self.len_features + self.len_classifier
        for i in range(start_idx, end_idx):
            if i < self.len_features:
                x = self.features[i](x)
            else:
                if i == self.len_features:
                    x = x.view(x.size(0), -1)
                x = self.classifier[i - self.len_features](x)
        return x

    def get_layer(self, idx):
        if idx < self.len_features:
            return self.features[idx]
        return self.classifier[idx - self.len_features]


if __name__ == '__main__':
    model = AlexNetbig()
    x = torch.randn(model.input_shape)
    output_shapes = []
    for layer in model.features:
        x = layer(x)
        output_shapes.append(tuple(x.shape))
    print(output_shapes)
