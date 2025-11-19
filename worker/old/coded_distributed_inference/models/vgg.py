import torch.nn as nn
import torch
import sys
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class vgg11(nn.Module):
    def __init__(self, num_classes=1000):
        super(vgg11, self).__init__()
        self.features = make_layers(cfg['A'])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.input_shape = (3, 224, 224)
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
class vgg16(nn.Module):
    def __init__(self, num_classes=1000):
        super(vgg16, self).__init__()
        self.features = make_layers(cfg['D'])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.input_shape = (3, 224, 224)
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
                x = self.classifier[i-self.len_features](x)
        return x

    def get_layer(self, idx):
        if idx < self.len_features:
            return self.features[idx]
        return self.classifier[idx - self.len_features]


if __name__ == '__main__':

    vgg = vgg16()
    # print(vgg.)
    # PATH = '../checkpoints/myNet/vgg16-397923af.pth'
    # vgg.load_state_dict(torch.load(PATH))
    input = torch.randn((10, 3, 224, 224))
    print(input)
    input = vgg(input)
    print(input.shape)
    print(input)
    for i in vgg.modules():
        print(i)

# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# Linear(in_features=4096, out_features=4096, bias=True)