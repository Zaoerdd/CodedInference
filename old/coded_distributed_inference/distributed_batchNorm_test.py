import torch.nn as nn
import torch.nn.functional as F
import torch

conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
bn = nn.BatchNorm2d(64, eps=0.001, affine=False, track_running_stats=True)
bn.eval()

input = torch.randn((1, 3, 224, 224))
input11 = conv(input)
input12 = F.conv2d(input, conv.weight, stride=1, padding=(1,1))

print(torch.equal(input11, input12))
print(bn.running_mean, bn.running_var)
input21 = bn(input11)
input221 = F.batch_norm(input12[..., :112], bn.running_mean, bn.running_var, bn.weight, eps=0.001, training=False, momentum=0.1)
input222 = F.batch_norm(input12[..., 112:], bn.running_mean, bn.running_var, bn.weight, eps=0.001, training=False, momentum=0.1)
input22 = torch.concat([input221, input222], dim=-1)
print(torch.allclose(input21, input22))
