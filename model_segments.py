import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model_utils import auto_segment_model

class SegmentEncoder(nn.Module):
    """
    对应图中的 Encoder E，使用 CNN 实现。
    """
    def __init__(self, k_workers, r_workers, in_channels):
        super().__init__()
        self.k = k_workers
        self.r = r_workers
        self.in_channels = in_channels * k_workers
        self.cnn_body = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1), nn.ReLU()
        )
        self.parity_heads = nn.ModuleList(
            [nn.Conv2d(32, in_channels, kernel_size=3, padding=1) for _ in range(r_workers)]
        )
    def forward(self, list_of_k_tensors):
        concatenated_tensor = torch.cat(list_of_k_tensors, dim=1)
        features = self.cnn_body(concatenated_tensor)
        parity_pieces = [head(features) for head in self.parity_heads]
        return parity_pieces

class SegmentFinalDecoder(nn.Module):
    """
    对应图中的 Decoder D，使用 CNN 实现。
    """
    def __init__(self, k_workers, out_channels, final_width):
        super().__init__()
        self.k = k_workers
        self.in_channels = out_channels * k_workers
        self.final_width = final_width
        self.reconstructor = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    def forward(self, k_outputs, k_indices):
        sorted_pairs = sorted(zip(k_indices, k_outputs), key=lambda p: p[0])
        sorted_outputs = [p[1] for p in sorted_pairs]
        concatenated_tensor = torch.cat(sorted_outputs, dim=1)
        reconstructed_full = self.reconstructor(concatenated_tensor)
        reconstructed_full = F.interpolate(reconstructed_full, size=(reconstructed_full.shape[2], self.final_width))
        return reconstructed_full

class BlockWorkerDecoder(nn.Module):
    """ 
    通用的卷积块 Worker 解码器。
    """
    def __init__(self, layer_configs):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for in_channels, out_channels in layer_configs:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        return x

# --- 主要修改: 动态模型加载器 ---
def load_segment_models_dynamically(model_name, k_workers, r_workers, input_shape):
    """
    动态地加载、分割任何兼容的模型，并为每个段创建所需的组件。
    """
    master_models = {}
    worker_models = {}

    # 1. 根据名称加载预训练模型
    if model_name.lower() == 'vgg16':
        # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model = models.vgg16()
    elif model_name.lower() == 'alexnet':
        # model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model = models.alexnet()
    else:
        raise ValueError(f"模型 '{model_name}' 尚不支持自动分割。")
    
    # 2. 调用我们的新工具来自动分割模型
    block_configs, pooling_layers = auto_segment_model(model, input_shape)
    print(f"模型 '{model_name}' 被自动分割为 {len(block_configs)} 个块。")
    
    if not block_configs:
        raise RuntimeError("模型分割失败，请检查模型结构。")

    # 3. 根据动态生成的配置来构建所有组件
    for block_name, configs in block_configs.items():
        encoder = SegmentEncoder(k_workers, r_workers, configs['in_c'])
        final_decoder = SegmentFinalDecoder(k_workers, configs['out_c'], configs['width'])
        worker_decoder = BlockWorkerDecoder(configs['layers'])
        
        master_models[block_name] = (encoder, final_decoder)
        worker_models[block_name] = worker_decoder

    # 将所有模型设置为评估模式
    for key in master_models:
        master_models[key][0].eval()
        master_models[key][1].eval()
        worker_models[key].eval()
    
    # Master 节点需要知道在段之间应用哪些池化层
    return master_models, worker_models, pooling_layers
