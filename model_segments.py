import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
from models.VGG16 import vgg16
from model_utils import auto_segment_model

# class SegmentEncoder(nn.Module):
#     """
#     对应图中的 Encoder E，使用 CNN 实现。
#     """
#     def __init__(self, k_workers, r_workers, in_channels):
#         super().__init__()
#         self.k = k_workers
#         self.r = r_workers
#         # 让我们计算单个分片的通道数 C_k:
#         C_k = in_channels // k_workers
#         # 拼接后的通道数 C_total 应该是 C_k * k = in_channels。
#         # (假设 in_channels 可以被 k_workers 整除)
#         self.in_channels = in_channels 
        
#         # self.in_channels = in_channels # <--- 使用这个修复

#         self.cnn_body = nn.Sequential(
#             nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 32, kernel_size=1), nn.ReLU()
#         )
#         self.parity_heads = nn.ModuleList(
#             [nn.Conv2d(32, C_k, kernel_size=3, padding=1) for _ in range(r_workers)]
#         )
#         # self.in_channels = in_channels * k_workers
#         # self.cnn_body = nn.Sequential(
#         #     nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
#         #     nn.Conv2d(64, 32, kernel_size=1), nn.ReLU()
#         # )
#         # self.parity_heads = nn.ModuleList(
#         #     [nn.Conv2d(32, in_channels, kernel_size=3, padding=1) for _ in range(r_workers)]
#         # )
#     def forward(self, list_of_k_tensors):
#         concatenated_tensor = torch.cat(list_of_k_tensors, dim=1)
#         features = self.cnn_body(concatenated_tensor)
#         parity_pieces = [head(features) for head in self.parity_heads]
#         return parity_pieces

class SegmentEncoder(nn.Module):
    """
    对应图中的 Encoder E，使用 CNN 实现。
    适用于空间分片：接收 k 个空间分片，沿通道维度拼接后进行编码。
    """
    def __init__(self, k_workers, r_workers, in_channels):
        super().__init__()
        self.k = k_workers
        self.r = r_workers
        
        # 1. 空间分片编码：SegmentEncoder的输入通道是 k 个分片沿通道拼接后的总通道数。
        # 每个分片有 in_channels 个通道，拼接后为 in_channels * k_workers。
        self.in_channels = in_channels * k_workers 
        
        self.cnn_body = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1), nn.ReLU()
        )
        # 2. 奇偶校验片的通道数是原始通道数 in_channels。
        self.parity_heads = nn.ModuleList(
            [nn.Conv2d(32, in_channels, kernel_size=3, padding=1) for _ in range(r_workers)]
        )
        
    def forward(self, list_of_k_tensors):
        """
        list_of_k_tensors: k 个空间分片，形状均为 [B, C, H/k, W]
        """
        # 3. 沿通道维度 (dim=1) 拼接 k 个空间分片
        # 结果形状: [B, C*k, H/k, W]
        concatenated_tensor = torch.cat(list_of_k_tensors, dim=1)
        
        features = self.cnn_body(concatenated_tensor)
        parity_pieces = [head(features) for head in self.parity_heads]
        return parity_pieces

# class SegmentFinalDecoder(nn.Module):
#     """
#     对应图中的 Decoder D，使用 CNN 实现。
#     """
#     def __init__(self, k_workers, out_channels, final_width):
#         super().__init__()
#         self.k = k_workers
#         self.in_channels = out_channels * k_workers
#         self.final_width = final_width
#         self.reconstructor = nn.Sequential(
#             nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1), nn.ReLU(),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, out_channels, kernel_size=1)
#         )
#     def forward(self, k_outputs, k_indices):
#         sorted_pairs = sorted(zip(k_indices, k_outputs), key=lambda p: p[0])
#         sorted_outputs = [p[1] for p in sorted_pairs]
#         concatenated_tensor = torch.cat(sorted_outputs, dim=1)
#         reconstructed_full = self.reconstructor(concatenated_tensor)
#         reconstructed_full = F.interpolate(reconstructed_full, size=(reconstructed_full.shape[2], self.final_width))
#         return reconstructed_full

class SegmentFinalDecoder(nn.Module):
    """
    对应图中的 Decoder D，使用 CNN 实现。
    修正为：兼容空间分片 (沿高度 dim=2 拼接)。
    """
    def __init__(self, k_workers, out_channels, final_width):
        super().__init__()
        self.k = k_workers
        # 🐛 修正：in_channels 现在应该是 out_channels（因为沿高度拼接，通道不变）
        self.in_channels = out_channels # <--- 修正: 沿高度拼接，通道数不变
        self.final_width = final_width
        self.reconstructor = nn.Sequential(
            # 🐛 修正：确保卷积层的输入通道数与拼接后的通道数一致
            nn.Conv2d(out_channels, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    def forward(self, k_outputs, k_indices):
        sorted_pairs = sorted(zip(k_indices, k_outputs), key=lambda p: p[0])
        sorted_outputs = [p[1] for p in sorted_pairs]
        
        # 1. 🐛 修正：沿【高度维度】 (dim=2) 拼接 k 个空间分片
        # 即使分块不均匀，也必须沿高度拼接
        concatenated_tensor = torch.cat(sorted_outputs, dim=2) # <--- 修正为 dim=2
        
        reconstructed_full = self.reconstructor(concatenated_tensor)
        
        # 2. 🐛 修正：移除或修正插值
        # 移除可能导致高度错误的 F.interpolate，因为空间分片要求所有张量 W 相同。
        # 如果需要调整 W，可以执行：
        if reconstructed_full.shape[3] != self.final_width:
             print(f"[WARN] Decoder 正在调整宽度: {reconstructed_full.shape[3]} -> {self.final_width}")
             reconstructed_full = F.interpolate(reconstructed_full, 
                                                 size=(reconstructed_full.shape[2], self.final_width))
        
        # 否则，直接返回
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
        model = vgg16()
    # elif model_name.lower() == 'alexnet':
    #     model = models.alexnet()
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
