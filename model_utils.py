import torch
import torch.nn as nn

from models.googlenet import BasicConv2d

def auto_segment_model(model, input_shape):
    """
    根据池化层自动分割一个CNN模型。

    Args:
        model (nn.Module): 需要被分割的模型 (例如 VGG16)。
        input_shape (tuple): 模型的输入形状, 例如 (1, 3, 224, 224)。

    Returns:
        dict: 一个包含所有块配置的字典。
        list: 一个包含所有被识别出的池化层的列表。
    """
    block_configs = {}
    pooling_layers = []
    
    # 我们假设模型的卷积部分在一个名为 'features' 的 nn.Sequential 模块中
    # 这对于 VGG, AlexNet 等模型是通用的
    if not hasattr(model, 'features') or not isinstance(model.features, nn.Sequential):
        print("警告: 模型没有 'features' 属性或该属性不是 nn.Sequential。无法自动分割。")
        return {}, []

    current_segment_layers = []
    in_channels = input_shape[1]
    dummy_input = torch.randn(input_shape)
    
    # ** 新增：辅助函数安全获取通道信息 **
    def get_channel_info(layer):
        if isinstance(layer, nn.Conv2d):
            return layer.in_channels, layer.out_channels
        elif isinstance(layer, BasicConv2d):
            # 通过访问 BasicConv2d 内部的 nn.Conv2d 模块来获取通道信息
            return layer.conv.in_channels, layer.conv.out_channels
        return None, None
    
    segment_count = 1

    # 遍历模型的所有特征层
# 遍历模型的所有特征层
    for layer in model.features:
        # ** 1. 允许 BasicConv2d 作为卷积层 **
        if isinstance(layer, (nn.Conv2d, BasicConv2d)): 
            if not current_segment_layers:
                # 打印 Block 的起始输入尺寸
                print(f"DEBUG: Block {segment_count} Input Shape: {list(dummy_input.shape)}")
                
            current_segment_layers.append(layer)
        
        elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            if current_segment_layers:
                block_name = f'block_{segment_count}'
                
                # ** 2. 使用辅助函数获取 out_channels **
                _, out_channels = get_channel_info(current_segment_layers[-1])
                layer_configs = [get_channel_info(l) for l in current_segment_layers]
                
                block_configs[block_name] = {
                    'layers': layer_configs,
                    'in_c': in_channels,
                    'out_c': out_channels,
                    'width': dummy_input.shape[-1]
                }
                
                # 通过“干运行”来计算当前段的输出
                temp_model = nn.Sequential(*current_segment_layers)
                dummy_input = temp_model(dummy_input)
                
                # 打印 Block 的卷积输出尺寸
                print(f"DEBUG: {block_name} Conv Output Shape: {list(dummy_input.shape)}")

                # 存储池化层
                pooling_layers.append(layer)
                
                # 让虚拟输入通过池化层，以获得下一个段的正确输入形状
                dummy_input = layer(dummy_input)
                
                # 打印 Block 的池化输出尺寸 (下一个 Block 的输入)
                print(f"DEBUG: {block_name} Pool Output Shape: {list(dummy_input.shape)}")
                
                in_channels = out_channels
                
                current_segment_layers = []
                segment_count += 1
        
        elif isinstance(layer, nn.ReLU):
            # ReLU层被视为卷积块的一部分，直接跳过
            pass
        else:
            print(f"警告: 遇到无法处理的层类型 {type(layer)}。分割中止。")
            break

    # 处理最后一组卷积层 (在最后一个池化层之后)
    if current_segment_layers:
        block_name = f'block_{segment_count}'
        
        # 再次使用辅助函数获取信息
        _, out_channels = get_channel_info(current_segment_layers[-1])
        layer_configs = [get_channel_info(l) for l in current_segment_layers]
        
        block_configs[block_name] = {
            'layers': layer_configs,
            'in_c': in_channels,
            'out_c': out_channels,
            'width': dummy_input.shape[-1]
        }
    
    return block_configs, pooling_layers
