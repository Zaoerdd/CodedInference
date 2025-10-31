# util.py

import torch
from torchvision import models
import socket

def load_model(model_name):
    """
    加载一个预训练模型。
    master.py 在主函数中调用此函数。
    """
    print(f"[util.py] 正在加载模型: {model_name}")
    
    # 禁用预训练权重以下载（如果本地没有缓存）
    # weights = 'DEFAULT'
    weights = None 

    if model_name.lower() == 'vgg16':
        # return models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        return models.vgg16(weights=None) 
    elif model_name.lower() == 'alexnet':
        # return models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        return models.alexnet(weights=None)
    else:
        raise ValueError(f"模型 {model_name} 在模拟 util.py 中不受支持")

def get_ip_addr(subnet=None):
    """
    worker.py 调用此函数。
    对于单机测试，我们强制它返回 '127.0.0.1'。
    """
    return '127.0.0.1'

def save_object(obj, filename):
    """
    master.py 调用的模拟函数。
    """
    print(f"[util.py] 模拟保存到: {filename}")
    pass
                    
def load_object(filename):
    """
    master.py 调用的模拟函数。
    它需要为 'total_run_latencies.tmp' 返回一个列表。
    """
    print(f"[util.py] 模拟从: {filename} 加载")
    if 'total_run_latencies.tmp' in filename:
         return [0.1, 0.2] # 返回模拟数据以防 np.mean() 失败
    return None