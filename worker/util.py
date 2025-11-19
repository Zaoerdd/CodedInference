import torch
import socket
import pickle
from models.VGG16 import vgg16
from models.AlexNet import AlexNet

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
        return vgg16()
    elif model_name.lower() == 'alexnet':
        # return models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        return AlexNet()
    else:
        raise ValueError(f"模型 {model_name} 在模拟 util.py 中不受支持")

def get_ip_addr(subnet=None):
    """
    master.py 调用的函数以获取主节点的 IP 地址。
    """

    return "127.0.0.1"

def save_object(obj, filename):
    """
    使用 pickle 将对象序列化并保存到文件。
    master.py 调用的实际保存函数。
    """
    try:
        with open(filename, 'wb') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        print(f"[util.py] 对象已保存到: {filename}")
    except Exception as e:
        print(f"[util.py] 错误: 无法保存对象到 {filename}. 错误: {e}")
                    
def load_object(filename):
    """
    使用 pickle 从文件反序列化并加载对象。
    master.py 调用的实际加载函数。
    """
    try:
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        print(f"[util.py] 对象已从: {filename} 加载")
        return obj
    except FileNotFoundError:
        print(f"[util.py] 警告: 文件未找到: {filename}. 返回 None。")
        # 保持对 master.py 中旧代码的兼容性
        if 'total_run_latencies.tmp' in filename:
             return [0.1, 0.2] # 返回模拟数据以防 np.mean() 失败
        return None
    except Exception as e:
        print(f"[util.py] 错误: 无法从 {filename} 加载对象. 错误: {e}")
        return None