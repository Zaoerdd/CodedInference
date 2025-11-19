from util import *
# from ExecutionUnit import ExecutionUnit
from models.googlenet import BasicConv2d
import queue
import numpy as np
import torch.nn as nn

def translate_next_array(next_array):
    for i in range(len(next_array)):  # 将next数组内的单个元素处理为长度为1的列表
        if not isinstance(next_array[i], list):
            next_array[i] = [next_array[i]]


def next_to_last(next):  # 将next数组转化为last数组，即last数组
    total = len(next)
    last = [[] for _ in range(total)]
    # last_array[0].append(-1)  # -1 represents the original input
    for i, nexts in enumerate(next):
        for l in nexts:
            last[l].append(i)
    return last


# print(len(layers))
# layers_dependency = next_to_last(next_array)


def topology_DAG(next_array, last_array):  # transfer the DAG network to topology list, starts from 0, bfs
    total = len(next_array)
    in_num = np.zeros(total)
    for i in range(total):
        in_num[i] = len(last_array[i])
    q = queue.Queue()
    q.put(0)
    ans = []
    while not q.empty():
        ele = q.get()
        ans.append(ele)
        if isinstance(next_array[ele], list):
            for i in next_array[ele]:
                in_num[i] -= 1
                if in_num[i] == 0:
                    q.put(i)
        else:
            in_num[next_array[ele]] -= 1
            if in_num[next_array[ele]] == 0:
                q.put(next_array[ele])
    return ans


# average distribution
# output features range of layer


def cal_output_shape(net, topology_list, last_array):
    layers = net.layers
    n_layers = len(topology_list)
    output_shapes = [[] for _ in range(n_layers)]
    mark = np.zeros(n_layers)
    for lth in topology_list:
        mark[lth] = 1
        if layers[lth] == 'concat':
            inputs = []
            for last in last_array[lth]:
                inputs.append(torch.randn(output_shapes[last]))
            output = torch.cat(inputs, 1)

        else:
            if lth == 0:
                input_shape = 1, *net.input_shape
            else:
                input_shape = output_shapes[last_array[lth][0]]
            x = torch.randn(input_shape)
            this_layer = net.layers[lth]
            if isinstance(this_layer, nn.Linear):
                x = torch.flatten(x, 1)
            output = this_layer(x)
        output_shapes[lth] = output.shape

    return output_shapes


def cal_output(layers, topology_list, last_array, x):
    n_layers = len(topology_list)
    outputs = [None for _ in range(n_layers)]
    mark = np.zeros(n_layers)
    for lth in topology_list:
        if lth == 0:
            outputs[0] = layers[0](x)
            continue
        mark[lth] = 1
        if layers[lth] == 'concat':
            inputs = [outputs[i] for i in last_array[lth]]
            output = torch.cat(inputs, 1)
        else:
            assert len(last_array[lth]) == 1
            last_layer = last_array[lth][0]
            output = layers[lth](outputs[last_layer])
        outputs[lth] = output
    return outputs


# layers_output_shapes = cal_output_shape(model, topology_layers, layers_dependency)


# def cal_inputFromOutput(output_shapes, last_layers):
#     n_layers = len(output_shapes)
#     input_shapes = [[] for _ in range(n_layers)]
#     for nl in topology_layers:
#         if nl == 0:
#             input_shape = [1, 3, 224, 224]
#         else:
#             lasts = last_layers[nl]
#             if len(lasts) == 1:
#                 last_array = lasts[0]
#                 input_shape = output_shapes[last_array]
#             else:  # have over 1 last_array layers
#                 input_shape = []
#                 for last_array in lasts:
#                     input_shape.append(output_shapes[last_array])
#         input_shapes[nl] = input_shape
#     return input_shapes
#
#
# # compute the layers' output shape and store in model
# model.input_shapes = cal_inputFromOutput(layers_output_shapes, layers_dependency)


# print(model.output_shapes)


# partitioning dimension: -1 # 假设从最后一维开始切
def workload_partition(output_shapes, num_device):  # temporarily average
    partitions = []
    for i, shape in enumerate(output_shapes):
        # if layers[i] == 'concat':
        #     partitions.append(1)
        # else:
        length = shape[-1]
        partition = [0 for _ in range(num_device + 1)]
        partition[-1] = length
        average = round(length / num_device)
        for i in range(1, num_device):
            partition[i] = average * i
        partitions.append(partition)
    return partitions


# def random_workload_partition(output_shapes, num_device):
#     partitions = []
#     for i, shape in enumerate(output_shapes):
#         length = shape[-1]
#         partition = []
#         recv_cnt = 0
#         while recv_cnt < num_device - 1:
#             p = random.randint(1, length - 1)
#             if p not in partition:
#                 recv_cnt += 1
#                 partition.append(p)
#         partition.sort()
#         partition = [0, *partition, length]
#         partitions.append(partition)
#     return partitions


def generate_layerConfigs(layers: list):
    configs = []
    for layer in layers:
        if isinstance(layer, BasicConv2d):
            conv = layer.conv
            layer_config = {'type': 'basicConv', 'kernel_size': conv.kernel_size, 'stride': conv.stride,
                            'padding': conv.padding}  # , 'bn_args': (bn.weight, bn.bias, False, bn.momentum, bn.eps)}
        elif isinstance(layer, nn.Conv2d):
            layer_config = {'type': 'conv', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                            'padding': layer.padding}
        elif isinstance(layer, nn.BatchNorm2d):
            layer_config = {'type': 'batchNorm2d'}
        elif isinstance(layer, nn.ReLU):
            layer_config = {'type': 'relu', 'inplace': layer.inplace}
        elif isinstance(layer, nn.MaxPool2d):
            layer_config = {'type': 'maxpool', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                            'padding': layer.padding, 'ceil_mode': layer.ceil_mode}
        elif isinstance(layer, nn.Upsample):
            layer_config = {'type': 'upsample', 'scale_factor': layer.scale_factor}
        elif layer == 'concat' or layer == 'add':
            layer_config = {'type': layer}
        else:  # only given kinds of layers
            layer_config = None
            print('This type of layer is not supported yet')
        configs.append(layer_config)

    return configs


def generate_layerConfig(layer):
    if isinstance(layer, BasicConv2d):
        conv = layer.conv
        layer_config = {'type': 'basicConv', 'kernel_size': conv.kernel_size, 'stride': conv.stride,
                        'padding': conv.padding}  # , 'bn_args': (bn.weight, bn.bias, False, bn.momentum, bn.eps)}
    elif isinstance(layer, nn.Conv2d):
        layer_config = {'type': 'conv', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                        'padding': layer.padding}
    elif isinstance(layer, nn.MaxPool2d):
        layer_config = {'type': 'maxpool', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                        'padding': layer.padding, 'ceil_mode': layer.ceil_mode}
    elif isinstance(layer, nn.Upsample):
        layer_config = {'type': 'upsample', 'scale_factor': layer.scale_factor}
    elif layer == 'concat':
        layer_config = {'type': layer}
    else:  # only given kinds of layers
        layer_config = None
        print('This type of layer is not supported yet')

    return layer_config


def conv_output_input(output_range: tuple, layer_config=None) -> tuple:
    o_s, o_e = output_range
    kernel_size, stride, padding = layer_config['kernel_size'], layer_config['stride'], layer_config['padding']
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if padding != 0:
        padding = padding[1]
    return o_s * stride[1] - padding, (o_e - 1) * stride[1] + kernel_size[1] - padding


# from output range to input range
def output_input(output_range: tuple, layer_config=None) -> tuple:
    o_s, o_e = output_range
    layer_type = layer_config['type']
    if layer_type in ['relu', 'concat']:  # most activation layers
        return output_range
    elif layer_type == 'upsample':
        scale_factor = layer_type['scale_factor']
        return round(o_s / scale_factor), round(o_e / scale_factor)
    elif layer_type in ('conv', 'basicConv', 'maxpool'):
        kernel_size, stride, padding = layer_config['kernel_size'], layer_config['stride'], layer_config['padding']
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if padding != 0:
            padding = padding[1]
        return o_s * stride[1] - padding, (o_e - 1) * stride[1] + kernel_size[1] - padding
    else:
        print('Unknown layer type')


# generate execution units
def gen_inputDependency(model, layer_list, topology_list, output_partitions, last_array):
    # required input: from which layer, input range(in the -1 dimension )
    ids = [list() for _ in range(len(topology_list))]
    for l in topology_list:  # 当前这层
        partition = output_partitions[l]

        # if layers[l] == 'concat':
        #     last_division = tuple(len(partitions[last_layer]) - 1 for last_layer in last_array[l])
        #     required_input = (last_array[l], last_division)
        #     layer_config = {'type': 'concat'}
        #     ids[l].append((required_input, layer_config, []))
        # else:
        if l == 0:
            H = model.input_shape[-1]
        else:
            H = model.output_shapes[last_array[l][0]][-1]
        for i in range(len(partition) - 1):
            # get output range
            output_range = partition[i: i + 2]  # [o_s, o_e)
            layer = layer_list[l]
            # get corresponding input range
            if isinstance(layer, BasicConv2d):
                # type = 'conv'
                # if isinstance(layer, BasicConv2d):
                conv = layer.conv
                # bn = layer.bn
                type = 'basicConv'
                layer_config = {'type': type, 'kernel_size': conv.kernel_size, 'stride': conv.stride,
                                'padding': conv.padding}  #, 'bn_args': (bn.weight, bn.bias, False, bn.momentum, bn.eps)}
                i_s, i_e = input_range = output_input(output_range, layer_config)  # [i_s, i_e)
                if conv.padding == 0:
                    padding = (0, 0, 0, 0)
                else:
                    if i_s < 0:
                        upper_padding = -i_s
                        i_s = 0
                    else:
                        upper_padding = 0
                    if i_e > H:
                        bottom_padding = i_e - H
                        i_e = H
                    else:
                        bottom_padding = 0
                    padding = (upper_padding, bottom_padding, *conv.padding)
                    input_range = (i_s, i_e)
                layer_config['padding'] = padding

            elif isinstance(layer, nn.MaxPool2d):  # padding = 0 for most maxpool layers
                layer_config = {'type': 'maxpool', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                                'padding': layer.padding, 'ceil_mode': layer.ceil_mode}
                i_s, i_e = input_range = output_input(output_range, layer_config)  # [i_s, i_e)

                if layer.padding == 0:
                    padding = (0, 0)
                else:
                    padding = layer.padding
                # else:
                if i_s < 0:
                    upper_padding = -i_s
                    i_s = 0
                else:
                    upper_padding = 0
                if i_e > H:
                    bottom_padding = i_e - H
                    i_e = H
                else:
                    bottom_padding = 0
                padding = (upper_padding, bottom_padding, *padding)
                input_range = (i_s, i_e)
                layer_config['padding'] = padding

            elif isinstance(layer, nn.Upsample):
                layer_config = {'type': 'upsample', 'scale_factor': layer.scale_factor}
                input_range = output_input(output_range, layer_config)
            elif layer == 'concat':
                layer_config = {'type': layer}
                input_range = output_range
            elif isinstance(layer, (nn.Sigmoid, nn.ReLU, nn.Dropout, nn.BatchNorm2d)):
                layer_config = {'type': 'bijective'}
                input_range = output_range
            else:
                input_range = None
                layer_config = None

            required_input = (last_array[l], input_range)
            ids[l].append((required_input, layer_config, []))

    return ids


# for w in workload_dependency:
#     print(w)


# def gen_forwarding(input_dependency: list, topology_list: list, dependency_list: list):
#     for nl in topology_list:  # this layer
#         lasts = dependency_list[nl]
#         if len(lasts) == 1:  # only depends on one layer, can be the next_array layer of concat
#             for i, n in enumerate(input_dependency[nl]):  # execution units of this layers
#                 last_layer, input_range = n[0]
#                 last_layer = last_layer[0]
#                 last_partition = partitions[last_layer]  # partition of last_array layer's output
#                 if last_partition == 1:  # last_array layer is concat, has whole input
#                     input_dependency[last_layer][0][2].append((i, input_range))
#                 else:
#                     # formation = []  # the formation of layer's input
#                     for j in range(len(last_partition) - 1):
#                         left_max = max(last_partition[j], input_range[0])
#                         right_min = min(last_partition[j + 1], input_range[1])
#                         if left_max < right_min:  # overlap
#                             overlap = (left_max, right_min)
#                             input_dependency[last_layer][j][2].append((i, overlap))
#                             # formation.append((n, overlap))
#         else:  # concat layer
#             for last_array in lasts:
#                 last_partition = partitions[last_array]
#                 for i in range(len(last_partition) - 1):
#                     input_dependency[last_array][i][2].append((0, (partitions[last_array][i], partitions[last_array][i + 1])))


def gen_forwarding(n_device: int, input_dependency, topology_list, next_layers, output_partitions):
    for l in topology_list:  # layer nl
        nexts = next_layers[l]  # next_array layers
        partition = output_partitions[l]
        if len(nexts) == 0:  # output of final layer should send back to master
            continue
            # input_dependency[l][0][2].append(1)
        # if layers[l] == 'concat':  # layer l is concat
        #     forwarding = [[] for _ in range(n_device)]
        #     for nl in nexts:
        #         for i, eu in enumerate(input_dependency[nl]):
        #             input_range = eu[0][1]
        #             forwarding[i].append(input_range)
        #     for to_device, f in enumerate(forwarding):  # d为设备号
        #         if len(f) > 0:  # 按照转发对应的设备号去重
        #             for interval in get_set_union(f):
        #                 input_dependency[l][0][2].append((to_device, interval))
        # elif len(nexts) == 1 and layers[nexts[0]] == 'concat':  # next_array layer is concat
        #     for i in range(len(partition) - 1):
        #         input_dependency[l][i][2].append((0, (0, partition[i + 1] - partition[i]), partition[i]))
        # else:  # no concat layer between this layer and next_array layer
        for nl in nexts:
            for i in range(len(partition) - 1):
                forwarding = [[] for _ in range(n_device)]  # 未去重的forwarding
                for j, eu in enumerate(input_dependency[nl]):
                    _, input_range = eu[0]
                    overlap = get_intersection((partition[i], partition[i + 1]), input_range)
                    if overlap is not None:
                        forwarding[j].append(overlap)
                for to_device, f in enumerate(forwarding):  # d: device_id
                    if len(f) > 0:  # 按照转发对应的设备号去重
                        for interval in get_set_union(f):
                            input_dependency[l][i][2].append(
                                (to_device, (interval[0] - partition[i], interval[1] - partition[i]), partition[i]))


# for eu in workload_dependency:
#     print(eu)
