# determine k* by layer according to given system parameters and conv parameters
import sys

import torch
from util import load_model, load_object
from functions import translate_next_array, next_to_last, output_input, generate_layerConfig
from models.googlenet import BasicConv2d
from scipy.optimize import Bounds, minimize
import numpy as np
import torch.nn as nn


def objective_function(k, *args):
    # k = x
    n, h1, h2, h3, h4 = args
    return h1 * k + h2 / k + h3 / k * np.log(n / (n - k)) + h4 * n * np.log(n / (n - k))


def solve_k_scipy(n, system_params, conv_params):  # 目前是nk
    B, C_i, H_i, W_i, C_o, H_o, W_o, kernel_size, stride, padding = conv_params
    mu_m, theta_m, mu_rec, theta_rec, mu_cmp, theta_cmp, mu_sen, theta_sen = system_params

    Iov = np.prod([C_i, H_i, kernel_size - stride])
    IW = np.prod([C_i, H_i, W_o, stride])
    O = np.prod([C_o, H_o, W_o])
    Nc = np.prod([2.0, C_o, H_o, C_i, kernel_size, kernel_size, W_o])  # 这个用int32表示会溢出了

    # h_1 = np.prod(2*(1/mu_m+theta_m)*(n*Iov+O))
    # h_2 = np.prod(4.0, IW, theta_rec) + np.prod(4.0, O, theta_sen) + np.prod(Nc, theta_cmp)
    # h_3 = np.prod(4.0, IW, 1/mu_rec) + np.prod(4.0, O, 1/mu_sen) + Nc / mu_cmp
    # h_4 = np.prod(4.0, Iov, mu_rec)
    h_1 = 2.0 * (1 / mu_m + theta_m) * (n * Iov + O)
    h_2 = 4.0 * IW * theta_rec + 4.0 * O * theta_sen + Nc * theta_cmp
    h_3 = 4.0 * IW / mu_rec + 4.0 * O / mu_sen + Nc / mu_cmp
    h_4 = 4.0 * Iov / mu_rec
    print(h_2/h_3)

    # print(h_1, h_2, h_3, h_4)

    # constraints = ({'type': 'ineq', 'fun': lambda x: x[0]}) # n >= k
    bounds = Bounds([2], [n])

    initial_guess = [n / 2]
    result = minimize(objective_function, initial_guess, args=(n, h_1, h_2, h_3, h_4), constraints=[], bounds=bounds)
    # print(f'k*={result.x}   T*={result.fun}')

    if not result.success:
        print(result.message)
    return (result.fun, *result.x)


model_name = 'resnet'
model = load_model(model_name)

layers = model.layers
next_array = model.next
translate_next_array(next_array)
last_array = next_to_last(next_array)

N_tr = 4 * np.prod((1, 10, 224, 224))

# mu_m, theta_m = 1375069010.349571, 1.1414616840348712e-09
mu_m, theta_m = 16904242506.376558, 1.5748831477819342e-09

# mu_sen, theta_sen = (664446.2862308504, 7.829987338890e-07)
# 0.62175678 0.43279159
# 1.16349416 0.71954717
# 62.69922425  1.68099049


lambdas_tr = np.arange(0, 11, 2)
mu_theta_recv_pairs = [
    (62.69922425, 1.68099049),
    (17.11664517, 1.62201568),
    (2.31140114, 1.09813618),
    (1.19833621, 0.73893584),
    (0.70131561, 0.37896212),
    (0.61142539, 0.27758693)
]

for mu_tr, theta_tr in mu_theta_recv_pairs:
    print(1 / mu_tr + theta_tr)

vgg16_conv_latency_lambdas = load_object('record/vgg16_layers_latency_lambdas.rec')
resnet_conv_latency_lambdas = load_object('record/resnet_layers_latency_lambdas.rec')
vgg16_layer_latency_local = load_object('vgg16_layer_latecy_local.rec')
resnet_layer_latency_local = load_object('resnet_layer_latecy_local.rec')

mu_sen, theta_sen = (0.61142539 * N_tr, 0.27758693 / N_tr)
# mu_sen, theta_sen = (16.47243177 * N_tr, 1.56259472 / N_tr)
# mu_rec, theta_rec = (1196567.3770074411, 3.3175139905527e-07)
mu_rec, theta_rec = (133.28784691 * N_tr, 1.68944582 / N_tr)
mu_cmp, theta_cmp = (16904242506.376558, 1.5748831477819342e-09)

system_params = mu_m, theta_m, mu_rec, theta_rec, mu_cmp, theta_cmp, mu_sen, theta_sen

B = 1


def get_input_output_shape(layer_id, input_shape, output_shapes, last_array):
    if layer_id == 0:
        return (1, *input_shape), output_shapes[layer_id]
    last_layer = last_array[layer_id][0]
    return output_shapes[last_layer], output_shapes[layer_id]


vgg16_conv_idxes = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
resnet_conv_idxes = [0, 3, 6, 10, 13, 17, 20, 22, 26, 29, 33, 36, 38, 42, 45, 49, 52, 54, 58, 61]
vgg_distributed_conv_idxes = [1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
resnet_distributed_conv_idxes = [3, 6, 10, 13, 17, 20, 26, 29, 33, 36, 42, 45, 49, 52, 58, 61]

if model_name == 'vgg16':
    conv_idxes = vgg16_conv_idxes
    distributed_conv_idxes = vgg_distributed_conv_idxes
    conv_latency_lambdas = vgg16_conv_latency_lambdas
    layer_latency_local = vgg16_layer_latency_local
else:
    conv_idxes = resnet_conv_idxes
    distributed_conv_idxes = resnet_distributed_conv_idxes
    conv_latency_lambdas = resnet_conv_latency_lambdas
    layer_latency_local = resnet_layer_latency_local

n = 10

layer_latency_estimate = []


def gen_conv_params(conv_layer: nn.Conv2d, input_shape: tuple, output_shape: tuple):
    if isinstance(conv_layer, BasicConv2d):
        conv_layer = conv_layer.conv
    _, C_i, H_i, W_i = input_shape
    _, C_o, H_o, W_o = output_shape
    kernel_size = conv_layer.kernel_size
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    stride = conv_layer.stride
    if isinstance(stride, tuple):
        stride = stride[0]
    padding = conv_layer.padding
    # input_shape, output_shape = get_input_output_shape(distributed_conv_idx, model.input_shape, model.output_shapes,
    #                                                    last_array)
    conv_params = (B, C_i, H_i, W_i, C_o, H_o, W_o, kernel_size, stride, padding)
    return conv_params


def conv_determined_k(conv_layer: nn.Conv2d | BasicConv2d, input_shape: tuple, output_shape: tuple,
                      system_params: tuple):
    '''
    the previous three params are used to generate conv_params, together with system_params to determine k*
    '''
    conv_params = gen_conv_params(conv_layer, input_shape, output_shape)
    objective, k = solve_k_scipy(n, system_params, conv_params)
    return objective, k


lambdas_latency = []
lambda_determined_kss = []
for i, lambda_tr in enumerate(lambdas_tr):
    print(f'--------lambda={lambda_tr}---------')
    mu_sen, theta_sen = mu_theta_recv_pairs[i]  # updated transmission params
    mu_sen *= N_tr
    theta_sen /= N_tr
    system_params = mu_m, theta_m, mu_rec, theta_rec, mu_cmp, theta_cmp, mu_sen, theta_sen
    print(system_params)
    sum_latency = 0
    conv_latency_lambda = conv_latency_lambdas[i]
    real_distributed_conv_layers = []
    determined_ks = []
    for layer_idx in range(len(model.features)):
        layer = model.layers[layer_idx]
        local_layer_latency = layer_latency_local[layer_idx]
        # print(f'layer {layer_idx}: ', end='')
        if layer_idx in distributed_conv_idxes:  # 说明该层通过分布式执行可能获得加速，因此进行k决策
            relative_conv_idx = conv_idxes.index(layer_idx)  # 当前层在所有conv层对应的序号
            layer_latency_distributed = conv_latency_lambda[relative_conv_idx]
            if min(layer_latency_distributed) < local_layer_latency:
                input_shape, output_shape = get_input_output_shape(layer_idx, model.input_shape,
                                                                   model.output_shapes,
                                                                   last_array)
                objective, k = conv_determined_k(layer, input_shape, output_shape, system_params)
                determined_k = min(n - 1, int(k + 1))
                r = n - determined_k
                layer_latency_determined = layer_latency_distributed[-r]
                determined_ks.append(determined_k)
                real_distributed_conv_layers.append(layer_idx)
                # print(f'{layer_latency_determined:.7f}s: distributed execution k={determined_k}')
            else:
                layer_latency_determined = local_layer_latency
                # print(f'{layer_latency_determined:.7f}s: local execution')
            sum_latency += layer_latency_determined
        else:
            # print(f'{local_layer_latency:.7f}s: local execution')
            sum_latency += local_layer_latency
    lambdas_latency.append(sum_latency)
    print('real distributed conv layers', real_distributed_conv_layers)
    print('determined ks', determined_ks)
    lambda_determined_kss.append(determined_ks)
print(lambdas_latency)

vgg16_lambda_diff = [0.10327186539276312, 0.10217877421300159, 0.07376974095179101, 0.11070911732396205,
                     0.4964202494800798, 0.5153671025275663]
resnet_lambda_diff = [0.4873897018438704, 0.3492181570309434, 0.15263996412002356, 0.9798541391077222,
                      0.37145333610488507, 1.3072533065479632]

vgg16_optimal_ks = [
[9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8],
[9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8],
[9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8],
[9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8],
[9, 8, 9, 8, 9, 9, 9, 8, 8, 9, 8, 8],
[9, 8, 8, 8, 9, 9, 9, 8, 8, 8, 8, 8]
]

resnet_optimal_ks = [
[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 9],
[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 8, 8],
[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 8, 8, 8, 8],
[9, 9, 9, 8, 9, 9, 9, 7, 9, 9, 9, 9, 9, 8, 8, 8],
[8, 9, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8],
[8, 9, 7, 8, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8]
]
optimal_kss = vgg16_optimal_ks if model_name == 'vgg16' else resnet_optimal_ks
for determined_ks, optimal_ks in zip(lambda_determined_kss, optimal_kss):
    assert len(determined_ks) == len(optimal_ks)
    determined_ks = np.asarray(determined_ks)
    optimal_ks = np.asarray(optimal_ks)
    k_diff = np.abs(determined_ks - optimal_ks)
    print(f'k_diff: max={np.max(k_diff)}, mean={np.mean(k_diff)}, sum={np.sum(k_diff)}')

#vgg16 lambda从0到1
# k_diff: max = 1, mean = 0.5833333333333334, sum = 7
# k_diff: max = 1, mean = 0.4166666666666667, sum = 5
# k_diff: max = 1, mean = 0.5, sum = 6
# k_diff: max = 1, mean = 0.5, sum = 6
# k_diff: max = 1, mean = 0.4166666666666667, sum = 5
# k_diff: max = 1, mean = 0.5, sum = 6
#resnet lambda从0到1
# k_diff: max=1, mean=0.375, sum=6
# k_diff: max=1, mean=0.1875, sum=3
# k_diff: max=1, mean=0.3125, sum=5
# k_diff: max=2, mean=0.375, sum=6
# k_diff: max=1, mean=0.375, sum=6
# k_diff: max=1, mean=0.6, sum=9


# for distributed_conv_idx in distributed_conv_idxes:  # 只从可能的需要分布式conv的层进行决策
#     print(f'layer {distributed_conv_idx} ', end='')
#     layer = layers[distributed_conv_idx]
#
#     input_shape, output_shape = get_input_output_shape(distributed_conv_idx, model.input_shape, model.output_shapes,
#                                                        last_array)
#
#     objective, k = conv_determined_k(layer, input_shape, output_shape, system_params)
#
#     round_k = int(k+1)
#     r = n - k
#
#     print(f't*={objective}, k*={min(n - 1, int(k+1))}')
#
#     # _, C_i, H_i, W_i = input_shape
#     # _, C_o, H_o, W_o = output_shape
#     # kernel_size = layer.kernel_size
#     # if isinstance(kernel_size, tuple):
#     #     kernel_size = kernel_size[0]
#     # stride = layer.stride
#     # if isinstance(stride, tuple):
#     #     stride = stride[0]
#     # padding = layer.padding
#     #
#     # # layer_config = generate_layerConfig(layer)
#     # # x = torch.randn(input_shape)
#     #
#     # conv_params = (B, C_i, H_i, W_i, C_o, H_o, W_o, kernel_size, stride, padding)
#     #
#     # objective, k = solve_k_scipy(n, system_params, conv_params)
#     layer_latency_estimate.append(objective)
#
# print('Estimated sum:', sum(layer_latency_estimate))
