import argparse
import asyncio
import socket
import sys
import threading
import time
from queue import SimpleQueue
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from comm import recv_data, async_recv_data, send_data, connect_to_other, put_recv_data, async_send_data
from functions import output_input, generate_layerConfig, generate_layerConfigs
from util import accept_connection, load_model
from socket import create_server, AF_INET
# from worker import send_input_output, recv_input_output
from models.googlenet import BasicConv2d
import numpy as np


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


def MDS_conv_optimal_k(perf_params: tuple, task_params: tuple, n: int):
    assert n > 2
    assert len(perf_params) == 5  # mu_m, mu_cmp, theta_cmp, mu_tr, theta_tr
    assert len(task_params) == 8  # C_i, H_i, W_i, C_o, H_o, W_o, kernel, stride
    mu_m, mu_cmp, theta_cmp, mu_tr, theta_tr = perf_params
    C_i, H_i, W_i, C_o, H_o, W_o, kernel, stride = task_params
    a = 2 * (np.prod([n, C_i, H_i, kernel - stride]) + np.prod([C_o, H_o, W_o]))
    b = 4 * (np.prod([C_i, H_i, stride]) + C_o * H_o) * W_o
    c = np.prod([2, C_o, H_o, C_i, kernel, kernel, W_o])
    d = np.prod([n, C_i, H_i, W_o, stride])
    A = a / mu_m
    B = b * theta_tr + c * theta_cmp - d / mu_m
    C = b / mu_tr + c / mu_cmp
    L_k = [A*k + B/k + C/k * np.log(n/(n-k)) for k in range(2, n)]
    return np.argmax(L_k)


def encode_conv_MDS(xs: list[torch.Tensor], n: int, k: int,
                    G: torch.Tensor):  # generate n encoded inputs from k inputs of same size
    '''
    :param xs: partitioned inputs in k pieces
    :param n: n in MDS code
    :param k: k in MDS code
    :param G: generation matrix, vandermonde matrix
    :param shape: shape of partitioned inputs
    :return: encoded inputs
    '''
    input_shape = xs[0].shape
    split_inputs = torch.concat(xs, dim=0)
    # print(f'split_inputs shape: {split_inputs.shape}')  # k*(1,C,I,I/k) -> (k,C,I,I/k)
    split_inputs = split_inputs.view(k, -1)  # (k,C*I*I/k)
    coded_inputs = torch.matmul(G, split_inputs)  # (n,C*I*I/k)
    coded_inputs = coded_inputs.view(n, *input_shape)  # (n,C,I,I/k)
    coded_inputs = [coded_inputs[i].clone().detach() for i in range(n)]
    return coded_inputs


def decode_conv_MDS(coded_outputs: list[torch.Tensor], n: int, k: int, G: torch.Tensor, kset: list[int]):
    assert len(kset) == k
    output_shape = coded_outputs[0].shape
    Gk = G[kset]
    Ginv = torch.linalg.inv(Gk)
    # Ck = torch.concat([coded_outputs[i] for i in kset], dim=0)  # k*(1,Co,O,O/k) -> (k,Co,O,O/k)
    Ck = torch.concat(coded_outputs, dim=0)  # k*(1,Co,O,O/k) -> (k,Co,O,O/k)
    Ck = Ck.view(k, -1)  # (k,Co,O,O/k) -> (k,Co*O*O/k)
    decoded_outputs = torch.matmul(Ginv, Ck)
    decoded_outputs = decoded_outputs.view(k, *output_shape)
    decoded_outputs = [decoded_outputs[i] for i in range(k)]
    # a = nn.BatchNorm2d()
    return decoded_outputs


def recv_data_thread(recv_socket: socket.socket, time_list: list):
    for i in range(3):
        _ = recv_data(recv_socket)
        print(f'recv No.{i} task')
        time_list.append(time.time())
        time.sleep(1e-3)


def send_data_thread(layer_num, data, send_socket: socket.socket, time_list: list):
    for i in range(3):
        time_list.append(time.time())
        send_data(send_socket, layer_num, data)
        print(f'send No.{i} task')
        time.sleep(1e-3)


def recv_output(index: int, recv_socket: socket.socket, recv_list: list, thread_is_working: bool):
    try:
        while thread_is_working:
            data = recv_data(recv_socket)
            recv_list.append((index, data))
            time.sleep(1e-3)
    except Exception as e:
        print(e)


def distributed_coded_convolution(master_ip: str, CNN_model: nn.Module):
    server_socket = create_server((master_ip, __port__), family=AF_INET)
    server_socket.settimeout(8)
    worker_sockets = []
    stop_thread = False
    available_workers = []
    print(f'Master ip is {master_ip}, recv connections from workers...')
    recv_thread = threading.Thread(target=accept_connection,
                                   args=[server_socket, available_workers, lambda: stop_thread])
    recv_thread.start()
    time.sleep(10)  # waiting for 10s
    stop_thread = True
    recv_thread.join()

    n = 10

    loop = asyncio.get_event_loop()  # for async sending
    num_ip = []
    for conn, addr in available_workers:
        if addr not in num_ip:  # first time to meet the addr
            num_ip.append(addr)
            worker_sockets.append(conn)
        else:
            conn.close()
    n_workers = len(num_ip)
    print(f'Recv connections from {n_workers} workers')
    if n_workers < n:
        print(f'Fail to receive {n} workers!')
        sys.exit(0)

    layers = CNN_model.features
    layer_configs = generate_layerConfigs(layers)
    layer_indexes = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # vgg16
    layers_to_test = [1]
    repetition = 10
    task_description = (layer_configs, layer_indexes, layers_to_test, repetition, n)
    send_tasks = [async_send_data(worker_sockets[i], task_description) for i in range(n_workers)]
    loop.run_until_complete(asyncio.gather(*send_tasks))

    # start recv threads
    recv_list = []
    thread_is_working = True
    recv_threads = [
        threading.Thread(target=recv_output, args=[i, worker_sockets[i], recv_list, lambda: thread_is_working]) for i in
        range(n_workers)]
    for t in recv_threads:
        t.start()
    time.sleep(1)

    Gs = [torch.vander(torch.arange(1, n+1), k).float() for k in range(2, n)]  # vandermonde matrix, any generation matrix
    for layer_num in layers_to_test:
        print(f'Test for layer={layer_num}')
        layer = layers[layer_num]
        if layer_num == 0:
            input_shape = model.input_shape
        else:
            input_shape = model.output_shapes[layer_num - 1]
        x = torch.randn(input_shape)
        output_shape = model.output_shapes[layer_num]
        O = output_shape[-1]
        layer_config = layer_configs[layer_num]
        padding = layer_config['padding']
        padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
        x_padded = F.pad(x, padding)
        layer_config['padding'] = 0
        for k in range(2, n):
            print(f'  Test for k={k}')
            G = Gs[k-2]
            worker_task = O // k
            master_task = O % k
            output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
            input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
            xs = [x_padded[..., s:e] for s, e in input_ranges]  # k*(1,C,I,I/k)
            records = []
            for r in tqdm(range(repetition)):
                recv_list.clear()
                start = time.time()
                encoded_inputs = encode_conv_MDS(xs, n, k, G)
                datas = [encoded_inputs[i] for i in range(n)]
                # send task (layer_num and coded inputs) to n workers asynchronously
                send_tasks = [async_send_data(worker_sockets[i], datas[i]) for i in range(n_workers)]
                loop.run_until_complete(asyncio.gather(*send_tasks))

                # finish master task if necessary
                if master_task > 0:
                    conv = layer.conv
                    m_output_range = (O - master_task, O)
                    i_s, i_e = output_input(m_output_range, layer_config)
                    split_input = x_padded[..., i_s:i_e]
                    master_output = F.conv2d(split_input, conv.weight, stride=conv.stride, padding=0)

                # receive coded outputs from k workers synchronously
                while len(recv_list) < k:
                    time.sleep(1e-3)

                kset = [t[0] for t in recv_list[:k]]
                coded_outputs = [t[1] for t in recv_list[:k]]
                outputs = decode_conv_MDS(coded_outputs, n, k, G, kset)

                end = time.time()
                consumption = end - start
                records.append(consumption)

                while len(recv_list) < n:
                    time.sleep(1e-1)
            print(f'k={k} records: {records}')

    thread_is_working = False



if __name__ == '__main__':
    __port__ = 49999
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master', type=str, required=True)
    args = parser.parse_args()
    master_ip = args.master
    # master_ip = '192.168.1.148'

    model = load_model('vgg16')
    # x = torch.randn(model.input_shape)
    distributed_coded_convolution(master_ip, model)