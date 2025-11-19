# distributed coded convolution
import asyncio
import socket
import sys
import threading
import time
from queue import SimpleQueue
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from coded_distributed_inference.comm_util import recv_output

from comm import recv_data, async_recv_data, send_data, connect_to_other, put_recv_data, async_send_data
from functions import output_input, generate_layerConfig, generate_layerConfigs
from util import accept_connection, load_model
from socket import create_server, AF_INET
# from worker import send_input_output, recv_input_output
from models.googlenet import BasicConv2d
import numpy as np


def coded_conv_toy():
    kernel_size = 7
    stride = 2
    padding = 3
    conv1 = nn.Conv2d(3, 64, kernel_size, stride, padding)
    # conv2 = nn.Conv2d(64, 256, kernel_size, stride, 0)
    layer_config = {'type': 'conv', 'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
    weight1 = conv1.weight
    # weight2 = conv2.weight

    output_ranges = [(0, 56), (56, 112)]
    input_ranges = [output_input(o, layer_config) for o in output_ranges]
    print(input_ranges)

    shape = (1, 3, 224, 224)
    H = shape[-1]
    input = torch.randn(shape)
    output = conv1(input)
    print(output.shape)
    padded_input = F.pad(input, [3, 3, 3, 3])
    print(padded_input.shape)
    input_range1, input_range2 = input_ranges[0], input_ranges[1]
    input1 = padded_input[..., input_range1[0] + padding: input_range1[1] + padding]
    input2 = padded_input[..., input_range2[0] + padding: input_range2[1] + padding]
    input3 = input1 + input2
    a = input1[0, 0, 0, 0]
    print(input1[0, 0, 3, 3], input2[0, 0, 3, 3], input3[0, 0, 3, 3])

    output1 = F.conv2d(input1, weight=weight1, stride=stride, padding=0)
    output2 = F.conv2d(input2, weight=weight1, stride=stride, padding=0)
    output3 = F.conv2d(input3, weight=weight1, stride=stride, padding=0)

    # output1 = F.conv2d(output1, weight=weight2, stride=stride, padding=0)
    # output2 = F.conv2d(output2, weight=weight2, stride=stride, padding=0)
    # output3 = F.conv2d(output3, weight=weight2, stride=stride, padding=0)

    output12 = output1 + output2
    # a = torch.isclose(output12, output3, rtol=0, atol=1e-16)
    concat_output = torch.concat([output1, output2], dim=-1)
    # print(torch.allclose(output, concat_output, rtol=0, atol=1e-4))
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)

    # print(output)

    equal = torch.allclose(output1 + output2, output3, rtol=0, atol=1e-5)
    print(output1[0, 0, 0, 1], output2[0, 0, 0, 1], output3[0, 0, 0, 1])
    print(equal)
    # True
    # 连续两个conv层计算也是coded也是可行的


def MDS_code():
    # mds = RSCodec()
    n = 10
    k = 7
    # G = np.vander(list(range(1, n+1)), k)
    # print(f'shape of G: {G.shape}')
    # M = np.random.random((k, 3, 5)) # 2d is ok, matrix with dimension over 2 will get problem
    # M = M.reshape((3, 15))  # all k data in M should be transformed to 1-dimensional vector for matrix multiplication
    # C = np.dot(G, M)
    # print(f'shape of M: {M.shape}')
    # print(M)
    # print(f'shape of C: {C.shape}')
    # kset = [2, 1, 0]
    # Ck = C[kset] # (3, 3, 5)
    # Gk = G[kset]
    # Ginv = np.linalg.inv(Gk)
    # M_rec = np.dot(Ginv, Ck)
    # M_rec = M_rec.reshape((3, 3, 5))
    # print(f'shape of M_rec: {M_rec.shape}')
    # print(M_rec)

    x = torch.arange(1, n + 1)
    print(x)
    G = torch.vander(x, k)  # vandermonde matrix, generation matrix
    print(f'shape of G: {G.shape}')
    mshape = (1, 512, 14, 14)
    M = torch.randn(mshape)  # input feature map (1,Ci,I,I) and output feature map (1,Co,O,O)
    M = M.view(k, -1)
    print(M.shape)
    conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)


def choose_k_MDS_conv(n: int, redundancy_rate):  # choose k with n that reach the minimal redundancy rate
    assert n > 1
    r = int(n * redundancy_rate + 0.5)
    return n - r


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


def coded_conv_MDS(x, input_shape, output_shape, n, redundancy_rate,
                   conv_layer: nn.Conv2d):  # (n, k) MDS, n divisible results
    layer_config = generate_layerConfig(conv_layer)
    in_dim = conv_layer.in_channels
    out_dim = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size[-1]
    stride = conv_layer.stride[-1]
    padding = [conv_layer.padding[1] for i in range(2)] + [conv_layer.padding[0] for i in range(2)]
    correct_output = conv_layer(x)
    x = F.pad(x, padding)  # 提前进行padding
    layer_config['padding'] = 0
    I = input_shape[-1]
    O = output_shape[-1]
    # choose k with n and redundancy_rate
    k = choose_k_MDS_conv(n, redundancy_rate)
    print(f'(n,k): {(n, k)}')
    # split feature map to n workers
    worker_task = O // k  # master_task < worker_task
    master_task = O % k
    output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
    print(output_ranges)
    input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
    print(input_ranges)
    split_inputs = torch.concat([x[..., s:e] for s, e in input_ranges], dim=0)  # k*(1,C,I,I/k) -> (k,C,I,I/k)
    print(f'split_inputs shape: {split_inputs.shape}')
    split_inputs = split_inputs.view(k, -1)  # (k,C*I*I/k)
    x = torch.arange(1, n + 1)
    G = torch.vander(x, k).float()  # vandermonde matrix, any generation matrix
    print(G.shape)
    print(type(G), type(split_inputs))
    coded_inputs = torch.matmul(G, split_inputs)
    print(f'coded_inputs shape: {coded_inputs.shape}')
    coded_inputs = coded_inputs.view(n, 1, in_dim, I + padding[2] + padding[3],
                                     kernel_size + stride * (worker_task - 1))  # (n,1*C*I*I/k)
    coded_inputs = [coded_inputs[i] for i in range(n)]  # (n,1,C,I,I/k), executing in distributed manner
    print(coded_inputs[0].shape)
    coded_outputs = [F.conv2d(i, conv_layer.weight, stride=conv_layer.stride) for i in coded_inputs]  # k*(1,Co,O,O/k)
    print(coded_outputs[0].shape)
    kset = [1, 2, 3]
    Gk = G[kset]
    print(f'Shape of Gk: {Gk.shape}')
    Ginv = torch.linalg.inv(Gk)
    Ck = torch.concat([coded_outputs[i] for i in kset], dim=0)  # (k,Co,O,O/k)
    Ck = Ck.view(k, -1)  # (k,Co*O*O/k)
    decoded_outputs = torch.matmul(Ginv, Ck)
    decoded_outputs = decoded_outputs.view(k, 1, out_dim, O, worker_task)
    decoded_outputs = [decoded_outputs[i] for i in range(k)]
    decoded_output = torch.concat(decoded_outputs, dim=-1)
    print(decoded_output.shape)
    # correct_outputaaaaaaa = correct_output[..., :12]
    print(torch.allclose(decoded_output, correct_output[..., :k * worker_task], atol=0.02))


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
    print(f'split_inputs shape: {split_inputs.shape}')  # k*(1,C,I,I/k) -> (k,C,I,I/k)
    split_inputs = split_inputs.view(k, -1)  # (k,C*I*I/k)
    coded_inputs = torch.matmul(G, split_inputs)  # (n,C*I*I/k)
    coded_inputs = coded_inputs.view(n, *input_shape)  # (n,C,I,I/k)
    coded_inputs = [coded_inputs[i].clone().detach() for i in range(n)]
    return coded_inputs


def decode_conv_MDS(coded_outputs: list[torch.Tensor], n: int, k: int, G: torch.Tensor, kset: list[int]):
    assert len(kset) == k
    output_shape = coded_outputs[0].shape
    Gk = G[kset]
    # Ginv = torch.linalg.inv(Gk)
    Ginv = Gk
    # Ck = torch.concat([coded_outputs[i] for i in kset], dim=0)  # k*(1,Co,O,O/k) -> (k,Co,O,O/k)
    Ck = torch.concat(coded_outputs, dim=0)  # k*(1,Co,O,O/k) -> (k,Co,O,O/k)
    Ck = Ck.view(k, -1)  # (k,Co,O,O/k) -> (k,Co*O*O/k)
    decoded_outputs = torch.matmul(Ginv, Ck)
    decoded_outputs = decoded_outputs.view(k, *output_shape)
    decoded_outputs = [decoded_outputs[i] for i in range(k)]
    # a = nn.BatchNorm2d()
    return decoded_outputs


# def encode_conv_LT(xs: list[torch.Tensor], n: int, k: int, G: torch.Tensor)


# def async_recv_k_outputs(socket_list: list[socket.socket], k):  #
#     loop2 = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop2)
#     while True:
#         read_ready, _, _ = select.select(socket_list, [], [], 0.2)
#         if len(read_ready) > k:
#             # 读取结果并准备解码

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


def distributed_coded_convolution_MDS(x: torch.Tensor, CNN_model: nn.Module):
    # def coded_vgg16_MDS():

    # build connections
    self_ip = '192.168.1.155'
    __port__ = 49999
    server_socket = create_server((self_ip, __port__), family=AF_INET)
    # connect_to_other(ip_list, __port__, )
    server_socket.settimeout(8)
    worker_sockets = []
    stop_thread = False
    available_workers = []
    print(f'Master ip is {self_ip}, recv connections from workers...')
    recv_thread = threading.Thread(target=accept_connection,
                                   args=[server_socket, available_workers, lambda: stop_thread])
    recv_thread.start()
    time.sleep(10)  # waiting for 10s
    stop_thread = True
    recv_thread.join()

    loop = asyncio.get_event_loop()  # for async sending
    num_ip = []
    for conn, addr in available_workers:
        # conn.settimeout(5)
        # ip, port = addr
        if addr not in num_ip:  # first time to meet the addr
            num_ip.append(addr)
            worker_sockets.append(conn)
        else:
            conn.close()
    n_workers = len(num_ip)
    print(f'Recv connections from {n_workers} workers')

    layers = CNN_model.features
    layer_configs = generate_layerConfigs(layers)
    layer_indexes = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # vgg16
    # layer_indexes = [0, 2, 4, 5, 6]  # alexnet
    send_tasks = [async_send_data(worker_sockets[i], (i, layer_configs, layer_indexes)) for i in range(n_workers)]
    loop.run_until_complete(asyncio.gather(*send_tasks))

    # tesing
    # worker_socket1 = worker_sockets[0]
    # x = torch.randn((1, 64, 224, 224))
    # _ = recv_data(worker_socket1)

    # serial
    # for _ in range(3):
    #     consumption = time.time()
    #     send_data(worker_socket1, (1, x))
    #     _ = recv_data(worker_socket1)
    #     consumption = time.time() - consumption
    #     print(f'Finish in {consumption}s')

    # parallel
    # send_time = []
    # recv_time = []
    # send_thread = threading.Thread(target=send_data_thread, args=[1, x, worker_socket1, send_time])
    # send_thread.start()
    # recv_thread = threading.Thread(target=recv_data_thread, args=[worker_socket1, recv_time])
    # recv_thread.start()
    # send_thread.join()
    # recv_thread.join()
    # layer_num, data = recv_data(worker_socket1)
    # print(send_time)
    # print(recv_time)
    # consumptions = [recv_time[i] - send_time[i] for i in range(3)]
    # print(consumptions)
    # print(recv_time[-1] - send_time[0])

    # assert isinstance(worker_socket1, socket.socket)
    # worker_socket1.close()

    recv_queue = SimpleQueue()  # 接收workers确认与所有其他workers完成建立连接的消息
    for conn in worker_sockets:
        recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
        recv_thread.start()
        recv_thread.join()
    if n_workers != 0 and recv_queue.qsize() == n_workers:
        print('Successfully initialized!')
    else:
        print('Fail to initialize')

    # start recv threads
    recv_list = []
    thread_is_working = True
    recv_threads = [
        threading.Thread(target=recv_output, args=[worker_sockets[i], recv_list, lambda: thread_is_working]) for i in
        range(n_workers)]
    for t in recv_threads:
        t.start()
    time.sleep(1)

    # start time of the task
    times = []

    coded = True
    n = n_workers
    if coded:
        ks = range(n // 2, n)
    else:
        ks = [n]

    for k in ks:

        G = torch.vander(torch.arange(1, n + 1), k).float()  # vandermonde matrix, any generation matrix

        output_shapes = CNN_model.output_shapes
        # 逐层处理，提取出其中计算量较大的线性计算，编码后发送给workers
        for layer_num, layer in enumerate(CNN_model.features):
            recv_list.clear()
            times.append(time.time())
            if isinstance(layer, BasicConv2d):  # for vgg16, only conv and linear layers are linear computation
                input_shape = x.shape
                output_shape = output_shapes[layer_num]
                # I = input_shape[-1]
                O = output_shape[-1]
                layer_config = layer_configs[layer_num]
                # linear = []
                # nonLinear = []
                # 将layer中的线性计算与非线性计算拆分出来，并将线性计算编码后发送给workers
                # if isinstance(layer, BasicConv2d):
                #     linear.append('conv')
                #     linear.append('bn')
                #     nonLinear.append('relu')
                # elif isinstance(layer, nn.MaxPool2d):
                #     nonLinear.append('maxpool')
                # else:
                #     print('Do not support this type of layer!')

                padding = layer_config['padding']
                padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
                x_padded = F.pad(x, padding)
                layer_config['padding'] = 0

                worker_task = O // k  # master_task < worker_task
                master_task = O % k

                output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
                print(output_ranges)
                input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
                xs = [x_padded[..., s:e] for s, e in input_ranges]  # k*(1,C,I,I/k)
                if coded:  # encoding input
                    encoded_inputs = encode_conv_MDS(xs, n, k, G)
                    datas = [(layer_num, encoded_inputs[i]) for i in range(n)]
                else:
                    datas = [(layer_num, xs[i]) for i in range(k)]

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

                if coded:  # decoding
                    kset = [t[0] for t in recv_list[:k]]
                    coded_outputs = [t[1] for t in recv_list[:k]]
                    outputs = decode_conv_MDS(coded_outputs, n, k, G, kset)
                else:
                    recv_list.sort(key=lambda t: t[0])
                    outputs = [data[1] for data in recv_list]

                if master_task > 0:
                    outputs.append(master_output)
                print([item.shape for item in outputs])
                x = torch.concat(outputs, dim=-1)
                x = F.relu(x, inplace=True)
            else:
                x = layer(x)
            times.append(time.time())
        thread_is_working = False
        print(f'{times[-1] - times[0]}s in total.')
        print(x.shape)
        print(times)
    # thread_is_working = False
    # for t in recv_threads:
    #     t.join()
    # send_tasks = [async_send_data(soc, (-1, 0)) for soc in worker_sockets]
    # loop.run_until_complete(asyncio.gather(*send_tasks))
    # recv_tasks = [async_recv_data(soc) for soc in worker_sockets]

    # times_workers = loop.run_until_complete(asyncio.gather(*recv_tasks))
    while len(recv_list[-1]) < n:
        time.sleep(1e-2)
    avgs = np.asarray(recv_list[-1]).mean(axis=0)
    draw_times(times, avgs, layer_indexes)
    return times


# def recv_output(index: int, recv_socket: socket.socket, recv_list: list, thread_is_working: bool):
#     while thread_is_working:
#         layer_num, data = recv_data(recv_socket)
#         if layer_num == -1:
#             recv_list[-1].append(data)
#         else:
#             recv_list[layer_num].append((index, data))
#             time.sleep(1e-3)


def draw_times(times_master, times_worker, layer_indexes):
    cnt = len(times_master) // 2
    durations = np.asarray([times_master[2*i + 1] - times_master[2*i] for i in range(cnt)])[layer_indexes]
    print(durations)
    labels = list(range(len(durations)))
    print(times_worker)
    comp = np.asarray(times_worker)
    comm = durations - comp
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.bar(labels, comp, width, label='comp')
    ax.bar(labels, comm, width, bottom=comp, label='comm')
    ax.set_ylabel('Latency')
    ax.set_xlabel('Layers')
    ax.legend()
    # ax.text(.87, -.08, '\nVisualization by DataCharm', transform=ax.transAxes,
    #         ha='center', va='center', fontsize=5, color='black', fontweight='bold', family='Roboto Mono')
    # plt.savefig(r'F:\DataCharm\SCI paper plots\sci_bar_guanwang', width=5, height=3,
    #             dpi=900, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # conv_layer = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    # padding = [conv_layer.padding[1] for i in range(2)] + [conv_layer.padding[0] for i in range(2)]
    # print(padding)
    # x = torch.randn(1, 512, 14, 14)
    # output_shape = input_shape = (1, 512, 14, 14)
    # n = 4
    # redundancy_rate = 0.2
    # coded_conv_MDS(x, input_shape, output_shape, n, redundancy_rate, conv_layer)
    model = load_model('vgg16')
    x = torch.randn(model.input_shape)
    distributed_coded_convolution_MDS(x, model)
