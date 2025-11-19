# 实验测试uncoded和coded方法对于单个conv计算的比较
# master
import argparse
import asyncio
import pickle
import sys
import threading
import time
from queue import SimpleQueue
from socket import create_server, AF_INET

from comm import put_recv_data, async_send_data
from comm_util import *
from coding_util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy.random as random
from functions import output_input, generate_layerConfigs, conv_output_input
from util import accept_connection, load_model

### 没记错python是单线程吧？那就不用考虑线程安全了
### 确保收到的数据能拼成需要的数据，至少确定形状是对的

class n_recv_k_list:
    def __init__(self, n, k):
        self.recv_list = []
        self.recv_cnt = 0
        self.n = n
        self.k = k
        self.full_time = None

    def append(self, item: tuple):
        self.recv_cnt += 1
        if self.recv_cnt == self.n:
            self.full_time = time.time()
        # stop when recv k results, while redundant results and 'fail' are still considered in recv_cnt
        if item[1] != 'fail' and len(self.recv_list) < self.k:
            self.recv_list.append(item)


    # def full_results_with_full_time(self):
    #     return self.recv_list, self.full_time

    def clear(self):
        self.recv_list.clear()
        self.recv_cnt = 0

    def enough(self):
        return len(self.recv_list) >= self.k

    def full(self):
        return self.recv_cnt >= self.n


def test_distributed_coded_convolution_k(master_ip: str, CNN_model: nn.Module, repetition):
    '''
    Test for coded convolution with different k(s)
    Correspond to coded_distributed_inference/worker.py
    :param master_ip: master_ip
    :param CNN_model: model instance
    :return: None
    '''
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
    fail_test = False  # 手动fail or 概率fail
    fail_num = 2
    # required worker numbers
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

    # CNN and conv task configuration
    layers = CNN_model.features
    layer_configs = generate_layerConfigs(layers)
    layer_indexes = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # vgg16
    layers_to_test = [1]

    # repetition = 10
    task_description = layer_configs, layer_indexes
    send_tasks = [async_send_data(worker_sockets[i], (i, *task_description)) for i in range(n_workers)]
    loop.run_until_complete(asyncio.gather(*send_tasks))

    recv_queue = SimpleQueue()  # recv 'ready' from workers
    for conn in worker_sockets:
        recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
        recv_thread.start()
        recv_thread.join()
    if n_workers != 0 and recv_queue.qsize() == n_workers:
        print('Successfully initialized!')
    else:
        print('Fail to initialize')

    # start recv threads
    recv_list = n_recv_k_list(n, 5)
    thread_is_working = True
    recv_threads = [
        threading.Thread(target=recv_output,
                         args=[worker_sockets[i], recv_list, lambda: thread_is_working]) for i in
        range(n_workers)]
    for t in recv_threads:
        t.start()
    time.sleep(1)
    # vandermonde matrix, any generation matrix: from (k=2 to k=9)
    Gs = [torch.vander(torch.arange(1, n + 1), k).float() for k in
          range(2, n)]

    records_layers = []

    for layer_num in layers_to_test:
        print(f'Test for layer={layer_num}')
        layer = layers[layer_num]
        # get input shape
        input_shape = model.input_shape if layer_num == 0 else model.output_shapes[layer_num - 1]
        x = torch.randn(input_shape)
        output_shape = model.output_shapes[layer_num]
        O = output_shape[-1]
        layer_config = layer_configs[layer_num]
        padding = layer_config['padding']
        padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
        x_padded = F.pad(x, padding)
        layer_config['padding'] = 0
        records = []
        for k in [5, 9]:  # test k from 2 to n-1, and extract k = 5 and optimal k*
            print(f'  Test for k={k}')
            recv_list.k = k
            G = Gs[k - 2]
            # print('Shape of G:', G.shape)
            worker_task = O // k
            master_task = O % k
            output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
            input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
            xs = [x_padded[..., s:e] for s, e in input_ranges]  # k*(1,C,I,I/k)
            test_encoded_inputs = encode_conv_MDS(xs, n, k, G)[0]
            print('Send_size:', len(pickle.dumps(test_encoded_inputs[0])))

            records1 = []
            # records2 = []
            for r in tqdm(range(repetition)):
                recv_list.clear()

                if fail_test and fail_num > 0:  # 先手动fail
                    fail_tasks = random.choice(range(n), fail_num, replace=False)
                    print('Fail', fail_tasks)
                    not_fail_tasks = list(set(range(n)).difference(fail_tasks))

                # start time
                start = time.time()
                encoded_inputs = encode_conv_MDS(xs, n, k, G)
                datas = [encoded_inputs[i] for i in range(n)]

                # set send task (layer_num and coded inputs) to n workers asynchronously
                if fail_test:
                    send_tasks = [async_send_data(worker_sockets[i], (layer_num, datas[i])) for i in not_fail_tasks] + \
                                 [async_send_data(worker_sockets[i], (layer_num, datas[i], 'fail')) for i in fail_tasks]
                else:
                    send_tasks = [async_send_data(worker_sockets[i], (layer_num, datas[i])) for i in range(n_workers)]
                # send tasks to workers
                loop.run_until_complete(asyncio.gather(*send_tasks))

                # finish master task if necessary
                if master_task > 0:
                    conv = layer.conv
                    m_output_range = (O - master_task, O)
                    i_s, i_e = output_input(m_output_range, layer_config)
                    split_input = x_padded[..., i_s:i_e]
                    master_output = F.conv2d(split_input, conv.weight, stride=conv.stride, padding=0)

                # receive coded outputs from k workers synchronously
                while not recv_list.enough():
                    time.sleep(1e-3)

                results = recv_list.recv_list[:k]

                kset = [t[0] for t in results]
                coded_outputs = [t[1] for t in results]
                outputs = decode_conv_MDS(coded_outputs, n, k, G, kset)
                # end time
                end = time.time()
                consumption = end - start
                records1.append(consumption)

                while not recv_list.full():
                    time.sleep(1e-1)

                # records2.append(recv_list.full_time - start)
            print(f'k={k} records: {records1}')
            # print(f'k=n records: {records2}')
            records.append(records1)

        if fail_test:
            record_file = 'record/'+f'vgg16_layer{layer_num}_r{repetition}_t1_fail{fail_num}.rec'
        else:
            record_file = 'record/'+f'vgg16_layer{layer_num}_r{repetition}_t1.rec'

        with open(record_file, 'wb') as f:
            pickle.dump(records, f)
        records_layers.append(records)

    thread_is_working = False


class k_2k_list():
    def __init__(self, n, k):
        self.__recv_data = []
        assert n % k == 0
        self.n = n
        self.k = k
        self.recv_cnt = 0
        self.__record = np.zeros(self.k)

    def append(self, index_data: tuple) -> None:
        self.recv_cnt += 1  # including 'fail'
        index = index_data[0] % self.k
        if self.__record[index] == 0:
            self.__recv_data.append(index_data)
            self.__record[index] = 1

    def enough(self):
        return len(self.__recv_data) == self.k

    def full(self):
        return self.recv_cnt

    def return_data(self):
        recv_datas = self.__recv_data
        self.__recv_data = []
        self.__record = np.zeros(self.k)
        return recv_datas

    def clear(self) -> None:
        self.__recv_data.clear()
        self.recv_cnt = 0
        self.__record = np.zeros(self.k)

    def size(self):
        return len(self.__recv_data)


def test_distributed_repetitive_convolution_k(master_ip: str, CNN_model: nn.Module, repetition):
    '''
    Test for distributed repetitive convolution with k=5 and n=10
    Correspond to coded_distributed_inference/worker.py
    :param master_ip: master_ip
    :param CNN_model: model instance
    :return: None
    '''
    print('Test for distributed repetitive convolution with k=5 and n=10')
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
    fail_test = False
    fail_num = 2
    # required worker numbers
    n = 10

    loop = asyncio.get_event_loop()  # for asyncio
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

    # CNN and conv task configuration
    layers = CNN_model.features
    layer_configs = generate_layerConfigs(layers)
    layer_indexes = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # vgg16
    layers_to_test = [1]
    # repetition = 10
    task_description = layer_configs, layer_indexes
    send_tasks = [async_send_data(worker_sockets[i], (i, *task_description)) for i in range(n_workers)]
    loop.run_until_complete(asyncio.gather(*send_tasks))

    recv_queue = SimpleQueue()  # recv 'ready' from workers
    for conn in worker_sockets:
        recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
        recv_thread.start()
        recv_thread.join()
    if n_workers != 0 and recv_queue.qsize() == n_workers:
        print('Successfully initialized!')
    else:
        print('Fail to initialize')

    # start recv threads
    k = 5
    recv_list = k_2k_list(n, k)  # or 4*3=12? 但是没有那么多树莓派。。
    thread_is_working = True
    recv_threads = [
        threading.Thread(target=recv_output, args=[worker_sockets[i], recv_list, lambda: thread_is_working]) for i in
        range(n_workers)]
    for t in recv_threads:
        t.start()
    time.sleep(1)

    for layer_num in layers_to_test:
        print(f'Test for layer={layer_num}')
        layer = layers[layer_num]
        # get input shape
        input_shape = model.input_shape if layer_num == 0 else model.output_shapes[layer_num - 1]
        x = torch.randn(input_shape)
        output_shape = model.output_shapes[layer_num]
        O = output_shape[-1]
        layer_config = layer_configs[layer_num]
        padding = layer_config['padding']
        padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
        x_padded = F.pad(x, padding)
        layer_config['padding'] = 0

        print(f'  Test for k={k}')  # 只分成k=5份，然后每个任务重复n/k=2份
        worker_task = O // k
        master_task = O % k
        output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
        input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
        xs = [x_padded[..., s:e].detach().clone() for s, e in input_ranges]  # k*(1,C,I,I/k)
        print('Send_size:', len(pickle.dumps(xs[0])))

        records = []
        for r in tqdm(range(repetition)):
            recv_list.clear()

            if fail_test and fail_num > 0:  # send task (layer_num and k datas) to n workers
                fail_tasks = random.choice(range(n), fail_num, replace=False)
                while fail_num == 2 and fail_tasks[0] % k == fail_tasks[1] % k:
                    fail_tasks = random.choice(range(n), fail_num, replace=False)
                print('Fail', fail_tasks)
                not_fail_tasks = list(set(range(n)).difference(fail_tasks))
                send_tasks = [async_send_data(worker_sockets[i], (layer_num, xs[i % k])) for i in not_fail_tasks] + \
                             [async_send_data(worker_sockets[i], (layer_num, xs[i % k], 'fail')) for i in fail_tasks]
            else:
                send_tasks = [async_send_data(worker_sockets[i], (layer_num, xs[i % k])) for i in range(n_workers)]

            # send_tasks = [async_send_data(worker_sockets[i], (layer_num, xs[i % k])) for i in range(n_workers)]
            start = time.time()
            loop.run_until_complete(asyncio.gather(*send_tasks))

            # finish master task if necessary
            if master_task > 0:
                conv = layer.conv
                m_output_range = (O - master_task, O)
                i_s, i_e = output_input(m_output_range, layer_config)
                split_input = x_padded[..., i_s:i_e]
                master_output = F.conv2d(split_input, conv.weight, stride=conv.stride, padding=0)

            # receive coded outputs from k workers synchronously
            while recv_list.size() < k:
                time.sleep(1e-3)

            outputs = recv_list.return_data()
            outputs.sort(key=lambda pair: pair[0])

            end = time.time()
            consumption = end - start
            records.append(consumption)

            while recv_list.recv_cnt < n_workers:
                time.sleep(1e-1)
        print(f'k={k} records: {records}')
        if fail_test:
            record_file = 'record/'+f'vgg16_layer{layer_num}_r{repetition}_t2_fail{fail_num}.rec'
        else:
            record_file = 'record/'+f'vgg16_layer{layer_num}_r{repetition}_t2.rec'
        with open(record_file, 'wb') as f:
            pickle.dump(records, f)
    thread_is_working = False


def fail_forwarding(layer_num, xs, fail_list: SimpleQueue,
                    worker_sockets: [socket.socket], not_fail_set: set, fail_num):
    sent = [False for _ in range(len(worker_sockets))]
    # while forward_thread_working:
    for _ in range(fail_num):
        task_idx = fail_list.get()
        if task_idx is None:
            continue
        target_idx = not_fail_set.pop()
        for i, worker_socket in enumerate(worker_sockets):
            if i == target_idx:
                send_data(worker_sockets[i], (layer_num, task_idx, xs[i]))
            else:
                if not sent[i]:
                    send_data(worker_sockets[i], xs[i])
            sent[i] = True
        # print(f'Task {task_idx} failed, forward to worker {target_idx}')
    # time.sleep(1e-2)


def test_distributed_repetitive_convolution_N(master_ip: str, CNN_model: nn.Module, repetition):
    '''
    Test for distributed repetitive convolution k=n=10, and repeat failure tasks in other workers
    :param master_ip: master_ip
    :param CNN_model: model instance
    :return: None
    '''
    print('Test for distributed repetitive convolution k=n=10')
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
    fail_test = False
    fail_num = 2
    # required worker numbers
    n = 10

    loop = asyncio.get_event_loop()  # for asyncio
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

    # CNN and conv task configuration
    layers = CNN_model.features
    layer_configs = generate_layerConfigs(layers)
    layer_indexes = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # vgg16
    layers_to_test = [1]
    # repetition = 10
    # task_description = (layer_configs, layer_indexes, layers_to_test, repetition, n)
    task_description = layer_configs, layer_indexes
    send_tasks = [async_send_data(worker_sockets[i], (i, *task_description)) for i in range(n_workers)]
    loop.run_until_complete(asyncio.gather(*send_tasks))

    recv_queue = SimpleQueue()  # 接收workers确认与所有其他workers完成建立连接的消息
    for conn in worker_sockets:
        recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
        recv_thread.start()
        recv_thread.join()
    if n_workers != 0 and recv_queue.qsize() == n_workers:
        print('All workers are ready!')
    else:
        print('Fail to initialize!')

    # start recv threads
    recv_list = []
    recv_thread_working = True
    fail_list = SimpleQueue()  # record failed tasks by response
    recv_threads = [
        threading.Thread(target=recv_output_with_fail,
                         args=[worker_sockets[i], recv_list, lambda: recv_thread_working, fail_list]) for i in
        range(n_workers)]
    for t in recv_threads:
        t.start()
    time.sleep(1)

    for layer_num in layers_to_test:
        print(f'Test for layer={layer_num}')
        layer = layers[layer_num]
        print(layer)
        # get input shape
        input_shape = model.input_shape if layer_num == 0 else model.output_shapes[layer_num - 1]
        x = torch.randn(input_shape)
        output_shape = model.output_shapes[layer_num]
        O = output_shape[-1]
        print(input_shape, output_shape)
        layer_config = layer_configs[layer_num]
        padding = layer_config['padding']
        padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
        x_padded = F.pad(x, padding)
        layer_config['padding'] = 0

        # Only for cases of k=n=10
        print(f'  Test for k=n=10')
        worker_task = O // n
        master_task = O % n
        output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
        input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
        xs = [x_padded[..., s:e].detach().clone() for s, e in input_ranges]  # k*(1,C,I,I/k)
        # show sending size
        print('Send_size:', len(pickle.dumps(xs[0])))

        records = []
        for r in tqdm(range(repetition)):
            recv_list.clear()
            if fail_test and fail_num > 0:  # 先手动fail
                fail_set = random.choice(range(n), fail_num, replace=False)
                # print('Fail', fail_set)
                not_fail_set = set(range(n)).difference(fail_set)
                # forward_target = set(random.choice(not_fail_set, fail_num, replace=False))

                # forwarding thread to forward failure tasks
                # repetition_set = set(range(n_workers))
                forward_thread_working = True
                fail_forwarding_thread = threading.Thread(target=fail_forwarding,
                                                          args=[layer_num, xs, fail_list,
                                                                worker_sockets, not_fail_set, fail_num])
                fail_forwarding_thread.start()

            # start time
            start = time.time()

            # set send task (layer_num and n datas) to n workers
            # send_tasks = [async_send_data(worker_sockets[i], (layer_num, xs[i])) for i in range(n_workers)]
            if fail_test:
                send_tasks = [async_send_data(worker_sockets[i], (layer_num, xs[i])) for i in not_fail_set] + \
                             [async_send_data(worker_sockets[i], (layer_num, xs[i], 'fail')) for i in fail_set]
            else:
                send_tasks = [async_send_data(worker_sockets[i], (layer_num, xs[i])) for i in range(n_workers)]
            # send tasks to workers
            loop.run_until_complete(asyncio.gather(*send_tasks))

            # finish master task if necessary
            if master_task > 0:
                conv = layer.conv
                m_output_range = (O - master_task, O)
                i_s, i_e = output_input(m_output_range, layer_config)
                split_input = x_padded[..., i_s:i_e]
                master_output = F.conv2d(split_input, conv.weight, stride=conv.stride, padding=0)

            # receive coded outputs from k workers synchronously
            while len(recv_list) < n:
                time.sleep(1e-3)

            # print(len(recv_list))
            # print([item.for item in recv_list])

            recv_list.sort(key=lambda pair: pair[0])

            end = time.time()
            if fail_test and fail_num > 0:
                forward_thread_working = False
                fail_forwarding_thread.join()
            consumption = end - start
            records.append(consumption)
        print(f'k=n=10 records: {records}')

        if fail_test:
            record_file = 'record/'+f'vgg16_layer{layer_num}_r{repetition}_t3_fail{fail_num}.rec'
        else:
            record_file = 'record/'+f'vgg16_layer{layer_num}_r{repetition}_t3.rec'

        with open(record_file, 'wb') as f:
            pickle.dump(records, f)

    recv_thread_working = False


if __name__ == '__main__':
    __port__ = 49999
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master', type=str, required=True)
    parser.add_argument('-t', '--test', type=int, required=False, default=0)
    parser.add_argument('-r', '--repetition', type=int, required=False, default=10)
    args = parser.parse_args()
    master_ip = args.master
    test_choice = args.test
    repetition = args.repetition
    # master_ip = '192.168.1.148'

    model = load_model('vgg16')
    # x = torch.randn(model.input_shape)

    repetition = 100

    print('Repetition', repetition)

    if test_choice == 0:
        test_distributed_coded_convolution_k(master_ip, model, repetition)
        # test_distributed_repetitive_convolution_k(master_ip, model, repetition)
        # test_distributed_repetitive_convolution_N(master_ip, model, repetition)
    elif test_choice == 1:
        test_distributed_coded_convolution_k(master_ip, model, repetition)
    elif test_choice == 2:
        test_distributed_repetitive_convolution_k(master_ip, model, repetition)
    elif test_choice == 3:
        test_distributed_repetitive_convolution_N(master_ip, model, repetition)
