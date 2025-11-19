from socket import AF_INET, create_server
import sys
from datetime import datetime

from comm_util import keep_recv_output_with_fail
from coding_util import *
from functions import *
from tqdm import tqdm
from util import save_object, load_object
import numpy.random as random
from fountain_code_test.LTCodes import *


def create_connections(master_ip: str, timeout=None):
    __port__ = 49999
    listen_addr = (master_ip, __port__)
    server_socket = create_server(listen_addr, family=AF_INET)

    # accept connections from workers
    server_socket.settimeout(8)
    stop_thread = False
    available_workers = []
    print(f'Master ip is {master_ip}, recv connections from workers...')
    # start a thread to collect connections from workers
    recv_thread = threading.Thread(target=accept_connection,
                                   args=[server_socket, available_workers, lambda: stop_thread, timeout])
    recv_thread.start()
    time.sleep(10)
    stop_thread = True
    recv_thread.join()
    server_socket.close()

    worker_sockets = []
    ip_set = set()
    for conn, addr in available_workers:
        if addr not in ip_set:  # first time to meet the addr
            ip_set.add(addr)
            worker_sockets.append(conn)
        else:
            conn.close()
    n_workers = len(ip_set)
    print(f'Recv connections from {n_workers} workers')
    return worker_sockets, ip_set, n_workers


class DistributedEngine:
    '''
    用于Distributed Coded Convolution
    所用涉及到分布式的方法都写到这个类里面
    '''

    def __init__(self, master_ip: str, timeout, model: nn.Module, n: int):
        '''
        params有model，说明当前的实验以及master的启动是对应于单个model的
        用于初始化master以及建立于workers的连接
        ！！！专门用于distributed convolution实验！！！
        '''
        # accept connections from workers
        worker_sockets, ip_set, n_workers = create_connections(master_ip, timeout)

        # try:
        #     self.loop = asyncio.get_running_loop()
        # except RuntimeError:
        #     self.loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(self.loop)
        self.loop = asyncio.get_event_loop()

        self.model = model
        self.layers_output_shapes = model.output_shapes
        self.layers = model.features
        layer_configs = generate_layerConfigs(self.layers)

        send_tasks = [async_send_data(worker_sockets[i], (i, layer_configs)) for i in range(n_workers)]
        self.loop.run_until_complete(asyncio.gather(*send_tasks))  # send the layer_config to all workers

        recv_queue = SimpleQueue()  # wait for all responses from all workers
        for conn in worker_sockets:
            recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
            recv_thread.start()
            recv_thread.join()
        if n_workers != 0 and recv_queue.qsize() == n_workers:  # workers的回复全部收到，初始化完成
            print('Successfully initialized!')
        else:
            print('Fail to initialize')

        # start recv threads to recv computing results from workers
        # finish the distributed coded engine

        self.worker_sockets = worker_sockets
        self.ip_set = ip_set
        self.n_workers = n_workers
        self.recv_queue = SimpleQueue()  # recv all responses from workers, the shared recv queue
        self.fail_task_queue = SimpleQueue()
        self.forwarding = False
        self.is_working = True
        self.recv_threads = [
            threading.Thread(target=keep_recv_output_with_fail,
                             args=[worker_sockets[worker_id], self.recv_queue, self.fail_task_queue,
                                   lambda: self.is_working]) for worker_id in
            range(n_workers)]
        if n > self.n_workers:
            raise Exception('Cannot set n greater than the actual number of workers')
        if n < 2:
            raise Exception('Cannot set n smaller than 3')
        self.n = n
        self.chosen_workers = None
        self.set_n(n)
        self.fail_num = 0  # no failure, and have failure if > 0
        self.not_fail_list = None
        self.forwarding_thread = threading.Thread(target=self.bad_fail_forwarding, args=[])
        self.cur_layer_id = None
        self.method = None
        self.inputs = None

        # master是否应该知道straggler的存在：理论上不知道，但是这个设置是为了方便设置默认的coded策略
        self.n_straggler = 0
        self.normal_workers = None  #
        self.stragglers = None  # 目前只有一个straggler, 存对应straggler的引用

        self.random_waiting = False
        self.upload_rate = 11 << 20  # MB/s -> B/s
        self.random_waiting_coefficient = 0  # in [1, n]
        self.waiting_latency = None

        # for async LT codes
        self.asyncio_thread = threading.Thread(target=self.run_loop)
        self.asyncio_thread.start()
        for conn in self.worker_sockets:
            conn.setblocking(False)

    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

        # self.forwarding_thread.start()
        # for t in self.recv_threads:
        #     t.start()

    def set_random_waiting(self, is_waiting=False, lambda_tr=0):
        if is_waiting:
            self.random_waiting = True
            assert lambda_tr > 0
            self.random_waiting_coefficient = lambda_tr

    def set_n(self, n_new: int):
        if n_new is not None:
            n_new = self.n if n_new is None else n_new
            assert self.n_workers >= n_new > 1
            if n_new != self.n and self.n < self.n_workers:
                chosen_workers = np.random.choice(self.n_workers, size=n_new, replace=False)
                self.chosen_workers = [self.worker_sockets[i] for i in chosen_workers]
            else:  # if all workers participate in computing, keep the order of workers
                self.chosen_workers = self.worker_sockets
            self.n = n_new

    def set_straggler(self, n_straggler: int):
        assert self.n > self.fail_num + n_straggler
        self.n_straggler = n_straggler
        if self.n_straggler > 0:
            self.stragglers = [self.worker_sockets[-n_straggler:]]  # 一般就是最慢的几个
            self.normal_workers = [self.worker_sockets[:-n_straggler]]

    def set_failure(self, n_failure_new: int):
        if n_failure_new is not None and n_failure_new != self.fail_num:
            assert self.n > n_failure_new + self.n_straggler
            self.fail_num = n_failure_new
            if self.fail_num > 0:
                self.forwarding_thread.start()

    def end(self):
        self.is_working = False
        for worker_socket in self.worker_sockets:
            worker_socket.close()
        # for recv_thread in self.recv_threads:
        #     # recv_thread.daemon = True
        # if self.forwarding_thread is not None:
        # self.forwarding_thread.daemon = True# 这里有问题

    def gen_send_tasks(self, taskID, layer_id):
        if self.fail_num == 0:
            datas = [(taskID, task_idx, layer_id, self.inputs[task_idx]) for task_idx in
                     range(self.n)]  # taskID as the identifier of conv task
            if self.random_waiting:  # only consider random waiting when n_f=0
                send_tasks = [async_send_data(self.chosen_workers[i], datas[i], self.waiting_latency[i]) for i in
                              range(self.n)]
            else:
                send_tasks = [async_send_data(self.chosen_workers[i], datas[i]) for i in range(self.n)]
        else:  # fail_num > 0
            fail_task_idx = random.choice(self.n, self.fail_num, replace=False)
            if self.fail_num == 2 and self.method == 'repetition':
                while fail_task_idx[0] % (self.n // 2) == fail_task_idx[1] % (self.n // 2):
                    fail_task_idx = random.choice(self.n - self.n_straggler, self.fail_num,
                                                  replace=False)  # - n_stragglers排除后面几个stragglers
            not_fail_task_idx = list(set(range(n)).difference(fail_task_idx))
            self.not_fail_list = not_fail_task_idx
            send_tasks = [async_send_data(self.chosen_workers[i], (taskID, i, layer_id, self.inputs[i])) for i in
                          not_fail_task_idx] + \
                         [async_send_data(self.chosen_workers[i], (taskID, i, 'fail', self.inputs[i])) for i in
                          fail_task_idx]
        return send_tasks

    def gen_random_waiting(self, tensor_shape, n: int):
        # rate_tr in self.random_waiting_coefficient
        Ntr = np.prod(tensor_shape) << 2
        scale = self.random_waiting_coefficient * (Ntr / self.upload_rate)  # *n/2
        self.waiting_latency = random.exponential(scale, n)

    def distributed_lt_conv(self, x: torch.Tensor, layer, layer_id, layer_config, k=None):
        self.method = 'lt'
        self.cur_layer_id = layer_id
        self.forwarding = False
        # 准备一个表示解码成功率的dict看至少要接收多少个symbols才到99%

        output_shape = model.output_shapes[layer_id]
        O = output_shape[-1]
        n = self.n
        padding = layer.padding
        if padding != 0:  # pre-padding of x
            padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
            x = F.pad(x, padding)
            # x_padded = F.pad(x, padding)
            layer_config['padding'] = 0

        if k is None:  # k=width of output shape: minimum split
            k = O
        else:  # 正常拆分，与MDS相同
            k = n - self.fail_num - self.n_straggler

        each_task = O // k  # master_task < worker_task
        master_task = O % k

        output_ranges = [(i, i + each_task) for i in range(0, O - master_task, each_task)]
        input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
        xs = [x[..., s:e].clone() for s, e in input_ranges]  # k*(1,C,I,I/k)
        lt_task = LTCodedTask(xs, layer_id, self.worker_sockets, self.loop)

        # todo: random_waiting for LT-codes cases

        start_or_taskID = time.perf_counter()
        lt_task.taskID = start_or_taskID

        assert self.loop.is_running()

        # start subtasks distribution
        future = asyncio.run_coroutine_threadsafe(lt_task.start_comm(), self.loop)
        # asyncio.run(lt_task.start_comm())

        # finish master task if necessary
        master_output = None
        if master_task > 0:
            m_output_range = (O - master_task, O)
            i_s, i_e = output_input(m_output_range, layer_config)
            # split_input = x_padded[..., i_s:i_e]
            split_input = x[..., i_s:i_e]
            master_output = F.conv2d(split_input, layer.weight, stride=layer.stride, padding=0)

        lt_task.decodable_event.wait()
        decoded_outputs = lt_task.decode_ge()
        consumption = time.perf_counter() - start_or_taskID

        if master_task > 0:
            decoded_outputs.append(master_output)

        output = torch.concat(decoded_outputs, dim=-1)

        future.result(timeout=10)
        # if not future.done():
        #     future.cancel()
        # self.loop.close()

        return output, consumption

    def distributed_mds_conv(self, x: torch.Tensor, layer, layer_id, layer_config, r=None):
        self.method = 'mds'
        self.cur_layer_id = layer_id
        self.forwarding = False
        n = self.n
        if r is None:
            k = n - self.fail_num - self.n_straggler
        else:
            assert 0 < r < n
            k = n - r

        # print(f'n,k={n},{k}')  # debug

        output_shape = model.output_shapes[layer_id]
        G = torch.vander(torch.arange(1, n + 1), k).float()  # vandermonde matrix, any generation matrix
        # output_shapes = self.model.output_shapes
        recv_list = []
        # pad the input in advance
        padding = layer.padding
        if padding != 0:
            padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
            x = F.pad(x, padding)
            # x_padded = F.pad(x, padding)
            layer_config['padding'] = 0

        O = output_shape[-1]
        worker_task = O // k  # master_task < worker_task
        master_task = O % k

        output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
        input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
        xs = [x[..., s:e] for s, e in input_ranges]  # k*(1,C,I,I/k)

        if self.random_waiting:
            self.gen_random_waiting(xs[0].shape, n)
            # print(self.waiting_latency)

        start_or_taskID = time.perf_counter()  # 用于计时的同时，作为task_id的唯一辨认符
        encoded_inputs = encode_conv_MDS(xs, n, k, G)
        self.inputs = encoded_inputs

        # # send task to all workers asynchronously based on the fail_num
        # if self.fail_num == 0:
        #     datas = [(start_or_taskID, layer_id, encoded_inputs[i]) for i in range(n)]  # 发送taskID作为任务辨认符
        #     send_tasks = [async_send_data(chosen_workers[i], datas[i]) for i in range(self.n)]
        # else:  # fail_num > 0
        #     fail_task_idx = random.choice(range(n), self.fail_num, replace=False)
        #     not_fail_task_idx = set(range(n)).difference(fail_task_idx)
        #     send_tasks = [async_send_data(chosen_workers[i], (start_or_taskID, layer_id, xs[i])) for i in not_fail_task_idx] + \
        #                  [async_send_data(chosen_workers[i], (layer_id, xs[i], 'fail')) for i in fail_task_idx]

        send_tasks = self.gen_send_tasks(start_or_taskID, layer_id)
        self.loop.run_until_complete(asyncio.gather(*send_tasks))

        # finish master task if necessary
        master_output = None
        if master_task > 0:
            m_output_range = (O - master_task, O)
            i_s, i_e = output_input(m_output_range, layer_config)
            # split_input = x_padded[..., i_s:i_e]
            split_input = x[..., i_s:i_e]
            master_output = F.conv2d(split_input, layer.weight, stride=layer.stride, padding=0)

        # receive coded outputs from k workers synchronously
        recv_cnt = 0
        while len(recv_list) < k:
            taskID, recv_data = self.recv_queue.get()
            recv_cnt += 1
            # taskID = recv_data[0]
            if taskID == start_or_taskID:
                recv_list.append(recv_data)

        kset = [t[0] for t in recv_list[:k]]
        coded_outputs = [t[-1] for t in recv_list[:k]]
        outputs = decode_conv_MDS(coded_outputs, n, k, G, kset)
        consumption = time.perf_counter() - start_or_taskID
        # 全部发出的任务收完防止影响后续任务的计算
        while recv_cnt < n - self.fail_num:
            _ = self.recv_queue.get()
            recv_cnt += 1

        if master_task > 0:
            outputs.append(master_output)

        output = torch.concat(outputs, dim=-1)
        # self.recv_queue = SimpleQueue()  # 不能修改recv_queue的引用
        return output, consumption

    def fail_forwarding(self):
        while self.is_working:
            try:
                for fail_cnt in range(self.fail_num):
                    taskID, task_idx = self.fail_task_queue.get(timeout=15)
                    if self.forwarding:
                        x = self.inputs[task_idx]
                        target_idx = self.not_fail_list[fail_cnt]
                        send_data(self.chosen_workers[target_idx], (taskID, task_idx, self.cur_layer_id, x))
                        print(f'Forward task {task_idx} to {target_idx}')
            except queue.Empty:
                break

    def bad_fail_forwarding(self):
        loop_this_thread = asyncio.new_event_loop()
        asyncio.set_event_loop(loop_this_thread)
        while self.is_working:
            try:
                for fail_cnt in range(self.fail_num):
                    taskID, task_idx = self.fail_task_queue.get(timeout=15)
                    if self.forwarding:
                        x = self.inputs[task_idx]
                        target_idx = self.not_fail_list[fail_cnt]
                        send_tasks = [async_send_data(self.chosen_workers[i], (0, x) if i != target_idx else (
                            taskID, task_idx, self.cur_layer_id, x)) for i in range(self.n)]
                        loop_this_thread.run_until_complete(asyncio.gather(*send_tasks))
            except queue.Empty:
                break

    def distributed_uncoded_conv(self, x: torch.Tensor, layer, layer_id, layer_config):
        self.method = 'uncoded'
        self.cur_layer_id = layer_id
        self.forwarding = True
        n = self.n
        # uncoded是否传padding呢？（大可不必）
        assert len(self.worker_sockets) >= n  # has been ensured in master

        # recv_times = []
        recv_list = []
        output_shape = model.output_shapes[layer_id]
        # pad the input in advance
        padding = layer.padding
        if padding != 0:
            padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
            x = F.pad(x, padding)
            # x_padded = F.pad(x, padding)
            layer_config['padding'] = 0
        # padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
        # x_padded = F.pad(x, padding)
        # layer_config['padding'] = 0

        O = output_shape[-1]
        worker_task = O // n  # master_task < worker_task
        master_task = O % n

        output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
        # print(output_ranges)
        input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
        # note the reference decoupling
        xs = [x[..., s:e].clone().detach() for s, e in input_ranges]  # k*(1,C,I,I/k)
        if self.random_waiting:
            self.gen_random_waiting(xs[0].shape, n)
            # print(self.waiting_latency)
        self.inputs = xs

        start_or_taskID = time.perf_counter()
        # datas = [(start_or_taskID, layer_id, xs[i]) for i in range(n)]
        # # send task (layer_num and coded inputs) to n workers asynchronously
        # send_tasks = [async_send_data(chosen_workers[i], datas[i]) for i in range(n)]

        send_tasks = self.gen_send_tasks(start_or_taskID, layer_id)
        self.loop.run_until_complete(asyncio.gather(*send_tasks))

        # finish master task if necessary
        if master_task > 0:
            m_output_range = (O - master_task, O)
            i_s, i_e = output_input(m_output_range, layer_config)
            split_input = x[..., i_s:i_e]
            master_output = F.conv2d(split_input, layer.weight, stride=layer.stride, padding=0)

        # receive coded outputs from all n workers synchronously
        recv_cnt = 0
        while len(recv_list) < n:
            taskID, recv_data = self.recv_queue.get()
            # recv_time = time.perf_counter() - start_or_taskID
            # recv_times.append(recv_time)
            recv_cnt += 1
            # taskID = recv_data[0]
            if taskID == start_or_taskID:
                recv_list.append(recv_data)

        recv_list.sort(key=lambda t: t[0])
        outputs = [data[1] for data in recv_list]
        consumption = time.perf_counter() - start_or_taskID
        # print(recv_times)

        # 全部发出的任务收完防止影响后续任务的计算
        while recv_cnt < n:
            _ = self.recv_queue.get()
            recv_cnt += 1

        if master_task > 0:
            outputs.append(master_output)
        # print([item.shape for item in outputs])
        output = torch.concat(outputs, dim=-1)

        return output, consumption

    def distributed_repetition_conv(self, x: torch.Tensor, layer, layer_id, layer_config):
        self.method = 'repetition'
        self.cur_layer_id = layer_id
        self.forwarding = False
        # assert len(self.worker_sockets) >= n and n % 2 == 0  # has been ensured in master
        n = self.n if self.n % 2 == 0 else self.n - 1
        k = n // 2

        recv_list = [None for _ in range(k)]
        output_shape = model.output_shapes[layer_id]
        # pad the input in advance
        padding = layer.padding
        if padding != 0:
            padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
            x = F.pad(x, padding)
            # x_padded = F.pad(x, padding)
            layer_config['padding'] = 0

        O = output_shape[-1]
        worker_task = O // k  # master_task < worker_task
        master_task = O % k

        output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
        # print(output_ranges)
        input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
        # reference decoupling before sending the splitting data
        xs = [x[..., s:e].clone().detach() for s, e in input_ranges]  # k*(1,C,I,I/k)\
        if self.random_waiting:
            self.gen_random_waiting(xs[0].shape, n)
            # print(self.waiting_latency)
        self.inputs = xs + xs

        start_or_taskID = time.perf_counter()
        # datas = [(start_or_taskID, layer_id, xs[i % k]) for i in range(n)]
        # # send task (layer_num and coded inputs) to n workers asynchronously
        # send_tasks = [async_send_data(chosen_workers[i], datas[i]) for i in range(n)]

        send_tasks = self.gen_send_tasks(start_or_taskID, layer_id)
        self.loop.run_until_complete(asyncio.gather(*send_tasks))

        # finish master task if necessary
        if master_task > 0:
            m_output_range = (O - master_task, O)
            i_s, i_e = output_input(m_output_range, layer_config)
            split_input = x[..., i_s:i_e]
            master_output = F.conv2d(split_input, layer.weight, stride=layer.stride, padding=0)

        # receive coded outputs from all n workers synchronously
        recv_cnt = 0
        while None in recv_list:
            taskID, recv_data = self.recv_queue.get()  # recv_data: task_idx, output
            recv_cnt += 1
            if taskID == start_or_taskID:
                recv_list[recv_data[0] % k] = recv_data[1]  # 实际只有k份任务，task_idx%k相等意味着任务重复
        # recv_list.sort(key=lambda t: t[0])
        outputs = recv_list

        if master_task > 0:
            outputs.append(master_output)
        # print([item.shape for item in outputs])
        output = torch.concat(outputs, dim=-1)
        consumption = time.perf_counter() - start_or_taskID
        # 全部发出的任务收完防止影响后续任务的计算：这里有bug啊，为什么加了之后就贼慢
        while recv_cnt < n - self.fail_num:
            _ = self.recv_queue.get()
            recv_cnt += 1

        return output, consumption


def execute_layer(input, layer_id, layer, layer_config=None,
                  distributed_engine=None, distributed=False, method=None):
    '''
    input: 传递的输入
    layer_id: 传递层号，用于workers设备根据层号读取权重
    layer: 层引用，用于判别层类型
    layer_config: 该网络层推理相关的必要配置参数
    distributed_engine: 提供分布式组件及分布式计算方法
    '''
    if isinstance(layer, nn.Conv2d):
        start = time.perf_counter()
        if not distributed:
            output = F.conv2d(input, layer.weight, bias=layer.bias, stride=layer.stride, padding=layer.padding)
        else:
            if method.startswith('mds'):
                r = None
                if '-' in method:
                    r = int(method.split('-')[1])
                output, latency_coded = distributed_engine.distributed_mds_conv(input, layer, layer_id, layer_config, r)
                return output, latency_coded
            elif method.startswith('lt'):
                if '-' in method and method.split('-')[1].isdigit():
                    k = int(method.split('-')[1])
                else:
                    k = None

                output, latency_coded = distributed_engine.distributed_lt_conv(input, layer, layer_id, layer_config, k)
            elif method == 'repetition':
                output, latency_repetition = distributed_engine.distributed_repetition_conv(input, layer, layer_id,
                                                                                            layer_config)
                return output, latency_repetition
            elif method == 'uncoded':
                output, latency_distributed = distributed_engine.distributed_uncoded_conv(input, layer, layer_id,
                                                                                          layer_config)
                return output, latency_distributed
            else:
                raise Exception('Unsupported distributed convolution method')
    elif isinstance(layer, BasicConv2d):
        _output, latency_conv = execute_layer(input, layer_id, layer.conv, layer_config, distributed_engine,
                                              distributed, method)
        _output, latency_bn = execute_layer(_output, layer_id, layer.bn, layer_config)
        output, latency_relu = execute_layer(_output, layer_id, layer.relu, layer_config)
        latency = latency_conv + latency_bn + latency_relu
        return output, latency
    elif isinstance(layer, nn.BatchNorm2d):
        start = time.perf_counter()
        output = F.batch_norm(input,
                              running_mean=layer.running_mean,
                              running_var=layer.running_var,
                              weight=layer.weight,
                              bias=layer.bias,
                              training=False,  # 设置为False，因为在推理模式下不更新统计量
                              # momentum=layer.momentum,  # 这个值在推理时不使用，但为了完整性可以保留
                              eps=layer.eps
                              )
    elif isinstance(layer, nn.ReLU):
        start = time.perf_counter()
        output = F.relu(input, inplace=True)
    elif isinstance(layer, nn.MaxPool2d):
        start = time.perf_counter()
        output = F.max_pool2d(input, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
    elif layer == 'add':
        start = time.perf_counter()
        assert isinstance(input, list)
        output = input[0] + input[1]
    elif layer == 'concat':
        start = time.perf_counter()
        assert isinstance(input, list)
        output = torch.concat(input, dim=1)
    else:
        raise Exception('Unsupported Layer Type')
    consumption = time.perf_counter() - start
    return output, consumption


def dnn_inference_by_layer(x: torch.Tensor, layers, next_array, methods=('Uncoded',), conv_repeat=1,
                           distributed_engine=None):
    '''
    DNN inference with input x, inference by layer in topological order
    Store necessary intermediate results, and drop useless results in time
    Mainly for testing the latency of executing the conv layers in distributed method
    :param x:
    :param model:
    :param layers:
    :param next_array:
    :param last_array:
    :return:
    '''
    n_layers = len(layers)
    translate_next_array(next_array)
    next_num = [len(next_layers) for next_layers in next_array]
    last_array = next_to_last(next_array)
    # the execute order of layers
    topology_layer = topology_DAG(next_array, last_array)
    print(topology_layer)

    # store layer output in list by layer_id if necessary, and clear useless results in time
    layer_output = [None for _ in range(n_layers)]
    layer_configs = generate_layerConfigs(layers)
    layer_latency = [None for _ in range(n_layers)]
    output = None
    # output_shapes = [None] * n_layers  # 用于计算并保存模型每一层的output_shape

    for layer_id in topology_layer:  # execute layers according to topology order
        layer = layers[layer_id]
        last_layers = last_array[layer_id]
        # next_layers = next_array[layer_id]
        layer_config = layer_configs[layer_id]

        if layer_id == 3:  # 仅用于debug
            a = 1

        # collect required input according to dependent layers
        if len(last_layers) == 0:  # 1st layer
            required_input = x
        elif len(last_layers) == 1:
            # if next_num[last_layers[0]] == 1:
            #     # have only one last layer, and this layer is the only one next layer of last layer
            #     required_input = this_output
            # else:
            required_input = layer_output[last_layers[0]]
        else:
            required_input = [layer_output[last_layer] for last_layer in last_layers]
            for last_layer in last_layers:  # 对层引用减小计数，当引用为0时丢弃该层储存的中间结果
                next_num[last_layer] -= 1
                if next_num[last_layer] == 0:
                    layer_output[last_layer] = None

        # compute output according to layer and input
        if isinstance(layer, (nn.Conv2d, BasicConv2d)):  # try to execute conv layers in distributed way
            print(f'Layer {layer_id}: {layer}')
            if len(methods) > 0 and distributed_engine is not None:  # distributed engine exists, execute conv layers distributedly
                this_layer_latency = []
                for method in methods:
                    method_latency = []
                    print(f'{method} test:')
                    time.sleep(0.1)
                    for _ in tqdm(range(conv_repeat)):
                        this_output, latency = execute_layer(required_input, layer_id, layer, layer_config,
                                                             distributed_engine=distributed_engine,
                                                             distributed=True, method=method)
                        method_latency.append(latency)
                    time.sleep(0.1)
                    this_layer_latency.append(method_latency)
                    print(f'{method}\'s mean: {np.mean(method_latency)}')
                layer_latency[layer_id] = this_layer_latency
            else:
                this_output, latency = execute_layer(required_input, layer_id, layer, layer_config)
                layer_latency[layer_id] = latency

        else:  # other layers, just execute locally
            this_output, latency = execute_layer(required_input, layer_id, layer, layer_config)
            layer_latency[layer_id] = latency

        # if len(next_layers) > 1:  # 若该层被接下来多层引用，储存该层的中间结果
        layer_output[layer_id] = this_output
        output = this_output
    #     output_shapes[layer_id] = tuple(output.shape)
    # print(output_shapes)

    return output, layer_latency


def latency_analysis(local_layers, conv_idxs, method_latency):
    '''
    local_layers: 每层在本地执行的时延
    conv_idxs: conv层对应的序号
    method_latency: conv_idxs对应的conv层使用某个方法对应的时延，每层包含所有重复的记录
    method: 方法名
    '''
    n_layers = len(local_layers)
    non_conv_latency = 0
    method_conv_latency_min = 0
    method_conv_latency_mean = 0
    method_conv_latency_max = 0
    layer_idx_distributed = []
    for layer_id in range(n_layers):
        if layer_id not in conv_idxs:
            non_conv_latency += local_layers[layer_id]
        else:
            idx = conv_idxs.index(layer_id)
            method_conv_latency_min += min(np.min(method_latency[idx]), local_layers[layer_id])
            method_conv_latency_mean += min(np.mean(method_latency[idx]), local_layers[layer_id])
            method_conv_latency_max += min(np.max(method_latency[idx]), local_layers[layer_id])
            if np.max(method_latency[idx]) < local_layers[layer_id]:
                layer_idx_distributed.append(layer_id)
    method_conv_latency_min += non_conv_latency
    method_conv_latency_mean += non_conv_latency
    method_conv_latency_max += non_conv_latency
    return method_conv_latency_min, method_conv_latency_mean, method_conv_latency_max, layer_idx_distributed


def latency_min_mean(local_layers, conv_idxs, conv_layers_latency):
    n_layers = len(local_layers)
    non_conv_latency = 0
    conv_layers_min_mean = []
    for layer_id in range(n_layers):
        if layer_id not in conv_idxs:
            non_conv_latency += local_layers[layer_id]
        else:
            conv_idx = conv_idxs.index(layer_id)
            method_layer_mean = np.mean(np.asarray(conv_layers_latency[conv_idx]), axis=1)
            print(f'Layer {layer_id}: {method_layer_mean}')
            layer_min_mean = np.min(method_layer_mean)
            conv_layers_min_mean.append(min(layer_min_mean, local_layers[layer_id]))
    print('Conv latency:', conv_layers_min_mean)
    return non_conv_latency + np.sum(conv_layers_min_mean)


# distributed execution: python worker.py -m 192.168.1.106 --model vgg16
def distributed_inference_testing(layer_repetition, methods: list, master: DistributedEngine, save=False):
    print(
        f'test_methods={methods}, repetition={layer_repetition}, n={master.n}, fail_num={master.fail_num}, straggler={master.n_straggler}')
    output, layer_latency = dnn_inference_by_layer(x, model.features, model.next, methods=methods,
                                                   conv_repeat=layer_repetition, distributed_engine=master)
    assert layer_latency.count(None) == 0

    print(output.shape)

    # latency summary
    conv_idxs = []
    distributed_conv_idxs = []
    conv_latency_methods = []
    centralize_latency = 0
    for layer_id, latency in enumerate(layer_latency):
        if isinstance(latency, list):
            conv_idxs.append(layer_id)
            conv_latency_methods.append(latency)
        else:
            centralize_latency += latency
    print(conv_idxs)
    layer_latency_local = load_object(f'{model_name}_layer_latecy_local.loc')
    print('Results of single method:')
    # 对于每层，单独统计每个方法的min，mean，max，然后汇总到该方法总体inference的时延
    for i, method in enumerate(methods):
        conv_latency_method = [l[i] for l in conv_latency_methods]
        l_min, l_mean, l_max, distributed_conv_idxs = latency_analysis(layer_latency_local, conv_idxs,
                                                                       conv_latency_method)
        print(f'\'{method}\': {{\'min\':{l_min}, \'mean\':{l_mean}, \'max\':{l_max}}}')
        print('Distributed conv idxes:', distributed_conv_idxs)

    # 对于不同方法执行每层的时延单独统计mean，然后每层都取时延最小对应方法的mean，汇总为混合方法能达到的最快时延
    print('Results of mixed method:')
    hybrid_min_mean = latency_min_mean(layer_latency_local, conv_idxs, conv_latency_methods)
    print('Hybrid minimum of mean layer latency:', hybrid_min_mean)

    if save:
        record = {'model': model_name, 'conv_idxs': conv_idxs, 'n': master.n, 'test_repetition': test_repetition,
                  'test_methods': methods}
        for i, method in enumerate(methods):
            conv_latency_method = [l[i] for l in conv_latency_methods]
            l_min, l_mean, l_max = latency_analysis(layer_latency_local, conv_idxs, conv_latency_method)
            record[method] = (l_min, l_mean, l_max, conv_latency_method)
        formatted_time = datetime.now().strftime('%m%d%H%M')
        save_object(record, f'record/{model_name}_n={10}_r={test_repetition}_f={fail_num}_{formatted_time}.dict')
    for worker_socket in master.worker_sockets:
        send_data(worker_socket, 'end')
    time.sleep(5)
    master.end()


# def layer_test(test_repetition, test_methods, master, save):
#     # start time of the task
#     times = []
#     coded = True
#     n = n_workers
#     if coded:
#         ks = range(n // 2, n)
#     else:
#         ks = [n]
#     for k in ks:
#         G = torch.vander(torch.arange(1, n + 1), k).float()  # vandermonde matrix, any generation matrix
#         output_shapes = model.output_shapes
#         # 逐层处理，提取出其中计算量较大的线性计算，编码后发送给workers
#         for layer_num, layer in enumerate(CNN_model.features):
#             recv_list.clear()
#             times.append(time.perf_counter())
#             if isinstance(layer, BasicConv2d):  # for vgg16, only conv and linear layers are linear computation
#                 input_shape = x.shape
#                 output_shape = output_shapes[layer_num]
#                 # I = input_shape[-1]
#                 O = output_shape[-1]
#                 layer_config = layer_configs[layer_num]
#
#                 padding = layer_config['padding']
#                 padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
#                 x_padded = F.pad(x, padding)
#                 layer_config['padding'] = 0
#
#                 worker_task = O // k  # master_task < worker_task
#                 master_task = O % k
#
#                 output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
#                 print(output_ranges)
#                 input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
#                 xs = [x_padded[..., s:e] for s, e in input_ranges]  # k*(1,C,I,I/k)
#                 if coded:  # encoding input
#                     encoded_inputs = encode_conv_MDS(xs, n, k, G)
#                     datas = [(layer_num, encoded_inputs[i]) for i in range(n)]
#                 else:
#                     datas = [(layer_num, xs[i]) for i in range(k)]
#
#                 # send task (layer_num and coded inputs) to n workers asynchronously
#                 send_tasks = [async_send_data(worker_sockets[i], datas[i]) for i in range(n_workers)]
#                 loop.run_until_complete(asyncio.gather(*send_tasks))
#
#                 # finish master task if necessary
#                 if master_task > 0:
#                     conv = layer.conv
#                     m_output_range = (O - master_task, O)
#                     i_s, i_e = output_input(m_output_range, layer_config)
#                     split_input = x_padded[..., i_s:i_e]
#                     master_output = F.conv2d(split_input, conv.weight, stride=conv.stride, padding=0)
#
#                 # receive coded outputs from k workers synchronously
#                 while len(recv_list) < k:
#                     time.sleep(1e-3)
#
#                 if coded:  # decoding
#                     kset = [t[0] for t in recv_list[:k]]
#                     coded_outputs = [t[1] for t in recv_list[:k]]
#                     outputs = decode_conv_MDS(coded_outputs, n, k, G, kset)
#                 else:
#                     recv_list.sort(key=lambda t: t[0])
#                     outputs = [data[1] for data in recv_list]
#
#                 if master_task > 0:
#                     outputs.append(master_output)
#                 print([item.shape for item in outputs])
#                 x = torch.concat(outputs, dim=-1)
#                 x = F.relu(x, inplace=True)
#             else:
#                 x = layer(x)
#             times.append(time.perf_counter())
#         thread_is_working = False
#         print(f'{times[-1] - times[0]}s in total.')
#         print(x.shape)
#         print(times)
#
#
#     # times_workers = loop.run_until_complete(asyncio.gather(*recv_tasks))
#     while len(recv_list[-1]) < n:
#         time.sleep(1e-2)
#     avgs = np.asarray(recv_list[-1]).mean(axis=0)
#     draw_times(times, avgs, layer_indexes)
#     return times


if __name__ == '__main__':
    master_ip = '192.168.1.168'
    worker_socket_timeout = None
    model_name = 'vgg16'
    model = load_model(model_name)
    input_shape = model.input_shape
    x = torch.randn((1, *input_shape))

    distributed = True
    if distributed:
        n = 1  # use at most n workers to collaboratively execute one task
        test_repetition = 4
        fail_num = 0
        random_waiting = True
        lambda_tr = 5
        master = DistributedEngine(master_ip, worker_socket_timeout, model, n)
        master.set_failure(fail_num)
        master.set_n(n)
        master.set_straggler(1)
        master.set_random_waiting(random_waiting, lambda_tr)
        # test_methods = ['uncoded']
        # test_methods = ['coded-3', 'coded-2', 'coded-1']
        test_methods = ['lt']
        # test_methods = ['repetition', 'uncoded', 'coded']
        save = False
        distributed_inference_testing(test_repetition, test_methods, master, save)
        # layer_test()
    else:
        output, layer_latency = dnn_inference_by_layer(x, model.features, model.next)
        print(output.shape)
        print(sum(layer_latency), layer_latency)

    sys.exit(0)
