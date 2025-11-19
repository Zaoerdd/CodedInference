import argparse
import traceback
from tqdm import tqdm
import struct
import threading
import time
from multiprocessing import Process, Pipe
from socket import create_connection
from queue import SimpleQueue
from comm import send_data, recv_data
from models.googlenet import BasicConv2d
import torch.nn.functional as F
from util import get_ip_addr, load_model

__subnet__ = '192.168.1'
__port__ = 49999
# master_ip = '192.168.1.122'
__worker_port__ = 50000

__format__ = 'IH'  # header: (data_size, layer_num)
__size__ = struct.calcsize(__format__)


class Worker:
    """
    Worker continuously accept task with encoded input and simply execute the task and send back results
    """

    def __init__(self, child_pipe: Pipe, master_ip=None, model_name=None):
        # self.number = None
        self.master_socket = None
        self.child_pipe = child_pipe
        print(f'Load {model_name} on worker')
        self.model = load_model(model_name)
        if master_ip is None:
            print('No master to connect!')
        else:
            self.master_ip = master_ip
        self.model.eval()  # keep the parameters of model unchanged
        self.layers = self.model.layers  # can generate the layer configuration in advance
        # self.layer_configs = generate_layerConfigs(self.layers)
        self.task_queue = None
        self.send_queue = None
        self.ip = get_ip_addr(__subnet__)
        self.time_records = []
        self.initialized = True
        self.sleep = False
        self.fail = True
        # self.fail_probability = 0.05

        # if self.ip is None:
        #     print('Fail to acquire ip, try again...')
        #     self.ip = get_ip_addr(__subnet__)
        #     print(f'Acquired ip: {self.ip}')
        #     if self.ip is None:
        #         print('Fail to acquire ip, exit')
        #         self.initialized = False
        #         return

        # print(f'ip_addr: {self.ip}')

    def start_inference(self):
        if not self.initialized:
            print('Cannot start the worker without initialization!')
            return

        # connect to master
        print(f'Connected to master(ip: {self.master_ip})...')
        master_socket = None
        try:
            master_socket = create_connection(address=(self.master_ip, __port__))
        except Exception as e:
            print('Error occurred when connecting to master:\n' + e.__str__())
        if master_socket is not None:
            print('Connected to master')
            self.master_socket = master_socket
        else:
            print('Fail to connect to master!')
            return

        index, layer_configs = recv_data(master_socket)
        print(f'Worker {index} receive initial data from master')
        # print(layer_configs)
        send_data(master_socket, 'ready')

        self.task_queue = SimpleQueue()
        self.send_queue = SimpleQueue()

        recv_thread = threading.Thread(target=self.receive_task_thread, args=[self.master_socket])
        recv_thread.start()
        send_thread = threading.Thread(target=self.send_data_thread, args=[self.master_socket])
        send_thread.start()
        start = end = 0

        if self.fail:
            print(f'Start Serving with Possible Failure')
            while True:
                task_data = self.task_queue.get()
                if len(task_data) == 4:
                    taskID, task_idx, layer_idx, x = task_data
                    if layer_idx == 'fail':
                        self.send_queue.put((taskID, task_idx, 'fail'))
                        print(f'Task {task_idx} fail')
                        continue
                    else:
                        print(f'Execute task {task_idx}')
                        start = time.time()
                        layer = self.layers[layer_idx]
                        layer_config = layer_configs[layer_idx]
                        y = self.execute_conv(x, layer_config, layer)
                        consumption = time.time() - start
                        print(f'Finish in {consumption} seconds.')
                        # self.time_records.append(consumption)
                        self.send_queue.put((taskID, task_idx, y))
                        # send_data(self.master_socket, (taskID, task_idx, y))
                elif task_data == 0:
                    break
                else:  # 用于接收没用的传输：用于一传多时限制传输速度
                    continue

        else:
            print('Start Normal Serving')
            while True:
                data = self.task_queue.get()
                if data == 0:
                    break
                taskID, task_idx, layer_idx, data = self.task_queue.get()  # recv taskID, task_idx, layer_idx, data
                print([type(item) for item in data])
                print(f'Execute task')
                # BasicConv2d set as default layer that contains linear computation
                start = time.time()
                # self.time_records.append(time.time())
                layer = self.layers[layer_idx]
                layer_config = layer_configs[layer_idx]
                # assert isinstance(layer, BasicConv2d)
                # y = F.conv2d(data, layer.conv.weight, stride=1, padding=1)
                y = self.execute_conv(data, layer_config, layer)
                # y = F.batch_norm(y, )
                consumption = time.time() - start
                print(f'Finish in {consumption} seconds.')
                self.time_records.append(consumption)

                data2send = (taskID, task_idx, y)
                # self.send_queue.put((index, y))
                send_data(master_socket, data2send)
                # 应该不用发时间？
                # if layer_num == layer_indexes[-1]:
                #     self.send_queue.put((-1, self.time_records))

    def start_repetition_task(self):
        # connect to master
        print(f'Connected to master(ip: {self.master_ip})...')
        master_socket = None
        try:
            master_socket = create_connection(address=(self.master_ip, __port__))
        except Exception as e:
            print('Error occurred when connecting to master:\n' + e.__str__())
        if master_socket is not None:
            print('Connected to master')
            self.master_socket = master_socket
        else:
            print('Fail to connect to master!')
            return

        try:
            layer_configs, layer_indexes, layers_to_test, repetition, n = recv_data(master_socket)
            for layer_num in layers_to_test:
                print(f'Test for layer={layer_num}')
                layer = self.layers[layer_num]
                layer_config = layer_configs[layer_num]
                for k in range(2, n):
                    print(f'  Test for k={k}')
                    for r in tqdm(range(repetition)):
                        x = recv_data(master_socket)
                        start = time.time()
                        y = self.execute_conv(x, layer_config, layer)
                        consumption = time.time() - start
                        send_data(master_socket, y)
                        print(f'Finish in {consumption} seconds.')
        except Exception as e:
            print(e)

    def receive_task_thread(self, recv_socket):  # put received task to worker's task queue
        while True:
            try:
                data = recv_data(recv_socket)
                print('Receive data')
                if data == 'end':  # 收到master的消息
                    print('Recv "end" from master')
                    self.task_queue.put(0)
                    self.send_queue.put(0)
                    time.sleep(0.1)
                    print('Send 0 to parent process')
                    self.child_pipe.send(0)
                    break
                self.task_queue.put(data)
                print('Put recv task in queue')
            except Exception:
                traceback.print_exc()
                break

    def send_data_thread(self, send_socket):
        while True:
            try:
                data = self.send_queue.get()

                send_data(send_socket, data)
                print('Send successfully')
            except Exception:
                traceback.print_exc()
                break

    def execute(self, x, layer_config, layer):
        if layer_config['type'] == 'conv':
            output = F.conv2d(x, layer.weight, stride=layer_config['stride'], padding=0)
            return output
        elif layer_config['type'] == 'basicConv':
            output = F.conv2d(x, layer.conv.weight, stride=layer_config['stride'], padding=0)
            return output
        else:
            print('Invalid layer type.')
            return None

    def execute_conv(self, x, layer_config, layer):
        if isinstance(layer, BasicConv2d):
            layer = layer.conv
        y = F.conv2d(x, layer.weight, stride=layer_config['stride'], padding=0)
        return y

def start_worker(child_conn, master_ip, model_name):
    w = Worker(child_conn, master_ip, model_name)
    w.start_inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master', type=str, required=True)
    parser.add_argument('--model', type=str, required=False, default='vgg16')
    args = parser.parse_args()
    master_ip = args.master
    model_name = args.model

    parent_conn, child_conn = Pipe()
    worker_process = Process(target=start_worker, args=(child_conn, master_ip, model_name))
    worker_process.start()
    pid = worker_process.pid

    try:
        data = parent_conn.recv()  # 收到子进程的关系消息，则关闭子进程
        time.sleep(1)
        # if data == 0:
    finally:
        worker_process.terminate()
        worker_process.join()

    # print(master_ip)
    # w = Worker(master_ip, model_name)
    # w.start_inference()
    # w.start_repetition_task()