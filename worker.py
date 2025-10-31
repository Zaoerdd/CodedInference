# worker.py

import argparse
import traceback
from tqdm import tqdm
import struct
import threading
import time
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from socket import create_connection
from queue import SimpleQueue
from comm import send_data, recv_data
# from models.googlenet import BasicConv2d # 不再需要
# import torch.nn.functional as F # 不再需要
from util import get_ip_addr, load_model

__subnet__ = '192.168.1'
__port__ = 59999
# master_ip = '192.168.1.122'
__worker_port__ = 50000

__format__ = 'IH'  # header: (data_size, layer_num)
__size__ = struct.calcsize(__format__)


class Worker:
    """
    Worker 接收 Master 发来的分段模型 (BlockWorkerDecoder)，
    然后根据任务指令执行对应的分段模型。
    """

    def __init__(self, child_pipe: Connection, master_ip=None, model_name=None):
        self.master_socket = None
        self.child_pipe = child_pipe
        self.master_ip = master_ip
        print(f'Worker 准备就绪 (用于 {model_name})')
        
        # --- 不再加载完整模型 ---
        # self.model = load_model(model_name)
        # self.layers = self.model.layers
        # self.layer_configs = generate_layerConfigs(self.layers)
        
        # worker_models 将是一个字典, e.g., {'block_1': BlockWorkerDecoder(...)}
        self.worker_models = None 
        
        self.task_queue = None
        self.send_queue = None
        self.ip = get_ip_addr(__subnet__)
        self.time_records = []
        self.initialized = True
        self.sleep = False
        self.fail = True # 保持这个标志，用于模拟故障

        if self.ip is None:
            print('Fail to acquire ip, try again...')
            self.ip = get_ip_addr(__subnet__)
            if self.ip is None:
                print('Fail to acquire ip, exit')
                self.initialized = False
                return

        print(f'ip_addr: {self.ip}')

    def start_inference(self):
        if not self.initialized:
            print('Cannot start the worker without initialization!')
            return

        # connect to master
        print(f'Connecting to master(ip: {self.master_ip})...')
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

        # --- 接收分段模型 ---
        try:
            index, self.worker_models = recv_data(master_socket)
            print(f'Worker {index} 收到 {len(self.worker_models)} 个模型分段。')
            
            # 确保所有模型都在评估模式
            if self.worker_models:
                for model_segment in self.worker_models.values():
                    model_segment.eval()
            
            send_data(master_socket, 'ready')
        except Exception as e:
            print("接收模型分段失败:")
            traceback.print_exc()
            master_socket.close()
            return
        # --------------------

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
                    # 任务格式: (taskID, task_idx, block_name, x)
                    taskID, task_idx, block_name, x = task_data
                    
                    if block_name == 'fail':
                        self.send_queue.put((taskID, task_idx, 'fail'))
                        print(f'Task {task_idx} fail')
                        continue
                    else:
                        print(f'Execute task {task_idx} (Segment: {block_name})')
                        start = time.time()
                        
                        # --- 执行分段模型 ---
                        try:
                            block_model = self.worker_models[block_name]
                            y = block_model(x)
                        except Exception as e:
                            print(f"执行 {block_name} 失败: {e}")
                            traceback.print_exc()
                            # 发送失败信号
                            self.send_queue.put((taskID, task_idx, 'fail'))
                            continue
                        # ---------------------
                        
                        consumption = time.time() - start
                        print(f'Finish in {consumption:.6f} seconds.')
                        self.send_queue.put((taskID, task_idx, y))
                        
                elif task_data == 0:
                    break
                else:  # 用于接收没用的传输
                    continue

        else:
            # 这部分逻辑与上面 'if self.fail' 基本重复，可以删除
            print('Start Normal Serving (逻辑已合并到 self.fail=True)')
            pass


    def receive_task_thread(self, recv_socket):  # put received task to worker's task queue
        while True:
            try:
                data = recv_data(recv_socket)
                # print('Receive data')
                if data == 'end':  # 收到master的消息
                    print('Recv "end" from master')
                    self.task_queue.put(0)
                    self.send_queue.put(0)
                    time.sleep(0.1)
                    print('Send 0 to parent process')
                    self.child_pipe.send(0)
                    break
                self.task_queue.put(data)
                # print('Put recv task in queue')
            except Exception:
                # traceback.print_exc()
                print("Recv 线程退出。")
                break

    def send_data_thread(self, send_socket):
        while True:
            try:
                data = self.send_queue.get()
                if data == 0:
                    print("Send 线程退出。")
                    break
                    
                send_data(send_socket, data)
                # print('Send successfully')
            except Exception:
                # traceback.print_exc()
                print("Send 线程退出。")
                break

    # --- execute 和 execute_conv 不再需要 ---
    # def execute(self, x, layer_config, layer):
    # def execute_conv(self, x, layer_config, layer):

# ... (main 函数保持不变) ...

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