# master.py
# python .\worker.py --master 127.0.0.1 --model vgg16

import sys
import os
import json
from datetime import datetime
import numpy as np
from comm_util import *
import asyncio
from functions import *
from tqdm import tqdm
from util import save_object, load_object
import numpy.random as random
from comm import *
from socket import create_server, AF_INET

# 导入新的分段模型工具
from model_segments import load_segment_models_dynamically

import time
import threading
import traceback
import queue
from queue import SimpleQueue

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_connections(master_ip: str, timeout=None):
    __port__ = 59999
    listen_addr = (master_ip, __port__)
    server_socket = create_server(listen_addr, family=AF_INET)

    # accept connections from workers
    server_socket.settimeout(8)
    stop_thread = False
    available_workers = []
    print(f'Master ip is {master_ip}, recving connections from workers...')
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
    用于Distributed Coded Convolution (分段编码卷积)
    '''

    def __init__(self, master_ip: str, timeout, model: nn.Module, k_workers: int, r_workers: int):
        '''
        用于初始化master以及建立于workers的连接
        !!! 采用 model_segments.py 中的分段编码逻辑 !!!
        '''
        # accept connections from workers
        worker_sockets, ip_set, n_workers = create_connections(master_ip, timeout)

        self.loop = asyncio.get_event_loop()
        self.model = model  # 原始的完整模型 (例如 VGG16)

        self.k = k_workers
        self.r = r_workers
        self.n = k_workers + r_workers  # 总共需要 n 个 worker
        self.fail_num = 0  # 模拟的故障数量

        if self.n > n_workers:
            raise Exception(f'编码方案需要 n={self.n} 个 worker，但只连接了 {n_workers} 个。')
        
        # 我们只使用 n 个 worker
        self.worker_sockets = worker_sockets
        self.chosen_workers = self.worker_sockets[:self.n] 
        print(f'使用 {self.n} 个 workers (k={self.k}, r={self.r})')

        # --- 新的分段模型加载 ---
        print("正在自动分割模型并创建编码器/解码器...")
        if not hasattr(model, 'model_name') or not hasattr(model, 'input_shape'):
             raise ValueError("模型对象必须具有 'model_name' 和 'input_shape' 属性。")
        
        self.master_models, self.worker_models, self.pooling_layers = \
            load_segment_models_dynamically(
                model.model_name,
                self.k,
                self.r,
                model.input_shape
            )
        print(f"模型加载完毕。共 {len(self.master_models)} 个分段。")
        # ------------------------

        # 将 worker 需要的模型 (BlockWorkerDecoder) 发送给它们
        send_tasks = [async_send_data(self.chosen_workers[i], (i, self.worker_models)) for i in range(self.n)]
        self.loop.run_until_complete(asyncio.gather(*send_tasks))  # send the layer_config to all workers

        recv_queue = SimpleQueue()  # wait for all responses from all workers
        for conn in self.chosen_workers:
            recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
            recv_thread.start()
            recv_thread.join()
        
        if recv_queue.qsize() == self.n:  # workers的回复全部收到，初始化完成
            print('Workers 初始化成功!')
        else:
            print(f'Workers 初始化失败 (收到 {recv_queue.qsize()}/{self.n} 个回复)')
            raise RuntimeError("Workers 初始化失败")

        # 启动接收线程
        self.recv_queue = SimpleQueue()  # recv all responses from workers, the shared recv queue
        self.fail_task_queue = SimpleQueue()
        self.forwarding = False
        self.is_working = True
        self.recv_threads = [
            threading.Thread(target=keep_recv_output_with_fail,
                             args=[self.chosen_workers[worker_id], self.recv_queue, self.fail_task_queue,
                                   lambda: self.is_working]) for worker_id in
            range(self.n)]
        
        self.not_fail_list = list(range(self.n))
        self.forwarding_thread = threading.Thread(target=self.bad_fail_forwarding, args=[]) # 保持故障转发线程
        self.cur_layer_id = None # 将用于存储 block_name
        self.inputs = None # 将用于存储 k+r 个分片

        # for async LT codes
        self.asyncio_thread = threading.Thread(target=self.run_loop)
        self.asyncio_thread.start()
        # for conn in self.worker_sockets:
        #     conn.setblocking(False)

        # 启动所有线程
        self.forwarding_thread.start()
        for t in self.recv_threads:
            t.start()
        
        print("Master 引擎已启动并准备就绪。")


    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()


    def set_failure(self, n_failure_new: int):
        if n_failure_new is not None and n_failure_new != self.fail_num:
            assert n_failure_new <= self.r, f"故障数 ({n_failure_new}) 不能超过冗余数 r ({self.r})"
            self.fail_num = n_failure_new
            print(f"设置模拟故障数: {self.fail_num}")
            # if self.fail_num > 0 and not self.forwarding_thread.is_alive():
            #     self.forwarding_thread.start() # 线程已在 init 中启动


    def end(self):
        """优化后的关闭流程"""
        if not self.is_working:  # 防止重复调用
            return
            
        self.is_working = False  # 设置停止标志
        print("正在关闭 master 引擎...")
        
        # 1. 关闭事件循环
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            
        # 2. 先关闭所有 socket
        for worker_socket in self.worker_sockets:
            try:
                worker_socket.shutdown(socket.SHUT_RDWR)  # 先尝试优雅关闭
                worker_socket.close()
            except Exception:
                pass  # 忽略关闭错误
                
        # 3. 等待转发线程退出
        if hasattr(self, 'forwarding_thread') and self.forwarding_thread.is_alive():
            self.forwarding_thread.join(timeout=3)
                
        # 4. 等待接收线程退出
        for t in self.recv_threads:
            if t.is_alive():
                t.join(timeout=3)  # 给每个线程3秒退出时间
                
        # 5. 清理其他资源
        if hasattr(self, 'recv_queue'):
            while not self.recv_queue.empty():
                try:
                    self.recv_queue.get_nowait()  # 清空队列
                except queue.Empty:
                    break
                    
        # 6. 确保 asyncio 线程退出
        if hasattr(self, 'asyncio_thread') and self.asyncio_thread.is_alive():
            self.asyncio_thread.join(timeout=3)
            
        print("Master 已关闭。")


    def gen_send_tasks(self, taskID, task_name: str):
        """
        生成 n=k+r 个发送任务。
        task_name 是 'block_name' (新) 或 'layer_id' (旧)。
        self.inputs 应该已经被设置为一个包含 n 个张量的列表。
        """
        datas = [(taskID, task_idx, task_name, self.inputs[task_idx]) for task_idx in
                 range(self.n)]  # taskID as the identifier of conv task
        
        if self.fail_num == 0:
            send_tasks = [async_send_data(self.chosen_workers[i], datas[i]) for i in
                          range(self.n)]
        else:  # fail_num > 0
            # 随机选择 fail_num 个 worker 发送 'fail' 任务
            fail_task_idx = random.choice(self.n, self.fail_num, replace=False)
            self.not_fail_list = list(set(range(self.n)).difference(fail_task_idx))
            
            send_tasks = []
            for i in range(self.n):
                if i in fail_task_idx:
                    # (taskID, task_idx, 'fail', data)
                    task_data = (taskID, i, 'fail', self.inputs[i]) 
                else:
                    task_data = datas[i]
                send_tasks.append(async_send_data(self.chosen_workers[i], task_data))

        return send_tasks

    def execute_coded_segment(self, x: torch.Tensor, block_name: str, encoder: nn.Module, final_decoder: nn.Module):
        """
        执行一个完整的编码分段计算 (对应图中的 E -> F_conv -> D)
        """
        self.cur_layer_id = block_name  # 用于故障转发
        self.forwarding = True  # 允许转发失败的任务
        k = self.k
        n = self.n

        detail_latencies = {}

        ## 分块和编码
        start_E = time.perf_counter()

        # --- [START] 修改点 1: 零填充逻辑 ---
        original_H = x.shape[2]
        pad_H = 0
        if original_H % k != 0:
            # 计算需要填充的高度
            required_H = ((original_H // k) + 1) * k
            pad_H = required_H - original_H
            
            print(f"[PAD] {block_name}: 高度 {original_H} 无法被 k={k} 整除。填充 {pad_H} 像素。")

            # 在高度维度 (dim=2) 进行填充: (pad_left, pad_right, pad_top, pad_bottom)
            # 我们只需要填充底部 (pad_bottom)。
            # 注意：PyTorch F.pad 从最后一个维度开始，这里是 (W_left, W_right, H_top, H_bottom)
            # W 维度不变 (0, 0)，H 维度只填充底部 (0, pad_H)
            x_padded = F.pad(x, (0, 0, 0, pad_H), mode='constant', value=0.0)
            x = x_padded # 使用填充后的张量进行分片
            
        # 1. 将输入 x 沿【空间维度】（高度维度 2）分割为 k 份
        # 现在 x.shape[2] % k == 0 应该成立
        
        # 1. 将输入 x 沿【空间维度】（高度维度 2）分割为 k 份
        # 此时 input_H % k == 0 已在外层检查过，可以直接使用 torch.chunk
        try:     
            k_pieces = torch.chunk(x, k, dim=2) # <-- 空间分片 (沿高度 dim=2)
            
        except RuntimeError as e:
            # 如果到达这里，说明外部检查失败或出现意外
            print(f"\n[错误] 无法为 {block_name} 分割输入张量。")
            print(f"输入张量高度: {x.shape[2]}, 目标 k: {k}")
            print(f"错误: {e}")
            raise e
        


        # 2. 使用编码器 E 生成 r 份奇偶校验张量
        parity_pieces = encoder(list(k_pieces))
        
        # 3. self.inputs 是 k 份数据 + r 份校验，共 n 份
        self.inputs = list(k_pieces) + parity_pieces

        # 记录编码时延
        detail_latencies['E_encode'] = time.perf_counter() - start_E # <--- 记录 E 时延
        

        ## 发送任务
        start_send = time.perf_counter()
        # 4. 发送 n=k+r 个任务
        start_or_taskID = time.perf_counter()

        send_tasks = self.gen_send_tasks(start_or_taskID, block_name)
        # 把 gather 包装成一个明确的 coroutine，再交给正在运行的 loop
        async def _gather_and_send(tasks):
            await asyncio.gather(*tasks)
        fut = asyncio.run_coroutine_threadsafe(_gather_and_send(send_tasks), self.loop)
        fut.result()  # 等待发送完成或抛出异常

        detail_latencies['Send_Tasks'] = time.perf_counter() - start_send # <--- 记录 Send 时延

        ## 接收和等待
        start_recv_wait = time.perf_counter()
        # 5. 等待并接收最快的 k 个结果
        recv_list = []
        recv_cnt = 0
        while len(recv_list) < k:
            # recv_data 是 (task_idx, output)
            taskID, recv_data = self.recv_queue.get(timeout=30) 
            recv_cnt += 1
            if taskID == start_or_taskID:
                recv_list.append(recv_data)
            # else:
            #     print(f"收到过期任务 {taskID}，丢弃")

        detail_latencies['Recv_Wait_k'] = time.perf_counter() - start_recv_wait # 记录 Recv Wait 时延

        # consumption = time.perf_counter() - start_or_taskID
        
        # 6. 使用解码器 D 重建

        start_D = time.perf_counter()
        k_indices = [p[0] for p in recv_list]
        k_outputs = [p[1] for p in recv_list]
        reconstructed_output = final_decoder(k_outputs, k_indices)

        detail_latencies['D_decode'] = time.perf_counter() - start_D # <--- 记录 D 时延
        
        # 7. 清理队列中多余的 (n-k) 个结果 (防止阻塞后续任务)
        clean_start = time.perf_counter()
        while recv_cnt < n - self.fail_num:
            _ = self.recv_queue.get(timeout=20)
            recv_cnt += 1
            
        clean_latency = time.perf_counter() - clean_start

        # 总时延
        total_segment_latency = detail_latencies['E_encode'] + detail_latencies['Send_Tasks'] + \
                                detail_latencies['Recv_Wait_k'] + detail_latencies['D_decode']
        detail_latencies['Total_Segment'] = total_segment_latency

        self.forwarding = False # 停止转发此任务

        # --- [START] 修改点 2: 移除填充逻辑 ---
        if pad_H > 0:
            # 移除填充的部分，恢复原始高度
            reconstructed_output = reconstructed_output[:, :, :original_H, :]
        # --- [END] 修改点 2: 移除填充逻辑 ---
        
        return reconstructed_output, detail_latencies # 返回详细时延字典

        # return reconstructed_output, consumption


    def fail_forwarding(self):
        while self.is_working:
            try:
                for fail_cnt in range(self.fail_num):
                    taskID, task_idx = self.fail_task_queue.get(timeout=15)
                    if self.forwarding:
                        x = self.inputs[task_idx]
                        target_idx = self.not_fail_list[fail_cnt % len(self.not_fail_list)] # 转发给一个未失败的
                        send_data(self.chosen_workers[target_idx], (taskID, task_idx, self.cur_layer_id, x))
                        print(f'Forward task {task_idx} to {target_idx}')
            except queue.Empty:
                continue
            except Exception:
                if self.is_working:
                    traceback.print_exc()
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
                        target_idx = self.not_fail_list[fail_cnt % len(self.not_fail_list)]
                        print(f'Forward task {task_idx} to {target_idx}')
                        # 只发送给目标 worker
                        task_data = (taskID, task_idx, self.cur_layer_id, x)
                        asyncio.run_coroutine_threadsafe(
                            async_send_data(self.chosen_workers[target_idx], task_data),
                            loop_this_thread
                        ).result(timeout=5)
                        
            except queue.Empty:
                continue
            except Exception:
                if self.is_working:
                    traceback.print_exc()
                break


def execute_layer(input, layer):
    """
    本地执行 (Master端)，用于池化层和分类器。
    """
    start = time.perf_counter()
    if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        output = layer(input)
    elif isinstance(layer, nn.ReLU):
        output = F.relu(input, inplace=True)
    elif isinstance(layer, nn.Sequential): # 用于分类器
        output = layer(input)
    else:
        raise Exception(f'Unsupported Local Layer Type: {type(layer)}')
    consumption = time.perf_counter() - start
    return output, consumption


# 辅助函数: 在 Master 本地执行卷积块
def execute_conv_block_locally(input, worker_decoder: nn.Module):
    """
    本地执行 (Master端)，用于不可分块的卷积块。
    """
    start = time.perf_counter()
    output = worker_decoder(input)
    consumption = time.perf_counter() - start
    return output, consumption


def dnn_inference_segmented(x: torch.Tensor, model, distributed_engine: DistributedEngine, conv_repeat=1):
    """
    新的推理循环，按分段模型执行，支持分布式/本地混合计算。
    """
    print("开始分段式推理...")
    
    # 从引擎获取分段模型、Worker 配置和池化层
    master_models = distributed_engine.master_models
    worker_models = distributed_engine.worker_models # WorkerDecoder 实例
    pooling_layers = distributed_engine.pooling_layers
    k_workers = distributed_engine.k
    
    # 确保按 'block_1', 'block_2' ... 的顺序执行
    block_names = sorted(master_models.keys()) 
    
    # segment_latencies = {block_name: [] for block_name in block_names}
    detail_segment_latencies = {
        block_name: {'E_encode': [], 'Send_Tasks': [], 'Recv_Wait_k': [], 'D_decode': [], 'Total_Segment': []} 
        for block_name in block_names
    }
    local_op_latencies = []
    total_run_latencies = []
    
    final_output = None

    for r in tqdm(range(conv_repeat)):
        current_input = x
        total_latency_run = 0
        
        # # 1. 遍历所有编码卷积块 (F_Conv)
        # for i, block_name in enumerate(block_names):
        #     encoder, final_decoder = master_models[block_name]
        #     worker_decoder = worker_models[block_name]
            
        #     input_H = current_input.shape[2]
            
        #     # --- 混合计算逻辑 ---
        #     if input_H % k_workers != 0:
        #         # 情况 1: 不可整除，在 Master 端本地执行 F_conv
        #         print(f"[LOCAL] {block_name}: 高度 {input_H} 无法被 k={k_workers} 整除。本地执行 F_Conv。")
                
        #         segment_output, segment_latency = execute_conv_block_locally(
        #             current_input, 
        #             worker_decoder
        #         )
        #         # 本地执行，所有详细时延都为 0，Total_Segment 为本地时延
        #         detail_lats = {
        #             'E_encode': 0.0, 'Send_Tasks': 0.0, 'Recv_Wait_k': 0.0, 
        #             'D_decode': 0.0, 'Total_Segment': segment_latency
        #         }
        #     else:
        #         # 情况 2: 可整除，执行分布式编码计算
        #         # 重新启用原始代码逻辑 (确保 execute_coded_segment 逻辑是正确的)
        #         # segment_output, segment_latency = distributed_engine.execute_coded_segment(
        #         #     current_input, 
        #         #     block_name, 
        #         #     encoder, 
        #         #     final_decoder
        #         # )
        #         segment_output, detail_lats = distributed_engine.execute_coded_segment( # <--- 接收详细时延字典
        #             current_input, 
        #             block_name, 
        #             encoder, 
        #             final_decoder
        #         )
        #         segment_latency = detail_lats['Total_Segment'] # <--- 提取总时延
            # ---------------------

            # 1. 遍历所有编码卷积块 (F_Conv)
        for i, block_name in enumerate(block_names):
            encoder, final_decoder = master_models[block_name]
            # worker_decoder = worker_models[block_name] # 不再需要，因为不再本地执行
            
            # input_H = current_input.shape[2] # 不再需要

            # --- 混合计算逻辑被简化 ---
            # 直接执行分布式编码计算，填充逻辑已移入 execute_coded_segment
            segment_output, detail_lats = distributed_engine.execute_coded_segment( 
                current_input, 
                block_name, 
                encoder, 
                final_decoder
            )
            segment_latency = detail_lats['Total_Segment'] # <--- 提取总时延
            
            # segment_latencies[block_name].append(segment_latency)
            # 记录详细时延
            for key, val in detail_lats.items():
                detail_segment_latencies[block_name][key].append(val)

            total_latency_run += segment_latency
            current_input = segment_output # 更新当前输入为解码或本地执行后的输出
            
            # 2. 在 Master 本地执行池化层 (Pooling)
            if i < len(pooling_layers):
                pooling_layer = pooling_layers[i]
                pool_output, pool_latency = execute_layer(current_input, pooling_layer)
                
                if r == 0: local_op_latencies.append(pool_latency)
                total_latency_run += pool_latency
                current_input = pool_output # 更新输入

        # 3. 在 Master 本地执行分类器 (F_Fc)
        fc_start = time.perf_counter()
        if hasattr(model, 'avgpool'): # 适配 VGG / AlexNet
            current_input = model.avgpool(current_input)
        current_input = torch.flatten(current_input, 1)
        if hasattr(model, 'classifier'):
            final_output = model.classifier(current_input)
        
        fc_latency = time.perf_counter() - fc_start
        if r == 0: local_op_latencies.append(fc_latency)
        total_latency_run += fc_latency
        
        total_run_latencies.append(total_latency_run)

    print("--- 分段式推理完成 ---")
    print(f"总运行 {conv_repeat} 次，平均端到端时延: {np.mean(total_run_latencies):.6f}s")
    
    # 返回最终输出和时延字典
    # return final_output, segment_latencies, local_op_latencies, total_run_latencies
    return final_output, detail_segment_latencies, local_op_latencies, total_run_latencies


def distributed_inference_testing(layer_repetition, methods: list, master: DistributedEngine, save=False):
    print(
        f'test_methods={methods}, repetition={layer_repetition}, n={master.n}, k={master.k}, r={master.r}, fail_num={master.fail_num}')
    
    output, detail_segment_latencies, local_latencies, total_run_latencies = dnn_inference_segmented(x, model,
                                                                         distributed_engine=master,
                                                                         conv_repeat=layer_repetition)
    print(f'最终输出 Shape: {output.shape}')

    # latency summary
    print("\n--- 详细时延总结 (平均值) ---")
    total_segment_latency_sum = 0
    
    # 时延键的显示顺序
    latency_keys = ['E_encode', 'Send_Tasks', 'Recv_Wait_k', 'D_decode', 'Total_Segment']
    
    for block_name in sorted(detail_segment_latencies.keys()):
        print(f"  [Segment] **{block_name}**:")
        block_lats = detail_segment_latencies[block_name]
        
        for key in latency_keys:
            mean_lat = np.mean(block_lats[key])
            print(f"    - {key:<12}: {mean_lat:.6f}s")
            
        total_segment_latency_sum += np.mean(block_lats['Total_Segment'])
    
    print("\n--- Master 本地操作时延 ---")
    print(f"  [Local Ops] (Pooling/FC)\t: {np.sum(local_latencies):.6f}s")
    print("-------------------------")
    print(f"  **总平均端到端时延** \t: {np.mean(total_run_latencies): .6f}s")


    # --- 保存逻辑更新 ---
    if save:
        # 定义结果保存目录
        save_dir = 'test_results' 
        # 检查并创建目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"已创建结果目录: {save_dir}")

        filename_prefix = f'result_n{master.n}_k{master.k}_r{master.r}_f{master.fail_num}_rep{layer_repetition}'
        
        # 1. 准备详细分段时延数据 (detail_segment_latencies)
        json_segment_data = {}
        for block_name, block_lats in detail_segment_latencies.items():
            # 将 NumPy 数组或列表转换为标准的 Python 列表
            json_segment_data[block_name] = {
                key: np.array(val).tolist() for key, val in block_lats.items()
            }
        
        # 保存详细分段时延
        json_filename_details = os.path.join(save_dir, f'{filename_prefix}_segment_details.json')
        with open(json_filename_details, 'w') as f:
            json.dump(json_segment_data, f, indent=4)
        print(f"详细分段时延已保存到 {json_filename_details}")
        
        # 2. 保存本地操作时延 (local_latencies)
        json_filename_local = os.path.join(save_dir, f'{filename_prefix}_local_ops.json')
        with open(json_filename_local, 'w') as f:
            json.dump(np.array(local_latencies).tolist(), f, indent=4)
        print(f"本地操作时延已保存到 {json_filename_local}")

        # 3. 保存总运行时间 (total_run_latencies)
        json_filename_total = os.path.join(save_dir, f'{filename_prefix}_total_run.json')
        with open(json_filename_total, 'w') as f:
            json.dump(np.array(total_run_latencies).tolist(), f, indent=4)
        print(f"总运行时间已保存到 {json_filename_total}")

    master.end()
    print("测试完成，Master 已关闭。")


# --- Main ---
if __name__ == '__main__':
    master_ip = '192.168.1.168'
    # master_ip = '127.0.0.1'
    worker_socket_timeout = None
    model_name = 'vgg16'
    
    model = load_model(model_name)
    # 动态地给模型对象添加必要的属性
    model.model_name = model_name 
    model.input_shape = (1, 3, 224, 224) # VGG16 standard input
    x = torch.randn(model.input_shape)

    print(f"模型 {model_name} 加载完毕，准备开始分布式推理测试。")

    distributed = True
    if distributed:
        # --- 编码参数 ---
        # 如果 k=1, 那么 "k_pieces" 就是 1 个完整的张量。
        # (k=1, r=1) 表示系统总共 2 个 worker，可以容忍 1 个失败。
        k = 4  # 数据分片数
        r = 2  # 奇偶校验分片数
        # ----------------
        
        n_total = k + r
        test_repetition = 10
        fail_num = 1      # 模拟故障数
        assert fail_num <= r, f"故障数(fail_num={fail_num}) 必须小于等于冗余数(r={r})"

        print(f"\n--- 分布式推理测试 (n={n_total}, k={k}, r={r}, fail_num={fail_num}) ---\n")

        master = DistributedEngine(master_ip, worker_socket_timeout, model, 
                                   k_workers=k, r_workers=r)
        master.set_failure(fail_num)

        test_methods = ['segment_coded'] # 仅用于日志记录
        save = True
        
        try:
            distributed_inference_testing(test_repetition, test_methods, master, save)
        except Exception as e:
            print("\n--- 发生未捕获的异常 ---")
            traceback.print_exc()
            master.end() # 确保在异常时关闭

    else:
        # 本地推理 (旧逻辑)
        # output, layer_latency = dnn_inference_by_layer(x, model.features, model.next)
        print("本地推理模式未更新。")

    sys.exit(0)