import socket
import time
from queue import SimpleQueue
import traceback

import torch

from comm import send_data, recv_data


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


def recv_output(recv_socket: socket.socket, recv_list, thread_is_working: bool):
    try:
        while thread_is_working:
            data = recv_data(recv_socket)  # (index, data | 'fail')
            recv_list.append(data)
            time.sleep(1e-3)
    except Exception as e:
        traceback.print_exc()


def recv_output_with_fail(recv_socket: socket.socket, recv_list, thread_is_working: bool, fail_list: SimpleQueue):
    try:
        while thread_is_working:
            data = recv_data(recv_socket)
            if data[1] == 'fail':  # recv_data: (failed_idx, 'fail')
                failed_idx, _ = data
                fail_list.put(failed_idx)
                # print(f'Recv Fail from worker {failed_idx}')

            else:  # recv_data: (task_idx, output)
                recv_list.append(data)
            time.sleep(1e-3)
    except Exception as e:
        traceback.print_exc()


def keep_recv_output_with_fail(recv_socket: socket.socket, shared_recv_queue: SimpleQueue, fail_queue: SimpleQueue, thread_is_working: bool):
    while thread_is_working:
        try:
            taskID, task_idx, data = recv_data(recv_socket)  # (taskID, task_idx, data)
            if not isinstance(data, torch.Tensor):  # (taskId, task_idx, 'fail') means fail
                print(f'Recv fail response of task {task_idx}')
                fail_queue.put((taskID, task_idx))
            else:  # (taskID, task_idx, output)
                shared_recv_queue.put((taskID, (task_idx, data)))
        except socket.timeout:
            continue
        except Exception:
            traceback.print_exc()
            break

def keep_recv_output(worker_id: int, recv_socket: socket.socket, shared_recv_queue: SimpleQueue, thread_is_working: bool):
    while thread_is_working:
        try:
            data = recv_data(recv_socket)  # (taskId, data)
            shared_recv_queue.put((worker_id, data))
        except socket.timeout:
            continue
        except Exception:
            traceback.print_exc()
            break