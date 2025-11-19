import socket
import time
from queue import SimpleQueue
import traceback

import torch

from comm import send_data, recv_data, recv_exact


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


# def keep_recv_output_with_fail(recv_socket: socket.socket, shared_recv_queue: SimpleQueue, fail_queue: SimpleQueue, thread_is_working: bool):
#     while thread_is_working:
#         try:
#             taskID, task_idx, data = recv_data(recv_socket)  # (taskID, task_idx, data)
#             if not isinstance(data, torch.Tensor):  # (taskId, task_idx, 'fail') means fail
#                 print(f'Recv fail response of task {task_idx}')
#                 fail_queue.put((taskID, task_idx))
#             else:  # (taskID, task_idx, output)
#                 shared_recv_queue.put((taskID, (task_idx, data)))
#         except socket.timeout:
#             continue
#         except Exception:
#             traceback.print_exc()
#             break

# def keep_recv_output_with_fail(recv_socket: socket.socket, shared_recv_queue: SimpleQueue, fail_queue: SimpleQueue, thread_is_working):
#     # 将 while thread_is_working 更改为 while thread_is_working()
#     # 并且在异常时检查 thread_is_working() 的状态
#     while thread_is_working():
#         try:
#             # ... (保持原有逻辑) ...
#             taskID, task_idx, data = recv_data(recv_socket)  # (taskID, task_idx, data)
#             if not isinstance(data, torch.Tensor):  # (taskId, task_idx, 'fail') means fail
#                 print(f'Recv fail response of task {task_idx}')
#                 fail_queue.put((taskID, task_idx))
#             else:  # (taskID, task_idx, output)
#                 shared_recv_queue.put((taskID, (task_idx, data)))
#         except socket.timeout:
#             continue
#         except ConnectionError: # 明确捕获连接断开，这是正常退出
#             break
#         except Exception as e:
#             # 捕获所有其他异常 (包括 WinError 10038 / OSError)
#             if thread_is_working():
#                 # 如果主引擎仍认为自己在工作，打印 traceback
#                 traceback.print_exc()
#             # 如果 thread_is_working() 是 False，说明主引擎正在关闭，静默退出
#             break

# def keep_recv_output_with_fail(conn, recv_queue, fail_task_queue, is_working):
#     """使用 socket.settimeout 控制阻塞，兼容 recv_data 返回格式并稳健处理异常"""
#     try:
#         # 设置短超时，保证循环能定期检查 is_working()
#         try:
#             conn.settimeout(1.0)
#         except Exception:
#             pass

#         while is_working():
#             try:
#                 pkt = recv_data(conn)  # 使用库函数，不传 timeout 参数，依赖 socket 超时
#                 if pkt is None:
#                     continue

#                 # 兼容不同返回格式
#                 if isinstance(pkt, tuple):
#                     if len(pkt) == 3:
#                         taskID, task_idx, data = pkt
#                     elif len(pkt) == 2:
#                         taskID, data = pkt
#                         task_idx = None
#                     else:
#                         # 非预期格式，丢弃并继续
#                         continue
#                 else:
#                     continue

#                 # data 不是 tensor 视为 fail 标记
#                 if not isinstance(data, torch.Tensor):
#                     fail_task_queue.put((taskID, task_idx))
#                 else:
#                     recv_queue.put((taskID, (task_idx, data)))

#             except socket.timeout:
#                 # 周期性超时以便检查 is_working()
#                 continue
#             except (ConnectionError, OSError) as e:
#                 if is_working():
#                     print(f"接收数据时发生错误: {e}")
#                 break
#             except Exception as e:
#                 # 其它异常（比如协议不匹配）在仍工作时打印
#                 if is_working():
#                     print(f"接收数据时发生未知错误: {e}")
#                 break
#     finally:
#         # 确保在退出时关闭连接（避免资源泄漏）
#         try:
#             conn.close()
#         except Exception:
#             pass

def keep_recv_output_with_fail(conn, recv_queue, fail_task_queue, is_working):
    """稳健接收：依赖 socket 超时，捕获 WinError 10038，不在此关闭 conn"""
    try:
        try:
            conn.settimeout(1.0)
        except Exception:
            pass

        while True:
            # 先检查退出标志，尽量避免在 socket 已被主线程关闭后再进行 recv
            if not is_working():
                break
            try:
                pkt = recv_data(conn)  # 依赖 socket timeout
                if pkt is None:
                    continue

                if isinstance(pkt, tuple):
                    if len(pkt) == 3:
                        taskID, task_idx, data = pkt
                    elif len(pkt) == 2:
                        taskID, data = pkt
                        task_idx = None
                    else:
                        continue
                else:
                    continue

                if not isinstance(data, torch.Tensor):
                    fail_task_queue.put((taskID, task_idx))
                else:
                    recv_queue.put((taskID, (task_idx, data)))

            except socket.timeout:
                continue
            except OSError as e:
                # Windows: 如果在非套接字上尝试操作 (WinError 10038)，说明主线程已关闭 socket，静默退出循环
                if getattr(e, 'winerror', None) == 10038:
                    break
                # 如果主线程正在关闭，也静默退出
                if not is_working():
                    break
                # 其他情况打印一次错误并退出
                print(f"接收数据时发生错误: {e}")
                break
            except Exception as e:
                if is_working():
                    print(f"接收数据时发生未知错误: {e}")
                break
    finally:
        # 不在此处 close(conn) —— socket 由主线程统一管理和关闭，避免重复 close 导致 WinError/资源竞争
        return

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