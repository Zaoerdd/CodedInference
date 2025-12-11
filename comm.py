import asyncio
import pickle
import time
from queue import SimpleQueue
import socket
import struct
from torch import Tensor
from subprocess import getstatusoutput
from sys import getsizeof
from typing import List, Tuple  # <-- ADD THIS LINE

__buffer__ = 65535
__timeout__ = 10


def ping(dest: str, ping_cnt: int, timeout: int):  # '1' means connected, else 0
    assert 0 < ping_cnt < 10
    status, output = getstatusoutput(
        f"ping {dest} -c {ping_cnt} -w {timeout} | tail -2 | head -1 | awk {{'print $4'}}")  # send one packet and check the recv packet
    print(f'ping result {output}')
    if status == 0 and int(output) > 0:
        return True  # ping得通
    else:
        return False


data_header_format = 'I'
data_header_size = struct.calcsize(data_header_format)


def accept_connection(server_socket: socket, recv_list: List, stop, timeout=None):
    while True:
        if stop():
            break
        try:
            conn, addr = server_socket.accept()
            conn.settimeout(timeout)
            recv_list.append((conn, addr))  # keep full addr (ip, port) so separate workers on same IP are distinct
            print(f'Recv connection from {addr}')
        except socket.timeout:
            continue
        except Exception as e:
            print(e)


def put_recv_data(conn: socket, recv_queue: SimpleQueue):
    try:
        data = recv_data(conn)
        recv_queue.put(data)
    except socket.timeout:
        print('Recv data time out')
    except Exception as e:
        print(e)


def connect_to_other(ip_list: List, port: int, socket_list: list, self_ip):
    try:
        for i, worker_ip in enumerate(ip_list):
            if worker_ip != self_ip:
                addr = worker_ip, port
                conn = socket.create_connection(addr, timeout=5)
                print(f'Connected to {worker_ip}')
                socket_list.append((conn, addr[0]))
    except socket.timeout:
        print('Create connections timeout')
    except Exception as e:
        print(e)


def send_data(send_socket: socket, data, waiting=0):
    if not isinstance(data, (bytes, bytearray)):
        data = pickle.dumps(data)
    data_size = len(data)
    header = struct.pack(data_header_format, data_size)
    if waiting:
        time.sleep(waiting)
    res = send_socket.sendall(header + data)
    return res

def send_all(sock: socket.socket, data: bytes, timeout: float = 30) -> None:
    """
    可靠的阻塞发送完整数据。
    """
    total_sent = 0
    size = len(data)
    sock.settimeout(timeout)  # 设置超时保护
    
    try:
        while total_sent < size:
            sent = sock.send(data[total_sent:])
            if sent == 0:
                raise ConnectionError("发送数据时连接断开")
            total_sent += sent
    except socket.timeout:
        raise TimeoutError(f"发送数据超时 (已发送 {total_sent}/{size} 字节)")
    # finally:
    #     sock.settimeout(None)  # 恢复默认超时设置

def recv_exact(sock: socket.socket, size: int, timeout: float = 30) -> bytes:
    """
    可靠的阻塞接收指定字节数的数据。
    """
    data = bytearray(size)
    view = memoryview(data)
    total_recv = 0
    sock.settimeout(timeout)  # 设置超时保护
    
    try:
        while total_recv < size:
            nbytes = sock.recv_into(view[total_recv:], size - total_recv)
            if nbytes == 0:  # 连接关闭
                raise ConnectionError("接收数据时连接断开")
            total_recv += nbytes
    except socket.timeout:
        raise TimeoutError(f"接收数据超时 (已接收 {total_recv}/{size} 字节)")
    # finally:
    #     sock.settimeout(None)  # 恢复默认超时设置
    
    return bytes(data)

async def async_send_data(send_socket: socket.socket, data, loop=None, waiting: float = 0):
    """
    异步发送数据。使用线程池执行阻塞的发送操作。
    
    Args:
        send_socket: 发送用的 socket
        data: 要发送的数据（会被 pickle 序列化）
        loop: 可选的事件循环对象
        waiting: 发送前等待的秒数
    """
    if loop is None:
        loop = asyncio.get_running_loop()
    
    if waiting > 0:
        await asyncio.sleep(waiting)
    
    # 序列化数据
    if not isinstance(data, (bytes, bytearray)):
        data = pickle.dumps(data)
    
    # 准备头部
    data_size = len(data)
    header = struct.pack(data_header_format, data_size)
    payload = header + data
    
    # 在线程池中执行阻塞发送
    try:
        await loop.run_in_executor(None, send_all, send_socket, payload)
    except (ConnectionError, TimeoutError) as e:
        print(f"发送数据失败: {e}")
        raise
    except Exception as e:
        print(f"发送数据时发生未知错误: {e}")
        raise


def recv_data(recv_socket: socket.socket) -> any:
    """
    从 socket 接收数据并反序列化。
    支持超时和连接断开的处理。
    """
    try:
        # 1. 接收固定大小的头部
        header_data = recv_exact(recv_socket, data_header_size)
        data_size = struct.unpack(data_header_format, header_data)[0]
        
        # 2. 接收数据体
        data = recv_exact(recv_socket, data_size)
        
        # 3. 反序列化
        return pickle.loads(data)
        
    except (ConnectionError, TimeoutError) as e:
        print(f"接收数据失败: {e}")
        raise
    except Exception as e:
        print(f"接收数据时发生未知错误: {e}")
        raise


async def async_recv_data1(recv_socket: socket.socket, loop=None):
    if loop is None:
        loop = asyncio.get_running_loop()
    # msg = recv_socket.recv(data_header_size)
    msg = await loop.sock_recv(recv_socket, data_header_size)
    header = struct.unpack(data_header_format, msg)
    data_size = header[0]
    data = bytearray(data_size)
    ptr = memoryview(data)
    # data = b''
    sofar = 0
    print(f'Start Recv {data_size} bytes')
    while sofar < data_size:
        ptr = ptr[sofar:]
        nrecv = await loop.sock_recv_into(recv_socket, ptr)
        sofar += nrecv
    print(f'Finish Recv {data_size} bytes')
    return pickle.loads(data)


async def async_recv_data(recv_socket: socket.socket, loop=None):
    if loop is None:
        loop = asyncio.get_running_loop()

    msg = await loop.sock_recv(recv_socket, data_header_size)
    header = struct.unpack(data_header_format, msg)
    recv_size = header[0]
    # data = bytearray(recv_size)
    data = b''
    # ptr = memoryview(data)
    while recv_size:
        recv = await loop.sock_recv(recv_socket, recv_size)
        data += recv
        recv_size -= len(recv)
    return pickle.loads(data)


__format__ = 'IHHH'  # header: (data size, layer num, left range, right range]
__size__ = struct.calcsize(__format__)


def send_tensor(send_socket: socket, data: Tensor, layer_num: int, data_range: Tuple[int, int]):
    """
    send the padding data of intermediate result of certain layer to other device
    :param send_socket: the socket of UDP protocol to send data
    :param data: the (Tensor type) data to send
    :param layer_num: the intermediate result of which layer
    :param data_range: the corresponding range of data to the layer output
    :return: the sending status
    """
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    header = struct.pack(__format__, data_size, layer_num, *data_range)
    res = send_socket.sendall(header + serialized)
    return res


async def async_send_tensor(send_socket: socket, data: Tensor, layer_num: int, data_range: Tuple[int, int]):
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    print(f'send output from layer {layer_num} of size {data_size/1024}KB')
    header = struct.pack(__format__, data_size, layer_num, *data_range)
    res = send_socket.sendall(header + serialized)
    return res


def recv_tensor(recv_socket: socket):
    data = b''
    header_size = __size__
    while header_size:
        recv = recv_socket.recv(header_size)
        data += recv
        header_size -= len(recv)
    header = struct.unpack(__format__, data)
    data_size = header[0]
    data = b''
    while data_size:
        recv = recv_socket.recv(min(4096, data_size))
        data += recv
        data_size -= len(recv)
    return (header[1], (header[2], header[3])), pickle.loads(data)  # (layer_num, range) data


async def async_recv_tensor(recv_socket: socket):
    data = b''
    header_size = __size__
    while header_size:
        recv = recv_socket.recv(header_size)
        data += recv
        header_size -= len(recv)
    header = struct.unpack(__format__, data)
    data_size = header[0]
    data = b''
    while data_size:
        recv = recv_socket.recv(min(4096, data_size))
        data += recv
        data_size -= len(recv)
    return (header[1], (header[2], header[3])), pickle.loads(data)  # (layer_num, range) data
