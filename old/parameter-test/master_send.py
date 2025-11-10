import argparse
import asyncio
import sys
from socket import socket, create_server, AF_INET
import numpy as np
from comm import accept_connection, async_send_data, async_recv_data, send_data, recv_data
import threading
import time
from util import load_model
from tqdm import tqdm
import pickle


def repe_recv_time(recv_socket: socket, recv_list: list, data_size: int, repetition: int):
    for _ in range(repetition):
        _data = recv_data(recv_socket)
        end = time.time()  # recv_time on master
        assert len(_data) == data_size
        recv_list.append(end)


repetition = 100
__port__ = 49999
# self_ip, port = '192.168.1.114', __port__
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ip', type=str, required=True)
# parser.add_argument('-p', '--port', type=int)
args = parser.parse_args()
self_ip = args.ip

# connect
server_socket = create_server((self_ip, __port__), family=AF_INET)
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

# collect connections
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
if n_workers < 1:
    sys.exit(0)

records = []
loop = asyncio.get_event_loop()

model = load_model('vgg16')

shapes = [(1, *model.input_shape), *model.output_shapes]
# sizes = [np.prod(shape) * 4 for shape in shapes]
# print(sizes)

try:
    for idx, shape in enumerate(shapes):  # 少测几次
        feature_size = np.prod(shape) * 4

        print(f'Repetition {idx}: size {feature_size}, k=[2-10]')
        for k in range(2, 10):
            master_send_time = []
            master_recv_time = [[] for i in range(n_workers)]

            data_size = feature_size // k
            data = b''.zfill(data_size)

            # 这算是同时发的吧
            send_tasks = [async_send_data(worker_socket, (feature_size, k, repetition)) for worker_socket in
                          worker_sockets]
            loop.run_until_complete(asyncio.gather(*send_tasks))

            recv_threads = [
                threading.Thread(target=repe_recv_time, args=[worker_sockets[i], master_recv_time[i], data_size, repetition])
                for i in range(n_workers)]
            for recv_thread in recv_threads:
                recv_thread.start()

            time.sleep(5)

            for r in tqdm(range(repetition)):
                # while np.min([len(mrt) for mrt in master_recv_time]) < r:  # 等全部收完再发下一次
                #     continue
                # for mrt in master_recv_time:
                #     assert len(mrt) == r
                send_tasks = [async_send_data(worker_socket, data) for worker_socket in worker_sockets]
                master_send_time.append(time.time())
                loop.run_until_complete(asyncio.gather(*send_tasks))
                time.sleep(1)

            for recv_thread in recv_threads:
                recv_thread.join()

            recv_tasks = [async_recv_data(worker_socket) for worker_socket in worker_sockets]
            recvs = loop.run_until_complete(asyncio.gather(*recv_tasks))

            workers_recv_time, workers_send_time = [], []
            assert len(recvs) == n_workers
            for recv_time, send_time in recvs:
                # for i in range(n_workers):
                assert len(recv_time) == len(send_time) == repetition
                workers_recv_time.append(np.asarray(recv_time))
                workers_send_time.append(np.asarray(send_time))

            for master_recv in master_recv_time:
                assert len(master_recv) == repetition
                master_recv = np.asarray(master_recv)

            master_send_time = np.asarray(master_send_time)
            send_time = [worker_recv_time - master_send_time for worker_recv_time in workers_recv_time]
            recv_time = [master_recv_time[i] - workers_send_time[i] for i in range(n_workers)]
            round_trip_time = [master_recv_time[i] - master_send_time for i in range(n_workers)]

            records.append((data_size, (send_time, recv_time, round_trip_time)))
except Exception as e:
    print(e)
    for w in worker_sockets:
        w.close()
    sys.exit(0)

with open(f'transmission{repetition}_device{n_workers}.rec', 'wb') as record_file:
    pickle.dump(records, record_file)
