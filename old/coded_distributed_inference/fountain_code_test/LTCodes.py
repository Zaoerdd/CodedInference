"""
test fountain code
- encode(): generate one encoded task
"""
import asyncio
import threading
from types import SimpleNamespace
import numpy as np
import math
# import random
# from random import choices, choice
import numpy.random as random
import torch
import socket
from comm import *
import torch.nn.functional as F
from queue import SimpleQueue

EPSILON = 0.0001
ROBUST_FAILURE_PROBABILITY = 0.01


def ideal_distribution(N):
    """ Create the ideal soliton distribution.
    In practice, this distribution gives not the best results
    Cf. https://en.wikipedia.org/wiki/Soliton_distribution
    """

    probabilities = [0, 1 / N]
    probabilities += [1 / (k * (k - 1)) for k in range(2, N + 1)]
    probabilities_sum = sum(probabilities)

    assert probabilities_sum >= 1 - EPSILON and probabilities_sum <= 1 + EPSILON, "The ideal distribution should be standardized"
    return probabilities


def robust_distribution(N):
    M = N // 2 + 1
    R = N / M

    extra_proba = [0] + [1 / (i * M) for i in range(1, M)]
    extra_proba += [math.log(R / ROBUST_FAILURE_PROBABILITY) / M]  # Spike at M
    extra_proba += [0 for k in range(M + 1, N + 1)]

    probabilities = np.add(extra_proba, ideal_distribution(N))
    probabilities /= np.sum(probabilities)
    probabilities_sum = np.sum(probabilities)

    assert probabilities_sum >= 1 - EPSILON and probabilities_sum <= 1 + EPSILON, "The robust distribution should be standardized"
    return probabilities


def create_one_hot_np(length, indexes: [int]):
    v = np.zeros((1, length), dtype=float)
    v[:, indexes] = 1
    return v


class Symbol:
    def __init__(self, index, degree, neighbours: set, data):
        assert degree > 0, 'Degree of any encoded symbol should be larger than 0'
        self.index = index
        self.degree = degree
        self.neighbours = neighbours
        self.data = data


class LTCodedTask:
    def __init__(self, tasks, layer_id, worker_sockets=None, loop=None):
        self.source_symbols = tasks
        self.source_num = len(tasks)
        self.population = range(self.source_num + 1)
        self.distribution = robust_distribution(self.source_num)  # len(N+1), probability[1-N]: self.distribution[1:]
        self.max_encoded_num = self.source_num + self.source_num//2
        self.taskID = None
        self.layer_id = layer_id

        # encoded_symbol generator
        self.gen_encoded_symbol = self.generate_encoded_symbol()

        self.generated_encoded_symbols = []
        self.solved_symbols = [None] * self.source_num
        self.solved_cnt = 0
        self.decodable = False
        self.decodable_event = threading.Event()

        self.recv_symbols = []
        self.encoding_matrix = None
        self.__decode__ = 'ge'  # gaussian elimination
        # self.__decode__ = 'bp'  # back propagation

        if worker_sockets is not None:
            self.distributed = True
            self.worker_sockets = worker_sockets
            # self.async_comm_tasks = self.start_comm()
        else:
            self.distributed = False
        self.loop = loop

    async def start_comm(self):
        print('Start async comm tasks')
        # comm_tasks = [asyncio.create_task(self.worker_comm(conn, i)) for i, conn in enumerate(self.worker_sockets)]
        cnts_and_events = [(SimpleNamespace(**{'send_cnt': 0, 'recv_cnt': 0}), asyncio.Event()) for _ in range(len(self.worker_sockets))]
        send_tasks = [asyncio.create_task(self.keep_send_to(conn, i, cnts_and_events[i])) for i, conn in enumerate(self.worker_sockets)]
        recv_tasks = [asyncio.create_task(self.keep_recv_from(conn, i, cnts_and_events[i])) for i, conn in enumerate(self.worker_sockets)]
        await asyncio.gather(*recv_tasks)

    async def keep_send_to(self, conn, worker_idx, send_recv_cnt):
        send_recv_cnt, event = send_recv_cnt
        try:
            while not self.decodable:
                encoded_symbol = next(self.gen_encoded_symbol)
                data = self.taskID, encoded_symbol.index, self.layer_id, encoded_symbol.data
                await async_send_data(conn, data, self.loop)
                send_recv_cnt.send_cnt += 1
                await asyncio.sleep(0)  # exchange to other coroutine
        except StopIteration:
            pass
        finally:
            print(worker_idx, 'No more encoded symbols', send_recv_cnt.send_cnt)
            return

    async def keep_recv_from(self, conn, worker_idx, send_recv_cnt):
        send_recv_cnt, event = send_recv_cnt
        while send_recv_cnt.send_cnt > send_recv_cnt.recv_cnt:  # delete "not self.decodable or"
            _, task_idx, y = await async_recv_data(conn, self.loop)
            send_recv_cnt.recv_cnt += 1
            await asyncio.sleep(0)
            # print(worker_idx, send_recv_cnt.send_cnt, send_recv_cnt.recv_cnt)
            if not self.decodable:
                self.put_recv(task_idx, y)
        # print('Done:', worker_idx, send_recv_cnt.send_cnt, send_recv_cnt.recv_cnt)

    # async def worker_comm(self, conn: socket.socket, worker_idx):
    #     # event = threading.Event()
    #     print(f'{worker_idx} start async comm task')
    #     # semaphore = asyncio.Semaphore(2)
    #     send_semaphore = asyncio.Semaphore(1)  # avoid overlapping of multiple send/recv tasks
    #     recv_semaphore = asyncio.Semaphore(1)
    #     comm_tasks = []
    #     while not self.decodable:
    #         try:
    #             # async with semaphore:
    #                 if not self.decodable:
    #                     print(f'{worker_idx} try to get a encoded symbol')
    #                     encoded_symbol = next(self.gen_encoded_symbol)
    #                     data = self.taskID, encoded_symbol.index, self.layer_id, encoded_symbol.data
    #                     if len(comm_tasks) > 1:
    #                         await comm_tasks[-2]
    #                     print(f'{worker_idx} the -2 task is done, continue')
    #                     comm_task = asyncio.create_task(self.distribute_task(conn, data, send_semaphore, recv_semaphore))
    #                     comm_tasks.append(comm_task)
    #                 else:
    #                     break
    #         except StopIteration:
    #             print('No more symbols, end')
    #             break
    #
    #     await asyncio.gather(*comm_tasks)
    #     print(f'{worker_idx} worker_comm end')
    #
    # async def distribute_task(self, conn: socket.socket, data, send_sema: asyncio.Semaphore, recv_sema: asyncio.Semaphore):
    #     async with send_sema:
    #         await async_send_data(conn, data, self.loop)
    #     async with recv_sema:
    #         _, task_idx, y = await async_recv_data(conn, self.loop)  # = symbol_idx, encoded_data
    #     print(y.shape)
    #     if not self.decodable:
    #         self.put_recv(task_idx, y)

    def put_recv(self, symbol_idx, data):
        self.generated_encoded_symbols[symbol_idx].data = data
        self.recv_symbols.append(self.generated_encoded_symbols[symbol_idx])
        if self.__decode__ == 'ge':  # gaussian elimination
            encoding_vector = create_one_hot_np(self.source_num,
                                                list(self.generated_encoded_symbols[symbol_idx].neighbours))
            if self.encoding_matrix is None:
                self.encoding_matrix = encoding_vector
            else:
                self.encoding_matrix = np.concatenate((self.encoding_matrix, encoding_vector), axis=0)
            if len(self.recv_symbols) >= self.source_num:
                rank = np.linalg.matrix_rank(self.encoding_matrix)
                if rank == self.source_num:
                    # print('====Rank is FULL!!!====', f'recv_size={len(self.recv_symbols)}')
                    self.decodable = True  # full-rank: start decoding
                    self.decodable_event.set()
        else:  # back propagation
            # reduce to solve symbols
            pass

    def generate_encoded_symbol(self):
        for index in range(self.max_encoded_num):
            degree = 1 if index == 0 else random.choice(self.population, size=1, p=self.distribution)

            random.seed(index)  # set seed with index
            selected_neighbours = set(random.choice(self.source_num, degree))
            selected_source = [self.source_symbols[idx] for idx in selected_neighbours]
            encoded_data = sum(selected_source)
            encoded_symbol = Symbol(index, degree, selected_neighbours, encoded_data)
            self.generated_encoded_symbols.append(encoded_symbol)

            yield encoded_symbol

    def reduce_neighbours(self, solved_index, other_symbols):
        for other_symbol in other_symbols:
            if other_symbol.degree > 1 and solved_index in other_symbol.neighbours:
                other_symbol.data -= self.solved_symbols[solved_index]
                other_symbol.neighbours.remove(solved_index)
                other_symbol.degree -= 1

    def reduce(self):
        solved_cnt = 0
        for i, encoded_symbol in enumerate(self.generated_encoded_symbols):
            if encoded_symbol.degree == 1:
                solved_cnt += 1
                solved_src_index = next(iter(encoded_symbol.neighbours))
                self.generated_encoded_symbols.pop(i)  # pop out of the encoded symbols

                if self.solved_symbols[solved_src_index] is not None:
                    continue

                self.solved_symbols[solved_src_index] = encoded_symbol.data
                self.solved_cnt += 1

                self.reduce_neighbours(solved_src_index, self.generated_encoded_symbols)
        print(f'Solved symbols cnt={self.solved_cnt}, solved in this iteration:{solved_cnt}')

    def decode_bp(self):
        pass

    def decode_ge(self):
        """
        Decode symbols using GE (gaussian elimination), start decoding when *rank of encoding matrix is full*.
        """
        Ginv = torch.linalg.pinv(torch.from_numpy(self.encoding_matrix)).to(torch.float)
        coded_outputs = [rs.data for rs in self.recv_symbols]
        output_shape = coded_outputs[0].shape
        Ck = torch.concat(coded_outputs, dim=0)  # k*(1,Co,O,O/k) -> (k,Co,O,O/k)
        Ck = Ck.view(len(self.recv_symbols), -1)  # (k,Co,O,O/k) -> (k,Co*O*O/k)
        decoded_outputs = torch.matmul(Ginv, Ck)
        decoded_outputs = decoded_outputs.view(self.source_num, *output_shape)
        decoded_outputs = [decoded_outputs[i] for i in range(self.source_num)]
        return decoded_outputs
