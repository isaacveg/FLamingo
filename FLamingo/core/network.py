import pickle
from time import sleep
import time
from mpi4py import MPI
import asyncio


async def send_data(comm, data, dest_rank, tag_epoch):
    data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)  
    comm.send(data, dest=dest_rank, tag=tag_epoch)
    # print("after send")


async def get_data(comm, received, src_rank, tag_epoch):
    data = comm.recv(source=src_rank, tag=tag_epoch)
    received[src_rank] = pickle.loads(data)
    return data


class NetworkHandler():
    """
    Network handler of client.
    """
    def __init__(self):
        world = MPI.COMM_WORLD
        rank = world.Get_rank()
        size = world.Get_size()
        self.comm = world
        self.received_data = {}
        self.rank = rank
        self.size = size
        self.MASTER_RANK = 0
        self.communication_tag = [0 for _ in range(self.size)]
        self.data_sent = [0 for _ in range(self.size)]
        # time
        self.last_send_time = 0.0
        self.last_recv_time = 0.0

    def send(self, data, dest_rank=0):
        """
        Send data to other process, default from server
        """
        s_t, self.last_send_time = time.time(), 0.0
        tag = self.communication_tag[dest_rank]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = asyncio.ensure_future(send_data(self.comm, data, dest_rank, tag))
        loop.run_until_complete(asyncio.wait([task]))
        loop.close()
        self.communication_tag[dest_rank] += 1
        self.last_send_time = time.time() - s_t
        # it is not neccessary to return the time, but you can use it yourself.
        return self.last_send_time

    def get(self, src_rank=0):
        """
        Get data from other process, default from server
        """
        s_t, self.last_recv_time = time.time(), 0.0
        tag = self.communication_tag[src_rank]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = asyncio.ensure_future(get_data(self.comm, self.received_data, src_rank, tag))
        loop.run_until_complete(asyncio.wait([task]))
        loop.close()
        # print(self.received_data[src_rank]['status'])
        self.communication_tag[src_rank] += 1
        self.last_recv_time = time.time() - s_t
        # receive time is not accurate due to the time of waiting for the data,
        # so it is not explicitly used in the code. But you can use it yourself.
        return self.received_data[src_rank]
