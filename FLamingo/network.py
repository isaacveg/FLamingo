from math import e
import pickle
from mpi4py import MPI
import asyncio


async def send_data(comm, data, dest_rank, tag_epoch):
    data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)  
    comm.send(data, dest=dest_rank, tag=tag_epoch)

async def get_data(comm, received, src_rank, tag_epoch):
    s_t = MPI.Wtime()
    data = comm.recv(source=src_rank, tag=tag_epoch)
    received[src_rank] = pickle.loads(data)
    e_t = MPI.Wtime()
    return e_t - s_t


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
        self.bytes_sent = [0 for _ in range(self.size)]
        # time
        self.last_send_time = 0.0
        self.last_recv_time = 0.0
        # event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def send(self, data, dest_rank=0):
        """
        Send data to other process, default from server.
        Args:
            data: data to be sent
            dest_rank: destination rank, default 0
        """
        tag = self.communication_tag[dest_rank]
        self.loop.run_until_complete(send_data(self.comm, data, dest_rank, tag))
        self.communication_tag[dest_rank] += 1

    def get(self, src_rank=0):
        """
        Get data from other process, default from server.
        Args:
            src_rank: source rank, default 0
        Returns:
            received data
        """
        tag = self.communication_tag[src_rank]
        self.last_recv_time = self.loop.run_until_complete(get_data(self.comm, self.received_data, src_rank, tag))
        self.communication_tag[src_rank] += 1
        return self.received_data[src_rank]

    def close(self):
        self.loop.close()
