# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)

import sys
# sys.path.append(".")
# sys.path.append("..") # Adds higher directory to python modules path.

# Now import FLamingo
from FLamingo.core.client import *


class FedAvgClient(Client):
    """
    FedAvg Client, the original FLamingo Client
    """


if __name__ == '__main__':
    client = FedAvgClient()
    client.run()