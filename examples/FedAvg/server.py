# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.

# Now import FLamingo
from FLamingo.core.server import *


class FedAvgServer(Server):
    """
    FedAvg Server, the original FLamingo Server
    """


if __name__ == "__main__":
    server = FedAvgServer()
    server.run()