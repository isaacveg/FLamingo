# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')

# Now import FLamingo
from FLamingo.client import *


class YourClient(Client):
    """
    Your own Client
    """
    def init(self):
        """
        Init model and network to enable customize these parts.   
        """
        self.network = NetworkHandler()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        self.loss_func = torch.nn.CrossEntropyLoss()


    def YourTrain(self):
        # Train your own model
        return 


    def run(self):
        """
        Client jobs, usually have a loop
        """
        while True:
            # get from server
            data = self.network.get(0)
            if data['status'] == 'TRAINING':
                print('training...')
                # Do something
            
            elif data['status'] == 'STOP':
                print('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        print('stopped')


if __name__ == '__main__':
    client = YourClient()
    client.run()