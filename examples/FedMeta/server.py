import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='1'


# Set cuda information before importing FLamingo
import sys
sys.path.append('../FLamingo/')
from FLamingo.core.server import *
from model import CNNMNIST, AlexNet


class MetaClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        # define your own information
        self.local_optimization_loss = 0.0
        self.local_optimization_samples = 0
        self.before_test_acc = 0.0
        self.before_test_loss = 0.0
        

class MetaServer(Server):
    def init(self):
        """
        Defining model and related information
        """
        self.network = NetworkHandler()
        self.model = CNNMNIST() if self.dataset_type == 'mnist' else AlexNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.993)

    def average_client_info(self, client_list):
        """
        Average client info and log it
        """
        length = len(client_list)
        # clients = [self.get_client_by_rank(rank) for rank in client_list]
        clients = client_list
        avg_train_loss = 0.0
        avg_test_loss, avg_test_acc = 0.0, 0.0
        avg_before_test_loss, avg_before_test_acc = 0.0, 0.0
        avg_local_optimization_loss, avg_local_optimization_samples = 0.0, 0.0
        for client in clients:
            avg_train_loss += client.train_loss
            avg_test_acc += client.test_acc
            avg_test_loss += client.test_loss
            avg_before_test_acc += client.before_test_acc
            avg_before_test_loss += client.before_test_loss
            avg_local_optimization_loss += client.local_optimization_loss
            avg_local_optimization_samples += client.local_optimization_samples
        self.log(f"Avg global info:\ntrain loss {avg_train_loss/length}, \
                 \ntest acc {avg_test_acc/length}, \
                 \ntest loss {avg_test_loss/length}, \
                 \nbefore test acc {avg_before_test_acc/length}, \
                 \nbefore test loss {avg_before_test_loss/length}, \
                 \nlocal optimization loss {avg_local_optimization_loss/length}, \
                 \nlocal optimization samples {avg_local_optimization_samples/length}")

    def run(self):
        self.init_clients(clientObj=MetaClientInfo)
        while True:
            """
            Server acts the same as before
            """
            self.select_clients()
            self.broadcast(data={'status':'TRAINING'})
            self.listen()
            self.aggregate(weight_by_sample=True)
            self.average_client_info(self.selected_clients)
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                break
        # out of loop
        self.log("Server stopped")
        self.stop_all()


if __name__ == "__main__": 
    server = MetaServer()
    server.run()