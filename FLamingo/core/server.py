import os
import sys
from mpi4py import MPI

# WORLD = MPI.COMM_WORLD
# rank = WORLD.Get_rank()
# size = WORLD.Get_size()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import numpy as np
import random
from copy import deepcopy
import time

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")


from FLamingo.core.utils.args_utils import get_args
from FLamingo.core.utils.data_utils import ClientDataset
from FLamingo.core.utils.model_utils import create_model_instance
from FLamingo.core.utils.chores import log, merge_several_dicts
from FLamingo.core.network import NetworkHandler


class ClientInfo():
    """
    Basical client info object, can update items using dic
    """
    def __init__(self, rank):
        self.rank = rank
        self.status = 'TRAINING'
        self.global_round = 0
        # train status
        self.train_loss = 0.0
        self.train_time = 0.0
        self.train_samples = 0
        # test status
        self.test_time = 0.0
        self.test_loss = 0.0
        self.test_acc = 0.0
        self.test_samples = 0
        # params and weight
        self.params = None
        self.weight = 0.0

    def update(self, data):
        """
        Update items in client info
        Args:
            data: dict containing items. 
        """
        for k,v in data.items():
            assert hasattr(self, k), f"{k} not in client info"
            setattr(self, k, v)



class Server():
    # def __init__(self, args):
    def __init__(self):
        """
        The basic Federated Learning Server, includes basic operations
        Args:
            args: passed in through config file or other way
        returns:
            None
        """
        args = get_args()
        
        WORLD = MPI.COMM_WORLD
        rank = WORLD.Get_rank()
        size = WORLD.Get_size()   

        self.comm_world = WORLD
        self.rank = rank
        self.size = size

        if torch.cuda.is_available():
            device_num = torch.cuda.device_count()
            if device_num > 1:
                print(f"Warning, multiple GPUs detected, this could probably cause problems. \
                      They will all work on first GPU if not specified in advance as it may be unable to set GPU after importing torch. \n \
                      Please use 'os.environ' to set 'CUDA_VISIBLE_DEVICES' before using FLamingo. \n \
                      For example, 'os.environ['CUDA_VISIBLE_DEVICES'] = '0', then import FLamingo.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.MASTER_RANK = 0
        self.status = "TRAINING"

        # Copy some info from args
        self.args = args
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.args = args

        self.global_round = 0
        self.model_save_path = os.path.join(self.run_dir, "saved_models")  

        if os.path.exists(self.run_dir) == False:
            os.makedirs(self.run_dir, exist_ok=True)
        if os.path.exists(self.model_save_path) == False:
            os.makedirs(self.model_save_path, exist_ok=True) 

        self.data_to_send = None
        self.buffer = []

        self.init()

        # If user didn't init model, network, optimizer, loss_func, lr_scheduler, do it here
        if self.network is None:
            self.network = NetworkHandler()
        if self.model is None:
            self.model = create_model_instance(self.model_type, self.dataset_type)
        if self.optimizer is None:
            if self.momentum is not None:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        if self.loss_func is None:
            self.loss_func = torch.nn.CrossEntropyLoss()
        
        self.start_time = time.localtime()


    def log(self, info_str):
        """
        Print info string with time and rank
        """
        log(self.rank, self.global_round, info_str)


    def save_model(self, model, epoch):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        model_path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
        torch.save(model.state_dict(), model_path)


    def load_model(self, model, epoch):
        model_path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
        assert os.path.exists(model_path), f"model for Server {self.rank} does not exist"
        model.load_state_dict(torch.load(model_path))


    def init(self):
        """
        Init model and network to enable customize these parts.   
        For model, pass in torch model or model_type to create coresponding model.   
        For network, pass in your own network module or leave blank for default.
        Returns:
            None of these will be returned. The function will set:  
            self.model, self.model_type(if given when customized)
            self.optimizer: default SGD
            self.loss_func: default CrossEntropyLoss
            self.lr_scheduler: default ExponetialLR with gamma=0.993
        If you want to customize, please define it.
        """
        self.network = NetworkHandler()
        self.model = create_model_instance(self.model_type, self.dataset_type)
        assert self.model is not None, f"Model initialized failed. Either not passed in correctly or failed to instantiate."
        self.model.to(self.device)
        if self.momentum is not None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        self.loss_func = torch.nn.CrossEntropyLoss()



    def set_model_parameter(self, params, model=None):
        """
        Set model parameters. Default self.model
        """
        model = self.model if model is None else model
        torch.nn.utils.vector_to_parameters(params, model.parameters())


    def export_model_parameter(self, model=None):
        """
        Export self.model.parameters() to a vector
        """
        model = self.model if model is None else model
        return torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()


    def print_model_info(self, model=None):
        """
        Print model related info
        """
        model = self.model if model is None else model
        model_params = self.export_model_parameter(model=model)
        para_nums = model_params.nelement()
        model_size = para_nums * 4 / 1024 / 1024
        self.log(f"Model type:{self.model_type} \nModel size: {model_size} MB\n Parameters: {para_nums}")


    def init_clients(self, clientObj=ClientInfo):
        """
        Init clients list on server, clients list must be a class
        """
        self.all_clients = [clientObj(rank) for rank in range(0, self.num_clients+1)]


    def stop_all(self):
        """
        Stop all your clients.
        """
        self.broadcast({"status":'STOP'}, dest_ranks=range(1, self.num_clients+1))
        self.log("Stopped all clients")


    def select_clients(self, selected_from=None, selected_num=None):
        """
        Randomly select select_num clients from list selected_from   
        Args:       
            selected_from: default self.all_clients
            selected_num: default self.num_training_clients
        """
        all_idx = [i for i in range(1, self.num_clients+1)]
        selected_from = all_idx if selected_from is None else selected_from
        selected_num = self.num_training_clients if selected_num is None else selected_num
        self.selected_idxes = random.sample(selected_from, selected_num)
        self.selected_clients = []
        for client_idx in self.selected_idxes:
            self.all_clients[client_idx].global_round = self.global_round
            # self.all_clients[client_idx].strategy = strategy
            self.all_clients[client_idx].params = self.export_model_parameter(self.model)
            self.selected_clients.append(self.all_clients[client_idx])
        
        if self.verb:self.log(f"Selected clients: {self.selected_idxes}")


    def get_client_by_rank(self, rank, client_list=None):
        """
        Get client by its rank
        Args:
            rank: client rank to return
            client_list [ClientInfo]: where to find, default self.all_clients
        Return:
            required ClientInfo
        """
        assert rank in range(1, self.size+1), f"Invalid rank {rank}"
        if client_list is None:
            client_list = self.all_clients
        for client in client_list:
            if client.rank == rank:
                return client
        raise IndexError(f"Client rank {rank} not found")


    def train(self, model, dataloader, local_epoch, loss_func, optimizer):
        model.train()
        model.to(self.device)
        epoch_loss, epoch_num = 0.0, 0
        for ep in range(local_epoch):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                batch_num, loss = self._train_one_batch(model, data, target, optimizer, loss_func)
                epoch_loss += loss * batch_num
                epoch_num += batch_num
        epoch_loss /= epoch_num
        return {'loss':epoch_loss, 'num':epoch_num}


    def broadcast(self, data, dest_ranks=None, network=None):
        """
        Broadcast data to dest_ranks(list: int)
        Args:
            data: data to send, default self.data_to_send
            dest_ranks: destinations, default self.selected_idxes
        """
        if network is None:
            network = self.network
        if data is None:
            assert self.data_to_send is not None, "No data to send in both data and self.data_to_send"
            data = self.data_to_send
        if not isinstance(data, dict):
            self.log(f"Data to send is not dict, type {type(data)}")
        if dest_ranks is None:
            assert self.selected_idxes is not None, "No dest_ranks to send in both dest_ranks and self.selected_idxes"
            dest_ranks = self.selected_idxes
        data.update({'global_round':self.global_round})
        for dest in dest_ranks:
            network.send(data, dest)
        if self.verb: self.log(f'Server broadcast to {dest_ranks} succeed')


    def listen(self, src_ranks=None, clientObj=None, network=None):
        """
        Listening data from src_ranks(list: int) and update client
        Args:
            src_ranks: list[int] destinations, default self.selected_idx
            buffer: list where to store client infomation, default self.buffer
        """
        if network is None:
            network = self.network
        if clientObj is None:
            assert self.selected_clients is not None, "No ClientInfo object to store data"
            clientObj = self.all_clients
        if src_ranks is None:
            assert self.selected_idxes is not None, "No src_ranks to send in both src_ranks and self.selected_idxes"
            src_ranks = self.selected_idxes
        for src in src_ranks:
            received_data = network.get(src_rank=src)
            client = self.get_client_by_rank(src, clientObj)
            client.update(received_data)     
        if self.verb: self.log(f'Server listening to {src_ranks} succeed')


    def aggregate(self, client_list=None, weight_by_sample=False):
        """
        Aggregating client params, vanilla
        Args:
            client_list: list[ClientInfo], default self.selected_clients
            weight_by_sample: bool, whether weight by sample number from client
        """
        if client_list is None:
            assert self.selected_clients is not None, "No ClientInfo object to aggregate"
            client_list = self.selected_clients
        global_param_vec = self.export_model_parameter()
        total_sample, total_client_len = 0, len(client_list)
        param_delta_vec = torch.zeros_like(global_param_vec)
        for client in client_list:
            assert client.params is not None, "Client params is None"
            if weight_by_sample:
                assert client.train_samples > 0, f"Client {client.rank} train samples is 0"
                total_sample += client.train_samples
        for client in client_list:
            if weight_by_sample:
                client.weight = client.train_samples / total_sample
            else:
                client.weight = 1 / total_client_len
            param_delta = client.params - global_param_vec
            param_delta_vec += param_delta * client.weight
        global_param_vec += param_delta_vec
        self.set_model_parameter(global_param_vec)


    def load_client_model(self, client_rank):
        """
        Load parameters of a client to new model
        """
        client = self.get_client_by_rank(client_rank)
        model = deepcopy(self.model)
        self.set_model_parameter(client.params, model=model)
        return model


    def average_client_info(self, client_list):
        """
        Average client info and log it
        """
        length = len(client_list)
        clients = [self.get_client_by_rank(rank) for rank in client_list]
        avg_train_loss = 0.0
        avg_test_loss, avg_test_acc = 0.0, 0.0
        for client in clients:
            avg_train_loss += client.train_loss
            avg_test_acc += client.test_acc
            avg_test_loss += client.test_loss
        self.log(f"Avg global info:\n train loss {avg_train_loss/length}, \
                 \ntest acc {avg_test_acc/length}, \
                 \ntest loss {avg_test_loss/length}")
        


    def evaluate_on_clients(self, client_list, model=None):
        """
        Evalutae model on given clients' datasets
        Args:
            client:list[int] need to eval
            model:default self.model
        """
        model = self.model if model is None else model
        evaluation_loss, evaluation_acc = [], []
        for client in client_list:
            loss, acc, _ = self._evaluate_single_client(model, client)
            evaluation_acc.append(acc)
            evaluation_loss.append(loss)
        if self.verb: 
            self.log(f'Evaluation on clients: {client_list}\n \
                  Avg acc {np.mean(evaluation_acc)}\n \
                  Avg loss {np.mean(evaluation_loss)}')


    def _evaluate_single_client(self, model, dataset_rank, model_rank=None):
        """
        Evaluate model on client dataset
        """
        model = self.model if model_rank is None else self.load_client_model(model_rank)
        client_set = ClientDataset(self.dataset_type, self.data_dir, dataset_rank)
        testloader = client_set.get_test_loader(self.test_batch_size)
        test_loss, test_acc, test_num = self.test(model, testloader, self.loss_func,self.device)
        del client_set, testloader
        return test_loss, test_acc, test_num


    def test(self, model, dataloader, loss_func, device='cpu'):
        model.eval()
        test_loss, test_num, correct_num = 0.0, 0, 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            batch_num, batch_correct, loss = self._test_one_batch(model, data, target, loss_func)
            test_loss += loss * batch_num
            test_num += batch_num
            correct_num += batch_correct
        test_loss /= test_num
        test_acc = correct_num / test_num
        return test_loss, test_acc , test_num


    def _test_one_batch(self, model, data, target, loss_func):
        """
        Test one batch
        """
        model.eval()
        output = model(data)
        loss = loss_func(output, target)
        _, pred = torch.max(output, 1)
        # Check test accuracy
        correct = (pred == target).sum().item()
        num = data.size(0)
        return num, correct, loss.item()


    def finalize_round(self):
        """
        Call this to update global_round and other routines
        """
        self.global_round += 1


    def run(self):
        """
        FedAvg procedure:
        0. Initialize
        1. Select clients
        2. Send requests to clients
        3. Waiting for training results
        4. Aggregating results
        5. Evaluating and record
        """
        # self.init()
        self.print_model_info()
        self.init_clients(clientObj=ClientInfo)
        while True:
            # Selecting and set params
            self.select_clients()
            # Sending requests
            self.broadcast(data={'status':'TRAINING',
                                 'params':self.export_model_parameter()})
            # waiting 
            self.listen()
            # aggregate
            self.aggregate(client_list=self.selected_clients)
            # Evaluate
            # self.evaluate_on_clients(client_list=self.selected_idxes)
            self.average_client_info(client_list=self.selected_idxes)
            
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.args.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        
        if self.args.verb: self.log(f'Server finished at round {self.global_round}')
        self.stop_all()
       
