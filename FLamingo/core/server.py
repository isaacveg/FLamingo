import os
import sys
from mpi4py import MPI

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
from FLamingo.core.utils.chores import log, merge_several_dicts, create_logger
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
        if not hasattr(self, 'network'):
            self.network = NetworkHandler()
        if not hasattr(self, 'model'):
            self.model = create_model_instance(self.model_type, self.dataset_type)
            self.model = self.model.to(self.device)
        if not hasattr(self, 'optimizer'):
            if hasattr(self, 'momentum') and self.args.momentum is not None:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        if not hasattr(self, 'loss_func'):
            self.loss_func = torch.nn.CrossEntropyLoss()
        
        self.start_time = time.localtime()
        
        self.logger = create_logger(os.path.join(self.run_dir, 'server.log'))

    def log(self, info_str):
        """
        Print info string with time and rank
        """
        # Printed log won't used here anymore. 
        # If you want it, you need to DIY
        # log(self.rank, self.global_round, info_str)
        self.logger.info(info_str)

    def save_model(self, model, epoch):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        model_path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, epoch):
        model_path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
        assert os.path.exists(model_path), f"model for Server {self.rank} does not exist"
        model.load_state_dict(torch.load(model_path))
        
    def get_clients_attr_tolist(self, attr_name, clients_list=None):
        """
        Get a list of attribute values from a list of clients indexes.
        Args:
            attr_name: attribute name to get
            clients_list: list of clients to get, default self.all_clients_idxes
        Return:
            list of attribute values
        """
        if clients_list is None:
            clients = self.all_clients
        else:
            clients = [self.get_client_by_rank(k) for k in clients_list]
        return [getattr(client, attr_name) for client in clients]
    
    def set_clients_attr_fromlist(self, attr_name, attr_list, clients_list=None):
        """
        Set a list of attribute values to a list of clients indexes.
        Args:
            attr_name: attribute name to set
            attr_list: list of attribute values to set
            clients_list: list of clients to set, default self.all_clients_idxes
        """
        if clients_list is None:
            clients = self.all_clients
        else:
            clients = [self.get_client_by_rank(k) for k in clients_list]
        for client, attr in zip(clients, attr_list):
            setattr(client, attr_name, attr)

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
        self.model = self.model.to(self.device)
        if self.args.momentum is not None:
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
        return torch.nn.utils.parameters_to_vector(model.parameters()).clone().detach()

    def print_model_info(self, model=None):
        """
        Print model related info.
        By default, it will log: model type, model size(MB), number of parameters.
        Args:
            model: model to print, default self.model
        """
        model = self.model if model is None else model
        model_params = self.export_model_parameter(model=model)
        para_nums = model_params.nelement()
        model_size = para_nums * 4 / 1024 / 1024
        self.log(f"Model type:{self.model_type} \nModel size: {model_size} MB\n Parameters: {para_nums}")

    def init_clients(self, clientObj=ClientInfo):
        """
        Init clients list on server, clients list must be a class.
        This will set self.all_clients with a list of clientObj, defined by you.
        """
        # use num_clients+1 to ensure the rank 0 is server itself.
        self.all_clients = [clientObj(rank) for rank in range(1, self.num_clients+1)]
        self.all_clients_idxes = [i for i in range(1, self.num_clients+1)]

    def stop_all(self):
        """
        Stop all your clients.
        This will broadcast a 'STOP' signal to all clients.
        """
        self.broadcast({"status":'STOP'}, dest_ranks=range(1, self.num_clients+1))
        self.log("Stopped all clients")

    def select_clients(self, selected_from=None, selected_num=None):
        """
        Randomly select select_num clients from list selected_from.  
        This will set self.selected_clients and self.selected_clients_idxes.  
        Args:       
            selected_from: int list, default self.all_clients_idxes
            selected_num: int, default self.num_training_clients
        """
        selected_from = self.all_clients_idxes if selected_from is None else selected_from
        selected_num = self.num_training_clients if selected_num is None else selected_num
        self.selected_clients_idxes = random.sample(selected_from, selected_num)
        self.selected_clients_idxes = sorted(self.selected_clients_idxes)
        self.selected_clients = []
        for client_idx in self.selected_clients_idxes:
            self.get_client_by_rank(client_idx).global_round = self.global_round
            # self.all_clients[client_idx].strategy = strategy
            # self.get_client_by_rank(client_idx).params = self.export_model_parameter(self.model)
            self.selected_clients.append(self.get_client_by_rank(client_idx))
        if self.verb:self.log(f"Selected clients: {self.selected_clients_idxes}")

    def get_client_by_rank(self, rank, client_list=None):
        """
        Get client by its rank, return ClientInfo object.
        In most cases, if you set client_list to None, then you can simply use self.all_clients[rank].
        Args:
            rank: client rank to return
            client_list [ClientInfo]: where to find, default self.all_clients
        Return:
            required ClientInfo
        """
        assert rank in range(1, self.num_clients+1), f"Invalid rank {rank}"
        if client_list is None:
            client_list = self.all_clients
            return self.all_clients[rank-1]     # rank 1--num_clients, index 0--num_clients-1
        for client in client_list:
            if client.rank == rank:
                return client
        raise IndexError(f"Client rank {rank} not found")

    def train(self, model, dataloader, local_epoch, loss_func, optimizer, scheduler=None):
        """
        Train given dataset on given dataloader.
        Args:
            model: model to be trained
            dataloader: dataloader for the dataset
            local_epoch: number of local epochs
            loss_func: loss function
            optimizer: optimizer
            scheduler: default None, learning rate scheduler, lr will be consistent if not given
        Return:
            dict: train_loss and train_samples
        """
        model.train()
        epoch_loss, num_samples = 0.0, 0
        for ep in range(local_epoch):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()  
                output = model(data)
                loss = loss_func(output, target)
                loss.backward()  
                optimizer.step() 
                batch_num_samples = len(target)
                epoch_loss += loss.item() * batch_num_samples  
                num_samples += batch_num_samples
            if scheduler is not None:
                scheduler.step()  # 更新学习率
        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples}

    def test(self, model, dataloader, loss_func=None, device=None):
        """
        Test dataset on given dataloader.
        Args:
            model (nn.Module): Model to be tested.
            dataloader (DataLoader): DataLoader for the test dataset.
            loss_func (nn.Module, optional): Loss function to be used for testing. Defaults to None.
            device (torch.device, optional): Device to be used for testing. Defaults to None.
        Returns:
            dict: Dictionary containing test_loss, test_acc and test_samples.
        """
        loss_func = loss_func or self.loss_func
        device = device or self.device
        model.eval()
        test_loss = 0.0
        correct = 0
        num_samples = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_samples += len(target)
        test_loss /= num_samples
        accuracy = 100. * correct / num_samples
        # self.log(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_samples} ({accuracy:.0f}%)')
        return {'test_loss': test_loss, 'test_acc': accuracy,'test_samples':num_samples}

    def broadcast(self, data, dest_ranks=None, network=None):
        """
        Broadcast data to dest_ranks(list: int)
        Args:
            data: data to send, default self.data_to_send
            dest_ranks: destinations, default self.selected_clients_idxes
        """
        if network is None:
            network = self.network
        if data is None:
            assert self.data_to_send is not None, "No data to send in both data and self.data_to_send"
            data = self.data_to_send
        if not isinstance(data, dict):
            self.log(f"Data to send is not dict, type {type(data)}")
        if dest_ranks is None:
            assert self.selected_clients_idxes is not None, "No dest_ranks to send in both dest_ranks and self.selected_clients_idxes"
            dest_ranks = self.selected_clients_idxes
        data.update({'global_round':self.global_round})
        for dest in dest_ranks:
            network.send(data, dest)
        if self.verb: self.log(f'Server broadcast to {dest_ranks} succeed')

    def listen(self, src_ranks=None, network=None):
        """
        Listening data from src_ranks(list: int) and update client
        Args:
            src_ranks: list[int] destinations, default self.selected_idx
            buffer: list where to store client infomation, default self.buffer
        """
        if network is None:
            network = self.network
        if src_ranks is None:
            assert self.selected_clients_idxes is not None, "No src_ranks to send in both src_ranks and self.selected_clients_idxes"
            src_ranks = self.selected_clients_idxes
        for src in src_ranks:
            received_data = network.get(src_rank=src)
            client = self.get_client_by_rank(src)
            client.update(received_data)     
        if self.verb: self.log(f'Server listening to {src_ranks} succeed')

    def aggregate(self, client_list=None, weight_by_sample=False):
        """
        Aggregating client params, vanilla
        Args:
            client_list: list[int], default self.selected_clients_idxes
            weight_by_sample: bool, whether weight by sample number from client
        """
        if client_list is None:
            assert self.selected_clients_idxes is not None, "No ClientInfo object to aggregate"
            client_list = self.selected_clients_idxes
        global_param_vec = self.export_model_parameter()
        total_sample, total_client_len = 0, len(client_list)
        param_delta_vec = torch.zeros_like(global_param_vec)
        for client_idx in client_list:
            client = self.get_client_by_rank(client_idx)
            assert client.params is not None, "Client params is None"
            if weight_by_sample:
                assert client.train_samples > 0, f"Client {client.rank} train samples is 0"
                total_sample += client.train_samples
        for client_idx in client_list:
            client = self.get_client_by_rank(client_idx)
            if weight_by_sample:
                client.weight = client.train_samples / total_sample
            else:
                client.weight = 1 / total_client_len
            # global on cpu and params on gpu, need to temporarily move to cpu
            # print(f"ClientParams:{client.params.device}, globalVec:{global_param_vec.device}")
            param_delta = client.params.to(self.device) - global_param_vec
            param_delta_vec += param_delta * client.weight
        global_param_vec += param_delta_vec
        self.set_model_parameter(global_param_vec)
    
    def generate_global_test_set(self):
        """
        Generate a global test set. This will check data_dir/test/0.npz and generate a global test set.
        After this you can use self.test_loader to test.
        """
        self.test_set = ClientDataset(self.dataset_type, self.data_dir, 0)
        self.test_loader = self.test_set.get_test_loader(self.batch_size)

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
            
    def weighted_average(self, clients_list=None,attr='weight',delete=False):
        """
        Aggregate parameters using weights stored indexed by self.selected_clients_idxes.
        
        Args:
            clients_list: The list of clients to use for aggregation.
            attr (default 'weight'): The attribute to use for weighting the clients. 
            delete: If True, delete the updated_vec, model_delta, and original_vec
                after the aggregation.
        """
        clients_list = clients_list or self.selected_clients_idxes
        original_vec = self.export_model_parameter(self.model)
        model_delta = torch.zeros_like(original_vec)
        for rank in clients_list:
            client = self.get_client_by_rank(rank)
            client_vec = client.params.to(self.device)
            client_weight = getattr(client, attr)
            model_delta += client_weight * (client_vec - original_vec)
        updated_vec = original_vec + model_delta
        self.set_model_parameter(updated_vec)
        if delete:
            del updated_vec, model_delta, original_vec
        # return updated_vec

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
    
    def _train_one_batch(self, model, data, target, optimizer, loss_func):
        """
        Trains the model on a single batch of data. 
        Parameters:
        - model (nn.Module): The model to be trained.
        - data (torch.Tensor): The input data for the model.
        - target (torch.Tensor): The target values for the model.
        - optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        - loss_func (callable): The loss function used to compute the loss between the model output and the target.

        Returns:
        - batch_size (int): The size of the target tensor.
        - loss (float): The computed loss value.
        """
        model.train()
        # model = model.to(self.device)
        data, target = data.to(self.device), target.to(self.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        return len(target), loss.item()

    def _test_one_batch(self, model, data, target, loss_func):
        """
        Test one batch.
        Args:
            model: model to test
            data: data to test
            target: target to test
            loss_func: loss function
        Returns:
            num: number of samples
            correct: number of correct samples
            loss: loss of this batch
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
        Call this to update global_round and other routines.
        For now it just add 1 to global_round
        """
        self.global_round += 1
        self.log(f"============End of Round {self.global_round}============")

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
            # Sending models and parameters.
            self.broadcast(data={'status':'TRAINING',
                                 'params':self.export_model_parameter()}
                           )
            # Waiting for responses
            self.listen()
            # Aggregating model parameters
            self.aggregate()
            # Evaluate
            # self.test(self.model, self.test_loader)
            # Average
            self.average_client_info(client_list=self.selected_clients_idxes)
            
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.args.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        
        if self.args.verb: self.log(f'Server finished at round {self.global_round}')
        self.stop_all()
       
