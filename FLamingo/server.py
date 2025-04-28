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

from .base import FLamingoBase
from .utils.args_utils import get_args
from .utils.data_utils import ClientDataset
from .utils.model_utils import create_model_instance
from .utils.chores import log, merge_several_dicts, create_logger, create_recorder
from .network import NetworkHandler


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
        self.train_samples = 0
        # test status
        self.test_time = 0.0
        self.test_loss = 0.0
        self.test_acc = 0.0
        self.test_samples = 0
        # params and weight
        self.params = None
        self.weight = 0.0
        # time cost
        self.send_time = 0.0
        self.train_time = 0.0
        self.round_time = 0.0

    def update(self, data):
        """
        Update items in client info
        Args:
            data: dict containing items. 
        """
        for k,v in data.items():
            assert hasattr(self, k), f"{k} not in client info"
            setattr(self, k, v)
    
    def round_time_calc(self):
        """
        Calculate round time cost. Currently it's train_time + 2*send_time since receive time is not accurate.
        """
        if self.train_time == 0.0 or self.send_time == 0.0:
            print(f"Warning, train_time or send_time is 0.0, train_time:{self.train_time}, send_time:{self.send_time}")
        self.round_time = self.train_time + 2*self.send_time


class Server(FLamingoBase):
    # def __init__(self, args):
    def __init__(self):
        """
        The basic Federated Learning Server, includes basic operations
        """
        super().__init__()
        args = self.args
    
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # self.MASTER_RANK = 0
        self.status = "TRAINING"
        
        self.logger = create_logger(os.path.join(self.run_dir, 'server.log'))
        
        # self.global_round = 0
        self.model_save_path = os.path.join(self.run_dir, "saved_models")  

        if os.path.exists(self.run_dir) == False:
            os.makedirs(self.run_dir, exist_ok=True)
        if os.path.exists(self.model_save_path) == False:
            os.makedirs(self.model_save_path, exist_ok=True) 

        self.data_to_send = None
        self.buffer = []

        self.init()
        
        # Server specific
        self.round_start_time = time.time()
        # self.round_time_cost = 0.0
        self.total_time_cost = 0.0
        self.time_budget = []

        # if hasattr(self, 'USE_TENSORBOARD'):
        if self.USE_TENSORBOARD:
            self.recorder = create_recorder(f'{self.run_dir}/event_log/{self.rank}/')
        else:
            self.USE_TENSORBOARD = False
            
    def quick_rec_dict(self, dict):
        """Quickly write key-value in dict to self.recorder(SummaryWriter)
        with current self.global_round. All values are default scalar
        """
        if self.USE_TENSORBOARD:
            for key, value in dict.items():
                self.recorder.add_scalar(key, value, self.global_round)

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

    def init_clients(self, clientObj=ClientInfo, ex_args=None):
        """
        Init clients list on server, clients list must be a class.
        This will set self.all_clients with a list of clientObj, defined by you.
        
        Args:
            clientObj: a ClientInfo object, default ClientInfo(rank)
            ex_args: a list that stores all arguments need for clientObj.
        """
        # use num_clients+1 to ensure the rank 0 is server itself.
        if not hasattr(self, 'num_eval_clients'):
            self.eval_clients_idxes = []
            self.num_trainable_clients = self.num_clients
            self.trainable_clients_idxes = [num for num in range(1, self.num_clients+1)]
        else:
            self.num_trainable_clients = self.num_clients-self.num_eval_clients
            self.eval_clients_idxes = [num for num in range(self.num_trainable_clients, self.num_clients+1)]
            self.trainable_clients_idxes = [num for num in range(1, self.num_trainable_clients+1)]
        if ex_args is None:
            self.all_clients = [clientObj(rank) for rank in range(1, self.num_clients+1)]
        else:
            self.all_clients = [clientObj(rank, *ex_args) for rank in range(1, self.num_clients+1)]
        self.all_clients_idxes = [i for i in range(1, self.num_clients+1)]

    def stop_all(self):
        """
        Stop all your clients.
        This will broadcast a 'STOP' signal to all clients.
        """
        self.broadcast({"status":'STOP'}, dest_ranks=range(1, self.num_clients+1))
        self.log("Stopped all clients")

    def select_clients(self, selected_from=None, selected_num=None,
                       select_all=False):
        """
        Randomly select select_num clients from list selected_from.  
        This will set self.selected_clients and self.selected_clients_idxes.  
        Args:       
            selected_from: int list, default self.all_clients_idxes
            selected_num: int, default self.num_training_clients
            select_all: bool, default False, whether to select all clients
        """
        if select_all:
            self.selected_clients_idxes = self.all_clients_idxes
            self.selected_clients = self.all_clients
            if self.verb: self.log(f"Selected all clients")
        else:
            selected_from = self.trainable_clients_idxes if selected_from is None else selected_from
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
        In most cases, if you set client_list to None, then you can simply use self.all_clients[rank-1].
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
    
    def get_clients_by_ranks(self, rank_list, client_list=None):
        """
        Get clients by their ranks, return a list of ClientInfo objects.
        Args:
            rank_list: list of client ranks
            client_list: list of clients to search, default self.all_clients
        Return:
            list of required ClientInfo objects
        """
        return [self.get_client_by_rank(rank, client_list) for rank in rank_list]

    def broadcast(self, data, dest_ranks=None, network=None):
        """
        Broadcast data to dest_ranks(list: int).
        
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
        data.update({
            'global_round': self.global_round
            })
        for dest in dest_ranks:
            network.send(data, dest)
        if self.verb: self.log(f'Server broadcast to {dest_ranks} succeed')
        
    def personalized_broadcast(self, common_data=None, personalized_attr=None, dest_rank=None, network=None):
        """
        Broadcast different attributes to clients.
        This function is useful when different clients have different attributes to send, 
        such as dynamically adjusted local updates. You must store all commonly used data 
        into common_data, the attributes will then be added from clients to the data, and 
        finally sent to the specified destination ranks.

        Args:
            common_data (dict, optional): Data to be sent to all clients. Default is None.
            personalized_attr (list of str, optional): List of attribute names to be sent 
                individually to each client. Default is None.
            dest_rank (list of int, optional): List of destination ranks for the clients. 
                Default is None, which uses self.selected_clients_idxes.
            network (object, optional): Network object used to send the data. Default is 
                None, which uses self.network.
        """
        if network is None:
            network = self.network
        if dest_rank is None:
            dest_rank = self.selected_clients_idxes
        if personalized_attr is None:
            print("There are no personalized attributes to send, you should use self.broadcast.")
            self.broadcast(data=common_data, dest_ranks=dest_rank)
        else:
            for rank in dest_rank:
                send_dic = {'global_round': self.global_round}
                if common_data is not None:
                    send_dic.update(common_data)
                client = self.get_client_by_rank(rank)
                for attr in personalized_attr:
                    if hasattr(client, attr):
                        send_dic.update({attr: getattr(client, attr)})
                    else:
                        print(f"Client {rank} does not have attribute {attr}. Skipping...")
                network.send(send_dic, dest_rank=rank)
            if self.verb: self.log(f'Server personalized broadcast to {dest_rank} succeed')

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
            if not self.USE_SIM_SYSHET:
                # use real receive time
                client.update({'send_time': network.last_recv_time})     
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
        self.test_loader = self.test_set.get_test_loader(self.test_batch_size)

    def load_client_model(self, client_rank):
        """
        Load parameters of a client to new model
        """
        client = self.get_client_by_rank(client_rank)
        model = deepcopy(self.model)
        self.set_model_parameter(client.params, model=model)
        return model

    def average_client_info(self, client_list, attrs=None, dic_prefix='avg'):
        """
        Average client info and log them. It will also return a dict containing
        these averaged params, with avg + original_name.
        
        Args:
            client_list: list[int] clients to average, default self.selected_clients_idxes
            attrs: list[str] attributes to average, default ['train_loss', 'test_loss', 'test_acc']
            dic_prefix: str, prefix for the returned dict, default 'avg'
        """
        client_list = client_list or self.selected_clients_idxes
        dic = {}
        if attrs is None:
            attrs = ['train_loss', 'test_loss', 'test_acc']
        for attr in attrs:
            attr_list = self.get_clients_attr_tolist(attr, client_list)
            mean_attr = np.mean(attr_list)
            self.log(f"{dic_prefix} {attr}: {mean_attr}")
            dic[f'{dic_prefix}_{attr}'] = mean_attr
        return dic
        
    def evaluate_clients_models_on_server(self, client_list, model=None):
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
            
    def evaluate_on_new_clients(self, client_list=None, model=None):
        """
        Evaluate on new clients, this will send model to eval_clients and evaluate.
        Args:
            client_list: list[int] new clients to evaluate
            model: model to evaluate
        """
        model = self.model if model is None else model
        client_list = client_list or self.eval_clients_idxes
        self.broadcast({'params': self.export_model_parameter(model),
                        'status':'EVAL'}, 
                       dest_ranks=client_list)
        self.listen(src_ranks=client_list)
            
    def weighted_average(self, clients_list=None,attr='weight',delete=False):
        """
        Aggregate parameters using weights stored indexed by self.selected_clients_idxes.
        Specifically, you can set client.weight and then use this function to aggregate.
        
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
 
    def finalize_round(self):
        """
        Call this to update global_round and other routines.
        For now it will:
        - add 1 to global_round
        - calculate round time usage for each client
        - log current round time usage and reset timer
        - append round_time to time_budget
        """
        self.global_round += 1
        for client in self.selected_clients:
            client.round_time_calc()    # update round time
        if not self.USE_SIM_SYSHET:
            time_used = time.time()-self.round_start_time   
            self.time_budget.append(time_used)
            self.log(f"Round time cost: {time_used:.4f}")
        else:
            time_list = self.get_clients_attr_tolist('round_time', self.selected_clients_idxes)
            max_idx = np.argmax(time_list)
            time_used = time_list[max_idx]
            slowest_client_idx = self.selected_clients_idxes[max_idx]
            self.time_budget.append(time_used)
            self.log(f"Simulated round time cost: {time_used:.4f} slowest client {slowest_client_idx}")
        self.log(f"{'='*10}End of Round {self.global_round}{'='*10}")
        self.round_start_time = time.time()
        # self.round_time_cost = 0.0
        
    def summarize(self):
        """
        Summarize the training process.
        Log the total time cost and average time cost.
        """
        self.total_time_cost = sum(self.time_budget)
        self.log(f"Total time cost: {sum(self.time_budget):.4f}")
        self.log(f"Average time cost: {np.mean(self.time_budget):.4f}")
            
    def stop(self):
        """
        Stop the server. This will close the network and log the message.
        However, you don't need to call this function in most cases.
        """
        self.network.close()
        self.log("Server stopped")

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
        self.init_clients(clientObj=ClientInfo)
        while True:
            # Althogh the self.round_start_time is set in init, it's better to set it again here.
            self.round_start_time = time.time()
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
        self.summarize()
        self.stop()
       
