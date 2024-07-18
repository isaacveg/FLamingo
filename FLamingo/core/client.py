import os
import sys
        
import numpy as np
import random
import time
from copy import deepcopy

import asyncio
from mpi4py import MPI

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import torch

from FLamingo.core.utils.args_utils import get_args
from FLamingo.core.utils.data_utils import ClientDataset
from FLamingo.core.utils.model_utils import create_model_instance
from FLamingo.core.utils.train_test_utils import infinite_dataloader
from FLamingo.core.utils.chores import log, merge_several_dicts, create_logger, create_recorder
from FLamingo.core.network import NetworkHandler


class Client():
    # def __init__(self, args):
    def __init__(self):
        """
        The basic Federated Learning Client, includes basic operations
        Args:
            args: passed in through config file or other way
        returns:
            None
        """
        args = get_args()

        WORLD = MPI.COMM_WORLD
        rank = WORLD.Get_rank()
        size = WORLD.Get_size()
        # self.network = NetworkHandler(WORLD, rank, size)
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

        # Copy info from args
        self.args = args
        for key, value in vars(args).items():
            # if value is not None:
            setattr(self, key, value)
        self.global_round = 0

        self.model_save_path = os.path.join(self.run_dir, "saved_models")    
         
        if os.path.exists(self.run_dir) == False:
            os.makedirs(self.run_dir, exist_ok=True)
        if os.path.exists(self.model_save_path) == False:
            os.makedirs(self.model_save_path, exist_ok=True)
        client_logs_path = os.path.join(self.run_dir, 'client_logs')
        if os.path.exists(client_logs_path) == False:
            os.makedirs(client_logs_path, exist_ok=True)

        self.init()

        # If user didn't init model, network, optimizer, loss_func, lr_scheduler, do it here
        if not hasattr(self, 'network'):
            self.network = NetworkHandler()
        if not hasattr(self, 'dataset'):
            self.dataset = ClientDataset(self.dataset_type, self.data_dir, self.rank)
            self.train_loader = self.dataset.get_train_loader(self.batch_size)
            self.test_loader = self.dataset.get_test_loader(self.test_batch_size)
            # self.log(f"Client {self.rank} initializing dataset {self.train_loader}")
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

        self.logger = create_logger(os.path.join(client_logs_path, f'client_{self.rank}.log'))
        if self.USE_SIM_SYSHET:
            self.log("Using simulated system heterogeneity. Time is computed through random generation.")
            # Randomly select a setting from args.sys_het_list[ ]
            assert len(self.sys_het_list) > 0, "Sys_het_list is empty or not given"
            # sys_het = random.choice(self.sys_het_list)
            sys_het = self.sys_het_list[self.rank % len(self.sys_het_list)]
            self.log(f"Using system heterogeneity setting: {sys_het}")
            for key, value in sys_het.items():
                setattr(self, key, value)
            del sys_het
        if self.USE_TENSORBOARD:
            self.recorder = create_recorder(f'{self.run_dir}/event_log/{self.rank}/')
        else:
            self.USE_TENSORBOARD = False
        self.start_time = time.localtime()

    def log(self, info_str):
        """
        Print info string with time and rank
        """
        # log(self.rank, self.global_round, info_str)
        self.logger.info(info_str)
    
    def quick_rec_dict(self, dict):
        """Quickly write key-value in dict to self.recorder(SummaryWriter)
        with current self.global_round. All values are default scalar
        """
        if self.USE_TENSORBOARD:
            for key, value in dict.items():
                self.recorder.add_scalar(key, value, self.global_round)

    def init(self):
        """
        Init model and network to enable customize these parts.   
        For model, pass in torch model or model_type to create coresponding model.   
        For network, pass in your own network module or leave blank for default.
        For dataset, pass in your own dataset, train_loader, and test_loader.
        Settings:
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
        if self.momentum is not None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def save_model(self, model, epoch):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        model_path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, epoch):
        model_path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
        assert os.path.exists(model_path), f"model for client {self.rank} does not exist"
        model.load_state_dict(torch.load(model_path))

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
            dict: train_loss, train_samples, train_time
        """
        model.train()
        epoch_loss, num_samples = 0.0, 0
        s_t = time.time()
        num_batches = 0
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
                num_batches += 1
            if scheduler is not None:
                scheduler.step()  # 更新学习率
        
        if self.USE_SIM_SYSHET:
            train_time = num_batches * self.rand_time(self.computation, self.dynamics)
        else:
            train_time = time.time() - s_t
        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples,'train_time':train_time}
    
    def train_iters(self, model, dataloader, loss_func, optimizer, scheduler=None, iters=None):
        """
        Train given dataset on given dataloader with given iters.
        Args:
            model: model to be trained
            dataloader: dataloader for the dataset
            loss_func: loss function
            optimizer: optimizer
            scheduler: default None, learning rate scheduler, lr will be consistent if not given
            iters: number of iterations
        Return:
            dict: train_loss, train_samples, train_time
        """
        model.train()
        epoch_loss, num_samples = 0.0, 0
        s_t = time.time()
        num_batches = 0
        inf_loader = infinite_dataloader(dataloader)
        for i in range(iters):
            data, target = next(inf_loader)
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()  
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()  
            optimizer.step() 
            batch_num_samples = len(target)
            epoch_loss += loss.item() * batch_num_samples  
            num_samples += batch_num_samples
            num_batches += 1
        if scheduler is not None:
            scheduler.step()  # 更新学习率
        
        if self.USE_SIM_SYSHET:
            train_time = num_batches * self.rand_time(self.computation, self.dynamics)
        else:
            train_time = time.time() - s_t
        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples,'train_time':train_time}

    def test(self, model, dataloader, loss_func=None, device=None):
        """
        Test dataset on given dataloader.
        Args:
            model (nn.Module): Model to be tested.
            dataloader (DataLoader): DataLoader for the test dataset.
            loss_func (nn.Module, optional): Loss function to be used for testing. Defaults to None.
            device (torch.device, optional): Device to be used for testing. Defaults to None.
        Returns:
            dict: Dictionary containing test_loss, test_acc, and test_samples.
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
        return {'test_loss': test_loss, 'test_acc': accuracy,'test_samples':num_samples}

    def evaluate(self):
        pass

    def finalize_round(self):
        """
        Finalize training round and execute given functions.
        The basic version will only update global_round.
        """
        self.global_round += 1
        self.log(f"============End of Round {self.global_round}============")
    
    def _train_one_batch(self, model, data, target, optimizer, loss_func):
        """
        Trains the model on a single batch of data. 
        You should manually send everything to device before calling this function.
        
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
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        return len(target), loss.item()
        
    def rand_time(self, loc, scale):
            """
            Generate a random time value within a given range. 
            You can override this method to implement a custom time distribution.
            For now, it generates a random value from a normal distribution with a mean of loc and a standard deviation of sqrt scale.

            Parameters:
                loc (float): The mean value of the normal distribution.
                scale (float): The standard deviation of the normal distribution.

            Returns:
                float: A random time value within the range of 0.1 to 1.0.
            """
            randed = np.random.normal(loc=loc, scale=np.sqrt(scale))
            while randed > 10 or randed < 1:
                randed = np.random.normal(loc=loc, scale=np.sqrt(scale))
            return randed / 10

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

    def listen(self, rank=0):
        """
        Listen from other processes, default from server
        """
        data = self.network.get(rank)
        self.global_round = data['global_round']
        return data
    
    def send(self, data, rank=0):
        """
        Send data to other processes, default to server
        """
        self.network.send(data, rank)
        return

    def run(self):
        """
        FedAvg procedure:
        1. Get request from server
        2. Training
        3. Send information back
        """
        # self.model = create_model_instance(self.model)
        # self.init()
        while True:
            # data = self.receive_data(self.MASTER_RANK)
            # data = self.network.get(self.MASTER_RANK)
            data = self.listen()
            # print([k for k,v in data.items()])
            if data['status'] == 'STOP':
                if self.verb: self.log('Stopped by server')
                break

            elif data['status'] == 'TRAINING':
                # print(f'{self.rank}, {self.verb}')
                # if self.verb: self.log(f'training, train slow: {self.train_slow}, send slow: {self.send_slow}')
                self.set_model_parameter(data['params'])
                trained_info = self.train(
                    self.model, self.train_loader, self.args.local_epochs, self.loss_func, self.optimizer)
                self.log(f"Client {self.rank} trained {trained_info['train_samples']} samples in {trained_info['train_time']}s, loss: {trained_info['train_loss']}")
                tested_info = self.test(
                    self.model, self.test_loader, self.loss_func, self.device)
                self.log(f"Client {self.rank} tested {tested_info['test_samples']} samples, loss: {tested_info['test_loss']}, acc: {tested_info['test_acc']}")
                # Construct data to send
                data_to_send = merge_several_dicts([trained_info, tested_info])
                data_to_send['params'] = self.export_model_parameter()
                if self.USE_SIM_SYSHET:
                    # send time usually larger than computation time
                    data_to_send['send_time'] = self.rand_time(self.communication, self.dynamics) * 10
                # print(data_to_send)
                # self.network.send(data_to_send, self.MASTER_RANK)
                self.send(data_to_send)
                # if self.verb: self.log('training finished')

            elif data['status'] == 'TEST':
                if self.verb: self.log('testing')
                self.model = torch.load(data['model'])
                test_info = self.test()
                if self.verb: 
                    self.log(f'test info: {test_info.items()}')
                    # for k, v in test_info.items():
                    #     self.log(f'{k}: {v}')
                self.send(self.MASTER_RANK, test_info)
        
            else:
                raise Exception('Unknown status')
            
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        
        if self.verb: self.log(f'finished at round {self.global_round}')
       