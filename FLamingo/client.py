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

from .utils.args_utils import get_args
from .utils.data_utils import ClientDataset
from .utils.model_utils import create_model_instance
from .utils.train_test_utils import infinite_dataloader
from .utils.chores import log, merge_several_dicts, create_logger, create_recorder
from .network import NetworkHandler
from .base import FLamingoBase


class Client(FLamingoBase):
    # def __init__(self, args):
    def __init__(self):
        """
        The basic Federated Learning Client, includes basic operations
        """
        super().__init__()
        args = self.args
    
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.MASTER_RANK = 0

        self.status = "TRAINING"

        self.model_save_path = os.path.join(self.run_dir, "saved_models")    
         
        if os.path.exists(self.run_dir) == False:
            os.makedirs(self.run_dir, exist_ok=True)
        if os.path.exists(self.model_save_path) == False:
            os.makedirs(self.model_save_path, exist_ok=True)
        client_logs_path = os.path.join(self.run_dir, 'client_logs')
        if os.path.exists(client_logs_path) == False:
            os.makedirs(client_logs_path, exist_ok=True)
        self.logger = create_logger(os.path.join(client_logs_path, f'client_{self.rank}.log'))
        self.init()

        if self.USE_SIM_SYSHET:
            assert self.sys_het_list is not None, "sys_het_list is not given"
            self.log("Using simulated system heterogeneity.")
            # Randomly select a setting from args.sys_het_list[ ]
            assert len(self.sys_het_list) > 0, "Sys_het_list is empty or not given"
            # sys_het = random.choice(self.sys_het_list)
            sys_het = self.sys_het_list[self.rank % len(self.sys_het_list)]
            self.log(f"Using system heterogeneity setting: {sys_het}")
            for key, value in sys_het.items():
                setattr(self, key, value)
            del sys_het
        else:
            self.log("Not using simulated system heterogeneity. Time are real.")

        if self.USE_TENSORBOARD:
            self.recorder = create_recorder(f'{self.run_dir}/event_log/{self.rank}/')
        else:
            self.USE_TENSORBOARD = False
        self.start_time = time.localtime()

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
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        self.loss_func = torch.nn.CrossEntropyLoss()

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
        train_dict = super().train(model, dataloader, local_epoch, loss_func, optimizer, scheduler)
        
        if self.USE_SIM_SYSHET:
            num_batches = len(dataloader) * local_epoch
            train_dict['train_time'] = num_batches * self.rand_comp()
        return train_dict

    def evaluate(self):
        pass

    def finalize_round(self):
        """
        Finalize training round and execute given functions.
        The basic version will only update global_round.
        """
        self.global_round += 1
        self.log(f"============End of Round {self.global_round}============")

    def rand_send(self):
        return self.rand_time(self.communication, self.cm_dynamics) * 10
    
    def rand_comp(self):
        return self.rand_time(self.computation, self.cp_dynamics)

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
        while True:
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
                    data_to_send['send_time'] = self.rand_send()
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
       