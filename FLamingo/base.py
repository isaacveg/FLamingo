import os
import sys
import torch
import time
import numpy as np
import random
import logging
from .utils.train_test_utils import infinite_dataloader
from .utils.args_utils import get_args
from mpi4py import MPI

class FLamingoBase:
    """
    Base class for both Server and Client in FLamingo.
    Contains common functionality shared between the two classes.
    """
    
    # def __init__(self, rank, device=None, seed=None):
    def __init__(self):
        """
        Initialize base attributes.
        """
        self.args = get_args()
        
        WORLD = MPI.COMM_WORLD
        self.rank = WORLD.Get_rank()
        self.size = WORLD.Get_size()  
        self.comm_world = WORLD
        # self.rank = rank
        
        # Set device
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
        
        # Common attributes
        for key, value in vars(self.args).items():
            setattr(self, key, value)
        
        self.global_round = 0
        self.MASTER_RANK = 0
        self.model = None
        # self.model_type = None
        self.optimizer = None
        self.loss_func = None
        
        # Time tracking
        self.round_start_time = 0.0
        self.time_budget = []
    
    def log(self, info_str):
        """Log info string with time and rank"""
        self.logger.info(info_str)
    
    def quick_rec_dict(self, dict_data):
        """Quickly write key-value in dict to self.recorder(SummaryWriter)
        with current self.global_round. All values are default scalar
        """
        if self.USE_TENSORBOARD:
            for key, value in dict_data.items():
                self.recorder.add_scalar(key, value, self.global_round)
    
    def set_model_parameter(self, params, model=None):
        """
        Set model parameters.
        
        Args:
            params: Parameters to set
            model: Model to set parameters, default self.model
        """
        model = self.model if model is None else model
        torch.nn.utils.vector_to_parameters(params, model.parameters())
    
    def export_model_parameter(self, model=None):
        """
        Export model parameters as a vector.
        
        Args:
            model: Model to export parameters, default self.model
            
        Returns:
            torch.Tensor: Parameter vector
        """
        model = self.model if model is None else model
        return torch.nn.utils.parameters_to_vector(model.parameters()).clone().detach()
    
    def print_model_info(self, model=None, precision=4):
        """
        Print model related info.
        By default, it will log: model type, model size(MB), number of parameters.
        
        Args:
            model: model to print, default self.model
            precision: precision of model, default 4 bytes
        """
        model = self.model if model is None else model
        model_params = self.export_model_parameter(model=model)
        para_nums = model_params.nelement()
        model_size = para_nums * precision / 1024 / 1024
        self.log(f"Model type: {self.model_type} \nModel size: {model_size} MB\nParameters: {para_nums}\n{model}")
    
    def save_model(self, model=None, path=None):
        """
        Save model to path.
        
        Args:
            model: Model to save, default self.model
            path: Path to save, default self.model_save_path/model_{self.rank}.pth
        """
        model = self.model if model is None else model
        if path is None:
            if hasattr(self, 'model_save_path'):
                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)
                path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
            else:
                raise ValueError("No path provided and self.model_save_path not set")
                
        torch.save(model.state_dict(), path)
        
    def load_model(self, model=None, path=None):
        """
        Load model from path.
        
        Args:
            model: Model to load into, default self.model
            path: Path to load from, default self.model_save_path/model_{self.rank}.pth
        """
        model = self.model if model is None else model
        if path is None:
            if hasattr(self, 'model_save_path'):
                path = os.path.join(self.model_save_path, f'model_{self.rank}.pth')
            else:
                raise ValueError("No path provided and self.model_save_path not set")
                
        model.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, model, dataloader, local_epoch, loss_func, optimizer, scheduler=None, device=None):
        """
        Train a model for local_epoch epochs.
        
        Args:
            model: Model to train
            dataloader: Dataloader for training
            local_epoch: Number of epochs to train
            loss_func: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler, default None
            device: Device to train on, default self.device
            
        Returns:
            dict: Training statistics
        """
        device = self.device if device is None else device
        model.train()
        model = model.to(device)
        
        epoch_loss = 0.0
        num_samples = 0
        s_t = time.time()
        for e in range(local_epoch):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(target)
                num_samples += len(target)
                
            if scheduler is not None:
                scheduler.step()
                
        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples,
                'train_time': time.time() - s_t}
    
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
            train_time = num_batches * self.rand_comp()
        else:
            train_time = time.time() - s_t
        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples,'train_time':train_time}

    def test(self, model, dataloader, loss_func=None, device=None):
        """
        Test a model.
        
        Args:
            model: Model to test
            dataloader: Dataloader for testing
            loss_func: Loss function, default self.loss_func
            device: Device to test on, default self.device
            
        Returns:
            dict: Testing statistics
        """
        device = self.device if device is None else device
        loss_func = self.loss_func if loss_func is None else loss_func
        
        model.eval()
        model = model.to(device)
        
        test_loss = 0.0
        correct = 0
        num_samples = 0
        s_t = time.time()
        with torch.no_grad():
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_func(output, target).item() * len(target)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_samples += len(target)
                
        test_loss /= num_samples
        accuracy = 100. * correct / num_samples
        
        return {
            'test_loss': test_loss, 
            'test_acc': accuracy,
            'test_samples': num_samples,
            'test_time': time.time()-s_t
        }
        
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