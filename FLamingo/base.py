import os
import sys
import torch
import time
import numpy as np
import random
import logging
from .utils.chores import create_logger, create_recorder

class FLamingoBase:
    """
    Base class for both Server and Client in FLamingo.
    Contains common functionality shared between the two classes.
    """
    
    def __init__(self, rank, device=None, seed=None):
        """
        Initialize base attributes.
        
        Args:
            rank (int): The rank of the process
            device (torch.device, optional): Device to use. If None, will auto-select
            seed (int, optional): Random seed for reproducibility
        """
        self.rank = rank
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Set random seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Common attributes
        self.global_round = 0
        self.MASTER_RANK = 0
        self.model = None
        self.model_type = None
        self.optimizer = None
        self.loss_func = None
        
        # Time tracking
        self.round_start_time = time.time()
        self.time_budget = []
        
        # Flags
        self.USE_TENSORBOARD = False
        self.verb = False
    
    def set_logger(self, log_path):
        """Set up logger for the class"""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logger = create_logger(log_path)
    
    def set_recorder(self, recorder_path):
        """Set up TensorBoard recorder"""
        self.recorder = create_recorder(recorder_path)
        self.USE_TENSORBOARD = True
    
    def log(self, info_str):
        """Log info string with time and rank"""
        if hasattr(self, 'logger'):
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
        
        epoch_loss = 0
        num_samples = 0
        
        for e in range(local_epoch):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
                epoch_loss += loss.item() * len(target)
                num_samples += len(target)
                
            if scheduler is not None:
                scheduler.step()
                
        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples, 'train_time': time.time()}
    
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
        
        test_loss = 0
        correct = 0
        num_samples = 0
        
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
            'test_time': time.time()
        }