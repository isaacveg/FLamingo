# FLamingo Framework Overview

FLamingo is a flexible and extensible framework for implementing Federated Learning (FL) algorithms. It provides a streamlined approach to developing, testing, and deploying FL systems with customizable components for various use cases.

## Framework Architecture

FLamingo follows a server-client architecture where:

- **Server**: Coordinates the federated learning process, aggregates model updates, and manages client participation.
- **Client**: Performs local training on private data and communicates model updates to the server.
- **Base Class**: Provides common functionality shared between server and client components.
- **Network Handler**: Manages communication between server and clients.

The framework is designed with modularity in mind, allowing researchers and practitioners to focus on algorithm implementation rather than infrastructure.

## Core Components

1. **FLamingoBase**: Base class containing shared functionality for both Server and Client classes.
2. **Server**: Manages client selection, aggregation, and global model updates.
3. **Client**: Handles local training and communication with the server.
4. **NetworkHandler**: Facilitates communication between server and clients with support for both blocking and non-blocking MPI operations.
5. **Utilities**: Helper modules for data loading, model creation, and other common tasks.

## Key Features

- **Flexible Communication**: Support for both blocking (sequential) and non-blocking (parallel) communication modes
- **Modular Design**: Easy to customize and extend individual components
- **MPI-based Networking**: Efficient distributed communication using Message Passing Interface
- **Heterogeneity Simulation**: Built-in support for simulating client system heterogeneity

## Getting Started with FLamingo

This guide will walk you through implementing a simple FedAvg algorithm using the FLamingo framework.

### Prerequisites

First, ensure you have the FLamingo package installed:

```bash
pip install flamingo
# or if installing from the repository
git clone https://github.com/yourusername/FLamingo.git
cd FLamingo
pip install -e .
```

### Step 1: Setup Configuration

FLamingo uses YAML configuration files to manage hyperparameters and settings. Create a `config.yaml` file:

```yaml
# Basic configuration for FedAvg
seed: 42
num_clients: 10  # Total number of clients
num_training_clients: 5  # Number of clients per round
max_epochs: 20  # Total number of FL rounds

# Model and dataset
model_type: "CNN"  # Model architecture (options: CNN, MLP, ResNet, etc.)
dataset_type: "mnist"  # Dataset (options: mnist, cifar10, etc.)
data_dir: "./data"  # Path to dataset directory

# Training parameters
local_epochs: 5  # Number of local training epochs per round
batch_size: 32  # Local training batch size
test_batch_size: 128  # Batch size for testing
lr: 0.01  # Learning rate
momentum: 0.9  # SGD momentum parameter
lr_decay: 0.995  # Learning rate decay factor

# System settings
USE_SIM_SYSHET: False  # Enable/disable system heterogeneity simulation
USE_TENSORBOARD: True  # Enable/disable TensorBoard logging
run_dir: "./runs/fedavg"  # Directory for logs and saved models
verb: True  # Verbose logging
```

### Step 2: Create Main Script

Create a `main.py` file as the entry point for your federated learning simulation:

```python
import os
import sys
from mpi4py import MPI
import torch
import numpy as np
import random

# Import FLamingo components
from FLamingo.server import Server
from FLamingo.client import Client
from FLamingo.utils.args_utils import get_args

def main():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set random seed for reproducibility
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create server or client based on rank
    if rank == 0:
        # Rank 0 is the server
        server = Server()
        server.run()
    else:
        # Ranks 1+ are clients
        client = Client()
        client.run()

if __name__ == "__main__":
    main()
```

### Step 3: Generate and Partition Dataset

Before running the FL simulation, you need to generate and partition the dataset:

```bash
# For MNIST dataset
python -m datasets.generate_mnist --num_clients 10 --iid 0 --data_dir ./data
```

This will partition the MNIST dataset into 10 client shards with non-IID distribution (controlled by the `--iid` parameter).

### Step 4: Run the Federated Learning Simulation

Use MPI to launch the federated learning simulation:

```bash
mpirun -np 11 python main.py --config_file config.yaml
```

The `-np 11` parameter specifies 11 processes: 1 server (rank 0) and 10 clients (ranks 1-10).

## Understanding the FedAvg Workflow

Let's break down the federated averaging process implemented in FLamingo:

### 1. Server Initialization
```python
# In Server class
def __init__(self):
    # Initialize base components (model, optimizer, etc.)
    super().__init__()
    # Set up logging and other server-specific components
    self.init()
    # Initialize client information objects
    self.init_clients(clientObj=ClientInfo)
```

### 2. Client Selection
```python
# In Server class
def select_clients(self, selected_from=None, selected_num=None, select_all=False):
    # Randomly select clients for this round
    self.selected_clients_idxes = random.sample(selected_from, selected_num)
    self.selected_clients = []
    for client_idx in self.selected_clients_idxes:
        self.selected_clients.append(self.get_client_by_rank(client_idx))
```

### 3. Model Distribution
```python
# In Server class
def broadcast(self, data, dest_ranks=None, network=None):
    # Send the current model to selected clients
    data.update({'global_round': self.global_round})
    for dest in dest_ranks:
        network.send(data, dest)
```

### 4. Local Training on Clients
```python
# In Client class
def train(self, model, dataloader, local_epoch, loss_func, optimizer, scheduler=None):
    # Train the model locally for specified number of epochs
    model.train()
    for e in range(local_epoch):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Standard training loop
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
    # Return training metrics
    return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples, 'train_time': time.time() - s_t}
```

### 5. Model Updates to Server
```python
# In Client class
def send(self, data, rank=0):
    # Send trained model and metrics back to server
    self.network.send(data, rank)
```

### 6. Aggregation on Server
```python
# In Server class
def aggregate(self, client_list=None, weight_by_sample=False):
    # Aggregate client model updates
    global_param_vec = self.export_model_parameter()
    param_delta_vec = torch.zeros_like(global_param_vec)
    for client_idx in client_list:
        client = self.get_client_by_rank(client_idx)
        # Calculate weight for this client (uniform or by sample size)
        client.weight = client.train_samples / total_sample if weight_by_sample else 1 / total_client_len
        # Compute weighted model delta
        param_delta = client.params - global_param_vec
        param_delta_vec += param_delta * client.weight
    # Update global model
    global_param_vec += param_delta_vec
    self.set_model_parameter(global_param_vec)
```

### 7. Evaluation and Metrics
```python
# In Server class
def average_client_info(self, client_list, attrs=None, dic_prefix='avg'):
    # Calculate and log average metrics across clients
    dic = {}
    for attr in attrs:
        attr_list = self.get_clients_attr_tolist(attr, client_list)
        mean_attr = np.mean(attr_list)
        self.log(f"{dic_prefix} {attr}: {mean_attr}")
        dic[f'{dic_prefix}_{attr}'] = mean_attr
    return dic
```

## Customizing FLamingo

FLamingo is designed to be easily customizable. Here are some common extension points:

### Custom Client Selection

You can implement custom client selection strategies by overriding the `select_clients` method in your Server subclass:

```python
class MyServer(Server):
    def select_clients(self, selected_from=None, selected_num=None, select_all=False):
        # Implement your custom selection strategy
        # For example, select clients based on their performance
        clients_performance = [(i, self.get_client_by_rank(i).test_acc) for i in self.trainable_clients_idxes]
        clients_performance.sort(key=lambda x: x[1], reverse=True)
        self.selected_clients_idxes = [client[0] for client in clients_performance[:self.num_training_clients]]
        self.selected_clients = self.get_clients_by_ranks(self.selected_clients_idxes)
```

### Custom Aggregation Method

You can implement different aggregation methods by overriding the `aggregate` method:

```python
class MyServer(Server):
    def aggregate(self, client_list=None, weight_by_sample=False):
        # Implement your custom aggregation logic
        # For example, implement FedProx with proximal term
        global_param_vec = self.export_model_parameter()
        param_delta_vec = torch.zeros_like(global_param_vec)
        
        # Add your custom logic here
        # ...
        
        self.set_model_parameter(updated_params)
```

### System Heterogeneity Simulation

FLamingo includes built-in support for simulating system heterogeneity, which is useful for testing algorithms under realistic conditions:

```yaml
# In config.yaml
USE_SIM_SYSHET: True
sys_het_list:
  - {'computation': 1.0, 'communication': 1.0, 'cm_dynamics': 0.1, 'cp_dynamics': 0.1}
  - {'computation': 2.0, 'communication': 1.5, 'cm_dynamics': 0.2, 'cp_dynamics': 0.2}
  - {'computation': 3.0, 'communication': 2.0, 'cm_dynamics': 0.3, 'cp_dynamics': 0.3}
```

## Monitoring and Analysis

FLamingo provides built-in support for logging and visualization:

1. **Logs**: All logs are saved in the specified `run_dir`.
2. **TensorBoard**: When enabled, metrics are logged for visualization with TensorBoard.
3. **Model Checkpoints**: Models are saved periodically in the `saved_models` directory.

To view TensorBoard metrics:

```bash
tensorboard --logdir=./runs/fedavg/event_log/
```

## Extensive Documentation

For more detailed information, refer to the other documentation files:
- [Server Documentation](server.md)
- [Client Documentation](client.md)
- [Datasets](datasets.md)
- [Hyperparameters](hyperparameters.md)