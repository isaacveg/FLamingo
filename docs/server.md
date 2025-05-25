# Server Documentation

The `Server` class in FLamingo is responsible for coordinating the federated learning process across multiple clients. It manages client selection, aggregation of model parameters, evaluation, and communication.

## Architecture

The `Server` class extends `FLamingoBase` and implements the federated learning server-side operations. It maintains client information through the `ClientInfo` class and handles communication via a `NetworkHandler`.

## Initialization

```python
server = Server()
```

During initialization, the server:
1. Sets up random seeds for reproducibility
2. Creates logging directories
3. Initializes model, optimizer, loss function, and learning rate scheduler
4. Prepares for client management

## Key Components

### ClientInfo Class

The `ClientInfo` class maintains information about each client, including:
- Training and testing metrics
- Model parameters
- Timing information
- Client status

### Core Methods

#### Client Management

- `init_clients(clientObj=ClientInfo, ex_args=None)`: Initializes client objects
- `select_clients(selected_from=None, selected_num=None, select_all=False)`: Selects a subset of clients for the current round
- `get_client_by_rank(rank, client_list=None)`: Retrieves a client by its rank
- `get_clients_by_ranks(rank_list, client_list=None)`: Retrieves multiple clients by their ranks
- `get_clients_attr_tolist(attr_name, clients_list=None)`: Gets a specific attribute from a list of clients
- `set_clients_attr_fromlist(attr_name, attr_list, clients_list=None)`: Sets an attribute for multiple clients

#### Communication

- `broadcast(data, dest_ranks=None, network=None, blocking=True)`: Broadcasts data to specified clients. Use `blocking=False` for non-blocking parallel sends.
- `personalized_broadcast(common_data=None, personalized_attr=None, dest_rank=None, network=None, blocking=True)`: Sends personalized data to clients. Use `blocking=False` for non-blocking parallel sends.
- `listen(src_ranks=None, network=None)`: Receives data from specified clients
- `stop_all()`: Sends stop signal to all clients

#### Model Operations

- `aggregate(client_list=None, weight_by_sample=False)`: Aggregates model parameters from clients
- `weighted_average(clients_list=None, attr='weight', delete=False)`: Performs weighted aggregation of client parameters
- `load_client_model(client_rank)`: Loads a specific client's model

#### Evaluation

- `generate_global_test_set()`: Creates a global test dataset
- `evaluate_clients_models_on_server(client_list, model=None)`: Evaluates client models on the server
- `evaluate_on_new_clients(client_list=None, model=None)`: Evaluates the model on new clients
- `average_client_info(client_list, attrs=None, dic_prefix='avg')`: Averages and logs client metrics

#### Round Management

- `finalize_round()`: Updates round statistics and prepares for the next round
- `summarize()`: Summarizes the training process statistics
- `stop()`: Terminates the server

## The Federated Learning Process

The main process is implemented in the `run()` method, which follows this pattern:

1. **Initialization**: Set up clients and model
2. **Client Selection**: Select a subset of clients for the current round
3. **Model Distribution**: Send the model to selected clients
4. **Client Training**: Wait for clients to train and return results
5. **Aggregation**: Combine client models into a global model
6. **Evaluation**: Test the global model performance
7. **Round Finalization**: Update round metrics and prepare for the next round
8. **Termination**: Stop when reaching the maximum epochs

## Example Usage

```python
server = Server()
server.run()
```

## Advanced Features

### System Heterogeneity Simulation

When `USE_SIM_SYSHET` is enabled, the server simulates heterogeneous client behavior by adjusting timing metrics.

### Blocking vs Non-Blocking Communication

The server supports both blocking and non-blocking communication modes:

- **Blocking mode (default)**: Sends are executed sequentially, ensuring stability and simplicity. Each client receives data only after the previous send completes.
- **Non-blocking mode**: Sends are executed in parallel using MPI's non-blocking operations, reducing overall communication time when sending to multiple clients.

```python
# Blocking mode (default)
server.broadcast(data)

# Non-blocking mode for faster parallel sends
server.broadcast(data, blocking=False)
server.personalized_broadcast(common_data, personalized_attr, blocking=False)
```

### Personalized Model Distribution

Using `personalized_broadcast()`, the server can send different configurations to different clients.

### Weighted Aggregation

Various aggregation strategies can be implemented by customizing the `weighted_average()` method and setting client weights appropriately.