# Client Documentation

The `Client` class in FLamingo represents an individual participant in the federated learning process. It handles local model training, evaluation, and communication with the server.

## Architecture

The `Client` class extends `FLamingoBase` and implements the client-side operations of the federated learning process. Each client has its own local dataset, and performs local training on its data before sending model updates to the server.

## Initialization

```python
client = Client()
```

During initialization, the client:
1. Sets up random seeds for reproducibility
2. Creates logging directories
3. Initializes model, optimizer, loss function, and learning rate scheduler
4. Sets up system heterogeneity simulation if enabled

## Key Methods

### Model and System Setup

- `init()`: Initializes the model, optimizer, loss function, and network handler
- `rand_send()`: Generates a simulated communication time for system heterogeneity
- `rand_comp()`: Generates a simulated computation time for system heterogeneity

### Training and Evaluation

- `train(model, dataloader, local_epoch, loss_func, optimizer, scheduler=None)`: Trains the model on the local dataset
  - Performs a specified number of local epochs
  - Calculates training loss and time
  - Supports system heterogeneity simulation

- `evaluate()`: Placeholder method for model evaluation 
  - The actual evaluation functionality is implemented in the `test()` method from the `FLamingoBase` class

### Communication

- `listen(rank=0)`: Receives data from the server
  - Updates the global round counter
  - Returns the received data

- `send(data, rank=0)`: Sends data to the server
  - By default, sends to the master server (rank 0)

### Round Management

- `finalize_round()`: Increments the global round counter and logs the completion of a round

## The Federated Learning Process

The client's main process is implemented in the `run()` method, which follows this pattern:

1. **Listening**: Wait for instructions from the server
2. **Processing Instructions**:
   - If 'STOP', exit the process
   - If 'TRAINING', perform local training and evaluation
   - If 'TEST', perform evaluation only
3. **Sending Results**: Send training/testing results back to the server
4. **Round Finalization**: Update the round counter and prepare for the next round
5. **Termination**: Stop when reaching the maximum epochs or when instructed by the server

### Training Workflow

When the client receives a 'TRAINING' instruction:

1. Update local model parameters with those received from the server
2. Train the model on local data for a specified number of epochs
3. Evaluate the trained model on local test data
4. Send model parameters and metrics back to the server

## Example Usage

```python
client = Client()
client.run()
```

## Advanced Features

### System Heterogeneity Simulation

When `USE_SIM_SYSHET` is enabled, the client simulates heterogeneous system behavior by:

1. Selecting heterogeneity parameters based on the client's rank
2. Simulating computation and communication times using a stochastic model
3. Reporting these simulated times instead of actual wall clock times

This allows for reproducing realistic federated learning scenarios with varying client capabilities without needing a physically heterogeneous environment.

### TensorBoard Integration

When `USE_TENSORBOARD` is enabled, the client logs metrics to TensorBoard for visualization and analysis.