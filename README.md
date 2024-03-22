# FLamingo 
A truly expansive and flexible MPI library for Federated Learning

*Some codes directly from PFL-Non-IID and FedLab, truly inspiring*

### How to run?
Suppose there are 30 clients in your cluster on cifar10 dataset. 
1. Generate dataset for each client
Modify client number in `generate_cifar10.py`, and generate cifar10 with noniid data, balanced samples' number and ordinary distribution with 10 classes each client (can be modeified in datasets/utils/dataset_utils.py. Change pat to dir to use Dirichlet distribution.

> This part is directly inheritated from Tsingz0/PFL-non-IID
> Leaf dataset is directly from Leaf and FedLab
```shell
cd FLMPI  
python datasets/generate_cifar10.py noniid balance pat
```
Then it'll be stored in `datasets/cifar10`, with train \test to use. Also storing meta data in json file.

If you need Leaf dataset, first download and generate according to official guidance and use `datasets/leaf_data/gen_pickle_dataset.sh`

2. Write configs in `config.yaml` and run.
```shell
python main.py
```
- or Use mpiexec to run your own script. 
```shell
mpiexec --oversubscribe -n 1 python core/server/base_server.py : -n 30 python core/client/base_client.py > run.log > 2 >&1
```

### How to add my own client, server, model or datasets?
0. Fork the repo and clone it to your machine.
1. Add your client and server in `core/client/` or `core/server/`, there're templates to get started. Better name it as `yourmethod_client.py` and `yourmethod_server.py`.
2. Add your models and datasets in `datasets/` or  `models/`. 
3. Implement your own dataset generater in `datasets`. Remember before each run datasets should be generated ahead. Implement dataset loader in `core/utils/data_utils.py` if you have new type of dataset other than CV.
4. `python main.py` if you followed standard naming paradigm or anything else you need to run.


### Get started
You can use `python -c "from FLamingo.templates import generate;generate('./FedProx')" ` to generate a template for your own method. Then you can modify it and run it.

The archecture is shown below.

- server.py implements `Server`, must have `run()`;    
- server.py implments `ClientInfo`, should be passed to Server to record client infomation on server;   
- client.py implements `Client`, must have `run()`, and other things you need;

Server and `Client` use `ClientDataset` to get dataset by its rank. So clients rank are normally started from 1 and server has MASTER_RANK 0. After round you can use `finalize_round()` on client to do some rountine job or `stop_all` on server to stop all.

Server use `broadcast` and `listen` to send and collect

Feel freeeee to use code already written.

To communicate, you can use `network.send()` or `network.get()` and pass in rank of process you want to communicate. Or just implement your own communication method.


### Hyperparams
Bellow are some hyperparams you can refer to.


| dataset/model         | test_acc | lr/min_lr | decay_rate | local_iteration | batch_size | epoch | momentum | weight_decay |
| --------------------- | -------- | --------- | ---------- | --------------- | ---------- | ----- | -------- | ------------ |
| CIFAR10/AlextNet      | 0.8      | 0.1/0.001 | 0.993      | 50              | 32         | 500   | -1       | 0.00         |
| CIFAR10/VGG9          | 0.85     | 0.1/0.001 | 0.993      | 50              | 32         | 500   | -1       | 0.00         |
| EMNIST/CNN            | 0.85     | 0.1/0.001 | 0.98       | 30              | 32         | 200   | -1       | 0.00         |
| EMINST/?              |          |           |            |                 |            |       |          |              |
| CIFAR100/ResNet9      | 0.55     | 0.1/0.001 | 0.99       | 30              | 32         | 500   | 0.9      | 0.001        |
| CIFAR100/VGG16        |          |           |            |                 |            |       |          |              |
| tinyImageNet/ResNet50 |          |           |            |                 |            |       |          |              |
| tinyImageNet/?        |          |           |            |                 |            |       |          |              |
| IMAGE100/AlexNet      | 0.54     | 0.1/0.001 | 0.99       | 50              | 64         | 120   | -1       | 0.00         |
| IMAGE100/VGG16        | 0.65     | 0.1/0.001 | 0.993      | 50              | 64         | 250   | -1       | 0.00         |
| SVHN/?                |          |           |            |                 |            |       |          |              |


### Results
| Dataset | Method | Accuracy | Accuracy (std) | Accuracy (min) | Accuracy (max) |
| ------- | ------ | -------- | -------------- | -------------- | -------------- |



### Contribute
Issue anything or pull request.
