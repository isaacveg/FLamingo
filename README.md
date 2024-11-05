# FLamingo 
A truly expansive and flexible MPI library for Federated Learning

*Some codes directly from PFL-Non-IID and FedLab, truly inspiring*

### How to run?
Suppose there are 30 clients in your cluster on cifar10 dataset. 
0. Clone FLamingo
```bash
git clone https://github.com/isaacveg/FLamingo.git
```
If you want to install it, do the followings, or just add the FLamingo to your system path
```python
# In your python file
import sys
sys.path.append(YOUR_PATH_TO_FLAMINGO)
import FLamingo
```
Or install it (Optional)
```bash
cd FLamingo
pip install .
```

1. Generate dataset for each client
Use FLamingo_datasets to generate dataset splits [FLamingo_datasets](github.com/isaacveg/FLamingo_datasets)
```bash
git clone https://github.com/isaacveg/FLamingo_datasets.git
cd FLamingo_datasets
python gen_cifar10.py --nc 30 --dist iid --seed 2024 --indir ../datasets/ --outdir ../datasets/
```
Generate cifar10 with iid data, change pat to dir to use Dirichlet distribution. More specific usages please check [FLamingo_datasets](github.com/isaacveg/FLamingo_datasets)

> This part is inheritated from Tsingz0/PFL-non-IID
> Leaf dataset is directly from Leaf and FedLab

Then it'll be stored in `datasets/cifar10`, with train \test to use. Also storing meta data in json file. 0.npz is the shard containing all the data which is easy to use.


> If you need Leaf dataset, first download and generate according to official guidance and use `datasets/leaf_data/gen_pickle_dataset.sh`. Then you can use gen_femnist.py or gen_shakespeare.py

2. Write configs in `config.yaml` and run. The main.py should contains process management.
```shell
python main.py
```
- or Use mpiexec to run on your own. 
```shell
mpiexec --oversubscribe -n 1 python core/server/base_server.py : -n 30 python core/client/base_client.py > run.log > 2 >&1
```

3. Check examples
Use [FLamingo_examples](github.com/isaacveg/FLamingo_examples) and there are some of examples available.

### How to add my own client, server, model or datasets?
You can use `python -c "from FLamingo.templates import generate;generate('./YOUR_METHOD_NAME')" ` to generate a template for your own method. Then you can modify it and run it.

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
