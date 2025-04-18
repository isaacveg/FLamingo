# Hyper-parameters Recommendation

### Hyperparams
Bellow are some hyperparams you can refer to.


| dataset/model         | test_acc | lr/min_lr | lr_decay_rate | local_iteration | batch_size | epoch | momentum | weight_decay |
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

We encourage you to contribute your results to this table.

| Dataset | Method | Accuracy | Accuracy (std) | Accuracy (min) | Accuracy (max) |
| ------- | ------ | -------- | -------------- | -------------- | -------------- |
