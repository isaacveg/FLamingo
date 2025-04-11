import torch
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from models import AlexNet, CNN, DNN, EncDec, LR, ResNet, RNN, VGG, MobileNet


def create_model_instance(model_type, dataset_type):
    """
    Create model instance by passing model type and dataset type.
    The model and dataset must collaborate, else will return None.
    Use init_model to initialize your own model structure.
    Or submitting pull request to add your own.
    """
    assert model_type and dataset_type is not None, f"Model type and dataset type cannot be None."
    models_dict = {
        ('mnist', 'lr'): LR.MNIST_LR_Net,
        ('mnist', 'cnn'): CNN.MNIST_Net,
        ('mnist', 'alexnet'): AlexNet.AlexNetMnist,
        ('emnist', 'vgg'): VGG.VGG19_EMNIST,
        ('emnist', 'cnn'): CNN.EMNIST_CNN,
        ('svhn', 'vgg'): VGG.VGG19_EMNIST,
        ('svhn', 'cnn'): CNN.EMNIST_CNN,
        ('cifar10', 'alexnet'): AlexNet.AlexNet,
        ('cifar10', 'vgg9'): VGG.VGG9,
        ('cifar10', 'alexnet2'): AlexNet.AlexNet2,
        ('cifar10', 'vgg16'): VGG.VGG16_Cifar10,
        ('cifar100', 'mobilenet'): MobileNet.MobileNetV2_cifar100,
        ('cifar100', 'resnet'): ResNet.ResNet9,
        ('cifar100', 'vgg16'): VGG.VGG16_Cifar100,
        ('tinyimageNet', 'resnet'): ResNet.ResNet50,
        ('image100', 'alexnet'): AlexNet.AlexNet_IMAGE,
        ('image100', 'vgg16'): VGG.VGG16_IMAGE,
        ('femnist', 'cnn'): CNN.EMNIST_CNN,
        ('shakespeare', 'rnn'): RNN.RNN_Shakespeare,
        ('avazu', 'lr'): LR.LR_Net,
        ('avazu', 'dnn'): DNN.DNN_REC_Net
    }

    # def create_model(dataset_type, model_type):
    model = models_dict.get((dataset_type, model_type), None)
    return model() if model else None
