import torch
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from models import AlexNet, CNN, DNN, EncDec, LR, ResNet, RNN, VGG


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
        ('EMNIST', 'vgg'): VGG.VGG19_EMNIST,
        ('EMNIST', 'cnn'): CNN.EMNIST_CNN,
        ('SVHN', 'vgg'): VGG.VGG19_EMNIST,
        ('SVHN', 'cnn'): CNN.EMNIST_CNN,
        ('CIFAR10', 'alexnet'): AlexNet.AlexNet,
        ('CIFAR10', 'vgg9'): VGG.VGG9,
        ('CIFAR10', 'alexnet2'): AlexNet.AlexNet2,
        ('CIFAR10', 'vgg16'): VGG.VGG16_Cifar10,
        ('CIFAR100', 'resnet'): ResNet.ResNet9,
        ('CIFAR100', 'vgg16'): VGG.VGG16_Cifar100,
        ('tinyImageNet', 'resnet'): ResNet.ResNet50,
        ('image100', 'alexnet'): AlexNet.AlexNet_IMAGE,
        ('image100', 'vgg16'): VGG.VGG16_IMAGE,
        ('femnist', 'cnn'): CNN.EMNIST_CNN,
        ('Avazu', 'lr'): LR.LR_Net,
        ('Avazu', 'dnn'): DNN.DNN_REC_Net
    }

    # def create_model(dataset_type, model_type):
    model = models_dict.get((dataset_type, model_type), None)
    return model() if model else None


def create_encoder(dataset_type):
    if dataset_type == 'cifar10':
        model = EncDec.AED_CIFAR()
    elif dataset_type == 'shakespear':
        model = EncDec.AED_SHAKESPEAR()
    elif 'mnist' in dataset_type:
        model = EncDec.AED_FEMNIST()
    else:
        raise ValueError('Not valid dataset')

    return model 
