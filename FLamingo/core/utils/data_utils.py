import torch
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
import pickle
import os, sys
import numpy as np
from PIL import Image
from torchvision import transforms

# append FLMPI
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir_path)
sys.path.append(os.path.join(dir_path, 'datasets'))
# sys.path.append(os.path.join(dir_path, 'datasets','leaf','nlp_utils'))
import leaf_data
from leaf_data.pickle_dataset import PickleDataset


class ClientDataset():
    # Initialize the ClientDataset class with the dataset type, data directory, and rank
    def __init__(self, dataset_type, data_dir, rank):
        # Set the type of the dataset
        self.type = dataset_type
        # Create a dataset instance for the training set
        self.trainset = create_dataset_instance(rank, self.type, data_dir, 'train')
        # Create a dataset instance for the test set
        self.testset = create_dataset_instance(rank, self.type, data_dir, 'test')
        # Set the rank
        self.rank = rank

    # Return a DataLoader for the training set
    def get_train_loader(self, batch_size):
        # Return a DataLoader for the training set
        return DataLoader(dataset=self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Return a DataLoader for the test set
    def get_test_loader(self, batch_size):
        # Return a DataLoader for the test set
        return DataLoader(dataset=self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

    
def create_dataset_instance(rank, dataset_type, data_dir, train_test):
    '''
    Create a dataset instance from a given rank, dataset type, data directory, and train/test split.
    Args:
        rank (int): the rank of the client
        dataset_type (str): the type of dataset
        data_dir (str): the directory of the dataset
        train_test (str): the train/test split
    
    Returns:
        (torch.utils.data.Dataset): the dataset instance
    '''
    if dataset_type in ['femnist', 'shakespeare', 'sent140', 'celeba', 'synthetic', 'reddit']:
        # LEAF dataset, use preprocess from FedLab-benchmarks
        dataset = PickleDataset(dataset_name=dataset_type, pickle_root=data_dir)
        return dataset.get_dataset_pickle(train_test, rank)

    else:
        if rank >= 0:
            pickle_dir = os.path.join(data_dir, dataset_type, train_test)
            pickle_file = os.path.join(pickle_dir, f'{rank-1}.npz')
        assert os.path.exists(data_dir), f'{data_dir} dataset {dataset_type} not found, plz generate it'
        # Check file numbers under pickle_dir to match the number of clients
        file_num = len(os.listdir(pickle_dir))
        assert os.path.exists(pickle_file), f'{pickle_file} Client {rank} dataset {dataset_type} not found, plz generate it.\n \
            Hint: file number under {pickle_dir} is {file_num}, check your client num matches or not.'
        with open(pickle_file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        X_train = torch.Tensor(data['x']).type(torch.float32)
        y_train = torch.Tensor(data['y']).type(torch.int64)

        # train_data = [(x, y) for x, y in zip(X_train, y_train)]
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        return dataset


def concatenate_datasets(dataset_type, data_dir, rank, train_test):
    '''
    Concatenate all the datasets from a given rank, dataset type, and data directory.
    Args:
        dataset_type (str): the type of dataset
        data_dir (str): the directory of the dataset
        rank list(int): the ranks of the clients
        train_test (str): whether train/test 
    
    Returns:
        (torch.utils.data.Dataset): the concatenated dataset
    '''
    # Create a list of datasets
    datasets = []
    # Loop through all the clients
    for i in rank:
        # Create a dataset instance for the client
        dataset = create_dataset_instance(i, dataset_type, data_dir, train_test)
        # Append the dataset to the list of datasets
        datasets.append(dataset)
    # Concatenate the list of datasets
    return ConcatDataset(datasets)



def save_data_to_picture(dataset_type, data, save_dir):
    """
    Transform picture of given data to save_dir
    Args:
        dataset_type (str): the type of dataset
        data (tuple): a tuple of two torch tensors, the first containing the images and the second containing the labels
        save_dir (str): the directory where to save the images
    Output:
        generate save_dir/rank/*.png
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # emnist and femnist
    name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z',
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z'
             ]
    cifar_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    images, labels = data
    if 'mnist' in dataset_type:
        shape = [28, 28]
        count_ = {}
        for i, (image, label) in enumerate(zip(images, labels)):
            image = np.array(image).reshape(shape)
            if int(label) not in count_:
                count_[int(label)] = 0
            # Image doesn't support 0-1 image, convert to 0-255
            image = image * 255
            image = Image.fromarray(image.astype('uint8'), mode='L')
            image.save(os.path.join(save_dir, f'{name_list[int(label)]}_{count_[int(label)]}.png'))
            count_[int(label)] += 1
    elif 'cifar'in dataset_type:
        shape = [32, 32, -1]
        count_ = {}
        original_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i, (image, label) in enumerate(zip(images, labels)):
            # Cifar image is already 0-255
            if int(label) not in count_:
                count_[int(label)] = 0
            image = transform_invert(image, original_transform)
            # image = np.array(image).reshape(shape)
            image = Image.fromarray(image.astype('uint8'), mode='RGB')
            image.save(os.path.join(save_dir, f'{cifar_list[int(label)]}_{count_[int(label)]}.png'))
            count_[int(label)] += 1
    else:
        raise NotImplementedError(f'{dataset_type} not implemented')
    


def transform_invert(img_, transform_train):
    """
    https://blog.csdn.net/DragonGirI/article/details/107542108
    inverse transforming data. 
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        # print('Normalize')
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
        # print(img_[0])
 
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255
    return img_
    # print(img_.shape)


if __name__ == '__main__':
    # debug
    # data_dir = '/data0/yzhu/FLMPI/datasets/'
    data_dir = '/data0/yzhu/datasets/'
    rank = 5
    # dataset_type = 'cifar10'
    dataset_type = 'femnist'
    dataset = ClientDataset(dataset_type, data_dir, rank=rank)
    loader = dataset.get_train_loader(batch_size=32)
    save_data_to_picture(dataset_type, next(iter(loader)), f'./{rank}')
    
    