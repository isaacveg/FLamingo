import os
from mpi4py import MPI
WRLD = MPI.COMM_WORLD
RANK = WRLD.Get_rank()
os.environ['CUDA_VISIBLE_DEVICES'] = str(RANK%4)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# import FLamingo after setting cuda information
import sys
sys.path.append('../FLamingo/')

from FLamingo.core.client import Client
from FLamingo.core.utils.chores import merge_several_dicts
from model import *


class MetaClient(Client):
    """
    Your own Client
    """
    def init(self):
        self.model = CNNMNIST() if self.dataset_type == 'mnist' else AlexNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.993)

    def reptile_train(self, model, alpha, beta, steps, dataloader, local_epoch):
        """
        Reptile training using given functions
        """
        model.train()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = 0.0
        samples = 0
        for ep in range(local_epoch):
            dataiter = iter(dataloader)
            for idx in range(len(dataloader)//steps):
                original_model_vector = self.export_model_parameter(model)
                data, label = next(dataiter)
                for st in range(steps):
                    num, batchloss = self._train_one_batch(model, data, label, optimizer, loss_func)
                new_vec = self.export_model_parameter(model)
                original_model_vector = self.update_model(original_model_vector, new_vec, beta)
                self.set_model_parameter(original_model_vector, model)
                samples += num
                loss += batchloss * num
        loss /= samples
        return {'train_loss':loss, 'train_samples':samples}

    def local_optimization(self, model, dataloader, steps=None):
        """
        Local optimization for meta-model
        """
        if steps is None:
            steps = len(dataloader)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss()
        num, loss = 0, 0.0
        for idx, (data, target) in enumerate(dataloader):
            num, loss = self._train_one_batch(model, data, target, optimizer, loss_func)
            num += num
            loss += loss * num
        loss /= num
        return {'local_optimization_loss':loss, 'local_optimization_samples':num}
    
    def update_model(self, original, after, lr):
        """
        Update original model according to trained model
        """
        delta = after - original
        original += lr * delta
        return original

    def run(self):
        """
        Client jobs, usually have a loop
        """
        while True:
            # get from server
            data = self.listen()
            if data['status'] == 'TRAINING':
                # Using Reptile to train
                info_dict = self.reptile_train(
                    self.model, self.alpha, self.beta, self.steps, 
                    self.train_loader, self.local_epochs
                    )
                # Test before local optimization
                bf_test_dic = self.test(self.model, self.test_loader, self.loss_func, self.device)
                # Local optimization
                optim_dic = self.local_optimization(self.model, self.train_loader, self.optim_steps)
                # Test after local optimization
                test_dict = self.test(self.model, self.test_loader, self.loss_func, self.device)
                # self.log(f"train: {info_dict}\noptim: {optim_dic}\ntest: {test_dict}")
                send_dic = merge_several_dicts([info_dict, optim_dic, test_dict])
                send_dic.update({'before_test_acc':bf_test_dic['test_acc'],
                                 'before_test_loss':bf_test_dic['test_loss']}
                                 )
                # self.network.send(send_dic, 0)
                self.send(send_dic)
            elif data['status'] == 'STOP':
                self.log('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        self.log('stopped')


if __name__ == '__main__':
    client = MetaClient()
    client.run()