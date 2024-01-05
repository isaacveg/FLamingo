import torch
import torch.nn as nn
import torch.nn.functional as F


# cnn model for MNIST
class CNNMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4 * 4 * 64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # flatten
        out = self.fc(out)
        out = self.softmax(out)
        return out
    

# AlexNet model for CIFAR10
class Base_Model():
    def get_num_paras(self):
        return sum(p.numel() for p in self.parameters())

    def apply_paras(self, new_para):
        count_paras = 0

        paras = self.named_parameters()
        para_dict = dict(paras)

        with torch.no_grad():
            for n,_ in para_dict.items():
                if 'bn' not in n:
                    para_dict[n].set_(new_para[count_paras].float().to(para_dict[n].device))
                count_paras += 1

        # para_dict = self.state_dict()
        # keys=list(self.state_dict())
        # for i in range(len(keys)):
        #     para_dict.update({keys[i]: new_para[i]})
        # self.load_state_dict(para_dict)
        # del para_dict
        # for p in self.parameters():
        #     p.data = 
    
    def get_grads(self):
        return [p[1].grad.clone().detach() for p in self.named_parameters()]

    def get_paras(self):
        return [p[1].data.clone().detach() for p in self.named_parameters()]
    
    def get_para_shapes(self):
        return [p[1].shape for p in self.named_parameters()]

    def get_para_names(self):        
        self_para_name = []
        for p in self.named_parameters():
            if p[1].requires_grad:
                self_para_name.append(p[0])
        return self_para_name


class AlexNet_IMAGE(nn.Module, Base_Model):
    def __init__(self):
        super(AlexNet_IMAGE, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,kernel_size=7, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*5*5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 100)

        self.mask = torch.ones([self.get_num_paras()])

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = F.relu(self.conv2(x), inplace=True)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = F.relu(self.conv3(x), inplace=True)
        x = self.bn3(x)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.bn4(x)
        x = F.relu(self.conv5(x), inplace=True)
        x = self.bn5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # F.batch_norm()
        # print(" x shape ",x.size())
        x = x.view(-1,256*5*5)
        F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
    
class AlexNet2(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    def forward(self, x):      # 如果输入参数为特征，那么不在使用conv_layer，而是直接从下一步计算
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class AlexNet(nn.Module):
    def __init__(self,class_num=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 3 32 32
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            # 64 32+2-2=32 32/2=16
            nn.ReLU(inplace=True),
            # 64 16 16
            nn.MaxPool2d(kernel_size=2),
            # 64 8 8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 192 8 8
            nn.MaxPool2d(kernel_size=2),
            # 192 4 4
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 384 4 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 256 4 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 256 2 2
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, class_num),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)