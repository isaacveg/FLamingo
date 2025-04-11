import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x*self.weight

def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m

class ConvBN(nn.Module):
    def __init__(self, do_batchnorm, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        super().__init__()
        self.pool = pool
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                              padding=1, bias=False)
        if do_batchnorm:
            self.bn = batch_norm(c_out, bn_weight_init=bn_weight_init, **kw)
        self.do_batchnorm = do_batchnorm
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.do_batchnorm:
            out = self.relu(self.bn(self.conv(x)))
        else:
            out = self.relu(self.conv(x))
        if self.pool:
            out = self.pool(out)
        return out

    def prep_finetune(self, iid, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = False
        layers = [self.conv]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = True
        return itertools.chain.from_iterable([l.parameters() for l in layers])

class Residual(nn.Module):
    def __init__(self, do_batchnorm, c, **kw):
        super().__init__()
        self.res1 = ConvBN(do_batchnorm, c, c, **kw)
        self.res2 = ConvBN(do_batchnorm, c, c, **kw)

    def forward(self, x):
        return x + F.relu(self.res2(self.res1(x)))

    def prep_finetune(self, iid, c, **kw):
        layers = [self.res1, self.res2]
        return itertools.chain.from_iterable([l.prep_finetune(iid, c, c, **kw) for l in layers])

class BasicNet(nn.Module):
    def __init__(self, do_batchnorm, channels, weight,  pool, num_classes, initial_channels=3, new_num_classes=None, **kw):
        super().__init__()
        self.new_num_classes = new_num_classes
        self.prep = ConvBN(do_batchnorm, initial_channels, channels['prep'], **kw)

        self.layer1 = ConvBN(do_batchnorm, channels['prep'], channels['layer1'],
                             pool=pool, **kw)
        self.res1 = Residual(do_batchnorm, channels['layer1'], **kw)

        self.layer2 = ConvBN(do_batchnorm, channels['layer1'], channels['layer2'],
                             pool=pool, **kw)

        self.layer3 = ConvBN(do_batchnorm, channels['layer2'], channels['layer3'],
                             pool=pool, **kw)
        self.res3 = Residual(do_batchnorm, channels['layer3'], **kw)

        self.pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(channels['layer3'], num_classes, bias=False)
        self.classifier = Mul(weight)

        self._initialize_weights()

    def forward(self, x):
        out = self.prep(x)
        out = self.res1(self.layer1(out))
        out = self.layer2(out)
        out = self.res3(self.layer3(out))

        out = self.pool(out).view(out.size()[0], -1)
        out = self.classifier(self.linear(out))
        return F.log_softmax(out, dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def finetune_parameters(self, iid, channels, weight, pool, **kw):
        #layers = [self.prep, self.layer1, self.res1, self.layer2, self.layer3, self.res3]
        self.linear = nn.Linear(channels['layer3'], self.new_num_classes, bias=False)
        self.classifier = Mul(weight)
        modules = [self.linear, self.classifier]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True
        return itertools.chain.from_iterable([m.parameters() for m in modules])
        """
        prep = self.prep.prep_finetune(iid, 3, channels['prep'], **kw)
        layer1 = self.layer1.prep_finetune(iid, channels['prep'], channels['layer1'],
                             pool=pool, **kw)
        res1 = self.res1.prep_finetune(iid, channels['layer1'], **kw)
        layer2 = self.layer2.prep_finetune(iid, channels['layer1'], channels['layer2'],
                             pool=pool, **kw)
        layer3 = self.layer3.prep_finetune(iid, channels['layer2'], channels['layer3'],
                             pool=pool, **kw)
        res3 = self.res3.prep_finetune(iid, channels['layer3'], **kw)
        layers = [prep, layer1, res1, layer2, layer3, res3]
        parameters = [itertools.chain.from_iterable(layers), itertools.chain.from_iterable([m.parameters() for m in modules])]
        return itertools.chain.from_iterable(parameters)
        """

class ResNet9(nn.Module):
    def __init__(self, do_batchnorm=False, channels=None, weight=0.125, pool=nn.MaxPool2d(2),
                 extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
        super().__init__()
        self.channels = channels or {'prep': 64, 'layer1': 128,
                                'layer2': 256, 'layer3': 512}
        self.weight = weight
        self.pool = pool
        print(f"Using BatchNorm: {do_batchnorm}")
        self.n = BasicNet(do_batchnorm, self.channels, weight, pool, **kw)
        self.kw = kw

    def forward(self, x):
        return self.n(x)

    def finetune_parameters(self):
        return self.n.finetune_parameters(self.iid, self.channels, self.weight, self.pool, **self.kw)

class ResNet(nn.Module):
    def __init__(
        self, block, num_block, base_width, num_classes=200, batch_norm=True,
    ):
        super().__init__()

        self.in_channels = 64

        self.batch_norm = batch_norm

        if self.batch_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, base_width)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, base_width)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, base_width)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, base_width)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, base_width):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(
                block(
                    self.in_channels, out_channels, stride, base_width, self.batch_norm
                )
            )
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layer_list)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """

    expansion = 4

    def __init__(
        self, in_channels, out_channels, stride=1, base_width=64, batch_norm=True
    ):
        super().__init__()

        self.batch_norm = batch_norm

        width = int(out_channels * (base_width / 64.0))
        layer_list = [
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
        ]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(width))
        layer_list += [
            nn.ReLU(inplace=True),
            nn.Conv2d(
                width, width, stride=stride, kernel_size=3, padding=1, bias=False
            ),
        ]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(width))
        layer_list += [
            nn.ReLU(inplace=True),
            nn.Conv2d(
                width, out_channels * BottleNeck.expansion, kernel_size=1, bias=False
            ),
        ]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(out_channels * BottleNeck.expansion))
        self.residual_function = nn.Sequential(*layer_list)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            layer_list = [
                nn.Conv2d(
                    in_channels,
                    out_channels * BottleNeck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
            ]
            if self.batch_norm:
                layer_list.append(nn.BatchNorm2d(out_channels * BottleNeck.expansion))
            self.shortcut = nn.Sequential(*layer_list)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

def _resnet(arch, block, num_block, base_width, num_classes, pretrained, batch_norm, model_dir="pretrained_models"):
    model = ResNet(block, num_block, base_width, num_classes, batch_norm)
    if pretrained:
        pretrained_path = "{}/{}-cifar{}.pt".format(model_dir, arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = pretrained_dict["model_state_dict"] # necessary because of our ckpt format
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def ResNet50(class_num=200, pretrained=False, model_dir="pretrained_models"):
    """ return a ResNet 50 object
    """
    return _resnet(
        "resnet50",
        BottleNeck,
        [3, 4, 6, 3],
        64,
        class_num,
        pretrained,
        batch_norm=True,
        model_dir=model_dir,
    )