import logging
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn

from models.layers_mnvi import ConvBlock, InitialBlock, FinalBlock

from models import varprop
import torch.nn.functional as F
import pdb
torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger()

def finitialize(modules, small=False):
    logger.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, varprop.Conv2dMNCL):
            #logger.info("Conv2dMNCL instance detected")
            nn.init.kaiming_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, varprop.LinearMNCL):
            #logger.info("LinearMNC instance detected")
            nn.init.kaiming_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, varprop.BatchNorm2d):
            #logger.info("BatchNorm2 instance detected")
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        expansion = 1
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x, x_variance):

        _out = self.conv1(x, x_variance)
        _out_mean, _out_variance = self.conv2(*_out)
        if self.downsample is not None:
            shortcut_mean, shortcut_variance = self.downsample(x, x_variance)
        else:
            shortcut_mean, shortcut_variance = x, x_variance
        _out_mean = _out_mean + shortcut_mean
        _out_variance = _out_variance + shortcut_variance
        return _out_mean, _out_variance


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        expansion = 4
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv3 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.downsample = downsample

    def forward(self, x):
        _out = self.conv1(x)
        _out = self.conv2(_out)
        _out = self.conv3(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        _out = _out + shortcut
        return _out


class ResidualBlock(nn.Module):
    def __init__(self, opt, block, inChannels, outChannels, depth, stride=1):
        super(ResidualBlock, self).__init__()
        if stride != 1 or inChannels != outChannels * block.expansion:
            downsample = ConvBlock(
                opt=opt,
                in_channels=inChannels,
                out_channels=outChannels * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            downsample = None
        self.blocks = varprop.Sequential()
        self.blocks.add_module(
            "block0", block(opt, inChannels, outChannels, stride, downsample)
        )
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module(
                "block{}".format(i), block(opt, inChannels, outChannels)
            )

    def forward(self, x, x_variance):
        return self.blocks(x, x_variance)


class ResNet(nn.Module):
    def __init__(self, opt):
        super(ResNet, self).__init__()
        
        depth = opt.depth
        self._kl_div_weight = opt.model_kl_div_weight

        if depth in [20, 32, 44, 56, 110, 1202]:
            blocktype, self.nettype = "BasicBlock", "cifar"
        elif depth in [164, 1001]:
            blocktype, self.nettype = "BottleneckBlock", "cifar"
        # we are here
        elif depth in [18, 34]:
            blocktype, self.nettype = "BasicBlock", "imagenet"
        elif depth in [50, 101, 152]:
            blocktype, self.nettype = "BottleneckBlock", "imagenet"
        assert depth in [20, 32, 44, 56, 110, 1202, 164, 1001, 18, 34, 50, 101, 152]

        if blocktype == "BasicBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 6 == 0, (
                "Depth should be 6n+2, and preferably one of 20, 32, 44, 56, 110, 1202"
            )
            n = (depth - 2) // 6
            block = BasicBlock
            in_planes, out_planes = 16, 64
        elif blocktype == "BottleneckBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 9 == 0, "Depth should be 9n+2, and preferably one of 164 or 1001"
            n = (depth - 2) // 9
            block = BottleneckBlock
            in_planes, out_planes = 16, 64
        # we are here 
        elif blocktype == "BasicBlock" and self.nettype == "imagenet":
            assert depth in [18, 34]
            num_blocks = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
            block = BasicBlock
            in_planes, out_planes = 64, 512  # 20, 160

        elif blocktype == "BottleneckBlock" and self.nettype == "imagenet":
            assert depth in [50, 101, 152]
            if depth == 50:
                num_blocks = [3, 4, 6, 3]
            elif depth == 101:
                num_blocks = [3, 4, 23, 3]
            elif depth == 152:
                num_blocks = [3, 8, 36, 3]
            block = BottleneckBlock
            in_planes, out_planes = 64, 512
        else:
            assert 1 == 2

        self.num_classes = opt.num_classes

        # initial block
        self.initial = InitialBlock(
            opt=opt, out_channels=in_planes, kernel_size=3, stride=1, padding=1
        )

        if self.nettype == "cifar":
            self.group1 = ResidualBlock(opt, block, 16, 16, n, stride=1)
            self.group2 = ResidualBlock(
                opt, block, 16 * block.expansion, 32, n, stride=2
            )
            self.group3 = ResidualBlock(
                opt, block, 32 * block.expansion, 64, n, stride=2
            )
        # we are here 
        elif self.nettype == "imagenet":
            self.group1 = ResidualBlock(
                opt, block, 64, 64, num_blocks[0], stride=1
            )  # For ResNet-S, convert this to 20,20
            self.group2 = ResidualBlock(
                opt, block, 64 * block.expansion, 128, num_blocks[1], stride=2
            )  # For ResNet-S, convert this to 20,40
            self.group3 = ResidualBlock(
                opt, block, 128 * block.expansion, 256, num_blocks[2], stride=2
            )  # For ResNet-S, convert this to 40,80
            self.group4 = ResidualBlock(
                opt, block, 256 * block.expansion, 512, num_blocks[3], stride=2
            )  # For ResNet-S, convert this to 80,160
        else:
            assert 1 == 2
        
        
        # replace this with bayesian adaptive pooling 
        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool  = varprop.AdaptiveAvgPool2d()
        
        self.dim_out = out_planes * block.expansion
        

        # fully conected
        self.fc = FinalBlock(opt=opt, in_channels=out_planes * block.expansion)
        # initialize weights

        finitialize(self.modules(), small=False)

        # original code
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Extra functions that only apply to the Bayesian model in CL scenario
    def kl_div(self):
        #kl_div_weight = self.opt["model_kl_div_weight"]
        kl = 0.0
        for module in self.modules():
            if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                kl += module.kl_div()
        # here we need to multiply the kl with the weight as well: self._kl_div_weight
        return kl*self._kl_div_weight

    def save_prior_and_weights(self, prior_conv_func):
            for module in self.modules():
                if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                    module.save_prior_and_weights(prior_conv_func)

    def update_prior(self):
        for module in self.modules():
            if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                module.update_prior()
    
    def update_prior_and_weights_from_saved(self):
            for module in self.modules():
                if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                    module.update_prior_and_weights_from_saved()

    def update_weights_from_saved(self):
        for module in self.modules():
            if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                module.update_weights_from_saved()

    def get_variance(self):
        raw_variances = []
        variances = []
        for module in self.modules():
            if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                raw_var, var = module.get_variance()
                raw_variances.append(raw_var)
                variances.append(var)
        
        return raw_variances, variances

    def get_prior_variance(self):
        prior_variances = []
        for module in self.modules():
            if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                prior_variances.append(module.get_prior_variance())

        return prior_variances

    def forward(self, x):
        x_variance = torch.zeros_like(x)

        out = self.initial(x, x_variance)
        out = self.group1(*out)
        out = self.group2(*out)
        out = self.group3(*out)
        if self.nettype == "imagenet":
            out = self.group4(*out)
        out = self.pool(*out)
        out_mean, out_variance = self.pool(*out)
        # here flattening should be done seperately for both tensors: mean and variance
        out_mean = out_mean.view(x.size(0), -1)
        out_variance = out_variance.view(x.size(0), -1)
        out_mean, out_variance = self.fc(out_mean, out_variance)
        return {'prediction_mean':out_mean, 'prediction_variance':out_variance, 'kl_div':self.kl_div}



