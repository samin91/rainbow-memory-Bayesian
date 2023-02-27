import torch
import torch.nn as nn

from models.layers import ConvBlock, InitialBlock, FinalBlock

from models import varprop
import torch.nn.functional as F
import logging
import pdb
#torch.use_deterministic_algorithms(True, warn_only=True)
torch.autograd.set_detect_anomaly(True)


def head_size(split_targets):
    t=0
    for targets in split_targets:
        t+=len(targets)
    return t

def keep_variance(x, min_variance):
    return x.clamp(min=min_variance)
# why are initializing our bayesian layers with different values tha than 
# the original pytorch implementation?
def finitialize(modules, small=False):
    logging.info("Initializing MSRA")
    for layer in modules:
        print("Layer: ", layer)
        if isinstance(layer, (varprop.Conv2dMNCL, varprop.LinearMNCL)):
            nn.init.kaiming_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, varprop.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, mnv_init=-3.0, prior_precision=1e0, prior_mean=0.0):
    """3x3 convolution with padding"""
    return varprop.Conv2dMNCL(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation, mnv_init=mnv_init, prior_precision=prior_precision, prior_mean=prior_mean)

def conv1x1(in_planes, out_planes, stride=1, mnv_init=-2.0, prior_precision=1e2, prior_mean=0.0):
    """1x1 convolution"""
    return varprop.Conv2dMNCL(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                    mnv_init=mnv_init, prior_precision=prior_precision, prior_mean=prior_mean)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, keep_variance_fn=None, mnv_init=-3.0, prior_precision=1e0, prior_mean=0.0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = varprop.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, mnv_init=mnv_init, prior_precision=prior_precision, prior_mean=prior_mean)
        self.bn1 = norm_layer(planes)
        self.relu = varprop.ReLU(keep_variance_fn=keep_variance_fn)
        self.conv2 = conv3x3(planes, planes, mnv_init=mnv_init, prior_precision=prior_precision, prior_mean=prior_mean)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward_sampling(self, inputs):
        identity = inputs

        out = self.conv1.forward_sampling(inputs)
        out = self.bn1.forward_sampling(out)
        out = F.relu(out)

        out = self.conv2.forward_sampling(out)
        out = self.bn2.forward_sampling(out)

        if self.downsample is not None:
            identity = self.downsample.forward_sampling(inputs)

        out += identity
        out = F.relu(out)

        return out


    def forward(self, inputs_mean, inputs_variance):
        identity_mean, identity_variance = inputs_mean, inputs_variance

        out = self.conv1(inputs_mean, inputs_variance)
        out = self.bn1(*out)
        out = self.relu(*out)

        out = self.conv2(*out)
        out_mean, out_variance = self.bn2(*out)

        if self.downsample is not None:
            identity_mean, identity_variance = self.downsample(inputs_mean, inputs_variance)

        out_mean += identity_mean
        out_variance += identity_variance
        out_mean, out_variance = self.relu(out_mean, out_variance)

        return out_mean, out_variance

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, keep_variance_fn=None, mnv_init=-3.0, prior_precision=1e0, prior_mean=0.0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = varprop.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = varprop.ReLU(keep_variance_fn=keep_variance_fn)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs_mean, inputs_variance):
        identity_mean, identity_variance = inputs_mean, inputs_variance

        out = self.conv1(inputs_mean, inputs_variance)
        out = self.bn1(*out)
        out = self.relu(*out)

        out = self.conv2(*out)
        out = self.bn2(*out)
        out = self.relu(*out)

        out = self.conv3(*out)
        out_mean, out_variance = self.bn3(*out)

        if self.downsample is not None:
            identity_mean, identity_variance = self.downsample(inputs_mean, inputs_variance)

        out_variance += identity_variance
        out_mean, out_variance = self.relu(out_mean, out_variance)

        return out_mean, out_variance
    
class ImageResNetMNCL(nn.Module):

    def __init__(self, opt, block, layers, zero_init_residual=False, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None
                 ):
        super(ImageResNetMNCL, self).__init__()

        self.opt = opt
        if norm_layer is None:
            norm_layer = varprop.BatchNorm2d
        self._norm_layer = norm_layer

        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=opt["min_variance"])
        self._mnv_init = opt["mnv_init"]
        self._prior_precision = opt["prior_precision"]
        self._prior_mean = opt["prior_mean"]
        # calculate num_classes
        '''
        self.num_classes = head_size(args.split_targets)
        self.split_targets = args.split_targets
        self.current_task=current_task
        self.label_trick_valid = args.label_trick_valid
        self.coreset_training = args.coreset_training
        self.coreset_kld = args.coreset_kld
        
        '''

        
        # Related to the multi head architecture
        '''
        self._num_classes_per_task = [len(targets) for targets in args.split_targets]
        self._active_task = 0
        self._num_tasks = args.num_tasks
        assert(self._num_tasks == len(self._num_classes_per_task))
        '''
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        #self.conv1 = varprop.Conv2dMNCL(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, mnv_init=self._mnv_init, prior_precision=self._prior_precision, prior_mean=self._prior_mean)
        self.conv1 = varprop.Conv2dMNCL(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, 
                                        mnv_init=self._mnv_init, prior_precision=self._prior_precision, prior_mean=self._prior_mean)
        self.bn1 = varprop.BatchNorm2d(self.inplanes)
        self.relu = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)

        #self.maxpool = varprop.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = varprop.AdaptiveAvgPool2d()
        # to adapt to RM, we need to replaece the fully connected layer
        # it is a sequential object
        #self.fc = FinalBlock(opt=opt, in_channels=512 * block.expansion)
        self.fc = varprop.LinearMNCL(512 * block.expansion, 
                                    self.opt["num_classes"],
                                    mnv_init=opt.mnv_init, 
                                    prior_precision=opt.prior_precision)
        # what about initializing the fc layre inside the final block? 
        finitialize(self.modules(), small=False)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # wasn't in the core initially
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = varprop.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, mnv_init=self._mnv_init, prior_precision=self._prior_precision, prior_mean=self._prior_mean),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            keep_variance_fn=self._keep_variance_fn, mnv_init=self._mnv_init, prior_precision=self._prior_precision, prior_mean=self._prior_mean))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                keep_variance_fn=self._keep_variance_fn, mnv_init=self._mnv_init, prior_precision=self._prior_precision, prior_mean=self._prior_mean))

        return varprop.Sequential(*layers)


    def _forward_impl(self, x):
        # check for nan values in the inputs and layers
       
        has_nan = torch.isnan(x).any()
        if has_nan:
            print("The tensor contains at least one NaN value")
        x_variance = torch.zeros_like(x)

        x = self.conv1(x, x_variance)
        for tensor in x:
            if torch.isnan(tensor).any():
                has_nan=True
                break
        if has_nan:
            print("The tensor contains at least one NaN value")
        
    
        x = self.bn1(*x)
        for tensor in x:
            if torch.isnan(tensor).any():
                has_nan=True
                break
        if has_nan:
            print("The tensor contains at least one NaN value")
        

        x = self.relu(*x)
        for tensor in x:
            if torch.isnan(tensor).any():
                has_nan=True
                break
        if has_nan:
            print("The tensor contains at least one NaN value")

       
        # Remove max_pooling because the implementation of 
        # this layer is non-deterministic 
        #x = self.maxpool(*x)
        
        x = self.layer1(*x)
        for tensor in x:
            if torch.isnan(tensor).any():
                has_nan=True
                break
        if has_nan:
            print("The tensor contains at least one NaN value")

        x = self.layer2(*x)
        for tensor in x:
            if torch.isnan(tensor).any():
                has_nan=True
                break
        if has_nan:
            print("The tensor contains at least one NaN value")

        x = self.layer3(*x)
        for tensor in x:
            if torch.isnan(tensor).any():
                has_nan=True
                break
        if has_nan:
            print("The tensor contains at least one NaN value")

        x = self.layer4(*x)
        for tensor in x:
            if torch.isnan(tensor).any():
                has_nan=True
                break
        if has_nan:
            print("The tensor contains at least one NaN value")

        x_mean, x_variance = self.avgpool(*x)
        has_nan = torch.isnan(x_mean).any()
        if has_nan:
            print("The tensor contains at least one NaN value")
        has_nan = torch.isnan(x_variance).any()
        if has_nan:
            print("The tensor contains at least one NaN value")
        x_mean = torch.flatten(x_mean, 1)
        x_variance = torch.flatten(x_variance, 1)
        out_mean, out_variance = self.fc.forward(x_mean, x_variance)
        has_nan = torch.isnan(out_mean).any()
        if has_nan:
            print("The tensor contains at least one NaN value")
        has_nan = torch.isnan(out_variance).any()
        if has_nan:
            print("The tensor contains at least one NaN value")
        # out_mean, out_variance = self.fc3[self._active_task](x_mean, x_variance)
        return {'prediction_mean':out_mean, 'prediction_variance':out_variance, 'kl_div':self.kl_div}

    def forward_sampling(self, inputs):
        
        x = self.conv1.forward_sampling(inputs)
        x = self.bn1.forward_sampling(x)
        x = F.relu(x)

        x = self.layer1.forward_sampling(x)
        x = self.layer2.forward_sampling(x)
        x = self.layer3.forward_sampling(x)
        x = self.layer4.forward_sampling(x)
        
        x = self.avgpool.forward_sampling(x)
        x = torch.flatten(x, 1)

        out = self.fc.forward_sampling(x)
        #out = self.fc3[self._active_task].forward_sampling(x)
        return out

    def forward(self, x):
        return self._forward_impl(x)
    
    # Extra functions that only apply to the Bayesian model in CL scenario
    def kl_div(self):
        kl_div_weight = self.opt["model_kl_div_weight"]
        kl = 0.0
        for module in self.modules():
            if isinstance(module, (varprop.LinearMNCL, varprop.Conv2dMNCL)):
                kl += module.kl_div()
        # here we need to multiply the kl with the weight as well: self._kl_div_weight

        return kl*kl_div_weight

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

def ResNet(opt):
    # how to access the above function when here we have a sunction and not a class? 
    if opt.depth==18:
        model = ImageResNetMNCL(opt, BasicBlock, [2, 2, 2, 2])
    elif opt.depth == 34:
        model = ImageResNetMNCL(opt, BasicBlock, [3, 4, 6, 3])
    # I can create a dictionary from the output of the model 
    return model