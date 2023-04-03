import torch
torch.use_deterministic_algorithms(True, warn_only=True)
from torch import nn
from models import varprop
import copy
import pdb

def keep_variance(x, min_variance):
            return x.clamp(min=min_variance)

class ConvBlock(nn.Module):
    def __init__(
        self,
        opt,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
    ):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size


        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=opt["min_variance"])
        self._mnv_init = opt["mnv_init"]
        self._prior_precision = opt["prior_precision"]
        self._prior_mean = opt["prior_mean"]
        self._kl_div_weight = opt['model_kl_div_weight']

        # here we need to change the conv layer to the Bayesian one
        '''
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        '''
        conv = varprop.Conv2dMNCL(in_channels, 
                                  out_channels, 
                                  kernel_size=kernel_size, 
                                  stride=stride, 
                                  padding=padding, 
                                  bias=bias, 
                                  mnv_init=self._mnv_init, 
                                  prior_precision=self._prior_precision, 
                                  prior_mean=self._prior_mean
                                  )
 
        layer = [conv]
        if opt.bn:
            if opt.preact:
                # here we need to change the bn layer to the Bayesian one
                '''
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                '''
                bn = varprop.BatchNorm2d(num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [bn]
            else:
                # here we need to change the bn layer to the Bayesian one
                '''
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                '''
                bn = varprop.BatchNorm2d(num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [conv, bn]

        if opt.activetype is not "None":
            # here we need to change the activation layer to the Bayesian one
            '''
            active = getattr(nn, opt.activetype)()
            '''
            active = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(conv)

        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)

# Shold we reimplelemnt the final block for the Bayesain model? 
class FCBlock(nn.Module):
    def __init__(self, opt, in_channels, out_channels, bias=False):
        super(FCBlock, self).__init__()

        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels
        self.out_features = out_channels

        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=opt["min_variance"])
        self._mnv_init = opt["mnv_init"]
        self._prior_precision = opt["prior_precision"]
        self._prior_mean = opt["prior_mean"]
        self._kl_div_weight = opt['model_kl_div_weight']


        # here we need to change the linear layer to the Bayesian one
        '''
        lin = nn.Linear(in_channels, out_channels, bias=bias)
        '''
        lin = varprop.LinearMNCL(in_channels, 
                                self.opt["num_classes"], 
                                prior_precision=opt.prior_precision,
                                prior_mean=opt.prior_mean,
                                mnv_init=opt.mnv_init)
        layer = [lin]
        # Why do we need these parts? 
       
        if opt.bn:
            if opt.preact:
                # here we need to change the bn layer to the Bayesian one
                # this needs to be implemented for the Bayesian case
                # ???? 
                bn = varprop.BatchNorm1d(num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                '''
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=in_channels, affine=opt.affine_bn, 
                    eps=opt.bn_eps
                )
                '''
                layer = [bn]
            else:
                # here we need to change the bn layer to the Bayesian one
                bn = varprop.BatchNorm1d(num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                '''
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=out_channels, affine=opt.affine_bn, 
                    eps=opt.bn_eps
                )
                '''
                layer = [lin, bn]

        if opt.activetype is not "None":
            # here we need to change the activation layer to the Bayesian one
            '''
            active = getattr(nn, opt.activetype)()
            '''
            active = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(lin)

        
        self.block = nn.Sequential(*layer)

    # how to rewrite this when we have one input?
    def forward(self, input):
            return self.block.forward(input)
        

def FinalBlock(opt, in_channels, bias=False):
    #pdb.set_trace()
    out_channels = opt.num_classes
    opt = copy.deepcopy(opt)
    if not opt.preact:
            opt.activetype = "None"
    return FCBlock(opt=opt, in_channels=in_channels, out_channels=out_channels, bias=bias)


def InitialBlock(opt, out_channels, kernel_size, stride=1, padding=0, bias=False):
    in_channels = opt.in_channels
    opt = copy.deepcopy(opt)
    return ConvBlock(
        opt=opt,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
