import torch
torch.use_deterministic_algorithms(True, warn_only=True)
from torch import nn
from models import varprop
import copy
import pdb

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
        
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        layer = [conv]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                layer = [conv, bn]

        if opt.activetype is not "None":
            active = getattr(nn, opt.activetype)()
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
        lin = nn.Linear(in_channels, out_channels, bias=bias)
        layer = [lin]
        # Why do we need these parts? 
       
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=in_channels, affine=opt.affine_bn, 
                    eps=opt.bn_eps
                )
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=out_channels, affine=opt.affine_bn, 
                    eps=opt.bn_eps
                )
                layer = [lin, bn]

        if opt.activetype is not "None":
            active = getattr(nn, opt.activetype)()
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(lin)

        
        self.block = nn.Sequential(*layer)

    # how to rewrite this when we have one input?
    def forward(self, input):
            return self.block.forward(input)
        

# FC class for Bayesian model
class FCBlock_bayes(nn.Module):
    def __init__(self, opt, in_channels, out_channels, bias=False):
        super(FCBlock_bayes, self).__init__()

        #pdb.set_trace()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels
        self.out_features = out_channels
        
        self.lin = varprop.LinearMNCL(in_channels, out_channels, 
                                      mnv_init=opt.mnv_init, 
                                      prior_precision=opt.prior_precision)

    # how to rewrite this when we have one input?
    def forward(self, input_mean, input_var):
        return self.lin(input_mean, input_var)
        

def FinalBlock(opt, in_channels, bias=False):
    #pdb.set_trace()
    out_channels = opt.num_classes
    opt = copy.deepcopy(opt)
    
    if opt["bayesian"] is True:
        return FCBlock_bayes(
            opt=opt, in_channels=in_channels, out_channels=out_channels, bias=bias
        )
    else: 
        if not opt.preact:
            opt.activetype = "None"
        return FCBlock(
            opt=opt, in_channels=in_channels, out_channels=out_channels, bias=bias
        )


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
