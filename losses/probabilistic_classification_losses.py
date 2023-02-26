from __future__ import absolute_import
from __future__ import print_function

import torch
#torch.use_deterministic_algorithms(True)
import torch.nn as nn
from torch.nn import functional as F
import pdb


def _accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ClassificationLossVI(nn.Module):
    def __init__(self, topk=3):
        super(ClassificationLossVI, self).__init__()
        self._topk = tuple(range(1, topk+1))

    def forward(self, output_dict, target_dict):
        samples = 64
        
        prediction_mean = output_dict['prediction_mean'].unsqueeze(dim=2).expand(-1, -1, samples)
        prediction_variance = output_dict['prediction_variance'].unsqueeze(dim=2).expand(-1, -1, samples)
        target = target_dict['target1']
        target_expanded = target.unsqueeze(dim=1).expand(-1, samples)
        normal_dist = torch.distributions.normal.Normal(torch.zeros_like(prediction_mean), torch.ones_like(prediction_mean))
        if self.training:
            losses = {}
            normals =  normal_dist.sample()
            prediction = prediction_mean + torch.sqrt(prediction_variance) * normals
            # this needs to be either computed on the cpu or reimplemented in cuda
            loss = F.cross_entropy(prediction, target_expanded, reduction='mean')
            kl_div = output_dict['kl_div']
            losses['total_loss'] = loss + kl_div()
            with torch.no_grad():
                p = F.softmax(prediction, dim=1).mean(dim=2)
                losses['xe'] =  F.cross_entropy(prediction, target_expanded, reduction='mean')
                acc_k = _accuracy(p, target, topk=self._topk)
                for acc, k in zip(acc_k, self._topk):
                    losses["top%i" % k] = acc         
        else:
            with torch.no_grad():
                normals =  normal_dist.sample()
                prediction = prediction_mean + torch.sqrt(prediction_variance) * normals
                p = F.softmax(prediction, dim=1).mean(dim=2)
                losses = {}
                kl_div = output_dict['kl_div']
                losses['total_loss'] = - torch.log(p[range(p.shape[0]), target]).mean() + kl_div()
                losses['xe'] = - torch.log(p[range(p.shape[0]), target]).mean()
                acc_k = _accuracy(p, target, topk=self._topk)
                for acc, k in zip(acc_k, self._topk):
                    losses["top%i" % k] = acc
        return losses
