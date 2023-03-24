from __future__ import absolute_import
from __future__ import print_function

import torch
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn
from torch.nn import functional as F
import pdb


class ClassificationLossVI(nn.Module):
    def __init__(self, topk=3):
        super(ClassificationLossVI, self).__init__()
        self._topk = tuple(range(1, topk+1))

    def forward(self, output_dict, target_dict):
        
        samples = 64
        
        prediction_mean = output_dict['prediction_mean'].unsqueeze(dim=2).expand(-1, -1, samples)
        # shape:torch.Size([10, 10, 64]) (batch_size, num_classes, samples)
        has_nan = torch.isnan(prediction_mean).any()
        if has_nan:
            print('prediction_mean tensor contains at least one nan value')
        prediction_variance = output_dict['prediction_variance'].unsqueeze(dim=2).expand(-1, -1, samples)
        # shape:torch.Size([10, 10, 64]) --> (batch_size, num_classes, samples)
        has_nan = torch.isnan(prediction_variance).any()
        if has_nan:
            print('prediction_variance tensor contains at least one nan value')
        #target = target_dict['target1']
        target= target_dict
        # shape: torch.Size([10]) --> (batch_size)
        target_expanded = target.unsqueeze(dim=1).expand(-1, samples)
        # shape: torch.Size([10, 64]) --> (batch_size, samples)
        normal_dist = torch.distributions.normal.Normal(torch.zeros_like(prediction_mean), 
                                                        torch.ones_like(prediction_mean))
        
        if self.training:
            losses = {}
            normals =  normal_dist.sample()
            # shape: torch.Size([10, 10, 64]) --> (batch_size, num_classes, samples)
            prediction = prediction_mean + torch.sqrt(prediction_variance) * normals
            # shape: torch.Size([10, 10, 64]) --> (batch_size, num_classes, samples)
            has_nan = torch.isnan(prediction).any()
            if has_nan:
                print('prediction tensor contains at least one nan value')
            # add prediction which is out logit to the loss dict
            losses['prediction'] = prediction
            # this needs to be either computed on the cpu or reimplemented in cuda
            # Implement the following yourself
            #loss = F.cross_entropy(prediction, target_expanded, reduction='mean')
            p = F.softmax(prediction, dim=1).mean(dim=2)
            # change the target type from double to long()
            # to(dtype=torch.long)
            loss = - torch.log(p[range(p.shape[0]), target.to(dtype=torch.long)]).mean()
            
            if torch.isnan(loss):
                print('loss is nan')
            losses['xe'] = loss
            kl_div = output_dict['kl_div']
            losses['total_loss'] = loss + kl_div()
     
        else:
            with torch.no_grad():
                normals =  normal_dist.sample()
                prediction = prediction_mean + torch.sqrt(prediction_variance) * normals
                # add prediction to the loss dictionary when we are evaluating 
                losses['prediction'] = prediction
                p = F.softmax(prediction, dim=1).mean(dim=2)
                losses = {}
                kl_div = output_dict['kl_div']
                _xe = - torch.log(p[range(p.shape[0]), target.to(dtype=torch.long)]).mean()
                losses['total_loss'] = _xe + kl_div()
                losses['xe'] = _xe
          
        return losses
