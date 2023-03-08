from __future__ import absolute_import
from __future__ import print_function
from tkinter import N

import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.autograd.set_detect_anomaly(True)
#torch.use_deterministic_algorithms(True)
import torch.nn as nn
from torch.nn import functional as F
import pdb


def _accuracy(output, target, topk=(1,)):
    #pdb.set_trace()
    maxk = max(topk) #3
    batch_size = target.size(0) #10
    _, pred = output.topk(maxk, 1, True, True) #torch.Size([10, 3]) gets three highest values
    pred = pred.t() # gets the indexes of the three highest values
    correct = pred.eq(target.view(1, -1))
    res = []
    for k in topk: #topk=(1,2,3) 
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ClassificationLossVI(nn.Module):
    def __init__(self, args, topk=3):
        super(ClassificationLossVI, self).__init__()
        self._topk = tuple(range(1, topk+1))
        self.label_trick = args.label_trick
        self.label_trick_valid = args.label_trick_valid
        self.coreset_training = args.coreset_training
        self.coreset_kld = args.coreset_kld
        self.merged_training = args.merged_training
        self.device = args.device

    def forward(self, output_dict, target_dict):
        
        samples = 64
        prediction_mean = output_dict['prediction_mean'].unsqueeze(dim=2).expand(-1, -1, samples)
        prediction_variance = output_dict['prediction_variance'].unsqueeze(dim=2).expand(-1, -1, samples)

        # whentraning on task_2+cr_1, do we get targets from task_1 as well? 
        target = target_dict['target1'] # this is the batch target
        #tensor([101, 150, 133,  40,  40, 133, 129,  57,  40,  40], device='cuda:0')
        target_expanded = target.unsqueeze(dim=1).expand(-1, samples) #torch.Size([10, 64])
        normal_dist = torch.distributions.normal.Normal(torch.zeros_like(prediction_mean), torch.ones_like(prediction_mean))
        
        if self.training:
            #pdb.set_trace()
            losses = {}
            normals =  normal_dist.sample()
            prediction = prediction_mean + torch.sqrt(prediction_variance) * normals  #torch.Size([10, 170, 64])
            
            # -------------------------------------------------------------------------------
            #                                 Labels trick
            # -------------------------------------------------------------------------------
            if self.label_trick is False or self.coreset_kld==1:
            #if self.label_trick is False or self.coreset_training is True: 
                # prediction and label type? float?? float16 easily overflows?
                # using autocast should be the norm
                # wrapping up the loss comutation in with torch.cuda.amp.autocast()
                # https://discuss.pytorch.org/t/cross-entropy-loss-outputting-nan/123002/4
                
                # check the dtype of prediction tensor 
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
                # label_trick is True
                #pdb.set_trace()
                #task_targets = target_dict['task_labels'][0] #shape: [10, 10]  #tensor([ 90,  77,  48,   7, 118, 135, 160, 115,  26,   4])
                task_targets = [item -30 for item in target_dict['task_labels']] #shape: [10, 10]  #tensor([ 90,  77,  48,   7, 118, 135, 160, 115,  26,   4])
                ordered_task_targets = torch.unique(torch.Tensor(task_targets).long(), sorted=True) #tensor([  4,   7,  26,  48,  77,  90, 115, 118, 135, 160])
                # Tensor([ 40,  48,  51,  57,  94, 101, 109, 129, 133, 150])
                if self.merged_training is True:
                    coreset_targets = target_dict['coresets_list'] # a list of lists [[84, 99, 87, 42, 39, 65, 156, 16, 43, 144]]
                    if len(coreset_targets)>0:
                        flat_coreset_targets = [item for sublist in coreset_targets for item in sublist] #[84, 99, 87, 42, 39, 65, 156, 16, 43, 144]
                        # minus 30
                        #flat_coreset_targets.append(task_targets.tolist())
                        seen_targets=torch.cat((torch.Tensor(flat_coreset_targets),torch.Tensor(task_targets)), 0)
                        #tensor([ 84.,  99.,  87.,  42.,  39.,  65., 156.,  16.,  43., 144.,  90.,  77., 48.,   7., 118., 135., 160., 115.,  26.,   4.])
                        #seen_targets = [item for sublist in flat_coreset_targets for item in sublist]
                        ordered_task_targets = torch.unique(seen_targets, sorted=True).long() # because tensors used as indices must be long or byte tensors or bool
                        #tensor([4,7,16,26,39,42,43,48,65,77,84,87,90,99,115,118,135,144,156,160])
                    # How to get the labels of the coresets?
                

                # Get the current batch labels (and sort them for reassignment)
                labels = target.clone().detach() #tensor([ 48, 160,  26,  43,   7,  65,   4,   4,  48,  26], device='cuda:0')
                #unq_labels = torch.unique(labels, sorted=True) # in an ascending order - tensor([0, 1], device='cuda:0')
                
                # ---------- ToDo: the unq_targets must be relabeld according to the indexes of task_targets ------------
                # this for loop is stupid: 
                for t_idx, t in enumerate(ordered_task_targets):
                    labels[labels==t] = t_idx
                '''
                # Assign new labels (0,1 ...)
                for t_idx, t in enumerate(unq_labels):
                    labels[labels == t] = t_idx
                '''   
                   
                # expand the target here

                labels_expanded = labels.unsqueeze(dim=1).expand(-1, samples)  #torch.Size([batch_size, 64])
  
                #loss_label_trick = F.cross_entropy(prediction[:, unq_labels, :], labels_expanded, reduction='mean')
                # should we use an ordered task_targets
                # calculate loss on copu and then send it back to the gpu 
                
                # Compute loss on the CPU
                prediction_cpu = prediction[:, ordered_task_targets, :].cpu() #grad_fn=<ToCopyBackward0>
                labels_cpu = labels_expanded.cpu() # does not contain device='cuda:0'
                loss_label_trick_cpu = F.cross_entropy(prediction_cpu, labels_cpu, reduction='mean') #grad_fn=<NllLoss2DBackward0>
                loss_label_trick = loss_label_trick_cpu.to(self.device)

                '''
                torch.use_deterministic_algorithms(False, warn_only=True)
                loss_label_trick = F.cross_entropy(prediction[:, ordered_task_targets, :], labels_expanded, reduction='mean')
                torch.use_deterministic_algorithms (True, warn_only=True)
                # What is this KLd? is it the reason behind all the weights in the classifier changing? 
                '''
                kl_div = output_dict['kl_div']
                # kld() needs to be changed 
                losses['total_loss'] = loss_label_trick + kl_div()

                with torch.no_grad():
                    # ToInvestigate: should we apply the label trick for caculating the accuracy and the xe as well?
                    p = F.softmax(prediction[:, ordered_task_targets, :], dim=1).mean(dim=2)
            
                    prediction_cpu = prediction[:, ordered_task_targets, :].cpu() #grad_fn=<ToCopyBackward0>
                    labels_cpu = labels_expanded.cpu() # does not contain device='cuda:0'
                    loss_label_trick_cpu = F.cross_entropy(prediction_cpu, labels_cpu, reduction='mean') #grad_fn=<NllLoss2DBackward0>
                    losses['xe'] = loss_label_trick_cpu.to(self.device)

                    '''
                    torch.use_deterministic_algorithms(False, warn_only=True)
                    losses['xe'] =  F.cross_entropy(prediction[:, ordered_task_targets, :], labels_expanded, reduction='mean')
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    '''
                    acc_k = _accuracy(p, labels, topk=self._topk)
                    for acc, k in zip(acc_k, self._topk):
                        losses["top%i" % k] = acc      
            # ---------------------------------------------------------------------------------------------------
            
        else:
              
              if self.label_trick and self.label_trick_valid: # similar to the testing of multi head architecture
                   
                    with torch.no_grad():
                        normals = normal_dist.sample()  # we have this sampling phase here
                        prediction = prediction_mean + torch.sqrt(prediction_variance) * normals
                        
                        labels = target.clone().detach() # all the labels in the batch- suffle=false, they are all the same
                        #unq_labels = torch.unique(labels, sorted=True) # in an ascending order - tensor([0, 1], device='cuda:0')
                        
                        task_targets = target_dict['task_labels'][0] #shape: [10, 10]
                        ordered_task_targets = torch.unique(task_targets, sorted=True)
                        
                        for t_idx, t in enumerate(ordered_task_targets):
                            labels[labels==t] = t_idx
                        '''
                        # Assign new labels (0,1 ...)
                        for t_idx, t in enumerate(unq_labels):
                            labels[labels == t] = t_idx
                        '''
                        losses = {}
                        kl_div = output_dict['kl_div']
                        
                        # ---ToDo: try the original loss for validation----
                        
                        p = F.softmax(prediction[:, ordered_task_targets, :], dim=1).mean(dim=2)
                        losses['total_loss'] = - torch.log(p[range(p.shape[0]), labels]).mean() + kl_div()
                        losses['xe'] = - torch.log(p[range(p.shape[0]), labels]).mean()
                        
                        
                        # expand the target here 
                        #labels_expanded = labels.unsqueeze(dim=1).expand(-1, samples)  #torch.Size([batch_size, 64])
                        #loss_label_trick = F.cross_entropy(prediction[:, ordered_task_targets, :], labels_expanded, reduction='mean')
                     
                        # calculate accuracy 
                        #losses['xe'] =  F.cross_entropy(prediction[:, ordered_task_targets, :], labels_expanded, reduction='mean')
                        acc_k = _accuracy(p, labels, topk=self._topk)
                        for acc, k in zip(acc_k, self._topk):
                            losses["top%i" % k] = acc
                        
                        
              
              else: 
                    #pdb.set_trace()
                    with torch.no_grad():
                        normals = normal_dist.sample()
                        prediction = prediction_mean + torch.sqrt(prediction_variance) * normals  #torch.Size([10, 170, 64])
                        p = F.softmax(prediction, dim=1).mean(dim=2) #torch.Size([10, 170])
                        losses = {}
                        kl_div = output_dict['kl_div']
                        # kld() is calculated considering the current task's indexed weights - here it needs to be calculated for the validation set considering the whole output - how to do this?
                        # ToInvestigate: why validation loss is calculated like this? I still do not know! 
                        ## target: all the labels in the batch- suffle=false, they are all the same
                        losses['total_loss'] = - torch.log(p[range(p.shape[0]), target]).mean() + kl_div()
                        losses['xe'] = - torch.log(p[range(p.shape[0]), target]).mean() #tensor(6.7160, device='cuda:0')
                    
                        acc_k = _accuracy(p, target, topk=self._topk)
                        for acc, k in zip(acc_k, self._topk):
                            losses["top%i" % k] = acc
        return losses

    def set_coreset_kld_flag(self, _flag):
        self.coreset_kld=_flag