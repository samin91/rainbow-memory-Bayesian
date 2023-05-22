"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import time
import logging
import random
import os
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Ray Tune
import ray
from ray.tune import Trainable 
from ray import tune
from utils.train_utils import select_model, select_optimizer

from utils.early_stopping import EarlyStopping
from methods.finetune import Finetune
# for ray.tune()
from utils.data_loader import cutmix_data, ImageDataset
import pdb
from tqdm import tqdm

logger = logging.getLogger()
# log = f"tensorboard/Run_{}" ???
#writer = SummaryWriter(f"test/run_{1}")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class RM(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        
        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.n_epochs = kwargs["n_epoch"]
        self.exp_env = kwargs["stream_env"]
        self.bayesian = kwargs["bayesian_model"]
        self.pretrain = kwargs['pretrain']
        self.scheduler_name = kwargs["sched_name"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"

   # --------------------------------------------------------------------------------------------------
   # For Ray Tune
   # --------------------------------------------------------------------------------------------------
    '''
    def setup(self, config):
        pdb.set_trace()
        self.optimizer = select_optimizer(self.opt_name, config['lr'], config['weight_decay'], self.model, self.sched_name)
    '''

    ''' add the dataloader function here and see if it makes a difference 
    
    '''

    # config
    def find_hyperparametrs(self, config):
        
        #batch_size = self.batch_size
        n_worker = self.n_workers
        cur_iter = 0
        batch_size = self.batch_size


        self.optimizer, self.scheduler = select_optimizer(self.opt_name, config['lr'], config['weight_decay'], self.model, self.sched_name)

       
        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=(batch_size // 2),
                num_workers=n_worker,
                pin_memory=True,
            )
            stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            stream_batch_size = batch_size
        

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list
        valid_list = self.valid_list
        random.shuffle(train_list)
        
        
        # Configuring a batch with streamed and memory data equally.
       
        train_loader, test_loader, valid_loader=self.get_dataloader(stream_batch_size, n_worker, train_list, test_list, valid_list)
      
        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(self.streamed_list)+len(self.memory_list)}")
        logger.info(f"Test samples: {len(self.test_list)}")
        logger.info(f"Valid samples: {len(self.valid_list)}")
        
        # TRAIN
        eval_dict = dict()
        #eval_dict_valid = dict()
        '''ToDo: should we also put the loss function on the device? 
        '''

        '''
        if self.pretrain is True:
            
            # automatically know wheich model is supposed to get the weights? 
            if self.bayesian is False:
                # Load the pre-trained weights
                pretrained_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
                #pretrained_dict = torch.load('path/to/pretrained_weights.pth')
                # Filter out unnecessary keys
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                   if k in model_dict and k not in ('fc.weight', 'fc.bias')}

                # Load the pre-trained weights into the model
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
        '''
     
        self.model = self.model.to(self.device)
        
        for epoch in range(self.n_epochs):
           
            # initialize for each task
            # optimizer.param_groups is a python list, which contains a dictionary.
            if self.scheduler_name == "cos":
                if epoch <= 0:  # Warm start of 1 epoch
                    for param_group in self.optimizer.param_groups:
                        # param_group is the dict inside the list and is the only item in this list.
                        if self.bayesian is True:
                            param_group["lr"] = self.lr *0.1  # self.lr * 0.1   this was changed due to inf error
                        else:
                            param_group["lr"] = self.lr * 0.1
                elif epoch == 1:  # Then set to maxlr
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lr
                else:  # Aand go!
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                if self.scheduler is not None:
                    self.scheduler.step()

            # Training
            
            train_loss, train_acc = self._train(train_loader=train_loader, memory_loader=memory_loader, optimizer=self.optimizer, criterion=self.criterion)
            

            # Validation (validating over all the test sets seen so far)
            eval_dict_valid = self.evaluation(
                valid_loader, criterion=self.criterion
            )

            # Communicate with Ray tune
            with tune.checkpoint_dir(epoch) as checkpoint_dir: # what should be the checkpoint_dir will be?
                path = os.path.join("/visinf/home/shamidi/Projects/rainbow-memory-Bayesian", "ray_checkpoints", "checkpoint")
                torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

            tune.report(
                loss=eval_dict_valid["avg_loss"], accuracy=eval_dict_valid["avg_acc"]
                )
            

            # Testing(testing over all the test sets seen so far)
            eval_dict = self.evaluation(
                test_loader, criterion=self.criterion
            )
            
            # Report the results on the current epoch
            
            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{self.n_epochs} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"valid_loss {eval_dict_valid['avg_loss']:.4f} | valid_acc {eval_dict_valid['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
                )
            

      
       
           
    

    # GENERAL TRAINING
    def train(self, cur_iter, n_epoch, batch_size, n_worker, writer, n_passes=0):
        
    
        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=(batch_size // 2),
                num_workers=n_worker,
                pin_memory=True,
            )
            stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            stream_batch_size = batch_size

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list
        valid_list = self.valid_list
        random.shuffle(train_list)
        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader, valid_loader  = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list, valid_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)+len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")
        logger.info(f"Valid samples: {len(valid_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        early_stopping = EarlyStopping(patience=20, verbose=True)

        '''ToDo: should we also put the loss function on the device? 
        '''
       
        if self.pretrain is True:
            
            # automatically know wheich model is supposed to get the weights? 
            if self.bayesian is False:
                # Load the pre-trained weights
                pretrained_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
                #pretrained_dict = torch.load('path/to/pretrained_weights.pth')
                # Filter out unnecessary keys
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                   if k in model_dict and k not in ('fc.weight', 'fc.bias')}

                # Load the pre-trained weights into the model
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)



        self.model = self.model.to(self.device)
        
        for epoch in range(n_epoch):
            train_start=0
            infer_start=0
            train_end=0
            infer_end=0
            '''
            # -------------------------------------------------------
            # For the first task we need a larger learning_rate
            #--------------------------------------------------------
            if cur_iter==0:
                lr = 0.03
                # initialize for each task
                # optimizer.param_groups is a python list, which contains a dictionary.
                if epoch <= 0:  # Warm start of 1 epoch
                    for param_group in self.optimizer.param_groups:
                        # param_group is the dict inside the list and is the only item in this list.
                        if self.bayesian is True:
                            param_group["lr"] = lr * 0.1 # self.lr * 0.1   this was changed due to inf error
                        else:
                            param_group["lr"] = lr * 0.1
                elif epoch == 1:  # Then set to maxlr
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                else:  # Aand go!
                    if self.scheduler is not None:
                        self.scheduler.step()


            else: 
            '''
            # initialize for each task
            # optimizer.param_groups is a python list, which contains a dictionary.
            if self.scheduler_name == "cos":
                if epoch <= 0:  # Warm start of 1 epoch
                    for param_group in self.optimizer.param_groups:
                        # param_group is the dict inside the list and is the only item in this list.
                        if self.bayesian is True:
                            param_group["lr"] = self.lr *0.1  # self.lr * 0.1   this was changed due to inf error
                        else:
                            param_group["lr"] = self.lr * 0.1
                elif epoch == 1:  # Then set to maxlr
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lr
                else:  # Aand go!
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                if self.scheduler is not None:
                    self.scheduler.step()

            # Training
            train_start = time.time()
            train_loss, train_acc = self._train(train_loader=train_loader, memory_loader=memory_loader,
                                                optimizer=self.optimizer, criterion=self.criterion)
            train_end = time.time() - train_start

            # Validation (validating over all the test sets seen so far)
            eval_dict_valid = self.evaluation(
                valid_loader, criterion=self.criterion
            )
        
            '''
            # Communicate with Ray tune
            with tune.checkpoint_dir(epoch) as checkpoint_dir: # what should be the checkpoint_dir will be?
                path = os.path.join(checkpoint_dir, "ray_checkpoints", "checkpoint")
                torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

            tune.report(
                loss=eval_dict_valid["avg_loss"], accuracy=eval_dict_valid["avg_acc"]
                )
            '''
            # Testing (testing over all the test sets seen so far)
            infer_start = time.time()
            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )
            infer_end = time.time() - infer_start

            # ------------------------------------------------------------
            # Tensorboard
            # ------------------------------------------------------------
            '''
            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            # add a plot for total loss and xe loss for the Bayesian case
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )
            '''
            # Train
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)

            # Valid
            
            writer.add_scalar('Accuracy/valid-',eval_dict_valid["avg_acc"], epoch)
            writer.add_scalar('Loss/valid-', eval_dict_valid["avg_loss"] , epoch)
            
            # Test
            writer.add_scalar('Accuracy/test-',eval_dict["avg_acc"], epoch)
            writer.add_scalar('Loss/test-', eval_dict["avg_loss"] , epoch)
            # -------------------------------------------------------------------
            # Logging to console
            # -------------------------------------------------------------------
            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"valid_loss {eval_dict_valid['avg_loss']:.4f} | valid_acc {eval_dict_valid['avg_acc']:.4f} | "
                f"training time {train_end:.2f} | inference time {infer_end:.2f} |"
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )
            '''
            # --------------------------------------------------------------------
            # Save the best model based on the validation accuracy
            # --------------------------------------------------------------------
            validation_losses = [avg_loss_dict[vkey] for vkey in args.validation_keys]
            for i, (vkey, vminimize) in enumerate(zip(args.validation_keys, args.validation_keys_minimize)):
                if vminimize:
                    store_as_best[i] = validation_losses[i] < best_validation_losses[i]
                else:
                    store_as_best[i] = validation_losses[i] > best_validation_losses[i]
                if store_as_best[i]:
                    best_validation_losses[i] = validation_losses[i]
            # ----------------------------------------------------------------
            # Store checkpoint
            # ----------------------------------------------------------------
            if checkpoint_saver is not None:
                checkpoint_saver.save_latest(
                    directory=args.save,
                    model_and_loss=model_and_loss,
                    stats_dict=dict(avg_loss_dict, epoch=epoch),
                    store_as_best=store_as_best,
                    store_prefixes=args.validation_keys)
            '''
            # --------------------------------------------------------------------
            # they report best eval accuracy and not the last one! 
            # --------------------------------------------------------------------
            best_acc = max(best_acc, eval_dict["avg_acc"])
            #best_acc = max(best_acc, eval_dict_valid["avg_acc"])
            # --------------------------------------------------------------------
            # --------------------------------------------------------------------
            # Early stopping
            # --------------------------------------------------------------------
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model

            #if cur_iter >=2: # for the first two tasks we train for longer 
            if self.early_stopping is True:
                early_stopping(eval_dict_valid["avg_loss"], self.model)
            
                if early_stopping.early_stop:
                    print(f"Early stopping for task_{cur_iter} on epoch {epoch+1}")
                    break
            
        return best_acc, eval_dict

    def update_model(self, x, y, criterion, optimizer):
        # chekc the label type, output of the bayesian model
        
        optimizer.zero_grad()
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            '''
            x = x.double()
            labels_a = labels_a.double()
            labels_b = labels_b.double()
            '''
            # take care of the output of the bayesian model and its probabilistic loss
            if self.bayesian:
                #self.model.double()
                logit_dict = self.model(x)

                loss = lam * criterion(logit_dict, labels_a)['total_loss'] + (1 - lam) * criterion(
                    logit_dict, labels_b)['total_loss']
                #loss = losses_dict['total_loss']
                logit = criterion(logit_dict, labels_a)['prediction']
                logit = logit.mean(dim=2)
            else:
                #self.model.double()
                logit = self.model(x)
                loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                    logit, labels_b
                )
        else:
            
            if self.bayesian:
                # measure forward pass time
                #t_start = time.time()
                #self.model.double()
                logit_dict = self.model(x)
                #t_end = time.time() - t_start
                # logger.info(f'forward pass time: {t_end:.2f} s')

                # criterion is the probabilistic loss class
                #t_s = time.time()
                losses_dict = criterion(logit_dict, y)
                #t_e = time.time() - t_s
                #logger.info(f'loss time: {t_e:.2f} s')
                
                loss = losses_dict['total_loss']
                logit = losses_dict['prediction'] # Shape: torch.Size([10, 10, 64]) --> (batch_size, num_classes, samples)
                # change the shape of the logit to be (batch_size, num_classes)
                logit = logit.mean(dim=2)
            else:
                #self.model.double()
                logit = self.model(x)
                loss = criterion(logit, y)
        
        # calculate the number of correct predictions per batch for the bayesian model as well here
        _, preds = logit.topk(self.topk, 1, True, True)

        loss.backward()
        ''' ToDo: is it necessary to clip the gradient? it was done in mnvi code
        Maybe they didn't need it but I'm not sure. For the Bayesian case, it is probably needed.
        '''
        if self.bayesian:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1, norm_type='inf')
        
        optimizer.step()
        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)

    def _train(
        self, train_loader, memory_loader, optimizer, criterion
    ):
        
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        self.model.train()
        if memory_loader is not None and train_loader is not None:
            data_iterator = zip(train_loader, cycle(memory_loader))
        elif memory_loader is not None:
            data_iterator = memory_loader
        elif train_loader is not None:
            data_iterator = train_loader
        else:
            raise NotImplementedError("None of dataloder is valid")
        
        for i, data in enumerate(tqdm(data_iterator)):
            if len(data) == 2:
                stream_data, mem_data = data
                x = torch.cat([stream_data["image"], mem_data["image"]])
                y = torch.cat([stream_data["label"], mem_data["label"]])
            else:
                x = data["image"]
                y = data["label"]
            # set to double
            #x = x.double().to(self.device)
            #y = y.double().to(self.device)

            x = x.to(self.device)
            y = y.to(self.device)

            
            # ------------------------------------------------------
            # this is equivalent to the step code in the test repo
            l, c, d = self.update_model(x, y, criterion, optimizer)
            # Compute the moving averages - equivalent to MovingAverage in the test repo
            total_loss += l
            correct += c
            num_data += d

        if train_loader is not None:
            n_batches = len(train_loader)
        else:
            n_batches = len(memory_loader)

        return total_loss / n_batches, correct / num_data

    def allocate_batch_size(self, n_old_class, n_new_class):
        new_batch_size = int(
            self.batch_size * n_new_class / (n_old_class + n_new_class)
        )
        old_batch_size = self.batch_size - new_batch_size
        return new_batch_size, old_batch_size
