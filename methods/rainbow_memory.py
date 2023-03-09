"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import time
import logging
import random

import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset
import pdb

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
        self.exp_env = kwargs["stream_env"]
        self.bayesian = kwargs["bayesian_model"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"

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
            )
            stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            stream_batch_size = batch_size

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list
        random.shuffle(train_list)
        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)+len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        early_stopping = EarlyStopping(patience=10, verbose=True)

        '''ToDo: should we also put the loss function on the device? hmmmm
        '''
        
        self.model = self.model.to(self.device)
        
        for epoch in range(n_epoch):
            train_start=0
            infer_start=0
            train_end=0
            infer_end=0
            # initialize for each task
            # optimizer.param_groups is a python list, which contains a dictionary.
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    # param_group is the dict inside the list and is the only item in this list.
                    if self.bayesian is True:
                        param_group["lr"] = self.lr
                    else:
                        param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                if self.scheduler is not None:
                    self.scheduler.step()

            train_start = time.time()
            train_loss, train_acc = self._train(train_loader=train_loader, memory_loader=memory_loader,
                                                optimizer=self.optimizer, criterion=self.criterion)
            train_end = time.time() - train_start

            # EVAL - testing over all the test sets seen so far
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
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar('Accuracy/valid-',eval_dict["avg_acc"], epoch)
            writer.add_scalar('Loss/valid-', eval_dict["avg_loss"] , epoch)
            # -------------------------------------------------------------------
            # Logging to console
            # -------------------------------------------------------------------
            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | training time {train_end:.2f} | inference time {infer_end:.2f} |"
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )
            # --------------------------------------------------------------------
            # they report best eval accuracy and not the last one! 
            # --------------------------------------------------------------------
            best_acc = max(best_acc, eval_dict["avg_acc"])

            # --------------------------------------------------------------------
            # Early stopping
            # --------------------------------------------------------------------
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(eval_dict["avg_loss"], self.model)
        
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
            # take care of the output of the bayesian model and its probabilistic loss
            if self.bayesian:
                logit_dict = self.model(x)

                loss = lam * criterion(logit_dict, labels_a)['total_loss'] + (1 - lam) * criterion(
                    logit_dict, labels_b)['total_loss']
                #loss = losses_dict['total_loss']
                logit = criterion(logit_dict, labels_a)['prediction']
                logit = logit.mean(dim=2)
            else:
                logit = self.model(x)
                loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                    logit, labels_b
                )
        else:
            
            if self.bayesian:
                # measure forward pass time
                #t_start = time.time() 
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

        for data in data_iterator:
            if len(data) == 2:
                stream_data, mem_data = data
                x = torch.cat([stream_data["image"], mem_data["image"]])
                y = torch.cat([stream_data["label"], mem_data["label"]])
            else:
                x = data["image"]
                y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)
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
