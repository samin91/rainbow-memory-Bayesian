"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
# When we make a new one, we should inherit the Finetune class.
import time
import logging
import os
import random

import PIL
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn
from randaugment.randaugment import RandAugment
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from utils.data_loader import ImageDataset
from utils.data_loader import cutmix_data
from utils.train_utils import select_model, select_optimizer
from utils.checkpoint_saver import CheckpointSaver
from models import varprop
from models import cub200_mnvi
import pdb



logger = logging.getLogger()
# log = f"tensorboard/Run_{}" ???
#writer = SummaryWriter(f"test/run_{1}")


class ICaRLNet(nn.Module):
    def __init__(self, model, feature_size, n_class):
        super().__init__()
        self.model = model
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_class, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x


class Finetune:
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        self.num_learned_class = 0
        self.num_learning_class = kwargs["n_init_cls"]
        self.n_classes = n_classes
        self.learned_classes = []
        self.class_mean = [None] * n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]

        self.device = device
        self.criterion = criterion
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        self.lr = kwargs["lr"]
        self.feature_size = kwargs["feature_size"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.prev_streamed_list = []
        self.streamed_list = []
        self.test_list = []
        self.memory_list = []
        self.memory_size = kwargs["memory_size"]
        self.mem_manage = kwargs["mem_manage"]
        # Growing Memroy --------------------------------
        self.coreset_size = kwargs["coreset_size"]
        self.expanding_memory = kwargs["expanding_memory"]
        #-------------------------------------------
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "random"
        # Bayeian Model --------------------------------
        self.bayesian = kwargs["bayesian_model"]
        self.kld_weight_atte = kwargs["kld_weight_atte"]
        self.kwargs = kwargs
        # here we create the model instance and pass it to the device
        
        self.model = select_model(self.model_name, self.dataset, kwargs["n_init_cls"], self.kwargs)    
        # ------------------------------------------
        # dtype
        # ------------------------------------------
        '''
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.dtype}")
        '''       
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        self.already_mem_update = False

        self.mode = kwargs["mode"]

        self.uncert_metric = kwargs["uncert_metric"]
        # self.cub200_mnvi = kwargs["cub200_mnview"]

        # running time of the samplers
        self.total_time_bayesian = 0
        self.total_time_montecarlo = 0

        self.early_stopping = kwargs["early_stopping"]


        # cuda events for measuring CUDA time? 
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        #-------------------------------------------

    def set_current_dataset(self, train_datalist, test_datalist):
        
        random.shuffle(train_datalist)
        self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist

    def before_task(self, datalist, cur_iter, init_model=False, init_opt=True, bayesian=False):
        logger.info("Apply before_task")
        self.incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + self.incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )

        if self.mem_manage == "prototype":
            self.model.fc = nn.Linear(self.model.fc.in_features, self.feature_size)
            self.feature_extractor = self.model
            self.model = ICaRLNet(
                self.feature_extractor, self.feature_size, self.num_learning_class
            )

        # Here the head size gets increased - fully connected layer should be reimplemented for the Bayesian model
        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        # To care for the case of decreasing head? I think this should be increasing head
        new_out_features = max(out_features, self.num_learning_class)
        
        if init_model:
            # initialize model parameters in every iteration
            logger.info("Reset model parameters")
            self.model = select_model(self.model_name, self.dataset, new_out_features, self.kwargs)
        else:
            
            if bayesian is True:
                # what happens to the pre-trained weights?
                #default values of the layer: prior_precision=1e0, prior_mean=0.0, mnv_init=-3.0
                
                self.model.fc = varprop.LinearMNCL(in_features, new_out_features, self.kwargs['prior_precision'], self.kwargs['prior_mean'], self.kwargs['mnv_init'])
                cub200_mnvi.finitialize([self.model.fc], small=False)
                #self.model.fc = self.model.fc.to(self.device)
                # does this weight initialization take place automatically? - since the model is defined once before the task training
                #, I do not think so. I prefer to initialize the classifier weights again
                ''' Another way to intialize fc for each task
                small = True
                nn.init.kaiming_normal_(salf.model.fc.weight)
                if small:
                    layer.weight.data.mul_(0.001)
                if self.model.fc.bias is not None:
                    nn.init.constant_(self.model.fc.bias, 0)
                '''

                # Set up kld weight
                if self.kld_weight_atte is True: 
                    if cur_iter==0:
                        self.model._kl_div_weight = 5e-8
                    elif cur_iter==1:
                        self.model._kl_div_weight = 5e-7
                    elif cur_iter==2:
                        self.model._kl_div_weight = 5e-9
                    elif cur_iter==3:
                        self.model._kl_div_weight = 5e-9
                    elif cur_iter==4:
                        self.model._kl_div_weight = 5e-9
                    else:
                        pass
                logger.info(f"kld weight is {self.model._kl_div_weight}")
            else:
                    self.model.fc = nn.Linear(in_features, new_out_features)
                    #self.model.fc = self.model.fc.to(self.device)
        
        '''ToDO: Check if this all the layers of the Bayesian model and 
                all the parameters are considered?
                https://discuss.pytorch.org/t/model-named-parameters-will-lose-some-layer-modules/14588/6
        '''
        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For regularzation methods
        self.model = self.model.to(self.device)

        # gives us the possibility to reinitialize the optimizer and scheduler
        if init_opt:
            logger.info("Reset the optimizer and scheduler states")
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, self.model, self.sched_name
            )

        logger.info(f"Increasing the head of fc {out_features} -> {new_out_features}")

        self.already_mem_update = False

    def after_task(self, cur_iter):
        
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes # classes seen so far
        self.num_learned_class = self.num_learning_class
        self.update_memory(cur_iter)

    def update_memory(self, cur_iter, num_class=None):
        
        if num_class is None:
            if self.expanding_memory is True:
                '''ToDo:
                    1. Check if this is correct for the blurry case in particular
                '''
                num_class = len(self.incoming_classes) # classes in the current task
            else:
                num_class = self.num_learning_class #classes seen so far
        
        if not self.already_mem_update:
            # -------------------------------------------
            # EXPANDING MEMORY
            # -------------------------------------------
            if self.expanding_memory is True:
                logger.info(f"Update the growing memory over {num_class} classes by {self.mem_manage}")
                candidates = self.streamed_list 
                if self.mem_manage == "random":
                        self.memory_list.extend(self.rnd_sampling(candidates, self.coreset_size)) # memory grows, hence using .extend()
                elif self.mem_manage == "uncertainty":
                        if cur_iter == 0:
                            # how does this work for a Bayesian model? the same as normal models
                            self.memory_list.extend(self.equal_class_sampling(candidates, num_class)) # memory grows, hence using .extend()
                        else:
                            self.memory_list.extend(self.uncertainty_sampling(candidates, num_class, cur_iter)) # memory grows, hence using .extend()
            # -------------------------------------------
            # FIXED MEMORY
            # -------------------------------------------
            else:
                logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
                candidates = self.streamed_list + self.memory_list
                if len(candidates) <= self.memory_size:
                    self.memory_list = candidates
                    self.seen = len(candidates)
                    logger.warning("Candidates < Memory size")
                else:
                    if self.mem_manage == "random":
                        self.memory_list = self.rnd_sampling(candidates, self.memory_size)
                    elif self.mem_manage == "reservoir":
                        self.reservoir_sampling(self.streamed_list)
                    elif self.mem_manage == "prototype":
                        self.memory_list = self.mean_feature_sampling(
                            exemplars=self.memory_list,
                            samples=self.streamed_list,
                            num_class=num_class,
                        )
                    elif self.mem_manage == "uncertainty":
                        if cur_iter == 0:
                            # how does this work for a Bayesian model? the same as normal models
                            self.memory_list = self.equal_class_sampling(
                                candidates, num_class
                            )
                        else:
                            self.memory_list = self.uncertainty_sampling(
                                candidates,
                                num_class,
                                cur_iter
                            )
                    else:
                        logger.error("Not implemented memory management")
                        raise NotImplementedError
            # -----------------------------------------------------------
            if self.expanding_memory is False: # this assertion needs to be checked only for the fixed memory case
                assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")

    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        # Loader
        train_loader = None
        test_loader = None
        if train_list is not None and len(train_list) > 0:
            train_dataset = ImageDataset(
                pd.DataFrame(train_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            # drop last becasue of BatchNorm1D in IcarlNet
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
                pin_memory=True,
            )

        if test_list is not None:
            test_dataset = ImageDataset(
                pd.DataFrame(test_list),
                dataset=self.dataset,
                transform=self.test_transform,
            )
            test_loader = DataLoader(
                test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker, pin_memory=True
            )

        return train_loader, test_loader

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=1):

        train_list = self.streamed_list + self.memory_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        for epoch in range(n_epoch):
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()

            train_loss, train_acc = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch=epoch,
                total_epochs=n_epoch,
                n_passes=n_passes,
            )

            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )
            '''
            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )
            '''
            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict

    def _train(
        self, train_loader, optimizer, criterion, epoch, total_epochs, n_passes=1
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            for pass_ in range(n_passes):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                do_cutmix = self.cutmix and np.random.rand(1) < 0.5
                if do_cutmix:
                    x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                    logit = self.model(x)
                    loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                        logit, labels_b
                    )
                else:
                    logit = self.model(x)
                    loss = criterion(logit, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)

        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data

    def evaluation_ext(self, test_list):
        # evaluation from out of class
        test_dataset = ImageDataset(
            pd.DataFrame(test_list),
            dataset=self.dataset,
            transform=self.test_transform,
        )
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=32, num_workers=2
        )
        eval_dict = self.evaluation(test_loader, self.criterion)

        return eval_dict

    def evaluation(self, test_loader, criterion):
        
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                if self.bayesian:
                    logit_dict = self.model(x)
                    losses_dict = criterion(logit_dict, y)
                    loss = losses_dict['total_loss']
                    logit = losses_dict['prediction'] # Shape: torch.Size([10, 10, 64]) --> (batch_size, num_classes, samples)
                    # change the shape of the logit to be (batch_size, num_classes)
                    logit = logit.mean(dim=2)
                else:
                    logit = self.model(x)
                    loss = criterion(logit, y)

                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def rnd_sampling(self, samples, size):
        random.shuffle(samples)
        return samples[: size]

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            self.seen += 1

    def mean_feature_sampling(self, exemplars, samples, num_class):
        """Prototype sampling

        Args:
            features ([Tensor]): [features corresponding to the samples]
            samples ([Datalist]): [datalist for a class]

        Returns:
            [type]: [Sampled datalist]
        """

        def _reduce_exemplar_sets(exemplars, mem_per_cls):
            if len(exemplars) == 0:
                return exemplars

            exemplar_df = pd.DataFrame(exemplars)
            ret = []
            for y in range(self.num_learned_class):
                cls_df = exemplar_df[exemplar_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )

            num_dups = pd.DataFrame(ret).duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

            return ret
        # this makes sense for the case of a fixed memory
        mem_per_cls = self.memory_size // num_class
        exemplars = _reduce_exemplar_sets(exemplars, mem_per_cls)
        old_exemplar_df = pd.DataFrame(exemplars)

        new_exemplar_set = []
        sample_df = pd.DataFrame(samples)
        for y in range(self.num_learning_class):
            cls_samples = []
            cls_exemplars = []
            if len(sample_df) != 0:
                cls_samples = sample_df[sample_df["label"] == y].to_dict(
                    orient="records"
                )
            if len(old_exemplar_df) != 0:
                cls_exemplars = old_exemplar_df[old_exemplar_df["label"] == y].to_dict(
                    orient="records"
                )

            if len(cls_exemplars) >= mem_per_cls:
                new_exemplar_set += cls_exemplars
                continue

            # Assign old exemplars to the samples
            cls_samples += cls_exemplars
            if len(cls_samples) <= mem_per_cls:
                new_exemplar_set += cls_samples
                continue

            features = []
            self.feature_extractor.eval()
            with torch.no_grad():
                for data in cls_samples:
                    image = PIL.Image.open(
                        os.path.join("dataset", self.dataset, data["file_name"])
                    ).convert("RGB")
                    x = self.test_transform(image).to(self.device)
                    feature = (
                        self.feature_extractor(x.unsqueeze(0)).detach().cpu().numpy()
                    )
                    feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
                    features.append(feature.squeeze())

            features = np.array(features)
            logger.debug(f"[Prototype] features: {features.shape}")

            # do not replace the existing class mean
            if self.class_mean[y] is None:
                cls_mean = np.mean(features, axis=0)
                cls_mean /= np.linalg.norm(cls_mean)
                self.class_mean[y] = cls_mean
            else:
                cls_mean = self.class_mean[y]
            assert cls_mean.ndim == 1

            phi = features
            mu = cls_mean
            # select exemplars from the scratch
            exemplar_features = []
            num_exemplars = min(mem_per_cls, len(cls_samples))
            for j in range(num_exemplars):
                S = np.sum(exemplar_features, axis=0)
                mu_p = 1.0 / (j + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p, axis=1, keepdims=True)

                dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                i = np.argmin(dist)

                new_exemplar_set.append(cls_samples[i])
                exemplar_features.append(phi[i])

                # Avoid to sample the duplicated one.
                del cls_samples[i]
                phi = np.delete(phi, i, 0)

        return new_exemplar_set

    def uncertainty_sampling(self, samples, num_class, cur_iter):
        
        """uncertainty based sampling
        Args:
            samples ([list]): [training_list + memory_list]
        """
        
        # ---------------------------------------------------------------------
        # Add majority voting based uncertainty values to the samples dictionary
        # ---------------------------------------------------------------------
        
        if self.bayesian:
            '''ToDo: compute time
            '''
            start_time_1 = time.time()
            self._Bayesian(samples)
            end_time_1 = time.time()
            running_time_1 = end_time_1 - start_time_1
            logger.info("Function _Bayesian took {:.2f} seconds to run".format(running_time_1))
            self.total_time_bayesian += running_time_1
            
        else:
            # RM original: ensmebling of 12 passes of augmented images
            start_time_2 = time.time()
            self.montecarlo(samples, uncert_metric=self.uncert_metric)
            end_time_2 = time.time()
            running_time_2 = end_time_2 - start_time_2
            logger.info("Function montecarlo took {:.2f} seconds to run".format(running_time_2))
            self.total_time_montecarlo += running_time_2

        sample_df = pd.DataFrame(samples)
        ret = []
        # ---------------------------------------------------------------------
        # EXPANDING MEMORY
        # ---------------------------------------------------------------------
        if self.expanding_memory:
            mem_per_cls = self.coreset_size // num_class
            '''ToDo: labels are task_0: 0-9, task_1: 10-19 and so on
                This needs to be considerd for the case of expanding memory where we only sample from the current task
            '''
            # define label range
            start = cur_iter*num_class
            End = (cur_iter+1)*num_class
            for i in range(start, End):
                cls_df = sample_df[sample_df["label"] == i] # class data frame
                if len(cls_df) <= mem_per_cls:
                    ret += cls_df.to_dict(orient="records") # converts the dataframe to a list of dictionaries
                else:
                    # RM jumping strategy
                    jump_idx = len(cls_df) // mem_per_cls
                    uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                    ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")
        # ---------------------------------------------------------------------
        # FIXED MEMORY
        # ---------------------------------------------------------------------
        else: 
            mem_per_cls = self.memory_size // num_class
            for i in range(num_class):
                cls_df = sample_df[sample_df["label"] == i] # class data frame
                if len(cls_df) <= mem_per_cls:
                    ret += cls_df.to_dict(orient="records") # converts the dataframe to a list of dictionaries
                else:
                    # RM jumping strategy
                    jump_idx = len(cls_df) // mem_per_cls
                    uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                    ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")
           
            num_rest_slots = self.memory_size - len(ret)
            if num_rest_slots > 0:
                logger.warning("Fill the unused slots by breaking the equilibrium.")
                ret += (
                    sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
                )

            num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        
        batch_size = 32
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            infer_df, dataset=self.dataset, transform=infer_transform
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )

        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"] #torch.Size([32, 3, 224, 224])
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu() #torch.Size([32, num_classes])

                for i, cert_value in enumerate(logit): #cert_value is the logit of the each image: torch.Size([30])
                    # infer list members: 
                    # {'klass': 'White_throated_Sparrow', 
                    # 'file_name': 'train/White_throated_Sparrow/White_Throated_Sparrow_0031_128808.jpg', 
                    # 'label': 29}
                    sample = infer_list[batch_size * n_batch + i] 
                    sample[uncert_name] = 1 - cert_value
                    '''
                    {'klass': 'White_throated_Sparrow', 'file_name': 'train/White_throated_Sparrow/White_Throated_Sparrow_0031_128808.jpg', 'label': 29, 
                    'uncert_0': tensor([ 0.6901,  1.8171,  1.0081,  2.2003,  1.2164,  0.1057,  2.0374,  0.0638,
                    0.1433,  0.5314,  0.7850,  2.5484,  0.5498,  0.8810,  0.2965,  2.0254,
                    2.0250,  1.8225,  1.8064,  2.4466, -0.5500,  0.0042,  1.8341,  2.0047,
                    0.9955,  1.3818, -3.4274,  0.5815,  1.7533, -0.1668]),
                    'uncert_1':
                    , ... }
                    '''

    def montecarlo(self, candidates, uncert_metric="vr"):
        
        transform_cands = []
        logger.info(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "vr":
            transform_cands = [
                Cutout(size=8),
                Cutout(size=16),
                Cutout(size=24),
                Cutout(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomRotation(90),
                Invert(),
                Solarize(v=128),
                Solarize(v=64),
                Solarize(v=32),
            ]
        elif uncert_metric == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        elif uncert_metric == "vr_cutout":
            transform_cands = [Cutout(size=16)] * 12
        elif uncert_metric == "vr_autoaug":
            transform_cands = [select_autoaugment(self.dataset)] * 12

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands): #Random Augment Policy
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)


    def variance_ratio(self, sample, cand_length):
        #pdb.set_trace()
        vote_counter = torch.zeros(sample["uncert_0"].size(0)) # troch.Size([30])
        for i in range(cand_length): #candidate length is 12
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item() # out of 12 predictions per sample, how many times the most voted class was not predicted




    def equal_class_sampling(self, samples, num_class):

        sample_df = pd.DataFrame(samples)
        ret = []
        # -------------------------------------------------
        # EXPANDING MEMORY
        # -------------------------------------------------
        if self.expanding_memory:
            mem_per_cls = self.coreset_size // num_class
            # Warning: assuming the classes were ordered following task number.
            for y in range(num_class):
                cls_df = sample_df[sample_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )
            assert len(ret) == self.coreset_size 
        # -------------------------------------------------
        # FIXED MEMORY
        # -------------------------------------------------
        else:
            mem_per_cls = self.memory_size // num_class
            # Warning: assuming the classes were ordered following task number.
            
            for y in range(self.num_learning_class):
                cls_df = sample_df[sample_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )

            num_rest_slots = self.memory_size - len(ret)
            if num_rest_slots > 0:
                logger.warning("Fill the unused slots by breaking the equilibrium.")
                ret += (
                    sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
                )

            num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def update_prior(self, prior_conv_func):
        """Update the prior of the bayesian model: posterior --> prior.

        Args:
            prior_conv_func (function): function to convert the posterior width.
        """
        
        # check the state dictionary
        self.model.save_prior_and_weights(prior_conv_func)
        # check the state dictioanry
        self.model.update_prior_and_weights_from_saved()
        # check the state dictionary

    def checkpoint_saver_loader(self, ):
        #pdb.set_trace()
        """Manage the checkpoint of the model."""
        #pdb.set_trace()
        checkpoint_saver = CheckpointSaver()
        checkpoint_stats = None

        if self.kwargs["checkpoint_path"] is None:
            logger.info("No checkpoint given.")
            logger.info("Starting from scratch with random initialization.")

        elif os.path.isfile(self.kwargs['checkpoint_path']):
            checkpoint_stats, filename = checkpoint_saver.restore(
                filename=self.kwargs['checkpoint_path'],
                model=self.model,
                include_params=self.kwargs['checkpoint_include_params'],
                exclude_params=self.kwargs['checkpoint_exclude_params'])

        elif os.path.isdir(self.kwargs['checkpoint_path']):
            if self.kwargs.checkpoint_mode in ["resume_from_best"]:
                logger.info("Loading best checkpoint in %s" % self.kwargs['checkpoint_path'])
                checkpoint_stats, filename = checkpoint_saver.restore_best(
                    directory=self.kwargs['checkpoint_path'],
                    model=self.model,
                    include_params=self.kwargs['checkpoint_include_params'],
                    exclude_params=self.kwargs['checkpoint_exclude_params'])

            elif self.kwargs.checkpoint_mode in ["resume_from_latest"]:
                logger.info("Loading latest checkpoint in %s" % self.kwargs['checkpoint_path'])
                checkpoint_stats, filename = checkpoint_saver.restore_latest(
                    directory=self.kwargs['checkpoint_path'],
                    model=self.model,
                    include_params=self.kwargs['checkpoint_include_params'],
                    exclude_params=self.kwargs['checkpoint_exclude_params'])
            else:
                #logger.info("Unknown checkpoint_restore '%s' given!" % self.kwargs['checkpoint_restore'])
                quit()
        else:
            logger.info("Could not find checkpoint file or directory '%s'" % self.kwargs['checkpoint_path'])
            quit()

        return checkpoint_saver, checkpoint_stats, filename
    
    def save_checkpoint(self, directory, stats_dict, store_as_best=False, store_prefixes="total_loss"):
        
        checkpoint_saver = CheckpointSaver()
        checkpoint_saver.save_latest(directory, self.model, stats_dict, store_as_best=False, store_prefixes="total_loss")
    
    # Bayesian uncertainty 
    def _Bayesian(self, infer_list):
        """uncertainty per sample computation for a bayesian model.
        Args:
            samples ([list]): [training_list]
        """
       
        batch_size=32
        infer_transform = transforms.Compose(self.test_transform.transforms)
        # Consider the bayesian case first
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            infer_df, dataset=self.dataset, transform=infer_transform
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )
        # ----------------------------------------
        # INFERENCE
        # ----------------------------------------
        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"] #torch.Size([32, 3, 224, 224])
                x = x.to(self.device)
                logit_dict = self.model(x)
                #logit_dict = logit_dict.detach().cpu() #torch.Size([32, num_classes])
                # -----------------------------------------------------------------------------------
                # Detaching tensors and moving them to CPU - I am not sure if this is necessary here!
                # ------------------------------------------------------------------------------------
                logit_dict_prediction_mean = logit_dict['prediction_mean'].detach().cpu()
                logit_dict_prediction_variance = logit_dict['prediction_variance'].detach().cpu()
                # Samples from the output distribution
                samples = 64
                prediction_mean = logit_dict_prediction_mean.unsqueeze(dim=2).expand(-1, -1, samples)
                prediction_variance = logit_dict_prediction_variance.unsqueeze(dim=2).expand(-1, -1, samples)
                normal_dist = torch.distributions.normal.Normal(torch.zeros_like(prediction_mean), 
                                                        torch.ones_like(prediction_mean))
                normals =  normal_dist.sample()
                logit = prediction_mean + torch.sqrt(prediction_variance) * normals #torch.Size([batch_size, num_classes, samples])
                for i, cert_value in enumerate(logit): #cert_value is the logit of the each image: torch.Size([num_classes, samples])
                    sample = infer_list[batch_size * n_batch + i] 
                    sample['uncertainties'] = 1 - cert_value #cert_value[:,0] for indexing cert_value so we see the individual outputs 
                
        # Do the majority voting per sample predictions 
        for sample in infer_list:
            self.variance_ratio_bayesian(sample, sample['uncertainties'].size(1))
        
    def variance_ratio_bayesian(self, sample, cand_length=64):
        
        vote_counter = torch.zeros(sample["uncertainties"].size(0)) # troch.Size([30])
        for i in range(cand_length): #candidate length is 64
            top_class = int(torch.argmin(sample['uncertainties'][:,i]))  # uncert argmin.
            vote_counter[top_class] += 1
        
        assert vote_counter.sum() == cand_length #number of samples per image=64
        
        # Does it make sense to use majority votig for a bayesian model?
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item() # out of 12 predictions per sample, how many times the most voted class was not predicted
        

    def measure_time(self, model, input):
            # Record the start time
            self.start_event.record()

            # Forward pass
            output = model(input)

            # Record the end time
            self.end_event.record()

            # Wait for the events to complete
            torch.cuda.synchronize()

            # Compute the elapsed time
            elapsed_time = self.start_event.elapsed_time(self.end_event)

            return elapsed_time, output