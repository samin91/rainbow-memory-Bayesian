import os
import random
import shutil
import re
import logging
import numpy as np
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
from torch import nn
from utils import strings

import pdb

logger = logging.getLogger()
# --------------------------------------------------------------------------
# Checkpoint loader/saver.
# --------------------------------------------------------------------------
class CheckpointSaver:
    def __init__(self,
                 prefix="checkpoint",
                 latest_postfix="_latest_",
                 best_postfix="_best_",
                 model_key="state_dict",
                 extension=".ckpt"):

        self._prefix = prefix
        self._model_key = model_key
        self._latest_postfix = latest_postfix
        self._best_postfix = best_postfix
        self._extension = extension

    # the purpose of rewriting the loading function is we sometimes want to
    # initialize parameters in modules without knowing the dimensions at runtime
    #
    # This function here will resize these parameters to whatever size required.
    #
    @staticmethod
    def _load_state_dict_into_module(state_dict, module, strict=True):
        
        own_state = module.state_dict()
        for name, param in state_dict.items():
            name = name.split(".", maxsplit=3)[-1]  # added by mar-ret
            if name in own_state:
                #print("Loading {} into module".format(name))  # added by mar-ret
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].resize_as_(param)
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            else:
                print("{} not in module".format(name))  # added by mar-ret
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def restore(self, filename, model, include_params="*", exclude_params=()):
        
        # -----------------------------------------------------------------------------------------
        # Make sure file exists
        # -----------------------------------------------------------------------------------------
        if not os.path.isfile(filename):
            logger.info("Could not find checkpoint file '%s'!" % filename)
            quit()

        # -----------------------------------------------------------------------------------------
        # Load checkpoint from file including the state_dict
        # -----------------------------------------------------------------------------------------
        checkpoint_with_state = torch.load(filename, map_location="cpu")

        # -----------------------------------------------------------------------------------------
        # Load filtered state dictionary
        # -----------------------------------------------------------------------------------------
        state_dict = checkpoint_with_state[self._model_key]
        restore_keys = strings.filter_list_of_strings(
            state_dict.keys(),
            include=include_params,
            exclude=exclude_params)
        # Exclude the keys (fc)
        state_dict = {key: value for key, value in state_dict.items() if key in restore_keys}

        # if parameter lists are given, don't be strict with loading from checkpoints
        strict = True
        if include_params != "*" or len(exclude_params) != 0:
            strict = False
        
        print("strict = ", strict)  # added by mar-ret

        self._load_state_dict_into_module(state_dict, model, strict=strict)
        #logging.info("  Restore keys:")
        #for key in restore_keys:
        #    logging.info("    %s" % key)

        # -----------------------------------------------------------------------------------------
        # Get checkpoint statistics without the state dict
        # -----------------------------------------------------------------------------------------
        checkpoint_stats = {
            key: value for key, value in checkpoint_with_state.items() if key != self._model_key
        }

        return checkpoint_stats, filename

    def restore_latest(self, directory, model_and_loss, include_params="*", exclude_params=()):
        latest_checkpoint_filename = os.path.join(
            directory, self._prefix + self._latest_postfix + self._extension)
        return self.restore(latest_checkpoint_filename, model_and_loss, include_params, exclude_params)

    def restore_best(self, directory, model_and_loss, include_params="*", exclude_params=()):
        best_checkpoint_filename = os.path.join(
            directory, self._prefix + self._best_postfix + self._extension)
        return self.restore(best_checkpoint_filename, model_and_loss, include_params, exclude_params)

    def save_latest(self, directory, model_and_loss, stats_dict,
                    store_as_best=False, store_prefixes="total_loss"):

        # -----------------------------------------------------------------------------------------
        # Mutable default args..
        # -----------------------------------------------------------------------------------------
        store_as_best = list(store_as_best)

        # -----------------------------------------------------------------------------------------
        # Make sure directory exists
        # -----------------------------------------------------------------------------------------
        system.ensure_dir(directory)

        # -----------------------------------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------------------------------
        save_dict = dict(stats_dict)
        save_dict[self._model_key] = model_and_loss.state_dict()

        latest_checkpoint_filename = os.path.join(
            directory, self._prefix + self._latest_postfix + self._extension)

        latest_statistics_filename = os.path.join(
            directory, self._prefix + self._latest_postfix + ".json")

        torch.save(save_dict, latest_checkpoint_filename)
        json.write_dictionary_to_file(stats_dict, filename=latest_statistics_filename)

        # -----------------------------------------------------------------------------------------
        # Possibly store as best
        # -----------------------------------------------------------------------------------------
        for store, prefix in zip(store_as_best, store_prefixes):
            if store:
                best_checkpoint_filename = os.path.join(
                    directory, self._prefix + self._best_postfix + prefix + self._extension)

                best_statistics_filename = os.path.join(
                    directory, self._prefix + self._best_postfix + prefix + ".json")
                
                if len(best_checkpoint_filename.rsplit("/", 1)) > 1:
                    shortfile = best_checkpoint_filename.rsplit("/", 1)[1]
                    shortpath = os.path.dirname(best_checkpoint_filename).rsplit("/", 1)[1]
                    shortname = os.path.join(shortpath, shortfile)
                    logger.info("Save ckpt to ../%s" % shortname)
                    shutil.copyfile(latest_checkpoint_filename, best_checkpoint_filename)
                    shutil.copyfile(latest_statistics_filename, best_statistics_filename)
                else:
                    shortfile = best_checkpoint_filename.rsplit("\\", 1)[1]
                    shortpath = os.path.dirname(best_checkpoint_filename).rsplit("\\", 1)[1]
                    shortname = os.path.join(shortpath, shortfile)
                    logger.info("Save ckpt to ..\\%s" % shortname)
                    shutil.copyfile(latest_checkpoint_filename, best_checkpoint_filename)
                    shutil.copyfile(latest_statistics_filename, best_statistics_filename)