import numpy as np
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
#(True, warn_only=True)
from torch import nn
from .probabilistic_classification_losses import ClassificationLossVI


class Loss(nn.Module):
    def __init__(self, loss):
        super(Loss, self).__init__()
        self.add_module("_loss", loss)
        
    @property
    def loss(self):
        return self._loss

    # -------------------------------------------------------------
    # Note: We merge inputs and targets into a single dictionary !
    # -------------------------------------------------------------
    def forward(self, output_dict, target_dict):
        # -------------------------------------
        # Compute losses
        # -------------------------------------
        loss_dict = self._loss(output_dict, target_dict)
        # -------------------------------------
        # Return losses and outputs
        # -------------------------------------
        return loss_dict


def configure_model_and_loss():
    # ----------------------------------------------------
    # Create loss object
    # ----------------------------------------------------
    loss = ClassificationLossVI()
    # ----------------------------------------------------
    # loss
    # ----------------------------------------------------
    _loss = Loss(loss)
    return _loss