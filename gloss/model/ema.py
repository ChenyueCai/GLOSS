# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Iterator, Optional, Tuple

import torch


class EMA():
    """Exponential Moving Average of parameters.
    
    Computes an exponential moving average of a given pytorch module's
    parameters:

    p_t = mu * p_{t-1} + (1-mu) * p
    """
    def __init__(self, mu: float, grad_only: bool = True, device: torch.device = "cpu"):
        """Constructor for EMA.

        Args:
            mu (float): The EMA weight.
            grad_only (bool, optional): Whether only parameters
                with gradients should be averaged. Defaults to True.
        """
        self.mu = mu
        self.grad_only = grad_only
        self.avg = {}
        self.stored_params = {}
        self.device = device

    def update(self, model: torch.nn.Module):
        """Update the internal EMA average with the current model.

        This method will also initialize the average for any
        parameters not currently in state.

        Args:
            model (torch.nn.Module): Model with parameter EMA being tracked.
        """
        with torch.no_grad():
            for name, p in model.named_parameters():
                if self.grad_only and not p.requires_grad:
                    continue
                if name in self.avg:
                    self.avg[name].mul_(self.mu).add_(p.data.to(self.device) * (1.0 - self.mu))
                else:
                    self.avg[name] = p.data.clone().detach().to(self.device)


    def reset(self):
        """Reset the internal average state.
        """
        self.avg = {}
        self.stored_params = {}

    def is_empty(self) -> bool:
        """Returns true if the EMA has an empty state.

        Returns:
            bool: True if the EMA has an empty state.
        """
        return len(self.avg) == 0


    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over the parameter average.

        Returns:
            Iterator[Tuple[str, torch.Tensor]]: Iterator over the named parameters.
        """
        return self.avg.items().__iter__()


    def set_model_to_ema(self, model: torch.nn.Module, store=True):
        """Set the provided model's parameters to the current EMA.

        Args:
            model (torch.nn.Module): Model to replace
            store (bool, optional): Whether the current model parameters
                should be stored for retrieval later. Useful for running
                a model evaluation during training. Defaults to True.
        """
        with torch.no_grad():
            if store:
                self.store(model)
            for name, p in model.named_parameters():
                if self.grad_only and not p.requires_grad:
                    continue
                p.data = self.avg[name].clone().detach()


    def store(self, model: torch.nn.Module):
        """Store the model parameters internally.

        Args:
            model (torch.nn.Module): Model to store.
        """
        for name, p in model.named_parameters():
            if self.grad_only and not p.requires_grad:
                continue
            self.stored_params[name] = p.data.clone()


    def restore(self, model: torch.nn.Module):
        """Restore the model parameters from the internal state.

        Args:
            model (torch.nn.Module): The model to restore from the internal EMA store.

        Raises:
            ValueError: One of the model's parameters was not found in the store.
        """
        for name, p in model.named_parameters():
            if self.grad_only and not p.requires_grad:
                continue
            if name in self.stored_params:
                p.data.copy_(self.stored_params[name])
            else:
                raise ValueError(f"Model's parameter {name} not stored within EMA store.")
        self.stored_params = {}


    def state_dict(self) -> Dict[str, Any]:
        """The state dictionary of the EMA.

        Has layout:
        state = {
            "mu": float,
            "avg": dict[str, torch.Tensor]
        }

        Returns:
            Dict[str, Any]: The state dictionary.
        """
        return {"mu": self.mu, "avg": self.avg}


    def load_state_dict(self, state_dict: Dict[str, Any], device: Optional[torch.device] = None):
        """Load the given state dictionary into the EMA.

        The state dictionary should have layout:
        state = {
            "mu": float,
            "avg": dict[str, torch.Tensor]
        }

        Args:
            state_dict (Dict[str, Any]): The input state dictionary.
            device (Optional[torch.device], optional): The device to load the state dictionary
                onto. Defaults to None.
        """
        if device is not None:
            self.device = device
        self.mu = state_dict['mu']
        self.avg = {
            k: p.to(self.device) for k,p in state_dict['avg'].items()
        }
