import torch

from torch import Tensor
from torch.nn import Module
from typing import Literal

class GlobalMassConservationLoss(Module):
    '''
    Implements global mass conservation loss. Behavior changes depending on mode (train/test).
    During training, we take the absolute value of the loss to convert it to a convex function.
    During testing, we return the original signed values.
    '''
    def __init__(self,
                 mode: Literal['train', 'test'],
                 delta_t: int = 30):
        super(GlobalMassConservationLoss, self).__init__()
        self.mode = mode
        self.delta_t = delta_t

    def forward(self,
                batch_node_pred: Tensor, # Actual predicted water volume (t+1)
                batch_node_input: Tensor, # Actual given water volume (t)
                total_inflow: Tensor, # Actual total inflow (not normalized), from global_mass_info
                total_outflow: Tensor, # Actual total outflow (not normalized), from global_mass_info
                total_rainfall: Tensor, # Actual total rainfall (not normalized), from global_mass_info
        ) -> Tensor:
        # Get current total water volume (t)
        total_water_volume = batch_node_input.sum()

        # Get next total water volume (t+1)
        total_next_water_volume = batch_node_pred.sum()

        # Compute Global Mass Conservation
        delta_v = total_next_water_volume - total_water_volume
        rf_volume = total_rainfall
        inflow_volume = total_inflow * self.delta_t
        outflow_volume = total_outflow * self.delta_t

        global_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
        if self.mode == 'train':
            global_volume_error = torch.abs(global_volume_error)

        global_loss = global_volume_error.mean()
        return global_loss
