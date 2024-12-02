from .neurons import LIF
import torch
from torch import nn


class deltaLeaky(LIF):
    """
    First-order leaky integrate-and-fire neuron model with delta RNN capabilities.
    This model combines the standard leaky integration with a delta update mechanism.
    The delta mechanism captures changes in the input or state to dynamically adjust the membrane potential.
    """

    def __init__(
        self,
        beta,
        delta_threshold=10.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            beta,
            delta_threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self.prev_state = None
        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.reset_delay = reset_delay

    def _init_mem(self):
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

    def reset_prev_state(self):
        """Resets prev_state to None."""
        self.prev_state = None

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def init_leaky(self):
        """Deprecated, use :class:`Leaky.reset_mem` instead"""
        return self.reset_mem()

def forward(self, input_, mem=None):
    # Initialize memory and prev_state if not set
    if self.prev_state is None:
        self.prev_state = torch.zeros_like(input_, device=input_.device)
    if self.mem.shape != input_.shape:
        self.mem = torch.zeros_like(input_, device=input_.device)

    # Update state with leaky dynamics
    curr_state = self.beta.clamp(0, 1) * self.mem + input_

    # Fire mechanism based on delta_fire
    spike = self.delta_fire(curr_state, self.prev_state)

    # Suppress state on spike
    state = curr_state * (~spike)

    # Update prev_state and mem
    self.prev_state = curr_state
    self.mem = state
