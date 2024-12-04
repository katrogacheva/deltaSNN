from .neurons import LIF
import torch
from torch import nn


class deltaLeaky(LIF):
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
        reset_mechanism="zero",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            beta=beta,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            inhibition=inhibition,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self._init_mem()
        self._init_prevmem()

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
        
    def _init_prevmem(self):
        prevmem = torch.zeros(0)
        self.register_buffer("prevmem", prevmem, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem
    
    def reset_prevmem(self):
        self.prevmem = torch.zeros_like(self.prevmem, device=self.prevmem.device)
        return self.prevmem

    def init_leaky(self):
        """Deprecated, use :class:`Leaky.reset_mem` instead"""
        return self.reset_mem(), self.reset_prevmem()

    def forward(self, input_, mem=None, prevmem=None):

        if not mem == None:
            self.mem = mem
            
        if not prevmem == None:
            self.prevmem = prevmem

        if self.init_hidden and not mem == None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)

        self.reset = self.mem_reset(self.mem)
        self.mem = self.state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.inhibition:
            spk = self.fire_inhibition(
                self.mem.size(0), self.mem
            )  # batch_size
        else:
            spk = self.delta_fire(self.mem, self.prevmem, self.delta_threshold)

        if not self.reset_delay:
            do_reset = (
                spk / self.graded_spikes_factor - self.reset
            )  # avoid double reset
            if self.reset_mechanism_val == 0:  # reset by subtraction
                self.mem = self.mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:  # reset to zero
                self.mem = self.mem - do_reset * self.mem

        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem

    def _base_state_function(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.threshold

    def _base_zero(self, input_):
        self.prevmem = self.mem
        self.mem = (1 - self.reset) * self.mem
        return self._base_state_function(input_)
    
    def _base_int(self, input_):
        return self._base_state_function(input_)

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], deltaLeaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], deltaLeaky):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )



# def forward(self, input_, mem=None):
#     # Initialize memory and prev_state if not set
#     if self.prev_state is None:
#         self.prev_state = torch.zeros_like(input_, device=input_.device)
#     if self.mem.shape != input_.shape:
#         self.mem = torch.zeros_like(input_, device=input_.device)

#     # Update state with leaky dynamics
#     curr_state = self.beta.clamp(0, 1) * self.mem + input_

#     # Fire mechanism based on delta_fire
#     spike = self.delta_fire(curr_state, self.prev_state)

#     # Suppress state on spike
#     state = curr_state * (~spike)

#     # Update prev_state and mem
#     self.prev_state = curr_state
#     self.mem = state
