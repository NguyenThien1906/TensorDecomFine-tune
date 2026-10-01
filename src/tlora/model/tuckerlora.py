import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy.ntheory
import json

import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.decomposition import tucker

from tensorly.random import random_tt
from tensorly.tt_tensor import tt_to_tensor

from .lora import processors, linear_layers, parameter_counter

tl.set_backend('pytorch')

def prime_factors(x: int):
  div = 2
  factors = []
  while div * div <= x:
    while x % div == 0:
      factors = factors + [div]
      x /= div
    div = int(sympy.ntheory.nextprime(div))

  if x > 1:
    factors = factors + [x]

  return factors

def get_integer_factors(x: int, num_factors: int):
  num = x
  factors_list = None # Renamed variable to avoid confusion

  factors_list = prime_factors(num)
  factors_list.sort()

  while len(factors_list) > num_factors:
    # simple balancing by multiplying the two smallest numbers
    # not completely optimal
    factors_list[0] *= factors_list[1]
    factors_list.pop(1)
    factors_list.sort()

  if len(factors_list) < num_factors:
    factors_list = [1] * (num_factors - len(factors_list)) + factors_list

  return [int(i) for i in factors_list]

def tucker_ranks(M: int, N: int, D: int, ratio: float):
  M_ranks = get_integer_factors(M, D)
  N_ranks = get_integer_factors(N, D)
  I_ranks = [M_ranks[i] * N_ranks[i] for i in range(D)]
  R_ranks = [int(ratio * I) if int(ratio * I) > 0 else 1 for I in I_ranks]
  return R_ranks, I_ranks

def total_tucker_parameters(R_ranks, I_ranks):
  U = sum([r * i for r, i in zip(R_ranks, I_ranks)])
  prod = 1
  for r in R_ranks:
    prod *= r
  R = prod
  return U + R

def list_prod(lst: list):
  prod = 1
  for i in lst:
    prod *= i
  return prod

@linear_layers.add_to_registry("tucker_lora")
class TuckerLoRALinearLayer(nn.Module):
    def __init__(self, in_f:int, out_f:int, mode:int, config_json:str):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.mode = mode
        self.config_path = config_json

        # get configuration
        self.config = None
        self.import_config()
        if self.config is None:
            raise ValueError("Failed to import Tucker config. Please check the config JSON file and path.")
        self.in_ranks = self.config["in_ranks"]
        self.out_ranks = self.config["out_ranks"]
        self.R_ranks = self.config["r_ranks"]
        self.I_ranks = [self.in_ranks[i] * self.out_ranks[i] for i in range(self.mode)]

        # init
        core = torch.zeros(self.R_ranks, requires_grad=True) #why is requires_grad=True here instead of assigning to 
        factors = [torch.randn(self.I_ranks[i], self.R_ranks[i], requires_grad=True) / self.R_ranks[i] for i in range(self.mode)]

        # sign up for torch.nn
        core = core.contiguous()
        for i in range(len(factors)):
            factors[i] = factors[i].contiguous()

        self.core = nn.Parameter(core)
        self.factors = nn.ParameterList([nn.Parameter(fa) for fa in factors])

        parameter_counter.add(core.numel() + sum([fa.numel() for fa in factors]))

    def import_config(self):
        # get config from json, to get ranks for specific layer sizes
        if self.config_path is None:
            raise ValueError("Config JSON is required for TuckerLoRALinearLayer to determine ranks.")
        with open(self.config_path, 'r') as f:
            linear_sizes = str(self.in_f)+"x"+str(self.out_f)
            self.config = json.load(f)
            if str(self.mode) not in self.config or linear_sizes not in self.config[str(self.mode)]:
                raise ValueError(f"Config JSON must contain ranks for mode {self.mode} and linear size {linear_sizes}.")
            self.config = self.config[str(self.mode)][linear_sizes]

        # check validity
        # keys must have "in_ranks", "out_ranks", "r_ranks"
        if not all(key in self.config for key in ["in_ranks", "out_ranks", "r_ranks"]):
            raise ValueError("Config JSON must contain 'in_ranks', 'out_ranks', and 'r_ranks' for TuckerLoRALinearLayer.")
        
        # all must be lists of integers
        if not all(isinstance(self.config[key], list) and all(isinstance(i, int) for i in self.config[key]) for key in ["in_ranks", "out_ranks", "r_ranks"]):
            raise ValueError("Config JSON values for 'in_ranks', 'out_ranks', and 'r_ranks' must be lists of integers.")

        # must have same length
        if not all(len(self.config[key]) == self.mode for key in ["in_ranks", "out_ranks", "r_ranks"]):
            raise ValueError("Config JSON lists for 'in_ranks', 'out_ranks', and 'r_ranks' must have the same length.")
        
        # check that ranks are valid for the given in_f and out_f
        if self.in_f != list_prod(self.config["in_ranks"]):
            raise ValueError(f"Product of 'in_ranks'({self.config['in_ranks']}) must equal in_f ({in_f}).")
        if self.out_f != list_prod(self.config["out_ranks"]):
            raise ValueError(f"Product of 'out_ranks'({self.config['out_ranks']}) must equal out_f ({out_f}).")

    def forward(self, input):
        #1. Chuyển đổi kiểu dữ liệu cho đồng bộ lớp
        input_dtype = input.dtype

        #2. Tính toán đầu ra
        tensor = tucker_to_tensor((self.core, self.factors)).reshape(self.out_f, self.in_f)
        # check nan
        if torch.isnan(tensor).any():
          raise ValueError("NaN detected in the reconstructed tensor. Please check the core and factor matrices for stability issues.")
        A_output = F.linear(input, tensor)

        #3. Chuyển đổi lại kiểu dữ liệu ban đầu
        A_output = A_output.to(input_dtype)

        return A_output

@processors.add_to_registry("tucker_lora")
class TuckerLoRACrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, mode, config_json, cross_attention_dim=None, 
                 linear_layer=TuckerLoRALinearLayer): #this must match trainer_sdxl.py/line 236
        super().__init__()
        self.W_Q = linear_layer(hidden_size, hidden_size, mode=mode, config_json=config_json)
        self.W_K = linear_layer(cross_attention_dim or hidden_size, hidden_size, mode=mode, config_json=config_json)
        self.W_V = linear_layer(cross_attention_dim or hidden_size, hidden_size, mode=mode, config_json=config_json)
        self.W_out = linear_layer(hidden_size, hidden_size, mode=mode, config_json=config_json)

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None
    ): # function này không đổi
        #1. Tính Q, K, V với LoRA
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        Q = attn.to_q(hidden_states) + self.W_Q(hidden_states)
        K = attn.to_k(encoder_hidden_states) + self.W_K(encoder_hidden_states)
        V = attn.to_v(encoder_hidden_states) + self.W_V(encoder_hidden_states)

        #2a. Uh... trick lỏ lấy từ phía trên
        Q = attn.head_to_batch_dim(Q)
        K = attn.head_to_batch_dim(K)
        V = attn.head_to_batch_dim(V)

        #2. Tính attention scores và output
        batch_size, seq_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_length, batch_size)
        attention_probs = attn.get_attention_scores(Q, K, attention_mask)
        hidden_states = torch.bmm(attention_probs, V)

        #3. Tính dropout
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) + self.W_out(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


@linear_layers.add_to_registry("tensor_train")
class TensorTrainLinearLayer(nn.Module):
    def __init__(self, in_f:int, out_f:int, rank:int):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.rank = rank

        factors = random_tt((in_f, out_f), rank=rank)

        self.factors = nn.ParameterList([nn.Parameter(fa / (self.rank**2)) for fa in factors])

        parameter_counter.add(sum([fa.numel() for fa in factors]))

    def forward(self, input):
      input_dtype = input.dtype

      tensor = tt_to_tensor(self.factors).reshape(self.out_f, self.in_f)
      A_output = F.linear(input, tensor.float())

      if torch.isnan(tensor).any():
        raise ValueError("NaN detected in the reconstructed tensor. Please check the factor matrices for stability issues.")

      return A_output.to(input_dtype)

@processors.add_to_registry("tensor_train")
class TensorTrainCrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, rank, cross_attention_dim=None, 
                 tt_linear_layer=TensorTrainLinearLayer): #this must match trainer_sdxl.py/line 236
        super().__init__()
        self.W_Q = tt_linear_layer(hidden_size, hidden_size, rank=rank)
        self.W_K = tt_linear_layer(cross_attention_dim or hidden_size, hidden_size, rank=rank)
        self.W_V = tt_linear_layer(cross_attention_dim or hidden_size, hidden_size, rank=rank)
        self.W_out = tt_linear_layer(hidden_size, hidden_size, rank=rank)

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None
    ): # function này không đổi
        #1. Tính Q, K, V với LoRA
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        Q = attn.to_q(hidden_states) + self.W_Q(hidden_states)
        K = attn.to_k(encoder_hidden_states) + self.W_K(encoder_hidden_states)
        V = attn.to_v(encoder_hidden_states) + self.W_V(encoder_hidden_states)

        #2a. Uh... trick lỏ lấy từ phía trên
        Q = attn.head_to_batch_dim(Q)
        K = attn.head_to_batch_dim(K)
        V = attn.head_to_batch_dim(V)

        #2. Tính attention scores và output
        batch_size, seq_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_length, batch_size)
        attention_probs = attn.get_attention_scores(Q, K, attention_mask)
        hidden_states = torch.bmm(attention_probs, V)

        #3. Tính dropout
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) + self.W_out(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
