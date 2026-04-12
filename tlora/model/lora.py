from concurrent.futures import InterpreterPoolExecutor
import copy
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tlora.utils.registry import ClassRegistry


lora_prosessors = ClassRegistry()
lora_linear_layers = ClassRegistry()


@lora_linear_layers.add_to_registry("lora")
class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, training=True, sig_type=None,):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )
        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank))
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype)) * mask.to(hidden_states.device) # BM
        up_hidden_states = self.up(down_hidden_states) # A

        return up_hidden_states.to(orig_dtype)

@lora_linear_layers.add_to_registry("ortho_lora")
class OrthogonalLoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, training=True, sig_type="last"):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.rank = rank

        # B, A, S
        self.q_layer = nn.Linear(in_features, rank, bias=False)
        self.p_layer = nn.Linear(rank, out_features, bias=False)
        self.lambda_layer = nn.Parameter(torch.ones(1, rank))

        if training:
            # Initialize with SVD - R
            base_m = torch.normal(
                size=(in_features, out_features), mean=0, std=1 / self.rank
            )
            u, s, v = torch.linalg.svd(base_m)

            # Initialize with data sources for R
            if sig_type == 'principal':
                self.q_layer.weight.data = u[: self.rank].clone()
                self.p_layer.weight.data = v[:, : self.rank].clone()
                self.lambda_layer.data = s[None, : self.rank].clone()
            elif sig_type == 'last':
                self.q_layer.weight.data = u[- self.rank :].clone()
                self.p_layer.weight.data = v[:, - self.rank :].clone()
                self.lambda_layer.data = s[None, - self.rank :].clone()
            elif sig_type == 'middle':
                start = math.ceil((u.shape[0] - self.rank) / 2)
                self.q_layer.weight.data = u[start : start + self.rank].clone()
                start = math.ceil((v.shape[1] - self.rank) / 2)
                self.p_layer.weight.data = v[:, start : start + self.rank].clone()
                start = math.ceil((s.shape[0] - self.rank) / 2)
                self.lambda_layer.data = s[None, start : start + self.rank].clone()

            del u, s, v, base_m
            gc.collect()
            torch.cuda.empty_cache()

        self.base_p = copy.deepcopy(self.p_layer)
        self.base_q = copy.deepcopy(self.q_layer)
        self.base_lambda = copy.deepcopy(self.lambda_layer)

        # re-arrange data to be contiguous: for fast access
        for param in self.parameters():
            param.data = param.data.contiguous()
        
        # freeze base parameters
        self.base_p.requires_grad_(False)
        self.base_q.requires_grad_(False)
        self.base_lambda.requires_grad_(False)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank))
        orig_dtype = hidden_states.dtype
        dtype = self.q_layer.weight.dtype

        q_hidden_states = self.q_layer(hidden_states.to(dtype)) * self.lambda_layer * mask.to(hidden_states.device) # BSM
        p_hidden_states = self.p_layer(q_hidden_states) # A

        result = p_hidden_states - self.base_p(
            self.base_q(hidden_states.to(dtype))
            * self.base_lambda
            * mask.to(hidden_states.device)
        ) # BSMA(adjusted) - BSMA(base)

        return result.to(orig_dtype)

    def regularization(self): #Frobenious norm: W^T W - I
        p_reg = torch.sum(
            (
                self.p_layer.weight.T @ self.p_layer.weight
                - torch.eye(self.rank, device=self.p_layer.weight.device)
            )
            ** 2
        )
        q_reg = torch.sum(
            (
                self.q_layer.weight @ self.q_layer.weight.T
                - torch.eye(self.rank, device=self.p_layer.weight.device)
            )
            ** 2
        )
        return p_reg, q_reg

@lora_linear_layers.add_to_registry("lortho_lora")
class LOrthogonalLoRALinearLayer(nn.Module):
    def __init__(
        self, original_layer, in_features, out_features, rank=4, sig_type="principal", do_training=True
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.rank = rank

        self.q_layer = nn.Linear(in_features, rank, bias=False)
        self.p_layer = nn.Linear(rank, out_features, bias=False)
        self.lambda_layer = nn.Parameter(torch.ones(1, rank))

        if do_training:
            u, s, v = torch.linalg.svd(original_layer.weight.data)

            if sig_type == "principal":
                self.q_layer.weight.data = v[: self.rank].clone()
                self.p_layer.weight.data = u[:, : self.rank].clone()
                self.lambda_layer.data = s[None, : self.rank].clone()

            elif sig_type == "last":
                self.q_layer.weight.data = v[-self.rank :].clone()
                self.p_layer.weight.data = u[:, -self.rank :].clone()
                self.lambda_layer.data = s[None, -self.rank :].clone()

            elif sig_type == "middle":
                start = math.ceil((v.shape[0] - self.rank) / 2)
                self.q_layer.weight.data = v[start : start + self.rank].clone()
                start = math.ceil((u.shape[1] - self.rank) / 2)
                self.p_layer.weight.data = u[:, start : start + self.rank].clone()
                start = math.ceil((s.shape[0] - self.rank) / 2)
                self.lambda_layer.data = s[None, start : start + self.rank].clone()

        self.base_p = copy.deepcopy(self.p_layer)
        self.base_q = copy.deepcopy(self.q_layer)
        self.base_lambda = copy.deepcopy(self.lambda_layer)

        for param in self.parameters():
            param.data = param.data.contiguous()

        self.base_p.requires_grad_(False)
        self.base_q.requires_grad_(False)
        self.base_lambda.requires_grad_(False)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank))
        orig_dtype = hidden_states.dtype
        dtype = self.q_layer.weight.dtype

        q_hidden_states = self.q_layer(hidden_states.to(dtype)) * self.lambda_layer * mask.to(hidden_states.device)
        p_hidden_states = self.p_layer(q_hidden_states)

        result = p_hidden_states - self.base_p(
            self.base_q(hidden_states.to(dtype))
            * self.base_lambda
            * mask.to(hidden_states.device)
        )

        return result.to(orig_dtype)

    def regularization(self):
        p_reg = torch.sum(
            (
                self.p_layer.weight.T @ self.p_layer.weight
                - torch.eye(self.rank, device=self.p_layer.weight.device)
            )
            ** 2
        )
        q_reg = torch.sum(
            (
                self.q_layer.weight @ self.q_layer.weight.T
                - torch.eye(self.rank, device=self.p_layer.weight.device)
            )
            ** 2
        )
        return p_reg, q_reg

@lora_prosessors.add_to_registry("lora")
class LoRACrossAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size,
        lora_linear_layer=LoRALinearLayer,
        cross_attention_dim=None,
        rank=4,
        do_training=True,
        sig_type='principal',
    ):
        super().__init__()

        self.do_training = do_training

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_q_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type)
        self.to_k_lora = lora_linear_layer(
            cross_attention_dim or hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type
        )
        self.to_v_lora = lora_linear_layer(
            cross_attention_dim or hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type
        )
        self.to_out_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
        sigma_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        # apply lora to Q for noisy mask
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(
            hidden_states, sigma_mask
        )
        query = attn.head_to_batch_dim(query)

        # cross-attention input for text embeddings
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        # apply lora to K, V for text embeddings
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states, sigma_mask
        )

        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states, sigma_mask
        )

        # get key and value into batch dimension for fast attention calculation
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # forward
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states, sigma_mask
        )

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

@lora_prosessors.add_to_registry("lortho_lora")
class LOrthogonalLoRACrossAttnProcessor(nn.Module):
    def __init__(
        self,
        original_layer,
        hidden_size,
        lora_linear_layer=LOrthogonalLoRALinearLayer,
        cross_attention_dim=None,
        rank=4,
        sig_type="principal",
        do_training=True,
    ):
        super().__init__()

        self.do_training = do_training

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        
        # setup lora parameters based on hyper-parameters
        self.to_q_lora = lora_linear_layer(
            original_layer.to_q, hidden_size, hidden_size, rank, sig_type, self.do_training
        )
        self.to_k_lora = lora_linear_layer(
            original_layer.to_k,
            cross_attention_dim or hidden_size,
            hidden_size,
            rank,
            sig_type,
            self.do_training

        )
        self.to_v_lora = lora_linear_layer(
            original_layer.to_v,
            cross_attention_dim or hidden_size,
            hidden_size,
            rank,
            sig_type,
            self.do_training
        )
        # set up the output place?
        self.to_out_lora = lora_linear_layer(
            original_layer.to_out[0], hidden_size, hidden_size, rank, sig_type, self.do_training
        )

    # sort of a run()
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
        sigma_mask=None,
    ):
        # basically inferencing the input through attention block.
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states, sigma_mask)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states, sigma_mask
        )

        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states, sigma_mask
        )

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states, sigma_mask
        )

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

lora_prosessors.classes["ortho_lora"] = LoRACrossAttnProcessor

# to-do
@lora_linear_layers.add_to_registry("tensor_lora")
class TensorizedLoRALinearLayer(nn.Module):
    def __init__(self, in_f:int, out_f:int, rank:int, bias:bool=False):
        #1. Ghi nhận và khởi tạo các ma trận
        super().__init__()
        self.rank = rank
        self.B_mat = nn.Linear(in_features=in_f, out_features=rank, bias=False)
        self.A_mat = nn.Linear(in_features=rank, out_features=out_f, bias=False)

        #2. Khởi tạo tham số
        nn.init.normal_(tensor=self.B_mat, mean=0.0, std=1.0, generator=None)
        nn.init.zeros_(tensor=self.A_mat)

    def forward(self, input):
        #1. Chuyển đổi kiểu dữ liệu cho đồng bộ lớp LoRA
        input_dtype = input.dtype
        input_formatted = input.to(self.B_mat.weight.dtype)

        #2. Tính toán đầu ra của LoRA
        B_output = self.B_mat(input_formatted)
        A_output = self.A_mat(B_output)

        #3. Chuyển đổi lại kiểu dữ liệu ban đầu
        A_output = A_output.to(input_dtype)

        return A_output

@lora_prosessors.add_to_registry("tensor_lora")
class TensorizedLoRACrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, rank, cross_attention_dim=None, 
                 lora_linear_layer=None, sig_type=None, do_training=None): #this must match trainer_sdxl.py/line 236
        super().__init__()
        self.W_Q = TensorizedLoRALinearLayer(hidden_size, hidden_size, rank)
        self.W_K = TensorizedLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.W_V = TensorizedLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.W_out = TensorizedLoRALinearLayer(hidden_size, hidden_size, rank)

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.rank = rank

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None
    ):
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
        
