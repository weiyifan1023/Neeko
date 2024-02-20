import torch
from torch import nn
from typing import Optional, List


def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0])
    return x.float().var() / (x.float().mean()**2 + eps)


def topk_mask_seq(tensor, capacity_factor: float = 1.0):
    batch_size, seq_len, num_experts = tensor.size()
    tensor_softmax = nn.functional.softmax(tensor, dim=-1)
    top_k = int(seq_len // num_experts * capacity_factor)
    _, topk_indices = tensor_softmax.topk(top_k, dim=1)
    mask = torch.full_like(tensor, -1e4)
    mask.scatter_(1, topk_indices, 0)
    masked_tensor = tensor + mask
    return masked_tensor


def topk_mask_tensor(tensor, top_k=2):
    topk_values, topk_indices = torch.topk(tensor, k=top_k, dim=2)
    mask = torch.full_like(tensor, -1e4)
    mask.scatter_(2, topk_indices, topk_values)
    return mask


class LossFunction:
    def __init__(self, loss_fn: str, L_factor: float, Z_factor: float, V_factor: float, var_lower_bound: float):
        self.loss_fn = loss_fn
        self.L_factor = L_factor
        self.Z_factor = Z_factor
        self.V_factor = V_factor
        self.var_lower_bound = var_lower_bound
        self.capacity_factor = 1.2

    def load_balance_loss_fn(self, probs: torch.Tensor):
        num_groups, tokens_per_group, n = probs.shape
        probs = probs.reshape(-1, n)
        importance = probs.sum(0)
    #     num_groups, tokens_per_group, n = probs.shape
    #     _, expert_index = torch.topk(probs, k=1, dim=-1)
    #     expert_index = torch.nn.functional.one_hot(expert_index, num_classes=n).reshape(num_groups, tokens_per_group, n)
    #     position_in_expert = expert_index.cumsum(dim=-2)
    #     expert_capacity = tokens_per_group // n * self.capacity_factor
    #     expert_mask = (position_in_expert <= expert_capacity).float()
    #     tokens_per_group_and_expert = torch.mean(expert_mask, dim=-2)
    #     router_prob_per_group_and_expert = torch.mean(probs, dim=-2)
    #     loss = torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (n ** 2)

        return cv_squared(importance)

    @staticmethod
    def z_loss_fn(logits: torch.Tensor) -> float:
        num_groups, tokens_per_group, _ = logits.shape
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = log_z ** 2
        return torch.sum(z_loss) / (num_groups * tokens_per_group)

    def compute_loss(
        self,
        logits: Optional[torch.Tensor],
        probs: Optional[torch.Tensor],
        layer_to_orthogonal: Optional[List[nn.Module]]
    ):
        loss_list = []
        if self.loss_fn == "None":
            return
        if "L" in self.loss_fn:
            loss_list.append(self.load_balance_loss_fn(probs) * self.L_factor)
        if "Z" in self.loss_fn:
            loss_list.append(self.z_loss_fn(logits) * self.Z_factor)
        if "V" in self.loss_fn:
            loss_list.append(self.var_loss_fn(logits) * self.V_factor)
        if len(loss_list):
            total_loss = sum(loss_list)
            total_loss.requires_grad_(True)
            total_loss.backward(retain_graph=True)


class Dense(nn.Module):
    def __init__(self, dim: int, num_moe: int, top_n: int, loss_fn: str, L_factor: float, Z_factor: float, V_factor: float, var_lower_bound: float) -> None:
        super().__init__()
        self.dim = dim
        self.num_moe = num_moe
        self.top_n = top_n
        self.loss_fn = loss_fn
        self.loss = LossFunction(loss_fn, L_factor, Z_factor, V_factor, var_lower_bound)
        self.linear_layer = nn.Linear(dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def calculate_temperature(self, x):
        max_x = torch.max(x, dim=-1, keepdim=True).values
        min_x = torch.min(x, dim=-1, keepdim=True).values
        temperature = torch.log(torch.tensor(4.0)) / (max_x - min_x)
        return temperature

    def forward(self, x):
        # weight = self.linear_layer.weight
        # cosine_distances = nn.functional.cosine_similarity(weight.unsqueeze(1), weight.unsqueeze(0), dim=-1)
        # print(cosine_distances)
        logits = self.linear_layer(x)
        # temperature = self.calculate_temperature(logits)
        probs = self.softmax(logits)
        # print(probs)
        # assert False
        if self.training:
            self.loss.compute_loss(logits, probs, [self.linear_layer])
        return probs


class Sparse(nn.Module):
    def __init__(self, dim: int, num_moe: int, top_n: int, loss_fn: str, L_factor: float, Z_factor: float, V_factor: float, var_lower_bound: float) -> None:
        super().__init__()
        self.dim = dim
        self.num_moe = num_moe
        self.top_n = top_n
        self.capacity_factor = 1.2
        self.loss = LossFunction(loss_fn, L_factor, Z_factor, V_factor, var_lower_bound)
        self.linear_layer = nn.Linear(dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.cnt = 0

    def forward(self, x):
        # weight = self.linear_layer.weight
        # cosine_distances = nn.functional.cosine_similarity(weight.unsqueeze(1), weight.unsqueeze(0), dim=-1)
        # print(cosine_distances)
        logits = self.linear_layer(x)
        logits_mask = topk_mask_tensor(logits, self.top_n)
        probs = self.softmax(logits_mask)
        if self.training:
            self.loss.compute_loss(logits, probs, [self.linear_layer])
        # self.cnt += 1
        # if self.cnt % 20 == 0:
        #     print(probs.reshape(-1, self.num_moe).sum(dim=-2))
            # print(probs[0])
            # print(len(probs[0]))
        # assert False
        return probs


class Noisy(nn.Module):
    def __init__(self, dim: int, num_moe: int, top_n: int, loss_fn: str) -> None:
        super().__init__()
        self.dim = dim
        self.num_moe = num_moe
        self.top_n = top_n
        self.loss = LossFunction(loss_fn)
        self.linear_layer_miu = nn.Linear(dim, num_moe, bias=False)
        self.linear_layer_sigma = nn.Linear(dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        miu = self.linear_layer_miu(x)
        sigma = self.linear_layer_sigma(x)
        noise = torch.randn_like(miu)
        logits = miu + sigma * noise
        logits_mask = topk_mask_tensor(logits, self.top_n)
        probs = self.softmax(logits_mask)
        if self.training:
            self.loss.compute_loss(logits, probs, [self.linear_layer_miu, self.linear_layer_sigma])
        return probs


class Random(nn.Module):
    def __init__(self, dim: int, num_moe: int, top_n: int, loss_fn: str) -> None:
        super().__init__()
        self.dim = dim
        self.num_moe = num_moe
        self.top_n = top_n
        self.loss = LossFunction(loss_fn)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        routing_weights = torch.rand(batch_size, seq_len, self.num_moe).to(x.device)
        routing_weights /= routing_weights.sum(dim=2, keepdim=True)
        return routing_weights


GATING_TO_MODEL_MAPPING = {
    "Dense": Dense,
    "Sparse": Sparse,
    "Noisy": Noisy,
    "Random": Random,
}


# --model_name_or_path ../../../dataset/hf_models/llama-7b --do_train --finetuning_type moelora --output_dir ./ckpt/wikitq_noisy_moelora --max_source_length 2048 --overwrite_cache --per_device_train_batch_size 2 --gradient_accumulation_steps 2 --lr_scheduler_type cosine --logging_steps 20 --save_steps 1000 --learning_rate 1e-4 --num_train_epochs 1.0 --plot_loss --lora_rank 32 --num_moe 4 --top_n_for_moe 2 --gating SwitchTransformersTop1Router --fp16
