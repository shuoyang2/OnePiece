# deepseek_moe.py

import math
import torch
import torch.nn.functional as F
from grouped_gemm.ops import permute, unpermute, gmm
from torch import nn


# --- 【新增】重新引入基尼系数计算函数 ---
def _calculate_gini(loads: torch.Tensor) -> float:
    """
    计算给定负载分布的基尼系数。
    """
    loads = loads.flatten().float()
    if torch.sum(loads) == 0:
        return 0.0
    sorted_loads = torch.sort(loads).values
    n = len(sorted_loads)
    index = torch.arange(1, n + 1, device=loads.device, dtype=loads.dtype)
    numerator = torch.sum((2 * index - n - 1) * sorted_loads)
    denominator = n * torch.sum(sorted_loads)
    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()


# --- 【新增】重新引入您的偏置负载均衡策略类 ---
class LoadBalancingStrategy(nn.Module):
    """
    无辅助损失的负载均衡策略 - 带有数据类型保持功能。
    """

    def __init__(self, num_experts, alpha=0.01, update_freq=1, warmup_steps=1):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.register_buffer('expert_biases', torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer('current_expert_loads', torch.zeros(num_experts, dtype=torch.float32))

    def ensure_float32_biases(self):
        """
        一个手动调用的方法，用于确保 expert_biases 是 float32 类型。
        在模型整体类型转换后调用此方法。
        """
        if self.expert_biases.dtype != torch.float32:
            self.expert_biases.data = self.expert_biases.data.to(torch.float32)
            print(f"INFO: Manually reset expert_biases dtype to {self.expert_biases.dtype}")

    def update_biases(self, expert_usage_counts: torch.Tensor):
        self.step_count += 1
        if self.step_count < self.warmup_steps or self.step_count % self.update_freq != 0:
            return

        expert_usage_counts_float = expert_usage_counts.to(device=self.expert_biases.device, dtype=torch.float32)
        with torch.no_grad():
            self.current_expert_loads.copy_(expert_usage_counts_float)

        avg_load = expert_usage_counts_float.mean() if expert_usage_counts_float.sum() > 0 else 0.0
        load_violation_error = avg_load - expert_usage_counts_float
        bias_updates = self.alpha * torch.sign(load_violation_error)

        with torch.no_grad():
            self.expert_biases.add_(bias_updates)

    def get_load_balancing_stats(self) -> dict:
        recent_loads = self.current_expert_loads
        recent_loads_cpu = recent_loads.cpu()
        stats = {
            'load_gini': _calculate_gini(recent_loads_cpu),
        }
        return stats


# --- 1. 【修改】配置类，加入负载均衡参数 ---
class MoEConfig:
    def __init__(self, args):
        self.hidden_size = args.hidden_units * args.dnn_hidden_units
        self.moe_intermediate_size = args.moe_intermediate_size
        self.hidden_act = "silu"
        self.n_routed_experts = args.moe_num_experts
        self.num_experts_per_tok = args.moe_top_k
        self.n_shared_experts = args.moe_shared_expert_num
        self.scoring_func = 'softmax'
        self.norm_top_k_prob = True
        self.seq_aux = args.moe_use_sequence_aux_loss
        self.aux_loss_alpha = args.moe_sequence_aux_loss_coeff
        self.routed_scaling_factor = 1.0
        self.drop_rate = args.dropout_rate

        # 为 LoadBalancingStrategy 添加参数
        self.load_balancing_alpha = getattr(args, 'moe_load_balancing_alpha', 0.01)
        self.load_balancing_update_freq = getattr(args, 'moe_load_balancing_update_freq', 10)


# --- 2. 专家网络模块 (无变化) ---
class FusedRoutedMLP(torch.nn.Module):
    def __init__(self, num_routed_experts, config: MoEConfig):
        super().__init__()
        self.num_experts = num_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.w1 = torch.nn.Parameter(torch.empty(
            self.num_experts * self.hidden_size, self.intermediate_size * 2))
        self.w2 = torch.nn.Parameter(torch.empty(
            self.num_experts * self.intermediate_size, self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

class FFN(nn.Module):
    def __init__(self, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size


        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# --- 3. 【修改】门控网络 (MoEGate)，整合偏置均衡策略 ---
class MoEGate(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_top_k_prob
        # 将alpha改为Parameter，方便DataParallel时只在主卡更新
        self.alpha = nn.Parameter(torch.tensor(config.aux_loss_alpha, dtype=torch.float32), requires_grad=False)
        self.seq_aux = config.seq_aux
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        # 实例化负载均衡策略
        self.load_balancing = LoadBalancingStrategy(
            num_experts=self.n_routed_experts,
            alpha=config.load_balancing_alpha,
            update_freq=config.load_balancing_update_freq
        )

        self.register_buffer('expert_token_counts', torch.zeros(self.n_routed_experts, dtype=torch.int64))
        
        # 用于动态aux loss调整的gini系数历史记录（最近100步）
        self.gini_history = []
        self.gini_window_size = 100
        self.gini_update_counter = 0
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, h).contiguous()

        # 1. 计算原始路由分数
        logits = F.linear(hidden_states_reshaped.to(torch.float32), self.weight.to(torch.float32))
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # 2. 【核心修改】添加偏置，并使用带偏置的分数选择专家
        bias_scores = scores + self.load_balancing.expert_biases.to(scores.device).unsqueeze(0)
        topk_weights, topk_indices = torch.topk(bias_scores, k=self.top_k, dim=-1, sorted=False)

        # 3. 【核心修改】最终权重从原始(无偏置)的分数中获取
        # 偏置只影响“选谁”，不影响“多重”
        topk_weights = torch.gather(scores, 1, topk_indices)

        # 4. 权重归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        # 5. 更新偏置负载均衡（仅在训练时）
        expert_usage_counts = torch.bincount(topk_indices.view(-1).cpu(), minlength=self.n_routed_experts)
        self.expert_token_counts.copy_(expert_usage_counts)
        if self.training:
            self.load_balancing.update_biases(expert_usage_counts.to(hidden_states.device))

        # 6. 计算辅助损失 (使用原始 scores)
        aux_loss = None
        if self.training and self.alpha.item() > 0.0:
            if self.seq_aux:
                scores_for_seq_aux = scores.view(bsz, seq_len, -1)
                topk_indices_for_aux_loss = topk_indices.view(bsz, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_indices_for_aux_loss,
                                torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device)
                                ).div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_indices.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                pi = scores.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.alpha
        
        # 7. 记录当前步的gini系数（用于动态调整）
        if self.training:
            current_gini = _calculate_gini(expert_usage_counts.float())
            self.gini_history.append(current_gini)
            if len(self.gini_history) > self.gini_window_size:
                self.gini_history.pop(0)

        return topk_indices.to(torch.int32), topk_weights.to(hidden_states.dtype), aux_loss

    def get_moe_statistics(self) -> dict:
        """
        【修改】从 LoadBalancingStrategy 获取完整的统计指标，并添加gini历史和当前alpha。
        """
        stats = self.load_balancing.get_load_balancing_stats()
        return stats
    
    def update_aux_loss_alpha(self, gini_target_min: float, gini_target_max: float, adjust_rate: float):
        """
        基于窗口内平均gini系数动态更新aux loss系数
        仅在窗口满100步时调用
        
        Args:
            gini_target_min: 目标gini系数下限
            gini_target_max: 目标gini系数上限
            adjust_rate: 调整速率
        """
        if len(self.gini_history) < self.gini_window_size:
            return  # 窗口未满，不更新
        
        # 计算窗口内平均gini系数
        avg_gini = sum(self.gini_history) / len(self.gini_history)
        print(avg_gini)
        print(f"尝试更新！")
        # 根据avg_gini调整alpha
        with torch.no_grad():
            current_alpha = self.alpha.item()
            if avg_gini < gini_target_min:
                print("DOWN")
                # 基尼系数过低（分布过于均匀），降低aux loss系数
                new_alpha = max(0.0, current_alpha - adjust_rate)
                self.alpha.copy_(torch.tensor(new_alpha, dtype=torch.float32))
            elif avg_gini > gini_target_max:
                print("UP")
                # 基尼系数过高（分布不均），增加aux loss系数
                new_alpha = min(0.03,current_alpha + adjust_rate)
                self.alpha.copy_(torch.tensor(new_alpha, dtype=torch.float32))


# --- 4. MoE层 (无变化) ---
class DeepseekMoE(torch.nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.experts = FusedRoutedMLP(self.num_experts, config)
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            self.shared_expert = FFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
            )
        else:
            self.shared_expert = None

    @torch.compiler.disable()
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        permuted_tokens, row_id_map = permute(hidden_states_reshaped, topk_idx)
        row_id_map = row_id_map.to(torch.int32)
        tokens_per_expert_cpu = self.gate.expert_token_counts
        active_expert_indices = torch.nonzero(tokens_per_expert_cpu > 0, as_tuple=False).squeeze(-1)

        if active_expert_indices.numel() == 0:
            moe_output_reshaped = torch.zeros_like(hidden_states_reshaped)
        else:
            active_expert_token_counts = tokens_per_expert_cpu.index_select(0, active_expert_indices)
            w1_weight = self.experts.w1.view(self.num_experts, hidden_dim, self.experts.intermediate_size * 2)
            w2_weight = self.experts.w2.view(self.num_experts, self.experts.intermediate_size, hidden_dim)
            idx_gpu = active_expert_indices.to(w1_weight.device)
            w1_active = w1_weight.index_select(0, idx_gpu).to(torch.bfloat16)
            w2_active = w2_weight.index_select(0, idx_gpu).to(torch.bfloat16)
            total_active_tokens = int(active_expert_token_counts.sum())
            permuted_tokens_active = permuted_tokens[:total_active_tokens].contiguous().to(torch.bfloat16)
            w1_output = gmm(permuted_tokens_active, w1_active, active_expert_token_counts.cpu(), trans_b=False)
            gate_output, up_output = torch.chunk(w1_output, 2, dim=-1)
            intermediate_activated = F.silu(gate_output) * up_output
            permuted_expert_outputs = gmm(intermediate_activated, w2_active, active_expert_token_counts.cpu(),
                                          trans_b=False)
            moe_output_reshaped = unpermute(permuted_expert_outputs, row_id_map, topk_weight.to(torch.float32))

        moe_output = moe_output_reshaped.view(batch_size, seq_len, -1)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(residual)
            final_output = moe_output + shared_output
        else:
            final_output = moe_output

        return final_output, topk_idx, aux_loss


# --- 5. Transformer Block (无变化) ---
class DeepseekMoEBlock(torch.nn.Module):
    def __init__(self, args, moe_config):
        super().__init__()
        from model import FlashMultiHeadAttention
        hidden_dim = args.hidden_units * args.dnn_hidden_units
        self.attn_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if args.rms_norm else torch.nn.LayerNorm(
            hidden_dim, eps=1e-8)
        self.ffn_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if args.rms_norm else torch.nn.LayerNorm(hidden_dim,
                                                                                                             eps=1e-8)
        self.attn = FlashMultiHeadAttention(
            hidden_units=hidden_dim, num_heads=args.num_heads,
            dropout_rate=args.dropout_rate, rope=args.rope,
            max_seq_len=args.maxlen + 1)
        self.moe_mlp = DeepseekMoE(config=moe_config)

    def forward(self, x, attn_mask=None):
        residual = x
        x_norm = self.attn_layernorm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + attn_output
        residual = x
        x_norm = self.ffn_layernorm(x)
        moe_output, topk_idx, aux_loss = self.moe_mlp(x_norm)
        x = residual + moe_output
        return x, topk_idx, aux_loss

    def get_moe_statistics(self):
        if hasattr(self, 'moe_mlp') and hasattr(self.moe_mlp, 'gate'):
            return self.moe_mlp.gate.get_moe_statistics()
        return None


# --- 6. 统计日志函数 (无变化) ---
def log_moe_statistics(args, moe_config):
    hidden_size = moe_config.hidden_size
    intermediate_size = moe_config.moe_intermediate_size
    num_experts = moe_config.n_routed_experts
    top_k = moe_config.num_experts_per_tok
    dense_mlp_params = (hidden_size * intermediate_size) * 2 + (intermediate_size * hidden_size)
    dense_mlp_flops = 2 * dense_mlp_params
    expert_params = num_experts * dense_mlp_params
    gate_params = hidden_size * num_experts
    moe_total_params = expert_params + gate_params
    gate_flops = 2 * gate_params
    active_expert_flops = top_k * dense_mlp_flops
    moe_total_flops = gate_flops + active_expert_flops
    param_ratio = moe_total_params / dense_mlp_params if dense_mlp_params > 0 else 0
    flops_ratio = moe_total_flops / dense_mlp_flops if dense_mlp_flops > 0 else 0

    def format_num(n):
        if n >= 1_000_000_000: return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000: return f"{n / 1_000_000:.2f}M"
        if n >= 1_000: return f"{n / 1_000:.2f}K"
        return str(int(n))

    print("\n" + "=" * 70)
    print(" " * 15 + "Sparse MoE vs. Dense MLP Statistics")
    print("=" * 70)
    print(
        f"  Base Model Config: hidden_size={args.hidden_units}, dnn_hidden_units={args.dnn_hidden_units} -> effective_dim={hidden_size}")
    print(f"  MoE Config: intermediate_size={intermediate_size}, num_experts={num_experts}, top_k={top_k}")
    print("-" * 70)
    print(f"{'Metric':<22} | {'Dense MLP (1 Expert)':<22} | {'Sparse MoE Layer':<22}")
    print("-" * 70)
    print(
        f"{'Total Parameters':<22} | {format_num(dense_mlp_params):<22} | {format_num(moe_total_params):<22} ({param_ratio:.2f}x)")
    print(
        f"{'Inference FLOPs/token':<22} | {format_num(dense_mlp_flops):<22} | {format_num(moe_total_flops):<22} ({flops_ratio:.2f}x)")
    print("=" * 70)
    print("  Note: FLOPs are an approximation for a single token forward pass (MACs * 2).\n")