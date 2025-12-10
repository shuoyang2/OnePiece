import math
import torch
import torch.nn.functional as F
from grouped_gemm.ops import permute, unpermute, gmm
from torch import nn


def _calculate_gini(loads: torch.Tensor) -> float:
    """
    计算给定负载分布的基尼系数。此函数被完整保留。
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


# --- 1. 配置类 ---
class MoEConfig:
    def __init__(self, args):
        self.hidden_size = args.hidden_units * args.dnn_hidden_units
        self.moe_intermediate_size = args.moe_intermediate_size
        self.hidden_act = "silu"

        # MoE 相关参数 (DeepSeek V2 风格)
        self.n_routed_experts = args.moe_num_experts
        self.num_experts_per_tok = args.moe_top_k
        self.n_shared_experts = args.moe_shared_expert_num

        # 门控和损失函数相关
        self.scoring_func = 'softmax'  # Deepseek 使用 softmax
        self.norm_topk_prob = True

        # 序列级辅助损失参数 (直接映射)
        self.seq_aux = args.moe_use_sequence_aux_loss
        self.aux_loss_alpha = args.moe_sequence_aux_loss_coeff

        # Deepseek MoE 特有参数
        self.routed_scaling_factor = 1.0
        self.topk_method = "gready"
        self.n_group = 0
        self.topk_group = 0


# --- 2. 专家网络模块 ---
class FusedRoutedMLP(torch.nn.Module):
    """
    使用连续内存存储可路由专家(Routed Experts)的权重，以适配 grouped_gemm
    """

    def __init__(self, num_routed_experts, config: MoEConfig):
        super().__init__()
        self.num_experts = num_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.w1 = torch.nn.Parameter(torch.empty(
            self.num_experts * self.hidden_size,
            self.intermediate_size * 2
        ))
        self.w2 = torch.nn.Parameter(torch.empty(
            self.num_experts * self.intermediate_size,
            self.hidden_size
        ))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))


class PointWiseFeedForward(torch.nn.Module):
    """
    标准的前馈网络，用于共享专家(Shared Expert)
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# --- 3. 新的门控网络 (MoEGate) ---
class MoEGate(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        # 用于存储指标计算所需的数据
        self.register_buffer('current_expert_loads', torch.zeros(self.n_routed_experts, dtype=torch.float32))
        self.register_buffer('expert_token_counts', torch.zeros(self.n_routed_experts, dtype=torch.int64))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # 注意：输入的 hidden_states 已经在 DeepseekMoE.forward 中被清理过 NaNs，这里直接使用
        bsz, seq_len, h = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, h).contiguous()

        # 1. 计算路由专家的logits和scores
        # 始终使用 float32 进行 logits 计算以保证精度
        logits = F.linear(hidden_states_reshaped.to(torch.float32), self.weight.to(torch.float32))
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # 2. top_k 选择
        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # [Safety Check] 确保索引范围
        topk_indices = torch.clamp(topk_indices, min=0, max=self.n_routed_experts - 1)

        # 3. 权重归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        # 4. 计算并存储负载指标
        topk_indices_cpu = topk_indices.view(-1).cpu()
        expert_usage_counts = torch.bincount(topk_indices_cpu, minlength=self.n_routed_experts)

        self.expert_token_counts.copy_(expert_usage_counts)
        with torch.no_grad():
            self.current_expert_loads.copy_(expert_usage_counts.float())

        # 5. 计算辅助损失
        aux_loss = None
        if self.training and self.alpha > 0.0:
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

        return topk_indices.to(torch.int32), topk_weights.to(hidden_states.dtype), aux_loss

    def get_moe_statistics(self) -> dict:
        recent_loads = self.current_expert_loads
        recent_loads_cpu = recent_loads.cpu()
        stats = {
            'load_gini': _calculate_gini(recent_loads_cpu),
            'load_min': recent_loads_cpu.min().item(),
            'load_max': recent_loads_cpu.max().item(),
        }
        return stats


# --- 4. 新的MoE层 ---
class DeepseekMoE(torch.nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts

        # 可路由专家模块 (Routed Experts)
        self.experts = FusedRoutedMLP(self.num_experts, config)
        self.gate = MoEGate(config)

        # 共享专家模块 (Shared Expert)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            self.shared_expert = PointWiseFeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
            )
        else:
            self.shared_expert = None

    @torch.compiler.disable()
    def forward(self, hidden_states):
        # [CRITICAL FIX] 在最开始清理 NaN/Inf。
        # 在 Beam Search 中，Padding Tokens 可能会产生 NaN，这不仅影响 Gate 索引计算，
        # 更会直接导致传入 permute 的数据含有 NaN，从而引发 Illegal Memory Access。
        # hidden_states 可能是 [Batch, Seq, Dim] 或 [Total_Seq, Dim]
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else 1
        hidden_dim = hidden_states.shape[-1]

        residual = hidden_states

        # [Vital] 确保输入到 permute 之前是绝对 contiguous 的 2D 视图
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim).contiguous()

        # 1. 门控网络计算 (使用已清洗的 hidden_states)
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        # [Vital] 展平 topk_idx 以匹配 flatten 后的 hidden_states
        # 并确保是 int32 且 contiguous
        topk_idx_flat = topk_idx.view(-1, topk_idx.shape[-1]).to(torch.int32).contiguous()

        # 2. grouped_gemm 核心路径
        # permute(input, indices) -> input: [N, D], indices: [N, K]
        try:
            permuted_tokens, row_id_map = permute(hidden_states_reshaped, topk_idx_flat)
        except RuntimeError as e:
            # 捕获并提供更多调试信息
            print(
                f"CRITICAL ERROR in permute: hidden shape {hidden_states_reshaped.shape}, idx shape {topk_idx_flat.shape}")
            raise e

        row_id_map = row_id_map.to(torch.int32)

        # 3. 准备专家计数 (CPU)
        tokens_per_expert_cpu = self.gate.expert_token_counts.cpu()
        active_expert_indices = torch.nonzero(tokens_per_expert_cpu > 0, as_tuple=False).squeeze(-1)

        if active_expert_indices.numel() == 0:
            moe_output_reshaped = torch.zeros_like(hidden_states_reshaped)
        else:
            active_expert_token_counts = tokens_per_expert_cpu.index_select(0, active_expert_indices)
            w1_weight = self.experts.w1.view(self.num_experts, hidden_dim, self.experts.intermediate_size * 2)
            w2_weight = self.experts.w2.view(self.num_experts, self.experts.intermediate_size, hidden_dim)

            idx_gpu = active_expert_indices.to(w1_weight.device)
            # 确保计算时使用 bf16，与 grouped_gemm 预期一致
            w1_active = w1_weight.index_select(0, idx_gpu).to(torch.bfloat16)
            w2_active = w2_weight.index_select(0, idx_gpu).to(torch.bfloat16)

            total_active_tokens = int(active_expert_token_counts.sum())
            permuted_tokens_active = permuted_tokens[:total_active_tokens].contiguous().to(torch.bfloat16)

            # GMM 计算
            w1_output = gmm(permuted_tokens_active, w1_active, active_expert_token_counts, trans_b=False)
            gate_output, up_output = torch.chunk(w1_output, 2, dim=-1)
            intermediate_activated = F.silu(gate_output) * up_output
            permuted_expert_outputs = gmm(intermediate_activated, w2_active, active_expert_token_counts,
                                          trans_b=False)

            # Unpermute
            topk_weight_flat = topk_weight.view(-1, topk_weight.shape[-1]).contiguous()
            # Unpermute 时可以使用原始精度或 float32
            moe_output_reshaped = unpermute(permuted_expert_outputs, row_id_map, topk_weight_flat.to(torch.float32))

        # 4. 恢复形状并添加共享专家
        if hidden_states.dim() == 3:
            moe_output = moe_output_reshaped.view(batch_size, seq_len, -1)
        else:
            moe_output = moe_output_reshaped

        if self.shared_expert is not None:
            shared_output = self.shared_expert(residual)
            final_output = moe_output + shared_output
        else:
            final_output = moe_output

        return final_output, topk_idx, aux_loss


# --- 5. 新的Transformer Block ---
class DeepseekMoEBlock(torch.nn.Module):
    """
    修改后的 Block，支持 past_key_value 的传入和传出。
    """

    def __init__(self, args, moe_config):
        super().__init__()
        from model import FlashMultiHeadAttention
        hidden_dim = args.hidden_units * args.dnn_hidden_units

        self.attn_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if args.rms_norm else torch.nn.LayerNorm(
            hidden_dim, eps=1e-8)
        self.ffn_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if args.rms_norm else torch.nn.LayerNorm(hidden_dim,
                                                                                                             eps=1e-8)

        self.attn = FlashMultiHeadAttention(
            hidden_units=hidden_dim,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            rope=args.rope,
            max_seq_len=args.maxlen + 1
        )
        self.moe_mlp = DeepseekMoE(config=moe_config)

    def forward(self, x, attn_mask=None, past_key_value=None):
        """
        Args:
            x: [Batch, Seq, Dim]
            attn_mask: Attention mask
            past_key_value: Tuple(K, V) from previous step

        Returns:
            x: Hidden states
            meta: (topk_idx, aux_loss) -> 适配 _, 占位符
            present_key_value: Tuple(K, V) -> updated cache
        """
        # Pre-LN 结构
        residual = x
        x_norm = self.attn_layernorm(x)

        # [MODIFIED] 传入 past_key_value，接收 updated cache
        attn_output, present_key_value = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            past_key_value=past_key_value
        )

        x = residual + attn_output

        residual = x
        x_norm = self.ffn_layernorm(x)

        # MoE MLP 不需要 KV Cache，它是 pointwise 的
        moe_output, topk_idx, aux_loss = self.moe_mlp(x_norm)
        x = residual + moe_output

        # [MODIFIED] 打包元数据，以便调用者可以使用 x, _, new_kv = block(...)
        meta_outputs = (topk_idx, aux_loss)

        return x, meta_outputs, present_key_value

    def get_moe_statistics(self):
        if hasattr(self, 'moe_mlp') and hasattr(self.moe_mlp, 'gate'):
            return self.moe_mlp.gate.get_moe_statistics()
        return None