from pathlib import Path
from signal import Sigmasks

import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dataset import save_emb
from utils import *
from deepseek_moe import *


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, offset=0):
        # [MODIFIED] 增加 offset 参数
        # x shape: [bs, num_heads, seq_len, head_dim]
        seq_len = x.shape[2]

        # 根据 offset 切片，获取对应的 cos/sin
        # 训练时 offset=0, slice=0:seq_len
        # 推理Step k时 offset=k, slice=k:k+1
        cos = self.cos_cached[:, :, offset:offset + seq_len, ...]
        sin = self.sin_cached[:, :, offset:offset + seq_len, ...]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.cat((-x_odd, x_even), dim=-1)

        return x * cos + x_rotated * sin


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, rope=False, max_seq_len=500):
        super(FlashMultiHeadAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.rope = rope
        max_seq_len = 500
        self.max_seq_len = 500
        if rope:
            self.rope_unit = RotaryEmbedding(self.head_dim, max_seq_len)
            self.rel_pos_bias = torch.nn.Embedding(2 * max_seq_len - 1, self.num_heads)

    def forward(self, query, key, value, attn_mask=None, past_key_value=None):
        """
        :param past_key_value: Tuple[torch.Tensor, torch.Tensor] (K_cache, V_cache)
        """
        batch_size, seq_len, _ = query.size()

        # 1. 计算当前输入的 Q, K, V
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 处理 RoPE 和 KV Cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 获取已经生成的序列长度作为 offset
            kv_seq_len_past = past_key.shape[2]
        else:
            kv_seq_len_past = 0

        if self.rope:
            # [MODIFIED] 应用 RoPE 时传入 offset
            # Q 的位置是 [past_len, past_len + seq_len]
            Q = self.rope_unit(Q, offset=kv_seq_len_past)
            # K 的位置也是 [past_len, past_len + seq_len] (对于当前的新 token)
            K = self.rope_unit(K, offset=kv_seq_len_past)

            # 注意：相对位置编码 rel_pos_bias 在 KV Cache 模式下较复杂，
            # 通常 RoPE 已经足够。如果必须保留 rel_pos_bias，需要根据 total length 重算 bias。
            # 这里为了适配 Beam Search 和 Cache，RoPE 是主要位置编码手段。
            # 如果 strict 依赖 rel_pos_bias，逻辑需要调整为 cat 后的完整长度。

        # 3. 更新 KV Cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 拼接: [B, H, L_past, D] cat [B, H, L_curr, D] -> [B, H, L_total, D]
            K = torch.cat((past_key, K), dim=2)
            V = torch.cat((past_value, V), dim=2)

        # 保存当前的 KV 为 present_key_value
        present_key_value = (K, V)

        # 4. Attention 计算
        # Q: [B, H, L_curr, D]
        # K, V: [B, H, L_total, D]

        # [Compatible] 兼容旧的 rel_pos_bias 逻辑 (仅当非 RoPE 或混合使用时)
        # 注意：下面的 rel_pos_bias 逻辑是基于 Q 和 K 长度计算的，
        # 如果使用了 Cache，K 的长度已经变长，需要确保 indices 计算正确。
        rel_bias = None
        if self.rope and hasattr(self, 'rel_pos_bias'):
            # 计算 total length
            kv_total_len = K.shape[2]
            q_len = Q.shape[2]

            # positions: [q_len, kv_total_len]
            # Q 的绝对位置是从 kv_seq_len_past 开始的
            q_indices = torch.arange(kv_seq_len_past, kv_seq_len_past + q_len, device=query.device).view(-1, 1)
            k_indices = torch.arange(kv_total_len, device=query.device).view(1, -1)
            positions = q_indices - k_indices

            rel_pos_indices = positions + self.max_seq_len - 1
            rel_pos_indices = torch.clamp(rel_pos_indices, 0, 2 * self.max_seq_len - 2)
            rel_bias = self.rel_pos_bias(rel_pos_indices).permute(2, 0, 1).unsqueeze(0)

        # 5. Scaled Dot Product Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            final_attn_mask = attn_mask

            # 处理 mask 形状广播
            # attn_mask 期望: [B, 1, Q_len, KV_len]
            if attn_mask is not None:
                if attn_mask.dim() == 2:  # [B, KV_len] -> [B, 1, 1, KV_len] (广播)
                    final_attn_mask = attn_mask.view(batch_size, 1, 1, -1)
                elif attn_mask.dim() == 3:  # [B, Q_len, KV_len] -> [B, 1, Q_len, KV_len]
                    final_attn_mask = attn_mask.unsqueeze(1)

            if self.rope and rel_bias is not None:
                # 复杂的 bias 处理，通常 FlashAttn 接口不直接支持加 bias tensor (除了 mask)
                # 这里回退到手动加 bias，或者将 mask 视为 float mask
                # 为简单起见，如果用了 RoPE + RelBias，降级到手动实现或者 mask hack

                if final_attn_mask is not None:
                    # bool mask -> float mask
                    float_mask = torch.zeros_like(final_attn_mask, dtype=torch.float, device=query.device)
                    float_mask.masked_fill_(final_attn_mask.logical_not(), float('-inf'))
                    final_attn_mask = float_mask + rel_bias
                else:
                    final_attn_mask = rel_bias

            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout_rate if self.training else 0.0,
                attn_mask=final_attn_mask
            )
        else:
            # Standard implementation fallback
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if rel_bias is not None:
                scores += rel_bias

            scores = torch.clamp(scores, min=-30, max=30)

            if attn_mask is not None:
                # 适配 cache 后的 mask
                if attn_mask.dim() == 2:
                    scores.masked_fill_(attn_mask.view(batch_size, 1, 1, -1).logical_not(), float('-inf'))
                elif attn_mask.dim() == 3:
                    scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
                else:
                    scores.masked_fill_(attn_mask.logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        attn_output = attn_output.clamp(min=-30, max=30)
        output = self.out_linear(attn_output)

        return output, present_key_value

    def infer(self, query, key, value, attn_mask=None):
        batch_size, q_seq_len, _ = query.size()
        kv_seq_len = key.size(1)

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            # 对query和key应用RoPE编码
            Q = self.rope_unit(Q)
            K = self.rope_unit(K)
            # 创建相对位置索引矩阵
            positions = torch.arange(q_seq_len, device=query.device).view(-1, 1) - torch.arange(kv_seq_len,
                                                                                                device=query.device).view(
                1, -1)
            # 偏移索引，使其从0开始，作为Embedding的输入
            rel_pos_indices = positions + self.max_seq_len - 1

            # 从Embedding表中查找偏置值
            # shape: [q_seq_len, kv_seq_len, num_heads]
            rel_bias = self.rel_pos_bias(rel_pos_indices)
            # 调整形状以匹配注意力分数矩阵 [batch_size, num_heads, q_seq_len, kv_seq_len]
            # [q_seq_len, kv_seq_len, num_heads] -> [num_heads, q_seq_len, kv_seq_len] -> [1, num_heads, q_seq_len, kv_seq_len]
            rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            final_attn_mask = attn_mask
            if self.rope:
                # 将布尔掩码转换为浮点数掩码，False 的位置为 -inf
                float_mask = torch.zeros_like(attn_mask, dtype=torch.float, device=query.device)
                float_mask.masked_fill_(attn_mask.logical_not(), float('-inf'))

                # 将 rel_bias 加到掩码上，PyTorch 会将其直接加到注意力分数上
                # 广播机制: [1, num_heads, q_seq_len, kv_seq_len] + [batch_size, 1, q_seq_len, kv_seq_len]
                final_attn_mask = float_mask.unsqueeze(1) + rel_bias
            else:
                final_attn_mask = final_attn_mask.unsqueeze(1) if attn_mask is not None else None
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=final_attn_mask
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if self.rope:
                scores += rel_bias

            scores = torch.clamp(scores, min=-30, max=30)

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)
        # if attn_output is not None:
        # attn_output = torch.clamp(attn_output, min=-30, max=30)
        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.hidden_units)
        attn_output = attn_output.clamp(min=-30, max=30)
        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, hidden_layer_units_multiplier):
        super(PointWiseFeedForward, self).__init__()

        self.linear1 = torch.nn.Linear(hidden_units, hidden_units * hidden_layer_units_multiplier)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units * hidden_layer_units_multiplier, hidden_units)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # 输入形状: (batch_size, seq_len, hidden_units)
        outputs = self.linear1(inputs)  # 形状: (batch_size, seq_len, hidden_units * multiplier)
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.linear2(outputs)  # 形状: (batch_size, seq_len, hidden_units)
        outputs = self.dropout2(outputs)
        return outputs


class SidRewardHSTUBlock(torch.nn.Module):
    """
    专用于 SID 和 Reward 模型的 HSTU (Hierarchical Sequential Transduction Unit) 模块。
    该模块可以处理自注意力和交叉注意力，并替代了标准的多头注意力和前馈网络层。
    """

    def __init__(self, hidden_units, num_heads, dropout_rate, max_seq_len):
        super(SidRewardHSTUBlock, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.max_seq_len = max_seq_len

        # 针对 query, key, value 和门控的独立投影层
        self.q_proj = torch.nn.Linear(hidden_units, hidden_units)
        self.k_proj = torch.nn.Linear(hidden_units, hidden_units)
        self.v_proj = torch.nn.Linear(hidden_units, hidden_units)
        self.u_proj = torch.nn.Linear(hidden_units, hidden_units)  # 门控 U 的投影

        self.f2_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.rel_pos_bias = torch.nn.Embedding(2 * max_seq_len - 1, self.num_heads)

    def forward(self, query, key, value, attn_mask=None, infer=False):
        batch_size, q_seq_len, _ = query.shape
        kv_seq_len = key.shape[1]

        # --- 1. 逐点投影 (Pointwise Projection) ---
        U = self.activation(self.u_proj(query))
        Q_proj = self.activation(self.q_proj(query))
        K_proj = self.activation(self.k_proj(key))
        V_proj = self.activation(self.v_proj(value))

        # 为多头注意力重塑形状
        Q = Q_proj.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K_proj.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V_proj.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # --- 2. 空间聚合 (Spatial Aggregation) ---
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # 添加相对位置偏置
        if infer:
            positions = (kv_seq_len - 1) * torch.ones(q_seq_len, dtype=torch.long, device=query.device).view(
                -1, 1
            ) - torch.arange(kv_seq_len, dtype=torch.long, device=query.device).view(1, -1)
        else:
            positions = torch.arange(q_seq_len, device=query.device).view(-1, 1) - torch.arange(
                kv_seq_len, device=query.device
            ).view(1, -1)
        rel_pos_indices = positions + self.max_seq_len - 1
        # 为了兼容更长的序列，这里对索引进行裁剪，避免越界
        rel_pos_indices = torch.clamp(rel_pos_indices, min=0, max=2 * self.max_seq_len - 2)
        rel_bias = self.rel_pos_bias(rel_pos_indices).permute(2, 0, 1).unsqueeze(0)
        scores += rel_bias

        # 应用激活函数 (替代 Softmax) 和注意力掩码
        attn_weights = self.activation(scores)
        if attn_mask is not None:
            # 确保掩码形状兼容 [B, H, q_len, kv_len]
            if attn_mask.dim() == 3:  # e.g., [B, q_len, kv_len]
                attn_mask_expanded = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:  # e.g., [B, 1, q_len, kv_len]
                attn_mask_expanded = attn_mask
            else:  # e.g., [q_len, kv_len]
                attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(attn_mask_expanded.logical_not(), 0.0)

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.hidden_units)

        # --- 3. 逐点变换 (Pointwise Transformation) ---
        gated_output = attn_output * U
        final_output = self.f2_linear(gated_output)
        return final_output

    def infer(self, query, key, value, attn_mask=None):
        # 推理时直接调用 forward
        return self.forward(query, key, value, attn_mask, infer=True)


class HSTUBlock(torch.nn.Module):
    """
    HSTU (Hierarchical Sequential Transduction Unit) 模块的实现。
    该模块同时替代了标准 Transformer 中的多头注意力和前馈网络层。
    """

    def __init__(self, hidden_units, num_heads, dropout_rate, max_seq_len):
        super(HSTUBlock, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.max_seq_len = max_seq_len

        # 根据论文，f1 和 f2 是简单的线性层
        # 这个单一的投影层用于一次性生成 Q, K, V 和门控向量 U
        self.f1_linear = torch.nn.Linear(hidden_units, hidden_units * 4)

        # 最终的输出投影层 f2
        self.f2_linear = torch.nn.Linear(hidden_units, hidden_units)

        # 论文中提到激活函数 φ1 和 φ2 均为 SiLU
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

        # 相对位置偏置 (rab)，类似于 T5 的位置偏置实现
        self.rel_pos_bias = torch.nn.Embedding(2 * max_seq_len - 1, self.num_heads)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch, seq_len, hidden_units]。假定输入已经经过了归一化 (Pre-LN)。
            attn_mask (torch.Tensor, optional): 注意力掩码，形状为 [batch, seq_len, seq_len]。默认为 None。
        """
        batch_size, seq_len, _ = x.shape
        # --- 1. 逐点投影 (Pointwise Projection)，对应 f1 和 φ1 ---
        projected = self.f1_linear(x)
        # 在分割前应用 φ1 (SiLU) 激活函数
        activated = self.activation(projected)

        # 分割成 U, Q, K, V 四个部分
        U, Q_proj, K_proj, V_proj = torch.chunk(activated, 4, dim=-1)

        # 为多头注意力重塑 Q, K, V 的形状
        Q = Q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # --- 2. 空间聚合 (Spatial Aggregation)，即修改后的注意力机制 ---
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # 添加相对位置偏置 (rab)
        positions = torch.arange(seq_len, device=x.device).view(-1, 1) - torch.arange(seq_len, device=x.device).view(
            1, -1
        )
        rel_pos_indices = positions + self.max_seq_len - 1
        # 为了支持“扩长序列”（长度可能大于 max_seq_len），这里对索引进行裁剪，避免越界
        rel_pos_indices = torch.clamp(rel_pos_indices, min=0, max=2 * self.max_seq_len - 2)
        rel_bias = self.rel_pos_bias(rel_pos_indices).permute(2, 0, 1).unsqueeze(0)
        scores += rel_bias

        # 先激活
        attn_weights = self.activation(scores)
        # 再应用注意力掩码
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1).logical_not(), 0.0)

        # 应用 φ2 (SiLU) 激活函数，替代 Softmax

        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # --- 3. 逐点变换 (Pointwise Transformation) ---
        # 使用 U 进行门控 (逐元素相乘)，然后通过 f2 进行最终投影
        # 论文公式为 f2(Norm(attn_output) * U)，由于我们采用 Pre-LN 架构，直接应用门控
        gated_output = attn_output * U
        final_output = self.f2_linear(gated_output)

        return final_output


class BaseSortMLP(torch.nn.Module):
    def __init__(self, seq_dim, item_dim, dropout_rate, num_heads, max_seq_len, args):
        super(BaseSortMLP, self).__init__()
        self.seq_dim = seq_dim
        self.item_dim = item_dim

        self.attention = FlashMultiHeadAttention(
            hidden_units=seq_dim, num_heads=args.num_heads,
            dropout_rate=args.dropout_rate, max_seq_len=args.maxlen + 1, ).to(args.device)
        self.layer_norm = torch.nn.RMSNorm(seq_dim, eps=1e-8).to(
            args.device) if args.rms_norm else torch.nn.LayerNorm(seq_dim, eps=1e-8).to(args.device)
        self.fwd_layer = PointWiseFeedForward(seq_dim, args.dropout_rate, args.feed_forward_hidden_units).to(
            args.device)

        # 2. MLP层，用于对拼接后的特征进行打分
        mlp_input_dim = seq_dim + item_dim

        self.mlp_layers = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(mlp_input_dim // 2, mlp_input_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(mlp_input_dim // 4, 1)
        )

    def forward(self, seq_embs, item_embs, attn_mask=None):
        attention_out, _ = self.attention(item_embs, seq_embs, seq_embs, attn_mask)
        attention_out = self.layer_norm(attention_out)
        attention_out = self.fwd_layer(attention_out)

        mlp_input = torch.cat([attention_out, item_embs], dim=-1)
        batch_size, seq_len, _ = mlp_input.size()

        mlp_input_flat = mlp_input.view(-1, mlp_input.size(-1))

        output = self.mlp_layers(mlp_input_flat)
        return output.view(batch_size, seq_len, -1)


class SortMLP(torch.nn.Module):
    def __init__(self, hidden_units, dnn_hidden_units, dropout_rate, num_heads, max_seq_len, args, num_classes=1):
        super(SortMLP, self).__init__()
        # base_model 输出的 embedding 维度是 hidden_units * dnn_hidden_units
        seq_dim = hidden_units * dnn_hidden_units
        item_dim = hidden_units * dnn_hidden_units

        self.click_mlp = BaseSortMLP(seq_dim, item_dim, dropout_rate, num_heads, max_seq_len, args)

    def forward(self, seq_embs, item_embs, attn_mask=None):
        return self.click_mlp(seq_embs, item_embs, attn_mask)


class EnhancedSortMLP(torch.nn.Module):
    def __init__(self, hidden_units, dnn_hidden_units, dropout_rate, num_heads, max_seq_len, args, num_classes=1):
        super(EnhancedSortMLP, self).__init__()
        # base_model 输出的 embedding 维度是 hidden_units * dnn_hidden_units
        seq_dim = hidden_units * dnn_hidden_units
        item_dim = hidden_units * dnn_hidden_units

        # 增强特征维度：ANN score + SID1 prob + SID2 prob
        enhanced_feat_dim = 1

        # 将增强特征投影到与embedding相同的维度
        self.enhanced_feat_projection = torch.nn.Linear(enhanced_feat_dim, seq_dim).to(args.device)

        # MODIFIED: 使用 SidRewardHSTUBlock 替换注意力和前馈层
        self.hstu_block = SidRewardHSTUBlock(
            hidden_units=seq_dim, num_heads=args.num_heads,
            dropout_rate=args.dropout_rate, max_seq_len=args.maxlen + 1,
        ).to(args.device)
        self.layer_norm = torch.nn.RMSNorm(seq_dim, eps=1e-8).to(
            args.device) if args.rms_norm else torch.nn.LayerNorm(seq_dim, eps=1e-8).to(args.device)

        # 最终MLP层：拼接后的特征 + 增强特征
        enhanced_input_dim = seq_dim + item_dim + enhanced_feat_dim

        self.mlp_layers = torch.nn.Sequential(
            torch.nn.Linear(enhanced_input_dim, enhanced_input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(enhanced_input_dim // 2, enhanced_input_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(enhanced_input_dim // 4, 1)
        )

    def forward(self, seq_embs, item_embs, attn_mask=None, ann_scores=None, sid1_probs=None, sid2_probs=None):
        """
        增强版forward，将ANN score和SID概率特征融入到注意力机制中

        Args:
            seq_embs: 序列embedding
            item_embs: 物品embedding
            attn_mask: 注意力掩码
            ann_scores: ANN相似度分数 [batch_size, seq_len]
            sid1_probs: SID1 softmax概率 [batch_size, seq_len]
            sid2_probs: SID2 softmax概率 [batch_size, seq_len]
        """
        batch_size, seq_len, _ = seq_embs.shape

        # 1. 准备增强特征
        enhanced_features = []

        if ann_scores is not None:
            enhanced_features.append(ann_scores.unsqueeze(-1))  # [batch_size, seq_len, 1]
        else:
            enhanced_features.append(torch.zeros(batch_size, seq_len, 1, device=seq_embs.device))

        # if sid1_probs is not None:
        #     enhanced_features.append(sid1_probs.unsqueeze(-1))  # [batch_size, seq_len, 1]
        # else:
        #     enhanced_features.append(torch.zeros(batch_size, seq_len, 1, device=seq_embs.device))

        # if sid2_probs is not None:
        #     enhanced_features.append(sid2_probs.unsqueeze(-1))  # [batch_size, seq_len, 1]
        # else:
        #     enhanced_features.append(torch.zeros(batch_size, seq_len, 1, device=seq_embs.device))

        # 拼接增强特征 [batch_size, seq_len, 3]
        enhanced_feats = torch.cat(enhanced_features, dim=-1)

        # 2. 将增强特征投影到embedding维度
        enhanced_feats_projected = self.enhanced_feat_projection(enhanced_feats)  # [batch_size, seq_len, seq_dim]

        # 3. 将增强特征融入到序列embedding中
        enhanced_seq_embs = seq_embs + enhanced_feats_projected  # 残差连接

        # 4. MODIFIED: 调用 HSTU block
        # HSTU 模块内部处理了注意力和类前馈网络的操作
        # 假设这里采用 Post-LN 架构，与原代码块的结构保持一致
        attention_out = self.hstu_block(item_embs, enhanced_seq_embs, enhanced_seq_embs, attn_mask)
        attention_out = self.layer_norm(attention_out)

        # 5. 拼接原始特征和增强特征用于最终MLP
        mlp_input = torch.cat([attention_out, item_embs, enhanced_feats], dim=-1)

        # 6. 通过MLP层
        mlp_input_flat = mlp_input.view(-1, mlp_input.size(-1))
        output = self.mlp_layers(mlp_input_flat)

        # [MODIFIED] 更改输出为 p_ctr (0, 1)
        output = torch.sigmoid(output)
        # output = (output-0.5)*2 # <-- [REMOVED]
        return output.view(batch_size, seq_len, -1)


# class EnhancedSortMLP(torch.nn.Module):
#     def __init__(self, hidden_units, dnn_hidden_units, dropout_rate, num_heads, max_seq_len, args, num_classes=1):
#         super(EnhancedSortMLP, self).__init__()
#         # base_model 输出的 embedding 维度是 hidden_units * dnn_hidden_units
#         seq_dim = hidden_units * dnn_hidden_units
#         item_dim = hidden_units * dnn_hidden_units

#         # 增强特征维度：ANN score + SID1 prob + SID2 prob
#         enhanced_feat_dim = 3

#         # 将增强特征投影到与embedding相同的维度
#         self.enhanced_feat_projection = torch.nn.Linear(enhanced_feat_dim, seq_dim).to(args.device)

#         # MODIFIED: 使用 SidRewardHSTUBlock 替换注意力和前馈层
#         self.hstu_block = SidRewardHSTUBlock(
#             hidden_units=seq_dim, num_heads=args.num_heads,
#             dropout_rate=args.dropout_rate, max_seq_len=args.maxlen + 1,
#         ).to(args.device)
#         self.layer_norm = torch.nn.RMSNorm(seq_dim, eps=1e-8).to(
#             args.device) if args.rms_norm else torch.nn.LayerNorm(seq_dim, eps=1e-8).to(args.device)

#         # 最终MLP层：拼接后的特征 + 增强特征
#         # enhanced_input_dim = seq_dim + item_dim + enhanced_feat_dim
#         enhanced_input_dim = seq_dim + item_dim + 1
#         self.mlp_layers = torch.nn.Sequential(
#             torch.nn.Linear(enhanced_input_dim, enhanced_input_dim // 2),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout_rate),
#             torch.nn.Linear(enhanced_input_dim // 2, enhanced_input_dim // 4),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout_rate),
#             torch.nn.Linear(enhanced_input_dim // 4, 1),
#             torch.nn.Sigmoid()
#         )

#     def forward(self, seq_embs, item_embs, attn_mask=None, ann_scores=None, sid1_probs=None, sid2_probs=None):
#         """
#         增强版forward，将ANN score和SID概率特征融入到注意力机制中

#         Args:
#             seq_embs: 序列embedding   b, s, hidden
#             item_embs: 物品embedding  b, s, hidden
#             attn_mask: 注意力掩码
#             ann_scores: ANN相似度分数 [batch_size, seq_len]  b, s
#             sid1_probs: SID1 softmax概率 [batch_size, seq_len]
#             sid2_probs: SID2 softmax概率 [batch_size, seq_len]
#         """
#         batch_size, seq_len, _ = seq_embs.shape

#         # 1. 准备增强特征
#         # enhanced_features = []

#         # if ann_scores is not None:
#         #     enhanced_features.append(ann_scores.unsqueeze(-1))  # [batch_size, seq_len, 1]
#         # else:
#         #     enhanced_features.append(torch.zeros(batch_size, seq_len, 1, device=seq_embs.device))

#         # if sid1_probs is not None:
#         #     enhanced_features.append(sid1_probs.unsqueeze(-1))  # [batch_size, seq_len, 1]
#         # else:
#         #     enhanced_features.append(torch.zeros(batch_size, seq_len, 1, device=seq_embs.device))

#         # if sid2_probs is not None:
#         #     enhanced_features.append(sid2_probs.unsqueeze(-1))  # [batch_size, seq_len, 1]
#         # else:
#         #     enhanced_features.append(torch.zeros(batch_size, seq_len, 1, device=seq_embs.device))

#         # # 拼接增强特征 [batch_size, seq_len, 3]
#         # enhanced_feats = torch.cat(enhanced_features, dim=-1)

#         # # 2. 将增强特征投影到embedding维度
#         # enhanced_feats_projected = self.enhanced_feat_projection(enhanced_feats)  # [batch_size, seq_len, seq_dim]

#         # # # 3. 将增强特征融入到序列embedding中
#         # enhanced_seq_embs = seq_embs + enhanced_feats_projected  # 残差连接

#         # # 4. MODIFIED: 调用 HSTU block
#         # # HSTU 模块内部处理了注意力和类前馈网络的操作
#         # # 假设这里采用 Post-LN 架构，与原代码块的结构保持一致
#         # # attention_out = self.hstu_block(item_embs, enhanced_seq_embs, enhanced_seq_embs, attn_mask)
#         # # attention_out = self.layer_norm(attention_out)

#         # # 5. 拼接原始特征和增强特征用于最终MLP
#         # mlp_input = torch.cat([enhanced_seq_embs, item_embs, enhanced_feats], dim=-1)

#         # # 6. 通过MLP层
#         # mlp_input_flat = mlp_input.view(-1, mlp_input.size(-1))
#         if ann_scores is None:
#             # 如果 ann_scores 未提供, 使用 0 作为占位符，以保证模型兼容性
#             ann_scores_feat = torch.zeros(batch_size, seq_len, 1, device=seq_embs.device, dtype=seq_embs.dtype)
#         else:
#             # 将 ann_scores 的维度从 [batch_size, seq_len] 转换为 [batch_size, seq_len, 1]
#             # 以便能和 embedding 张量进行拼接
#             ann_scores_feat = ann_scores.unsqueeze(-1)

#         mlp_input = torch.cat([seq_embs, item_embs, ann_scores_feat], dim=-1)
#         output = self.mlp_layers(mlp_input)

#         # 2. DEBUG: 直接使用seq_embs和item_embs的内积
#         # output = (seq_embs * item_embs).sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
#         # output = ann_scores_feat
#         return output.view(batch_size, seq_len, -1)


ACTION_TYPE_NUM = 3


class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.rms_norm = args.rms_norm
        self.sparse_embedding = args.sparse_embedding
        self.rope = args.rope
        self.use_hstu = args.use_hstu
        self.mm_emb_gate = args.mm_emb_gate
        self.random_perturbation = args.random_perturbation
        self.random_perturbation_value = args.random_perturbation_value
        self.mode = args.mode

        # MoE 相关参数
        self.use_moe = args.use_moe
        self.moe_num_experts = args.moe_num_experts
        self.moe_top_k = args.moe_top_k
        self.moe_intermediate_size = args.moe_intermediate_size
        self.moe_load_balancing_alpha = args.moe_load_balancing_alpha
        self.moe_load_balancing_update_freq = args.moe_load_balancing_update_freq
        self.hidden_units = args.hidden_units * args.dnn_hidden_units # 保存 hidden_units 用于 beam search

        hidden_dim = args.hidden_units * args.dnn_hidden_units

        if args.learnable_temp:
            self.learnable_temp = torch.nn.Parameter(torch.tensor(args.infonce_temp))

        self.item_emb = torch.nn.Embedding(self.item_num + 1, 32, padding_idx=0, sparse=self.sparse_embedding)
        self.item_hash_prime_a = 2000003
        self.item_hash_prime_b = 3000017
        self.item_hash_emb_a = torch.nn.Embedding(self.item_hash_prime_a + 1, args.hash_emb_size, padding_idx=0, sparse=self.sparse_embedding)
        self.item_hash_emb_b = torch.nn.Embedding(self.item_hash_prime_b + 1, args.hash_emb_size, padding_idx=0, sparse=self.sparse_embedding)
        self.next_action_type_emb = torch.nn.Embedding(ACTION_TYPE_NUM + 1, hidden_dim, padding_idx=0, sparse=self.sparse_embedding).to(self.dev)

        if not self.rope and not self.use_hstu:
            self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, hidden_dim, padding_idx=0, sparse=self.sparse_embedding)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        # 初始化 MoE 配置
        if self.use_moe:
            self.moe_config = MoEConfig(args)
            print(f"模型信息：正在使用 MoE 模块，专家数量: {self.moe_num_experts}, Top-K: {self.moe_top_k}")
            self.moe_blocks = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        if self.mm_emb_gate:
            self.gate_item_feature_types = ["100", "101", "112", "114", "115", "116", "117", "118", "119", "120"]
            self.mm_emb_count = sum(self.ITEM_EMB_FEAT.values())
            self.mm_emb_gate_unit = torch.nn.Linear(
                self.mm_emb_count + args.hidden_units * (len(self.gate_item_feature_types)),
                len(self.ITEM_EMB_FEAT) + 1)
            self.output_scores = []

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + len(self.USER_ARRAY_FEAT)) + len(self.USER_CONTINUAL_FEAT)
        itemdim = (
                32 + args.hash_emb_size + args.hash_emb_size
                + args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + len(self.ITEM_ARRAY_FEAT))
                + len(self.ITEM_CONTINUAL_FEAT)
        )
        itemdim += args.hidden_units * len(self.ITEM_EMB_FEAT) if not self.mm_emb_gate else 0

        self.userdnn = torch.nn.Linear(userdim, hidden_dim)
        self.itemdnn = torch.nn.Linear(itemdim, hidden_dim)

        self.last_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)
        self.num_blocks = args.num_blocks

        # 构建 Backbone Layers
        for _ in range(args.num_blocks):
            if self.use_moe:
                self.moe_blocks.append(DeepseekMoEBlock(args, self.moe_config))
            # 注意：此处移除了对非 MoE 架构的初始化支持，因为 Prompt 要求只使用 MoE
            # 如果需要兼容旧代码，可保留 else 分支，但 predict_beam_search 将强制使用 MoE

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0, sparse=self.sparse_embedding)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0, sparse=self.sparse_embedding)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0, sparse=self.sparse_embedding)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0, sparse=self.sparse_embedding)
        for k in self.ITEM_EMB_FEAT:
            if self.mm_emb_gate:
                self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], itemdim)
            else:
                self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

        self.similarity_function = args.similarity_function
        self.sid = False
        if args.sid:
            self.sid = True
            self.sid_embedding = torch.nn.Embedding(args.sid_codebook_size + 1, hidden_dim * 2, padding_idx=0, sparse=self.sparse_embedding)
            self.sid_token_proj = torch.nn.Linear(hidden_dim * 2, hidden_dim)

            # 辅助 HSTU 块 (用于 SID 预测头，保持不变)
            self.sid1_hstu_block = SidRewardHSTUBlock(
                hidden_units=hidden_dim, num_heads=args.num_heads,
                dropout_rate=args.dropout_rate, max_seq_len=self.maxlen + 1,
            )
            self.sid1_layer_norm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8).to(self.dev)
            self.sid1_output_projection = torch.nn.Linear(hidden_dim, args.sid_codebook_size + 1)

            self.sid2_hstu_block_list = torch.nn.ModuleList([
                SidRewardHSTUBlock(
                    hidden_units=hidden_dim, num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate, max_seq_len=self.maxlen + 1,
                ) for _ in range(self.num_blocks)
            ])
            self.sid2_layer_norm_list = torch.nn.ModuleList([
                (torch.nn.RMSNorm(hidden_dim, eps=1e-8) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)).to(self.dev)
                for _ in range(self.num_blocks)
            ])
            self.sid2_output_projection = torch.nn.Linear(hidden_dim, args.sid_codebook_size + 1)
            self.sid2_query_projection = torch.nn.Linear(3 * hidden_dim, hidden_dim)

            # label -> next sid1, next sid1 -> next sid2
            self.label_to_next_sid1 = torch.nn.Linear(hidden_dim, args.sid_codebook_size + 1)
            self.next_sid1_to_next_sid2 = torch.nn.Linear(hidden_dim * 2, args.sid_codebook_size + 1)

        self.item_to_label_head = torch.nn.Linear(hidden_dim, 3)

    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_SPARSE_FEAT.update({k: feat_statistics[k] for k in feat_types['context_item_sparse']})
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    def feat2tensor(self, seq_feature, k):
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = 0;
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        if include_user:
            item_mask = (mask == 1).to(self.dev)
            seq = seq.to(self.dev)

            base_ids = (item_mask * seq)
            item_embedding = self.item_emb(base_ids)
            # Prime-hash indices and lookups
            hash_a_ids = (base_ids % self.item_hash_prime_a)
            hash_b_ids = (base_ids % self.item_hash_prime_b)
            item_hash_emb_a = self.item_hash_emb_a(hash_a_ids)
            item_hash_emb_b = self.item_hash_emb_b(hash_b_ids)

            item_feat_list = [item_embedding, item_hash_emb_a, item_hash_emb_b]
            user_feat_list = []
        else:
            seq = seq.to(self.dev)
            base_ids = seq
            item_embedding = self.item_emb(base_ids)
            # Prime-hash indices and lookups
            hash_a_ids = (base_ids % self.item_hash_prime_a)
            hash_b_ids = (base_ids % self.item_hash_prime_b)
            item_hash_emb_a = self.item_hash_emb_a(hash_a_ids)
            item_hash_emb_b = self.item_hash_emb_b(hash_b_ids)

            item_feat_list = [item_embedding, item_hash_emb_a, item_hash_emb_b]

        if self.mm_emb_gate: mm_emb_feat_list = []

        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]
        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
            ])

        for feat_dict, feat_type, feat_list in all_feat_types:
            # Handle cases where feat_dict might be None or a list
            current_features = feat_dict.keys() if isinstance(feat_dict, dict) else (feat_dict or [])

            for k in current_features:
                tensor_feature = feature_array[k].to(self.dev)
                if feat_type.endswith('sparse'):
                    emb = self.sparse_emb[k](tensor_feature)
                    feat_list.append(emb)
                    if self.mm_emb_gate and k in self.gate_item_feature_types: mm_emb_feat_list.append(emb)
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    # Ensure continual float features match model dtype (e.g., bf16)
                    desired_dtype = self.item_emb.weight.dtype
                    feat_list.append(tensor_feature.to(desired_dtype).unsqueeze(2))

        if not self.mm_emb_gate:
            for k in self.ITEM_EMB_FEAT:
                # Cast input to the Linear weight dtype to avoid matmul dtype mismatch
                x = feature_array[k].to(self.dev)
                x = x.to(self.emb_transform[k].weight.dtype)
                item_feat_list.append(self.emb_transform[k](x))
        else:
            mm_emb_list = []
            for k in self.ITEM_EMB_FEAT:
                # Ensure bf16 dtype path consistency for multimodal/gated features
                raw = feature_array[k].to(self.dev)
                desired_dtype = self.item_emb.weight.dtype
                raw = raw.to(desired_dtype)
                mm_emb_feat_list.append(raw)
                mm_emb_list.append(self.emb_transform[k](raw.unsqueeze(2)))
            all_mm_emb = torch.cat(mm_emb_feat_list, dim=2)
            batchsize, maxlen, mm_emb_shape = all_mm_emb.shape
            # Match gate unit input dtype to its weights
            all_mm_emb = all_mm_emb.to(self.mm_emb_gate_unit.weight.dtype)
            output_score = F.softmax(self.mm_emb_gate_unit(all_mm_emb.view(batchsize * maxlen, mm_emb_shape)),
                                     dim=-1).view(batchsize, maxlen, -1)
            self.output_scores.append(output_score)

        all_item_emb = torch.cat(item_feat_list, dim=2)
        if self.mm_emb_gate:
            all_emb_list = [all_item_emb.unsqueeze(2)] + mm_emb_list
            mm_emb_feat_list = torch.cat(all_emb_list, dim=2)
            all_item_emb = torch.sum(output_score.unsqueeze(-1) * mm_emb_feat_list, dim=2, keepdim=True).squeeze(2)

        if self.random_perturbation and self.mode == "train":
            all_item_emb += (all_item_emb != 0) * (
                    torch.rand_like(all_item_emb) - 0.5) * 2 * self.random_perturbation_value
        all_item_emb = self.itemdnn(all_item_emb)
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = torch.relu(all_item_emb) + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature, next_action_type=None, infer=False):
        batch_size, maxlen = log_seqs.shape
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        if not self.rope and not self.use_hstu:
            poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).detach()
            poss = self.pos_emb(poss * (log_seqs != 0))
            seqs += poss
        seqs = self.emb_dropout(seqs)

        attention_mask_tril = torch.tril(torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev))
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
        attention_mask_infer = attention_mask_pad.unsqueeze(1)
        # user, sid1, sid2, item feat
        # 初始化变量
        mlp_pos_embs = seqs
        mlp_logfeats = seqs
        sid_logfeats = seqs
        all_seq_logfeats = []

        # 强制使用 MoE 路径
        if self.use_moe:
            for i, block in enumerate(self.moe_blocks):
                seqs, topk_idx, aux_loss = block(seqs, attn_mask=attention_mask)
                if aux_loss is not None and self.training:
                    if not hasattr(self, '_moe_aux_losses'):
                        self._moe_aux_losses = []
                    self._moe_aux_losses.append(aux_loss)
                # 收集每一层的输出用于 SID level 2 attention
                all_seq_logfeats.append(seqs)
                if i == self.num_blocks - 1:
                    sid_logfeats = seqs
        else:
             # 如果配置错误未开启 MoE，但类初始化时已移除非 MoE 代码，这里做个兜底
             pass

        log_feats = self.last_layernorm(seqs)
        sid_logfeats = log_feats

        if infer:
            return log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats, attention_mask_infer
        else:
            # 【修改】训练模式也返回 6 个值，与 forward 解包保持一致
            return log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats

    def forward(self, user_item, pos_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, sid, pos_log_p,
                ranking_loss_mask, args=None, dataset=None):
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        # 【修改】这里解包 6 个返回值
        log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats = self.log2feats(
            user_item, mask, seq_feature, next_action_type)
        loss_mask = (next_mask == 1).to(self.dev)

        sid_level_1_logits, sid_level_2_logits = None, None

        if self.similarity_function == 'cosine':
            pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
            log_feats_normalized = F.normalize(log_feats, p=2, dim=-1)
        else:
            pos_embs_normalized, log_feats_normalized = pos_embs, log_feats

        return (
            log_feats_normalized, loss_mask, pos_embs_normalized,
            attention_mask, mlp_logfeats, sid_logfeats,
            sid_level_1_logits, sid_level_2_logits, None,
            pos_embs, mlp_pos_embs  # 【修改】确保返回正确的 mlp_pos_embs
        )

    def forward_infer(self, user_item, pos_seqs, mask, next_mask, next_action_type,
                      seq_feature, pos_feature, sid, pos_log_p, ranking_loss_mask, args=None, dataset=None):
        """
        推理时的前向传播，包含指标计算
        """
        # 调用核心逻辑获取模型输出
        (
            seq_embs, loss_mask, pos_embs,
            causal_mask, mlp_logfeats, sid_logfeats,
            sid_level_1_logits, sid_level_2_logits, all_scores, _, mlp_pos_embs
        ) = self.forward(
            user_item, pos_seqs, mask, next_mask, next_action_type,
            seq_feature, pos_feature, sid, pos_log_p, ranking_loss_mask,
            args, dataset
        )

        # 计算指标
        metrics = self._calculate_metrics_infer(
            seq_embs, loss_mask, pos_embs, pos_seqs,
            next_action_type, sid_level_1_logits, sid_level_2_logits,
            all_scores, sid, sid_logfeats, pos_log_p, args, dataset, causal_mask, mlp_logfeats, ranking_loss_mask,
            mlp_pos_embs
        )

        return metrics

    def _calculate_loss(self, user_item, pos_seqs, mask, next_mask, next_action_type,
                       seq_feature, pos_feature, sid, pos_log_p,
                       args, dataset=None):
        """
        全新的 Loss 计算逻辑。
        不依赖 forward 的输出，而是独立构造扩长序列：User -> SID1 -> SID2 -> Item Feat -> Label
        """
        loss = torch.tensor(0.0, device=self.dev)
        loss_dict = {}

        # 1. 基础维度与类型准备
        # 【修改】使用 user_item 获取全序列，并计算有效 item 长度 (去掉 User 位)
        # user_item shape: [B, L+1] (idx 0 is user)
        batch_size = user_item.shape[0]
        # 我们只关注 Item 部分的长度
        seq_len = user_item.shape[1] - 1

        common_dtype = self.item_emb.weight.dtype
        hidden_dim = self.hidden_units

        # 2. 生成各部分的 Embedding
        # 2.1 User Embedding (位于序列起点)
        # 调用 feat2emb 获取包含 User 的序列，取第一个位置即为 User 表征
        full_seq_emb = self.feat2emb(user_item, seq_feature, mask=mask, include_user=True).to(common_dtype)
        user_token = full_seq_emb[:, 0, :]  # [B, D]

        # 2.2 Item Feature Embedding (对应 Item Feat 位置)
        item_feat_tokens = full_seq_emb[:, 1:, :] # [B, S, D]

        # 2.3 SID Embeddings (对应 SID1, SID2 位置)
        if args.sid and sid is not None:
            # sid shape: [B, L+1, 2] -> Slice to [B, S, 2]
            current_sid = sid[:, 1:, :]
            sid1_raw = self.sid_embedding(current_sid[:, :seq_len, 0].long())  # [B, S, 2D]
            sid2_raw = self.sid_embedding(current_sid[:, :seq_len, 1].long())  # [B, S, 2D]
            sid1_tokens = self.sid_token_proj(sid1_raw).to(common_dtype)  # [B, S, D]
            sid2_tokens = self.sid_token_proj(sid2_raw).to(common_dtype)  # [B, S, D]
        else:
            raise Exception("没有SID！")

        # 2.4 Action/Label Embeddings (对应 Label 位置)
        # next_action_type 记录了每一步的真实行为 (0,1,2...)
        current_next_action_type = next_action_type[:, 1:]
        action_tokens = self.next_action_type_emb(current_next_action_type[:, :seq_len].to(self.dev)).to(
            common_dtype)  # [B, S, D]

        # 3. 构造扩长序列 (Extended Sequence)
        # 序列模式: [User, (SID1_0, SID2_0, Feat_0, Act_0), (SID1_1, ...), ...]
        # 总长度 = 1 (User) + 4 * seq_len
        extended_len = 1 + 4 * seq_len

        extended_seq = torch.zeros(batch_size, extended_len, hidden_dim, device=self.dev, dtype=common_dtype)
        extended_mask = torch.zeros(batch_size, extended_len, dtype=torch.bool, device=self.dev)

        # 3.1 填充 User
        extended_seq[:, 0, :] = user_token
        extended_mask[:, 0] = True  # User 始终有效

        # 3.2 计算索引位置
        idx_range = torch.arange(seq_len, device=self.dev)
        sid1_pos = 1 + 4 * idx_range  # 1, 5, 9...
        sid2_pos = sid1_pos + 1  # 2, 6, 10...
        feat_pos = sid1_pos + 2  # 3, 7, 11...
        act_pos = sid1_pos + 3  # 4, 8, 12...

        # 3.3 填充 Item 相关的 Token
        extended_seq[:, sid1_pos, :] = sid1_tokens
        extended_seq[:, sid2_pos, :] = sid2_tokens
        extended_seq[:, feat_pos, :] = item_feat_tokens
        extended_seq[:, act_pos, :] = action_tokens

        # 3.4 填充 Mask
        # 原始 mask: [B, S], 1 表示有效 item
        item_valid = (mask[:, 1:seq_len+1] == 1).to(self.dev)  # [B, S]

        extended_mask[:, sid1_pos] = item_valid
        extended_mask[:, sid2_pos] = item_valid
        extended_mask[:, feat_pos] = item_valid
        extended_mask[:, act_pos] = item_valid

        # 4. 经过 Backbone (MoE)
        # 4.1 构造因果掩码 (Causal Mask)
        # 保证每个位置只能看到自己和之前的 token
        tril_mask = torch.tril(torch.ones((extended_len, extended_len), dtype=torch.bool, device=self.dev))
        # 结合 Padding Mask: [B, 1, L] & [1, L, L] -> [B, L, L]
        attn_mask = tril_mask.unsqueeze(0) & extended_mask.unsqueeze(1)

        # 4.2 Forward Pass
        x = extended_seq
        if self.use_moe:
            for block in self.moe_blocks:
                x, meta_outputs, _ = block(x, attn_mask=attn_mask)
                topk_idx, aux_loss = meta_outputs
                # 收集负载均衡 Loss
                if aux_loss is not None:
                    print(aux_loss)
                    loss += aux_loss
                    loss_dict['moe_aux_loss'] = loss_dict.get('moe_aux_loss', 0.0) + aux_loss.item()
        else:
            # 兼容非 MoE 逻辑 (如果需要)
            pass

        # 经过最终的 Norm
        hidden_states = self.last_layernorm(x)

        # 5. 计算损失
        # 提取各个位置的输出向量
        # out_user = hidden_states[:, 0, :] # User 位置的输出，用于预测第一个 SID1

        out_sid1 = hidden_states[:, sid1_pos, :]  # [B, S, D] -> 预测 SID2
        out_sid2 = hidden_states[:, sid2_pos, :]  # [B, S, D] -> 预测 Feat (隐式) 或作为 Feat 的上下文
        out_feat = hidden_states[:, feat_pos, :]  # [B, S, D] -> 预测 Label (CTR)
        out_act = hidden_states[:, act_pos, :]  # [B, S, D] -> 预测 Next SID1

        # 5.1 任务一: CTR 预测 (Feat -> Label)
        # 使用 Item Feature 位置的输出预测当前的 Action Type
        ctr_logits = self.item_to_label_head(out_feat)  # [B, S, 3]

        ctr_valid_mask = item_valid.view(-1)
        if ctr_valid_mask.any():
            valid_ctr_logits = ctr_logits.view(-1, 3)[ctr_valid_mask]
            valid_ctr_labels = next_action_type[:, 1:seq_len+1].contiguous().view(-1)[ctr_valid_mask].long()

            ctr_loss = F.cross_entropy(valid_ctr_logits, valid_ctr_labels)
            loss += ctr_loss
            loss_dict['ctr_bce'] = ctr_loss.item()
            with torch.no_grad():
                # 1. 转 CPU numpy
                probs = torch.softmax(valid_ctr_logits, dim=-1).detach().cpu().numpy()  # [N, 3]
                labels = valid_ctr_labels.detach().cpu().numpy()  # [N]

                # --- 计算 AUC Type 1 (点击 vs 曝光) ---
                # 只看 Label 为 0(曝光) 和 1(点击) 的样本
                mask_1 = (labels == 0) | (labels == 1)
                if mask_1.any():
                    y_true_1 = (labels[mask_1] == 1).astype(int)
                    y_score_1 = probs[mask_1, 1]  # 取 Class 1 的概率

                    # 只有当同时存在正负样本时才能计算
                    if len(np.unique(y_true_1)) > 1:
                        loss_dict['auc_type1'] = roc_auc_score(y_true_1, y_score_1)

                # --- 计算 AUC Type 2 (转化 vs 曝光) ---
                # 只看 Label 为 0(曝光) 和 2(转化) 的样本
                mask_2 = (labels == 0) | (labels == 2)
                if mask_2.any():
                    y_true_2 = (labels[mask_2] == 2).astype(int)
                    y_score_2 = probs[mask_2, 2]  # 取 Class 2 的概率

                    if len(np.unique(y_true_2)) > 1:
                        loss_dict['auc_type2'] = roc_auc_score(y_true_2, y_score_2)



        if args.sid and sid is not None:
            sid1_labels = sid[:, 1:seq_len+1, 0].long()  # [B, S]
            sid2_labels = sid[:, 1:seq_len+1, 1].long()  # [B, S]

            # 5.2 任务二: SID1 预测 (Context -> SID1)
            # 逻辑:
            # 第 1 个 SID1 由 User Token 预测
            # 后续 SID1 由前一个 Item 的 Action Token 预测

            # 构造预测源张量: [User, Act_0, Act_1, ..., Act_{S-2}] -> 预测 [SID1_0, SID1_1, ..., SID1_{S-1}]
            # 拼接 User 输出和 Action 输出的前 S-1 个
            if seq_len > 0:
                # user_out: [B, 1, D] -> 对应预测 sid1_labels[:, 0]
                user_out = hidden_states[:, 0, :].unsqueeze(1)

                if seq_len > 1:
                    # act_out_prev: [B, S-1, D] -> 对应预测 sid1_labels[:, 1:]
                    act_out_prev = out_act[:, :-1, :]
                    pred_source_sid1 = torch.cat([user_out, act_out_prev], dim=1)  # [B, S, D]
                else:
                    pred_source_sid1 = user_out

                # 使用 label_to_next_sid1 投影层
                sid1_logits = self.label_to_next_sid1(pred_source_sid1)  # [B, S, V]

                sid1_valid_mask = item_valid & (sid1_labels > 0)
                if sid1_valid_mask.any():
                    valid_sid1_logits = sid1_logits[sid1_valid_mask]
                    valid_sid1_labels = sid1_labels[sid1_valid_mask]

                    sid1_loss = F.cross_entropy(valid_sid1_logits, valid_sid1_labels)
                    loss += sid1_loss
                    loss_dict['sid1'] = sid1_loss.item()

                    # 统计 Hit@10
                    with torch.no_grad():
                        top10 = torch.topk(valid_sid1_logits, k=min(10, valid_sid1_logits.size(-1)), dim=-1).indices
                        hits = (top10 == valid_sid1_labels.unsqueeze(-1)).any(dim=-1).float()
                        loss_dict['SID/Top10HitRate1'] = hits.mean().item()

            # 5.3 任务三: SID2 预测 (SID1 -> SID2)
            # 使用 SID1 位置的输出预测当前的 SID2
            sid2_logits = self.sid2_output_projection(out_sid1)  # [B, S, V]

            sid2_valid_mask = item_valid & (sid2_labels > 0)
            if sid2_valid_mask.any():
                valid_sid2_logits = sid2_logits[sid2_valid_mask]
                valid_sid2_labels = sid2_labels[sid2_valid_mask]

                sid2_loss = F.cross_entropy(valid_sid2_logits, valid_sid2_labels)
                loss += sid2_loss
                loss_dict['sid2'] = sid2_loss.item()

                # 统计 Hit@10
                with torch.no_grad():
                    top10_2 = torch.topk(valid_sid2_logits, k=min(10, valid_sid2_logits.size(-1)), dim=-1).indices
                    hits_2 = (top10_2 == valid_sid2_labels.unsqueeze(-1)).any(dim=-1).float()
                    loss_dict['SID/Top10HitRate2'] = hits_2.mean().item()

        loss_dict['total'] = loss.item()
        return loss, loss_dict

    def _calculate_metrics_infer(self, seq_embs, loss_mask, pos_embs, pos,
                                 next_action_type, sid_level_1_logits, sid_level_2_logits,
                                 mlp_output, sid, sid_logfeats_detach, pos_log_p, args, dataset, causal_mask,
                                 mlp_logfeats, ranking_loss_mask, mlp_pos_embs):
        """
        在推理阶段计算指标，避免显存问题
        """
        metrics = {}

        # 计算损失
        loss, loss_dict = self._calculate_loss(
            seq_embs, loss_mask, pos_embs, next_action_type,
            sid_level_1_logits, sid_level_2_logits, mlp_output, sid, pos_log_p, mlp_logfeats, causal_mask, args,
            ranking_loss_mask, mlp_pos_embs
        )

        # 将损失添加到指标中
        for key, value in loss_dict.items():
            metrics[f'loss_{key}'] = value

        # 计算基础指标
        pos_logits = torch.sum(seq_embs * pos_embs, dim=-1) * loss_mask
        pos_sim = similarity(pos_logits)

        metrics['pos_sim'] = pos_sim.item()

        # 构建候选池（简化版本，避免显存问题）
        hidden_unit = pos_embs.shape[-1]
        mask_flat = loss_mask.view(-1).bool()

        # 只使用当前batch的pos和neg，不进行复杂的去重操作
        active_pos_embs = pos_embs.view(-1, hidden_unit)[mask_flat]
        active_pos_ids = pos.view(-1)[mask_flat]

        # 简单的候选池构建
        candidate_embs = torch.cat([active_pos_embs], dim=0)
        candidate_ids = torch.cat([active_pos_ids], dim=0)

        # 限制候选池大小以避免显存问题
        max_candidates = candidate_embs.shape[0]  # 限制最大候选数
        if candidate_embs.shape[0] > max_candidates:
            indices = torch.randperm(candidate_embs.shape[0], device=candidate_embs.device)[:max_candidates]
            candidate_embs = candidate_embs[indices]
            candidate_ids = candidate_ids[indices]

        # 计算Top-K指标
        query_emb = seq_embs[next_action_type != 0]
        labels = pos[next_action_type != 0]

        if len(query_emb) > 0 and len(candidate_embs) > 0:
            scores = torch.matmul(query_emb, candidate_embs.T)
            _, topk_indices = torch.topk(scores, k=min(10, candidate_embs.shape[0]), dim=1)
            topk_items = candidate_ids[topk_indices]

            # 计算HR@10和NDCG@10
            hr10 = calculate_hitrate(topk_items, labels)
            ndcg10 = calculate_ndcg(topk_items, labels)
            score = 0.31 * hr10 + 0.69 * ndcg10

            metrics['hr10'] = hr10
            metrics['ndcg10'] = ndcg10
            metrics['score'] = score

        # 计算Last Step指标
        eval_mask_last_step = (next_action_type[:, -1] != 0)
        if eval_mask_last_step.sum() > 0 and len(candidate_embs) > 0:
            query_emb_last = seq_embs[:, -1, :][eval_mask_last_step]
            labels_last = pos[:, -1][eval_mask_last_step]

            scores_last = torch.matmul(query_emb_last, candidate_embs.T)
            _, topk_indices_last = torch.topk(scores_last, k=min(10, candidate_embs.shape[0]), dim=1)
            topk_items_last = candidate_ids[topk_indices_last]

            hr10_last = calculate_hitrate(topk_items_last, labels_last)
            ndcg10_last = calculate_ndcg(topk_items_last, labels_last)
            score_last = 0.31 * hr10_last + 0.69 * ndcg10_last

            metrics['hr10_last'] = hr10_last
            metrics['ndcg10_last'] = ndcg10_last
            metrics['score_last'] = score_last

        # SID重排指标（如果启用）

        # if args.sid:
        # sid_hr, score_diff, hr_diff, ndcg_diff = calculate_score_sid(
        # self, sid_logfeats_detach, seq_embs, pos_embs, pos, next_action_type,
        # dataset, loss_mask, self.dev
        # )
        # metrics["sid_hr"] = sid_hr
        # metrics["score_diff"] = score_diff
        # metrics["hr_diff"] = hr_diff
        # metrics["ndcg_diff"] = ndcg_diff

        return metrics

    def forward_train(
            self, user_item, pos_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, sid, pos_log_p,
            ranking_loss_mask, args, dataset=None
    ):
        """
        训练时的前向传播，包含损失计算和指标计算
        """

        # 计算损失函数
        loss, loss_dict = self._calculate_loss(
            user_item, pos_seqs, mask, next_mask, next_action_type,
            seq_feature, pos_feature, sid, pos_log_p,
            args, dataset
        )

        # 构建训练日志字典
        log_dict = {}
        log_dict['Sid1Loss/train'] = loss_dict.get('sid1', 0.0)
        log_dict['Sid2Loss/train'] = loss_dict.get('sid2', 0.0)
        log_dict['CTR_BCE_Loss/train'] = loss_dict.get('ctr_bce', 0.0)
        log_dict['NextSid1Loss/train'] = loss_dict.get('next_sid1', 0.0)
        log_dict['NextSid2Loss/train'] = loss_dict.get('next_sid2', 0.0)
        log_dict['MoE_AuxLoss/train'] = loss_dict.get('moe_aux_loss', 0.0)
        log_dict['Loss/train'] = loss_dict['total']
        log_dict['SID/Top10HitRate1'] = loss_dict.get('SID/Top10HitRate1', 0.0)
        log_dict['SID/Top10HitRate2'] = loss_dict.get('SID/Top10HitRate2', 0.0)
        # InfoNCE 温度已不再使用ctr_bce
        log_dict['ctr_bce/train'] = loss_dict.get('ctr_bce', 0.0)
        log_dict["auc_type1"] = loss_dict.get("auc_type1",0.0)
        log_dict["auc_type2"] = loss_dict.get("auc_type2", 0.0)
        # 添加MoE指标到log_dict
        if self.use_moe:
            moe_metrics = self._collect_moe_metrics()
            for key, value in moe_metrics.items():
                if isinstance(value, (int, float)):
                    log_dict[key] = value

        # 只在 cuda:0 上每 50 步计算一次指标
        if not hasattr(self, '_metric_counter'):
            self._metric_counter = 0

        if str(self.dev) == 'cuda:0':
            self._metric_counter += 1

        return loss, log_dict

    def predict_sid(self, log_feats, sid1_list, attention_mask=None):
        """
        专门用于验证/预测阶段的SID预测，采用自回归方式，避免数据泄漏。
        """
        # 确保在无梯度的环境下执行
        with torch.no_grad():
            # 1. MODIFIED: 使用 HSTU block 预测第一层SID的分布
            sid1_attn_output = self.sid1_hstu_block.infer(log_feats[:, -1:, :], log_feats, log_feats)
            sid1_attn_output = self.sid1_layer_norm(sid1_attn_output)
            sid_level_1_logits = self.sid1_output_projection(sid1_attn_output)

            # 2. 找出预测的最可能的第一层SID
            # (batch_size, seq_len)
            # 用ann出来的32位真实sid1，来查第一步出来的sid1embedding，softmax之后的结果相乘出top
            # 真实的sid1[b,32,d] [log_feats[:,-1:,:]
            sid1_embedding = self.sid_embedding(sid1_list)

            # 3. 使用预测出的s1的embedding来预测第二层SID的分布
            # 将sid1_embedding和log_feats进行concat，然后通过投影层映射回原始维度（与训练模式保持一致）
            log_feats_expanded = log_feats[:, -1:, :].expand_as(sid1_embedding)
            sid2_q_concat = torch.cat([sid1_embedding, log_feats_expanded], dim=-1)
            sid2_q = self.sid2_query_projection(sid2_q_concat)

            # MODIFIED: 使用 HSTU block
            # sid2的key/value从sid1的输出中获取
            sid2_attn_output = self.sid2_hstu_block.infer(sid2_q, sid1_attn_output, sid1_attn_output)
            sid2_attn_output = self.sid2_layer_norm(sid2_attn_output)
            sid_level_2_logits = self.sid2_output_projection(sid2_attn_output)
            sid_level_1_prob = F.softmax(sid_level_1_logits, dim=-1)
            sid_level_2_prob = F.softmax(sid_level_2_logits, dim=-1)
        return sid_level_1_prob, sid_level_2_prob  # [B,1,D], [B,32,D]

    def beamsearch_sid(self, log_feats, all_seq_logfeats, attention_mask_infer, top_k, top_k_2):
        """
        使用 Beam Search 预测两层的 SID (显存优化版).

        SID1 的预测使用完整的 log_feats 作为上下文。
        SID2 的预测仅使用 SID1 的注意力输出作为上下文，以节省显存。

        Args:
            log_feats (torch.Tensor): 输入的特征，形状为 (B, S, D)，其中 B 是批次大小，S 是序列长度，D 是特征维度。
            top_k (int): Beam Search 的宽度 (beam size)。

        Returns:
            tuple:
                - top_sequences (torch.Tensor): 最终预测出的 top_k 个 SID 序列，形状为 (B, top_k, 2)。
                - top_scores (torch.Tensor): 对应序列的最终对数概率分数，形状为 (B, top_k)。
        """
        with torch.no_grad():
            B, S, D = log_feats.shape
            vocab_size = self.sid_embedding.num_embeddings

            # 1. 预测第一位SID
            sid1_attn_output = self.sid1_hstu_block.infer(log_feats[:, -1:, :], log_feats, log_feats)
            sid1_attn_output = self.sid1_layer_norm(sid1_attn_output)
            sid_level_1_logits = self.sid1_output_projection(sid1_attn_output).squeeze(1)

            log_probs_1 = F.log_softmax(sid_level_1_logits, dim=-1)
            top_scores_1, top_indices_1 = torch.topk(log_probs_1, top_k, dim=-1)

            # 2. 基于第一位的top_k结果预测第二位SID
            sid1_embeddings = self.sid_embedding(top_indices_1)  # (B, top_k, D_emb)
            log_feats_last_step = all_seq_logfeats[0][:, -1:, :]  # (B, 1, D)
            expanded_log_feats = log_feats_last_step.expand(-1, top_k, -1)  # (B, top_k, D)

            # 拼接sid嵌入和log_feats
            sid_q_concat = torch.cat([sid1_embeddings, expanded_log_feats], dim=-1)
            sid_q_projected = self.sid2_query_projection(sid_q_concat)  # (B, top_k, D_query)
            query = sid_q_projected.view(B * top_k, 1, -1)

            # === 多层 cross-attention：使用 all_seq_logfeats 的每一层作为 KV ===
            sid2_attn_output = query
            # 对应的 KV padding 掩码同样扩展到 beam 维度：[B, 1, S] -> [B, top_k, 1, S] -> [B*top_k, 1, S]
            expanded_mask = attention_mask_infer.unsqueeze(1).expand(-1, top_k, -1, -1).reshape(B * top_k, 1, -1)
            for i in range(len(all_seq_logfeats)):
                # 使用 all_seq_logfeats[i] 作为 KV，expand 到 top_k 个 beam
                kv_feats = all_seq_logfeats[i]  # (B, S, D)
                expanded_kv = kv_feats.unsqueeze(1).expand(-1, top_k, -1, -1)  # (B, top_k, S, D)
                expanded_kv = expanded_kv.reshape(B * top_k, kv_feats.shape[1], kv_feats.shape[2])  # (B*top_k, S, D)
                # 对应的 KV padding 掩码同样扩展到 beam 维度：[B, 1, S] -> [B, top_k, 1, S] -> [B*top_k, 1, S]
                # expanded_mask = attention_mask_infer.unsqueeze(1).expand(-1, top_k, -1, -1).reshape(B * top_k, 1, kv_feats.shape[1])

                # Pre-norm 模式：先对 query 做 LayerNorm，然后做 cross-attention，最后残差连接
                sid2_attn_output_norm = self.sid2_layer_norm_list[i](sid2_attn_output)
                attn_output = self.sid2_hstu_block_list[i].infer(sid2_attn_output_norm, expanded_kv, expanded_kv,
                                                                 expanded_mask)
                sid2_attn_output = sid2_attn_output + attn_output  # 残差连接：原始 query + attention 输出

            sid_level_2_logits = self.sid2_output_projection(sid2_attn_output).squeeze(1)

            log_probs_2 = F.log_softmax(sid_level_2_logits, dim=-1)

            # 2.4. 合并分数并筛选 (逻辑不变)
            expanded_scores_1 = top_scores_1.view(B * top_k, 1)
            total_scores = expanded_scores_1 + log_probs_2
            total_scores = total_scores.view(B, top_k * vocab_size)
            top_scores, final_indices = torch.topk(total_scores, top_k_2, dim=-1)

            # 2.5. 回溯路径，构建最终序列 (逻辑不变)
            beam_indices = torch.div(final_indices, vocab_size, rounding_mode='floor')
            token_indices_2 = final_indices % vocab_size
            token_indices_1 = torch.gather(top_indices_1, 1, beam_indices)
            top_sequences = torch.stack([token_indices_1, token_indices_2], dim=2)

        return top_sequences, top_scores
    def predict_beam_search(self, user_item, pos_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, sid,
                            args, dataset):
        """
        OneRec 风格的 Beam Search 推理，遵循 SID1 -> SID2 的预测范式。
        [FIXED] 增加了显式的内存连续性保证，防止 MoE 模块出现 Illegal Memory Access。
        [OPTIMIZED] 引入 KV Cache 加速 Step 2 的推理过程。
        """
        batch_size = pos_seqs.shape[0]
        device = self.dev
        common_dtype = self.item_emb.weight.dtype
        # seq_len 是历史序列长度
        seq_len = pos_seqs.shape[1] - 1

        beam_size = args.beam_search_beam_size

        # ===========================
        # 1. 构建初始扩展序列 (History Context)
        # ===========================
        # 1.1 获取 User Token
        seqs_emb = self.feat2emb(user_item, seq_feature, mask=mask, include_user=True)
        user_token = seqs_emb[:, 0, :].to(common_dtype)  # [B, D]

        # 1.2 获取 History Tokens
        pos_embs = seqs_emb[:,1:,:].to(common_dtype)
        act_toks = self.next_action_type_emb(next_action_type[:, 1:].to(device)).to(common_dtype)  # Action Tokens

        if self.sid and sid is not None:
            sid1_raw = self.sid_embedding(sid[:, 1:, 0].long())
            sid2_raw = self.sid_embedding(sid[:, 1:, 1].long())
            sid1_toks = self.sid_token_proj(sid1_raw).to(common_dtype)
            sid2_toks = self.sid_token_proj(sid2_raw).to(common_dtype)
        else:
            raise ValueError("SID information is required for SID-based Beam Search.")

        # 1.3 拼装 Extended Sequence
        extended_len = 1 + 4 * seq_len
        extended_seq = torch.zeros(batch_size, extended_len, self.hidden_units, device=device, dtype=common_dtype)
        extended_mask = torch.zeros(batch_size, extended_len, dtype=torch.bool, device=device)

        # 填充 User Token (Pos 0)
        extended_seq[:, 0, :] = user_token
        extended_mask[:, 0] = True

        # 填充 History Tokens
        idx = torch.arange(seq_len, device=device)
        sid1_pos = 1 + 4 * idx
        sid2_pos = sid1_pos + 1
        feat_pos = sid1_pos + 2
        act_pos = sid1_pos + 3

        # 使用原始 mask
        item_valid = (mask[:, :seq_len] == 1).to(device)

        extended_seq[:, sid1_pos, :] = sid1_toks
        extended_seq[:, sid2_pos, :] = sid2_toks
        extended_seq[:, feat_pos, :] = pos_embs
        extended_seq[:, act_pos, :] = act_toks
        extended_mask[:, sid1_pos] = item_valid
        extended_mask[:, sid2_pos] = item_valid
        extended_mask[:, feat_pos] = item_valid
        extended_mask[:, act_pos] = item_valid

        # ===========================
        # 2. Beam Search 初始化
        # ===========================

        # 初始化 beam 状态
        accum_scores = torch.zeros(batch_size, beam_size, device=device)
        if beam_size > 1:
            accum_scores[:, 1:] = float('-inf')  # 除第一个 beam 外，其余初始化为负无穷
        accum_scores = accum_scores.view(-1)

        # 扩展序列及 Mask 初始化 (复制 beam_size 份)
        # [CRITICAL FIX] 确保初始 expand 后的 tensor 是连续的
        current_seqs = extended_seq.unsqueeze(1).expand(-1, beam_size, -1, -1).contiguous().view(batch_size * beam_size,
                                                                                                 extended_len, -1)
        current_masks = extended_mask.unsqueeze(1).expand(-1, beam_size, -1).contiguous().view(batch_size * beam_size,
                                                                                               extended_len)

        # 记录生成的 SIDs: [B, beam_size, 2]
        generated_sids = torch.zeros(batch_size, beam_size, 2, dtype=torch.long, device=device)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, beam_size)

        # ===========================
        # 定义支持 KV Cache 的 Backbone Forward
        # ===========================
        def run_backbone_moe(seqs, masks, past_key_values=None):
            """
            :param seqs: [B_total, L_curr, D]
            :param masks: [B_total, L_total] (注意这里是完整的mask)
            :param past_key_values: List[Tuple[Tensor, Tensor]], 对应每一层的 KV Cache
            :return: output [B_total, L_curr, D], present_key_values
            """
            # [CRITICAL FIX] 确保输入 Tensor 绝对连续
            x = seqs.contiguous()

            batch_curr, seq_len_curr, _ = x.shape
            # masks 应该是完整的长度，即 past_len + seq_len_curr
            total_len = masks.shape[1]

            # 构造 Attention Mask
            if past_key_values is None:
                # First Step (Prompt Phase): Causal Mask
                # tril: [L, L]
                tril = torch.tril(torch.ones((seq_len_curr, seq_len_curr), dtype=torch.bool, device=device))
                # masks.unsqueeze(1): [B, 1, L]
                # 结果: [B, L, L]
                attn_mask_ext = tril.unsqueeze(0) & masks.unsqueeze(1)
            else:
                # Generation Phase (Incremental):
                # Query Length = 1 (usually), Key Length = Total Length
                # seqs 只包含新的 token
                # Mask 需要允许当前 token 关注所有之前的有效 tokens
                # masks: [B, Total_L] -> unsqueeze -> [B, 1, Total_L]
                attn_mask_ext = masks.unsqueeze(1)
                # 注意：这里不需要 tril，因为 Query 只有一个 token，自然只能看到所有的 Past Keys + Self

            present_key_values = []

            # 逐层经过 MoE Blocks
            for i, block in enumerate(self.moe_blocks):
                # 获取当前层的 cache
                layer_past = past_key_values[i] if past_key_values is not None else None

                # [ASSUMPTION] 这里的 block 必须支持 past_key_value 参数并返回元组
                # x: [B, L_curr, D]
                # new_kv: Tuple[Tensor, Tensor]
                x, _, new_kv = block(x, attn_mask=attn_mask_ext, past_key_value=layer_past)

                present_key_values.append(new_kv)

            out = self.last_layernorm(x)
            return out, present_key_values

        # ===========================
        # Step 1: Predict SID1 (Prompt Phase)
        # ===========================

        # 运行 Backbone (无 Cache，建立 Cache)
        # output: [B*beam, L, D]
        backbone_out, past_key_values = run_backbone_moe(current_seqs, current_masks, past_key_values=None)

        # 获取最后一个 token 的输出
        act_hidden = backbone_out[:, -1, :]  # [B*beam, D]

        # 预测 SID1
        sid1_logits = self.label_to_next_sid1(act_hidden)  # [B*beam, vocab]
        sid1_log_probs = F.log_softmax(sid1_logits, dim=-1)

        # Top-K 选择
        sid1_log_probs = sid1_log_probs.view(batch_size, beam_size, -1)
        candidate_scores = sid1_log_probs + accum_scores.view(batch_size, beam_size, 1)

        candidate_scores = candidate_scores.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(candidate_scores, beam_size, dim=-1)

        vocab_size = sid1_logits.shape[-1]
        prev_beam_indices = torch.div(topk_indices, vocab_size, rounding_mode='floor')  # [B, beam]
        sid1_tokens = topk_indices % vocab_size

        accum_scores = topk_scores.view(-1)
        generated_sids = generated_sids[batch_indices, prev_beam_indices]
        generated_sids[:, :, 0] = sid1_tokens

        # ===========================
        # KV Cache Reordering (Critical for Beam Search)
        # ===========================
        # 计算展平后的索引 [B*beam]
        flat_prev_beam = (prev_beam_indices + batch_indices * beam_size).view(-1)

        # 重排 KV Cache
        # past_key_values 是 List[Tuple[K, V]]
        # K, V shape 通常是 [B*beam, Num_Heads, Seq_Len, Head_Dim]
        reordered_past_key_values = []
        for layer_past in past_key_values:
            layer_k, layer_v = layer_past
            # [CRITICAL FIX] index_select 后必须 contiguous，否则后续 MoE 计算会报错
            new_k = layer_k.index_select(0, flat_prev_beam).contiguous()
            new_v = layer_v.index_select(0, flat_prev_beam).contiguous()
            reordered_past_key_values.append((new_k, new_v))

        past_key_values = reordered_past_key_values

        # 更新 Mask (只需要 append 新的一列 1)
        # 注意：Current Seqs 不需要 append 了，因为 Step 2 只输入增量 token
        # 但 Mask 必须是全局长度，用于 Attention
        # [B*beam, 1]
        new_mask_col = torch.ones(batch_size * beam_size, 1, dtype=torch.bool, device=device)
        # current_masks 之前对应的是 Step 1 选中的那些 beam 的 mask，虽然都是 True，但逻辑上需要重排
        current_masks = current_masks.index_select(0, flat_prev_beam).contiguous()
        current_masks = torch.cat([current_masks, new_mask_col], dim=1).contiguous()

        # ===========================
        # Step 2: Predict SID2 (Generation Phase)
        # ===========================

        # 准备 SID1 的 Embedding 作为 Step 2 的输入
        sid1_emb_raw = self.sid_embedding(sid1_tokens.view(-1))
        # [B*beam, 1, D]
        sid1_emb_proj = self.sid_token_proj(sid1_emb_raw).to(common_dtype).unsqueeze(1).contiguous()

        # 运行 Backbone (使用 Cache，只输入增量 token)
        # output: [B*beam, 1, D]
        backbone_out, _ = run_backbone_moe(sid1_emb_proj, current_masks, past_key_values=past_key_values)

        # 此时 backbone_out 长度为 1，直接取 squeeze
        sid1_hidden = backbone_out.squeeze(1)  # [B*beam, D]

        sid2_logits = self.sid2_output_projection(sid1_hidden)
        sid2_log_probs = F.log_softmax(sid2_logits, dim=-1)

        sid2_log_probs = sid2_log_probs.view(batch_size, beam_size, -1)
        candidate_scores = sid2_log_probs + accum_scores.view(batch_size, beam_size, 1)

        candidate_scores = candidate_scores.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(candidate_scores, beam_size, dim=-1)

        prev_beam_indices = torch.div(topk_indices, vocab_size, rounding_mode='floor')
        sid2_tokens = topk_indices % vocab_size

        accum_scores = topk_scores.view(-1)
        # Step 2 的 beam 来源于 Step 1 生成的结果，不需要再次重排 generated_sids 的第一列
        # 因为 generated_sids 目前存的就是 Step 1 胜出的结果
        # 但这里还是需要根据 Step 2 的 prev_beam_indices 来选择最终保留哪条路径
        generated_sids = generated_sids[batch_indices, prev_beam_indices]
        generated_sids[:, :, 1] = sid2_tokens

        return generated_sids, accum_scores.view(batch_size, beam_size)

    def predict(self, log_seqs, seq_feature, mask, next_action_type=None):
        log_feats, attention_mask, mlp_logits, _, _, _ = self.log2feats(log_seqs, mask, seq_feature, next_action_type,
                                                                        infer=True)
        final_feat = log_feats[:, -1, :]
        mlp_logits_last = mlp_logits[:, -1, :]
        if self.similarity_function == 'cosine':
            final_feat = F.normalize(final_feat, p=2, dim=-1)
        return final_feat, mlp_logits_last, attention_mask

    def predict(self, log_seqs, seq_feature, mask, next_action_type=None):  # next_action_type 在推理时可以为 None
        log_feats, attention_mask, mlp_logits, _, _, _, _ = self.log2feats(log_seqs, mask, seq_feature, True)
        final_feat = log_feats[:, -1, :]
        mlp_logits_last = mlp_logits[:, -1, :]
        if self.similarity_function == 'cosine':
            final_feat = F.normalize(final_feat, p=2, dim=-1)

        # 返回最终的用户向量、完整的序列特征和注意力掩码
        return final_feat, mlp_logits_last, attention_mask

    def set_mode(self, mode):
        self.mode = mode

    def _collect_moe_metrics(self):
        """收集MoE相关指标"""
        moe_metrics = {}

        if self.use_moe and hasattr(self, 'hstu_layers'):
            for i, layer in enumerate(self.hstu_layers):
                stats = layer.get_moe_statistics()
                if stats:
                    # 为每个层添加前缀
                    for key, value in stats.items():
                        moe_metrics[f'MoE_Layer{i}_{key}'] = value
        return moe_metrics


class MLPScorer(torch.nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.2):
        super(MLPScorer, self).__init__()
        self.input_dim = hidden_size
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_size // 2), torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4), torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(hidden_size // 4, hidden_size // 8), torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(hidden_size // 8, 1)
        )

    def forward(self, x):
        return self.dnn(x)