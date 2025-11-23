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


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()

        # dim: 最后一个维度的维度，即 self.head_dim
        # max_seq_len: 模型能处理的最大序列长度

        # 计算 theta = 1 / (base^(2i/d))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预先计算好 cos 和 sin 的值
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # freqs 的形状是 (max_seq_len, dim/2)

        # 将 freqs 扩展为 (max_seq_len, dim) 以便后续计算
        emb = torch.cat((freqs, freqs), dim=-1)

        # register_buffer 将这些张量注册为模块的 buffer
        # 它们是模块状态的一部分，会被 .to(device) 移动，但不会被视为模型参数
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        # x 的形状: [bs, num_heads, seq_len, head_dim]
        seq_len = x.shape[2]

        # 从缓存中获取对应长度的 cos 和 sin
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]

        # 将 x 的特征维度两两一组，进行旋转
        # x_even: 偶数索引的特征 (x0, x2, x4, ...)
        # x_odd: 奇数索引的特征 (x1, x3, x5, ...)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 创建一个与 x_even 形状相同，但值为 x_odd 乘以 -1 的张量
        x_rotated = torch.cat((-x_odd, x_even), dim=-1)

        # 应用旋转公式: x' = x * cos(m*theta) + rotate(x) * sin(m*theta)
        return x * cos + x_rotated * sin


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, rope=False, max_seq_len=101):
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
        self.max_seq_len = max_seq_len
        if rope:
            self.rope_unit = RotaryEmbedding(self.head_dim, max_seq_len)
            # 相对位置范围是 [-(max_seq_len-1), max_seq_len-1]，共 2*max_seq_len - 1 个值
            # 每个相对位置对应一个 num_heads 维的偏置向量，每个头一个偏置值
            self.rel_pos_bias = torch.nn.Embedding(2 * max_seq_len - 1, self.num_heads)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            Q = self.rope_unit(Q)
            K = self.rope_unit(K)
            # 创建相对位置索引矩阵
            positions = torch.arange(seq_len, device=query.device).view(-1, 1) - torch.arange(seq_len,
                                                                                              device=query.device).view(
                1, -1)
            # 偏移索引，使其从0开始，作为Embedding的输入
            rel_pos_indices = positions + self.max_seq_len - 1

            # 从Embedding表中查找偏置值
            # shape: [seq_len, seq_len, num_heads]
            rel_bias = self.rel_pos_bias(rel_pos_indices)
            # 调整形状以匹配注意力分数矩阵 [batch_size, num_heads, seq_len, seq_len]
            # [seq_len, seq_len, num_heads] -> [num_heads, seq_len, seq_len] -> [1, num_heads, seq_len, seq_len]
            rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            final_attn_mask = attn_mask
            if self.rope:
                # 将布尔掩码转换为浮点数掩码，False 的位置为 -inf
                float_mask = torch.zeros_like(attn_mask, dtype=torch.float, device=query.device)
                float_mask.masked_fill_(attn_mask.logical_not(), float('-inf'))

                # 将 rel_bias 加到掩码上，PyTorch 会将其直接加到注意力分数上
                # 广播机制: [1, num_heads, seq_len, seq_len] + [batch_size, 1, seq_len, seq_len]
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
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        attn_output = attn_output.clamp(min=-30, max=30)
        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None

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
            positions = (kv_seq_len - 1) * torch.ones(q_seq_len, dtype=torch.long, device=query.device).view(-1,
                                                                                                             1) - torch.arange(
                kv_seq_len, dtype=torch.long, device=query.device).view(1, -1)
        else:
            positions = torch.arange(q_seq_len, device=query.device).view(-1, 1) - torch.arange(kv_seq_len,
                                                                                                device=query.device).view(
                1, -1)
        rel_pos_indices = positions + self.max_seq_len - 1
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
        positions = torch.arange(seq_len, device=x.device).view(-1, 1) - torch.arange(seq_len, device=x.device).view(1,
                                                                                                                     -1)
        rel_pos_indices = positions + self.max_seq_len - 1
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
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        # 在多GPU训练中，device应该指向当前模型所在的设备
        # 如果是SimpleDataParallel，会在初始化后被更新为正确的设备
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

        # MoE相关参数
        self.use_moe = args.use_moe
        self.moe_num_experts = args.moe_num_experts
        self.moe_top_k = args.moe_top_k
        self.moe_intermediate_size = args.moe_intermediate_size
        self.moe_load_balancing_alpha = args.moe_load_balancing_alpha
        self.moe_load_balancing_update_freq = args.moe_load_balancing_update_freq

        hidden_dim = args.hidden_units * args.dnn_hidden_units

        if args.learnable_temp:
            self.learnable_temp = torch.nn.Parameter(torch.tensor(args.infonce_temp))

        # Item id embedding reduced to 32-dim
        self.item_emb = torch.nn.Embedding(self.item_num + 1, 32, padding_idx=0,
                                           sparse=self.sparse_embedding)
        self.item_emb = self.item_emb
        # Additional two prime-hash item embeddings (200w-level and 300w-level), 256-dim each
        self.item_hash_prime_a = 2000003  # ~2M prime
        self.item_hash_prime_b = 3000017  # ~3M prime
        self.item_hash_emb_a = torch.nn.Embedding(self.item_hash_prime_a + 1, args.hash_emb_size, padding_idx=0,
                                                  sparse=self.sparse_embedding)
        self.item_hash_emb_b = torch.nn.Embedding(self.item_hash_prime_b + 1, args.hash_emb_size, padding_idx=0,
                                                  sparse=self.sparse_embedding)
        self.next_action_type_emb = torch.nn.Embedding(ACTION_TYPE_NUM + 1, args.hidden_units * args.dnn_hidden_units,
                                                       padding_idx=0, sparse=self.sparse_embedding).to(
            self.dev)

        if not self.rope and not self.use_hstu:
            self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, hidden_dim,
                                              padding_idx=0, sparse=self.sparse_embedding)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        # 初始化MoE配置
        if self.use_moe:
            from deepseek_moe import MoEConfig, log_moe_statistics
            self.moe_config = MoEConfig(args)
            print(f"模型信息：正在使用 MoE 模块，专家数量: {self.moe_num_experts}, Top-K: {self.moe_top_k}")
            log_moe_statistics(args, self.moe_config)

        if not self.use_hstu:
            self.attention_layernorms = torch.nn.ModuleList()
            self.attention_layers = torch.nn.ModuleList()
            self.forward_layernorms = torch.nn.ModuleList()
            self.forward_layers = torch.nn.ModuleList()
        else:
            print("模型信息：正在使用 HSTU 模块替代标准的 Transformer 模块。")
            self.hstu_layernorms = torch.nn.ModuleList()
            self.hstu_layers = torch.nn.ModuleList()
            self.append_layernorms = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        if self.mm_emb_gate:
            self.gate_item_feature_types = ["100", "101", "112", "114", "115", "116", "117", "118", "119", "120"]
            self.gate_exclude_num = torch.nn.Parameter(torch.tensor(1.0))
            self.mm_emb_count = sum(self.ITEM_EMB_FEAT.values())
            self.mm_emb_gate_unit = torch.nn.Linear(
                self.mm_emb_count + args.hidden_units * (len(self.gate_item_feature_types)),
                len(self.ITEM_EMB_FEAT) + 1)
            self.output_scores = []

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        # Build item feature input dim:
        # - item id emb: 32
        # - two hash embs: 256 + 256
        # - item sparse feats: each args.hidden_units
        # - item array feats: each args.hidden_units (summed on last dim)
        # - item continual feats: each scalar
        itemdim = (
                32  # item id emb
                + args.hash_emb_size + args.hash_emb_size  # two hash emb dims
                + args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + len(self.ITEM_ARRAY_FEAT))
                + len(self.ITEM_CONTINUAL_FEAT)
        )
        itemdim += args.hidden_units * len(self.ITEM_EMB_FEAT) if not self.mm_emb_gate else 0

        self.userdnn = torch.nn.Linear(userdim, hidden_dim)
        self.itemdnn = torch.nn.Linear(itemdim, hidden_dim)

        self.last_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if self.rms_norm else torch.nn.LayerNorm(
            hidden_dim, eps=1e-8)
        self.num_blocks = args.num_blocks
        for _ in range(args.num_blocks):
            if not self.use_hstu:
                new_attn_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8).to(
                    self.dev) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)
                self.attention_layernorms.append(new_attn_layernorm)
                new_attn_layer = FlashMultiHeadAttention(hidden_dim, args.num_heads, args.dropout_rate, rope=self.rope,
                                                         max_seq_len=self.maxlen + 1)
                self.attention_layers.append(new_attn_layer)
                new_fwd_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8).to(
                    self.dev) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)
                self.forward_layernorms.append(new_fwd_layernorm)
                new_fwd_layer = PointWiseFeedForward(hidden_dim, args.dropout_rate, args.feed_forward_hidden_units).to(
                    self.dev)
                self.forward_layers.append(new_fwd_layer)
            else:
                new_hstu_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8).to(
                    self.dev) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)
                self.hstu_layernorms.append(new_hstu_layernorm)

                # 为每一层创建独立的 append LayerNorm
                new_append_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8).to(
                    self.dev) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)
                self.append_layernorms.append(new_append_layernorm)

                # 根据是否使用MoE选择不同的HSTU实现
                if self.use_moe:
                    from deepseek_moe import MoEHSTUBlock
                    hstu_block = MoEHSTUBlock(args, self.moe_config)
                else:
                    hstu_block = HSTUBlock(hidden_units=hidden_dim, num_heads=args.num_heads,
                                           dropout_rate=args.dropout_rate, max_seq_len=self.maxlen + 1, ).to(
                        self.dev)
                self.hstu_layers.append(hstu_block)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0,
                                                    sparse=self.sparse_embedding)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0,
                                                    sparse=self.sparse_embedding)
            self.sparse_emb[k] = self.sparse_emb[k]
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0,
                                                    sparse=self.sparse_embedding)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0,
                                                    sparse=self.sparse_embedding)
        for k in self.ITEM_EMB_FEAT:
            if self.mm_emb_gate:
                self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], itemdim)
            else:
                self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

        self.similarity_function = args.similarity_function

        self.reward = False
        if args.reward:
            self.reward = True
            # MODIFIED: 使用增强版SortMLP，该模型内部已更新为使用 SidRewardHSTUBlock
            self.reward_model = EnhancedSortMLP(args.hidden_units, args.dnn_hidden_units, args.dropout_rate,
                                                args.num_heads,
                                                args.maxlen, args).to(args.device)
            # [MODIFIED] 将损失函数更改为 BCELoss
            self.mlp_bce_loss = torch.nn.BCEWithLogitsLoss()

        self.item_attention_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8).to(
            self.dev) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)
        self.item_attention_layer = FlashMultiHeadAttention(
            hidden_units=hidden_dim, num_heads=args.num_heads,
            dropout_rate=args.dropout_rate, max_seq_len=self.maxlen + 1, )

        self.item_attention_fwd_layernorm = torch.nn.RMSNorm(hidden_dim, eps=1e-8).to(
            self.dev) if self.rms_norm else torch.nn.LayerNorm(hidden_dim, eps=1e-8)
        self.item_fwd_layer = PointWiseFeedForward(hidden_dim, args.dropout_rate, args.feed_forward_hidden_units).to(
            self.dev)
        self.sid = False
        if args.sid:
            self.sid = True
            # 增大SID embedding维度到 hidden_dim * 2
            self.sid_embedding = torch.nn.Embedding(args.sid_codebook_size + 1, hidden_dim * 2, padding_idx=0,
                                                    sparse=self.sparse_embedding)

            # MODIFIED: 使用 SidRewardHSTUBlock 替换注意力和前馈层
            self.sid1_hstu_block = SidRewardHSTUBlock(
                hidden_units=hidden_dim, num_heads=args.num_heads,
                dropout_rate=args.dropout_rate, max_seq_len=self.maxlen + 1,
            )
            self.sid1_layer_norm = torch.nn.RMSNorm(hidden_dim, eps=1e-8) if self.rms_norm else torch.nn.LayerNorm(
                hidden_dim, eps=1e-8).to(self.dev)
            self.sid1_output_projection = torch.nn.Linear(hidden_dim, args.sid_codebook_size + 1)

            # 多层 cross-attention：为每层准备独立的 HSTU block 和 LayerNorm
            self.sid2_hstu_block_list = torch.nn.ModuleList([
                SidRewardHSTUBlock(
                    hidden_units=hidden_dim, num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate, max_seq_len=self.maxlen + 1,
                ) for _ in range(self.num_blocks)
            ])
            self.sid2_layer_norm_list = torch.nn.ModuleList([
                (torch.nn.RMSNorm(hidden_dim, eps=1e-8) if self.rms_norm else torch.nn.LayerNorm(hidden_dim,
                                                                                                 eps=1e-8)).to(self.dev)
                for _ in range(self.num_blocks)
            ])
            self.sid2_output_projection = torch.nn.Linear(hidden_dim, args.sid_codebook_size + 1)

            # 添加投影层，将concat后的query (3 * hidden_dim) 映射回原始维度 (hidden_dim)
            # SID embedding: hidden_dim * 2, sid_logfeats: hidden_dim, 拼接后: hidden_dim * 3
            self.sid2_query_projection = torch.nn.Linear(3 * hidden_dim, hidden_dim)

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

    def log2feats(self, log_seqs, mask, seq_feature, infer=False):
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
        # 推理用mask：query长度为1，仅保留KV侧padding掩码，形状 [B, 1, S]
        attention_mask_infer = attention_mask_pad.unsqueeze(1)

        # Initialize variables to avoid UnboundLocalError
        mlp_pos_embs = seqs
        mlp_logfeats = seqs
        sid_logfeats = seqs

        all_seq_logfeats = []

        for i in range(len(self.attention_layers if not self.use_hstu else self.hstu_layers)):
            if not self.use_hstu:
                if self.norm_first:
                    x = self.attention_layernorms[i](seqs)
                    mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                    seqs = seqs + mha_outputs
                    seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
                else:
                    mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                    seqs = self.attention_layernorms[i](seqs + mha_outputs)
                    seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
            else:
                x_norm = self.hstu_layernorms[i](seqs)

                # 处理MoE HSTU的返回值
                if self.use_moe:
                    hstu_output, topk_idx, aux_loss = self.hstu_layers[i](x_norm, attn_mask=attention_mask)
                    # 可以在这里记录MoE的统计信息
                    if hasattr(self, '_moe_stats'):
                        self._moe_stats.append(topk_idx)
                    # 收集辅助损失
                    if aux_loss is not None:
                        if not hasattr(self, '_moe_aux_losses'):
                            self._moe_aux_losses = []
                        self._moe_aux_losses.append(aux_loss)
                else:
                    hstu_output = self.hstu_layers[i](x_norm, attn_mask=attention_mask)

                seqs = seqs + hstu_output
                if i == 1:
                    mlp_logfeats = seqs
                if i == self.num_blocks - 1:
                    sid_logfeats = seqs

                all_seq_logfeats.append(self.append_layernorms[i](seqs))

        log_feats = self.last_layernorm(seqs)
        sid_logfeats = log_feats
        if infer:
            return log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats, attention_mask_infer
        else:
            return log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats
    def forward(
            self, user_item, pos_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature
            , sid, pos_log_p, ranking_loss_mask, args=None, dataset=None
    ):
        # --- Start of old `forward` logic ---
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats = self.log2feats(
            user_item, mask, seq_feature,)
        loss_mask = (next_mask == 1).to(self.dev)

        sid_level_1_logits, sid_level_2_logits, mlp_output = None, None, None

        # with torch.no_grad():
        #     sid_logfeats_detach = sid_logfeats.detach()
        #     mlp_logfeats_detach = mlp_logfeats.detach()
        #     pos_embs_detach = pos_embs.detach()

        sid_logfeats_detach = sid_logfeats
        mlp_logfeats_detach = mlp_logfeats

        if self.sid and sid is not None:
            # MODIFIED: 使用 HSTU block
            sid1_attn_output = self.sid1_hstu_block(sid_logfeats_detach, sid_logfeats_detach, sid_logfeats_detach,
                                                    attention_mask)
            sid1_attn_output = self.sid1_layer_norm(sid1_attn_output)
            sid_level_1_logits = self.sid1_output_projection(sid1_attn_output)

            sid_emb = self.sid_embedding(sid[:, :, 0])
            sid_q_concat = torch.cat([sid_emb, all_seq_logfeats[0]], dim=-1)
            sid_q = self.sid2_query_projection(sid_q_concat)

            for i in range(0, len(all_seq_logfeats)):
                # Pre-norm 模式：先对 query 做 LayerNorm，然后做 cross-attention，最后残差连接
                sid_q_norm = self.sid2_layer_norm_list[i](sid_q)
                sid2_attn_output = self.sid2_hstu_block_list[i](sid_q_norm, all_seq_logfeats[i], all_seq_logfeats[i],
                                                                attention_mask)
                sid_q = sid_q + sid2_attn_output  # 残差连接：原始 query + attention 输出

            sid_level_2_logits = self.sid2_output_projection(sid_q)

        if self.similarity_function == 'cosine':
            pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
            log_feats_normalized = F.normalize(log_feats, p=2, dim=-1)
        else:
            pos_embs_normalized, log_feats_normalized = pos_embs, log_feats

        return (
            log_feats_normalized, loss_mask, pos_embs_normalized,
            attention_mask, mlp_logfeats_detach, sid_logfeats,
            sid_level_1_logits, sid_level_2_logits, mlp_output,
            pos_embs, mlp_pos_embs
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

    def _calculate_loss(self, seq_embs, loss_mask, pos_embs, next_action_type,
                        sid_level_1_logits, sid_level_2_logits, mlp_output, sid, pos_log_p, mlp_logfeats, causal_mask,
                        args, ranking_loss_mask, mlp_pos_embs):
        """
        计算损失函数，训练和验证都可以使用
        """
        loss = torch.tensor(0.0, device=self.dev)
        loss_dict = {}

        # 计算并输出 ANN score
        ann_scores = None
        if args.infonce:
            if args.learnable_temp:
                infonce = info_nce_loss_inbatch(seq_embs, loss_mask, pos_embs, pos_log_p, self.dev,
                                                self.learnable_temp)

            else:
                infonce = info_nce_loss_inbatch(seq_embs, loss_mask, pos_embs, pos_log_p, self.dev,
                                                args.infonce_temp)
            loss += infonce
            loss_dict['infonce'] = infonce.item()

            # 重新计算 ANN score 用于 reward model
            if self.similarity_function == 'cosine':
                pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
                log_feats_normalized = F.normalize(seq_embs, p=2, dim=-1)
                ann_scores = torch.sum(log_feats_normalized * pos_embs_normalized, dim=-1)  # [batch_size, seq_len]
            else:
                ann_scores = torch.sum(seq_embs * pos_embs, dim=-1)  # [batch_size, seq_len]

        # 计算并输出 SID 概率
        sid1_probs = None
        sid2_probs = None
        if args.sid and sid_level_1_logits is not None and sid_level_2_logits is not None:
            sid1_loss = sid_loss_func(sid_level_1_logits, sid[:, :, 0], loss_mask, self.dev)
            sid2_loss = sid_loss_func(sid_level_2_logits, sid[:, :, 1], loss_mask, self.dev)
            loss += sid1_loss
            loss += sid2_loss
            loss_dict['sid1'] = sid1_loss.item()
            loss_dict['sid2'] = sid2_loss.item()

            # 计算 SID 概率用于 reward model
            sid1_probs = torch.softmax(sid_level_1_logits, dim=-1)  # [batch_size, seq_len, num_classes]
            sid2_probs = torch.softmax(sid_level_2_logits, dim=-1)  # [batch_size, seq_len, num_classes]

            # 使用真实标签获取对应概率
            sid1_labels = sid[:, :, 0].long()  # [batch_size, seq_len]
            sid2_labels = sid[:, :, 1].long()  # [batch_size, seq_len]
            sid_loss_mask = loss_mask.bool()
            # 计算 hit@10 for sid1
            sid1_top10_preds = torch.topk(sid1_probs, k=10, dim=-1).indices
            sid1_targets_expanded = sid1_labels.unsqueeze(-1).expand_as(sid1_top10_preds)
            sid1_hits = (sid1_top10_preds == sid1_targets_expanded).any(dim=-1).float()
            sid1_hit10 = (sid1_hits * sid_loss_mask.float()).sum() / sid_loss_mask.float().sum()

            # 计算 hit@10 for sid2
            sid2_top10_preds = torch.topk(sid2_probs, k=10, dim=-1).indices
            sid2_targets_expanded = sid2_labels.unsqueeze(-1).expand_as(sid2_top10_preds)
            sid2_hits = (sid2_top10_preds == sid2_targets_expanded).any(dim=-1).float()
            sid2_hit10 = (sid2_hits * sid_loss_mask.float()).sum() / sid_loss_mask.float().sum()
            # 使用gather获取对应标签的概率
            sid1_probs = torch.gather(sid1_probs, dim=-1, index=sid1_labels.unsqueeze(-1)).squeeze(
                -1)  # [batch_size, seq_len]
            sid2_probs = torch.gather(sid2_probs, dim=-1, index=sid2_labels.unsqueeze(-1)).squeeze(
                -1)  # [batch_size, seq_len]

            # 输出 SID 概率监控指标
            if sid1_probs is not None:
                loss_dict['SID/Prob1Mean'] = sid1_probs.mean().item()
            if sid2_probs is not None:
                loss_dict['SID/Prob2Mean'] = sid2_probs.mean().item()

            # 添加到loss_dict
            loss_dict['SID/Top10HitRate1'] = sid1_hit10.item()
            loss_dict['SID/Top10HitRate2'] = sid2_hit10.item()

        if args.reward:
            # [MODIFIED] mlp_output 现在是 ctr_logits (raw logits), 范围 [-inf, inf]
            # 假设 self.reward_model 现在返回 logits
            ctr_logits_full = self.reward_model(
                mlp_logfeats.detach(),
                pos_embs.detach(),
                causal_mask.detach(),
                ann_scores.detach() if ann_scores is not None else None,
                sid1_probs.detach() if sid1_probs is not None else None,
                sid2_probs.detach() if sid2_probs is not None else None
            )
            pos_log_p = pos_log_p.to(self.dev)
            # ctr_logits_full = ctr_logits_full/args.infonce_temp
            # ctr_logits_full = ctr_logits_full - pos_log_p.unsqueeze(-1)
            # [MODIFIED] 准备计算 BCEWithLogitsLoss
            ranking_loss_mask = ranking_loss_mask.to(self.dev)
            ranking_loss_mask[:, -1:] = 0  # 忽略最后一个
            combined_mask = (ranking_loss_mask == 1).to(self.dev)  # [B, S]

            # Squeeze ctr_logits to match mask
            # ctr_logits_full shape [B, S, 1]
            ctr_logits = ctr_logits_full.squeeze(-1)[combined_mask]  # [N]
            cos_similarity = ann_scores[combined_mask]  # [N]
            pos_log_p = pos_log_p[combined_mask]  # [N]

            # 温度缩放后直接相减（保持一维）
            adjusted_scores = cos_similarity / args.infonce_temp - pos_log_p
            # labels (ctr_label)
            labels = next_action_type.long()[combined_mask]  # [N]

            if ctr_logits.shape[0] > 0:

                # --- [START] 用户的安全转换逻辑 ---

                # 计算p_ctr概率（仅用于加权，不用于损失计算）
                p_ctr = ctr_logits  # 使用掩码后的 logits

                # 调整余弦相似度范围 [0, 1]
                adjusted_cos_sim = (cos_similarity.detach() + 1) / 2

                # 计算加权的CTR预测值
                weighted_ctr = adjusted_cos_sim * p_ctr

                # 计算损失 - 注意这里使用BCEWithLogitsLoss
                # 我们需要将加权后的值转换为logits空间
                # 使用logit函数进行逆变换（注意数值稳定性）
                weighted_logits = torch.logit(weighted_ctr.clamp(min=1e-7, max=1 - 1e-7))

                # 准备标签
                ctr_label = labels.clone().float()
                ctr_label[ctr_label == 2] = 1  # [N]

                # 计算损失
                # 假设 self.mlp_bce_loss 已被更改为 torch.nn.BCEWithLogitsLoss()
                bce_loss = self.mlp_bce_loss(weighted_logits.float(), ctr_label.float())

                # --- [END] 用户的安全转换逻辑 ---

                # 保持与之前margin loss一致的权重
                loss += 0.5 * bce_loss
                loss_dict['reward'] = bce_loss.item()

                # 6. 计算 AUC (using the new weighted_ctr for scoring)
                with torch.no_grad():
                    scores_proba = weighted_ctr.cpu().float().numpy()  # 使用加权后的 *概率*
                    labels_np = ctr_label.cpu().numpy()

                    try:
                        auc = roc_auc_score(labels_np, scores_proba)
                        loss_dict['MLP_AUC/train'] = auc
                    except ValueError:
                        # Handle case where only one class is present
                        loss_dict['MLP_AUC/train'] = 0.5

                        # Keep ANN AUC for comparison
                    dot_scores = cos_similarity.cpu().numpy()
                    try:
                        ann_auc = roc_auc_score(labels_np, dot_scores)
                        loss_dict['ANN_AUC/train'] = ann_auc
                    except ValueError:
                        loss_dict['ANN_AUC/train'] = 0.5

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
        # 获取模型输出
        (
            seq_embs, loss_mask, pos_embs,
            causal_mask, mlp_logfeats, sid_logfeats,
            sid_level_1_logits, sid_level_2_logits, all_scores, _, mlp_pos_embs
        ) = self.forward(
            user_item, pos_seqs, mask, next_mask, next_action_type,
            seq_feature, pos_feature, sid, pos_log_p,
            args, dataset
        )

        # 计算损失函数
        loss, loss_dict = self._calculate_loss(
            seq_embs, loss_mask, pos_embs, next_action_type,
            sid_level_1_logits, sid_level_2_logits, all_scores, sid, pos_log_p, mlp_logfeats, causal_mask, args,
            ranking_loss_mask, mlp_pos_embs
        )

        # 构建训练日志字典
        log_dict = {}
        log_dict['InfoNCE/train'] = loss_dict.get('infonce', 0.0)
        log_dict['Sid1Loss/train'] = loss_dict.get('sid1', 0.0)
        log_dict['Sid2Loss/train'] = loss_dict.get('sid2', 0.0)
        log_dict['MLP_BCE_Loss/train'] = loss_dict.get('reward', 0.0)
        log_dict['MoE_AuxLoss/train'] = loss_dict.get('moe_aux_loss', 0.0)
        log_dict['Loss/train'] = loss_dict['total']
        log_dict['SID/Top10HitRate1'] = loss_dict.get('SID/Top10HitRate1', 0.0)
        log_dict['SID/Top10HitRate2'] = loss_dict.get('SID/Top10HitRate2', 0.0)
        if args.learnable_temp:
            log_dict['infonce_temp'] = self.learnable_temp

        # 添加MoE指标到log_dict
        if self.use_moe:
            moe_metrics = self._collect_moe_metrics()
            for key, value in moe_metrics.items():
                if isinstance(value, (int, float)):
                    log_dict[key] = value

        # 只在 cuda:0 上每 50 步计算一次指标
        if not hasattr(self, '_metric_counter'):
            self._metric_counter = 0
        do_metric = (str(self.dev) == 'cuda:0' and (
                (self._metric_counter % self.args.log_interval == 0) or self._metric_counter < 100))

        if do_metric:
            # [MODIFIED] AUC计算已在 _calculate_loss 中完成, 这里直接获取
            if args.reward:
                log_dict['MLP_AUC/train'] = loss_dict.get('MLP_AUC/train', 0.0)
                log_dict['ANN_AUC/train'] = loss_dict.get('ANN_AUC/train', 0.0)

            # 计算Acc@10
            pos_logits = torch.sum(seq_embs * pos_embs, dim=-1) * loss_mask
            pos_sim = similarity(pos_logits)
            log_dict['Similarity/positive_train'] = pos_sim.item()

            # 计算HR@10 和 NDCG@10
            hr10_last, ndcg10_last, score_last = calculate_score_fix(
                seq_embs, pos_embs, pos_seqs, next_action_type, loss_mask, self.dev
            )
            hr10, ndcg10, score = calculate_score(
                seq_embs, pos_embs, pos_seqs, next_action_type, loss_mask, self.dev
            )

            log_dict['HR@10/train'] = hr10
            log_dict['NDCG@10/train'] = ndcg10
            log_dict['Score/train'] = score
            log_dict['HR@10_last/train'] = hr10_last
            log_dict['NDCG@10_last/train'] = ndcg10_last
            log_dict['Score_last/train'] = score_last

            # [MODIFIED] 确保AUC键存在
            if 'MLP_AUC/train' not in log_dict:
                log_dict['MLP_AUC/train'] = 0.0
            if 'ANN_AUC/train' not in log_dict:
                log_dict['ANN_AUC/train'] = 0.0

            # if args.sid and dataset is not None:
            #     sid_hr, score_diff, hr_diff, ndcg_diff = calculate_score_sid(
            #         self, sid_logfeats, seq_embs, pos_embs, pos_seqs, next_action_type,
            #         dataset, loss_mask, self.dev
            #     )
            #     log_dict["SID/sid_hr"] = sid_hr
            #     log_dict["SID/score_diff"] = score_diff
            #     log_dict["SID/hr_diff"] = hr_diff
            #     log_dict["SID/ndcg_diff"] = ndcg_diff

            if args.reward and hasattr(self, 'reward_model') and self.reward_model is not None and dataset is not None:
                # 调用 utils 中的函数来计算 reward score
                reward_hr, reward_ndcg, reward_score = calculate_score_reward(
                    seq_embs=seq_embs,
                    pos_embs=pos_embs,
                    pos=pos_seqs,
                    next_action_type=next_action_type,
                    model=self,
                    loss_mask=loss_mask,
                    device=self.dev,
                    sid_logfeats=sid_logfeats,
                    mlp_logfeats=mlp_logfeats,
                    dataset=dataset
                )

                # 将计算结果存入 log_dict，键名与 main_dist.py 中的期望保持一致
                if reward_hr is not None:
                    log_dict['Reward/RewardHR@10/train'] = reward_hr
                    log_dict['Reward/RewardNDCG@10/train'] = reward_ndcg
                    log_dict['Reward/RewardScore/train'] = reward_score

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