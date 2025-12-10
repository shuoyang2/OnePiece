import json
import os
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor


def info_nce_loss_inbatch(seq_embs, loss_mask, pos_embs, pos_log_p, device, temp=0.1):
    """
    计算带有采样偏差校正的 InfoNCE 损失。
    Args:
        pos_log_p (torch.Tensor): 正样本的对数采样概率。
        neg_log_p (torch.Tensor): 负样本的对数采样概率。
    """
    batch_size, max_len, hidden_units = seq_embs.shape
    pos_log_p = pos_log_p.to(device)
    loss_mask = loss_mask.view(-1).bool()
    pos_log_p = pos_log_p.view(-1)[loss_mask].contiguous()
    query_embs = seq_embs.view(-1, hidden_units)[loss_mask].contiguous()
    all_neg_embs = pos_embs.view(-1, hidden_units)[loss_mask].contiguous()
    # 计算原始 neg logits
    sim_matrix = torch.matmul(query_embs, all_neg_embs.t()) / temp
    sim_matrix -= pos_log_p.unsqueeze(0)
    labels = torch.arange(sim_matrix.shape[0], device=device, dtype=torch.int64)

    return F.cross_entropy(sim_matrix, labels)


def sid_loss_func(sid_logits, sid, loss_mask, device):
    batch_size, seq_len, num_classes = sid_logits.shape
    mask_flat = loss_mask.view(-1).bool()
    sid_logits_reshaped = sid_logits.view(batch_size * seq_len, num_classes)
    sid_logits_reshaped = sid_logits_reshaped[mask_flat]
    sid_logits_reshaped = torch.clamp(sid_logits_reshaped, min=-20, max=20)
    sid_reshaped = sid.view(batch_size * seq_len)
    sid_reshaped = sid_reshaped.long()
    sid_reshaped = sid_reshaped[mask_flat]
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(sid_logits_reshaped, sid_reshaped)
    return loss.to(device)


def model_params(model, writer, global_step):
    grad_norms = []

    # 检查是否是SimpleDataParallel模型
    if hasattr(model, 'module'):
        # 如果是SimpleDataParallel，只使用主模块的参数
        target_model = model.module
    else:
        target_model = model

    for name, param in target_model.named_parameters():
        if param.grad is not None:
            try:
                # 确保梯度在正确的设备上
                grad_norm = param.grad.norm(2).item()
                grad_norms.append(grad_norm)
            except RuntimeError as e:
                # 如果访问梯度时出现CUDA错误，跳过这个参数
                print(f"Warning: Failed to access gradient for parameter {name}: {e}")
                continue

    if grad_norms:
        grad_norm_mean = np.mean(grad_norms)
        grad_norm_max = np.max(grad_norms)
        grad_norm_min = np.min(grad_norms)
        writer.add_scalar("Gradient/Mean", grad_norm_mean, global_step)
        writer.add_scalar("Gradient/Max", grad_norm_max, global_step)
        writer.add_scalar("Gradient/Min", grad_norm_min, global_step)


def model_grad_norms(model, writer, global_step):
    param_means = []

    # 检查是否是SimpleDataParallel模型
    if hasattr(model, 'replicas'):
        # 如果是SimpleDataParallel，使用第一个replica的参数
        target_model = model.replicas[0]
    elif hasattr(model, 'module'):
        # 如果是其他类型的DataParallel，使用主模块的参数
        target_model = model.module
    else:
        target_model = model

    for name, param in target_model.named_parameters():
        if param.grad is not None:
            try:
                # 确保参数在正确的设备上
                param_mean = param.data.mean().item()
                param_means.append(param_mean)
            except RuntimeError as e:
                # 如果访问参数时出现CUDA错误，跳过这个参数
                print(f"Warning: Failed to access parameter {name}: {e}")
                continue

    if param_means:
        param_mean_mean = np.mean(param_means)
        param_mean_max = np.max(param_means)
        param_mean_min = np.min(param_means)
        writer.add_scalar("Parameter/Mean", param_mean_mean, global_step)
        writer.add_scalar("Parameter/Max", param_mean_max, global_step)
        writer.add_scalar("Parameter/Min", param_mean_min, global_step)


def similarity(pos_logits):
    return pos_logits.mean()


def calculate_acc(pos_logits, neg_logits, k=10):
    batch_size, maxlen = pos_logits.shape
    concat_logits = torch.cat((pos_logits, neg_logits), dim=1)
    sort_logits = torch.argsort(concat_logits, dim=1, descending=True)[:, :k]
    is_positive = (sort_logits[:, :] < maxlen).float().mean()
    return is_positive


def calculate_hitrate(top_k_items, labels):
    labels = labels.unsqueeze(1).expand_as(top_k_items)
    hits = (top_k_items == labels).any(dim=1)
    return hits.float().mean().item()


def calculate_ndcg(top_k_items, labels):
    labels = labels.unsqueeze(1)
    ranks = (top_k_items == labels).nonzero(as_tuple=True)
    ndcg = torch.zeros(len(labels)).to(labels.device)
    positions = ranks[1]
    dcg = 1.0 / torch.log2(positions.float() + 2)
    ndcg[ranks[0]] = dcg
    return ndcg.mean().item()


def gate_score(mm_emb_names, scores):
    gate_num = 1 + len(mm_emb_names)
    scores = torch.cat(scores, dim=0).view(-1, gate_num).mean(dim=0)
    result = {}
    result["item_id"] = scores[0]
    for i in range(1, len(mm_emb_names) + 1):
        result[mm_emb_names[i - 1]] = scores[i]
    return result


# ===============================================================
# 以下是按新要求再次修改的函数
# ===============================================================

def calculate_score(seq_embs, pos_embs, pos, next_action_type, loss_mask, device="cuda"):
    """
    修改版v2：loss_mask 同时过滤 pos 和 neg。
    """
    hidden_unit = pos_embs.shape[-1]
    labels = pos[next_action_type != 0].to(device)
    query_emb = seq_embs[next_action_type != 0].to(device)

    # --- 核心修改：使用 loss_mask 同时过滤 pos 和 neg ---
    pos_embs_flat = pos_embs.view(-1, hidden_unit)
    pos_ids_flat = pos.view(-1)
    mask_flat = loss_mask.view(-1).bool()

    active_pos_embs = pos_embs_flat[mask_flat]
    active_pos_ids = pos_ids_flat[mask_flat]

    candidate_embs = torch.cat([active_pos_embs], dim=0).to(device)
    candidate_ids = torch.cat([active_pos_ids], dim=0).to(device)
    # --- 修改结束 ---

    scores = torch.matmul(query_emb, candidate_embs.T)
    _, topk_indices = torch.topk(scores, k=10, dim=1)
    topk_items = candidate_ids[topk_indices]

    hr10 = calculate_hitrate(topk_items, labels)
    ndcg10 = calculate_ndcg(topk_items, labels)
    score = 0.31 * hr10 + 0.69 * ndcg10

    return hr10, ndcg10, score


def calculate_score_fix(seq_embs, pos_embs, pos, next_action_type, loss_mask, device="cuda"):
    """
    修改版v2：loss_mask 同时过滤 pos 和 neg。
    """
    hidden_unit = pos_embs.shape[-1]

    query_emb = seq_embs[:, -1, :].to(device)
    query_emb = query_emb[next_action_type[:, -1] != 0]
    if len(query_emb) == 0:
        return 0, 0, 0

    # --- 核心修改：使用 loss_mask 同时过滤 pos 和 neg ---
    pos_embs_flat = pos_embs.view(-1, hidden_unit)
    pos_ids_flat = pos.view(-1)
    mask_flat = loss_mask.view(-1).bool()

    active_pos_embs = pos_embs_flat[mask_flat]
    active_pos_ids = pos_ids_flat[mask_flat]

    candidate_embs = torch.cat([active_pos_embs], dim=0).to(device)
    candidate_ids = torch.cat([active_pos_ids], dim=0).to(device)
    # --- 修改结束 ---

    scores = torch.matmul(query_emb, candidate_embs.T)
    _, topk_indices = torch.topk(scores, k=10, dim=1)
    topk_items = candidate_ids[topk_indices]

    labels = pos[:, -1].to(device)
    labels = labels[next_action_type[:, -1] != 0]
    hr10 = calculate_hitrate(topk_items, labels)
    ndcg10 = calculate_ndcg(topk_items, labels)
    score = 0.31 * hr10 + 0.69 * ndcg10

    return hr10, ndcg10, score


# def calculate_score_sid(
#         model,
#         sid_logfeats,
#         seq_embs,  # 新增: 传入Transformer的输出
#         pos_embs,
#         pos,
#         next_action_type,
#         dataset,
#         loss_mask,
#         device="cuda"
# ):
#     """
#     修复版的SID评分函数，专门用于评估阶段。
#     - 接收 model 对象，用于调用无数据泄漏的 sid 预测方法。
#     - 接收 log_feats 和 attention_mask 作为预测 sid 的输入。
#     """
#     ann_k = 32
#     hidden_unit = seq_embs.shape[-1]
#
#     # 1. 筛选出最后一个时间步需要评估的 query 和 labels，这里改成全选
#     eval_mask_last = (next_action_type[:, -1] != -1)
#     if eval_mask_last.sum() == 0:
#         return 0.0, 0.0, 0.0
#
#     query_emb = seq_embs[:, -1, :][eval_mask_last]
#     labels = pos[:, -1][eval_mask_last]
#
#     # 2. 构建去重后的候选池
#     mask_flat = loss_mask.view(-1).bool()
#     active_pos_embs = pos_embs.view(-1, hidden_unit)[mask_flat]
#     active_pos_ids = pos.view(-1)[mask_flat]
#
#     candidate_embs_all = torch.cat([active_pos_embs], dim=0)
#     candidate_ids_all = torch.cat([active_pos_ids], dim=0)
#
#     # 使用一种更稳健的方式去重
#     unique_ids, perm = torch.unique(candidate_ids_all, sorted=False, return_inverse=True)
#     unique_map_indices = torch.arange(perm.size(0), dtype=perm.dtype, device=perm.device)
#     first_occurrence_indices = perm.new_empty(unique_ids.size(0)).scatter_(0, perm, unique_map_indices)
#     candidate_ids_unique = candidate_ids_all[first_occurrence_indices]
#     candidate_embs_unique = candidate_embs_all[first_occurrence_indices]
#
#     # 3) ANN 召回 topK 候选
#     ann_scores = torch.matmul(query_emb, candidate_embs_unique.T)
#     topk_ann_scores, topk_indices_ann = torch.topk(ann_scores, k=ann_k, dim=1)
#     topk_item_ids = candidate_ids_unique[topk_indices_ann]
#
#     # Baseline metrics using ANN topK before SID rerank (Top10)
#     baseline_hr = calculate_hitrate(topk_item_ids[:, :10], labels)
#     baseline_ndcg = calculate_ndcg(topk_item_ids[:, :10], labels)
#     baseline_score = 0.31 * baseline_hr + 0.69 * baseline_ndcg
#
#     # 4) 回查 SID1/SID2
#     topk_item_ids_cpu = topk_item_ids.cpu().numpy()
#     topk_sids_list = []
#     for user_ids in topk_item_ids_cpu:
#         user_sids_list = []
#         for item_id in user_ids:
#             c_id = dataset.indexer_i_rev[item_id.item()]
#             try:
#                 user_sids = dataset.sid[int(c_id)]
#             except (KeyError, ValueError):
#                 user_sids = np.array([0, 0])
#             user_sids_list.append(user_sids)
#         topk_sids_list.append(user_sids_list)
#
#     sid_logfeats_valid = sid_logfeats[eval_mask_last]
#     topk_sids_np = np.array(topk_sids_list, dtype=np.int64)
#     topk_sids = torch.from_numpy(topk_sids_np).to(device=device, dtype=torch.long)
#     topk_s1, topk_s2 = topk_sids[..., 0], topk_sids[..., 1]
#
#     # 5) 计算每个sid的曝光概率
#     sid_level_1_prob, sid_level_2_prob = model.predict_sid(sid_logfeats_valid, topk_s1)
#
#     # 修改gather操作，去掉repeat
#     # 使用unsqueeze增加维度，使索引与概率矩阵维度匹配
#     topk_s1_expanded = topk_s1.unsqueeze(-1)  # 形状变为 [batch_size, ann_k, 1]
#     topk_s2_expanded = topk_s2.unsqueeze(-1)  # 形状变为 [batch_size, ann_k, 1]
#     sid_level_1_prob = sid_level_1_prob.expand(-1, ann_k, -1)  # 形状变为 [batch_size, ann_k, num_classes]
#     # 在最后一个维度上进行gather操作
#     topk_sid1_probs = torch.gather(sid_level_1_prob, 2, topk_s1_expanded).squeeze(-1)
#     topk_sid2_probs = torch.gather(sid_level_2_prob, 2, topk_s2_expanded).squeeze(-1)
#
#     # 6) 计算最终的打分
#     # 方案1: 加权线性融合 (推荐)
#     # final_scores = topk_ann_scores + 0.1 * topk_sid1_probs + 0.1 * topk_sid2_probs
#
#     # 方案2: 对数空间融合 (数值稳定)
#     # final_scores = topk_ann_scores * torch.exp(0.1 * torch.log(topk_sid1_probs + 1e-3) + 0.1 * torch.log(topk_sid2_probs + 1e-3))
#
#     # 方案3: 改进的乘法融合 (当前使用)
#     final_scores = topk_ann_scores * (1 + topk_sid1_probs) * (1 + topk_sid2_probs)
#
#     # 方案4: 幂函数融合
#     # final_scores = topk_ann_scores * torch.pow(1 + topk_sid1_probs, 0.5) * torch.pow(1 + topk_sid2_probs, 0.5)
#     reranked_order = torch.argsort(final_scores, dim=1, descending=True)
#     reranked_item_ids = torch.gather(topk_item_ids, 1, reranked_order)
#
#     # 只取 Top 10 进行最终评估
#     reranked_hr = calculate_hitrate(reranked_item_ids[:, :10], labels)
#     reranked_ndcg = calculate_ndcg(reranked_item_ids[:, :10], labels)
#     reranked_score = 0.31 * reranked_hr + 0.69 * reranked_ndcg
#
#     # 计算 diff 指标（SID重排后的提升）
#     diff_hr = reranked_hr - baseline_hr
#     diff_ndcg = reranked_ndcg - baseline_ndcg
#     diff_score = reranked_score - baseline_score
#
#     # 7) SID Reward Model 评估方式（新增）
#     sid_reward_hr = None
#     sid_reward_ndcg = None
#     sid_reward_score = None
#     sid_reward_scores_mean = None
#
#     if hasattr(model, 'reward_model') and model.reward_model is not None:
#         try:
#             # 准备reward model的输入特征
#             batch_size, ann_k = topk_ann_scores.shape
#
#             # 构建序列embedding和物品embedding
#             seq_embs_expanded = query_emb.unsqueeze(1).expand(-1, ann_k, -1)  # [batch_size, ann_k, hidden_dim]
#
#             # 从已有的candidate_embs_unique中获取对应的embeddings
#             topk_candidate_embs = candidate_embs_unique[topk_indices_ann]  # [batch_size, ann_k, hidden_dim]
#
#             # 构建注意力掩码：EnhancedSortMLP/FlashMultiHeadAttention 期望 [batch, seq_len, seq_len]
#             attn_mask = torch.ones(batch_size, ann_k, ann_k, dtype=torch.bool, device=device)
#
#             # 调用reward model
#             sid_reward_scores = model.reward_model(
#                 seq_embs_expanded, topk_candidate_embs, attn_mask,
#                 topk_ann_scores, topk_sid1_probs, topk_sid2_probs
#             ).squeeze(-1)  # [batch_size, ann_k]
#
#             # 使用reward model的输出进行重新排序
#             sid_reward_order = torch.argsort(sid_reward_scores, dim=1, descending=True)
#             sid_reward_item_ids = torch.gather(topk_item_ids, 1, sid_reward_order)
#
#             # 计算SID Reward Model的评估指标
#             sid_reward_hr = calculate_hitrate(sid_reward_item_ids[:, :10], labels)
#             sid_reward_ndcg = calculate_ndcg(sid_reward_item_ids[:, :10], labels)
#             sid_reward_score = 0.31 * sid_reward_hr + 0.69 * sid_reward_ndcg
#             sid_reward_scores_mean = sid_reward_scores.mean().item()
#
#             print(f"SID Reward Model: HR@10={sid_reward_hr:.4f}, NDCG@10={sid_reward_ndcg:.4f}, Score={sid_reward_score:.4f}")
#
#         except Exception as e:
#             print(f"SID Reward Model evaluation failed: {e}")
#             sid_reward_hr = None
#             sid_reward_ndcg = None
#             sid_reward_score = None
#             sid_reward_scores_mean = None
#
#     # 使用更高精度计算避免 bf16 问题
#     ann_base_contribution = topk_ann_scores.float().mean().item()
#     sid1_increment = (topk_ann_scores.float() * topk_sid1_probs.float()).mean().item()
#     sid2_increment = (topk_ann_scores.float() * topk_sid2_probs.float()).mean().item()
#     sid_interaction = (topk_ann_scores.float() * topk_sid1_probs.float() * topk_sid2_probs.float()).mean().item()
#
#     # 验证计算一致性
#     calculated_total = ann_base_contribution + sid1_increment + sid2_increment + sid_interaction
#     actual_total = final_scores.float().mean().item()
#
#     # 使用实际总值计算比例
#     total_contribution = actual_total
#
#     # 计算比例
#     ann_ratio = ann_base_contribution / total_contribution
#     sid1_ratio = sid1_increment / total_contribution
#     sid2_ratio = sid2_increment / total_contribution
#     interaction_ratio = sid_interaction / total_contribution
#
#     # 调试信息：检查计算一致性
#     ratio_sum = ann_ratio + sid1_ratio + sid2_ratio + interaction_ratio
#     if abs(ratio_sum - 1.0) > 0.01:  # 如果总和偏离1.0超过1%
#         print(f"Warning: Ratio sum = {ratio_sum:.6f}, calculated_total = {calculated_total:.6f}, actual_total = {actual_total:.6f}")
#
#     # 计算各个分数的均值，用于监控
#     ann_scores_mean = topk_ann_scores.mean().item()
#     sid1_probs_mean = topk_sid1_probs.mean().item()
#     sid2_probs_mean = topk_sid2_probs.mean().item()
#
#     return diff_hr, diff_ndcg, diff_score, ann_ratio, sid1_ratio, sid2_ratio, interaction_ratio, ann_scores_mean, sid1_probs_mean, sid2_probs_mean, sid_reward_hr, sid_reward_ndcg, sid_reward_score, sid_reward_scores_mean

def calculate_score_sid(
        model,
        seq_embs,
        pos_embs,
        pos,
        next_action_type,
        dataset,
        loss_mask,
        device="cuda",
        # 新增/修改的参数，用于支持 predict_beam_search
        user_item=None,
        pos_seqs=None,
        mask=None,
        next_mask=None,
        seq_feature=None,
        pos_feature=None,
        sid=None,
        args=None
):
    """
    修改版的SID评分函数，调用 model.predict_beam_search
    """
    ann_k = 32
    hidden_unit = seq_embs.shape[-1]

    # 1. 筛选出最后一个时间步需要评估的 query 和 labels
    eval_mask_last = (next_action_type[:, -1] != -1)
    num_eval_samples = eval_mask_last.sum()
    if num_eval_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    query_emb = seq_embs[:, -1, :][eval_mask_last]
    labels = pos[:, -1][eval_mask_last]

    # 2. 构建去重后的候选池
    mask_flat = loss_mask.view(-1).bool()
    active_pos_embs = pos_embs.view(-1, hidden_unit)[mask_flat]
    active_pos_ids = pos.view(-1)[mask_flat]

    candidate_embs_all = torch.cat([active_pos_embs], dim=0)
    candidate_ids_all = torch.cat([active_pos_ids], dim=0)

    unique_ids, perm = torch.unique(candidate_ids_all, sorted=False, return_inverse=True)
    unique_map_indices = torch.arange(perm.size(0), dtype=perm.dtype, device=perm.device)
    first_occurrence_indices = perm.new_empty(unique_ids.size(0)).scatter_(0, perm, unique_map_indices)
    candidate_ids_unique = candidate_ids_all[first_occurrence_indices]
    candidate_embs_unique = candidate_embs_all[first_occurrence_indices]

    reid_to_emb_map = {reid.item(): emb for reid, emb in zip(candidate_ids_unique, candidate_embs_unique)}

    # 3) ANN 召回 topK 候选
    ann_scores = torch.matmul(query_emb, candidate_embs_unique.T)
    topk_ann_scores, topk_indices_ann = torch.topk(ann_scores, k=ann_k, dim=1)
    topk_item_ids_ann = candidate_ids_unique[topk_indices_ann]

    baseline_hr = calculate_hitrate(topk_item_ids_ann[:, :10], labels)
    baseline_ndcg = calculate_ndcg(topk_item_ids_ann[:, :10], labels)
    baseline_score = 0.31 * baseline_hr + 0.69 * baseline_ndcg

    # ======================== SID Reranking Part ========================

    # 4. 执行 Beam Search 获取 SID 预测
    # 使用新接口 predict_beam_search
    # top_sequences: [B, beam, 2], top_scores: [B, beam]
    top_sequences, top_scores = model.predict_beam_search(
        user_item, pos_seqs, mask, next_mask, next_action_type,
        seq_feature, pos_feature, sid, args, dataset
    )

    # 注意：predict_beam_search 对所有 sample 进行了预测，这里我们需要根据 eval_mask_last 筛选
    top_sequences = top_sequences[eval_mask_last]

    # 5. 计算 SID Hit Rate
    sid_hit_count = 0
    ground_truth_cids = []
    for reid in labels:
        if reid.item() != 0:
            ground_truth_cids.append(dataset.indexer_i_rev[reid.item()])
        else:
            print(reid)

    ground_truth_sids = [dataset.sid.get(cid, [-1, -1]) for cid in ground_truth_cids]

    for i in range(num_eval_samples):
        true_sid_pair = ground_truth_sids[i]
        if (np.array(true_sid_pair) == [-1, -1]).all(): continue

        predicted_pairs = top_sequences[i].cpu().numpy().tolist()
        if any((np.array(p) == np.array(true_sid_pair)).all() for p in predicted_pairs):
            sid_hit_count += 1

    sid_hr = sid_hit_count / num_eval_samples if num_eval_samples > 0 else 0.0

    # 6. SID 反查召回，并与 ANN 结果合并
    rerank_candidate_ids = []
    active_candidate_set = set(candidate_ids_all.cpu().numpy())

    for i in range(num_eval_samples):
        # SID 召回
        sid_retrieved_cids = []
        for sid_pair in top_sequences[i]:
            sid_key = f"{sid_pair[0].item()}_{sid_pair[1].item()}"
            retrieved_cids = dataset.sid_reverse.get(sid_key, None)
            if retrieved_cids is not None:
                sid_retrieved_cids.append(retrieved_cids)
        sid_retrieved_cids = set(sid_retrieved_cids)

        sid_retrieved_reids = {dataset.indexer['i'][cid] for cid in sid_retrieved_cids if cid in dataset.indexer['i']}
        valid_sid_reids = {reid for reid in sid_retrieved_reids if reid in active_candidate_set}

        ann_reids = set(topk_item_ids_ann[i].cpu().numpy())
        final_candidate_reids = list(ann_reids.union(valid_sid_reids))

        num_candidates = len(final_candidate_reids)
        if num_candidates == 0:
            rerank_candidate_ids.append(torch.tensor([], dtype=torch.long, device=device))
            continue

        mlp_item_embs = torch.stack([reid_to_emb_map[reid] for reid in final_candidate_reids])

        reid_to_ann_score_map = {reid.item(): score.item() for reid, score in
                                 zip(topk_item_ids_ann[i], topk_ann_scores[i])}
        mlp_ann_scores = torch.tensor([
            reid_to_ann_score_map.get(reid, torch.matmul(query_emb[i], reid_to_emb_map[reid]).item())
            for reid in final_candidate_reids
        ], device=device)

        # 调用精排模型 (Reward Model)
        current_query_emb = query_emb[i].unsqueeze(0).expand(num_candidates, -1).unsqueeze(0)
        current_item_embs = mlp_item_embs.unsqueeze(0)
        current_ann_scores = mlp_ann_scores.unsqueeze(0)

        # 如果没有 reward_model，这里会报错，需保证 model 有此属性
        if hasattr(model, 'reward_model') and model.reward_model is not None:
            with torch.no_grad():
                rerank_scores = model.reward_model(current_query_emb, current_item_embs,
                                                   ann_scores=current_ann_scores).squeeze()
        else:
            # Fallback: 使用 ANN score
            rerank_scores = mlp_ann_scores

        _, sorted_indices = torch.topk(rerank_scores, k=min(10, num_candidates), dim=0)
        final_top10_ids = torch.tensor(final_candidate_reids, device=device)[sorted_indices]
        rerank_candidate_ids.append(final_top10_ids)

    # 7. 计算精排后的指标
    reranked_hr_list = []
    reranked_ndcg_list = []
    for i in range(num_eval_samples):
        topk_items = rerank_candidate_ids[i].view(1, -1)
        label = labels[i].view(1)
        if topk_items.numel() > 0:
            reranked_hr_list.append(calculate_hitrate(topk_items, label))
            reranked_ndcg_list.append(calculate_ndcg(topk_items, label))

    reranked_hr = np.mean(reranked_hr_list) if reranked_hr_list else 0.0
    reranked_ndcg = np.mean(reranked_ndcg_list) if reranked_ndcg_list else 0.0
    reranked_score = 0.31 * reranked_hr + 0.69 * reranked_ndcg

    score_diff = reranked_score - baseline_score
    hr_diff = reranked_hr - baseline_hr
    ndcg_diff = reranked_ndcg - baseline_ndcg

    return sid_hr, score_diff, hr_diff, ndcg_diff





def calculate_score_reward(seq_embs, pos_embs, pos, next_action_type, model, loss_mask,
                           device="cuda", sid_logfeats=None, mlp_logfeats=None, dataset=None):
    """
    修改版v2：loss_mask 同时过滤 pos 和 neg。
    """
    ann_k = 32
    hidden_unit = seq_embs.shape[-1]

    # 1. 筛选出序列中所有需要评估的 query 和 labels（非 padding 的交互）
    eval_mask = (next_action_type != 0)
    if eval_mask.sum() == 0:
        # 如果没有有效的评估样本，则返回 0
        return 0.0, 0.0, 0.0

    query_emb = seq_embs[eval_mask]
    query_mlp_emb = mlp_logfeats[eval_mask]
    labels = pos[eval_mask]

    # 2. 构建去重后的候选池
    mask_flat = loss_mask.view(-1).bool()
    active_pos_embs = pos_embs.view(-1, hidden_unit)[mask_flat]
    active_pos_ids = pos.view(-1)[mask_flat]

    candidate_embs_all = torch.cat([active_pos_embs], dim=0)
    candidate_ids_all = torch.cat([active_pos_ids], dim=0)

    # 使用一种更稳健的方式去重
    unique_ids, perm = torch.unique(candidate_ids_all, sorted=False, return_inverse=True)
    unique_map_indices = torch.arange(perm.size(0), dtype=perm.dtype, device=perm.device)
    first_occurrence_indices = perm.new_empty(unique_ids.size(0)).scatter_(0, perm, unique_map_indices)
    candidate_ids_unique = candidate_ids_all[first_occurrence_indices]
    candidate_embs_unique = candidate_embs_all[first_occurrence_indices]

    # 3) ANN 召回 topK 候选
    ann_scores = torch.matmul(query_emb, candidate_embs_unique.T)
    topk_ann_scores, topk_indices_ann = torch.topk(ann_scores, k=ann_k, dim=1)
    topk_item_ids = candidate_ids_unique[topk_indices_ann]

    # 计算baseline指标
    baseline_hr = calculate_hitrate(topk_item_ids[:, :10], labels)
    baseline_ndcg = calculate_ndcg(topk_item_ids[:, :10], labels)
    baseline_score = 0.31 * baseline_hr + 0.69 * baseline_ndcg

    # 4) Reward Model 评估
    reward_hr = None
    reward_ndcg = None
    reward_score = None

    if hasattr(model, 'reward_model') and model.reward_model is not None:
        try:
            # 准备reward model的输入特征
            batch_size, _ = topk_ann_scores.shape

            # 构建序列embedding和物品embedding
            # seq_embs_expanded = query_emb.unsqueeze(1).expand(-1, ann_k, -1)  # [N, ann_k, hidden_dim]
            query_mlp_emb_expanded = query_mlp_emb.unsqueeze(1).expand(-1, ann_k, -1)  # [N, ann_k, hidden_dim]

            # 从已有的candidate_embs_unique中获取对应的embeddings
            topk_candidate_embs = candidate_embs_unique[topk_indices_ann]  # [N, ann_k, hidden_dim]

            # 构建注意力掩码
            attn_mask = torch.ones(batch_size, ann_k, ann_k, dtype=torch.bool, device=device)

            # [MODIFIED] 调用reward model，并应用您要求的加权逻辑
            with torch.no_grad():
                # 1. Get p_ctr from reward model
                p_ctr_scores = model.reward_model(
                    query_mlp_emb_expanded, topk_candidate_embs, attn_mask,
                    topk_ann_scores, None, None  # sid1_probs 和 sid2_probs 设为 None
                ).squeeze(-1)  # [N, ann_k]

                # 2. Get cos_similarity (which is topk_ann_scores)
                cos_similarity = topk_ann_scores.detach()  # [N, ann_k]
                
                # 3. Apply user's logic: adjusted_cos_sim = (cos_similarity + 1) / 2
                adjusted_cos_sim = (cos_similarity + 1) / 2.0
                
                # 4. weighted_p_ctr = adjusted_cos_sim * p_ctr
                reward_scores = adjusted_cos_sim * p_ctr_scores  # This is weighted_p_ctr

            # 使用reward model的输出 (weighted_p_ctr) 进行重新排序
            reward_order = torch.argsort(reward_scores, dim=1, descending=True)
            reward_item_ids = torch.gather(topk_item_ids, 1, reward_order)

            # 计算Reward Model的评估指标
            reward_hr = calculate_hitrate(reward_item_ids[:, :10], labels)
            reward_ndcg = calculate_ndcg(reward_item_ids[:, :10], labels)
            reward_score = 0.31 * reward_hr + 0.69 * reward_ndcg
            print(f"Reward Model: HR@10={reward_hr:.4f}, NDCG@10={reward_ndcg:.4f}, Score={reward_score:.4f}")
            print(f"Baseline: HR@10={baseline_hr:.4f}, NDCG@10={baseline_ndcg:.4f}, Score={baseline_score:.4f}")
            # 计算相对于baseline的提升
            reward_hr = reward_hr - baseline_hr
            reward_ndcg = reward_ndcg - baseline_ndcg
            reward_score = reward_score - baseline_score

        except Exception as e:
            print(f"Reward Model evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            reward_hr = None
            reward_ndcg = None
            reward_score = None

    return reward_hr, reward_ndcg, reward_score




# ===============================================================
# 以下是未修改的优化器代码
# ===============================================================

def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class SingleDeviceMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


class PreprocessedDataset(Dataset):
    """
    使用预处理的批次数据的Dataset类
    支持高效的多worker并发访问，使用pickle存储完整batch
    """

    def __init__(self, batch_data_dir, shuffle=True):
        """
        初始化预处理数据集

        Args:
            batch_data_dir: 预处理数据目录路径
            shuffle: 是否随机打乱批次顺序
        """
        self.batch_data_dir = Path(batch_data_dir)

        # 加载元数据
        metadata_file = self.batch_data_dir / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        # 获取所有批次文件
        self.batch_files = sorted(list(self.batch_data_dir.glob("batch_*.pkl")))
        self.total_batches = len(self.batch_files)

        if self.total_batches == 0:
            raise ValueError(f"没有在 {batch_data_dir} 中找到预处理的批次文件")

        print(f"加载了 {self.total_batches} 个预处理批次 (pickle格式)")

        # 创建批次索引
        self.batch_indices = list(range(self.total_batches))
        if shuffle:
            random.shuffle(self.batch_indices)

    def __len__(self):
        return self.total_batches

    def __getitem__(self, idx):
        """
        加载单个预处理的批次

        Args:
            idx: 批次索引

        Returns:
            batch: 完整的批次数据元组，与原始DataLoader返回的格式完全一致
        """
        batch_idx = self.batch_indices[idx]
        batch_file = self.batch_files[batch_idx]

        try:
            # 直接加载完整的batch
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f)

            return batch

        except Exception as e:
            print(f"加载批次文件 {batch_file} 时发生错误: {e}")
            raise

    def reshuffle(self):
        """重新打乱批次顺序，用于新的epoch"""
        random.shuffle(self.batch_indices)


class DynamicPreprocessedDataset(Dataset):
    """
    动态预处理数据集 - 支持训练过程中动态等待预处理文件
    训练到第i个batch时，只需要预处理到第i个batch即可
    """

    def __init__(self, batch_data_dir, max_batches=None, wait_timeout=300):
        """
        初始化动态预处理数据集

        Args:
            batch_data_dir: 预处理数据目录路径
            max_batches: 最大批次数（如果为None，则动态检测）
            wait_timeout: 等待预处理文件的最大时间（秒）
        """
        self.batch_data_dir = Path(batch_data_dir)
        self.wait_timeout = wait_timeout
        self.max_batches = max_batches
        self.current_batch = 0

        # 检查元数据文件
        metadata_file = self.batch_data_dir / "metadata.pkl"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                # 如果传入了max_batches，优先使用传入的值
                if self.max_batches is None:
                    self.max_batches = self.metadata.get('total_batches', None)
                print(f"找到元数据文件，总批次数: {self.max_batches}")
            except Exception as e:
                print(f"读取元数据文件失败: {e}")
                self.metadata = {}
                # 如果传入了max_batches，保持使用传入的值
                if self.max_batches is None:
                    self.max_batches = None
        else:
            self.metadata = {}
            # 如果传入了max_batches，保持使用传入的值
            if self.max_batches is None:
                self.max_batches = None
            print(f"元数据文件不存在，使用传入的max_batches: {self.max_batches}")

        print(f"DynamicPreprocessedDataset初始化:")
        print(f"  - 数据目录: {self.batch_data_dir}")
        print(f"  - 最大批次数: {self.max_batches}")
        print(f"  - 等待超时: {self.wait_timeout}秒")

    def __len__(self):
        """返回数据集长度（如果已知）"""
        if self.max_batches is not None:
            return self.max_batches
        else:
            # 动态检测当前可用的批次数
            available_batches = len(list(self.batch_data_dir.glob("batch_*.pkl")))
            if available_batches > 0:
                print(f"动态检测到 {available_batches} 个可用批次")
                return available_batches
            else:
                # 如果没有预处理文件，返回一个较大的数字，让训练能够等待
                # 这样训练不会立即跳过，而是会等待预处理文件生成
                print("没有预处理文件，将等待预处理开始...")
                return 10000  # 返回一个较大的数字，让训练能够等待

    def get_actual_length(self):
        """获取实际可用的批次数"""
        available_batches = len(list(self.batch_data_dir.glob("batch_*.pkl")))
        return available_batches

    def __getitem__(self, idx):
        """
        动态加载预处理批次，如果文件不存在则等待

        Args:
            idx: 批次索引

        Returns:
            batch: 完整的批次数据
        """
        batch_file = self.batch_data_dir / f"batch_{idx:06d}.pkl"

        # 如果文件不存在，等待预处理完成
        if not batch_file.exists():
            print(f"等待预处理批次 {idx}...")
            start_time = time.time()

            while not batch_file.exists():
                elapsed = time.time() - start_time
                if elapsed > self.wait_timeout:
                    raise TimeoutError(f"等待批次 {idx} 超时 ({self.wait_timeout}秒)")

                time.sleep(1)  # 每秒检查一次

                # 每10秒输出一次等待信息
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    print(f"已等待 {int(elapsed)} 秒，继续等待批次 {idx}...")

        # 等待文件写入完成，检查文件完整性
        # self._wait_for_file_completion(batch_file, idx)

        # 加载批次数据，添加重试机制
        max_retries = 3
        for retry in range(max_retries):
            try:
                time_1 = time.time()
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f)
                time_2 = time.time()
                return batch
            except (pickle.UnpicklingError, EOFError) as e:
                if retry < max_retries - 1:
                    print(f"加载批次文件 {batch_file} 时发生错误: {e}，重试 {retry + 1}/{max_retries}")
                    time.sleep(2)  # 等待2秒后重试
                    # 重新等待文件完成
                    self._wait_for_file_completion(batch_file, idx)
                else:
                    print(f"加载批次文件 {batch_file} 时发生错误: {e}，重试次数已用完")
                    raise
            except Exception as e:
                print(f"加载批次文件 {batch_file} 时发生错误: {e}")
                raise

    def _wait_for_file_completion(self, batch_file, idx):
        """
        等待文件写入完成，检查文件完整性

        Args:
            batch_file: 批次文件路径
            idx: 批次索引
        """
        max_wait_time = 30  # 最多等待30秒
        start_time = time.time()
        last_size = 0
        stable_count = 0

        while True:
            if not batch_file.exists():
                time.sleep(0.5)
                print(f"Wait 0.5 time!")
                continue

            try:
                current_size = batch_file.stat().st_size
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 2:  # 文件大小稳定3次检查
                        break
                else:
                    stable_count = 0
                    last_size = current_size

                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    print(f"等待批次 {idx} 文件完成超时")
                    break
            except Exception:
                print(f"Wait 0.5 time!")
                time.sleep(0.5)
                continue


class PreprocessedDataLoader:
    """
    使用预处理数据的高效DataLoader
    完全解耦的设计，直接使用pickle存储完整batch
    便于后续扩展batch内容而不需要修改存储逻辑
    """

    def __init__(self, batch_data_dir, num_workers=1, shuffle=True):
        """
        初始化预处理数据加载器

        Args:
            batch_data_dir: 预处理数据目录路径
            num_workers: worker进程数，用于并发加载
            shuffle: 是否随机打乱批次顺序
        """
        self.dataset = PreprocessedDataset(batch_data_dir, shuffle=shuffle)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.current_epoch = 0
        # 默认预取因子与 PyTorch 行为一致
        self.prefetch_factor = 4 if self.num_workers > 0 else 0

        # 为了兼容，记录一些属性
        self.batch_size = self.dataset.metadata['batch_size']

        print(f"PreprocessedDataLoader初始化完成:")
        print(f"  - 批次数量: {len(self.dataset)}")
        print(f"  - 批次大小: {self.batch_size}")
        print(f"  - Workers: {self.num_workers}")
        print(f"  - 存储格式: {self.dataset.metadata.get('storage_format', 'unknown')}")

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """
        迭代器接口，支持多worker并发加载
        """
        if self.shuffle and self.current_epoch > 0:
            # 在新的epoch开始时重新打乱
            self.dataset.reshuffle()

        if self.num_workers <= 1:
            # 单worker模式
            for i in range(len(self.dataset)):
                yield self.dataset[i]
        else:
            # 多worker并发模式，滑动窗口预取，保证按序输出
            total = len(self.dataset)
            if total == 0:
                return
            window = max(1, self.prefetch_factor * self.num_workers)
            next_submit = 0
            next_yield = 0
            futures = {}
            results = {}
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 初始提交
                initial = min(total, window)
                for i in range(initial):
                    futures[executor.submit(self.dataset.__getitem__, i)] = i
                    next_submit += 1

                while next_yield < total:
                    # 等任意一个完成
                    done_any = False
                    for future in list(futures.keys()):
                        if future.done():
                            idx = futures.pop(future)
                            try:
                                results[idx] = future.result()
                            except Exception as e:
                                print(f"Worker加载批次 {idx} 时发生错误: {e}")
                                raise
                            done_any = True
                    if not done_any:
                        # 主动阻塞等待一个完成（减少自旋）
                        future, idx = None, None
                        try:
                            future = next(iter(futures.keys()))
                        except StopIteration:
                            pass
                        if future is not None:
                            idx_wait = futures.pop(future)
                            try:
                                results[idx_wait] = future.result()
                            except Exception as e:
                                print(f"Worker加载批次 {idx_wait} 时发生错误: {e}")
                                raise

                    # 尝试按序输出
                    while next_yield in results:
                        yield results.pop(next_yield)
                        next_yield += 1

                        # 补提交新的任务，维持窗口
                        if next_submit < total:
                            idx_new = next_submit
                            futures[executor.submit(self.dataset.__getitem__, idx_new)] = idx_new
                            next_submit += 1

        self.current_epoch += 1

    def set_epoch(self, epoch):
        """设置当前epoch，用于确定性的shuffle"""
        self.current_epoch = epoch
        if self.shuffle:
            # 使用epoch作为随机种子确保确定性
            random.seed(epoch)
            self.dataset.reshuffle()
            random.seed()  # 恢复随机种子


class DynamicPreprocessedDataLoader:
    """
    动态预处理数据加载器 - 支持训练过程中动态等待预处理文件
    训练到第i个batch时，只需要预处理到第i个batch即可
    """

    def __init__(self, batch_data_dir, num_workers=1, shuffle=True, max_batches=None, wait_timeout=300, batch_size=768):
        """
        初始化动态预处理数据加载器

        Args:
            batch_data_dir: 预处理数据目录路径
            num_workers: worker进程数，用于并发加载
            shuffle: 是否随机打乱批次顺序
            max_batches: 最大批次数（如果为None，则动态检测）
            wait_timeout: 等待预处理文件的最大时间（秒）
        """
        self.dataset = DynamicPreprocessedDataset(batch_data_dir, max_batches, wait_timeout)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.current_epoch = 0
        self.wait_timeout = wait_timeout

        # 为了兼容，记录一些属性
        self.batch_size = batch_size  # 默认值

        print(f"DynamicPreprocessedDataLoader初始化完成:")
        print(f"  - 数据目录: {batch_data_dir}")
        print(f"  - 最大批次数: {self.dataset.max_batches}")
        print(f"  - 批次大小: {self.batch_size}")
        print(f"  - Workers: {self.num_workers}")
        print(f"  - 等待超时: {self.wait_timeout}秒")

    def __len__(self):
        """返回数据加载器长度"""
        return len(self.dataset)

    def __iter__(self):
        """
        动态迭代器接口，支持等待预处理文件
        """
        if self.shuffle and self.current_epoch > 0:
            # 在新的epoch开始时重新打乱
            if hasattr(self.dataset, 'reshuffle'):
                self.dataset.reshuffle()

        if self.num_workers <= 1:
            # 单worker模式 - 直接等待文件
            for i in range(len(self.dataset)):
                yield self.dataset[i]
        else:
            # 多worker并发模式，但需要处理文件可能不存在的情况
            total = len(self.dataset)
            if total == 0:
                print("数据集长度为0，等待预处理文件生成...")
                return

            # 使用更保守的预取窗口，避免同时读取太多不存在的文件
            window = max(1, min(8, self.num_workers))  # 限制窗口大小为2
            next_submit = 0
            next_yield = 0
            futures = {}
            results = {}

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 初始提交 - 只提交第一个批次
                initial = min(total, window)  # 只提交第一个批次
                for i in range(initial):
                    futures[executor.submit(self.dataset.__getitem__, i)] = i
                    next_submit += 1

                while next_yield < total:
                    # 等任意一个完成
                    done_any = False
                    for future in list(futures.keys()):
                        if future.done():
                            idx = futures.pop(future)
                            try:
                                results[idx] = future.result()
                            except Exception as e:
                                print(f"Worker加载批次 {idx} 时发生错误: {e}")
                                raise
                            done_any = True

                    if not done_any:
                        # 主动阻塞等待一个完成
                        future, idx = None, None
                        try:
                            future = next(iter(futures.keys()))
                        except StopIteration:
                            pass
                        if future is not None:
                            idx_wait = futures.pop(future)
                            try:
                                results[idx_wait] = future.result()
                            except Exception as e:
                                print(f"Worker加载批次 {idx_wait} 时发生错误: {e}")
                                raise

                    # 尝试按序输出
                    while next_yield in results:
                        yield results.pop(next_yield)
                        next_yield += 1

                        # 补提交新的任务，维持窗口
                        if next_submit < total and len(futures) < window:
                            idx_new = next_submit
                            futures[executor.submit(self.dataset.__getitem__, idx_new)] = idx_new
                            next_submit += 1

        self.current_epoch += 1

    def set_epoch(self, epoch):
        """设置当前epoch，用于确定性的shuffle"""
        self.current_epoch = epoch
        if self.shuffle and hasattr(self.dataset, 'reshuffle'):
            # 使用epoch作为随机种子确保确定性
            random.seed(epoch)
            self.dataset.reshuffle()
            random.seed()  # 恢复随机种子



