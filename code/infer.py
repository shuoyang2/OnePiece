import json
import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import OrderedDict
# (移除了 importlib.util)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
# (移除了 SummaryWriter)
from tqdm import tqdm
import random
# 修正：同时导入 MyDataset
from dataset import MyDataset, MyTestDataset
from model import BaselineModel
from main_dist import get_args


def get_ckpt_path():
    """
    Loads checkpoint path from MODEL_OUTPUT_PATH environment variable.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    # 增加一个 .pth 的检查，以防万一
    for item in os.listdir(ckpt_path):
        if item.endswith(".pth"):
            return os.path.join(ckpt_path, item)
    raise FileNotFoundError(f"No .pt or .pth checkpoint under {ckpt_path}")


class TrainInferDataset(MyTestDataset):
    """
    从 train_infer.py 移植的数据集。
    仅用于“训练资源上的推理”。
    - 继承 MyTestDataset 的取样与特征构造逻辑（不丢最后一个 item）。
    - 但读取 train 数据目录下的 seq.jsonl / seq_offsets.pkl。
    - 为避免 user_action_type 文件在训练目录缺失，做了兜底。
    """

    def __init__(self, data_dir, args):
        MyDataset.__init__(self, data_dir, args)
        uat_path = Path(self.data_dir, "user_action_type.json")
        try:
            import orjson
            if uat_path.exists():
                self.user_action_type = orjson.loads(open(uat_path, 'rb').read())
            else:
                self.user_action_type = {}
                print(f"No USER ACTION TYPE!!!")
        except Exception:
            self.user_action_type = {}

    def _load_data_and_offsets(self):
        self.data_file = open(Path(os.environ.get('USER_CACHE_PATH')) / "seq.jsonl", 'rb')
        import pickle as _pkl
        with open(Path(Path(os.environ.get('USER_CACHE_PATH')), 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = _pkl.load(f)

    def __len__(self):
        import pickle as _pkl
        with open(Path(Path(os.environ.get('USER_CACHE_PATH')), 'seq_offsets.pkl'), 'rb') as f:
            temp = _pkl.load(f)
        return len(temp)


def _build_idx2creative_id(test_dataset: TrainInferDataset) -> Dict[int, int]:
    """
    从 train_infer.py 移植的辅助函数。
    """
    creative_id2idx = test_dataset.indexer['i']
    try:
        idx2creative_id = {idx: int(cid) for cid, idx in creative_id2idx.items()}
    except Exception:
        idx2creative_id = {idx: cid for cid, idx in creative_id2idx.items()}
    return idx2creative_id


# (移除了 _shard_bounds 函数)


def _find_candidates_by_sid(sid1: int, sid2: int, retrieval_ids: List, dataset, retrieve_id2creative_id: Dict) -> List[
    int]:
    """
    从 train_infer.py 移植的辅助函数，用于 SID 重排。
    """
    matched_indices = []

    if not hasattr(dataset, 'sid'):
        return matched_indices

    sid_dict = dataset.sid
    for idx, retrieval_id in enumerate(retrieval_ids):
        creative_id = retrieve_id2creative_id.get(int(retrieval_id))
        if creative_id is None:
            continue
        item_sid = sid_dict.get(str(creative_id))
        if item_sid and len(item_sid) >= 2 and item_sid[0] == sid1 and item_sid[1] == sid2:
            matched_indices.append(idx)
    return matched_indices


# (移除了 _save_part 函数)


# --- infer.py 中原有的辅助函数 (保留) ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# (移除了 read_result_ids 和 struct)


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_all_candidate_embs_train(indexer, feat_types, feat_default_value, mm_emb_dict, model, dataset, batch_size=1024,
                                 candidate_path=None):
    """
    训练时生成候选库所有 item 的 id 和 embedding，从指定路径读取候选库
    (此函数在原 infer.py 中已存在，train_infer.py 依赖此函数)
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

    if candidate_path is None:
        # 修正：使用 USER_CACHE_PATH 构造路径
        candidate_path = Path(os.environ.get('USER_CACHE_PATH'), 'predict_set.jsonl')
    else:
        candidate_path = Path(candidate_path)

    # 检查 USER_CACHE_PATH 下的 predict_set.jsonl 是否存在
    if not candidate_path.exists():
        # 如果不存在，尝试从 EVAL_DATA_PATH 复制
        eval_candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
        if eval_candidate_path.exists():
            print(f"Copying predict_set.jsonl from {eval_candidate_path} to {candidate_path}...")
            import shutil
            shutil.copy(eval_candidate_path, candidate_path)
        else:
            raise FileNotFoundError(f"Candidate file not found at {candidate_path} or {eval_candidate_path}")

    item_ids, retrieval_ids, features = [], [], []
    retrieve_id2creative_id = {}

    if type(indexer) != type({}):
        for i in indexer:
            indexer = i

    print("Step 1: Loading and processing candidate item features from training cache...")
    cold_start_items = []

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer.get(creative_id, 0)
            if item_id == 0:
                cold_start_items.append(creative_id)
                continue
            feature = process_cold_start_feat(feature)
            feature = dataset.fill_missing_feat(feature, item_id, creative_id=creative_id)

            item_ids.append(item_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    all_embs_list = []
    model.eval()

    item_feats_type = set(
        feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual'] + feat_types[
            'context_item_sparse'])
    item_emb_type = feat_types['item_emb']

    with torch.no_grad():
        print("Step 2: Generating candidate embeddings in batches...")
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Generating item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=model.dev).unsqueeze(0)
            item_features = [features[start_idx:end_idx]]
            seq_dict = {}
            for k in item_feats_type:
                seq_dict[k] = model.feat2tensor(item_features, k)

            SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 4096, "86": 3584}
            for k in item_emb_type:
                emb_dim = SHAPE_DICT[k]
                seq_default_value = torch.zeros(emb_dim, dtype=torch.float32)
                batch_data_list = np.array([
                    [item.get(k, seq_default_value) for item in seq]
                    for seq in item_features
                ])
                seq_dict[k] = torch.tensor(batch_data_list, dtype=torch.float32)

            batch_emb = model.feat2emb(item_seq, seq_dict, include_user=False).squeeze(0)

            if model.similarity_function == 'cosine':
                batch_emb = batch_emb / (1e-8 + torch.norm(batch_emb, p=2, dim=-1, keepdim=True))

            all_embs_list.append(batch_emb)

    candidate_embs = torch.cat(all_embs_list, dim=0)

    return candidate_embs, retrieval_ids, retrieve_id2creative_id, set(cold_start_items)


def get_all_candidate_embs(indexer, feat_types, feat_default_value, mm_emb_dict, model, dataset, batch_size=1024):
    """
    (原 infer.py 中的函数，保留但不在此新流程中调用)
    """
    # ... (此函数代码与上一版本相同，此处省略以保持简洁) ...
    # 完整的 get_all_candidate_embs 代码...
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, retrieval_ids, features = [], [], []
    retrieve_id2creative_id = {}

    if type(indexer) != type({}):
        for i in indexer:
            indexer = i

    print("Step 1: Loading and processing candidate item features...")
    cold_start_items = []
    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer.get(creative_id, 0)
            if item_id == 0:
                cold_start_items.append(creative_id)
                continue

            feature = process_cold_start_feat(feature)
            feature = dataset.fill_missing_feat(feature, item_id, creative_id=creative_id)

            item_ids.append(item_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    all_embs_list = []
    model.eval()

    item_feats_type = set(
        feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual'] + feat_types[
            'context_item_sparse'])
    item_emb_type = feat_types['item_emb']
    with torch.no_grad():
        print("Step 2: Generating candidate embeddings in batches...")
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Generating item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=model.dev).unsqueeze(0)
            item_features = [features[start_idx:end_idx]]
            seq_dict = {}
            for k in item_feats_type:
                seq_dict[k] = model.feat2tensor(item_features, k)

            SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 4096, "86": 3584}
            for k in item_emb_type:
                emb_dim = SHAPE_DICT[k]
                seq_default_value = torch.zeros(emb_dim, dtype=torch.float32)
                batch_data_list = np.array([
                    [item.get(k, seq_default_value) for item in seq]
                    for seq in item_features
                ])
                seq_dict[k] = torch.tensor(batch_data_list, dtype=torch.float32)

            batch_emb = model.feat2emb(item_seq, seq_dict, include_user=False).squeeze(0)

            if model.similarity_function == 'cosine':
                batch_emb = batch_emb / (1e-8 + torch.norm(batch_emb, p=2, dim=-1, keepdim=True))

            all_embs_list.append(batch_emb)

    candidate_embs = torch.cat(all_embs_list, dim=0)
    return candidate_embs, retrieval_ids, retrieve_id2creative_id, set(cold_start_items)


# ==========================================================
# 替换 infer() 函数
# ==========================================================

def infer():
    """
    基于 train_infer.py 的 main() 函数修改而来。
    - 仅执行标准推理 (Mode 0)。
    - 移除所有 .pkl 保存、TensorBoard 日志、多模式（avg/agg/sid）逻辑。
    - 假设在单卡上运行 (移除了 sharding 逻辑)。
    - 最终返回 all_top10s 和 all_user_ids。
    """

    # 1) 读取基础训练参数
    args = get_args()
    args.mode = "infer"
    args.feature_dropout_rate = 0
    print(f"Dropout rate set to: {args.feature_dropout_rate}")

    # 3) 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    args.device = device

    # (移除了 TensorBoard Writer)

    # 4) 构造“训练端推理数据集”
    data_path = os.environ.get('EVAL_DATA_PATH')
    dataset = TrainInferDataset(data_path, args)

    print(f"Running inference for all {len(dataset)} users...")

    current_batch_size = 2048

    test_loader = DataLoader(
        dataset,  # 直接使用完整的数据集
        batch_size=current_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=dataset.collate_fn,
        prefetch_factor=8
    )

    # 5) 构造模型并加载 checkpoint
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    candidate_path = Path(args.user_cache_path) / 'predict_set.jsonl'

    # --- 仅保留标准推理 (Mode 0) 逻辑 ---
    print("INFO: Entering Standard Inference Mode.")

    # 5.1) 构造模型并加载 checkpoint
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()
    ckpt_path = get_ckpt_path()  # 使用新的函数
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading standard model from {ckpt_path}: {e}")
        raise RuntimeError(f"Failed to load standard model from {ckpt_path}") from e

    # 6) 候选库 embedding
    print("Loading candidate embeddings for the standard model...")
    candidate_embs, retrieval_ids, retrieve_id2creative_id, _ = get_all_candidate_embs_train(
        dataset.indexer['i'], dataset.feature_types, dataset.feature_default_value, dataset.mm_emb_dict,
        model, dataset, batch_size=args.batch_size, candidate_path=str(candidate_path)
    )
    print("Candidate embeddings loaded.")

    # 7) 准备映射表和 logp
    candidate_creative_ids = [retrieve_id2creative_id.get(int(rid)) for rid in retrieval_ids]
    creative_id_to_candidate_idx = {int(cid): idx for idx, cid in enumerate(candidate_creative_ids) if cid is not None}
    idx2creative_id = _build_idx2creative_id(dataset)

    candidate_log_p = []
    for cid in candidate_creative_ids:
        if cid is not None:
            log_p = dataset.item_log_p.get(cid, dataset.min_log_p)
        else:
            log_p = dataset.min_log_p
        candidate_log_p.append(log_p)
    candidate_log_p = torch.tensor(candidate_log_p, dtype=torch.float32, device=device)
    print(f"Candidate log_p stats: min={candidate_log_p.min().item():.4f}, max={candidate_log_p.max().item():.4f}")

    # (移除了 Beam Search Generation / SID Legality 逻辑)

    top_k = 10

    # 初始化用于存储最终结果的列表
    all_top10_creative_ids: List[List[Any]] = []
    all_user_ids: List[Any] = []

    # (移除了所有统计变量)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Processing user batches"):

            seq, token_type, seq_feat, user_ids_in_batch, next_action_type = batch

            seq = seq.to(device)
            token_type = token_type.to(device)
            if next_action_type is not None:
                next_action_type = next_action_type.to(device)

            # (移除了 sid_resort 和 beam_search_generate 逻辑)

            # --- A/B) 核心分数计算 ---
            query_batch, mlp_logfeats, attention_mask = model.predict(seq, seq_feat, token_type, next_action_type)
            batch_scores = torch.matmul(query_batch, candidate_embs.T)

            # --- C) logp 修正 ---
            if args.infer_logq:
                batch_scores_corrected = batch_scores / args.infonce_temp
                batch_scores_corrected = batch_scores_corrected - candidate_log_p.unsqueeze(0)
            else:
                batch_scores_corrected = batch_scores

            # D) 历史行为过滤
            user_sequences = seq.cpu().numpy()
            for i in range(len(user_ids_in_batch)):
                history_indices = user_sequences[i]
                history_cids = {idx2creative_id.get(idx) for idx in history_indices if
                                idx != 0 and idx in idx2creative_id}
                candidate_indices_to_mask = [creative_id_to_candidate_idx.get(cid) for cid in history_cids if
                                             cid in creative_id_to_candidate_idx]
                if candidate_indices_to_mask:
                    batch_scores_corrected[i, candidate_indices_to_mask] = -torch.inf

            # E) 获取 ANN Top-K 结果
            _, ann_topk_indices_batch = torch.topk(batch_scores_corrected, k=top_k, dim=1)

            # F) 最终的topk结果初始化为ANN的结果
            final_topk_indices_batch = ann_topk_indices_batch.clone()

            # ====== G) Reward模型重排 ======
            if model.args.reward and hasattr(model, 'reward_model'):
                topk_reward = 32
                _, reward_topk_indices_batch = torch.topk(batch_scores_corrected, k=topk_reward, dim=1)
                batch_size_current = seq.shape[0]

                expanded_candidate_embs = candidate_embs[reward_topk_indices_batch]
                seq_embs_for_reward = mlp_logfeats[:, -1:, :].expand(-1, topk_reward, -1)
                ann_scores_for_reward = batch_scores_corrected.gather(1, reward_topk_indices_batch)
                reward_attn_mask = torch.ones(batch_size_current, topk_reward, topk_reward, dtype=torch.bool,
                                              device=device)

                reward_scores = model.reward_model_ctr(
                    seq_embs_for_reward,
                    expanded_candidate_embs,
                    reward_attn_mask,
                    ann_scores=ann_scores_for_reward,
                    sid1_probs=None,
                    sid2_probs=None
                ).squeeze(-1)

                _, reward_rerank_indices = torch.topk(reward_scores, k=top_k, dim=1)
                reward_topk_indices_batch = reward_topk_indices_batch.gather(1, reward_rerank_indices)

                # (移除了 Reward 统计)

                final_topk_indices_batch = reward_topk_indices_batch

            # ====== H) SID Beam Search重排 ======
            elif model.args.sid and hasattr(model, 'beamsearch_sid'):
                batch_size_current = seq.shape[0]

                batch_mask = (seq != 0).long()
                log_feats, _, mlp_logfeats_full, sid_logfeats = model.log2feats(
                    seq, batch_mask, seq_feat, next_action_type
                )

                sid_sequences, sid_scores = model.beamsearch_sid(
                    sid_logfeats,
                    top_k=5,
                    top_k_2=10
                )

                sid_topk_indices_list = []
                # (移除了 SID 统计)

                for i in range(batch_size_current):
                    user_sid_seqs = sid_sequences[i]
                    user_sid_scores = sid_scores[i]

                    sid_matched_indices = []
                    sid_matched_scores = []
                    for j in range(user_sid_seqs.shape[0]):
                        sid1 = user_sid_seqs[j, 0].item()
                        sid2 = user_sid_seqs[j, 1].item()
                        sid_score = user_sid_scores[j].item()

                        matched_indices_list = _find_candidates_by_sid(
                            sid1, sid2, retrieval_ids, dataset, retrieve_id2creative_id
                        )
                        for matched_idx in matched_indices_list:
                            sid_matched_indices.append(matched_idx)
                            sid_matched_scores.append(sid_score)

                    ann_top10_list = ann_topk_indices_batch[i].cpu().numpy().tolist()
                    combined_candidates = []
                    combined_scores = []
                    seen_indices = set()

                    for idx, score in zip(sid_matched_indices, sid_matched_scores):
                        if idx not in seen_indices:
                            combined_candidates.append(idx)
                            combined_scores.append(score)
                            seen_indices.add(idx)

                    for j, idx in enumerate(ann_top10_list):
                        if idx not in seen_indices:
                            combined_candidates.append(idx)
                            ann_score = batch_scores_corrected[i, idx].item()
                            combined_scores.append(ann_score)
                            seen_indices.add(idx)

                    if len(combined_candidates) > 0:
                        sorted_pairs = sorted(
                            zip(combined_candidates, combined_scores),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        final_top10 = [pair[0] for pair in sorted_pairs[:top_k]]
                    else:
                        final_top10 = ann_top10_list[:top_k]

                    # (移除了 SID 统计累加)
                    sid_topk_indices_list.append(torch.tensor(final_top10, dtype=torch.long))

                # (移除了 SID 统计的 TensorBoard 记录)

                sid_topk_indices_batch = torch.stack(sid_topk_indices_list, dim=0).to(device)
                final_topk_indices_batch = sid_topk_indices_batch

            # I) 保存当前批次的结果
            # (修改：不再保存索引，而是直接映射为 creative_id 并保存)

            all_user_ids.extend(list(user_ids_in_batch))

            # 将 final_topk_indices_batch (GPU tensor) 转换为 creative ID 列表 (CPU)
            topk_indices_np = final_topk_indices_batch.cpu().numpy()

            # 确保 retrieval_ids_np 只被转换一次
            if 'retrieval_ids_np' not in locals():
                retrieval_ids_np = np.array(retrieval_ids)

            topk_retrieval_ids = retrieval_ids_np[topk_indices_np]

            batch_top10s: List[List[Any]] = [
                [retrieve_id2creative_id.get(int(rid), 0) for rid in user_top_k]
                for user_top_k in topk_retrieval_ids
            ]
            all_top10_creative_ids.extend(batch_top10s)

            # J/K) (移除了速度统计、日志打印、全局步数增加)

    # 9) 聚合所有批次的结果并返回
    # (移除了所有保存、统计、日志逻辑)

    print(f"\nInference complete.")
    print(f"Total users processed: {len(all_user_ids)}")
    print(f"Total recommendations generated: {len(all_top10_creative_ids)}")

    # 最终返回
    return all_top10_creative_ids, all_user_ids

# (移除了 if __name__ == '__main__':)
# 使得 infer() 函数可以像原 infer.py 一样被其他脚本导入和调用

