import os
import json
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 复用现有模块，确保随模型/数据集改动自动适配
from main_dist import get_args
from dataset import MyDataset, MyTestDataset
from model import BaselineModel
from infer import get_all_candidate_embs_train


def _get_ckpt_path(name) -> str:
    # 修正：允许 name 直接就是 user_cache_path / train_infer / run_name 的形式
    # 同时也兼容旧的 run_name 形式
    base_path = Path(os.environ.get("USER_CACHE_PATH")) / "train_infer"

    # 尝试将 name 视为相对路径
    ckpt_path = base_path / name

    # 如果 name 不是一个目录（旧逻辑）
    if not ckpt_path.is_dir():
        # 尝试从 USER_CACHE_PATH 根目录查找
        ckpt_path_alt = Path(os.environ.get("USER_CACHE_PATH")) / name
        if ckpt_path_alt.is_dir():
            ckpt_path = ckpt_path_alt
        elif (base_path.parent / name).is_dir():  # 尝试 user_cache_path / name
            ckpt_path = base_path.parent / name
        else:
            # 回退到旧的逻辑：name 是 run_name
            ckpt_path = base_path / name
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Cannot find checkpoint directory for name: {name} under {base_path} or {base_path.parent}")

    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set or ckpt_path is None")

    for item in os.listdir(ckpt_path):
        if item.endswith(".pth"):
            return str(Path(ckpt_path) / item)
    raise FileNotFoundError(f"No .pth checkpoint under {ckpt_path}")


class TrainInferDataset(MyTestDataset):
    """
    仅用于“训练资源上的推理”。
    - 继承 MyTestDataset 的取样与特征构造逻辑（不丢最后一个 item）。
    - 但读取 train 数据目录下的 seq.jsonl / seq_offsets.pkl。
    - 为避免 user_action_type 文件在训练目录缺失，做了兜底。
    """

    def __init__(self, data_dir, args):
        # 只执行 MyDataset.__init__ 完成通用初始化，避免 MyTestDataset.__init__ 强制读取 predict_* 文件
        MyDataset.__init__(self, data_dir, args)
        # 兜底加载 user_action_type
        uat_path = Path(self.data_dir, "user_action_type.json")
        try:
            import orjson
            if uat_path.exists():
                self.user_action_type = orjson.loads(open(uat_path, 'rb').read())
            else:
                # 默认全部使用 1（点击）类型，避免下游计算报错
                self.user_action_type = {}
                print(f"No USER ACTION TYPE!!!")
        except Exception:
            self.user_action_type = {}

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        import pickle as _pkl
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = _pkl.load(f)

    def __len__(self):
        import pickle as _pkl
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            temp = _pkl.load(f)
        return len(temp)


def _build_idx2creative_id(test_dataset: TrainInferDataset) -> Dict[int, int]:
    creative_id2idx = test_dataset.indexer['i']
    try:
        idx2creative_id = {idx: int(cid) for cid, idx in creative_id2idx.items()}
    except Exception:
        idx2creative_id = {idx: cid for cid, idx in creative_id2idx.items()}
    return idx2creative_id


def _shard_bounds(total: int, num_shards: int, shard_id: int) -> Tuple[int, int]:
    per = total // num_shards
    rem = total % num_shards
    start = shard_id * per + min(shard_id, rem)
    end = start + per + (1 if shard_id < rem else 0)
    return start, end


def _find_candidates_by_sid(sid1: int, sid2: int, retrieval_ids: List, dataset, retrieve_id2creative_id: Dict) -> List[
    int]:
    """
    根据SID在候选库中查找匹配的item索引

    Args:
        sid1: 第一层SID
        sid2: 第二层SID
        retrieval_ids: 候选库中所有item的retrieval id列表
        dataset: 数据集对象，包含item的SID映射信息
        retrieve_id2creative_id: 从 retrieval_id 到 creative_id (即 itemid) 的映射

    Returns:
        匹配的候选索引列表（在retrieval_ids中的索引）
    """
    matched_indices = []

    # 检查dataset中是否有sid属性
    if not hasattr(dataset, 'sid'):
        return matched_indices

    sid_dict = dataset.sid  # sid_dict 的 key 是 itemid(creative_id)

    # 遍历候选库，查找匹配的items
    for idx, retrieval_id in enumerate(retrieval_ids):
        # 1. 从 retrieval_id 映射到 creative_id (itemid)
        # retrieval_id 可能是字符串或数字，确保转换
        creative_id = retrieve_id2creative_id.get(int(retrieval_id))
        if creative_id is None:
            continue

        # 2. 使用 creative_id 从 sid_dict 获取该 item 的 sid 信息
        # 根据 dataset.py, creative_id (item_id) 是字符串
        item_sid = sid_dict.get(str(creative_id))

        # 3. 检查是否匹配
        if item_sid and len(item_sid) >= 2 and item_sid[0] == sid1 and item_sid[1] == sid2:
            matched_indices.append(idx)

    return matched_indices


def _save_part(out_dir: Path, shard_id: int, user_list: List[Any], top10s: List[List[Any]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"part_{shard_id:03d}.pkl"
    with open(part_path, 'wb') as f:
        pickle.dump({
            'user_ids': user_list,
            'top10s': top10s,
        }, f)
    meta = {
        'num_users': len(user_list),
        'shard_id': shard_id,
    }
    with open(out_dir / f"part_{shard_id:03d}.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)


def calculate_sid_similarity_with_correct_top10s(generated_sids_tensor, generated_scores_tensor,
                                                 user_ids, dataset, max_users=10000):
    """
    计算生成SID对应的CID与correct_top10s的相似度
    确保top10 SID都在predict_set中，如果不够10个就继续向后查找

    Args:
        generated_sids_tensor: 生成的SID张量 [N, K, 2]
        generated_scores_tensor: 生成的SID分数张量 [N, K]
        user_ids: 用户ID列表
        dataset: 数据集对象，包含sid_reverse映射
        max_users: 最大计算用户数，默认10000

    Returns:
        dict: 包含相似度统计的字典
    """
    print(f"开始计算前{max_users}个用户的SID相似度...")

    # 限制用户数量
    actual_users = min(max_users, len(user_ids))
    print(f"实际计算用户数: {actual_users}")

    # 加载predict_set，构建creative_id集合用于过滤
    predict_set_creative_ids = set()
    try:
        predict_set_path = Path(os.environ.get("USER_CACHE_PATH")) / 'predict_set.jsonl'
        if predict_set_path.exists():
            print("加载predict_set.jsonl...")
            with open(predict_set_path, 'r') as f:
                for line in f:
                    line_data = json.loads(line)
                    creative_id = int(line_data['creative_id'])
                    predict_set_creative_ids.add(creative_id)
            print(f"predict_set包含{len(predict_set_creative_ids)}个creative_id")
        else:
            print("未找到predict_set.jsonl文件，将跳过SID过滤")
    except Exception as e:
        print(f"加载predict_set时出错: {e}")
        print("将跳过SID过滤")

    # 加载correct_top10s和correct_user_id
    try:
        correct_top10s_path = Path(os.environ.get("USER_CACHE_PATH")) / 'correct_top10s.pkl'
        correct_user_id_path_a = Path(os.environ.get("USER_CACHE_PATH")) / 'correct_user_id.pkl.pkl'
        correct_user_id_path_b = Path(os.environ.get("USER_CACHE_PATH")) / 'correct_user_id.pkl'

        if not correct_top10s_path.exists():
            print("未找到correct_top10s.pkl文件，跳过相似度计算")
            return None

        with open(correct_top10s_path, 'rb') as f:
            correct_top10s = pickle.load(f)

        # 确定正确的用户ID文件路径
        correct_user_id_path = None
        if correct_user_id_path_a.exists():
            correct_user_id_path = correct_user_id_path_a
        elif correct_user_id_path_b.exists():
            correct_user_id_path = correct_user_id_path_b

        if not correct_user_id_path:
            print("未找到correct_user_id文件，跳过相似度计算")
            return None

        with open(correct_user_id_path, 'rb') as f:
            correct_user_ids = pickle.load(f)

        print(f"加载了{len(correct_top10s)}个correct_top10s和{len(correct_user_ids)}个correct_user_ids")

    except Exception as e:
        print(f"加载correct文件时出错: {e}")
        return None

    # 构建用户ID到索引的映射
    uid2idx = {u: i for i, u in enumerate(correct_user_ids)}

    # 统计变量
    total_overlaps = []
    valid_users_count = 0
    total_jaccard_scores = []
    total_filtered_sids = 0  # 统计被过滤掉的SID数量
    total_valid_sids = 0  # 统计有效的SID数量

    print("开始计算相似度...")

    # 处理前max_users个用户
    for i in range(actual_users):
        user_id = user_ids[i]

        # 检查用户是否在correct数据中
        if user_id not in uid2idx:
            continue

        # 获取该用户的correct_top10
        correct_top10 = correct_top10s[uid2idx[user_id]]
        correct_set = set(correct_top10)

        # 获取该用户生成的SID
        user_sids = generated_sids_tensor[i]  # [K, 2]
        user_scores = generated_scores_tensor[i]  # [K]

        # 将SID转换为CID，并确保都在predict_set中
        generated_cids = []
        user_valid_sids = 0
        user_filtered_sids = 0

        for j in range(user_sids.shape[0]):
            sid1 = int(user_sids[j, 0])
            sid2 = int(user_sids[j, 1])
            sid_key = f"{sid1}_{sid2}"
            cid = dataset.sid_reverse.get(sid_key, 0)

            if cid != 0:
                # 检查CID是否在predict_set中
                if not predict_set_creative_ids or int(cid) in predict_set_creative_ids:
                    generated_cids.append(cid)
                    user_valid_sids += 1
                else:
                    user_filtered_sids += 1
            else:
                user_filtered_sids += 1

            # 如果已经找到10个有效的CID，就停止
            if len(generated_cids) >= 10:
                break

        # 更新统计信息
        total_valid_sids += user_valid_sids
        total_filtered_sids += user_filtered_sids

        # 取前10个作为top10
        generated_top10 = generated_cids[:10]
        if len(generated_top10) < 10:
            # 如果不够10个，用0填充
            generated_top10.extend([0] * (10 - len(generated_top10)))

        generated_set = set(generated_top10)

        # 计算重叠数量
        overlap_count = len(correct_set.intersection(generated_set))
        total_overlaps.append(overlap_count)

        # 计算Jaccard相似度
        union_size = len(correct_set.union(generated_set))
        jaccard_score = overlap_count / union_size if union_size > 0 else 0.0
        total_jaccard_scores.append(jaccard_score)

        valid_users_count += 1

        # 每1000个用户打印一次进度
        if (i + 1) % 1000 == 0:
            avg_overlap = sum(total_overlaps) / len(total_overlaps) if total_overlaps else 0
            avg_jaccard = sum(total_jaccard_scores) / len(total_jaccard_scores) if total_jaccard_scores else 0
            print(f"已处理{i + 1}个用户，平均重叠数: {avg_overlap:.2f}, 平均Jaccard: {avg_jaccard:.4f}")

    # 计算最终统计
    if valid_users_count > 0:
        avg_overlap = sum(total_overlaps) / len(total_overlaps)
        avg_jaccard = sum(total_jaccard_scores) / len(total_jaccard_scores)
        max_overlap = max(total_overlaps)
        max_jaccard = max(total_jaccard_scores)

        print(f"\n===== SID生成相似度统计 (前{actual_users}个用户) =====")
        print(f"有效用户数: {valid_users_count}")
        print(f"平均重叠数: {avg_overlap:.2f}/10")
        print(f"平均Jaccard相似度: {avg_jaccard:.4f}")
        print(f"最大重叠数: {max_overlap}/10")
        print(f"最大Jaccard相似度: {max_jaccard:.4f}")
        print(f"SID过滤统计:")
        print(f"  有效SID数量: {total_valid_sids}")
        print(f"  被过滤SID数量: {total_filtered_sids}")
        if total_valid_sids + total_filtered_sids > 0:
            filter_rate = total_filtered_sids / (total_valid_sids + total_filtered_sids) * 100
            print(f"  过滤率: {filter_rate:.2f}%")
        print(f"==========================================\n")

        return {
            'valid_users': valid_users_count,
            'avg_overlap': avg_overlap,
            'avg_jaccard': avg_jaccard,
            'max_overlap': max_overlap,
            'max_jaccard': max_jaccard,
            'total_overlaps': total_overlaps,
            'total_jaccard_scores': total_jaccard_scores
        }
    else:
        print("没有找到有效的用户进行相似度计算")
        return None


def main():
    # 1) 读取基础训练参数（模型/特征/路径等）
    args = get_args()
    args.mode = "infer"
    args.feature_dropout_rate = 0
    print(args.feature_dropout_rate)

    # ====== 新增功能控制变量 ======
    # 注意：这两个列表不能同时非空。

    # 1. 模型参数平均:
    #    填入 'run_name' 列表, e.g., ['exp_run_1', 'exp_run_2']
    #    程序会从 args.user_cache_path / "train_infer" / run_name / *.pth 查找
    model_agg_paths: List[str] = []

    # 2. 模型分数聚合:
    #    填入 'run_name' 列表, e.g., ['model_A', 'model_B']
    score_agg_paths: List[str] = []
    # ================================

    # 检查互斥条件
    if model_agg_paths and score_agg_paths:
        print("警告: model_agg_paths 和 score_agg_paths 不能同时非空。")
        print("将优先执行功能 1 (模型参数平均)。")
        score_agg_paths = []  # 强制清空

    # 2) 从 Bash 环境读取并行信息：NUM_SHARDS, SHARD_ID, 以及任务名 NAME
    num_shards = int(os.environ.get('NUM_SHARDS', '1'))
    shard_id = int(os.environ.get('SHARD_ID', '0'))
    run_name = args.train_infer_result_path  # 默认输出路径

    # 3) 设备：要求外部用 CUDA_VISIBLE_DEVICES 指定单卡
    # 若未设置，则按 args.device 使用默认 cuda:0
    device = args.device if torch.cuda.is_available() else 'cpu'
    args.device = device

    # 4) 初始化 TensorBoard Writer（每个shard都有自己的writer）
    tb_path = args.tb_path
    tb_infer_path = Path(tb_path)
    tb_infer_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_infer_path))
    print(f"✓ TensorBoard writer initialized at: {tb_infer_path}")

    # 4) 构造“训练端推理数据集”（完整序列）并切分
    data_path = args.data_path
    dataset = TrainInferDataset(data_path, args)
    total_len = len(dataset)
    start, end = _shard_bounds(total_len, num_shards, shard_id)
    indices = list(range(start, end))
    subset = Subset(dataset, indices)

    # 默认 batch_size
    current_batch_size = 2048
    if score_agg_paths:
        # 功能 2：分数聚合，减小 batch_size
        current_batch_size = 512  # 减小 batch size
        print(f"INFO: Mode 2 (Score Agg) detected, reducing batch size to {current_batch_size}")

    test_loader = DataLoader(
        subset, batch_size=current_batch_size, shuffle=False, num_workers=2,
        collate_fn=dataset.collate_fn, prefetch_factor=8
    )

    # 5) 构造模型并加载 checkpoint
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    candidate_path = Path(args.user_cache_path) / 'predict_set.jsonl'

    # --- 模式选择与模型/候选库加载 ---

    models_for_scoring: List[BaselineModel] = []
    candidate_embs_list: List[torch.Tensor] = []
    model: BaselineModel = None
    candidate_embs: torch.Tensor = None
    retrieval_ids: List = None
    retrieve_id2creative_id: Dict = None

    if model_agg_paths:
        # ====== 功能 1: 模型参数平均 ======
        print(f"INFO: Entering Mode 1 - Model Parameter Averaging from {len(model_agg_paths)} models.")

        # 1.1) 在 CPU 上初始化模型
        model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)  # 在 CPU
        model.eval()

        # 1.2) 加载所有 state_dicts 到 CPU
        all_state_dicts = []
        for path_name in model_agg_paths:
            try:
                ckpt_path = _get_ckpt_path(path_name)
                print(f"Loading model for averaging: {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
                all_state_dicts.append(state_dict)
            except Exception as e:
                print(f"Error loading model {path_name}: {e}. Skipping.")

        if not all_state_dicts:
            raise ValueError("model_agg_paths was provided, but no models could be loaded.")

        # 1.3) 平均参数
        num_models = len(all_state_dicts)
        print(f"Averaging parameters from {num_models} models...")

        # 使用 OrderedDict 保证顺序
        avg_state_dict = OrderedDict()
        base_state_dict = all_state_dicts[0]

        for key in base_state_dict.keys():
            if all(key in sd for sd in all_state_dicts):
                # 确保所有张量都在CPU上
                sum_tensor = torch.zeros_like(base_state_dict[key], dtype=torch.float32, device='cpu')
                for sd in all_state_dicts:
                    sum_tensor += sd[key].cpu().float()

                # 求平均并转回原始类型
                avg_state_dict[key] = (sum_tensor / num_models).to(base_state_dict[key].dtype)
            else:
                print(f"Warning: Key {key} not found in all models. Skipping this key.")

        # 1.4) 加载平均后的参数并移到 device
        model.load_state_dict(avg_state_dict)
        model.to(args.device)
        print("Model loaded with averaged parameters and moved to device.")

        # 1.5) 加载这个平均模型的候选库
        print("Loading candidate embeddings for the averaged model...")
        candidate_embs, retrieval_ids, retrieve_id2creative_id, _ = get_all_candidate_embs_train(
            dataset.indexer['i'], dataset.feature_types, dataset.feature_default_value, dataset.mm_emb_dict,
            model, dataset, batch_size=2048, candidate_path=str(candidate_path)
        )
        print("Candidate embeddings loaded.")

    elif score_agg_paths:
        # ====== 功能 2: 模型分数聚合 ======
        print(f"INFO: Entering Mode 2 - Model Score Aggregation from {len(score_agg_paths)} models.")

        # 2.1) 确保 test_loader 已使用较小的 batch_size (在前面已处理)

        # 2.2) 加载所有模型及其对应的候选库 embeddings
        for path_name in score_agg_paths:
            try:
                ckpt_path = _get_ckpt_path(path_name)
                print(f"Loading model for scoring: {ckpt_path}")
                # 加载模型
                m = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
                m.eval()
                checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
                m.load_state_dict(checkpoint)
                models_for_scoring.append(m)

                # 加载此模型对应的候选库
                print(f"Loading candidate embs for model: {path_name}")
                c_embs, r_ids, r_id2c_id, _ = get_all_candidate_embs_train(
                    dataset.indexer['i'], dataset.feature_types, dataset.feature_default_value, dataset.mm_emb_dict,
                    m, dataset, batch_size=2048, candidate_path=str(candidate_path)
                )
                # 确保候选库在 device 上
                candidate_embs_list.append(c_embs.to(args.device))

                # 保存第一组候选库元信息
                if retrieval_ids is None:
                    retrieval_ids = r_ids
                    retrieve_id2creative_id = r_id2c_id

            except Exception as e:
                print(f"Error loading model or candidates {path_name}: {e}. Skipping.")

        if not models_for_scoring:
            raise ValueError("score_agg_paths was provided, but no models could be loaded.")

        print(f"Loaded {len(models_for_scoring)} models and their candidate embeddings for score aggregation.")

        # (model 和 candidate_embs 保持为 None)

    elif args.sid_resort:
        # ====== 功能 3: SID Resort 模式 ======
        print("INFO: Entering Mode 3 - SID Resort Mode.")

        # 3.1) 构造模型并加载 checkpoint
        model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
        model.eval()
        ckpt_path = _get_ckpt_path(run_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint)

        # [删除] SID文件加载逻辑 (将移动到外面)

        # 3.4) 加载候选库embedding
        print("Loading candidate embeddings for SID resort mode...")
        candidate_embs, retrieval_ids, retrieve_id2creative_id, _ = get_all_candidate_embs_train(
            dataset.indexer['i'], dataset.feature_types, dataset.feature_default_value, dataset.mm_emb_dict,
            model, dataset, batch_size=2048, candidate_path=str(candidate_path)
        )
        print("Candidate embeddings loaded.")

    else:
        # ====== 功能 0: 原始推理逻辑 ======
        print("INFO: Entering Mode 0 - Standard Inference.")

        # 5.1) 构造模型并加载 checkpoint
        model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
        model.eval()
        ckpt_path = _get_ckpt_path(run_name)
        try:
            checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
            model.load_state_dict(checkpoint)
        except Exception as e:
            # [修改]：标准模式加载失败也报错
            print(f"Error loading standard model from {ckpt_path}: {e}")
            raise RuntimeError(f"Failed to load standard model: {run_name}") from e

        # 6) 候选库 embedding (仅在非 beam search 模式下加载)
        if not args.beam_search_generate:
            print("Loading candidate embeddings for the standard model...")
            candidate_embs, retrieval_ids, retrieve_id2creative_id, _ = get_all_candidate_embs_train(
                dataset.indexer['i'], dataset.feature_types, dataset.feature_default_value, dataset.mm_emb_dict,
                model, dataset, batch_size=2048, candidate_path=str(candidate_path)
            )
            print("Candidate embeddings loaded.")

    # --- 模式加载结束 ---

    # [修改] 仅在 ANN 模式下计算
    if not args.beam_search_generate:
        # creative_id -> 索引 映射，用于去历史去重
        candidate_creative_ids = [retrieve_id2creative_id.get(int(rid)) for rid in retrieval_ids]
        creative_id_to_candidate_idx = {int(cid): idx for idx, cid in enumerate(candidate_creative_ids) if
                                        cid is not None}

        # 7) 获取候选库中所有 item 的 log 修正概率
        candidate_log_p = []
        for cid in candidate_creative_ids:
            if cid is not None:
                log_p = dataset.item_log_p.get(cid, dataset.min_log_p)
            else:
                log_p = dataset.min_log_p
            candidate_log_p.append(log_p)

        candidate_log_p = torch.tensor(candidate_log_p, dtype=torch.float32, device=device)
        print(
            f"候选库 log_p 统计: min={candidate_log_p.min().item():.4f}, max={candidate_log_p.max().item():.4f}, mean={candidate_log_p.mean().item():.4f}")

    # +++ 新增：加载 SID 数据 (如果 args.sid_resort 为 True, 无论哪种模式) +++
    user_id_to_sid_cids = {}
    if args.sid_resort:
        print(f"INFO: args.sid_resort=True. Loading SID data for filtering...")
        if args.sid_path is None:
            # [修改]：sid_path 在 Mode 3 (sid_resort) 中已检查，但在 Mode 2 (score_agg) 中未检查
            # 如果在 Mode 2 中，sid_path 为 None，则发出警告并禁用 sid_resort
            if score_agg_paths:
                print("WARNING: args.sid_resort is True but sid_path is None. Disabling SID resort for score_agg.")
                args.sid_resort = False  # 禁用它
            else:
                raise ValueError("sid_path must be specified when using sid_resort mode")

        if args.sid_resort:  # 再次检查，因为它可能已被禁用
            sid_base_path = Path(args.user_cache_path) / 'train_infer' / args.sid_path

            # 读取所有part文件
            all_sid_user_ids = []
            all_sid_generated_cids = []

            print(f"Loading all SID files from {sid_base_path}...")
            part_files = sorted(sid_base_path.glob("part_*.pkl"))

            if not part_files:
                raise FileNotFoundError(f"No SID part files found in {sid_base_path}")

            for part_file in part_files:
                # print(f"Loading SID data from {part_file}...") # 注释掉
                with open(part_file, 'rb') as f:
                    sid_data = pickle.load(f)

                all_sid_user_ids.extend(sid_data['user_ids'])
                all_sid_generated_cids.extend(sid_data['generated_cids'])

            # 转换为numpy数组
            all_sid_generated_cids = np.array(all_sid_generated_cids)  # [Total_N, K]

            print(
                f"Loaded SID data for {len(all_sid_user_ids)} users with {all_sid_generated_cids.shape[1]} candidates per user")

            # +++ 新增：构建用户ID到SID生成结果的映射 (全局) +++
            for user_idx, user_id in enumerate(all_sid_user_ids):
                user_id_to_sid_cids[user_id] = all_sid_generated_cids[user_idx]
            print(f"Built SID mapping for {len(user_id_to_sid_cids)} users")

    # idx->creative_id (所有模式都需要)
    idx2creative_id = _build_idx2creative_id(dataset)

    # ====== 新增：加载用于 Beam Search 合法率校验的 SID 集合 ======
    # 修正逻辑：只从候选集 (predict_set.jsonl) 中加载合法的 SID
    legal_sid_set = None
    full_sid_set = None  # 新增：用于存储整个训练数据集的SID集合
    if args.beam_search_generate:  # 仅在需要时加载
        print("INFO: Building legal SID set from candidate set (predict_set.jsonl)...")

        # 1. 从 predict_set.jsonl 加载 creative_id
        candidate_creative_ids = set()
        if candidate_path.exists():
            with open(candidate_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # creative_id 在 dataset.sid 中是字符串 key
                        candidate_creative_ids.add(data['creative_id'])
                    except (json.JSONDecodeError, KeyError):
                        continue  # 跳过格式不正确的行
        else:
            print(f"WARNING: Candidate set file not found at {candidate_path}. SID legality check will be SKIPPED.")

        # 2. 基于候选集的 creative_id 构建合法的 SID 集合
        if candidate_creative_ids and hasattr(dataset, 'sid'):
            legal_sid_set = set()
            # 遍历候选集中的 creative_id
            for creative_id_str in candidate_creative_ids:
                # 在整个数据集的 SID 字典中查找
                sids = dataset.sid.get(creative_id_str)
                if sids is not None and len(sids) >= 2:
                    legal_sid_set.add((sids[0], sids[1]))
            print(f"✓ Built legal SID set with {len(legal_sid_set)} unique pairs from the candidate set.")
        elif not hasattr(dataset, 'sid'):
            print("WARNING: dataset.sid not found. SID legality check will be SKIPPED.")
        else:
            print("WARNING: No creative_ids loaded from candidate set. SID legality check will be SKIPPED.")

        # 新增：构建整个训练数据集的SID集合
        if hasattr(dataset, 'sid'):
            full_sid_set = set()
            for creative_id, sids in dataset.sid.items():
                if sids is not None and len(sids) >= 2:
                    full_sid_set.add((sids[0], sids[1]))
            print(f"✓ Built full SID set with {len(full_sid_set)} unique pairs from the entire training dataset.")
        else:
            print("WARNING: dataset.sid not found. Full SID set will not be built.")
    # ==========================================================

    top_k = 10

    # 初始化用于存储最终结果的列表
    all_topk_indices: List[torch.Tensor] = []
    all_user_ids: List[Any] = []

    # ====== 新增：为 Beam Search 生成模式初始化列表 ======
    all_generated_sids: List[torch.Tensor] = []
    all_generated_scores: List[torch.Tensor] = []

    # ====== 新增：用于实时相似度计算的变量 ======
    similarity_calculated = False  # 标记是否已经计算过相似度
    similarity_test_users = 10000  # 测试用户数量
    # =================================================

    # ====== 初始化统计变量 ======
    # Reward模型统计（全局累积）
    reward_stats_global = {
        'total_items': 0,
        'ann_top10_items': 0,  # 最终top10中来自ANN top10的数量
        'new_items': 0,  # 最终top10中新出现的item数量
    }

    # SID模型统计（全局累积）
    sid_stats_global = {
        'total_items': 0,
        'ann_items': 0,  # 最终top10中来自ANN检索的数量
        'sid_items': 0,  # 最终top10中通过SID search出来的数量
    }

    # ====== 新增：Beam Search 合法率统计 (全局累积) ======
    sid_legality_stats_global = {
        'total_generated': 0,
        'legal_generated': 0,
        'full_generated': 0,  # 新增：在整个训练数据集中存在的SID数量
    }

    count = 0
    global_step = 0  # 全局步数计数器

    # ====== 初始化速度统计变量 ======
    run_start_time = time.time()
    accumulated_samples = 0  # 累积处理的用户数
    printed_user_count = 0  # 控制打印的用户数量

    # ====== (MODIFIED) SID Resort 模式处理 (初始化结果字典) ======
    sid_resort_results = {}
    if args.sid_resort and not score_agg_paths:  # 仅在纯 SID Resort 模式下
        print("Processing SID resort mode (Mode 3)...")

        # 为每个topk值创建结果存储
        for topk in args.sid_resort_topk:
            sid_resort_results[topk] = {
                'user_ids': [],
                'top10s': []
            }

    # +++ 新增: 确保 retrieval_ids_np 在 ANN 模式下可用 +++
    retrieval_ids_np = None
    if not args.beam_search_generate and retrieval_ids is not None:
        retrieval_ids_np = np.array(retrieval_ids)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Processing user batches for shard {shard_id}"):

            count += 1
            seq, token_type, seq_feat, user_ids_in_batch, next_action_type = batch

            # 将当前批次数据移到设备
            seq = seq.to(device)
            token_type = token_type.to(device)
            if next_action_type is not None:
                next_action_type = next_action_type.to(device)

            # ==================================================
            # ====== H-S) 新增：SID Resort 模式处理 ======
            # ==================================================
            if args.sid_resort:
                # 1. 获取用户embedding
                query_batch, mlp_logfeats, attention_mask = model.predict(seq, seq_feat, token_type, next_action_type)

                # 2. 计算相似度分数
                batch_scores = torch.matmul(query_batch, candidate_embs.T)

                # 3. 应用logp修正
                batch_scores_corrected = batch_scores

                # 4. 为每个topk值处理
                for topk in args.sid_resort_topk:
                    # 复制分数用于当前topk处理
                    batch_scores_topk = batch_scores_corrected.clone()

                    # 5. SID筛选：对每个用户，只保留前topk个SID生成的item
                    for i, user_id in enumerate(user_ids_in_batch):
                        if user_id in user_id_to_sid_cids:
                            user_sid_cids = user_id_to_sid_cids[user_id]
                            # 跳过0值，获取前topk个有效CID
                            valid_cids = [cid for cid in user_sid_cids if cid != 0]
                            user_sid_cids_topk = valid_cids[:topk]

                            # 找到这些CID在候选库中的索引
                            candidate_indices = []
                            for cid in user_sid_cids_topk:
                                if cid in creative_id_to_candidate_idx:
                                    candidate_indices.append(creative_id_to_candidate_idx[cid])

                            # 创建mask，只保留SID生成的items
                            if candidate_indices:
                                mask = torch.zeros(batch_scores_topk.shape[1], dtype=torch.bool, device=device)
                                # 修正: 确保索引在范围内
                                valid_candidate_indices = [idx for idx in candidate_indices if
                                                           idx < batch_scores_topk.shape[1]]
                                if valid_candidate_indices:
                                    mask[valid_candidate_indices] = True
                                    # 将不在SID候选中的分数设为负无穷
                                    batch_scores_topk[i, ~mask] = -torch.inf
                                else:
                                    batch_scores_topk[i, :] = -torch.inf
                            else:
                                # 如果没有找到任何候选，全部设为负无穷
                                batch_scores_topk[i, :] = -torch.inf
                        else:
                            # 如果用户不在SID数据中，全部设为负无穷
                            batch_scores_topk[i, :] = -torch.inf

                    # 6. 历史行为过滤
                    user_sequences = seq.cpu().numpy()
                    for i in range(len(user_ids_in_batch)):
                        history_indices = user_sequences[i]
                        # 从序列中直接获取历史 creative IDs
                        history_cids = {idx2creative_id.get(idx) for idx in history_indices if
                                        idx != 0 and idx in idx2creative_id}
                        # 找到这些 creative IDs 在候选集中的索引
                        candidate_indices_to_mask = [creative_id_to_candidate_idx.get(cid) for cid in history_cids if
                                                     cid in creative_id_to_candidate_idx]

                        # 将历史物品的分数设为负无穷
                        if candidate_indices_to_mask:
                            # 修正: 确保索引在范围内
                            valid_mask_indices = [idx for idx in candidate_indices_to_mask if
                                                  idx < batch_scores_topk.shape[1]]
                            if valid_mask_indices:
                                batch_scores_topk[i, valid_mask_indices] = -torch.inf

                    # 7. 检查过滤后剩余的有效item数量
                    for i in range(len(user_ids_in_batch)):
                        valid_items_count = torch.sum(batch_scores_topk[i] != -torch.inf).item()
                        if valid_items_count < 10:
                            user_id = user_ids_in_batch[i]
                            # print( # 暂时注释掉，避免过多打印
                            #     f"WARNING: User {user_id} has only {valid_items_count} valid items after SID+history filtering (topk={topk})"
                            # )

                    # 8. 获取top10结果 (SID + 历史过滤)
                    _, topk_indices = torch.topk(batch_scores_topk, k=10, dim=1)

                    # 9. 转换为CID并保存
                    topk_retrieval_ids = retrieval_ids_np[topk_indices.cpu().numpy()]
                    top10s = [
                        [retrieve_id2creative_id.get(int(rid), 0) for rid in user_top_k]
                        for user_top_k in topk_retrieval_ids
                    ]

                    # 保存到结果中
                    sid_resort_results[topk]['user_ids'].extend(list(user_ids_in_batch))
                    sid_resort_results[topk]['top10s'].extend(top10s)

                # 速度统计（每个step都记录）
                current_batch_size_dynamic = len(user_ids_in_batch)
                accumulated_samples += current_batch_size_dynamic

                # 计算当前速度（样本数/秒）
                elapsed_time = time.time() - run_start_time
                current_speed = accumulated_samples / elapsed_time if elapsed_time > 0 else 0
                runtime_minutes = elapsed_time / 60.0

                # 每个step都记录到TensorBoard（每个shard独立记录）
                writer.add_scalar(f'Speed/{shard_id}_current_samples_per_sec', current_speed, global_step)
                writer.add_scalar(f'Speed/{shard_id}_runtime_minutes', runtime_minutes, global_step)
                writer.add_scalar(f'Speed/{shard_id}_Total_Samples', accumulated_samples, global_step)

                # 每10个step打印一次速度信息
                if global_step % 10 == 0:
                    print(f"[Shard {shard_id}] Step {global_step}: "
                          f"Speed={current_speed:.2f} users/sec, "
                          f"Total={accumulated_samples} users, "
                          f"Runtime={runtime_minutes:.2f} min")

                # 增加全局步数
                global_step += 1

                # 跳过其他处理逻辑
                continue

            # ==================================================
            # ====== H-S) 新增：Beam Search 生成模式 (模式 3) ======
            # ==================================================
            if args.beam_search_generate:
                # 1. 获取 SID 特征 (需要重新运行 log2feats)
                batch_mask = (seq != 0).long()
                # 我们需要 log2feats 的第四个输出: sid_logfeats
                log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats, attention_mask_infer = model.log2feats(
                    seq, batch_mask, seq_feat,True
                )

                # 2. 计时开始
                beam_search_start_time = time.time()

                # 3. 调用 beamsearch_sid
                sid_sequences, sid_scores = model.beamsearch_sid(
                    sid_logfeats,
                    all_seq_logfeats,
                    attention_mask_infer,
                    top_k=args.beam_search_beam_size,  # (e.g., 10)
                    top_k_2=args.beam_search_top_k  # (e.g., 512 or 1024)
                )  # sid_sequences: [B, beam_search_top_k, 2]

                # 4. 计时结束并记录
                beam_search_end_time = time.time()
                batch_beam_time = beam_search_end_time - beam_search_start_time
                batch_size_current = seq.shape[0]
                # 计算当前批次的 beam search 吞吐量
                batch_speed_beam = batch_size_current / batch_beam_time if batch_beam_time > 0 else 0.0
                # ====== 新增：打印前几个用户的生成结果 ======
                # 初始化 batch 合法率以便在打印时可用
                batch_legality_rate = -1.0
                batch_full_rate = -1.0

                # ====== H-S.X) 新增：计算 SID 合法率 ======
                if legal_sid_set is not None or full_sid_set is not None:
                    # sid_sequences 仍在GPU上: [B, K, 2]
                    generated_sids_cpu = sid_sequences.cpu().numpy()

                    batch_total_generated = 0
                    batch_legal_generated = 0
                    batch_full_generated = 0  # 新增：在完整训练数据集中的SID数量

                    # 遍历 batch
                    for user_sids in generated_sids_cpu:  # user_sids is [K, 2]
                        # 遍历该用户生成的 K 个 SID
                        for sid_pair in user_sids:  # sid_pair is [sid1, sid2]
                            sid_tuple = (sid_pair[0], sid_pair[1])

                            batch_total_generated += 1

                            # 检查是否在候选集合法集合中
                            if legal_sid_set is not None and sid_tuple in legal_sid_set:
                                batch_legal_generated += 1

                            # 新增：检查是否在整个训练数据集的SID集合中
                            if full_sid_set is not None and sid_tuple in full_sid_set:
                                batch_full_generated += 1

                    # 更新全局统计
                    sid_legality_stats_global['total_generated'] += batch_total_generated
                    sid_legality_stats_global['legal_generated'] += batch_legal_generated
                    sid_legality_stats_global['full_generated'] += batch_full_generated

                    # --- 记录到 TensorBoard ---

                    # 计算并记录 batch 统计
                    if batch_total_generated > 0:
                        # 候选集合法率
                        batch_legality_rate = (batch_legal_generated / batch_total_generated) * 100
                        writer.add_scalar(f'SID_Legality_Batch/{shard_id}_Candidate_Legality_Rate_Percent',
                                          batch_legality_rate,
                                          global_step)

                        # 新增：完整数据集合法率
                        batch_full_rate = (batch_full_generated / batch_total_generated) * 100
                        writer.add_scalar(f'SID_Legality_Batch/{shard_id}_Full_Dataset_Legality_Rate_Percent',
                                          batch_full_rate,
                                          global_step)

                    writer.add_scalar(f'SID_Legality_Batch/{shard_id}_Legal_Count', batch_legal_generated, global_step)
                    writer.add_scalar(f'SID_Legality_Batch/{shard_id}_Full_Dataset_Count', batch_full_generated,
                                      global_step)

                    # 计算并记录 cumulative 统计
                    global_total = sid_legality_stats_global['total_generated']
                    if global_total > 0:
                        # 候选集合法率
                        global_legality_rate = (sid_legality_stats_global['legal_generated'] / global_total) * 100
                        writer.add_scalar(f'SID_Legality_Cumulative/{shard_id}_Candidate_Legality_Rate_Percent',
                                          global_legality_rate,
                                          global_step)

                        # 新增：完整数据集合法率
                        global_full_rate = (sid_legality_stats_global['full_generated'] / global_total) * 100
                        writer.add_scalar(f'SID_Legality_Cumulative/{shard_id}_Full_Dataset_Legality_Rate_Percent',
                                          global_full_rate,
                                          global_step)

                    writer.add_scalar(f'SID_Legality_Cumulative/{shard_id}_Total_Generated', global_total, global_step)
                # =========================================

                # 5. 记录到 TensorBoard (使用 global_step)
                writer.add_scalar(f'Speed/{shard_id}_beam_search_users_per_sec', batch_speed_beam, global_step)
                writer.add_scalar(f'Speed/{shard_id}_beam_search_batch_time_ms', batch_beam_time * 1000, global_step)

                # 6. 保存结果 (CPU)
                all_generated_sids.append(sid_sequences.cpu())
                all_generated_scores.append(sid_scores.cpu())

                # 7. 复制步骤 I (保存 user_ids)
                all_user_ids.extend(list(user_ids_in_batch))

                # ====== 新增：实时相似度计算 ======
                if not similarity_calculated and len(all_user_ids) >= similarity_test_users:
                    print(f"\n===== 达到{similarity_test_users}个用户，开始实时相似度计算 =====")

                    # 聚合当前已生成的数据
                    current_generated_sids = torch.cat(all_generated_sids, dim=0).numpy()  # [Current_Users, K, 2]
                    current_generated_scores = torch.cat(all_generated_scores, dim=0).numpy()  # [Current_Users, K]

                    # 计算相似度（只取前similarity_test_users个用户）
                    similarity_stats = calculate_sid_similarity_with_correct_top10s(
                        current_generated_sids[:similarity_test_users],
                        current_generated_scores[:similarity_test_users],
                        all_user_ids[:similarity_test_users],
                        dataset,
                        max_users=similarity_test_users
                    )

                    if similarity_stats:
                        # 将相似度统计记录到TensorBoard
                        writer.add_scalar(f'SID_Similarity_RealTime/Valid_Users', similarity_stats['valid_users'],
                                          global_step)
                        writer.add_scalar(f'SID_Similarity_RealTime/Avg_Overlap', similarity_stats['avg_overlap'],
                                          global_step)
                        writer.add_scalar(f'SID_Similarity_RealTime/Avg_Jaccard', similarity_stats['avg_jaccard'],
                                          global_step)
                        writer.add_scalar(f'SID_Similarity_RealTime/Max_Overlap', similarity_stats['max_overlap'],
                                          global_step)
                        writer.add_scalar(f'SID_Similarity_RealTime/Max_Jaccard', similarity_stats['max_jaccard'],
                                          global_step)
                        print("实时相似度统计已记录到TensorBoard")

                    similarity_calculated = True
                    print(f"===== 实时相似度计算完成，继续处理剩余用户 =====")
                # =========================================

                # 8. 复制步骤 J (全局速度统计)
                current_batch_size_dynamic = len(user_ids_in_batch)
                accumulated_samples += current_batch_size_dynamic
                elapsed_time = time.time() - run_start_time
                current_speed = accumulated_samples / elapsed_time if elapsed_time > 0 else 0
                runtime_minutes = elapsed_time / 60.0

                writer.add_scalar(f'Speed/{shard_id}_current_samples_per_sec', current_speed, global_step)
                writer.add_scalar(f'Speed/{shard_id}_runtime_minutes', runtime_minutes, global_step)
                writer.add_scalar(f'Speed/{shard_id}_Total_Samples', accumulated_samples, global_step)

                # 打印速度信息（包含 beam search 速度和合法率）
                if global_step % 10 == 0:
                    legality_info = f"Candidate Legality={batch_legality_rate:.2f}% (Batch)" if batch_legality_rate >= 0.0 else "Candidate Legality=N/A"
                    full_info = f"Full Dataset Legality={batch_full_rate:.2f}% (Batch)" if batch_full_rate >= 0.0 else "Full Dataset Legality=N/A"
                    print(f"[Shard {shard_id}] Step {global_step}: "
                          f"Speed={current_speed:.2f} users/sec (Total), "
                          f"BeamSpeed={batch_speed_beam:.2f} users/sec (Batch), "
                          f"{legality_info}, "
                          f"{full_info}, "
                          f"Runtime={runtime_minutes:.2f} min")

                # 9. 复制步骤 K (增加全局步数)
                global_step += 1

                # 10. 跳过本批次剩余的标准 ANN/重排 逻辑
                continue

            # ==================================================
            # ====== 结束 Beam Search 生成模式 ======
            # ==================================================

            # --- A/B) 模式相关的分数计算 ---

            # 用于 Reward/SID 模型的中间变量
            mlp_logfeats = None

            if score_agg_paths:
                # ====== 功能 2: 分数聚合 ======

                # 初始化总分
                current_batch_size_dynamic = seq.shape[0]
                num_candidates = candidate_embs_list[0].shape[0]
                batch_scores = torch.zeros(
                    current_batch_size_dynamic,
                    num_candidates,
                    dtype=torch.float32,
                    device=device
                )

                for i, (m, c_embs) in enumerate(zip(models_for_scoring, candidate_embs_list)):
                    # A) 获取 user embeddings
                    query_batch_m, mlp_logfeats_m, attention_mask_m = m.predict(seq, seq_feat, token_type,
                                                                                next_action_type)

                    # B) 计算相似度并累加
                    batch_scores += torch.matmul(query_batch_m, c_embs.T)

                    # 保存第一个模型的 mlp_logfeats 供后续 G/H 步骤使用
                    if i == 0:
                        mlp_logfeats = mlp_logfeats_m
                        # 确保 model 变量被设置（用于 G/H 步骤）
                        model = m

            else:
                # ====== 功能 0 (标准) / 功能 1 (平均) ======
                # 此时 `model` 和 `candidate_embs` 已经被正确设置

                # A) 获取当前批次的 user embeddings (logits)
                query_batch, mlp_logfeats, attention_mask = model.predict(seq, seq_feat, token_type, next_action_type)

                # B) 计算相似度分数（正常分数）
                batch_scores = torch.matmul(query_batch, candidate_embs.T)

            # --- C) 进行 logp 修正 (所有模式通用) ---
            batch_scores_corrected = batch_scores

            # D) 历史行为过滤 (所有模式通用)
            user_sequences = seq.cpu().numpy()
            for i in range(len(user_ids_in_batch)):
                history_indices = user_sequences[i]
                # 从序列中直接获取历史 creative IDs
                history_cids = {idx2creative_id.get(idx) for idx in history_indices if
                                idx != 0 and idx in idx2creative_id}
                # 找到这些 creative IDs 在候选集中的索引
                candidate_indices_to_mask = [creative_id_to_candidate_idx.get(cid) for cid in history_cids if
                                             cid in creative_id_to_candidate_idx]

                # 将历史物品的分数设为负无穷（对修正后的分数进行过滤）
                if candidate_indices_to_mask:
                    batch_scores_corrected[i, candidate_indices_to_mask] = -torch.inf

            # E) 获取 ANN Top-K 结果（使用修正后+过滤后的分数）
            _, ann_topk_indices_batch = torch.topk(batch_scores_corrected, k=top_k, dim=1)

            # F) 最终的topk结果初始化为ANN的结果
            final_topk_indices_batch = ann_topk_indices_batch.clone()

            # ====== G) Reward模型重排 ======
            # (在模式2下, model = models_for_scoring[0])
            if args.reward and hasattr(model, 'reward_model'):
                topk_reward = 32
                _, reward_topk_indices_batch = torch.topk(batch_scores_corrected, k=topk_reward, dim=1)
                batch_size_current = seq.shape[0]  # 使用 seq.shape[0] 保证动态 batch

                # G.1) 对每个用户，扩展其top-k候选的embeddings
                # ann_topk_indices_batch: [B, top_k]

                # **MODIFIED**: 候选库选择
                current_candidate_embs = candidate_embs if not score_agg_paths else candidate_embs_list[0]

                expanded_candidate_embs = current_candidate_embs[reward_topk_indices_batch]  # [B, top_k, hidden_dim]

                # G.2) 扩展query_batch以匹配候选数量
                # mlp_logfeats的最后一个时间步作为序列表征
                seq_embs_for_reward = mlp_logfeats[:, -1:, :].expand(-1, topk_reward, -1)  # [B, top_k, hidden_dim]

                # G.3) 获取ANN分数（归一化）
                ann_scores_for_reward = batch_scores_corrected.gather(1, reward_topk_indices_batch)  # [B, top_k]

                # G.4) 构建attention mask（全True，因为都是有效候选）
                # reward_model中的attention参数期望的是[B, seq_len, seq_len]的格式
                # 这里seq_len就是top_k
                reward_attn_mask = torch.ones(batch_size_current, topk_reward, topk_reward, dtype=torch.bool,
                                              device=device)

                # G.5) 调用reward模型
                reward_scores = model.reward_model_ctr(
                    seq_embs_for_reward,
                    expanded_candidate_embs,
                    reward_attn_mask,  # [B, top_k, top_k]
                    ann_scores=ann_scores_for_reward,
                    sid1_probs=None,
                    sid2_probs=None
                )  # [B, top_k, 1]

                reward_scores = reward_scores.squeeze(-1)  # [B, top_k]

                # G.6) 根据reward分数重排
                _, reward_rerank_indices = torch.topk(reward_scores, k=top_k, dim=1)
                reward_topk_indices_batch = reward_topk_indices_batch.gather(1, reward_rerank_indices)

                # G.7) 统计：计算ANN top10和Reward重排后的重叠度
                batch_reward_stats = {
                    'total_items': 0,
                    'ann_top10_items': 0,
                    'new_items': 0,
                }

                for i in range(batch_size_current):
                    ann_top10_set = set(ann_topk_indices_batch[i].cpu().numpy())
                    reward_top10_set = set(reward_topk_indices_batch[i].cpu().numpy())

                    # 计算交集（来自原ANN top10的item）
                    overlap_count = len(ann_top10_set & reward_top10_set)
                    # 新出现的item数量
                    new_count = len(reward_top10_set - ann_top10_set)

                    # 累积到batch统计
                    batch_reward_stats['total_items'] += top_k
                    batch_reward_stats['ann_top10_items'] += overlap_count
                    batch_reward_stats['new_items'] += new_count

                    # 累积到全局统计
                    reward_stats_global['total_items'] += top_k
                    reward_stats_global['ann_top10_items'] += overlap_count
                    reward_stats_global['new_items'] += new_count

                # G.8) 记录当前batch的统计到TensorBoard
                if batch_reward_stats['total_items'] > 0:
                    batch_ann_percentage = (batch_reward_stats['ann_top10_items'] / batch_reward_stats[
                        'total_items']) * 100
                    batch_new_percentage = (batch_reward_stats['new_items'] / batch_reward_stats['total_items']) * 100

                    writer.add_scalar(f'Reward_Infer_Batch_/{shard_id}_ANN_Top10_Percentage', batch_ann_percentage,
                                      global_step)
                    writer.add_scalar(f'Reward_Infer_Batch/{shard_id}_New_Items_Percentage', batch_new_percentage,
                                      global_step)
                    writer.add_scalar(f'Reward_Infer_Batch/{shard_id}_ANN_Top10_Count',
                                      batch_reward_stats['ann_top10_items'], global_step)
                    writer.add_scalar(f'Reward_Infer_Batch/{shard_id}_New_Items_Count', batch_reward_stats['new_items'],
                                      global_step)

                    # 同时记录累积统计
                    cumulative_ann_percentage = (reward_stats_global['ann_top10_items'] / reward_stats_global[
                        'total_items']) * 100
                    cumulative_new_percentage = (reward_stats_global['new_items'] / reward_stats_global[
                        'total_items']) * 100

                    writer.add_scalar(f'Reward_Infer_Cumulative/{shard_id}_ANN_Top10_Percentage',
                                      cumulative_ann_percentage, global_step)
                    writer.add_scalar(f'Reward_Infer_Cumulative/{shard_id}_New_Items_Percentage',
                                      cumulative_new_percentage, global_step)
                    writer.add_scalar(f'Reward_Infer_Cumulative/{shard_id}_Total_Items',
                                      reward_stats_global['total_items'], global_step)

                # 使用reward重排后的结果
                final_topk_indices_batch = reward_topk_indices_batch

            # I) 保存当前批次的结果 (转移到CPU以节省GPU内存)
            all_topk_indices.append(final_topk_indices_batch.cpu())
            all_user_ids.extend(list(user_ids_in_batch))

            # J) 速度统计（每个step都记录）
            current_batch_size_dynamic = len(user_ids_in_batch)
            accumulated_samples += current_batch_size_dynamic

            # 计算当前速度（样本数/秒）
            elapsed_time = time.time() - run_start_time
            current_speed = accumulated_samples / elapsed_time if elapsed_time > 0 else 0
            runtime_minutes = elapsed_time / 60.0

            # 每个step都记录到TensorBoard（每个shard独立记录）
            writer.add_scalar(f'Speed/{shard_id}_current_samples_per_sec', current_speed, global_step)
            writer.add_scalar(f'Speed/{shard_id}_runtime_minutes', runtime_minutes, global_step)
            writer.add_scalar(f'Speed/{shard_id}_Total_Samples', accumulated_samples, global_step)

            # 每10个step打印一次速度信息
            if global_step % 10 == 0:
                print(f"[Shard {shard_id}] Step {global_step}: "
                      f"Speed={current_speed:.2f} users/sec, "
                      f"Total={accumulated_samples} users, "
                      f"Runtime={runtime_minutes:.2f} min")

            # K) 增加全局步数
            global_step += 1

    # 9) 聚合所有批次的结果并保存
    if args.sid_resort:
        # ====== 新增：保存 SID Resort 结果 ======
        print(f"Saving SID resort results for shard {shard_id}...")

        # 保存结果到不同文件夹
        for topk in args.sid_resort_topk:
            result_dir = Path(args.user_cache_path) / 'train_infer' / run_name / f'sid_resort_{topk}'
            result_dir.mkdir(parents=True, exist_ok=True)

            part_path = result_dir / f"part_{shard_id:03d}.pkl"
            with open(part_path, 'wb') as f:
                pickle.dump({
                    'user_ids': sid_resort_results[topk]['user_ids'],
                    'top10s': sid_resort_results[topk]['top10s'],
                }, f)

            # 保存元数据
            meta = {
                'num_users': len(sid_resort_results[topk]['user_ids']),
                'shard_id': shard_id,
                'mode': 'sid_resort',
                'topk': topk
            }
            with open(result_dir / f"part_{shard_id:03d}.json", 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False)

            print(f"SID resort results for topk={topk} saved to {part_path}")

        print("SID resort mode processing completed.")

    elif args.beam_search_generate:
        # ====== 新增：保存 Beam Search 生成的结果 ======
        print(f"Aggregating beam search results for shard {shard_id}...")

        # 聚合所有批次
        generated_sids_tensor = torch.cat(all_generated_sids, dim=0).numpy()  # [Total_Users, K, 2]
        generated_scores_tensor = torch.cat(all_generated_scores, dim=0).numpy()  # [Total_Users, K]

        # ====== 注意：相似度计算已在生成过程中实时完成 ======

        out_dir = Path(args.user_cache_path) / 'train_infer' / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        part_path = out_dir / f"part_{shard_id:03d}.pkl"

        print(f"Saving {len(all_user_ids)} users' generated SIDs to {part_path}...")

        # 使用dataset.sid_reverse字典进行回查，将sid转换为cid
        generated_cids_tensor = []
        for user_sids in generated_sids_tensor:  # user_sids: [K, 2]
            user_cids = []
            for sid_pair in user_sids:  # sid_pair: [sid1, sid2]
                sid_key = "_".join([str(int(sid_pair[0])), str(int(sid_pair[1]))])
                cid = dataset.sid_reverse.get(sid_key, 0)  # 如果找不到对应的cid，赋值为0
                user_cids.append(cid)
            generated_cids_tensor.append(user_cids)

        generated_cids_tensor = np.array(generated_cids_tensor)  # [N, K]

        with open(part_path, 'wb') as f:
            pickle.dump({
                'user_ids': all_user_ids,
                'generated_cids': generated_cids_tensor,  # [N, K] - 修改为dump cid
            }, f)

        # 保存元数据
        meta = {
            'num_users': len(all_user_ids),
            'shard_id': shard_id,
            'mode': 'beam_search_generate',
            'top_k_generated': args.beam_search_top_k,
            'beam_size': args.beam_search_beam_size
        }
        with open(out_dir / f"part_{shard_id:03d}.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)

        print(f"Beam search results saved successfully for shard {shard_id}.")
    else:
        # ====== 原始的 Top-10 保存逻辑 ======
        topk_indices = torch.cat(all_topk_indices, dim=0).numpy()
        retrieval_ids_np = np.array(retrieval_ids)
        topk_retrieval_ids = retrieval_ids_np[topk_indices]

        top10s: List[List[Any]] = [
            [retrieve_id2creative_id.get(int(rid), 0) for rid in user_top_k]
            for user_top_k in topk_retrieval_ids
        ]

        out_dir = Path(args.user_cache_path) / 'train_infer' / run_name
        _save_part(out_dir, shard_id, all_user_ids, top10s)

    # ====== 速度统计总结 ======
    total_elapsed_time = time.time() - run_start_time
    final_speed = accumulated_samples / total_elapsed_time if total_elapsed_time > 0 else 0
    print(f"\n===== 推理速度统计 (Shard {shard_id}) =====")
    print(f"总处理用户数: {accumulated_samples}")
    print(f"总耗时: {total_elapsed_time:.2f}秒 ({total_elapsed_time / 60:.2f}分钟)")
    print(f"平均吞吐量: {final_speed:.2f} 用户/秒")
    print(f"======================================\n")

    # ====== 最终统计总结 ======
    # Reward模型最终统计
    if args.reward and reward_stats_global['total_items'] > 0:
        # 计算百分比
        ann_percentage = (reward_stats_global['ann_top10_items'] / reward_stats_global['total_items']) * 100
        new_percentage = (reward_stats_global['new_items'] / reward_stats_global['total_items']) * 100

        print(f"\n===== Reward模型重排最终统计 (Shard {shard_id}) =====")
        print(f"总item数: {reward_stats_global['total_items']}")
        print(f"来自ANN top10的item数: {reward_stats_global['ann_top10_items']} ({ann_percentage:.2f}%)")
        print(f"Reward重排后新出现的item数: {reward_stats_global['new_items']} ({new_percentage:.2f}%)")
        print(f"======================================================\n")


    # ====== SID Legality 最终统计 ======
    # (legal_sid_set 在循环外围定义)
    if args.beam_search_generate and sid_legality_stats_global['total_generated'] > 0:
        global_total = sid_legality_stats_global['total_generated']
        global_legal = sid_legality_stats_global['legal_generated']
        global_full = sid_legality_stats_global['full_generated']

        if global_total > 0:
            # 候选集合法率
            candidate_legality_rate = (global_legal / global_total) * 100
            # 完整数据集合法率
            full_legality_rate = (global_full / global_total) * 100

            print(f"\n===== SID Beam Search 合法率最终统计 (Shard {shard_id}) =====")
            print(f"总生成SID数: {global_total}")
            print(f"在候选集中的SID数: {global_legal} ({candidate_legality_rate:.2f}%)")
            print(f"在整个训练数据集中的SID数: {global_full} ({full_legality_rate:.2f}%)")
            print(f"======================================================\n")
    elif args.beam_search_generate:
        print(f"\n[Shard {shard_id}] 未进行SID合法率统计 (可能是 dataset.sid 未加载)。\n")

    # 关闭writer
    writer.close()
    print(f"✓ TensorBoard统计已写入 (Shard {shard_id}, 总步数: {global_step})")

    print(json.dumps({
        'msg': 'train_infer shard done',
        'name': run_name,
        'num_shards': num_shards,
        'shard_id': shard_id,
        'users': len(all_user_ids)
    }))


if __name__ == '__main__':
    main()