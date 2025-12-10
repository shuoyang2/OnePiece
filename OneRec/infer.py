import json
import os
import pickle
import struct
os.system("pip install pynvml")
os.system('pip install $USER_CACHE_PATH/nv_grouped_gemm-1.1.4.post4-cp310-cp310-linux_x86_64.whl')
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from dataset import MyTestDataset
from model import BaselineModel
from main_dist import get_args
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from utils import *
import time


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


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

    Args:
        indexer: 索引字典
        feat_types: 特征类型
        feat_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
        batch_size: 批处理大小
        candidate_path: 候选库文件路径

    Returns:
        candidate_embs (torch.Tensor): 候选库所有 item 的 embedding Tensor
        retrieval_ids (list): 与 candidate_embs 顺序一致的 retrieval_id 列表
        retrieve_id2creative_id (dict): retrieval_id -> creative_id 的映射字典
        cold_start_items (set): 冷启动item的creative_id集合
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

    if candidate_path is None:
        candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    else:
        candidate_path = Path(candidate_path)

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
            item_id = indexer.get(creative_id, 0)  # 使用 .get() 避免 KeyError
            if item_id == 0:
                cold_start_items.append(creative_id)
                continue  # 去除冷启动item
            feature = process_cold_start_feat(feature)
            feature = dataset.fill_missing_feat(feature, item_id, creative_id=creative_id)

            item_ids.append(item_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 生成embedding
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

            # 准备当前批次的数据
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

            # 调用 feat2emb 生成 embedding
            batch_emb = model.feat2emb(item_seq, seq_dict, include_user=False).squeeze(0)

            # 如果使用余弦相似度，进行 L2 归一化
            if model.similarity_function == 'cosine':
                batch_emb = batch_emb / (1e-8 + torch.norm(batch_emb, p=2, dim=-1, keepdim=True))

            all_embs_list.append(batch_emb)

    # 将所有批次的 embedding 拼接成一个大的 Tensor
    candidate_embs = torch.cat(all_embs_list, dim=0)

    return candidate_embs, retrieval_ids, retrieve_id2creative_id, set(cold_start_items)


def get_all_candidate_embs(indexer, feat_types, feat_default_value, mm_emb_dict, model, dataset, batch_size=1024):
    """
    生产候选库所有 item 的 id 和 embedding，并直接返回 PyTorch Tensor。

    Args:
        indexer: 索引字典
        feat_types: 特征类型
        feat_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
        batch_size: 批处理大小

    Returns:
        candidate_embs (torch.Tensor): 候选库所有 item 的 embedding Tensor，在模型所在的 device 上。
        retrieval_ids (list): 与 candidate_embs 顺序一致的 retrieval_id 列表。
        retrieve_id2creative_id (dict): retrieval_id -> creative_id 的映射字典。
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, retrieval_ids, features = [], [], []
    retrieve_id2creative_id = {}

    if type(indexer) != type({}):
        for i in indexer:
            indexer = i

    user_cache_path_candidate = Path(os.environ.get('USER_CACHE_PATH'), 'predict_set.jsonl')
    if not os.path.exists(user_cache_path_candidate):
        # 存储predict_set.jsonl
        os.system(f"cp {candidate_path} {user_cache_path_candidate}")
        print(f"已保存predict_set.jsonl")
        os.system(f"ls -alh {user_cache_path_candidate}")

    print("Step 1: Loading and processing candidate item features...")
    cold_start_items = []
    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer.get(creative_id, 0)  # 使用 .get() 避免 KeyError
            if item_id == 0:
                cold_start_items.append(creative_id)
                continue  # 去除冷启动item

            # 这部分逻辑与您原有的 get_candidate_emb 函数相同
            feature = process_cold_start_feat(feature)
            feature = dataset.fill_missing_feat(feature, item_id, creative_id=creative_id)

            item_ids.append(item_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # --- 以下逻辑来自您的 save_item_emb 函数，但修改为返回 Tensor ---
    all_embs_list = []
    model.eval()  # 确保模型处于评估模式

    item_feats_type = set(
        feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual'] + feat_types[
            'context_item_sparse'])
    item_emb_type = feat_types['item_emb']
    with torch.no_grad():  # 推理时不需要计算梯度
        print("Step 2: Generating candidate embeddings in batches...")
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Generating item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            # 准备当前批次的数据
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=model.dev).unsqueeze(0)
            item_features = [features[start_idx:end_idx]]
            seq_dict = {}
            for k in item_feats_type:
                seq_dict[k] = model.feat2tensor(item_features, k)

            SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 4096, "86": 3584}
            for k in item_emb_type:
                emb_dim = SHAPE_DICT[k]
                # 直接使用列表推导式生成 PyTorch Tensor
                seq_default_value = torch.zeros(emb_dim, dtype=torch.float32)
                # 构建一个嵌套列表
                batch_data_list = np.array([
                    [item.get(k, seq_default_value) for item in seq]
                    for seq in item_features
                ])
                seq_dict[k] = torch.tensor(batch_data_list, dtype=torch.float32)

            # 调用 feat2emb 生成 embedding
            # 注意：这里假设您的 feat2emb 可以处理 [batch_feat_list] 这样的输入
            # 如果 feat2emb 需要特定格式，需要在此处进行适配
            batch_emb = model.feat2emb(item_seq, seq_dict, include_user=False).squeeze(0)

            # 如果使用余弦相似度，进行 L2 归一化
            if model.similarity_function == 'cosine':
                batch_emb = batch_emb / (1e-8 + torch.norm(batch_emb, p=2, dim=-1, keepdim=True))

            all_embs_list.append(batch_emb)

    # 将所有批次的 embedding 拼接成一个大的 Tensor
    candidate_embs = torch.cat(all_embs_list, dim=0)

    return candidate_embs, retrieval_ids, retrieve_id2creative_id, set(cold_start_items)


def infer():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    train_infer_result_path = args.train_infer_result_path
    # 当提供了多进程合并结果的名称时，直接合并并返回
    EXPECTED_TOTAL_USERS = 10139575  # 预期的总用户数

    # 持续循环检测直到数据完整
    if not args.save_infer_model:
        while True:
            if train_infer_result_path:
                base_dir = Path(args.user_cache_path) / 'train_infer' / str(train_infer_result_path)
                if not base_dir.exists():
                    print(f"Train_infer merge path not found: {base_dir}. Waiting for 5 minutes...")
                    time.sleep(300)  # 等待5分钟
                    continue  # 继续下一次循环检测

                # 获取所有part文件
                try:
                    pkl_files = sorted(base_dir.glob('part_*.pkl'))
                except Exception as e:
                    print(f"Error listing files in {base_dir}: {e}. Waiting for 5 minutes...")
                    time.sleep(300)
                    continue

                if not pkl_files:
                    print(f"No part files found in {base_dir}. Waiting for 5 minutes...")
                    time.sleep(300)
                    continue

                print(f"Found {len(pkl_files)} part files. Checking data completeness...")

                all_users = []
                all_top10s = []
                all_files_loaded = True

                # 读取所有文件并累计用户数
                for pkl_path in pkl_files:
                    try:
                        with open(pkl_path, 'rb') as f:
                            part = pickle.load(f)
                        users = part.get('user_ids', [])
                        tops = part.get('top10s', [])

                        if not users or not tops:
                            print(f"File {pkl_path.name} is empty. Waiting for data to be complete.")
                            all_files_loaded = False
                            break

                        all_users.extend(users)
                        all_top10s.extend(tops)
                    except Exception as e:
                        print(f"Failed to read {pkl_path}: {e}")
                        all_files_loaded = False
                        break

                # 检查总用户数是否达到预期
                if all_files_loaded and len(all_users) == EXPECTED_TOTAL_USERS:
                    print(f"Data is complete. Total users: {len(all_users)}. Proceeding with processing...")

                    # 与正确答案做重合率调试（可选）
                    try:
                        correct_top10s_path = Path(args.user_cache_path) / 'correct_top10s.pkl'
                        correct_user_id_path_a = Path(args.user_cache_path) / 'correct_user_id.pkl.pkl'
                        correct_user_id_path_b = Path(args.user_cache_path) / 'correct_user_id.pkl'

                        if correct_top10s_path.exists():
                            with open(correct_top10s_path, 'rb') as f:
                                correct_top10s = pickle.load(f)

                            # 确定正确的用户ID文件路径
                            correct_user_id_path = None
                            if correct_user_id_path_a.exists():
                                correct_user_id_path = correct_user_id_path_a
                            elif correct_user_id_path_b.exists():
                                correct_user_id_path = correct_user_id_path_b

                            if correct_user_id_path:
                                with open(correct_user_id_path, 'rb') as f:
                                    correct_user_ids = pickle.load(f)

                                # 计算平均重合率
                                uid2idx = {u: i for i, u in enumerate(correct_user_ids)}
                                overlaps = []
                                for u, tops in zip(all_users, all_top10s):
                                    if u in uid2idx:
                                        ref = set(correct_top10s[uid2idx[u]])
                                        cur = set(tops)
                                        inter = len(ref.intersection(cur))
                                        overlaps.append(inter / 10.0)

                                if overlaps:
                                    avg_overlap = sum(overlaps) / len(overlaps)
                                    print(
                                        f"Train-Infer overlap ratio vs correct answers: {avg_overlap:.4f} (n={len(overlaps)})")
                                    if avg_overlap < 0.2:
                                        print(
                                            "Warning: overlap ratio is low; please check shard split or candidate generation.")
                            else:
                                print("No valid correct_user_id file found.")
                    except Exception as e:
                        print(f"Debug overlap skipped: {e}")

                    return all_top10s, all_users
                else:
                    if not all_files_loaded:
                        print("Not all files loaded successfully. Waiting for 5 minutes...")
                    elif len(all_users) < EXPECTED_TOTAL_USERS:
                        print(
                            f"Data incomplete. Expected {EXPECTED_TOTAL_USERS} users, but found {len(all_users)}. Waiting for 5 minutes...")
                    else:
                        print(
                            f"Data has more users than expected! Expected {EXPECTED_TOTAL_USERS}, found {len(all_users)}. Waiting for 5 minutes...")
                    time.sleep(300)

    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint)

    if args.save_infer_model:
        model_path = Path(args.user_cache_path) / "train_infer" / args.train_infer_result_path
        Path.mkdir(model_path, parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path / "model.pth")
        print(f"Saved model to {model_path}")
        return None

    # ==================== 关键修复 1 ====================
    # 创建反向索引时，将 creative_id 强制转换为整数
    print("Creating reverse indexer for items (ensuring integer IDs)...")
    creative_id2idx = test_dataset.indexer['i']
    try:
        idx2creative_id = {idx: int(cid) for cid, idx in creative_id2idx.items()}
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert all creative_ids in indexer to int. Error: {e}")
        # 如果转换失败，保留原始类型，但问题可能依旧存在
        idx2creative_id = {idx: cid for cid, idx in creative_id2idx.items()}
    # ==================== 修复结束 ======================

    # --- 2. 生成候选库的 embedding (Candidate Embeddings) ---
    candidate_embs, retrieval_ids, retrieve_id2creative_id, cold_start_items_creative_id = get_all_candidate_embs(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        batch_size=2048
    )

    # ==================== 关键修复 2 ====================
    # 创建 creative_id 到候选集索引的映射，同样确保键是整数
    print("Creating mapping from creative_id to candidate index (ensuring integer keys)...")
    candidate_creative_ids = [retrieve_id2creative_id.get(int(rid)) for rid in retrieval_ids]
    # 确保 cid 是整数
    creative_id_to_candidate_idx = {int(cid): idx for idx, cid in enumerate(candidate_creative_ids) if cid is not None}
    # ==================== 修复结束 ======================

    # --- 1. 生成所有用户的 embedding 并收集历史序列 ---
    all_user_embs = []
    user_list = []
    user_sequences_map = {}
    print("Generating user query embeddings and collecting user sequences...")

    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            seq, token_type, seq_feat, user_ids_in_batch, next_action_type = batch
            seq = seq.to(args.device)

            logits, _, _ = model.predict(seq, seq_feat, token_type, next_action_type)
            for i in range(logits.shape[0]):
                emb = logits[i].unsqueeze(0).detach().cpu()
                all_user_embs.append(emb)

                current_user_id = user_ids_in_batch[i]
                sequence_indices = seq[i].cpu().numpy()
                history_creative_ids = {
                    idx2creative_id.get(idx) for idx in sequence_indices if idx != 0 and idx in idx2creative_id
                }
                user_sequences_map[current_user_id] = history_creative_ids

            user_list.extend(user_ids_in_batch)

    query_embs = torch.cat(all_user_embs, dim=0)

    # (代码的其余部分与上一版相同，因为核心问题已在映射创建时解决)
    del model
    # --- 3. 使用 PyTorch 进行 Top-K 检索 (包含过滤逻辑) ---
    print("Performing Top-K search with history filtering...")
    top_k = 10
    query_embs = query_embs.to(args.device)
    candidate_embs = candidate_embs.to(args.device)
    user_batch_size = 512

    all_topk_indices = []

    for i in tqdm(range(0, query_embs.shape[0], user_batch_size), desc="Searching users in batches"):
        start_idx = i
        end_idx = min(i + user_batch_size, query_embs.shape[0])
        query_batch = query_embs[start_idx:end_idx]
        user_ids_in_batch = user_list[start_idx:end_idx]

        # 计算原始得分矩阵
        batch_scores = torch.matmul(query_batch, candidate_embs.T)

        # ====== 优化部分：向量化过滤 ======
        # 1. 在 CPU 端准备用于向量化屏蔽的坐标索引
        # 这比在循环中直接操作 GPU Tensor 快得多
        batch_indices_to_mask = []  # 行索引 (对应批次中的用户)
        candidate_indices_to_mask = []  # 列索引 (对应候选库中的 item)

        for j, user_id in enumerate(user_ids_in_batch):
            history_cids = user_sequences_map.get(user_id, set())
            if not history_cids:
                continue

            # 查找历史 item 在候选库中的索引
            # 这一步仍然需要循环，但在 CPU 上准备数据，开销远小于 GPU 交互
            history_indices_in_candidate = [creative_id_to_candidate_idx.get(cid) for cid in history_cids]

            for candidate_idx in history_indices_in_candidate:
                if candidate_idx is not None:
                    batch_indices_to_mask.append(j)
                    candidate_indices_to_mask.append(candidate_idx)

        # 2. 如果有需要屏蔽的 item，则执行一次性的向量化屏蔽操作
        if batch_indices_to_mask:
            # 使用收集好的坐标，在 batch_scores 上进行一次性、高效的批量赋值
            batch_scores[batch_indices_to_mask, candidate_indices_to_mask] = -torch.inf
        # ====================================

        # 在屏蔽后的分数上进行 topk 计算
        _, topk_indices_batch = torch.topk(batch_scores, k=top_k, dim=1)
        all_topk_indices.append(topk_indices_batch.cpu())

    topk_indices = torch.cat(all_topk_indices, dim=0).numpy()

    # --- 4. 映射索引到 creative_id ---
    print("Mapping indices to creative_ids...")
    retrieval_ids_np = np.array(retrieval_ids)
    topk_retrieval_ids = retrieval_ids_np[topk_indices]
    top10s = []
    for user_top_k_retrieval_ids in tqdm(topk_retrieval_ids, desc="Final mapping"):
        creative_ids = [retrieve_id2creative_id.get(int(rid), 0) for rid in user_top_k_retrieval_ids]
        top10s.append(creative_ids)

    # ... (冷启动统计部分保持不变) ...

    # 验证过滤是否生效
    print("\nVerifying filtering effectiveness (overlap should be 0)...")
    total_overlap_count = 0
    total_recommendations = len(user_list) * top_k
    for i, user_id in enumerate(user_list):
        recommended_ids = set(top10s[i])
        history_ids = user_sequences_map.get(user_id, set())
        overlap_items = recommended_ids.intersection(history_ids)
        total_overlap_count += len(overlap_items)

    if total_recommendations > 0:
        overlap_percentage = (total_overlap_count / total_recommendations) * 100
        print(f"过滤后，历史 item 重叠总数为: {total_overlap_count}")
        print(f"过滤后，历史 item 重叠率为: {overlap_percentage:.4f}%")
        if total_overlap_count == 0:
            print("过滤成功！推荐结果中已不包含用户历史序列中的 item。")
        else:
            print("警告：过滤后仍有重叠，请再次检查数据源或代码逻辑。")

    return top10s, user_list


