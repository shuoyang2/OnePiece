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

from main_dist import get_args
from dataset import MyDataset, MyTestDataset
from model import BaselineModel
from infer import get_all_candidate_embs_train


def _get_ckpt_path(name) -> str:
    base_path = Path(os.environ.get("USER_CACHE_PATH")) / "train_infer"
    ckpt_path = base_path / name

    if not ckpt_path.is_dir():
        ckpt_path_alt = Path(os.environ.get("USER_CACHE_PATH")) / name
        if ckpt_path_alt.is_dir():
            ckpt_path = ckpt_path_alt
        elif (base_path.parent / name).is_dir():
            ckpt_path = base_path.parent / name
        else:
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


class TrainInferDataset(MyDataset):
    """
    用于在训练集上进行推理的数据集类。
    继承自 MyDataset 以复用其数据加载逻辑 (seq.jsonl)，但会额外返回 user_id，
    同时保留训练模式下的 pos/label 数据以便计算 Metrics。
    """

    def __init__(self, data_dir, args):
        # [Critical] 临时将 args.mode 设为 'train'
        # 这确保 MyDataset.__init__ 加载 seq_offsets.pkl (训练集索引) 而非测试集索引
        original_mode = args.mode
        args.mode = 'train'
        super().__init__(data_dir, args)
        args.mode = original_mode

        # 强制当前实例的模式为 'train'
        # 这样 super().__getitem__ 才会生成 pos, pos_feat, next_token_type 等训练目标数据
        self.mode = 'train'

        # 尝试加载 user_action_type，辅助某些特定的推理逻辑
        uat_path = Path(self.data_dir, "user_action_type.json")
        try:
            import orjson
            if uat_path.exists():
                with open(uat_path, 'rb') as f:
                    self.user_action_type = orjson.loads(f.read())
            else:
                self.user_action_type = {}
        except Exception:
            self.user_action_type = {}

    def collate_fn(self, batch):
        """
        自定义 collate_fn，处理包含 user_id 的 12 元组数据。
        """
        # 解包 12 个元素
        seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, action_type, sid, pos_log_p, ranking_loss_mask, user_id = zip(
            *batch)


        # 转换为 Tensor
        sid = torch.from_numpy(np.array(sid))
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        action_type = torch.from_numpy(np.array(action_type))
        pos_log_p = torch.from_numpy(np.array(pos_log_p))
        ranking_loss_mask = torch.from_numpy(np.array(ranking_loss_mask))

        # 处理特征字典 (seq_feat 和 pos_feat)
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)

        seq_feat_dict = {}
        pos_feat_dict = {}

        # 处理常规特征 (Sparse/Array)
        for k in self.all_feats:
            seq_feat_dict[k] = self.feat2tensor(seq_feat, k)
            pos_feat_dict[k] = self.feat2tensor(pos_feat, k)

        # 处理 Embedding 特征
        for k in self.feature_types["item_emb"]:
            emb_dim = self.feature_default_value[k].shape[0]
            seq_default_value = torch.zeros(emb_dim, dtype=torch.float32)

            # 使用 numpy 列表推导优化构建 Tensor
            # seq_feat processing
            batch_data_list_seq = np.array([
                [item.get(k, seq_default_value) for item in s]
                for s in seq_feat
            ])
            seq_feat_dict[k] = torch.tensor(batch_data_list_seq, dtype=torch.float32)

            # pos_feat processing
            batch_data_list_pos = np.array([
                [item.get(k, seq_default_value) for item in p]
                for p in pos_feat
            ])
            pos_feat_dict[k] = torch.tensor(batch_data_list_pos, dtype=torch.float32)

        # 返回顺序需严格匹配 train_infer.py 中的解包顺序:
        # batch[:11] -> seq, token_type, seq_feat, user_ids, next_action_type, pos, pos_feat, sid, next_token_type, pos_log_p, ranking_loss_mask
        return seq, token_type, seq_feat_dict, user_id, next_action_type, pos, pos_feat_dict, sid, next_token_type, pos_log_p, ranking_loss_mask


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


def calculate_sid_validity(generated_sids, sid_reverse_map):
    """
    计算生成的 SID 组合是否在反向字典中（即是否是合法的 Item）
    Args:
        generated_sids: Tensor [Batch, Beam, 2]
        sid_reverse_map: Dict {"sid1_sid2": item_id}
    Returns:
        valid_count, total_count
    """
    valid_count = 0
    total_count = 0

    # 转移到 CPU 并转为 numpy 进行快速迭代
    sids_np = generated_sids.detach().cpu().numpy()

    # sids_np shape: [Batch, Beam, 2]
    # 展平前两维以便遍历
    B, Beam, _ = sids_np.shape
    sids_flat = sids_np.reshape(-1, 2)

    for i in range(sids_flat.shape[0]):
        s1, s2 = sids_flat[i]
        # 构建 key，确保格式与 dataset.py 中构建 sid_reverse 的方式一致
        # dataset.py: "_".join([str(i) for i in v])
        key = f"{int(s1)}_{int(s2)}"

        if key in sid_reverse_map:
            valid_count += 1
        total_count += 1

    return valid_count, total_count


def main():
    args = get_args()
    args.mode = "train"
    args.feature_dropout_rate = 0 # 保持默认，或者根据需要设置
    print(f"Feature Dropout Rate: {args.feature_dropout_rate}")

    num_shards = int(os.environ.get('NUM_SHARDS', '1'))
    shard_id = int(os.environ.get('SHARD_ID', '0'))
    run_name = args.train_infer_result_path

    device = args.device if torch.cuda.is_available() else 'cpu'
    args.device = device

    tb_path = args.tb_path
    tb_infer_path = Path(tb_path)
    tb_infer_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_infer_path))

    data_path = args.data_path
    dataset = TrainInferDataset(data_path, args)
    total_len = len(dataset)
    start, end = _shard_bounds(total_len, num_shards, shard_id)
    indices = list(range(start, end))
    subset = Subset(dataset, indices)

    current_batch_size = 2
    test_loader = DataLoader(
        subset, batch_size=current_batch_size, shuffle=False, num_workers=2,
        collate_fn=dataset.collate_fn, prefetch_factor=8
    )

    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    print("INFO: Entering Mode 0 - Standard Inference.")
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()
    ckpt_path = _get_ckpt_path(run_name)
    checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint)

    # 结果容器
    all_generated_sids: List[torch.Tensor] = []
    all_generated_scores: List[torch.Tensor] = []
    all_user_ids: List[Any] = []

    count = 0
    global_step = 0
    run_start_time = time.time()
    accumulated_samples = 0

    # 统计相关变量
    global_valid_sids = 0
    global_total_sids = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Processing user batches for shard {shard_id}"):
            count += 1
            # 鲁棒解包
            if len(batch) >= 11:
                seq, token_type, seq_feat, user_ids_in_batch, next_action_type, pos, pos_feat, sid, next_token_type, pos_log_p, ranking_loss_mask = batch[
                                                                                                                                                    :11]
            else:
                seq, token_type, seq_feat, user_ids_in_batch, next_action_type = batch

            seq = seq.to(device)
            token_type = token_type.to(device)
            if next_action_type is not None:
                next_action_type = next_action_type.to(device)
            if sid is not None:
                sid = sid.to(device)

            # Mask (seq != 0)
            mask = (seq != 0).long().to(device)

            # next_mask
            next_mask = (pos != 0).long().to(device) if pos is not None else mask

            if args.beam_search_generate:
                beam_search_start_time = time.time()

                sid_sequences, sid_scores = model.predict_beam_search(
                    user_item=seq,
                    pos_seqs=seq,
                    mask=mask,
                    next_mask=next_mask,
                    next_action_type=next_action_type,
                    seq_feature=seq_feat,
                    pos_feature=seq_feat,
                    sid=sid,
                    args=args,
                    dataset=dataset
                )

                beam_search_end_time = time.time()
                batch_beam_time = beam_search_end_time - beam_search_start_time
                batch_speed_beam = seq.shape[0] / batch_beam_time if batch_beam_time > 0 else 0.0

                # --- 新增：计算 SID 反解有效率 ---
                # 检查 dataset 是否加载了 sid_reverse
                batch_valid_rate = 0.0
                if hasattr(dataset, 'sid_reverse'):
                    valid_cnt, total_cnt = calculate_sid_validity(sid_sequences, dataset.sid_reverse)

                    global_valid_sids += valid_cnt
                    global_total_sids += total_cnt

                    batch_valid_rate = valid_cnt / total_cnt if total_cnt > 0 else 0.0
                    global_valid_rate = global_valid_sids / global_total_sids if global_total_sids > 0 else 0.0

                    # 写入 TensorBoard
                    writer.add_scalar(f'Validity/Batch_SID_Valid_Rate', batch_valid_rate, global_step)
                    writer.add_scalar(f'Validity/Global_SID_Valid_Rate', global_valid_rate, global_step)
                else:
                    if count == 1: print("Warning: dataset.sid_reverse not found, skipping validity check.")

                # 记录速度
                writer.add_scalar(f'Speed/{shard_id}_beam_search_users_per_sec', batch_speed_beam, global_step)

                all_generated_sids.append(sid_sequences.cpu())
                all_generated_scores.append(sid_scores.cpu())
                all_user_ids.extend(list(user_ids_in_batch))

                current_batch_size_dynamic = len(user_ids_in_batch)
                accumulated_samples += current_batch_size_dynamic
                elapsed_time = time.time() - run_start_time

                if global_step % 10 == 0:
                    valid_str = f", ValidRate={batch_valid_rate:.4f}" if hasattr(dataset, 'sid_reverse') else ""
                    print(
                        f"[Shard {shard_id}] Step {global_step}: BeamSpeed={batch_speed_beam:.2f} users/sec{valid_str}")

                global_step += 1
                continue

    if args.beam_search_generate:
        print(f"Aggregating beam search results for shard {shard_id}...")
        generated_sids_tensor = torch.cat(all_generated_sids, dim=0).numpy()

        # --- 新增：计算整体 Valid Rate ---
        # 假设我们只关心 Top-1 的合法率
        total_samples = len(generated_sids_tensor)
        valid_count_total = 0

        # 转换为 CID 的同时检查合法性
        generated_cids_tensor = []
        for user_sids in tqdm(generated_sids_tensor, desc="Calculating Validity"):
            user_cids = []
            # 检查第一个 beam 的合法性
            first_sid_key = f"{user_sids[0][0]}_{user_sids[0][1]}"
            if first_sid_key in dataset.sid_reverse:
                valid_count_total += 1

            for sid_pair in user_sids:
                sid_key = f"{sid_pair[0]}_{sid_pair[1]}"
                cid = dataset.sid_reverse.get(sid_key, 0)  # 0 表示无效/冷启动/未命中
                user_cids.append(cid)
            generated_cids_tensor.append(user_cids)

        final_valid_rate = valid_count_total / total_samples if total_samples > 0 else 0
        print(f"Shard {shard_id} Final Top-1 Validity Rate: {final_valid_rate:.4f}")
        writer.add_scalar(f'Validity/{shard_id}_final_validity', final_valid_rate, global_step)

        generated_cids_tensor = np.array(generated_cids_tensor)

        out_dir = Path(args.user_cache_path) / 'train_infer' / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        part_path = out_dir / f"part_{shard_id:03d}.pkl"

        with open(part_path, 'wb') as f:
            pickle.dump({
                'user_ids': all_user_ids,
                'generated_cids': generated_cids_tensor,
            }, f)

        print(f"Beam search results saved to {part_path}")

    writer.close()
    print("Done.")


if __name__ == '__main__':
    main()