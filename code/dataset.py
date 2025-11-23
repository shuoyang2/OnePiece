import json
import os
import random

import numpy

print("正在安装 orjson...")
os.system('pip3 install orjson')
import orjson
import pickle
import struct
from collections import defaultdict, Counter
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import os
from tqdm import tqdm
from datetime import datetime


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_file = None
        self.mode = args.mode
        if args.mode == "train":
            with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
                self.seq_offsets = pickle.load(f)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(args.user_cache_path, "item_feat_dict.json"), 'r'))  # reid: id feature
        self.user_action_type = orjson.loads(open(self.data_dir / "user_action_type.json", 'rb').read())

        # 加载物品曝光与行为分析数据
        exposure_data_path = Path(args.user_cache_path) / 'item_exposure' / 'item_exposure_data.pkl'
        print("Loading item action analysis data...")
        with open(exposure_data_path, 'rb') as f:
            # 将数据列表转换为 item_id -> item_data 的字典，以实现 O(1) 快速查找
            action_data_list = pickle.load(f)
            self.action_data = {item['item_id']: item for item in action_data_list}
        self.exposure_enabled = True
        print("Item action analysis data loaded successfully.")

        print("Loading item interaction counts (exposure + click + conversion) for InfoNCE correction...")
        # 提取每个item的总交互次数 (曝光 + 点击 + 转化)
        item_counts = {
            item['item_id']: (
                    item['total_counts'].get('exposures', 0) +
                    item['total_counts'].get('clicks', 0) +
                    item['total_counts'].get('conversions', 0)
            )
            for item in action_data_list
        }
        # 过滤掉总交互次数为0的item
        item_counts = {k: v for k, v in item_counts.items() if v > 0}

        total_interactions = sum(item_counts.values())

        if total_interactions > 0:
            # 计算 log(P_i), 其中 P_i = count_i / total_interactions
            self.item_log_p = {item_id: np.log(count / total_interactions) for item_id, count in item_counts.items()}
            # 为不在字典中的item设置一个默认的最小log p
            self.min_log_p = min(self.item_log_p.values()) if self.item_log_p else -20.0  # 如果为空则使用一个小的默认值
        else:
            self.item_log_p = {}
            self.min_log_p = -20.0  # 如果没有交互数据，则使用一个小的默认值

        self.exposure_correction_enabled = True
        print("Item interaction counts loaded successfully for InfoNCE correction.")

        # 加载时间戳桶化数据
        timestamp_buckets_path = Path(args.user_cache_path) / 'item_exposure' / 'timestamp_buckets.pkl'
        if timestamp_buckets_path.exists():
            print("Loading timestamp buckets data...")
            with open(timestamp_buckets_path, 'rb') as f:
                self.timestamp_buckets = pickle.load(f)
            self.timestamp_bucket_enabled = True
            print("Timestamp buckets data loaded successfully.")
        else:
            self.timestamp_buckets = None
            self.timestamp_bucket_enabled = False
            print("Warning: Timestamp buckets data not found, bucket features will be disabled.")

        # 加载每桶 item 出现次数数据
        item_counts_per_bucket_path = Path(args.user_cache_path) / 'item_exposure' / 'item_counts_per_bucket.pkl'
        if item_counts_per_bucket_path.exists():
            print("Loading item counts per bucket data...")
            with open(item_counts_per_bucket_path, 'rb') as f:
                # list[dict]，下标为bucket_id，值为 {item_id: count}
                self.item_counts_per_bucket = pickle.load(f)
            self.item_count_per_bucket_enabled = True
            print("Item counts per bucket data loaded successfully.")
        else:
            self.item_counts_per_bucket = None
            self.item_count_per_bucket_enabled = False
            print("Warning: Item counts per bucket not found; timestamp_bucket feature will fall back to 0.")

        # 新增: 预计算每个物品在桶内的热度百分位排名
        self.item_percentile_ranks_per_bucket = None
        if self.item_count_per_bucket_enabled:
            print("Pre-calculating item percentile ranks per bucket...")
            self.item_percentile_ranks_per_bucket = []
            for bucket_counts in tqdm(self.item_counts_per_bucket, desc="Calculating Percentiles"):
                if not bucket_counts:
                    self.item_percentile_ranks_per_bucket.append({})
                    continue

                all_counts = list(bucket_counts.values())
                total_items_in_bucket = len(all_counts)

                if total_items_in_bucket == 0:
                    self.item_percentile_ranks_per_bucket.append({})
                    continue

                # 统计每个计数值的频率
                count_freq = Counter(all_counts)
                unique_sorted_counts = sorted(count_freq.keys())

                # 计算每个计数值对应的排名（即有多少物品的计数值比它小）
                count_to_rank = {}
                items_with_smaller_count = 0
                for unique_count in unique_sorted_counts:
                    count_to_rank[unique_count] = items_with_smaller_count
                    items_with_smaller_count += count_freq[unique_count]

                # 计算每个物品的百分位排名，并转换为整数类别
                percentile_ranks_for_bucket = {}
                for item_id, count in bucket_counts.items():
                    rank = count_to_rank[count]
                    percentile = (rank / total_items_in_bucket) * 100.0
                    # 乘以10并取整，将 0.0-100.0 映射到 0-1000
                    percentile_rank_int = int(round(percentile, 1) * 10)
                    percentile_ranks_for_bucket[item_id] = percentile_rank_int

                self.item_percentile_ranks_per_bucket.append(percentile_ranks_for_bucket)
            print("Item percentile ranks pre-calculated successfully.")

        timestamp_buckets_span_path = Path(args.user_cache_path) / 'item_exposure' / 'timestamp_buckets_span.pkl'
        if timestamp_buckets_span_path.exists():
            print("Loading timestamp buckets data...")
            with open(timestamp_buckets_span_path, 'rb') as f:
                self.timestamp_buckets_span = pickle.load(f)
            self.timestamp_bucket_span_enabled = True
            print("Timestamp buckets data loaded successfully.")
        else:
            self.timestamp_buckets_span = None
            self.timestamp_bucket_span_enabled = False
            print("Warning: Timestamp buckets data not found, bucket features will be disabled.")
        # 加载多种桶配置 - 支持4096, 2048, 1024等
        # 获取用户配置的桶大小列表
        bucket_sizes = getattr(args, 'bucket_sizes', [])

        # 初始化存储多个桶配置的字典
        self.timestamp_buckets_multi = {}  # {bucket_size: buckets_data}

        for bucket_size in bucket_sizes:
            # 根据桶数量加载对应文件
            bucket_path = Path(args.user_cache_path) / 'item_exposure' / f'timestamp_buckets_{bucket_size}.pkl'

            if bucket_path.exists():
                print(f"Loading timestamp buckets data (bucket_size={bucket_size})...")
                with open(bucket_path, 'rb') as f:
                    self.timestamp_buckets_multi[bucket_size] = pickle.load(f)
                print(f"Timestamp buckets {bucket_size} loaded successfully.")
            else:
                print(f"Warning: Timestamp buckets {bucket_size} not found at {bucket_path}")

        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids, args.debug)
        self.SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 4096, "86": 3584}
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        # 自定义特征处理
        self.custom_feat_statistics = {
            "time_diff_day": 32,
            "time_diff_hour": 24,
            "time_diff_minute": 60,
            "next_action_type": 3,
            "action_type": 3,
            "date_year": 100,
            "date_month": 12,
            "date_day": 31,
            "exposure_start_year": 4,  # 0: unknown, 1: 2024, 2: 2025
            "exposure_start_month": 14,  # 0: unknown, 1-12: months
            "exposure_start_day": 33,  # 0: unknown, 1-31: days
            "exposure_end_year": 4,
            "exposure_end_month": 14,
            "exposure_end_day": 33,
            "hot_bucket_1000": 1001,  # 修改: 作为稀疏特征，词表大小为1001 (0-1000)
            "timestamp_bucket_id": 8193,  # 时间戳桶ID，8192个桶
            "timestamp_bucket_span": 8193
        }

        # 动态添加多种桶配置的特征
        for bucket_size in bucket_sizes:
            # 为每个桶大小添加bucket_id特征 (桶ID从0开始，+1是为了留出0作为默认值)
            self.custom_feat_statistics[f"timestamp_bucket_{bucket_size}"] = bucket_size + 1

        if args.item_sparse is not None and "sid" in args.item_sparse:
            for layer in range(args.sid_codebook_layer):
                self.custom_feat_statistics[f"sid_{layer}"] = args.sid_codebook_size
                args.item_sparse.append(f"sid_{layer}")

        self.time_diff_enabled = False
        self.action_enabled = False
        self.hot_bucket_1000_enabled = False
        self.timestamp_bucket_id_enabled = False
        self.timestamp_bucket_span_enabled = False
        # 为每个桶大小添加启用标志
        self.timestamp_bucket_enabled_dict = {bucket_size: False for bucket_size in bucket_sizes}

        if args.sid:
            if args.debug:
                sid_path = args.user_cache_path
                path = f"{sid_path}/sid_81.pkl"
            else:
                sid_path = args.user_cache_path + f"/sid"
                path = f"{sid_path}/sid_{'_'.join(args.mm_sid)}.pkl"
            with open(path, 'rb') as ff:
                self.sid = pickle.load(ff)
            self.sid_enabled = True
            # 反查表
            self.sid_reverse = {}
            for k, v in self.sid.items():
                self.sid_reverse["_".join([str(i) for i in v])] = k

        else:
            self.sid = None
            self.sid_enabled = False

        self.feature_dropout_list = args.feature_dropout_list
        self.feature_dropout_rate = args.feature_dropout_rate

        if args.item_sparse and "sid" in args.item_sparse:
            args.item_sparse.remove("sid")

        self.feature_default_value, self.feature_types, self.feat_statistics, self.all_feats = self._init_feat_info(
            args)

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        if self.data_file is None:
            self._load_data_and_offsets()
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = orjson.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _get_timestamp_bucket(self, timestamp):
        """
        根据时间戳获取对应的桶ID

        Args:
            timestamp: 时间戳

        Returns:
            bucket_id: 桶ID，如果未找到则返回0
        """
        if not self.timestamp_bucket_enabled or not self.timestamp_buckets:
            return 0

        # 使用二分查找找到对应的桶
        left, right = 0, len(self.timestamp_buckets) - 1

        while left <= right:
            mid = (left + right) // 2
            bucket = self.timestamp_buckets[mid]

            if bucket['start_timestamp'] is None or bucket['end_timestamp'] is None:
                # 空桶，跳过
                if mid < len(self.timestamp_buckets) - 1:
                    left = mid + 1
                else:
                    right = mid - 1
                continue

            if bucket['start_timestamp'] <= timestamp <= bucket['end_timestamp']:
                return bucket['bucket_id'] + 1
            elif timestamp < bucket['start_timestamp']:
                right = mid - 1
            else:
                left = mid + 1

        # 如果没找到，返回0（默认桶）
        return 0

    def _get_item_percentile_rank_in_bucket(self, item_id, timestamp):
        """
        修改: 返回该时间戳所在桶内该 item 的热度百分位排名类别 (0-1000)。
        """
        if not self.item_count_per_bucket_enabled or not self.item_percentile_ranks_per_bucket:
            return 0
        if timestamp is None or item_id is None or item_id == 0:
            return 0

        b_idx = self._get_timestamp_bucket(timestamp)
        if b_idx == 0:
            return 0

        bucket_index = b_idx - 1

        if bucket_index < 0 or bucket_index >= len(self.item_percentile_ranks_per_bucket):
            return 0

        # 直接查询预计算好的百分位排名整数类别
        percentile_rank_map = self.item_percentile_ranks_per_bucket[bucket_index]
        return percentile_rank_map.get(item_id, 0)

    def _get_timestamp_bucket_by_size(self, timestamp, bucket_size):
        """
        根据时间戳和桶大小获取对应的桶ID

        Args:
            timestamp: 时间戳
            bucket_size: 桶的数量（4096, 2048, 1024等）

        Returns:
            bucket_id: 桶ID（从1开始，0表示未找到或无效）
        """
        if bucket_size not in self.timestamp_buckets_multi:
            return 0

        buckets = self.timestamp_buckets_multi[bucket_size]
        if not buckets:
            return 0

        # 使用二分查找
        left, right = 0, len(buckets) - 1

        while left <= right:
            mid = (left + right) // 2
            bucket = buckets[mid]

            if bucket['start_timestamp'] is None or bucket['end_timestamp'] is None:
                if mid < len(buckets) - 1:
                    left = mid + 1
                else:
                    right = mid - 1
                continue

            if bucket['start_timestamp'] <= timestamp <= bucket['end_timestamp']:
                # 返回bucket_id + 1，因为0用作默认值
                return bucket['bucket_id'] + 1
            elif timestamp < bucket['start_timestamp']:
                right = mid - 1
            else:
                left = mid + 1

        return 0

    def _get_timestamp_bucket_span(self, timestamp):
        """
        根据时间戳获取8192桶(span)的桶ID

        Args:
            timestamp: 时间戳

        Returns:
            bucket_id: 桶ID，如果未找到则返回0
        """
        if not self.timestamp_bucket_span_enabled or not self.timestamp_buckets_span:
            return 0

        left, right = 0, len(self.timestamp_buckets_span) - 1
        while left <= right:
            mid = (left + right) // 2
            bucket = self.timestamp_buckets_span[mid]

            if bucket['start_timestamp'] is None or bucket['end_timestamp'] is None:
                if mid < len(self.timestamp_buckets_span) - 1:
                    left = mid + 1
                else:
                    right = mid - 1
                continue

            if bucket['start_timestamp'] <= timestamp <= bucket['end_timestamp']:
                return bucket['bucket_id'] + 1
            elif timestamp < bucket['start_timestamp']:
                right = mid - 1
            else:
                left = mid + 1

        return 0
    def _get_timediff_bucket(self, timediff):
        """
        根据时间差（秒）获取对应的桶ID

        Args:
            timediff: 时间差（秒）

        Returns:
            bucket_id: 桶ID（从1开始，0表示未找到或无效）
        """
        if not self.timediff_bucket_enabled or not self.timediff_buckets:
            return 0

        if timediff is None or timediff < 0:
            return 0

        # 使用二分查找找到对应的桶
        left, right = 0, len(self.timediff_buckets) - 1

        while left <= right:
            mid = (left + right) // 2
            bucket = self.timediff_buckets[mid]

            if bucket['start_timediff'] is None or bucket['end_timediff'] is None:
                # 空桶，跳过
                if mid < len(self.timediff_buckets) - 1:
                    left = mid + 1
                else:
                    right = mid - 1
                continue

            if bucket['start_timediff'] <= timediff <= bucket['end_timediff']:
                # 返回bucket_id + 1，因为0用作默认值
                return bucket['bucket_id'] + 1
            elif timediff < bucket['start_timediff']:
                right = mid - 1
            else:
                left = mid + 1

        # 如果没找到，返回0（默认桶）
        return 0

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
            pos_log_p: 正样本的对数概率
            neg_log_p: 负样本的对数概率
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u:
                user_id = self.indexer_u_rev[u]
                if user_feat:
                    ext_user_sequence.insert(0, (0, user_feat, 2, action_type, timestamp))
                elif not item_feat:
                    ext_user_sequence.insert(0, (0, {}, 2, action_type, timestamp))

            if i and item_feat:
                dict_item_feat = self.item_feat_dict[str(i)]
                missing_keys = set(dict_item_feat.keys()) - set(item_feat.keys())
                for key in missing_keys:
                    item_feat[key] = dict_item_feat[key]

                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32) if self.mode == "train" else None
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32) if self.mode == "train" else None

        # ========== logq: 追加 pos/neg 对数概率 ==========
        pos_log_p = np.zeros([self.maxlen + 1], dtype=np.float32) if self.mode == "train" else None

        # ========== 5月31号之后的ranking_loss_mask ==========
        ranking_loss_mask = np.zeros([self.maxlen + 1], dtype=np.float32)

        sid = np.zeros([self.maxlen + 1, 2], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object) if self.mode == "train" else None
        nxt = ext_user_sequence[-1] if self.mode == "train" else None
        idx = self.maxlen

        ext_user_sequence = ext_user_sequence[:-1] if self.mode == 'train' else ext_user_sequence
        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence):
            i, feat, type_, act_type, timestamp = record_tuple

            next_i, next_feat, next_type, next_act_type, next_timestamp = None, None, None, None, None
            if nxt is not None:
                next_i, next_feat, next_type, next_act_type, next_timestamp = nxt

            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            else:
                if self.mode != "train" and idx == self.maxlen:  # 如果是最后一个位置，则需要查询user_action_type
                    next_action_type[idx] = self.user_action_type[user_id]
                else:
                    next_action_type[idx] = 0

            if type_ == 2:
                action_type[idx] = 0
            else:
                action_type[idx] = act_type if act_type is not None else 0

            # 判断是否在5月31号之后 (2025年5月31日的时间戳)
            may_31_2024 = 1748620800  # 2025年5月31日 00:00:00 UTC时间戳
            if next_timestamp is not None and next_timestamp > may_31_2024:
                ranking_loss_mask[idx] = 0
            else:
                ranking_loss_mask[idx] = 1

            context_feat = {}
            if self.action_enabled:
                context_feat["action_type"] = action_type[idx]
                context_feat['next_action_type'] = next_action_type[idx]
            if self.hot_bucket_1000_enabled and timestamp is not None:
                # 修改: 调用新函数获取百分位排名整数类别
                if type_ == 2:
                    context_feat["hot_bucket_1000"] = 0
                else:
                    context_feat["hot_bucket_1000"] = self._get_item_percentile_rank_in_bucket(i, timestamp)
            if self.timestamp_bucket_id_enabled and timestamp is not None:
                if type_ == 2:
                    context_feat["timestamp_bucket_id"] = 0
                else:
                    # 使用时间戳桶ID作为特征值
                    context_feat["timestamp_bucket_id"] = self._get_timestamp_bucket(timestamp)

            # 添加多种桶配置的特征
            if timestamp is not None:
                if self.timestamp_bucket_span_enabled and type_ == 1:
                    context_feat["timestamp_bucket_span"] = self._get_timestamp_bucket_span(timestamp)
                for bucket_size in self.timestamp_buckets_multi.keys():
                    # 只有当该桶特征被启用时才填充
                    if self.timestamp_bucket_enabled_dict.get(bucket_size, False):
                        feat_name = f"timestamp_bucket_{bucket_size}"
                        if type_ == 2:
                            context_feat[feat_name] = 0
                        else:
                            context_feat[feat_name] = self._get_timestamp_bucket_by_size(timestamp, bucket_size)

            # 添加时间差桶特征

            feat = self.fill_missing_feat(feat, i, context_feat)
            if self.mode == "train":
                next_feat = self.fill_missing_feat(next_feat, next_i)

            origin_idx = idx  # 4 = 0
            if type_ == 2:
                idx = 0

            seq[idx] = i
            token_type[idx] = type_

            if self.mode == "train":
                next_token_type[idx] = next_type

            seq_feat[idx] = feat
            if next_type is not None and next_i is not None and next_type == 1 and next_i != 0:
                if self.mode == "train":
                    pos[idx] = next_i
                    pos_feat[idx] = next_feat
                    if self.exposure_correction_enabled:
                        pos_log_p[idx] = self.item_log_p.get(next_i, self.min_log_p)
                if self.time_diff_enabled and origin_idx < self.maxlen:
                    # 加入时间差特征
                    if type_ == 2:
                        # 若当前为User, 则说明是第一个Item, 时间差为0
                        time_diff = 0
                    else:
                        # 若当前也是Item, 则说明是中间Item, 用next_timestamp - timestamp得到时间差
                        time_diff = next_timestamp - timestamp

                    day, hour, minute = second2timediff(time_diff)
                    seq_feat[origin_idx + 1]["time_diff_day"] = day
                    seq_feat[origin_idx + 1]["time_diff_hour"] = hour
                    seq_feat[origin_idx + 1]["time_diff_minute"] = minute


                if self.sid:  # 传的是下一个item 的sid
                    c_i = self.indexer_i_rev[next_i]
                    if c_i in self.sid:
                        sid[idx] = self.sid[c_i]

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        if self.mode == "train":
            pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)

            return seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, action_type, sid, pos_log_p, ranking_loss_mask
        else:
            return seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, action_type, sid, pos_log_p, ranking_loss_mask, user_id

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self, args):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = args.base_user_sparse + (args.user_sparse if args.user_sparse else [])
        feat_types['item_sparse'] = args.base_item_sparse + (args.item_sparse if args.item_sparse else [])
        feat_types['context_item_sparse'] = args.context_item_sparse if args.context_item_sparse is not None else []
        feat_types['item_array'] = args.item_array if args.item_array else []
        feat_types['user_array'] = args.base_user_array + (args.user_array if args.user_array else [])
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = args.user_continual if args.user_continual else []
        feat_types['item_continual'] = args.item_continual if args.item_continual else []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id in self.custom_feat_statistics:
                feat_statistics[feat_id] = self.custom_feat_statistics[feat_id]
            else:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0.0

        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0
        # 加入上下文特征
        for feat_id in feat_types['context_item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = self.custom_feat_statistics[feat_id]
            if 'time_diff' in feat_id:
                self.time_diff_enabled = True
            if "action_type" in feat_id:
                self.action_enabled = True
            if "hot_bucket_1000" == feat_id:
                # 如果用户仍将其放在稀疏特征中，也启用它
                self.hot_bucket_1000_enabled = True
            if "timestamp_bucket_id" == feat_id:
                self.timestamp_bucket_id_enabled = True
            if "timestamp_bucket_span" == feat_id:
                self.timestamp_bucket_span_enabled = True
            # 检查多桶特征是否启用
            if feat_id.startswith("timestamp_bucket_"):
                # 提取桶大小，例如从 "timestamp_bucket_4096" 提取 4096
                parts = feat_id.split("_")
                if len(parts) == 3 and parts[2].isdigit():
                    bucket_size = int(parts[2])
                    if bucket_size in self.timestamp_bucket_enabled_dict:
                        self.timestamp_bucket_enabled_dict[bucket_size] = True

        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float16
            )

        all_feats = [item for sublist in feat_types.values() for item in sublist if item is not None]
        # 去除item_emb
        for k in feat_types['item_emb']:
            if k in all_feats:
                all_feats.remove(k)

        # 输出特征名称，便于核对
        try:
            print("================ Feature Summary ================")
            for feat_group_name, feat_group_list in feat_types.items():
                print(f"{feat_group_name} (count={len(feat_group_list)}): {feat_group_list}")
            print(f"all_non_emb_feats (count={len(all_feats)}): {all_feats}")
            print("=================================================")
        except Exception:
            pass

        return feat_default_value, feat_types, feat_statistics, all_feats

    def fill_missing_feat(self, feat, item_id, context_feats={}, creative_id=""):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat is None:
            feat = {}
        filled_feat = feat.copy()

        default_time_feats = {
            'exposure_start_year': 0, 'exposure_start_month': 0, 'exposure_start_day': 0,
            'exposure_end_year': 0, 'exposure_end_month': 0, 'exposure_end_day': 0,
        }

        if self.exposure_enabled and item_id in self.action_data:
            item_stats = self.action_data[item_id]

            # 1. 处理时间戳特征
            if item_stats.get('exposure_start_ts') is not None:
                start_dt = datetime.fromtimestamp(item_stats['exposure_start_ts'])
                filled_feat['exposure_start_year'] = start_dt.year - 2023 if start_dt.year in [2024, 2025] else 0
                filled_feat['exposure_start_month'] = start_dt.month
                filled_feat['exposure_start_day'] = start_dt.day

            if item_stats.get('exposure_end_ts') is not None:
                end_dt = datetime.fromtimestamp(item_stats['exposure_end_ts'])
                filled_feat['exposure_end_year'] = end_dt.year - 2023 if end_dt.year in [2024, 2025] else 0
                filled_feat['exposure_end_month'] = end_dt.month
                filled_feat['exposure_end_day'] = end_dt.day
        else:
            filled_feat.update(default_time_feats)

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)

        missing_fields = set(all_feat_ids) - set(filled_feat.keys())
        for feat_id in missing_fields:
            if feat_id not in self.custom_feat_statistics:
                filled_feat[feat_id] = self.feature_default_value[feat_id]
            else:
                if "time_diff" in feat_id:
                    filled_feat[feat_id] = self.feature_default_value[feat_id]
                if "action_type" in feat_id or "timestamp_bucket" in feat_id or "hot_bucket_1000" in feat_id:
                    filled_feat[feat_id] = context_feats.get(feat_id, self.feature_default_value[feat_id])
                if "sid" in feat_id:
                    # item sparse
                    sid = self.sid[self.indexer_i_rev[item_id]]
                    filled_feat[feat_id] = sid[int(feat_id.split("_")[-1])]

        for feat_id in self.feature_types["item_emb"]:
            if item_id != 0 and self.indexer_i_rev.get(item_id) in self.mm_emb_dict.get(feat_id, {}):
                item_key = self.indexer_i_rev[item_id]
                if isinstance(self.mm_emb_dict[feat_id][item_key], np.ndarray):
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][item_key]
            elif creative_id and creative_id in self.mm_emb_dict.get(feat_id, {}):
                filled_feat[feat_id] = self.mm_emb_dict[feat_id][creative_id]
            else:
                filled_feat[feat_id] = np.zeros(self.SHAPE_DICT[feat_id], dtype=np.float16)

        if self.feature_dropout_list is not None:
            for feat_id in filled_feat.keys():
                if feat_id in self.feature_dropout_list:
                    random_num = random.random()
                    if random_num <= self.feature_dropout_rate:
                        empty_feature = 0
                        filled_feat[feat_id] = empty_feature

        return filled_feat

    def collate_fn(self, batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
            pos_log_p: 正样本的对数概率 tensor
            neg_log_p: 负样本的对数概率 tensor
        """
        seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, action_type, sid, pos_log_p, ranking_loss_mask = zip(
            *batch)

        sid = torch.from_numpy(np.array(sid))
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        action_type = torch.from_numpy(np.array(action_type))

        pos_log_p = torch.from_numpy(np.array(pos_log_p))
        ranking_loss_mask = torch.from_numpy(np.array(ranking_loss_mask))

        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)

        seq_feat_dict = {}
        pos_feat_dict = {}

        for k in self.all_feats:
            seq_feat_dict[k] = self.feat2tensor(seq_feat, k)
            pos_feat_dict[k] = self.feat2tensor(pos_feat, k)

        for k in self.feature_types["item_emb"]:
            emb_dim = self.feature_default_value[k].shape[0]
            # 直接使用列表推导式生成 PyTorch Tensor
            seq_default_value = torch.zeros(emb_dim, dtype=torch.float32)
            # 构建一个嵌套列表
            batch_data_list = np.array([
                [item.get(k, seq_default_value) for item in seq]
                for seq in seq_feat
            ])
            seq_feat_dict[k] = torch.tensor(batch_data_list, dtype=torch.float32)

            seq_default_value = torch.zeros(emb_dim, dtype=torch.float32)
            # 构建一个嵌套列表
            batch_data_list = np.array([
                [item.get(k, seq_default_value) for item in seq]
                for seq in pos_feat
            ])
            pos_feat_dict[k] = torch.tensor(batch_data_list, dtype=torch.float32)

        return seq, pos, token_type, next_token_type, next_action_type, seq_feat_dict, pos_feat_dict, action_type, sid, pos_log_p, ranking_loss_mask

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.feature_types['item_array'] or k in self.feature_types['user_array']:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
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
            res = torch.from_numpy(batch_data)
            return res
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item.get(k, self.feature_default_value.get(k, 0)) for item in seq_feature[i]]
                batch_data[i, :len(seq_data)] = seq_data

            res = torch.from_numpy(batch_data)
            return res


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
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

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    def collate_fn(self, batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, _, token_type, _, next_action_type, seq_feat, _, _, _, _, _, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat_dict = {}

        for k in self.all_feats:
            seq_feat_dict[k] = self.feat2tensor(seq_feat, k)

        for k in self.feature_types["item_emb"]:
            emb_dim = self.feature_default_value[k].shape[0]
            # 直接使用列表推导式生成 PyTorch Tensor
            seq_default_value = torch.zeros(emb_dim, dtype=torch.float32)
            # 构建一个嵌套列表
            batch_data_list = np.array([
                [item.get(k, seq_default_value) for item in seq]
                for seq in seq_feat
            ])
            seq_feat_dict[k] = torch.tensor(batch_data_list, dtype=torch.float32)

        return seq, token_type, seq_feat_dict, user_id, next_action_type


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def process_json_file(json_file):
    emb_dict = {}
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data_dict_origin = json.loads(line.strip())
                insert_emb = data_dict_origin['emb']
                if isinstance(insert_emb, list):
                    insert_emb = np.array(insert_emb, dtype=np.float16)
                emb_dict[data_dict_origin['anonymous_cid']] = insert_emb
            except Exception as e:
                # print(data_dict_origin)
                pass
    return emb_dict


def load_mm_emb(mm_path, feat_ids, debug):
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 4096, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81' or (feat_id == '81' and debug):
            base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
            json_files = list(base_path.glob('*.json'))
            with Pool() as pool:
                results = pool.map(process_json_file, json_files)
            for result in results:
                emb_dict.update(result)
        else:
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


def second2timediff(second):
    day = second // 86400 + 1
    second = second % 86400
    hour = second // 3600 + 1
    second = second % 3600
    minute = second // 60 + 1

    if day > 31:
        day = 32
        hour = 1
        minute = 1

    return day, hour, minute
