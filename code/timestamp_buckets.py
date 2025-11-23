#!/usr/bin/env python3
"""
时间戳桶化分析脚本
从序列文件中提取所有时间戳，按时间跨度分成指定数量的桶
"""

import os
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys
import time
import ujson
from collections import Counter

def get_data_paths():
    """
    获取数据路径配置
    """
    data_path = os.environ.get('USER_CACHE_PATH', './user_cache')
    
    return {
        'output_file': Path(data_path) / 'item_exposure' / 'timestamp_buckets.pkl',
        'item_count_file': Path(data_path) / 'item_exposure' / 'item_counts_per_bucket.pkl',
    }

def create_timestamp_buckets_by_time_span(seq_file_path, num_buckets=16384):
    """
    按时间跨度创建桶（等时间间隔分桶，向量化实现）

    - 仅统计 item 记录（跳过 user 占位行：item_id 为 None 或 0）
    - 使用 numpy 直方图一次性统计每桶数量，避免 O(N*B)

    Args:
        seq_file_path (Path): 序列文件路径
        num_buckets (int): 桶的数量

    Returns:
        list: 桶信息列表，每个桶包含区间边界与该桶的时间戳个数
    """
    print(f"🚀 等时间间隔分桶：读取 {seq_file_path} 并创建 {num_buckets} 个桶...")
    start_time = time.time()

    timestamps = []
    line_count = 0

    try:
        with open(seq_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    user_sequence = ujson.loads(line)
                    for record in user_sequence:
                        if len(record) < 6:
                            continue
                        item_id = record[1]
                        ts = record[5]
                        # 仅保留 item 行
                        if (item_id is None) or (item_id == 0):
                            continue
                        if ts is None or ts <= 0:
                            continue
                        timestamps.append(ts)
                    line_count += 1
                    if line_count % 1000000 == 0:
                        elapsed = max(1e-9, time.time() - start_time)
                        print(f"  已处理 {line_count} 行，速度 {line_count/elapsed:.1f} 行/秒，累计时间戳 {len(timestamps)}")
                except Exception as e:
                    # 跳过异常行
                    continue
    except FileNotFoundError:
        print(f"❌ 错误: 序列文件 {seq_file_path} 未找到")
        return []
    except Exception as e:
        print(f"❌ 处理文件时发生错误: {e}")
        return []

    if not timestamps:
        print("❌ 未找到任何时间戳")
        return []

    ts_np = np.asarray(timestamps, dtype=np.float64)
    min_ts = float(ts_np.min())
    max_ts = float(ts_np.max())

    print(f"📅 时间戳范围: {datetime.fromtimestamp(min_ts)} 到 {datetime.fromtimestamp(max_ts)}")
    print(f"⏱️  总时间跨度: {(max_ts - min_ts) / 86400:.2f} 天")

    if num_buckets <= 0:
        num_buckets = 1
    # 生成等距边界（包含右端点），长度 num_buckets+1
    edges = np.linspace(min_ts, max_ts, num_buckets + 1, dtype=np.float64)
    # 使用直方图统计每个半开区间 [edges[i], edges[i+1]) 的数量（最后一桶包含右端点）
    counts, _ = np.histogram(ts_np, bins=edges)
    
    buckets = []
    for i in range(num_buckets):
        start_ts = float(edges[i])
        end_ts = float(edges[i + 1])
        # 时间跨度（小时）
        span_hours = max(0.0, (end_ts - start_ts) / 3600.0)
        buckets.append({
            'bucket_id': i,
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
            'start_datetime': datetime.fromtimestamp(start_ts).isoformat(),
            'end_datetime': datetime.fromtimestamp(end_ts).isoformat(),
            'timestamp_count': int(counts[i]),
            'time_span_hours': span_hours,
        })

    print(f"✅ 分桶完成，共 {num_buckets} 个桶，非空桶 {int((counts>0).sum())} 个")
    print(f"  每桶平均 {counts.mean():.2f}，最大 {counts.max()}，最小 {counts.min()}")
    print(f"⏱️  总用时: {time.time() - start_time:.1f}秒")

    return buckets

def save_buckets(buckets, output_file):
    """
    保存桶信息到文件
    
    Args:
        buckets (list): 桶信息列表
        output_file (Path): 输出文件路径
    """
    print("💾 正在保存桶信息...")
    save_start = time.time()
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(buckets, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        save_time = time.time() - save_start
        print(f"✅ 桶信息已保存到: {output_file}，保存用时: {save_time:.1f}秒")
        
        # 打印统计信息
        non_empty_buckets = [b for b in buckets if b['timestamp_count'] > 0]
        print(f"\n📊 桶统计信息:")
        print(f"  总桶数: {len(buckets)}")
        print(f"  非空桶数: {len(non_empty_buckets)}")
        print(f"  总时间戳数: {sum(b['timestamp_count'] for b in buckets)}")
        
        if non_empty_buckets:
            total_days = (buckets[-1]['end_timestamp'] - buckets[0]['start_timestamp']) / 86400
            print(f"  时间范围: {buckets[0]['start_datetime']} 到 {buckets[-1]['end_datetime']}")
            print(f"  总天数: {total_days:.2f} 天")
            avg_time_span = np.mean([b['time_span_hours'] for b in buckets if b['time_span_hours'] > 0])
            print(f"  平均每桶时间跨度: {avg_time_span:.2f} 小时")
            counts = [b['timestamp_count'] for b in non_empty_buckets]
            print(f"  时间戳分布: 最小 {min(counts)}, 最大 {max(counts)}, "
                  f"平均 {np.mean(counts):.1f}")
        
    except Exception as e:
        print(f"❌ 保存桶信息失败: {e}")

def create_timestamp_buckets_by_frequency(seq_file_path, num_buckets=16384):
    """
    等频分桶（原版，两次文件读取）

    两阶段：
    1) 第一次读取，提取全部时间戳并排序，构建分位边界(boundaries)
    2) 第二次读取，依据边界将记录映射到桶，统计每个桶内的 item 出现次数

    Args:
        seq_file_path (Path): 序列文件路径
        num_buckets (int): 桶的数量

    Returns:
        tuple[list, list[dict]]: (桶元数据列表, 每桶 item->count 映射列表)
    """
    print(f"🚀 [原版方法] 开始等频分桶并统计每桶 item 次数，目标桶数: {num_buckets} ...")
    global_start = time.time()

    # 第一阶段：收集并排序所有时间戳
    timestamps = []
    line_count = 0
    try:
        with open(seq_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    user_sequence = ujson.loads(line)
                    for record in user_sequence:
                        if len(record) >= 6:
                            item_id = record[1]
                            if (item_id is None) or (item_id == 0):
                                continue
                            timestamp = record[5]
                            if timestamp is not None and timestamp > 0:
                                timestamps.append(timestamp)
                    line_count += 1
                    if line_count % 100000 == 0:
                        elapsed = time.time() - global_start
                        speed = line_count / max(elapsed, 1e-9)
                        print(f"  [阶段1] 已处理 {line_count} 行，速度: {speed:.1f} 行/秒，时间戳数量: {len(timestamps)}")
                except Exception as e:
                    print(f"处理行时出错: {line[:100]}..., 错误: {e}")
                    continue
    except FileNotFoundError:
        print(f"❌ 错误: 序列文件 {seq_file_path} 未找到")
        return [], []
    
    if not timestamps:
        print("❌ 未找到任何时间戳")
        return [], []

    print(f"📊 总共收集到 {len(timestamps)} 个时间戳")
    print("🔄 正在排序时间戳用于等频分桶...")
    sort_start = time.time()
    timestamps.sort()
    print(f"✅ 排序完成，用时: {time.time() - sort_start:.1f}秒")

    total = len(timestamps)
    num_buckets = max(1, min(num_buckets, total))

    # 构建边界
    boundaries = [timestamps[int(i * total / num_buckets)] for i in range(num_buckets)]
    boundaries.append(timestamps[-1] + 1)

    buckets = []
    for i in range(num_buckets):
        start_ts = boundaries[i]
        end_ts_inclusive = boundaries[i + 1] - 1
        buckets.append({
            'bucket_id': i, 'start_timestamp': start_ts, 'end_timestamp': end_ts_inclusive,
            'start_datetime': datetime.fromtimestamp(start_ts).isoformat(),
            'end_datetime': datetime.fromtimestamp(end_ts_inclusive).isoformat(),
            'timestamp_count': 0, 'time_span_hours': max(0.0, (end_ts_inclusive - start_ts) / 3600)
        })

    # 第二阶段：统计
    print("📦 正在按边界统计每桶 item 次数...")
    import bisect
    counts_per_bucket = [Counter() for _ in range(num_buckets)]
    
    line_count = 0
    start_phase2 = time.time()
    with open(seq_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip();
            if not line: continue
            try:
                user_sequence = ujson.loads(line)
                for record in user_sequence:
                    if len(record) >= 6:
                        item_id = record[1]
                        if (item_id is None) or (item_id == 0): continue
                        ts = record[5]
                        if ts is None or ts <= 0: continue
                        b_idx = bisect.bisect_right(boundaries, ts) - 1
                        b_idx = max(0, min(b_idx, num_buckets - 1))
                        counts_per_bucket[b_idx][item_id] += 1
                        buckets[b_idx]['timestamp_count'] += 1
                line_count += 1
                if line_count % 100000 == 0:
                    elapsed = time.time() - start_phase2
                    speed = line_count / max(elapsed, 1e-9)
                    print(f"  [阶段2] 已处理 {line_count} 行，速度: {speed:.1f} 行/秒")
            except Exception: continue

    total_time = time.time() - global_start
    print(f"✅ 等频分桶与统计完成，总用时: {total_time:.1f}秒")
    return buckets, [dict(c) for c in counts_per_bucket]

def create_timestamp_buckets_by_frequency_accelerated(seq_file_path, num_buckets=32768):
    """
    等频分桶加速版 (单次读取)

    通过一次性将 (timestamp, item_id) 读入内存并排序，避免对大文件进行第二次IO扫描，
    从而大幅提升处理速度。

    注意：此方法会消耗更多内存，因为它需要存储所有的 (timestamp, item_id) 对。
    如果内存不足，原有的双次扫描方法可能更适用。

    流程:
    1) 读取所有 item 记录的 (timestamp, item_id) 到内存。
    2) 基于 timestamp 对记录进行排序。
    3) 将排序后的记录列表按数量平均切分成 num_buckets 份。
    4) 为每个切片生成桶信息并统计 item 出现次数。

    Args:
        seq_file_path (Path): 序列文件路径
        num_buckets (int): 桶的数量

    Returns:
        tuple[list, list[dict]]: (桶元数据列表, 每桶 item->count 映射列表)
    """
    print(f"🚀 [加速版] 开始等频分桶，目标桶数: {num_buckets} ...")
    global_start = time.time()

    # 1. 一次性读取所有 (timestamp, item_id) 对
    records = []
    line_count = 0
    print("  [阶段1/3] 正在读取所有记录到内存...")
    try:
        with open(seq_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    user_sequence = ujson.loads(line)
                    for record in user_sequence:
                        if len(record) >= 6:
                            item_id = record[1]
                            if (item_id is None) or (item_id == 0):
                                continue
                            timestamp = record[5]
                            if timestamp is not None and timestamp > 0:
                                records.append((timestamp, item_id))
                    line_count += 1
                    if line_count % 100000 == 0:
                        elapsed = time.time() - global_start
                        speed = line_count / max(elapsed, 1e-9)
                        print(f"    已处理 {line_count} 行，速度: {speed:.1f} 行/秒，记录数: {len(records)}")
                except Exception:
                    # 跳过格式错误的行
                    continue
    except FileNotFoundError:
        print(f"❌ 错误: 序列文件 {seq_file_path} 未找到")
        return [], []
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return [], []

    if not records:
        print("❌ 未找到任何有效记录")
        return [], []

    print(f"  ✅ 读取完成，总共收集到 {len(records)} 条记录。")

    # 2. 排序
    print("  [阶段2/3] 正在基于时间戳排序记录...")
    sort_start = time.time()
    records.sort(key=lambda x: x[0]) # 按时间戳排序
    print(f"  ✅ 排序完成，用时: {time.time() - sort_start:.1f}秒")

    # 3. 分桶与统计
    print(f"  [阶段3/3] 正在创建 {num_buckets} 个桶并统计 item 次数...")
    bucketing_start = time.time()
    
    buckets = []
    item_counts_per_bucket = []
    
    total_records = len(records)
    num_buckets = max(1, min(num_buckets, total_records))

    for i in range(num_buckets):
        # 计算当前桶在 records 列表中的起止索引
        start_index = int(i * total_records / num_buckets)
        end_index = int((i + 1) * total_records / num_buckets)
        
        # 获取当前桶的记录切片
        bucket_slice = records[start_index:end_index]
        
        if not bucket_slice:
            start_ts = buckets[-1]['end_timestamp'] if buckets else records[0][0]
            item_counts = {}
            count = 0
            end_ts = start_ts
        else:
            # 使用生成器表达式和 Counter 高效统计
            item_counts = Counter(rec[1] for rec in bucket_slice)
            start_ts = bucket_slice[0][0]
            end_ts = bucket_slice[-1][0]
            count = len(bucket_slice)

        buckets.append({
            'bucket_id': i,
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
            'start_datetime': datetime.fromtimestamp(start_ts).isoformat(),
            'end_datetime': datetime.fromtimestamp(end_ts).isoformat(),
            'timestamp_count': count,
            'time_span_hours': max(0.0, (end_ts - start_ts) / 3600)
        })
        item_counts_per_bucket.append(dict(item_counts))

    print(f"  ✅ 分桶与统计完成，用时: {time.time() - bucketing_start:.1f}秒")
    total_time = time.time() - global_start
    print(f"✅ [加速版] 等频分桶与统计完成，总用时: {total_time:.1f}秒")
    
    return buckets, item_counts_per_bucket

def save_item_counts(item_counts_per_bucket, output_file):
    """
    保存每个桶内的 item 计数字典列表到文件

    Args:
        item_counts_per_bucket (list[dict]): 每个桶一个字典，键为 item_id，值为次数
        output_file (Path): 输出文件路径
    """
    print("💾 正在保存每桶 item 计数...")
    save_start = time.time()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(item_counts_per_bucket, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ item 计数已保存到: {output_file}，保存用时: {time.time() - save_start:.1f}秒")
    except Exception as e:
        print(f"❌ 保存 item 计数失败: {e}")

def print_bucket_preview(buckets, preview_count=10):
    """
    打印桶信息预览
    
    Args:
        buckets (list): 桶信息列表
        preview_count (int): 预览的桶数量
    """
    if not buckets:
        return
    print(f"\n--- 桶信息预览 (前{preview_count}个桶) ---")
    
    for i, bucket in enumerate(buckets[:preview_count]):
        print(f"桶 {bucket['bucket_id']}: {bucket['start_datetime']} - {bucket['end_datetime']} "
              f"({bucket['timestamp_count']} 个时间戳, {bucket['time_span_hours']:.2f} 小时)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='时间戳桶化分析脚本，通过单次读取内存排序加速等频分桶。'
    )
    parser.add_argument(
        '--seq_file',
        type=str,
        help='序列文件路径'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='输出文件路径，如果未指定则使用默认路径'
    )
    parser.add_argument(
        '--buckets',
        type=int,
        default=8192,  # 修改：默认桶数设置为 32k
        help='桶的数量 (默认: 32768)'
    )
    parser.add_argument(
        '--method',
        choices=['accelerated_frequency', 'frequency', 'timespan'], # 修改：添加新方法并设为默认
        default='frequency',
        help='分桶方法: accelerated_frequency(单次读取加速等频，推荐) / frequency(原版等频) / timespan(等时间跨度)'
    )
    parser.add_argument(
        '--item_count_file',
        type=str,
        help='每桶 item 计数输出文件路径（不指定则使用默认路径）'
    )
    
    args = parser.parse_args()
    
    # 获取路径
    if args.seq_file:
        seq_file_path = Path(args.seq_file)
    else:
        data_path = os.environ.get('TRAIN_DATA_PATH', './data')
        seq_file_path = Path(data_path) / 'seq.jsonl'
    
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        paths = get_data_paths()
        output_file = paths['output_file']

    if args.item_count_file:
        item_count_file = Path(args.item_count_file)
    else:
        paths = get_data_paths()
        item_count_file = paths['item_count_file']
    
    print("="*60)
    print("=== 时间戳桶化分析脚本 ===")
    print(f"序列文件: {seq_file_path}")
    print(f"输出文件: {output_file}")
    print(f"桶数量: {args.buckets}")
    print(f"分桶方法: {args.method}")
    if 'frequency' in args.method:
        print(f"item计数输出: {item_count_file}")
    print("="*60)
    
    if not seq_file_path.exists():
        print(f"❌ 错误: 序列文件不存在 {seq_file_path}")
        sys.exit(1)
    
    # 根据选择的方法创建桶
    item_counts = None
    if args.method == 'accelerated_frequency':
        buckets, item_counts = create_timestamp_buckets_by_frequency_accelerated(seq_file_path, args.buckets)
    elif args.method == 'frequency':
        buckets, item_counts = create_timestamp_buckets_by_frequency(seq_file_path, args.buckets)
    elif args.method == 'timespan':
        buckets = create_timestamp_buckets_by_time_span(seq_file_path, args.buckets)
    else:
        print(f"❌ 错误：未知的分桶方法 '{args.method}'")
        sys.exit(1)

    if not buckets:
        print("❌ 未能创建任何桶，程序终止。")
        sys.exit(1)

    # 保存结果
    save_buckets(buckets, output_file)
    if item_counts is not None:
        save_item_counts(item_counts, item_count_file)

    print_bucket_preview(buckets)
    
    print("\n🎯 时间戳桶化分析完成！")

if __name__ == "__main__":
    main()