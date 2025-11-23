#!/usr/bin/env python3
"""
本脚本用于分析用户行为序列数据，计算每个物品的关键指标（优化版）
性能优化：
1. 使用ujson替代json加速解析
2. 批量处理减少循环开销
3. 优化日期转换缓存
4. 使用numpy向量化操作
5. 减少内存使用
6. 添加进度监控
"""

import os
import numpy as np
import pickle
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import sys
import time
import ujson  # 更快的JSON解析库
import mmap  # 内存映射文件加速读取
import platform  # 检测操作系统


# =============================================================================
# 核心功能模块
# =============================================================================

def get_data_paths():
    """
    获取并配置数据输入和输出路径。
    通过环境变量 TRAIN_DATA_PATH 和 USER_CACHE_PATH 进行配置，如果未设置则使用默认值。
    """
    data_path = os.environ.get('TRAIN_DATA_PATH', './data')
    output_path = os.environ.get('USER_CACHE_PATH', './user_cache')

    return {
        'seq_file': Path(data_path) / 'seq.jsonl',
        'output_dir': Path(output_path) / 'item_exposure',
    }


def analyze_item_actions(seq_file_path, output_dir):
    """
    分析每个item的曝光、点击、转化行为，并计算在平均曝光日的相关指标（优化版）

    Args:
        seq_file_path (Path): 行为序列数据文件 (seq.jsonl) 的路径。
        output_dir (Path): 输出结果文件的存放目录。
    """
    print("🚀 开始分析物品的曝光、点击和转化行为 (优化版)...")
    start_time = time.time()

    # 使用更高效的数据结构
    item_stats = defaultdict(lambda: {
        'all_timestamps': [],
        'exposures': [],
        'clicks': [],
        'conversions': []
    })

    # 使用Counter进行快速计数
    item_daily_counts = defaultdict(lambda: defaultdict(Counter))
    global_daily_counts = defaultdict(Counter)

    # action_type 到行为名称的映射
    action_map = {0: 'exposures', 1: 'clicks', 2: 'conversions'}

    # 日期转换缓存
    date_cache = {}

    def get_date_from_timestamp(ts):
        """缓存日期转换结果"""
        if ts not in date_cache:
            # 简化日期转换，只保留日期部分
            date_cache[ts] = date.fromtimestamp(ts)
        return date_cache[ts]

    # 批量处理大小
    BATCH_SIZE = 1000000
    processed_lines = 0
    processed_records = 0
    line_count = 0

    try:
        with open(seq_file_path, 'r', encoding='utf-8') as f:
            # 使用内存映射加速读取
            if platform.system() == 'Windows':
                # Windows需要特殊处理
                mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                # Linux/macOS处理
                mmapped_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            batch_lines = []

            # 使用迭代器读取行
            pos = 0
            mmapped_file.seek(0)
            while True:
                line = mmapped_file.readline()
                if not line:
                    break

                try:
                    # 解码字节为字符串
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line:
                        continue

                    batch_lines.append(decoded_line)
                    line_count += 1

                    # 每50万行打印一次进度
                    if line_count % 500000 == 0:
                        elapsed = time.time() - start_time
                        speed = line_count / elapsed
                        print(f"  已处理 {line_count} 行，速度: {speed:.1f} 行/秒，用时: {elapsed:.1f}秒")

                    # 批量处理
                    if len(batch_lines) >= BATCH_SIZE:
                        processed_records += process_batch(
                            batch_lines, item_stats, item_daily_counts,
                            global_daily_counts, action_map, get_date_from_timestamp
                        )
                        processed_lines += len(batch_lines)
                        batch_lines = []

                except Exception as e:
                    print(f"处理行时出错: {line}, 错误: {e}")

            # 处理剩余记录
            if batch_lines:
                processed_records += process_batch(
                    batch_lines, item_stats, item_daily_counts,
                    global_daily_counts, action_map, get_date_from_timestamp
                )
                processed_lines += len(batch_lines)

            # 关闭内存映射
            mmapped_file.close()

    except FileNotFoundError:
        print(f"❌ 错误: 数据文件 {seq_file_path} 未找到。请检查路径或环境变量配置。")
        return
    except Exception as e:
        print(f"❌ 处理文件时发生未知错误: {e}")
        return

    data_processing_time = time.time() - start_time
    print(f"✅ 数据聚合完成，处理了 {processed_lines} 行，{processed_records} 条记录，用时: {data_processing_time:.1f}秒")
    print("🔢 开始计算各项指标...")

    calc_start_time = time.time()
    results = []

    # 批量计算，减少重复操作
    total_items = len(item_stats)
    item_ids = list(item_stats.keys())

    # 批量处理物品，每1000000个物品处理一次
    for i in range(0, total_items, 1000000):
        batch_ids = item_ids[i:i + 1000000]
        batch_results = []

        for item_id in batch_ids:
            stats = item_stats[item_id]
            all_timestamps = stats['all_timestamps']

            # 使用numpy进行快速统计计算
            if all_timestamps:
                all_timestamps_array = np.array(all_timestamps)

                # 获取历史总量
                total_exposures = len(stats['exposures'])
                total_clicks = len(stats['clicks'])
                total_conversions = len(stats['conversions'])

                # 使用numpy进行快速计算
                start_time_ts = float(all_timestamps_array.min())
                end_time_ts = float(all_timestamps_array.max())
                avg_all_time_ts = float(all_timestamps_array.mean())

                avg_day = get_date_from_timestamp(avg_all_time_ts)

                # 快速获取当天数据
                exposures_on_avg_day = item_daily_counts[item_id]['exposures'].get(avg_day, 0)
                clicks_on_avg_day = item_daily_counts[item_id]['clicks'].get(avg_day, 0)
                conversions_on_avg_day = item_daily_counts[item_id]['conversions'].get(avg_day, 0)

                # 获取当天全局统计
                global_exposures = global_daily_counts['exposures'].get(avg_day, 0)
                global_clicks = global_daily_counts['clicks'].get(avg_day, 0)
                global_conversions = global_daily_counts['conversions'].get(avg_day, 0)

                # 快速计算百分比
                exposure_pct = (exposures_on_avg_day / global_exposures * 100) if global_exposures > 0 else 0
                click_pct = (clicks_on_avg_day / global_clicks * 100) if global_clicks > 0 else 0
                conversion_pct = (conversions_on_avg_day / global_conversions * 100) if global_conversions > 0 else 0

                metrics_on_avg_day = {
                    'absolute_counts': {
                        'exposures': exposures_on_avg_day,
                        'clicks': clicks_on_avg_day,
                        'conversions': conversions_on_avg_day,
                    },
                    'global_counts_on_day': {
                        'exposures': global_exposures,
                        'clicks': global_clicks,
                        'conversions': global_conversions,
                    },
                    'percentage_of_global': {
                        'exposures_pct': f"{exposure_pct:.2f}%",
                        'clicks_pct': f"{click_pct:.2f}%",
                        'conversions_pct': f"{conversion_pct:.2f}%",
                    }
                }
            else:
                # 空数据的默认值
                start_time_ts = None
                end_time_ts = None
                avg_all_time_ts = None
                total_exposures = 0
                total_clicks = 0
                total_conversions = 0
                metrics_on_avg_day = {
                    'absolute_counts': {
                        'exposures': 0,
                        'clicks': 0,
                        'conversions': 0,
                    },
                    'global_counts_on_day': {
                        'exposures': 0,
                        'clicks': 0,
                        'conversions': 0,
                    },
                    'percentage_of_global': {
                        'exposures_pct': "0.00%",
                        'clicks_pct': "0.00%",
                        'conversions_pct': "0.00%",
                    }
                }

            # 构建结果
            batch_results.append({
                'item_id': item_id,
                'exposure_start_ts': start_time_ts,
                'exposure_end_ts': end_time_ts,
                'exposure_avg_ts': avg_all_time_ts,
                'metrics_on_avg_day': metrics_on_avg_day,
                'total_counts': {
                    'exposures': total_exposures,
                    'clicks': total_clicks,
                    'conversions': total_conversions,
                    'all_actions': len(all_timestamps),
                }
            })

        results.extend(batch_results)

        # 每处理1000000个物品打印一次进度
        processed_count = min(i + 1000000, total_items)
        if processed_count % 10000000 == 0 or processed_count == total_items:
            elapsed = time.time() - calc_start_time
            progress = processed_count / total_items * 100
            speed = processed_count / elapsed if elapsed > 0 else float('inf')
            print(f"  计算进度: {processed_count}/{total_items} ({progress:.1f}%), 速度: {speed:.1f} item/秒")

    calc_time = time.time() - calc_start_time
    total_time = time.time() - start_time
    print(f"📊 指标计算完成，共处理 {len(results)} 个物品，计算用时: {calc_time:.1f}秒")
    print(f"⏱️  总用时: {total_time:.1f}秒 (数据处理: {data_processing_time:.1f}秒, 指标计算: {calc_time:.1f}秒)")

    # 快速排序
    print("🔄 正在排序结果...")
    sort_start = time.time()
    results.sort(key=lambda x: x['total_counts']['all_actions'], reverse=True)
    sort_time = time.time() - sort_start
    print(f"✅ 排序完成，用时: {sort_time:.1f}秒")

    # 打印预览（限制数量以提高速度）
    print("\n--- 分析结果预览 (按总行为数排序) ---")
    preview_count = min(10, len(results))
    for res in results[:preview_count]:
        if res['exposure_avg_ts'] is not None:
            avg_day_str = get_date_from_timestamp(res['exposure_avg_ts']).isoformat()
        else:
            avg_day_str = "N/A"

        if res['exposure_start_ts'] is not None:
            start_day_str = get_date_from_timestamp(res['exposure_start_ts']).isoformat()
            end_day_str = get_date_from_timestamp(res['exposure_end_ts']).isoformat()
        else:
            start_day_str = "N/A"
            end_day_str = "N/A"

        print(f"\n[ Item ID: {res['item_id']} ]")
        print(f"  所有行为开始/结束时间: {start_day_str} / {end_day_str}")
        print(f"  平均曝光时间: {avg_day_str}")
        counts = res['total_counts']
        print(
            f"  历史总量: 曝光={counts['exposures']}, 点击={counts['clicks']}, 转化={counts['conversions']}, 总行为={counts['all_actions']}")
        metrics = res['metrics_on_avg_day']
        abs_counts = metrics['absolute_counts']
        global_counts = metrics['global_counts_on_day']
        pcts = metrics['percentage_of_global']
        print(
            f"  在平均日的指标: 该物品({abs_counts['exposures']}/{abs_counts['clicks']}/{abs_counts['conversions']}) / 全局({global_counts['exposures']}/{global_counts['clicks']}/{global_counts['conversions']}) = 占比({pcts['exposures_pct']}/{pcts['clicks_pct']}/{pcts['conversions_pct']})")

    # 保存结果
    print("\n💾 正在保存结果...")
    save_start = time.time()
    output_file = output_dir / 'item_exposure_data.pkl'
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_time = time.time() - save_start
        print(f"✅ 完整分析结果已保存到: {output_file}，保存用时: {save_time:.1f}秒")

        # 性能总结
        final_total_time = time.time() - start_time
        print(f"\n🎯 性能总结:")
        print(f"  总用时: {final_total_time:.1f}秒")
        print(f"  数据处理: {data_processing_time:.1f}秒 ({data_processing_time / final_total_time * 100:.1f}%)")
        print(f"  指标计算: {calc_time:.1f}秒 ({calc_time / final_total_time * 100:.1f}%)")
        print(f"  排序: {sort_time:.1f}秒 ({sort_time / final_total_time * 100:.1f}%)")
        print(f"  保存: {save_time:.1f}秒 ({save_time / final_total_time * 100:.1f}%)")
        print(f"  处理速度: {line_count / final_total_time:.0f} 行/秒")
        print(f"  记录处理速度: {processed_records / final_total_time:.0f} 记录/秒")

    except Exception as e:
        print(f"\n❌ 保存结果文件失败: {e}")


def process_batch(batch_lines, item_stats, item_daily_counts,
                  global_daily_counts, action_map, get_date_func):
    """处理一批记录，返回处理的记录数"""
    records_count = 0
    for line in batch_lines:
        try:
            user_sequence = ujson.loads(line)

            # 处理当前用户的所有记录
            for record in user_sequence:
                _, item_id, _, _, action_type, timestamp = record
                records_count += 1

                # 只处理有效的、已知的行为类型
                if item_id is not None and action_type in action_map:
                    action_name = action_map[action_type]

                    # 使用缓存的日期转换
                    day_key = get_date_func(timestamp)

                    # 记录时间戳到所有行为列表
                    item_stats[item_id]['all_timestamps'].append(timestamp)

                    # 记录特定行为类型的时间戳
                    if action_name in item_stats[item_id]:
                        item_stats[item_id][action_name].append(timestamp)
                    else:
                        # 如果行为类型不存在，创建新列表
                        item_stats[item_id][action_name] = [timestamp]

                    # 使用Counter进行快速计数
                    item_daily_counts[item_id][action_name][day_key] += 1
                    global_daily_counts[action_name][day_key] += 1

        except Exception as e:
            print(f"处理记录时出错: {line}, 错误: {e}")

    return records_count


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """主函数，用于处理命令行参数和启动分析。"""
    parser = argparse.ArgumentParser(
        description='物品行为分析脚本，计算曝光、点击、转化等关键指标 (性能优化版)。'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='action_analysis',
        choices=['action_analysis'],
        help='指定运行模式。当前仅支持 "action_analysis"。'
    )

    args = parser.parse_args()

    if args.mode == 'action_analysis':
        paths = get_data_paths()
        print("=" * 60)
        print("=== 物品曝光与行为分析 (性能优化版) ===")
        print(f"序列文件: {paths['seq_file']}")
        print(f"输出目录: {paths['output_dir']}")
        print("=" * 60)

        if not paths['seq_file'].exists():
            print(f"错误: 序列文件不存在 {paths['seq_file']}")
            sys.exit(1)

        analyze_item_actions(paths['seq_file'], paths['output_dir'])
    else:
        print(f"❌ 未知模式: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()