#!/usr/bin/env python3
"""
数据预处理脚本 - 将数据预处理和训练分离
完全模拟main_dist.py中的预处理逻辑，使用相同的args参数
使用DataLoader的num_workers进行处理，而不是多进程池
"""

import os
import pickle
import time
from pathlib import Path
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

# 导入main_dist.py中的参数和函数
from main_dist import get_args, set_path, set_seed
from dataset import MyDataset


def preprocess_dataset(args, dataset, train_dataset, log_file=None):
    """
    预处理整个训练数据集，将每个batch的处理结果存储到磁盘
    完全模拟main_dist.py中的预处理逻辑，使用DataLoader的num_workers
    
    Args:
        args: 参数配置（来自main_dist.py）
        dataset: MyDataset实例
        train_dataset: 训练数据子集
        log_file: 日志文件
    
    Returns:
        Path: 预处理数据存储目录
    """
    print("开始预处理数据集...")
    if log_file:
        log_file.write("开始预处理数据集...\n")
        log_file.flush()

    # 创建预处理数据存储目录
    batch_data_dir = Path(args.log_path) / "batch_data"
    batch_data_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧的预处理文件
    for old_file in batch_data_dir.glob("batch_*.pkl"):
        old_file.unlink()

    # 创建临时DataLoader进行预处理 - 完全模拟main_dist.py中的逻辑
    temp_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=dataset.collate_fn, prefetch_factor=8, persistent_workers=False, pin_memory=False
    )

    print(f"总共需要预处理 {len(temp_loader)} 个批次")
    if log_file:
        log_file.write(f"总共需要预处理 {len(temp_loader)} 个批次\n")
        log_file.flush()

    # 预处理每个batch - 完全模拟main_dist.py中的逻辑
    for batch_idx, batch in tqdm(enumerate(temp_loader), total=len(temp_loader), desc="预处理批次"):
        # 直接保存完整的batch数据，保持原有结构
        # 这种方式便于后续在batch中新增变量而不需要修改存储逻辑

        batch_file = batch_data_dir / f"batch_{batch_idx:06d}.pkl"

        # 将batch数据保存为pickle文件
        # batch是一个元组，包含所有必要的数据
        with open(batch_file, 'wb') as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)

        if batch_idx % 100 == 0:
            print(f"{batch_idx} done.")

    # 保存预处理元数据 - 完全模拟main_dist.py中的逻辑
    metadata = {
        'total_batches': len(temp_loader),
        'batch_size': args.batch_size,
        'maxlen': args.maxlen,
        'feature_types': dataset.feature_types,
        'feat_statistics': dataset.feat_statistics,
        'storage_format': 'pickle_batch'  # 标识存储格式
    }

    metadata_file = batch_data_dir / "metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"预处理完成，共处理 {len(temp_loader)} 个批次，数据保存在 {batch_data_dir}")
    if log_file:
        log_file.write(f"预处理完成，共处理 {len(temp_loader)} 个批次，数据保存在 {batch_data_dir}\n")
        log_file.flush()

    # 统计并输出预处理文件总大小 - 完全模拟main_dist.py中的逻辑
    try:
        total_bytes = 0
        file_count = 0
        for fpath in batch_data_dir.glob("batch_*.pkl"):
            try:
                total_bytes += fpath.stat().st_size
                file_count += 1
            except Exception:
                pass

        def _format_size(num_bytes: int) -> str:
            units = ["B", "KB", "MB", "GB", "TB"]
            size = float(num_bytes)
            idx = 0
            while size >= 1024.0 and idx < len(units) - 1:
                size /= 1024.0
                idx += 1
            return f"{size:.2f} {units[idx]}"

        size_str = _format_size(total_bytes)
        msg = f"预处理文件统计: {file_count} 个batch文件，总大小 {size_str}"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()
    except Exception as e:
        print(f"统计预处理文件大小失败: {e}")
        if log_file:
            log_file.write(f"统计预处理文件大小失败: {e}\n")
            log_file.flush()

    return batch_data_dir


def main():
    """主函数 - 完全使用main_dist.py中的参数和逻辑"""
    # 使用main_dist.py中的参数解析
    args = get_args()
    
    # 设置路径和随机种子 - 完全模拟main_dist.py
    set_seed(args.seed)
    set_path(args)
    
    print("=" * 50)
    print("数据预处理脚本启动")
    print("=" * 50)
    print(f"数据路径: {args.data_path}")
    print(f"日志路径: {args.log_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大序列长度: {args.maxlen}")
    print(f"Worker数量: {args.num_workers}")
    print(f"强制重新处理: {args.force_reprocess}")
    print("=" * 50)
    
    # 检查数据路径
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"数据路径不存在: {args.data_path}")
    
    # 创建日志目录
    Path(args.log_path).mkdir(parents=True, exist_ok=True)
    
    # 初始化数据集 - 完全模拟main_dist.py
    print("初始化数据集...")
    dataset = MyDataset(args.data_path, args)
    
    # 创建训练数据子集 - 完全模拟main_dist.py
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.99, 0.01])
    
    print(f"数据集大小: {len(dataset)}")
    print(f"训练集大小: {len(train_dataset)}")
    
    # 检查是否已存在预处理数据 - 完全模拟main_dist.py中的逻辑
    batch_data_dir = Path(args.log_path) / "batch_data"
    metadata_file = batch_data_dir / "metadata.pkl"
    
    if metadata_file.exists() and not args.force_reprocess:
        try:
            with open(metadata_file, 'rb') as f:
                existing_metadata = pickle.load(f)
            
            # 检查预处理数据是否与当前配置匹配
            if (existing_metadata.get('batch_size') == args.batch_size and
                existing_metadata.get('maxlen') == args.maxlen and
                len(list(batch_data_dir.glob("batch_*.pkl"))) == existing_metadata.get('total_batches', 0)):
                print("找到匹配的预处理数据，跳过预处理")
                print(f"预处理数据目录: {batch_data_dir}")
                return batch_data_dir
            else:
                print("预处理数据配置不匹配，将重新预处理")
        except Exception as e:
            print(f"检查预处理数据时发生错误: {e}")
            print("将重新进行预处理")
    
    # 开始预处理 - 使用完全模拟main_dist.py的预处理逻辑
    try:
        batch_data_dir = preprocess_dataset(args, dataset, train_dataset)
        print("预处理完成！")
        return batch_data_dir
    except Exception as e:
        print(f"预处理过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()
