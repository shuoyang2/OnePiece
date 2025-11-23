import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.system('pip3 install orjson')

import numpy as np

import torch
import random

torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from threading import Lock

from dataset import MyDataset
from model import BaselineModel
from utils import *
from dataparallel import create_my_dataparallel, create_my_dataparallel_optimizer
from torch.cuda.amp import autocast, GradScaler
import multiprocessing
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed


def _get_ckpt_path(name) -> str:
    ckpt_path = Path(os.environ.get("USER_CACHE_PATH")) / "train_infer" / name
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pth"):
            return str(Path(ckpt_path) / item)
    raise FileNotFoundError(f"No .pt checkpoint under {ckpt_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_path(args):
    args.log_path = os.environ.get('TRAIN_LOG_PATH') if not args.debug else "./results/log"
    args.tb_path = os.environ.get('TRAIN_TF_EVENTS_PATH') if not args.debug else "./results/tensorboard"
    args.data_path = os.environ.get('TRAIN_DATA_PATH') if not args.debug else "./data"
    args.ckpt_path = os.environ.get('TRAIN_CKPT_PATH') if not args.debug else "./results/ckpt"
    args.user_cache_path = os.environ.get('USER_CACHE_PATH') if not args.debug else "./user_cache/"


def get_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--mode', type=str, default="train", help='batch size')
    # seed
    parser.add_argument('--seed', type=int, default=42)
    # Train params
    parser.add_argument('--single_device_batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    # 添加梯度累积参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates to accumulate gradients for before performing a backward/step pass.')

    parser.add_argument('--similarity_function', default="cosine", type=str, choices=["cosine", "dot"])
    parser.add_argument('--clip_grad_norm', default=1.0, type=float)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--use_cosine_annealing', action='store_true', default=True)
    parser.add_argument('--lr_eta_min', default=0.0, type=float, help='Minimum learning rate for cosine annealing.')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--sparse_embedding', action='store_true', default=False)
    parser.add_argument('--embedding_zero_init', action='store_true', default=True)
    parser.add_argument('--bf16', action='store_true', default=True)
    parser.add_argument('--pure_bf16', action='store_true', default=False,
                        help='Use pure bf16 training instead of mixed precision')
    ## Train loss function

    parser.add_argument('--infonce', action="store_true", default=True)
    parser.add_argument('--infonce_temp', type=float, default=0.02)
    parser.add_argument('--learnable_temp', action='store_true', default=False)
    parser.add_argument('--muon', action='store_true', default=True) #moe要改成False，方便调整学习率
    parser.add_argument('--muon_lr', default=0.02, type=float)
    parser.add_argument('--muon_momentum', default=0.95, type=float)

    parser.add_argument('--interest_k', type=int, default=1, help='Number of interests for multi-interest mechanism')
    parser.add_argument('--use_multi_interest', action='store_true', default=False)
    #
    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=24, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    # MyDataParallel options
    parser.add_argument('--use_my_dataparallel', action='store_true', default=True,
                        help='Use MyDataParallel for multi-GPU training without torch.dist')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                        help='GPU IDs to use for MyDataParallel (e.g., --gpu_ids 0 1 2 3)')
    parser.add_argument('--norm_first', action='store_true', default=True)
    parser.add_argument('--rope', action='store_true', default=False)#moe需要改为True

    parser.add_argument('--mm_emb_gate', action='store_true', default=False)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--random_perturbation', action='store_true', default=False)
    parser.add_argument('--random_perturbation_value', default=5e-3, type=float)

    parser.add_argument('--hash_emb_size', default=256, type=int)
    parser.add_argument('--timestamp_bucket_emb_size', default=128, type=int,
                        help='Embedding size for timestamp_bucket features')
    parser.add_argument('--infer_logq', action="store_true", default=False)

    # hstu
    parser.add_argument('--use_hstu', action='store_true', default=True, help='Use HSTU blocks instead of standard Transformer blocks.')
    parser.add_argument('--hstu_rope', action='store_true', default=False,
                        help='Use RoPE in HSTU blocks.')

    parser.add_argument('--rms_norm', action='store_true', default=False)
    parser.add_argument('--dnn_hidden_units', default=4, type=int)  # 设置模型的dnn是多少倍hidden units
    parser.add_argument('--feed_forward_hidden_units', default=2, type=int)  # 设置模型Transformer的隐藏层维度是多少倍hidden units

    # dataset
    # Dataset Construction
    parser.add_argument('--base_user_sparse', default=['103', '104', '105', '109'], type=str, nargs='+')
    parser.add_argument('--base_item_sparse',
                        default=['100', '117', '118', '101', '102', '119', '120', '114', '112', '121', '115',
                                 '122', '116', ], type=str, nargs='+')  # base的特征都是Baseline模型自带的特征, 通过设置, 可以很方便的去除不需要的特征
    parser.add_argument('--base_user_array', default=['106', '107', '108', '110'], type=str, nargs='+')

    parser.add_argument('--user_sparse', default=None, type=str, nargs='+')  # 非base特征都是新增加的特征
    # 'exposure_start_year', 'exposure_start_month', 'exposure_start_day','exposure_end_year', 'exposure_end_month', 'exposure_end_day',
    # "sid"
    parser.add_argument('--item_sparse', default=['exposure_start_year', 'exposure_start_month', 'exposure_start_day',
                                                  'exposure_end_year', 'exposure_end_month', 'exposure_end_day'],
                        type=str, nargs='+')  # 非base特征都是新增加的特征
    parser.add_argument('--user_array', default=None, type=str, nargs='+')  # 非base特征都是新增加的特征
    parser.add_argument('--item_array', default=None, type=str, nargs='+')  # 非base特征都是新增加的特征
    parser.add_argument('--user_continual', default=None, type=str, nargs='+')  # 非base特征都是新增加的特征
    parser.add_argument('--item_continual', default=None, type=str, nargs='+')  # 非base特征都是新增加的特征

    # timediff: 'time_diff_day', 'time_diff_hour', 'time_diff_minute'
    # 'next_action_type'
    # 'timestamp_bucket_4096', 'timestamp_bucket_2048', 'timestamp_bucket_1024'
    parser.add_argument('--context_item_sparse',
                        default=['time_diff_day', 'time_diff_hour', 'time_diff_minute', 'action_type','next_action_type',
                                 'timestamp_bucket_id','hot_bucket_1000'], type=str,
                        nargs='+')
    parser.add_argument('--feature_dropout_list', default=['timestamp_bucket_id','hot_bucket_1000'], type=str,
                        nargs='+')
    parser.add_argument('--feature_dropout_rate', default=0.5, type=float)

    parser.add_argument('--bucket_sizes', type=int, nargs='+', default=[],
                        help='List of bucket sizes for timestamp bucketing features (e.g., 4096 2048 1024)')

    parser.add_argument('--user_id_exclude', action="store_true", default=True)
    parser.add_argument('--num_workers', default=32, type=int)
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--mm_sid', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # 本地调试
    parser.add_argument('--debug', action="store_true", default=False)

    parser.add_argument('--test_train_valid_process', action="store_true", default=False)
    parser.add_argument('--generate_sid', action="store_true", default=True)
    # sid
    parser.add_argument('--sid', action="store_true", default=False)
    parser.add_argument('--sid_codebook_layer', default=2, type=int)
    parser.add_argument('--sid_codebook_size', default=16384, type=int)
    parser.add_argument('--mlp_layers', default=2, type=int)

    parser.add_argument('--reward', default=False, action="store_true")
    parser.add_argument('--reward_only', default=False, action="store_true")

    # 预处理选项
    parser.add_argument('--use_preprocessing', action='store_true', default=False,
                        help='Use preprocessing to store batched data for faster training')
    parser.add_argument('--force_reprocess', action='store_true', default=False,
                        help='Force reprocessing even if preprocessed data exists')
    parser.add_argument('--train_infer_result_path', type=str, default="")
    parser.add_argument('--save_infer_model', action="store_true", default=True)

    parser.add_argument('--use_moe', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', default=64, type=int)
    parser.add_argument('--moe_top_k', default=3, type=int)
    parser.add_argument('--moe_intermediate_size', default=512, type=int)
    parser.add_argument('--moe_load_balancing_alpha', default=0, type=float)
    parser.add_argument('--moe_load_balancing_update_freq', default=1, type=int)

    parser.add_argument('--moe_shared_expert_num', default=1, type=int)  # 共享专家

    # Sequence-level auxiliary loss parameters
    parser.add_argument('--moe_use_sequence_aux_loss', action='store_true', default=True,
                        help='Enable sequence-level auxiliary loss for MoE load balancing')
    parser.add_argument('--moe_sequence_aux_loss_coeff', default=0.02, type=float,
                        help='Coefficient for sequence-level auxiliary loss')

    # Dynamic aux loss adjustment based on Gini coefficient
    parser.add_argument('--moe_dynamic_aux_loss', action='store_true', default=False,
                        help='Enable dynamic adjustment of aux loss coefficient based on Gini coefficient')
    parser.add_argument('--moe_aux_loss_adjust_rate', default=0.0001, type=float,
                        help='Adjustment rate for aux loss coefficient')
    parser.add_argument('--moe_gini_target_min', default=0.09, type=float,
                        help='Target minimum Gini coefficient for expert load balance')
    parser.add_argument('--moe_gini_target_max', default=0.31, type=float,
                        help='Target maximum Gini coefficient for expert load balance')
    parser.add_argument('--moe_dynamic_aux_loss_start_step', default=700, type=int,
                        help='Epoch to start dynamic aux loss adjustment')

    # ====== 新增：SID Beam Search 生成模式 (用于 train_infer.py) ======
    parser.add_argument('--beam_search_generate', action="store_true", default=False,
                        help='(train_infer.py) 启用 SID beam search 批量生成模式')
    parser.add_argument('--beam_search_top_k', default=256, type=int,
                        help='(train_infer.py) Beam search 模式下要生成的候选数量 (top_k_2)')
    parser.add_argument('--beam_search_beam_size', default=20, type=int,
                        help='(train_infer.py) Beam search 模式下的 beam size (top_k)')
    # ==========================================================
    
    # ====== 新增：SID Resort 模式参数 ======
    parser.add_argument('--sid_resort', action='store_true', default=False,
                        help='启用SID resort模式，对SID文件中的item进行相似度排序')
    parser.add_argument('--sid_path', type=str, default=None,
                        help='SID文件路径，用于sid_resort模式')
    parser.add_argument('--sid_resort_topk', type=int, nargs='+', default=[32, 128, 256],
                        help='SID resort模式下的topk列表，例如 [32, 128, 256]')
    # ==========================================================


    args = parser.parse_args()

    args.batch_size = args.single_device_batch_size * torch.cuda.device_count()
    set_seed(args.seed)
    set_path(args)
    return args


def evaluation(args, model, valid_loader, writer, epoch, global_step, dataset, device):
    model.eval()
    # 指标聚合变量
    metrics_sum = {}
    metrics_count = 0

    with torch.no_grad():
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):

            if args.test_train_valid_process and step > 10:
                break
            seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, action_type, sid, pos_log_p, ranking_loss_mask = batch
            seq, pos, sid = seq.to(args.device), pos.to(args.device), sid.to(args.device)
            pos_log_p = pos_log_p.to(args.device)
            next_action_type = next_action_type.to(args.device)
            ranking_loss_mask = ranking_loss_mask.to(args.device)
            # 使用forward_infer进行推理，包含指标计算
            if args.use_my_dataparallel and hasattr(model, 'replicas'):
                # MyDataParallel模式：使用forward_infer
                batch_metrics = model(
                    seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, sid,
                    pos_log_p, ranking_loss_mask,
                    args, dataset
                )
            else:
                # 单GPU模式：使用forward_infer
                batch_metrics = model.forward_infer(
                    seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, sid,
                    pos_log_p, ranking_loss_mask,
                    args, dataset
                )

            # 损失计算已移到模型内部，这里只需要聚合指标

            # 2. --- 聚合从模型返回的指标 (已在DataParallel层面聚合) ---
            if batch_metrics:  # 指标已经在DataParallel层面聚合过了
                # 累加指标
                for key, value in batch_metrics.items():
                    if key not in metrics_sum:
                        metrics_sum[key] = 0.0
                    metrics_sum[key] += value
                metrics_count += 1

    # 4. --- 汇总并计算最终指标 ---
    # 计算聚合指标的平均值
    final_metrics = {}
    if metrics_count > 0:
        for key, value in metrics_sum.items():
            final_metrics[key] = value / metrics_count

    # 输出指标结果
    if 'hr10' in final_metrics:
        hr10 = final_metrics['hr10']
        ndcg10 = final_metrics['ndcg10']
        score = final_metrics['score']
        print(f"Epoch {epoch} Validation Results: HR@10 = {hr10:.8f}, NDCG@10 = {ndcg10:.8f}, score = {score:.8f}")
        writer.add_scalar('infoMetrics/score@10_valid', score, global_step)
        writer.add_scalar('infoMetrics/HR@10_valid', hr10, global_step)
        writer.add_scalar('infoMetrics/NDCG@10_valid', ndcg10, global_step)

    if 'hr10_last' in final_metrics:
        hr10_last = final_metrics['hr10_last']
        ndcg10_last = final_metrics['ndcg10_last']
        score_last = final_metrics['score_last']
        print(
            f"Epoch {epoch} Validation Results: HR@10_last = {hr10_last:.8f}, NDCG@10_last = {ndcg10_last:.8f}, score_last = {score_last:.8f}")
        writer.add_scalar('valMetrics/score@10_last_valid', score_last, global_step)
        writer.add_scalar('valMetrics/HR@10_last_valid', hr10_last, global_step)
        writer.add_scalar('valMetrics/NDCG@10_last_valid', ndcg10_last, global_step)

    # metrics["sid_hr"] = sid_hr
    #             metrics["score_diff"] = score_diff
    #             metrics["hr_diff"] = hr_diff
    #             metrics["ndcg_diff"] = ndcg_diff
    if 'sid_hr' in final_metrics:
        hr10_sid = final_metrics['hr_diff']
        ndcg10_sid = final_metrics['ndcg_diff']
        score_sid = final_metrics['score_diff']
        sid_hr = final_metrics['sid_hr']
        writer.add_scalar('SID/Score_diff_val', score_sid, global_step)
        writer.add_scalar('SID/HR_diff_val', hr10_sid, global_step)
        writer.add_scalar('SID/NDCG_diff_val', ndcg10_sid, global_step)
        writer.add_scalar('SID/sid_hr', sid_hr, global_step)

    # ===============================================================
    # MODIFICATION START: 分别输出 CTR 和 CVR 的 AUC 和 Reward 指标
    # ===============================================================
    if 'loss_MLP_AUC/ctr_train' in final_metrics:
        mlp_auc_ctr = final_metrics['loss_MLP_AUC/ctr_train']
        ann_auc_ctr = final_metrics.get('loss_ANN_AUC/ctr_train', 0.5)
        print(f"Epoch {epoch} AUC (CTR): MLP={mlp_auc_ctr:.4f}, ANN={ann_auc_ctr:.4f}")
        writer.add_scalar('AUC/MLP/ctr_valid', mlp_auc_ctr, global_step)
        writer.add_scalar('AUC/ANN/ctr_valid', ann_auc_ctr, global_step)

    if 'loss_MLP_AUC/cvr_train' in final_metrics:
        mlp_auc_cvr = final_metrics['loss_MLP_AUC/cvr_train']
        ann_auc_cvr = final_metrics.get('loss_ANN_AUC/cvr_train', 0.5)
        print(f"Epoch {epoch} AUC (CVR): MLP={mlp_auc_cvr:.4f}, ANN={ann_auc_cvr:.4f}")
        writer.add_scalar('AUC/MLP/cvr_valid', mlp_auc_cvr, global_step)
        writer.add_scalar('AUC/ANN/cvr_valid', ann_auc_cvr, global_step)

    if 'reward_hr_ctr' in final_metrics:
        reward_hr_ctr = final_metrics.get('reward_hr_ctr', 0.0)
        reward_score_ctr = final_metrics.get('reward_score_ctr', 0.0)
        reward_ndcg_ctr = final_metrics.get('reward_ndcg_ctr', 0.0)
        print(
            f"Epoch {epoch} Reward Model (CTR) Validation:NDCG_diff@10={reward_ndcg_ctr}, HR_diff@10={reward_hr_ctr:.4f}, Score_diff={reward_score_ctr:.4f}")
        writer.add_scalar('Reward/RewardScore_CTR/valid', reward_score_ctr, global_step)
        writer.add_scalar('Reward/RewardHR_CTR/valid', reward_hr_ctr, global_step)
        writer.add_scalar('Reward/RewardNDCG_CTR/valid', reward_ndcg_ctr, global_step)

    if 'reward_hr_cvr' in final_metrics:
        reward_hr_cvr = final_metrics.get('reward_hr_cvr', 0.0)
        reward_score_cvr = final_metrics.get('reward_score_cvr', 0.0)
        reward_ndcg_cvr = final_metrics.get('reward_ndcg_cvr', 0.0)
        print(
            f"Epoch {epoch} Reward Model (CVR) Validation:NDCG_diff@10={reward_ndcg_cvr}, HR_diff@10={reward_hr_cvr:.4f}, Score_diff={reward_score_cvr:.4f}")
        writer.add_scalar('Reward/RewardScore_CVR/valid', reward_score_cvr, global_step)
        writer.add_scalar('Reward/RewardHR_CVR/valid', reward_hr_cvr, global_step)
        writer.add_scalar('Reward/RewardNDCG_CVR/valid', reward_ndcg_cvr, global_step)

    # ===============================================================
    # MODIFICATION END
    # ===============================================================

    # 记录损失指标
    if 'loss_total' in final_metrics:
        valid_loss = final_metrics['loss_total']
        print(f"Epoch {epoch} Validation Loss: {valid_loss:.6f}")
        writer.add_scalar('Loss/valid', valid_loss, global_step)

        if 'loss_infonce' in final_metrics:
            writer.add_scalar('InfoNCE/valid', final_metrics['loss_infonce'], global_step)
        if 'loss_sid1' in final_metrics:
            writer.add_scalar('Sid1Loss/valid', final_metrics['loss_sid1'], global_step)
        if 'loss_sid2' in final_metrics:
            writer.add_scalar('Sid2Loss/valid', final_metrics['loss_sid2'], global_step)

        # ===============================================================
        # MODIFICATION START: 分别记录 CTR 和 CVR 的 Reward Loss
        # ===============================================================
        if 'loss_reward/ctr' in final_metrics:
            writer.add_scalar('Reward/ctr_valid', final_metrics['loss_reward/ctr'], global_step)
        if 'loss_reward/cvr' in final_metrics:
            writer.add_scalar('Reward/cvr_valid', final_metrics['loss_reward/cvr'], global_step)
        # ===============================================================
        # MODIFICATION END
        # ===============================================================
    else:
        valid_loss = 0.0

    # 返回主要指标用于模型保存
    score = final_metrics.get('score', 0.0)
    return valid_loss, score


def train(args, run_name="default_run"):
    Path(args.log_path).mkdir(parents=True, exist_ok=True)
    Path(args.tb_path).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(args.log_path, 'train.log'), 'w')

    if multiprocessing.current_process().name == 'MainProcess':
        writer = SummaryWriter(args.tb_path)
        print("✓ TensorBoard writer initialized in main process")
    else:
        writer = None
        print("✓ Skipping TensorBoard writer in subprocess")
    # args.tb_writer = writer

    # global dataset
    data_path = args.data_path
    # 物品曝光开始和结束时间以及平均曝光日的点击转化和曝光，历史全部曝光点击转化
    # exposure_data_path = Path(args.user_cache_path) / 'item_exposure' / 'item_exposure_data.pkl'
    # if exposure_data_path.exists():
    #     print("Loading item action analysis data...")
    #     with open(exposure_data_path, 'rb') as f:
    #         # 将数据列表转换为 item_id -> item_data 的字典，以实现 O(1) 快速查找
    #         action_data_list = pickle.load(f)
    #         action_data = {item['item_id']: item for item in action_data_list}
    #     print("Item action analysis data loaded successfully.")
    # 物品曝光开始和结束时间以及平均曝光时长

    dataset = MyDataset(data_path, args)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.99, 0.01])
    train_dataset = dataset
    # 根据参数决定是否使用预处理
    if args.use_preprocessing:
        # 检查预处理数据目录
        batch_data_dir = Path(args.log_path) / "batch_data"
        metadata_file = batch_data_dir / "metadata.pkl"

        # 创建预处理数据目录（如果不存在）
        batch_data_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否有任何预处理文件存在
        existing_batches = list(batch_data_dir.glob("batch_*.pkl"))

        if not existing_batches and not metadata_file.exists():
            print("警告: 预处理数据不存在，将使用动态预处理模式")
            print("训练过程中将等待预处理文件生成...")
            print("请确保 preprocess_batch.py 正在运行")
        elif metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    existing_metadata = pickle.load(f)

                # 检查预处理数据是否与当前配置匹配
                if (existing_metadata.get('batch_size') == args.batch_size and
                        existing_metadata.get('maxlen') == args.maxlen):
                    print("找到匹配的预处理数据，将使用预处理模式")
                else:
                    print("警告: 预处理数据配置不匹配，将使用动态模式")
                    print(f"期望: batch_size={args.batch_size}, maxlen={args.maxlen}")
                    print(
                        f"实际: batch_size={existing_metadata.get('batch_size')}, maxlen={existing_metadata.get('maxlen')}")
            except Exception as e:
                print(f"检查预处理数据时发生错误: {e}")
                print("将使用动态预处理模式")
        else:
            print(f"发现 {len(existing_batches)} 个预处理文件，将使用动态预处理模式")

        # 计算真实的dataset长度
        print("计算真实的dataset长度...")
        temp_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            collate_fn=dataset.collate_fn, persistent_workers=False, pin_memory=False
        )
        real_max_batches = len(temp_loader)
        print(f"真实dataset长度: {real_max_batches} 个批次")

        # 使用动态预处理数据加载器，传入真实的max_batches
        from utils import DynamicPreprocessedDataLoader
        train_loader = DynamicPreprocessedDataLoader(
            batch_data_dir, num_workers=args.num_workers, shuffle=True,
            max_batches=real_max_batches, wait_timeout=300, batch_size=args.batch_size, drop_last=True
        )
        print(f"使用动态预处理数据加载器，真实长度: {real_max_batches} 个批次")
    else:
        # 使用原始DataLoader
        print("使用原始DataLoader（未启用预处理）")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            collate_fn=dataset.collate_fn, prefetch_factor=8, persistent_workers=False, pin_memory=False
        )

    # 验证集仍使用原始方式（因为验证集较小且不频繁使用）
    # valid_loader = DataLoader(
    #     valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    #     collate_fn=dataset.collate_fn, prefetch_factor=8, persistent_workers=False, pin_memory=False
    # )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 在MyDataParallel模式下强制不使用稀疏embedding
    if args.use_my_dataparallel:
        if args.sparse_embedding:
            print("Warning: Sparse embedding is not supported in MyDataParallel mode. Forcing dense embedding.")
            args.sparse_embedding = False

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)

    # Convert to pure bf16 if requested
    if args.pure_bf16:
        print("Converting model to pure bf16...")
        model = model.to(dtype=torch.bfloat16)
        print("✓ Model converted to pure bf16")
        # Disable mixed precision when using pure bf16
        print("Ensuring MoE load balancing biases remain float32...")
        # 检查模型是否启用了 MoE 并且包含了 HSTU 层
        if model.use_moe and hasattr(model, 'moe_blocks'):
            # 遍历每一个 HSTU block，因为每个 MoE 层都有自己的负载均衡策略
            for block in model.moe_blocks:
                # 确保 block 是 MoE 版本的 HSTUBlock
                if hasattr(block, 'moe_mlp'):
                    # 调用我们之前定义的修复方法
                    block.moe_mlp.gate.load_balancing.ensure_float32_biases()

            print("MoE biases have been successfully set to float32.")
        args.bf16 = False
        print("✓ Mixed precision disabled for pure bf16 mode")
    elif args.bf16:
        print("Using mixed precision training with bf16")

    # 0初始化
    for name, param in model.named_parameters():
        try:
            if args.embedding_zero_init and ("item_emb" in name):
                torch.nn.init.zeros_(param.data)
            else:
                torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    if args.reward_only:
        print(f"Set reward only!")
        args.num_epochs = 1

        if not args.debug:
            try:
                ckpt_path = _get_ckpt_path(args.train_infer_result_path)
            except Exception:
                # 回退到默认目录: {USER_CACHE_PATH}/train_infer/relu1016/*.pth
                args.train_infer_result_path = 'relu1016'
                ckpt_path = _get_ckpt_path(args.train_infer_result_path)
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda:0"))
            model.load_state_dict(checkpoint, strict=False)

    # Wrap model with MyDataParallel if requested
    if args.use_my_dataparallel:
        if args.gpu_ids is None:
            # Use all available GPUs
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = args.gpu_ids

        if len(gpu_ids) > 1:
            print(f"Using MyDataParallel with GPUs: {gpu_ids}")
            model = create_my_dataparallel(model, gpu_ids)
        else:
            model = model.to("cuda")
            print("Only one GPU specified, using single GPU training")
    else:
        print("Using single GPU training")

    epoch_start_idx = 1

    # Create optimizers
    if args.use_my_dataparallel and hasattr(model, 'replicas'):
        # 使用密集embedding的MyDataParallelOptimizer（稀疏embedding已在模型初始化前被强制禁用）
        # optimizer = create_my_dataparallel_optimizer(model, torch.optim.SGD, args=args, lr=args.lr, weight_decay=args.weight_decay)
        optimizer = create_my_dataparallel_optimizer(model, torch.optim.AdamW, args=args, betas=(0.9, 0.98),
                                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Regular single GPU optimization
        if not args.sparse_embedding:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                                          weight_decay=args.weight_decay)
        else:
            sparse_params = []
            dense_params = []
            for name, param in model.named_parameters():
                # 假设只有 embedding.weight 是稀疏的
                if 'emb' in name and 'transform' not in name:
                    sparse_params.append(param)
                else:
                    dense_params.append(param)

            print(f"Creating sparse optimizer for {len(sparse_params)} parameters")
            print(f"Creating dense optimizer for {len(dense_params)} parameters")

            optimizer_sparse = torch.optim.SparseAdam(sparse_params, lr=args.lr, betas=(0.9, 0.98))
            optimizer_dense = torch.optim.AdamW(dense_params, lr=args.lr, betas=(0.9, 0.98),
                                                weight_decay=args.weight_decay)

            # 打印参数信息用于调试
            for i, param in enumerate(sparse_params):
                print(f"Sparse param {i}: {param.shape}, device: {param.device}, dtype: {param.dtype}")
                if param.is_sparse:
                    print(f"  - Is sparse tensor")
            for i, param in enumerate(dense_params[:3]):  # 只打印前3个
                print(f"Dense param {i}: {param.shape}, device: {param.device}, dtype: {param.dtype}")

    scheduler = None
    scheduler_dense = None
    scheduler_sparse = None
    schedulers_dp = []
    if args.use_cosine_annealing:
        # 预热步数：总训练步数的10%，至少1步
        total_steps = args.num_epochs * len(train_loader)
        warmup_steps = max(1, int(0.1 * total_steps))
        # 学习率预热调度器
        if args.use_my_dataparallel and hasattr(model, 'replicas'):
            # For MyDataParallelOptimizer, build schedulers for all underlying optimizers
            if hasattr(optimizer, 'get_all_optimizers'):
                for opt in optimizer.get_all_optimizers():
                    warm = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
                    main = CosineAnnealingLR(opt, T_max=args.num_epochs * len(train_loader) - warmup_steps,
                                             eta_min=args.lr_eta_min)
                    schedulers_dp.append(SequentialLR(opt, schedulers=[warm, main], milestones=[warmup_steps]))
            else:
                warmup_scheduler = LinearLR(optimizer.optimizer, start_factor=1e-8, end_factor=1.0,
                                            total_iters=warmup_steps)
                main_scheduler = CosineAnnealingLR(optimizer.optimizer,
                                                   T_max=args.num_epochs * len(train_loader) - warmup_steps,
                                                   eta_min=args.lr_eta_min)
                scheduler = SequentialLR(optimizer.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                         milestones=[warmup_steps])
        elif not args.sparse_embedding:
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)

            # 余弦退火调度器
            main_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader) - warmup_steps,
                                               eta_min=args.lr_eta_min)
            # 将它们串联起来
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                     milestones=[warmup_steps])
        else:
            warmup_scheduler_sparse = LinearLR(
                optimizer_sparse, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
            warmup_scheduler_dense = LinearLR(optimizer_dense, start_factor=1e-8, end_factor=1.0,
                                              total_iters=warmup_steps)

            # 余弦退火调度器
            main_scheduler_sparse = CosineAnnealingLR(optimizer_sparse,
                                                      T_max=args.num_epochs * len(train_loader) - warmup_steps,
                                                      eta_min=args.lr_eta_min)
            main_scheduler_dense = CosineAnnealingLR(optimizer_dense,
                                                     T_max=args.num_epochs * len(train_loader) - warmup_steps,
                                                     eta_min=args.lr_eta_min)
            # 将它们串联起来
            scheduler_sparse = SequentialLR(optimizer_sparse,
                                            schedulers=[warmup_scheduler_sparse, main_scheduler_sparse],
                                            milestones=[warmup_steps])
            scheduler_dense = SequentialLR(optimizer_dense, schedulers=[warmup_scheduler_dense, main_scheduler_dense],
                                           milestones=[warmup_steps])

    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        run_start_time = time.time()
        model.train()

        # 为预处理数据设置epoch（确保每个epoch的数据顺序不同）
        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)

        # 初始化累积变量
        accumulated_loss = 0

        loss_sum = 0.0
        infonce_sum = 0.0
        sid1_loss_sum = 0.0
        sid2_loss_sum = 0.0
        mlp_ctr_loss_sum = 0.0
        mlp_cvr_loss_sum = 0.0
        moe_aux_loss_sum = 0.0
        mlp_auc_ctr_sum, ann_auc_ctr_sum = 0.0, 0.0
        mlp_auc_cvr_sum, ann_auc_cvr_sum = 0.0, 0.0
        auc_ctr_count, auc_cvr_count = 0, 0
        pos_sim_sum = 0.0
        loss_count = 0
        accumulated_samples = 0

        time_start = time.time()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 记录batch开始时间
            time_end = time.time()
            print(f"Read Batch Time: {time_end - time_start}")

            if args.test_train_valid_process and step > 10:
                break

            seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, action_type, sid, pos_log_p, ranking_loss_mask = batch

            seq, pos, sid = seq.to(args.device), pos.to(args.device), sid.to(args.device)
            token_type, next_token_type = token_type.to(args.device), next_token_type.to(args.device)
            pos_log_p = pos_log_p.to(args.device)
            next_action_type = next_action_type.to(args.device)
            ranking_loss_mask = ranking_loss_mask.to(args.device)
            # 将张量移动到正确的设备
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            token_type = token_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)
            sid = sid.to(args.device)
            pos_log_p = pos_log_p.to(args.device)
            ranking_loss_mask = ranking_loss_mask.to(args.device)

            # 记录当前batch的样本数量
            current_batch_size = seq.shape[0]

            # Use autocast for mixed precision (bf16) or no autocast for pure bf16
            if len(gpu_ids) == 1:
                loss, train_log_dict = model.forward_train(
                    seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, sid,
                    pos_log_p, ranking_loss_mask,
                    args=args, dataset=dataset
                )
            else:
                loss, train_log_dict = model.forward(
                    seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat, sid,
                    pos_log_p, ranking_loss_mask,
                    args=args, dataset=dataset
                )

            # 构建log_dict，包含全局信息
            log_dict = {'global_step': global_step, 'epoch': epoch, 'time': time.time()}
            log_dict.update(train_log_dict)

            # MODIFICATION: 累加损失和指标
            loss_sum += loss.item()
            loss_count += 1

            if args.infonce and 'InfoNCE/train' in train_log_dict:
                infonce_sum += train_log_dict['InfoNCE/train']
            if args.sid:
                if 'Sid1Loss/train' in train_log_dict:
                    sid1_loss_sum += train_log_dict['Sid1Loss/train']
                if 'Sid2Loss/train' in train_log_dict:
                    sid2_loss_sum += train_log_dict['Sid2Loss/train']
            if args.reward:
                if 'MLP_Loss/train_ctr' in train_log_dict:
                    mlp_ctr_loss_sum += train_log_dict['MLP_Loss/train_ctr']
                if 'MLP_Loss/train_cvr' in train_log_dict:
                    mlp_cvr_loss_sum += train_log_dict['MLP_Loss/train_cvr']

                if 'MLP_AUC/ctr_train' in train_log_dict:
                    mlp_auc_ctr_sum += train_log_dict['MLP_AUC/ctr_train']
                    auc_ctr_count += 1
                if 'ANN_AUC/ctr_train' in train_log_dict:
                    ann_auc_ctr_sum += train_log_dict['ANN_AUC/ctr_train']

                if 'MLP_AUC/cvr_train' in train_log_dict:
                    mlp_auc_cvr_sum += train_log_dict['MLP_AUC/cvr_train']
                    auc_cvr_count += 1
                if 'ANN_AUC/cvr_train' in train_log_dict:
                    ann_auc_cvr_sum += train_log_dict['ANN_AUC/cvr_train']
            # ===============================================================
            # MODIFICATION END
            # ===============================================================

            if args.use_moe and 'MoE_AuxLoss/train' in train_log_dict:
                moe_aux_loss_sum += train_log_dict['MoE_AuxLoss/train']

            if args.learnable_temp:
                log_dict["infonce_temp"] = float(train_log_dict['infonce_temp'].detach().cpu())
                writer.add_scalar('infonce_temp/HR@10', float(train_log_dict['infonce_temp']), global_step)

            for k, v in train_log_dict.items():
                if "MoE" in k:
                    writer.add_scalar(f"MoE/{k}", v, global_step)

            print(train_log_dict)
            if 'Similarity/positive_train' in train_log_dict:
                # 累加Acc和相似度
                pos_sim_sum += train_log_dict['Similarity/positive_train']

                # 从train_log_dict中获取HR@10和NDCG@10指标
                hr10_last = train_log_dict.get('HR@10_last/train', 0.0)
                ndcg10_last = train_log_dict.get('NDCG@10_last/train', 0.0)
                score_last = train_log_dict.get('Score_last/train', 0.0)
                hr10 = train_log_dict.get('HR@10/train', 0.0)
                ndcg10 = train_log_dict.get('NDCG@10/train', 0.0)
                score = train_log_dict.get('Score/train', 0.0)

                if args.sid:
                    hr10_sid = train_log_dict.get('SID/hr_diff', 0.0)
                    ndcg10_sid = train_log_dict.get('SID/ndcg_diff', 0.0)
                    score_sid = train_log_dict.get('SID/score_diff', 0.0)
                    sid_hitrate = train_log_dict.get('SID/sid_hr', 0.0)
                if step == 0:
                    hr10_list, ndcg10_list, score_list = [], [], []
                    hr10_last_list, ndcg10_last_list, score_last_list = [], [], []
                    if args.sid:
                        hr10_sid_list, ndcg10_sid_list, score_sid_list, sid_hitrate_list = [], [], [], []
                # 加一个和真实对比
                hr10_list.append(hr10)
                ndcg10_list.append(ndcg10)
                score_list.append(score)
                hr10_last_list.append(hr10_last)
                ndcg10_last_list.append(ndcg10_last)
                score_last_list.append(score_last)
                if args.sid:
                    hr10_sid_list.append(hr10_sid)
                    ndcg10_sid_list.append(ndcg10_sid)
                    score_sid_list.append(score_sid)
                    sid_hitrate_list.append(sid_hitrate)

                # --- 原有的 HR/NDCG 指标 ---
                hr10_avg = sum(hr10_list) / len(hr10_list)
                ndcg10_avg = sum(ndcg10_list) / len(ndcg10_list)
                score_avg = sum(score_list) / len(score_list)
                hr10_last_avg = sum(hr10_last_list) / len(hr10_last_list)
                ndcg10_last_avg = sum(ndcg10_last_list) / len(ndcg10_last_list)
                score_last_avg = sum(score_last_list) / len(score_last_list)

                log_dict['infoMetrics/HR@10_train'] = hr10_avg
                log_dict['infoMetrics/NDCG@10_train'] = ndcg10_avg
                log_dict['infoMetrics/score@10_train'] = score_avg
                if multiprocessing.current_process().name == 'MainProcess':
                    writer.add_scalar('infoMetrics/HR@10', hr10_avg, global_step)
                    writer.add_scalar('infoMetrics/NDCG@10', ndcg10_avg, global_step)
                    writer.add_scalar('infoMetrics/score@10', score_avg, global_step)

                log_dict['Metrics/HR@10_last_train'] = hr10_last_avg
                log_dict['Metrics/NDCG@10_last_train'] = ndcg10_last_avg
                log_dict['Metrics/score@10_last_train'] = score_last_avg
                if multiprocessing.current_process().name == 'MainProcess':
                    writer.add_scalar('Metrics/HR@10_last', hr10_last_avg, global_step)
                    writer.add_scalar('Metrics/NDCG@10_last', ndcg10_last_avg, global_step)
                    writer.add_scalar('Metrics/score@10_last', score_last_avg, global_step)

                if args.sid:
                    hr10_sid_avg = sum(hr10_sid_list) / len(hr10_sid_list)
                    ndcg10_sid_avg = sum(ndcg10_sid_list) / len(ndcg10_sid_list)
                    score_sid_avg = sum(score_sid_list) / len(score_sid_list)
                    sid_hitrate_avg = sum(sid_hitrate_list) / len(sid_hitrate_list)

                    log_dict['SID/HR_diff'] = hr10_sid_avg
                    log_dict['SID/NDCG_diff'] = ndcg10_sid_avg
                    log_dict['SID/Score_diff'] = score_sid_avg
                    log_dict['SID/sid_hr'] = sid_hitrate_avg
                    if multiprocessing.current_process().name == 'MainProcess':
                        writer.add_scalar('SID/HR_diff', hr10_sid_avg, global_step)
                        writer.add_scalar('SID/NDCG_diff', ndcg10_sid_avg, global_step)
                        writer.add_scalar('SID/Score_diff', score_sid_avg, global_step)
                        writer.add_scalar('SID/sid_hr', sid_hitrate_avg, global_step)

                if args.reward:
                    reward_hr_ctr = train_log_dict.get('Reward/RewardHR_CTR/train', None)
                    reward_score_ctr = train_log_dict.get('Reward/RewardScore_CTR/train',
                                                          None)  # Reward/RewardNDCG_CTR/train
                    reward_ndcg_ctr = train_log_dict.get('Reward/RewardNDCG_CTR/train', None)

                    reward_hr_cvr = train_log_dict.get('Reward/RewardHR_CVR/train', None)
                    reward_score_cvr = train_log_dict.get('Reward/RewardScore_CVR/train', None)
                    reward_ndcg_cvr = train_log_dict.get('Reward/RewardNDCG_CVR/train', None)

                    if multiprocessing.current_process().name == 'MainProcess':
                        if reward_hr_ctr is not None:
                            log_dict['Reward/RewardHR_CTR/train'] = reward_hr_ctr
                            log_dict['Reward/RewardNDCG_CTR/train'] = reward_ndcg_ctr
                            log_dict['Reward/RewardScore_CTR/train'] = reward_score_ctr
                            writer.add_scalar('Reward/RewardScore_CTR/train', reward_score_ctr, global_step)
                            writer.add_scalar('Reward/RewardHR_CTR/train', reward_hr_ctr, global_step)
                            writer.add_scalar('Reward/RewardNDCG_CTR/train', reward_ndcg_ctr, global_step)

                        if reward_hr_cvr is not None:
                            log_dict['Reward/RewardHR_CVR/train'] = reward_hr_cvr
                            log_dict['Reward/RewardNDCG_CVR/train'] = reward_ndcg_cvr
                            log_dict['Reward/RewardScore_CVR/train'] = reward_score_cvr
                            writer.add_scalar('Reward/RewardHR_CVR/train', reward_hr_cvr, global_step)
                            writer.add_scalar('Reward/RewardNDCG_CVR/train', reward_ndcg_cvr, global_step)
                            writer.add_scalar('Reward/RewardScore_CVR/train', reward_score_cvr, global_step)
                # ===============================================================
                # MODIFICATION END
                # ===============================================================

                # --- 新增的平均指标写入 ---
                if multiprocessing.current_process().name == 'MainProcess':
                    writer.add_scalar('Loss/train_avg', loss_sum / loss_count, global_step)
                    if args.infonce: writer.add_scalar('InfoNCE/train_avg', infonce_sum / loss_count, global_step)
                    if args.sid:
                        writer.add_scalar('Sid1Loss/train_avg', sid1_loss_sum / loss_count, global_step)
                        writer.add_scalar('Sid2Loss/train_avg', sid2_loss_sum / loss_count, global_step)
                    if args.reward:
                        writer.add_scalar('MLP_Loss/ctr_train_avg', mlp_ctr_loss_sum / loss_count, global_step)
                        writer.add_scalar('MLP_Loss/cvr_train_avg', mlp_cvr_loss_sum / loss_count, global_step)
                        if auc_ctr_count > 0:
                            writer.add_scalar('AUC/MLP/ctr_train', mlp_auc_ctr_sum / auc_ctr_count, global_step)
                            writer.add_scalar('AUC/ANN/ctr_train', ann_auc_ctr_sum / auc_ctr_count, global_step)
                        if auc_cvr_count > 0:
                            writer.add_scalar('AUC/MLP/cvr_train', mlp_auc_cvr_sum / auc_cvr_count, global_step)
                            writer.add_scalar('AUC/ANN/cvr_train', ann_auc_cvr_sum / auc_cvr_count, global_step)
                    # ===============================================================
                    # MODIFICATION END
                    # ===============================================================

                    if args.use_moe:
                        writer.add_scalar('MoE_AuxLoss/train_avg', moe_aux_loss_sum / loss_count, global_step)

                    writer.add_scalar('Similarity/positive_train_avg', pos_sim_sum, global_step)

                # --- 清空所有列表和累加器 ---
                hr10_list.clear()
                ndcg10_list.clear()
                score_list.clear()
                hr10_last_list.clear()
                ndcg10_last_list.clear()
                score_last_list.clear()
                if args.sid:
                    hr10_sid_list.clear()
                    ndcg10_sid_list.clear()
                    score_sid_list.clear()

                # ===============================================================
                # MODIFICATION START: 清空分离后的累加器
                # ===============================================================
                loss_sum, infonce_sum, sid1_loss_sum, sid2_loss_sum = 0.0, 0.0, 0.0, 0.0
                mlp_ctr_loss_sum, mlp_cvr_loss_sum = 0.0, 0.0
                moe_aux_loss_sum = 0.0
                mlp_auc_ctr_sum, ann_auc_ctr_sum = 0.0, 0.0
                mlp_auc_cvr_sum, ann_auc_cvr_sum = 0.0, 0.0
                auc_ctr_count, auc_cvr_count = 0, 0
                pos_sim_sum = 0.0
                loss_count = 0
                # ===============================================================
                # MODIFICATION END
                # ===============================================================

            loss = loss / args.gradient_accumulation_steps

            # 只有在非MyDataParallel模式下才调用loss.backward()
            # MyDataParallel模式下，各个GPU已经各自调用了loss.backward()
            if not (args.use_my_dataparallel and hasattr(model, 'replicas')):
                loss.backward()

            accumulated_loss += loss.item() * args.gradient_accumulation_steps

            # 仅在达到累积步数时才进行更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # 注意：这里的 'Loss/train_accumulated' 是每个优化器步骤记录一次，逻辑不变
                if multiprocessing.current_process().name == 'MainProcess':
                    writer.add_scalar('Loss/train_accumulated', accumulated_loss, global_step)
                accumulated_loss = 0

                if multiprocessing.current_process().name == 'MainProcess':
                    model_grad_norms(model, writer, global_step)
                if args.clip_grad_norm > 0:
                    if not args.use_my_dataparallel:
                        if args.sparse_embedding:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(dense_params, max_norm=args.clip_grad_norm)

                if args.use_my_dataparallel and hasattr(model, 'replicas'):
                    try:
                        optimizer.step(args.gradient_accumulation_steps)
                        optimizer.zero_grad()
                    except RuntimeError as e:
                        if "CUDA error" in str(e) or "illegal memory access" in str(e):
                            print(f"CUDA memory error detected: {e}")
                            print("Clearing CUDA cache and retrying...")
                            # 清理所有GPU的内存
                            for device_id in model.device_ids:
                                with torch.cuda.device(device_id):
                                    torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            # 重试一次
                            try:
                                optimizer.step(args.gradient_accumulation_steps)
                                optimizer.zero_grad()
                            except RuntimeError as retry_e:
                                print(f"Retry failed: {retry_e}")
                                print("Consider reducing batch size or using fewer GPUs")
                                raise retry_e
                        else:
                            raise e
                elif not args.sparse_embedding:
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    # 添加内存检查，避免CUDA内存访问错误
                    try:
                        optimizer_sparse.step()
                        optimizer_dense.step()
                        optimizer_sparse.zero_grad()
                        optimizer_dense.zero_grad()
                    except RuntimeError as e:
                        if "CUDA error" in str(e) or "illegal memory access" in str(e):
                            print(f"CUDA memory error detected: {e}")
                            print("Clearing CUDA cache and retrying...")
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            # 重试一次
                            try:
                                optimizer_sparse.step()
                                optimizer_dense.step()
                                optimizer_sparse.zero_grad()
                                optimizer_dense.zero_grad()
                            except RuntimeError as retry_e:
                                print(f"Retry failed: {retry_e}")
                                print("Consider disabling sparse_embedding to avoid this issue")
                                raise retry_e
                        else:
                            raise e

                # if multiprocessing.current_process().name == 'MainProcess':
                #     model_params(model, writer, global_step)
                # 动态调整MoE aux loss系数 - 每100步更新一次
                if args.use_moe and args.moe_dynamic_aux_loss and global_step >= args.moe_dynamic_aux_loss_start_step:
                    if global_step % 100 == 0:
                        # 只更新主卡（第一个replica或单GPU）的MoE层
                        if args.use_my_dataparallel and hasattr(model, 'replicas'):
                            # MyDataParallel模式：只更新第一个replica
                            target_model = model.replicas[0]
                        else:
                            # 单GPU模式
                            target_model = model
                        if hasattr(target_model, 'moe_blocks'):
                            # 对每一层独立更新
                            for layer_idx, block in enumerate(target_model.moe_blocks):
                                if hasattr(block.moe_mlp, 'gate'):
                                    gate = block.moe_mlp.gate
                                    # 调用每一层自己的update方法
                                    gate.update_aux_loss_alpha(
                                        args.moe_gini_target_min,
                                        args.moe_gini_target_max,
                                        args.moe_aux_loss_adjust_rate
                                    )

                                    # 获取统计信息并记录到TensorBoard
                                    if multiprocessing.current_process().name == 'MainProcess':
                                        stats = gate.get_moe_statistics()
                                        if stats:
                                            writer.add_scalar(f'MoE/Layer{layer_idx}/load_gini',
                                                              stats.get('load_gini', 0.0), global_step)

                            # 在DataParallel模式下，将主卡更新的alpha参数同步到其他replica
                            if args.use_my_dataparallel and hasattr(model, 'replicas') and len(
                                    model.replicas) > 1:
                                for replica_idx in range(1, len(model.replicas)):
                                    if hasattr(model.replicas[replica_idx], 'moe_blocks'):
                                        for layer_idx, (src_block, dst_block) in enumerate(
                                                zip(model.replicas[0].moe_blocks,
                                                    model.replicas[replica_idx].moe_blocks)
                                        ):
                                            if hasattr(src_block.moe_mlp, 'gate') and hasattr(
                                                    dst_block.moe_mlp, 'gate'):
                                                with torch.no_grad():
                                                    # 复制alpha参数从主卡到其他卡
                                                    dst_block.moe_mlp.gate.alpha.copy_(
                                                        src_block.moe_mlp.gate.alpha)

                if schedulers_dp:
                    for sch in schedulers_dp:
                        sch.step()
                    if multiprocessing.current_process().name == 'MainProcess':
                        # log first group's lr from each optimizer
                        try:
                            for idx, opt in enumerate(optimizer.get_all_optimizers()):
                                writer.add_scalar(f'LR/train_opt{idx}', opt.param_groups[0]['lr'], global_step)
                        except Exception:
                            pass
                elif scheduler is not None:
                    scheduler.step()
                    if multiprocessing.current_process().name == 'MainProcess':
                        if args.use_my_dataparallel and hasattr(model, 'replicas'):
                            writer.add_scalar('LR/train', optimizer.optimizer.param_groups[0]['lr'], global_step)
                        else:
                            writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], global_step)

                if scheduler_dense is not None and scheduler_sparse is not None:
                    scheduler_dense.step()
                    scheduler_sparse.step()
                    # 学习率也是每个优化器步骤记录一次，逻辑不变
                    if multiprocessing.current_process().name == 'MainProcess':
                        writer.add_scalar('LR/train', optimizer_dense.param_groups[0]['lr'], global_step)

                accumulated_samples += current_batch_size
                log_dict['Speed/Total_Samples'] = accumulated_samples
                # Runtime minutes since training started
                runtime_minutes = (time.time() - run_start_time) / 60.0
                log_dict['Speed/runtime_minutes'] = runtime_minutes

                current_speed = accumulated_samples / (time.time() - run_start_time)

                log_dict['Speed/current_samples_per_sec'] = current_speed

                if multiprocessing.current_process().name == 'MainProcess':
                    writer.add_scalar('Speed/current_samples_per_sec', current_speed, global_step)
                    writer.add_scalar('Speed/runtime_minutes', runtime_minutes, global_step)
                    writer.add_scalar('Speed/Total_Samples', accumulated_samples, global_step)

                log_json = json.dumps(log_dict)
                log_file.write(log_json + '\n')
                log_file.flush()
                print(log_json)
            time_start = time.time()

        save_dir = Path(args.ckpt_path, f"epoch{epoch}_global_step{global_step}")
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.use_my_dataparallel and hasattr(model, 'replicas'):
            # MyDataParallel模式：保存第一个replica的state_dict
            torch.save(model.replicas[0].state_dict(), save_dir / "model.pt")
        else:
            # 单GPU模式：直接保存
            torch.save(model.state_dict(), save_dir / "model.pt")

        # AFTER EVALUATION
        # Clear CUDA cache and garbage collect before evaluation
        if args.use_my_dataparallel and hasattr(model, 'device_ids'):
            for device_id in model.device_ids:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Reset metric counters after each epoch
        try:
            if args.use_my_dataparallel and hasattr(model, 'replicas'):
                for replica in model.replicas:
                    if hasattr(replica, '_metric_counter'):
                        replica._metric_counter = 0
            else:
                if hasattr(model, '_metric_counter'):
                    model._metric_counter = 0
        except Exception:
            pass

    print(f"{run_name} Done")
    if writer:
        writer.close()
    log_file.close()

    return 0  # 不再返回验证损失


if __name__ == '__main__':
    args = get_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Starting a standard training run...")
    train(args, run_name="standard_train")