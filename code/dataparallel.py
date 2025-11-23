#!/usr/bin/env python3
"""
MyDataParallel - My DataParallel implementation without torch.dist
"""

import time
import torch
import torch.nn as nn
import copy
import threading
from typing import List, Optional, Any, Dict

from torch.cuda.amp import autocast
from torch.optim import Optimizer


class MyDataParallel(nn.Module):
    """
    A MyDataParallel implementation that doesn't rely on torch.dist.
    This is useful when TCP communication is disabled on the server.
    """

    def __init__(self, module: nn.Module, device_ids: Optional[List[int]] = None):
        """
        Initialize MyDataParallel.

        Args:
            module: The module to parallelize
            device_ids: List of GPU device IDs to use
        """
        super().__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if len(device_ids) < 1:
            raise ValueError("At least one device ID is required")

        self.device_ids = device_ids
        self.primary_device = device_ids[0]

        # Replicate the module to all devices
        self.replicas = nn.ModuleList()
        for device_id in device_ids:
            replica = copy.deepcopy(module)
            replica = replica.to(f'cuda:{device_id}')

            # Update device attribute if the model has one (for BaselineModel)
            if hasattr(replica, 'dev'):
                replica.dev = f'cuda:{device_id}'

            self.replicas.append(replica)

        # Ensure all replicas have the same parameters initially
        self._sync_parameters()
        self.module = self.replicas[0]

        # Update device attribute for primary module
        if hasattr(self.module, 'dev'):
            self.module.dev = f'cuda:{self.primary_device}'

        print(f"MyDataParallel initialized with {len(device_ids)} GPUs: {device_ids}")

        # Initialize step tracking attributes
        self.step_count = 0
        self.sync_frequency = 10  # Sync parameters every 10 steps

        # Cache parameters for efficient gradient sync
        self._cache_parameters()

        # 性能优化：预分配线程池和结果缓存
        self._init_performance_optimizations()

    def _cache_parameters(self):
        """Cache parameters for efficient gradient synchronization"""
        self._cached_primary_params = list(self.replicas[0].parameters())
        self._cached_replica_params = []
        for replica in self.replicas[1:]:
            self._cached_replica_params.append(list(replica.parameters()))

    def _init_performance_optimizations(self):
        """初始化性能优化相关的缓存和线程池"""
        # 预分配结果缓存，避免每次重新分配
        self._results_cache = [None] * len(self.replicas)
        self._threads_cache = [None] * len(self.replicas)

        # 批处理优化参数
        self._min_batch_size_per_gpu = 4  # 降低最小批次大小，提高GPU利用率
        self._enable_async_transfer = True  # 启用异步数据传输

        # 内存池优化
        self._memory_pool = {}
        for device_id in self.device_ids:
            self._memory_pool[device_id] = []

    def _async_transfer_batch(self, batch, target_device):
        """异步传输批次数据到目标设备"""
        if not self._enable_async_transfer:
            # 如果异步传输被禁用，使用同步传输
            return tuple(
                item.to(f'cuda:{target_device}') if isinstance(item, torch.Tensor)
                else item for item in batch
            )

        # 异步传输优化
        device_batch = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                # 使用non_blocking=True进行异步传输
                device_batch.append(item.to(f'cuda:{target_device}', non_blocking=True))
            elif isinstance(item, dict):
                # 处理字典类型
                device_dict = {}
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        device_dict[key] = value.to(f'cuda:{target_device}', non_blocking=True)
                    else:
                        device_dict[key] = value
                device_batch.append(device_dict)
            else:
                device_batch.append(item)

        return tuple(device_batch)

    def _all_devices_safe_check(self):
        """Wait for all devices to complete their operations with error handling"""
        cuda_errors = []
        for device_id in self.device_ids:
            try:
                torch.cuda.synchronize(device_id)
            except RuntimeError as e:
                if "CUDA" in str(e) or "illegal memory access" in str(e):
                    cuda_errors.append((device_id, e))
                    print(f"CUDA synchronization error on device {device_id}: {e}")
                else:
                    raise e

        # H20兼容性修复：如果有CUDA错误，尝试清理但不继续执行
        if cuda_errors:
            print(f"Detected {len(cuda_errors)} CUDA errors during synchronization")
            # 不尝试清理内存，直接抛出第一个错误
            raise cuda_errors[0][1]

    def __getattr__(self, name):
        """
        Forward attribute access to the primary module.
        This allows access to methods like mlp_bce_loss, etc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If not found in MyDataParallel, try the primary module
            return getattr(self.module, name)

    def parameters(self):
        """
        Return parameters from the primary module only.
        This ensures gradient clipping works correctly.
        """
        return self.module.parameters()

    def named_parameters(self):
        """
        Return named parameters from the primary module only.
        """
        return self.module.named_parameters()

    def _get_dense_parameters(self):
        """
        Get only dense parameters (excluding sparse embeddings) for gradient clipping.
        This avoids the SparseCUDA backend issue with gradient clipping.
        """
        dense_params = []
        for param in self.module.parameters():
            # Only include dense parameters (not sparse embeddings)
            if param.is_sparse:
                continue
            if param.grad is not None and param.grad.is_sparse:
                continue
            dense_params.append(param)
        return dense_params

    def _sync_parameters(self):
        """Synchronize parameters across all replicas."""
        if len(self.replicas) <= 1:
            return

        # Get parameters from the first replica
        primary_params = list(self.replicas[0].parameters())

        # Copy parameters to all other replicas
        for replica in self.replicas[1:]:
            replica_params = list(replica.parameters())
            for primary_param, replica_param in zip(primary_params, replica_params):
                replica_param.data.copy_(primary_param.data)

    def _parallel_forward(self, args_kwargs_list):
        """并行执行forward计算 - 优化版本"""
        if len(args_kwargs_list) != len(self.replicas):
            raise ValueError("Input list length must match number of replicas")

        # 使用预分配的结果缓存
        results = self._results_cache
        threads = []

        def worker(device_idx, replica, args, kwargs):
            try:
                if args is not None and kwargs is not None:
                    # H20兼容性修复：显式设置当前CUDA设备
                    device_id = self.device_ids[device_idx]
                    torch.cuda.set_device(device_id)

                    with torch.cuda.device(device_id):
                        # 使用*args和**kwargs的方式调用
                        output = replica(*args, **kwargs)
                        results[device_idx] = output
                        # H20兼容性修复：确保GPU操作完成
                        torch.cuda.synchronize(device_id)
            except Exception as e:
                results[device_idx] = e

        # 启动并行线程
        for i, (replica, args_kwargs) in enumerate(zip(self.replicas, args_kwargs_list)):
            if args_kwargs is not None:
                args, kwargs = args_kwargs
                thread = threading.Thread(target=worker, args=(i, replica, args, kwargs))
                thread.start()
                threads.append(thread)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 只在最后进行一次同步，而不是每个worker都同步
        self._all_devices_safe_check()

        # 检查错误
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise result

        return results

    def _parallel_forward_train(self, args_kwargs_list):
        """并行执行forward_train计算（训练模式）- 支持多卡负样本交换"""
        if len(args_kwargs_list) != len(self.replicas):
            raise ValueError("Input list length must match number of replicas")

        # 使用预分配的结果缓存
        results = self._results_cache
        threads = []

        def worker_train(device_idx, replica, args, kwargs):
            try:
                if args is not None and kwargs is not None:
                    # H20兼容性修复：显式设置当前CUDA设备
                    device_id = self.device_ids[device_idx]
                    torch.cuda.set_device(device_id)

                    with torch.cuda.device(device_id):
                        # 调用forward_train方法
                        with autocast(dtype=torch.bfloat16, enabled=True):
                            loss, log_dict = replica.forward_train(*args, **kwargs)
                            # 在每个GPU上调用loss.backward()来累积梯度
                            loss.backward()
                            # 保存结果，用于返回第一个GPU的结果
                            results[device_idx] = (loss, log_dict)
                            # H20兼容性修复：强制同步确保所有操作完成
                            torch.cuda.synchronize(device_id)
            except RuntimeError as e:
                # H20兼容性修复：捕获CUDA错误但不尝试清理（可能导致二次错误）
                print(f"Runtime error in GPU {self.device_ids[device_idx]} during training: {e}")
                results[device_idx] = e
            except Exception as e:
                print(f"Unexpected error in GPU {self.device_ids[device_idx]} during training: {e}")
                results[device_idx] = e

        # 启动并行线程
        for i, (replica, args_kwargs) in enumerate(zip(self.replicas, args_kwargs_list)):
            if args_kwargs is not None:
                args, kwargs = args_kwargs
                thread = threading.Thread(target=worker_train, args=(i, replica, args, kwargs))
                thread.start()
                threads.append(thread)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 检查错误
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise result

        # 批量同步所有GPU，而不是逐个同步
        self._all_devices_safe_check()

        # 返回第一个GPU（cuda:0）的结果
        if results[0] is not None:
            return results[0]
        else:
            return None, {}

    def _parallel_forward_infer(self, args_kwargs_list):
        """并行执行forward_infer计算（推理模式）- 优化版本"""
        if len(args_kwargs_list) != len(self.replicas):
            raise ValueError("Input list length must match number of replicas")

        # 使用预分配的结果缓存
        results = self._results_cache
        threads = []

        def worker_infer(device_idx, replica, args, kwargs):
            try:
                if args is not None and kwargs is not None:
                    # H20兼容性修复：显式设置当前CUDA设备
                    device_id = self.device_ids[device_idx]
                    torch.cuda.set_device(device_id)

                    with torch.cuda.device(device_id):
                        # 调用forward_infer方法
                        output = replica.forward_infer(*args, **kwargs)
                        results[device_idx] = output
                        # H20兼容性修复：添加同步确保操作完成
                        torch.cuda.synchronize(device_id)
            except Exception as e:
                results[device_idx] = e

        # 启动并行线程
        for i, (replica, args_kwargs) in enumerate(zip(self.replicas, args_kwargs_list)):
            if args_kwargs is not None:
                args, kwargs = args_kwargs
                thread = threading.Thread(target=worker_infer, args=(i, replica, args, kwargs))
                thread.start()
                threads.append(thread)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 检查错误
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise result

        # 批量同步所有GPU
        self._all_devices_safe_check()

        # 处理结果并合并
        outputs = []
        for result in results:
            if result is not None:
                outputs.append(result)

        if not outputs:
            return None

        # Concatenate outputs
        if len(outputs) == 1:
            return outputs[0]

        # Handle tuple outputs
        if isinstance(outputs[0], tuple):
            result = []
            for i in range(len(outputs[0])):
                if outputs[0][i] is not None:
                    # 检查是否是metrics字典（最后一个元素）
                    if i == len(outputs[0]) - 1 and isinstance(outputs[0][i], dict):
                        # 聚合指标：计算所有GPU上指标的平均值
                        aggregated_metrics = {}
                        for output in outputs:
                            if output[i] is not None and isinstance(output[i], dict):
                                for key, value in output[i].items():
                                    if key not in aggregated_metrics:
                                        aggregated_metrics[key] = []
                                    aggregated_metrics[key].append(value)

                        # 计算平均值
                        for key, values in aggregated_metrics.items():
                            if values:  # 确保有值
                                aggregated_metrics[key] = sum(values) / len(values)

                        result.append(aggregated_metrics)
                    else:
                        # Concatenate tensors along batch dimension
                        concat_tensors = []
                        for output in outputs:
                            if output[i] is not None:
                                # 保持原始数据类型
                                tensor = output[i]
                                concat_tensors.append(
                                    tensor.to(device=f'cuda:{self.primary_device}', dtype=tensor.dtype))
                        if concat_tensors:
                            result.append(torch.cat(concat_tensors, dim=0))
                        else:
                            result.append(None)
                else:
                    result.append(None)
            return tuple(result)
        else:
            # Single output (could be tensor or dict)
            if isinstance(outputs[0], dict):
                # Handle dict output (e.g., metrics from forward_infer)
                aggregated_metrics = {}
                for output in outputs:
                    if output is not None and isinstance(output, dict):
                        for key, value in output.items():
                            if key not in aggregated_metrics:
                                aggregated_metrics[key] = []
                            aggregated_metrics[key].append(value)

                # 计算平均值
                for key, values in aggregated_metrics.items():
                    if values:  # 确保有值
                        aggregated_metrics[key] = sum(values) / len(values)

                return aggregated_metrics
            else:
                # Single tensor output
                concat_tensors = []
                for output in outputs:
                    # 保持原始数据类型
                    concat_tensors.append(output.to(device=f'cuda:{self.primary_device}', dtype=output.dtype))
                return torch.cat(concat_tensors, dim=0)

    def _distribute_batch(self, batch: Any) -> List[Any]:
        """
        Distribute batch across multiple GPUs.

        Args:
            batch: Input batch (tuple or list of tensors)

        Returns:
            List of batches, one for each GPU
        """
        if not isinstance(batch, (tuple, list)):
            raise ValueError("Batch must be a tuple or list")

        # Find the first tensor to determine batch size
        batch_size = None
        for item in batch:
            if isinstance(item, torch.Tensor) and item.dim() > 0:
                batch_size = item.size(0)
                break

        if batch_size is None:
            # No tensors found, return original batch for all GPUs
            return [batch] * len(self.device_ids)

        # 优化批处理分配：降低最小批次大小，提高GPU利用率
        min_samples_per_gpu = self._min_batch_size_per_gpu  # 使用配置的最小批次大小
        if batch_size < len(self.device_ids) * min_samples_per_gpu:
            # 对于小批次，仍然尝试分配到多个GPU以提高利用率
            if batch_size >= min_samples_per_gpu:
                # 如果批次大小足够，仍然分配到多个GPU
                pass
            else:
                # 只有批次非常小时才只用第一个GPU
                return [batch] + [None] * (len(self.device_ids) - 1)

        chunk_size = batch_size // len(self.device_ids)

        distributed_batches = []
        for i, device_id in enumerate(self.device_ids):
            start_idx = i * chunk_size
            if i == len(self.device_ids) - 1:
                # Last GPU gets remaining samples
                end_idx = batch_size
            else:
                end_idx = (i + 1) * chunk_size

            if start_idx < batch_size:
                chunked_batch = []
                for tensor in batch:
                    if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                        # Only slice tensors with batch dimension
                        chunked_batch.append(tensor[start_idx:end_idx])
                    elif isinstance(tensor, dict):
                        # Handle dictionary of tensors
                        chunked_dict = {}
                        for key, value in tensor.items():
                            if isinstance(value, torch.Tensor) and value.dim() > 0:
                                chunked_dict[key] = value[start_idx:end_idx]
                            else:
                                chunked_dict[key] = value
                        chunked_batch.append(chunked_dict)
                    else:
                        # Keep non-tensors or 0-dim tensors as-is
                        chunked_batch.append(tensor)
                distributed_batches.append(tuple(chunked_batch))
            else:
                distributed_batches.append(None)

        return distributed_batches

    def _is_training_mode(self, kwargs):
        """
        检测是否为训练模式
        通过检查kwargs中是否包含args参数来判断
        """
        return 'args' in kwargs and hasattr(kwargs['args'], 'device')

    def forward(self, *args, **kwargs):
        """
        Forward pass through all replicas.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            For training mode: (loss, log_dict) from the first GPU (cuda:0)
            For inference mode: Concatenated outputs from all replicas
        """
        if len(self.replicas) == 1:
            # Single GPU case
            if self._is_training_mode(kwargs):
                # Training mode: call forward_train
                return self.replicas[0].forward_train(*args, **kwargs)
            else:
                # Inference mode: call forward_infer
                return self.replicas[0].forward_infer(*args, **kwargs)

        # Multi-GPU case
        is_training = self._is_training_mode(kwargs)
        print(
            f"MyDataParallel forward: {len(self.replicas)} replicas, device_ids: {self.device_ids}, training_mode: {is_training}")

        batch = args
        distributed_batches = self._distribute_batch(batch)
        print(f"Distributed {len(distributed_batches)} batches across {len(self.replicas)} GPUs")

        # 准备每个replica的输入参数 - 使用异步传输优化
        kwargs_list = []
        for i, device_batch in enumerate(distributed_batches):
            if device_batch is not None:
                # 使用异步传输优化数据传输
                device_batch = self._async_transfer_batch(device_batch, self.device_ids[i])

                # 创建kwargs字典，优雅地处理参数
                replica_kwargs = dict(kwargs)
                # 将device_batch作为位置参数传递
                kwargs_list.append((device_batch, replica_kwargs))
            else:
                kwargs_list.append(None)

        if is_training:
            # 训练模式：并行执行forward_train，返回第一个GPU的结果
            return self._parallel_forward_train(kwargs_list)
        else:
            # 推理模式：并行执行forward_infer，合并结果
            return self._parallel_forward_infer(kwargs_list)

    def get_gpu_losses(self):
        """Get losses from each GPU for debugging"""
        return getattr(self, '_gpu_losses', [])

    def clear_cache(self):
        """Clear GPU cache to free memory with comprehensive cleanup"""
        try:
            # 清理所有设备的内存
            for device_id in self.device_ids:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()

            # 强制垃圾回收
            import gc
            gc.collect()

            # 同步所有设备
            self._all_devices_safe_check()

        except Exception as e:
            print(f"Error during cache clearing: {e}")
            # 即使出错也要尝试基本清理
            torch.cuda.empty_cache()
            import gc
            gc.collect()


class MyDataParallelOptimizer:
    """
    Optimizer wrapper for MyDataParallel that handles gradient synchronization.
    """

    def __init__(self, model: MyDataParallel, optimizer_class, args, **optimizer_kwargs):
        """
        Initialize the optimizer wrapper.

        Args:
            model: MyDataParallel model
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            **optimizer_kwargs: Arguments for the optimizer
        """
        self.model = model

        # Create optimizers for each replica
        self.replica_optimizers = []
        # Flat list of optimizers for schedulers/logging
        self._flat_optimizers = []
        for replica in model.replicas[:1]:

            # muon optimizer branch: build param groups per replica
            if hasattr(args, 'muon') and args.muon:
                from utils import SingleDeviceMuon
                muon_params = []
                adam_params = []

                for name, param in replica.named_parameters():
                    if not param.requires_grad:
                        continue
                    is_muon_candidate = False
                    # transformer/hstu linear weights (no bias)
                    if (('attention_layers' in name or 'forward_layers' in name or 'hstu_layers' in name)
                            and param.ndim >= 2 and 'bias' not in name):
                        is_muon_candidate = True
                    # optional dnn weights
                    if (('userdnn' in name or 'itemdnn' in name) and 'weight' in name):
                        is_muon_candidate = True

                    if is_muon_candidate:
                        muon_params.append(param)
                        # print(f"[Muon Param]: {name}")
                    else:
                        adam_params.append(param)
                        # print(f"[Adam Param]: {name}")

                optimizers = []
                if len(adam_params) > 0:
                    adamw_optimizer = ManualAdamW(
                        adam_params,
                        lr=optimizer_kwargs.get('lr', getattr(args, 'lr', 3e-4)),
                        betas=optimizer_kwargs.get('betas', (0.9, 0.98)),
                        eps=optimizer_kwargs.get('eps', 1e-9),
                        weight_decay=optimizer_kwargs.get('weight_decay', getattr(args, 'weight_decay', 0.0))
                    )
                    # from torch.optim import SM3
                    # adamw_optimizer = SM3(
                    #     adam_params,
                    #     lr=optimizer_kwargs.get('lr', getattr(args, 'lr', 3e-4)),
                    #     betas=optimizer_kwargs.get('betas', (0.9, 0.98)),
                    #     eps=optimizer_kwargs.get('eps', 1e-9),
                    #     weight_decay=optimizer_kwargs.get('weight_decay', getattr(args, 'weight_decay', 0.0))
                    # )
                    # Use internal ManualAdafactor to avoid external dependency
                    # adamw_optimizer = ManualAdafactor(
                    #     adam_params,
                    #     lr=optimizer_kwargs.get('lr', getattr(args, 'lr', 1e-3)),
                    #     eps=optimizer_kwargs.get('eps', (1e-30, 1e-3)),
                    #     weight_decay=optimizer_kwargs.get('weight_decay', getattr(args, 'weight_decay', 0.0)),
                    #     relative_step=False,
                    #     scale_parameter=False,
                    #     warmup_init=False,
                    # )
                    optimizers.append(adamw_optimizer)
                    print(f"Setting Manual optimizer successfully for replica {replica.dev}")
                if len(muon_params) > 0:
                    muon_optimizer = SingleDeviceMuon(
                        muon_params,
                        lr=getattr(args, 'muon_lr', 0.02),
                        weight_decay=optimizer_kwargs.get('weight_decay', getattr(args, 'weight_decay', 0.0)),
                        momentum=getattr(args, 'muon_momentum', 0.95)
                    )
                    optimizers.append(muon_optimizer)
                    print(f"Setting muon optimizer successfully for replica {replica.dev}")

                self.replica_optimizers.append(optimizers)
                self._flat_optimizers.extend(optimizers)
                # Backward-compat single handle
                self.optimizer = optimizers[0] if len(optimizers) > 0 else None
                continue

            # default branch: dense/sparse separation
            dense_params = []
            sparse_params = []
            for name, param in replica.named_parameters():
                if 'emb' in name and 'transform' not in name and args.sparse_embedding:
                    sparse_params.append(param)
                else:
                    dense_params.append(param)

            optimizers = []
            if len(dense_params) > 0:
                dense_optimizer = optimizer_class(dense_params, **optimizer_kwargs)
                optimizers.append(dense_optimizer)
                print("Setting dense optimizer successfully for replica {replica.dev}")

            if len(sparse_params) > 0:
                sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=optimizer_kwargs.get('lr', 0.001))
                optimizers.append(sparse_optimizer)
                print("Setting sparse optimizer successfully for replica {replica.dev}")

            self.replica_optimizers.append(optimizers)
            self._flat_optimizers.extend(optimizers)
            # Backward-compat single handle
            self.optimizer = optimizers[0] if len(optimizers) > 0 else None

    def get_all_optimizers(self):
        """Return a flat list of underlying torch.optim.Optimizer instances for LR scheduling/logging."""
        return list(self._flat_optimizers)

    def step(self, gradient_accumulation_steps=1):
        """Perform optimization step on all replicas with optimized gradient synchronization."""
        # import time
        # start_time = time.time()

        # Synchronize gradients before optimization
        if len(self.model.replicas) > 1:
            self._sync_gradients(gradient_accumulation_steps)

        # Perform optimization step on all replicas
        for optimizers in self.replica_optimizers[:1]:
            for optimizer in optimizers:
                optimizer.step()

        # Update step count and sync parameters if needed
        # self.model.step_count += 1
        # if self.model.step_count % self.model.sync_frequency == 0:
        self.model._sync_parameters()

        # step_time = time.time() - start_time
        # self.model.step_times.append(step_time)

    def _sync_gradients(self, gradient_accumulation_steps=1):
        """Synchronize gradients across all replicas with safe memory operations."""
        if len(self.model.replicas) == 1:
            return

        try:
            # Wait for all devices to complete their operations
            self.model._all_devices_safe_check()

            primary_params = self.model._cached_primary_params
            replica_params_list = self.model._cached_replica_params

            # 缩放系数
            accumulation_factor = 1.0 / (len(self.model.device_ids) * gradient_accumulation_steps)

            for param_idx, primary_param in enumerate(primary_params):
                if primary_param.grad is None:
                    continue

                # 使用更安全的内存操作，避免创建临时张量
                primary_grad = primary_param.grad.data
                primary_grad.mul_(accumulation_factor)

                # 收集所有副本的梯度
                for replica_idx in range(len(self.model.replicas) - 1):
                    replica_params = replica_params_list[replica_idx]
                    replica_param = replica_params[param_idx]
                    if replica_param.grad is None:
                        continue

                    replica_grad = replica_param.grad.data

                    # 检查梯度类型和稀疏性
                    if replica_grad.is_sparse:
                        print(f"Warning: Sparse gradient detected for param {param_idx}, skipping")
                        continue

                    # 安全的设备间拷贝
                    replica_device = replica_param.device
                    if replica_grad.device != primary_grad.device:
                        replica_grad = replica_grad.to(primary_grad.device, non_blocking=False)

                    # 就地累加，减少内存分配
                    primary_grad.add_(replica_grad, alpha=accumulation_factor)
                    replica_grad.to(replica_device, non_blocking=False)

                # 确保梯度数据正确设置
                primary_param.grad.data = primary_grad

            # 最终同步检查
            self.model._all_devices_safe_check()

        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error during gradient synchronization: {e}")
                # 尝试清理内存并重试
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                raise e
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error during gradient synchronization: {e}")
            raise e

    def zero_grad(self):
        """Zero gradients on all replicas with safe memory management."""
        try:
            # 使用更安全的方式清零梯度，避免内存泄漏
            for replica in self.model.replicas:
                for param in replica.parameters():
                    if param.grad is not None:
                        param.grad = None  # 直接设置为None，避免zero_()的内存开销

            # 同时调用优化器的zero_grad
            for optimizers in self.replica_optimizers:
                for optimizer in optimizers:
                    optimizer.zero_grad()
        except Exception as e:
            print(f"Error during zero_grad: {e}")
            # 尝试清理内存
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            raise e

    def state_dict(self):
        """Return state dict from primary optimizer."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict to primary optimizer."""
        self.optimizer.load_state_dict(state_dict)


def create_my_dataparallel(model: nn.Module, device_ids: Optional[List[int]] = None) -> MyDataParallel:
    """
    Create a MyDataParallel wrapper for the given model.

    Args:
        model: The model to parallelize
        device_ids: List of GPU device IDs to use

    Returns:
        MyDataParallel wrapped model
    """
    return MyDataParallel(model, device_ids)


def create_my_dataparallel_optimizer(model: MyDataParallel, optimizer_class, args, **optimizer_kwargs):
    """
    Create a MyDataParallelOptimizer for the given model.

    Args:
        model: MyDataParallel model
        optimizer_class: Optimizer class
        **optimizer_kwargs: Arguments for the optimizer

    Returns:
        MyDataParallelOptimizer
    """
    return MyDataParallelOptimizer(model, optimizer_class, args, **optimizer_kwargs)


class ManualAdamW(Optimizer):
    """
    A minimal AdamW implementation compatible with torch LR schedulers.
    - Decoupled weight decay (as in AdamW)
    - Supports param_groups with keys: lr, betas, eps, weight_decay
    - Skips sparse gradients
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group.get('lr', 1e-3)
            beta1, beta2 = group.get('betas', (0.9, 0.999))
            eps = group.get('eps', 1e-8)
            weight_decay = group.get('weight_decay', 0.0)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    # AdamW does not support sparse gradients in this simple impl
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()


class ManualAdafactor(Optimizer):
    """
    Minimal Adafactor (no transformers dependency), fixed-lr mode only.
    Supports arguments: lr, eps (tuple: (eps1, eps2)), weight_decay,
    relative_step (must be False), scale_parameter (ignored here), warmup_init (ignored).
    This is a simplified variant using factored second-moment when tensor is 2D.
    """

    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3), weight_decay=0.0,
                 relative_step=False, scale_parameter=False, warmup_init=False):
        if relative_step:
            raise ValueError("ManualAdafactor: relative_step=True not supported in this minimal impl.")
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group.get('lr', 1e-3)
            eps1, eps2 = group.get('eps', (1e-30, 1e-3))
            weight_decay = group.get('weight_decay', 0.0)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    # Not supported in minimal version
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    if p.ndim >= 2:
                        r, c = p.shape[-2], p.shape[-1]
                        state['v_row'] = torch.zeros(*p.shape[:-2], r, device=p.device, dtype=p.dtype)
                        state['v_col'] = torch.zeros(*p.shape[:-2], c, device=p.device, dtype=p.dtype)
                    else:
                        state['v'] = torch.zeros_like(p)

                state['step'] += 1

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Compute second-moment estimate (factored if possible)
                if p.ndim >= 2 and ('v_row' in state and 'v_col' in state):
                    v_row = state['v_row']
                    v_col = state['v_col']
                    grad_sq = g.float().pow(2).mean(dim=-1)
                    v_row.mul_(0.999).add_(grad_sq, alpha=1 - 0.999)
                    grad_sq_col = g.float().pow(2).mean(dim=-2)
                    v_col.mul_(0.999).add_(grad_sq_col, alpha=1 - 0.999)
                    # Reconstruct RMS
                    r_factor = (v_row + eps1).rsqrt().unsqueeze(-1)
                    c_factor = (v_col + eps1).rsqrt().unsqueeze(-2)
                    rms = (r_factor * c_factor).to(g.dtype)
                else:
                    v = state['v']
                    v.mul_(0.999).addcmul_(g, g, value=1 - 0.999)
                    rms = (v + eps1).rsqrt()

                update = g / (rms / eps2 + 1e-12)
                p.add_(update, alpha=-lr)

        return loss