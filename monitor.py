import torch
import psutil
import numpy as np
from typing import List, Set, Dict, Optional, Tuple, Iterator, Type, Union, Any
from collections import defaultdict, deque
import logging
from threading import Lock

class ResourceMonitor:
    """Monitor system resources and provide optimization recommendations"""
    def __init__(self, target_gpu_util=0.85, target_cpu_util=0.75, 
                 target_memory_util=0.85, window_size=5):
        self.target_gpu_util = target_gpu_util
        self.target_cpu_util = target_cpu_util
        self.target_memory_util = target_memory_util
        self.window_size = window_size
        
        # Performance tracking
        self.processing_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.gpu_utils = deque(maxlen=window_size)
        
        # Resource monitoring
        self.total_memory = psutil.virtual_memory().total
        self.cpu_count = psutil.cpu_count()
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Adjustment locks
        self.adjustment_lock = Lock()

    def get_current_resources(self) -> Dict[str, float]:
        """Get current system resource utilization"""
        resources = {
            'cpu_util': psutil.cpu_percent() / 100,
            'memory_util': psutil.virtual_memory().percent / 100,
            'gpu_util': 0.0,
            'gpu_memory_util': 0.0
        }
        
        if self.has_gpu:
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                resources['gpu_memory_util'] = gpu_memory_allocated / self.gpu_memory
                if hasattr(torch.cuda, 'utilization'):
                    resources['gpu_util'] = torch.cuda.utilization() / 100
            except Exception as e:
                logging.warning(f"Error getting GPU metrics: {e}")
        
        return resources

    def calculate_optimal_batch_size(self, current_batch_size: int, 
                                   processing_time: float) -> Tuple[int, bool]:
        """Calculate optimal batch size based on resource utilization and performance"""
        with self.adjustment_lock:
            resources = self.get_current_resources()
            
            # Store metrics
            self.processing_times.append(processing_time)
            self.batch_sizes.append(current_batch_size)
            if self.has_gpu:
                self.gpu_utils.append(resources['gpu_util'])
            
            # Don't adjust until we have enough data points
            if len(self.processing_times) < self.window_size:
                return current_batch_size, False
            
            # Calculate performance metrics
            avg_processing_time = np.mean(self.processing_times)
            processing_time_trend = np.gradient(self.processing_times).mean()
            
            # Resource headroom
            memory_headroom = self.target_memory_util - resources['memory_util']
            cpu_headroom = self.target_cpu_util - resources['cpu_util']
            
            # Base adjustment factor
            adjustment_factor = 1.0
            
            # Adjust based on resource utilization
            if memory_headroom < 0 or cpu_headroom < 0:
                adjustment_factor = 0.8
            elif self.has_gpu:
                gpu_util = resources['gpu_util']
                gpu_memory_headroom = self.target_gpu_util - resources['gpu_memory_util']
                
                if gpu_memory_headroom < 0:
                    adjustment_factor = 0.8
                elif gpu_util < self.target_gpu_util * 0.8:
                    adjustment_factor = 1.2
            else:
                if cpu_headroom > 0.3:
                    adjustment_factor = 1.1
                elif cpu_headroom > 0.1:
                    adjustment_factor = 1.05
            
            if processing_time_trend > 0:
                adjustment_factor *= 0.95
            
            new_batch_size = max(1, min(
                int(current_batch_size * adjustment_factor),
                self._get_max_safe_batch_size()
            ))
            
            should_adjust = abs(new_batch_size - current_batch_size) >= max(1, current_batch_size * 0.1)
            
            return new_batch_size, should_adjust

    def _get_max_safe_batch_size(self) -> int:
        """Calculate maximum safe batch size based on available resources"""
        if self.has_gpu:
            free_gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            max_gpu_batch = int(free_gpu_memory * 0.8 / (1024 * 1024))
        else:
            max_gpu_batch = float('inf')
        
        free_system_memory = psutil.virtual_memory().available
        max_system_batch = int(free_system_memory * 0.8 / (1024 * 1024))
        
        return min(max_gpu_batch, max_system_batch, 512)
