import os
import torch
import torch.distributed as dist
import torch.nn as nn
from contextlib import contextmanager

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.require_backward_grad_sync = True
        
        # Broadcast initial weights from rank 0 to all other processes
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # Broadcast buffers
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)
            
        # Register hooks for async communication
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook())
                
    def _make_hook(self):
        def hook(param):
            if param.grad is None or not self.require_backward_grad_sync:
                return
            
            with torch.no_grad():
                param.grad.div_(dist.get_world_size())
                handle = dist.all_reduce(param.grad, async_op=True)
                self.handles.append(handle)
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        old_val = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_val

    def finish_gradient_synchronization(self):
        # Wait for all async communication to finish
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        # Ensure all CUDA operations are completed
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 25):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.handles = []
        self.require_backward_grad_sync = True
        
        # Broadcast initial weights
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        # Bucket parameters
        self.buckets = []
        self.param_to_bucket_idx = {}
        
        current_bucket_params = []
        current_bucket_size_bytes = 0
        threshold_bytes = bucket_size_mb * 1024 * 1024
        
        # Iterate in REVERSE order (model output -> input) to maximize overlap
        params_with_grad = [p for p in self.module.parameters() if p.requires_grad]
        
        for param in reversed(params_with_grad):
            param_bytes = param.numel() * param.element_size()
            
            current_bucket_params.append(param)
            current_bucket_size_bytes += param_bytes
            
            if current_bucket_size_bytes >= threshold_bytes:
                self._create_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size_bytes = 0
        
        # Add remaining
        if current_bucket_params:
            self._create_bucket(current_bucket_params)
            
        # Register hooks
        for param in params_with_grad:
             param.register_post_accumulate_grad_hook(self._make_hook())
    
    def _create_bucket(self, params):
        bucket_idx = len(self.buckets)
        total_numel = sum(p.numel() for p in params)
        
        first_p = params[0]
        buffer = torch.zeros(total_numel, device=first_p.device, dtype=first_p.dtype)
        
        self.buckets.append({
            'params': params,
            'total_numel': total_numel,
            'buffer': buffer,
            'ready_count': 0
        })
        
        for p in params:
            self.param_to_bucket_idx[p] = bucket_idx

    def _make_hook(self):
        def hook(param):
            if param.grad is None or not self.require_backward_grad_sync:
                return
            
            bucket_idx = self.param_to_bucket_idx[param]
            bucket = self.buckets[bucket_idx]
            bucket['ready_count'] += 1
            
            if bucket['ready_count'] == len(bucket['params']):
                self._all_reduce_bucket(bucket)
        return hook

    def _all_reduce_bucket(self, bucket):
        buffer = bucket['buffer']
        
        # Flatten grads
        offset = 0
        for p in bucket['params']:
            numel = p.numel()
            buffer[offset:offset+numel].copy_(p.grad.view(-1))
            offset += numel
            
        # Div by world size
        buffer.div_(dist.get_world_size())
        
        # Async all_reduce
        handle = dist.all_reduce(buffer, async_op=True)
        self.handles.append((handle, bucket))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        old_val = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_val

    def finish_gradient_synchronization(self):
        for handle, bucket in self.handles:
            handle.wait()
            # Copy back
            buffer = bucket['buffer']
            offset = 0
            for p in bucket['params']:
                numel = p.numel()
                p.grad.view(-1).copy_(buffer[offset:offset+numel])
                offset += numel
            bucket['ready_count'] = 0
            
        self.handles.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
