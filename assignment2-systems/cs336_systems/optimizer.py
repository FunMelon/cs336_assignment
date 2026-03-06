import torch
import torch.distributed as dist
from typing import Type, Any

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        
        # Setup rank/world_size BEFORE calling super().__init__
        # because super().__init__ calls add_param_group which uses them
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.param_to_rank = {}
        # self.local_param_groups needs to be managed carefully because add_param_group called by super
        # will try to add to local_optimizer, but local_optimizer isn't created yet!
        # So we need to separate initialization.
        
        # We can't fully support add_param_group during super().__init__ because 
        # local_optimizer doesn't exist yet.
        # Strategy: 
        # 1. Initialize self properties.
        # 2. Call super().__init__(params, defaults)
        #    This calls self.add_param_group for each group.
        #    Inside add_param_group, we check if local_optimizer exists.
        #    If not, we just build up a list of local groups to be used later.
        # 3. After super().__init__, create local_optimizer with the collected local groups.
        
        self.local_param_groups_buffer = [] 
        self.local_optimizer = None
        
        # Initialize parent class to handle param_groups structure
        super().__init__(params, defaults=kwargs)
        
        # Now create the actual local optimizer
        self.local_optimizer = self.optimizer_cls(self.local_param_groups_buffer, **self.kwargs)
        
        # Clear buffer to save memory, though not strictly necessary
        self.local_param_groups_buffer = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update local parameters using local optimizer
        self.local_optimizer.step()

        # Broadcast updated parameters to all other ranks
        # Iterate over all parameters in the main optimizer's groups
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                owner_rank = self.param_to_rank[param]
                dist.broadcast(param.data, src=owner_rank)
                
        return loss

    def zero_grad(self, set_to_none: bool = False):
        # Clear gradients for all parameters (handled by parent class)
        super().zero_grad(set_to_none=set_to_none)
        
        # Also ensure local optimizer is clear, though it shares the same tensor objects
        self.local_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]):
        # Add to super (updates self.param_groups)
        super().add_param_group(param_group)
        
        # Calculate start index based on existing params minus new ones
        total_params = sum(len(g['params']) for g in self.param_groups)
        start_idx = total_params - len(param_group['params'])
        
        local_new_params = []
        for i, param in enumerate(param_group['params']):
            global_idx = start_idx + i
            owner_rank = global_idx % self.world_size
            self.param_to_rank[param] = owner_rank
            
            if owner_rank == self.rank:
                local_new_params.append(param)
        
        # Prepare local group config
        local_group_config = {k: v for k, v in param_group.items() if k != 'params'}
        local_group_config['params'] = local_new_params
        
        if self.local_optimizer is None:
            # During init phase
            self.local_param_groups_buffer.append(local_group_config)
        else:
            # Runtime phase
            self.local_optimizer.add_param_group(local_group_config)
