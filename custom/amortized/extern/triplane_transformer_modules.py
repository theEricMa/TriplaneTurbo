import torch
import torch.nn as nn
import numpy as np

from diffusers.models.attention_processor import Attention, AttnProcessor, LoRAAttnProcessor, AttentionProcessor
from threestudio.utils.typing import *


class ModLN(nn.Module):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]

class ConditionModulationBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,  mlp_ratio: float = 4.,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_drop: float = 0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(inner_dim, eps)
        self.cross_attn = Attention(
            query_dim=inner_dim, heads=num_heads, dim_head=inner_dim // num_heads,
            cross_attention_dim=cond_dim, 
            dropout=attn_drop, bias=attn_bias, )
        self.norm2 =  nn.LayerNorm(inner_dim, eps)
        self.self_attn = Attention(
            query_dim=inner_dim, heads=num_heads, dim_head=inner_dim // num_heads,
            cross_attention_dim=inner_dim, 
            dropout=attn_drop, bias=attn_bias, )
        self.norm3 =  nn.LayerNorm(inner_dim, eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        # mod: [N, D_mod]
        x = x + self.cross_attn(self.norm1(x), cond)
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa)
        x = x + self.mlp(self.norm3(x))
        return x

class ConditionModulationBlockwoCrossAttn(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float, mlp_ratio: float = 4.,
                 attn_drop: float = 0., attn_bias: bool = False, mlp_drop: float = 0.):
        super().__init__()
        self.norm2 = nn.LayerNorm(inner_dim, eps)
        self.self_attn = Attention(
            query_dim=inner_dim, heads=num_heads, dim_head=inner_dim // num_heads,
            cross_attention_dim=inner_dim, 
            dropout=attn_drop, bias=attn_bias, )
        self.norm3 = nn.LayerNorm(inner_dim, eps)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        # mod: [N, D_mod]

        # concatenate the condition to the input
        x = torch.cat([cond, x], dim=1)
        before_sa = self.norm2(x)

        # self-attention, leave out the 1st token
        x = x + self.self_attn(before_sa)
        x = x + self.mlp(self.norm3(x))

        # remove the 1st token, which is the condition
        x = x[:, 1:, :]
        return x



class TriplaneTransformer(nn.Module):
    """
    Transformer with condition and modulation that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(self, 
                inner_dim: int, condition_dim: int,
                triplane_low_res: int, triplane_high_res: int, triplane_dim: int,
                num_layers: int, num_heads: int,
                flash_attention: bool, local_text: bool,
                mlp_ratio: float = 4., eps: float = 1e-6):
        super().__init__()

        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, inner_dim) * (1. / inner_dim) ** 0.5)
        
        # condition modulation
        self.needs_local_text = local_text
        self.layers = nn.ModuleList([
            ConditionModulationBlockwoCrossAttn(
                inner_dim=inner_dim, cond_dim=condition_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, eps=eps,
            ) if not local_text else \
            ConditionModulationBlock(
                inner_dim=inner_dim, cond_dim=condition_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, eps=eps,
            ) for _ in range(num_layers)
        ])
        
        if flash_attention:
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
            self.set_use_memory_efficient_attention_xformers(True, MemoryEfficientAttentionFlashAttentionOp)

        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.deconv = nn.ConvTranspose2d(inner_dim, triplane_dim, kernel_size=2, stride=2, padding=0, bias=False) 
   
        if not local_text:
            self.proj = nn.Linear(condition_dim, inner_dim)

    def forward(self, text_embed):
        N = text_embed.shape[0]
        H = W = self.triplane_low_res
        L = 3 * H * W

        # project text_embed to inner_dim
        if not self.needs_local_text:
            text_embed = self.proj(text_embed)
            text_embed = text_embed.unsqueeze(1)

        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        n = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            x = layer(x, text_embed)
            # x += self.pos_embed * (n - idx) / n # trick: help convergence
        x = self.norm(x)

        # separate each plane and apply deconv
        x = x.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.deconv(x)  # [3*N, D', H', W']
        # x = torch.tanh(x) # tanh help convergence
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()

        assert self.triplane_high_res == x.shape[-2], \
            f"Output triplane resolution does not match with expected: {x.shape[-2]} vs {self.triplane_high_res}"
        assert self.triplane_dim == x.shape[-3], \
            f"Output triplane dimension does not match with expected: {x.shape[-3]} vs {self.triplane_dim}"

        return x
         
    # the following are copied from diffusers
    # for use_memory_efficient_attention_xformers ########################################
    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    # for attention processors ########################################
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)