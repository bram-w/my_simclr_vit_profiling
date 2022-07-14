# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import numpy as np
import timm
import torch
from torch import nn

import losses

class ParallelLayerNorm(nn.Module):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, groups, d):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.groups = groups
        self.d = d
        self.weight = nn.Parameter(torch.ones(1, 1, groups * d))
        self.bias = nn.Parameter(torch.zeros(1, 1, groups * d))

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # input L x N x (groups * d)
        # x is L x N x (GD) where each models features are continuous (D + D + D)
        # reshaping this tensor with (groups, d) gives (D, D, D, D) as desired
        x = x.view(x.shape[0], x.shape[1], self.groups, self.d)
        x = self.ln(x.type(torch.float32))
        x = x.flatten(start_dim=2).type(orig_type)
        x = x * self.weight + self.bias
        return x

#[RESOLVED] Multihead attention might be fine since it relegates itself to heads?
#       FINAL STAGE is NOT FINE: HAS FULL LINEAR LAYER AT END
#       Might be able to just do set to identity, turn off grad, put in new linear layer?
#       It's called the projection dig into tomorrow
#    NEED TO TURN OFF IN PROJ TOO, FIGURE OUT SHUFFLING
# THEN MULTIHEAD MIGHT BE OK
# w: projection weights for q, k and v, packed into a single tensor. Weights
#             are packed along dimension 0, in q, k, v order.
# so do need weight to 

# [RESOLVED] Linear can be changed to a grouped 1d conv I think
#    Unsqueeze last dim, do grouped 1d conv, can do once per MLP

# [RESOLVED] LayerNorm needs to be modified to act in grouped way
#     do layernorm normalization in a layer
#     pull out elementwise affine calculation and handle on one at end

# [RESOLVED] Need to look into causal attention mask thing, not sure abou that
# causal attention is valid choice, prevents attending to padding tokens, works with my sequence length
# Make sure to add in! but keep


class ParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, groups, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.conv = nn.Conv1d(in_features, out_features, 1, groups=groups, bias=bias)

    def forward(self, x):
        # x is L x N x (GD) where each models features are continuous (D + D + D)
        # conv1d groups act as expected: continuous as above
        # so just need to combine L and N, add dimension at end so can do 1d conv
        L, N = x.shape[:2]
        x = x.view(L*N, -1, 1)
        x = self.conv(x)
        # Might be more efficient to refactor into single MLP block to avoid extra views but idk
        x = x.view(L, N, -1)
        return x

class ParallelMultiheadAttention(nn.Module):
    
    def __init__(self, groups, base_d, n_head_per_group):
        super().__init__()
        self.groups = groups
        self.base_d = base_d
        self.n_head_per_group = n_head_per_group

        self.in_proj = ParallelLinear(groups * base_d, 3 * groups * base_d, groups)

        self.attn_base = nn.MultiheadAttention(groups * base_d, groups * n_head_per_group, bias=False)
        # TODO: Can I bypass this without the huge matrix mult? Is this an issue for backward?
        self.attn_base.in_proj_weight.data = torch.cat([torch.eye(groups * base_d) for _ in range(3)])
        self.attn_base.out_proj.weight.data = torch.eye(groups * base_d)
        self.attn_base.in_proj_weight.requires_grad = False
        self.attn_base.out_proj.weight.requires_grad = False

        self.out_proj = ParallelLinear(groups * base_d, groups * base_d, groups)

    def forward(self, x, attn_mask, need_weights=False):
        # x is L x N x (GD) where each models features are continuous (D + D + ... + D)
        x = self.in_proj(x) # Now x is (3D + 3D + .. 3D)
        # Might be more optimized to do this within linear
        x = x.view(x.shape[0], x.shape[1], self.groups, 3, self.base_d).transpose(2, 3).flatten(start_dim=3)
        # Now formerly clustered chunks are interwoven
        # Before L X N X [a1 a2 a3 b1 b2 b3]
        # now L X N X [[a1 b1] [a2 b2] [a3 b3]]
        # so just send in rows

        # Run pure attention (weights are frozen identity)
        x = self.attn_base(x[:, :, 0], x[:, :, 1], x[:, :, 2], need_weights=need_weights, attn_mask=attn_mask)[0]
        # Do final proj in groups
        x = self.out_proj(x)
        return x



class ParallelResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head_per_group: int, attn_mask: torch.Tensor = None, groups: int = 1):
        super().__init__()

        self.attn = ParallelMultiheadAttention(groups, d_model, n_head_per_group)

        self.ln_1 = ParallelLayerNorm(groups, d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", ParallelLinear(groups * d_model, groups * d_model * 4, groups)),
            ("gelu", QuickGELU()),
            ("c_proj", ParallelLinear(groups * d_model * 4, groups * d_model, groups))
        ]))
        self.ln_2 = ParallelLayerNorm(groups, d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ParallelTransformer(nn.Module):
    # TODO: Build attn mask if doing language
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 num_groups=1):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_groups = num_groups
        self.resblocks = nn.Sequential(*[ParallelResidualAttentionBlock(width, heads, attn_mask, groups=num_groups) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ParallelTextEncoder(nn.Module):

    def __init__(self,
                num_models: int,
                output_dim_per_model: int, # 1 for most of work
                context_length: int,
                vocab_size: int,
                transformer_width: int,
                transformer_heads: int,
                transformer_layers: int):

        super().__init__()

        self.context_length = context_length
        self.num_models = num_models
        self.output_dim_per_model = output_dim_per_model

        self.transformer = ParallelTransformer(width=transformer_width,
                                            layers=transformer_layers,
                                            heads=transformer_heads,
                                            attn_mask = self.build_attention_mask(),
                                            num_groups=num_models)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = ParallelLayerNorm(self.num_models, transformer_width)

        self.text_projection = ParallelLinear(num_models * transformer_width, num_models * output_dim_per_model,
                                                groups=num_models, bias=False)
        
        self.initialize_parameters() 

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj.conv.weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.conv.weight, std=proj_std)
            
            nn.init.zeros_(block.attn.in_proj.conv.bias)
            nn.init.zeros_(block.attn.out_proj.conv.bias)
            
            # biases on linear aren't changed in orig
            nn.init.normal_(block.mlp.c_fc.conv.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.conv.weight, std=proj_std)

        nn.init.normal_(self.text_projection.conv.weight, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Now repeat to get LND --> LN(GD)
        x = x.repeat(1, 1, self.num_models)
        # Process (still LN(GD) )
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LN(GD) -> NL(GD)
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
        #Now N(GD) tensor and want to do single projection to N x (G * output_dim_per_model)
        x = self.text_projection(x.unsqueeze(0)).squeeze(0)
        
        return x

###########
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)


    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, return_logit_scale=True):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp() if return_logit_scale else None}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


