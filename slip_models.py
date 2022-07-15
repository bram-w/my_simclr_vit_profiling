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

from torchvision.models import mobilenet_v3_small
def text_binary_call():
    # base is 512 8 12
    # quadrating in width, linear in depth
    # to reduce by 64 can reudce width by 8
    # I don't think heads matters, but could be wrong.
    # dropping down to maintain more width for each head
    return CLIP(embed_dim=1,
            vision_width=1,
            vision_model=None,
            context_length=77,
            vocab_size=49408,
            transformer_width=64,
            transformer_heads=4,
            transformer_layers=12
            )

class MultiBinaryText(nn.Module):

    def __init__(self, num_models=64):
        super().__init__()
        self.model_list = nn.ModuleList([text_binary_call() for _ in range(num_models)])

    def forward(self, x):
        # this is raw logits, should sigmoid or tanh
        return torch.cat([m.encode_text(x) for m in self.model_list], dim=1)

def vision_binary_call():
    net = mobilenet_v3_small(num_classes=1)
    net.classifier[2].p = 0
    return net

class MultiBinaryVision(nn.Module):

    def __init__(self, num_models=64):
        super().__init__()
        self.model_list = nn.ModuleList([vision_binary_call() for _ in range(num_models)])

    def forward(self, x):
        # this is raw logits, should sigmoid or tanh
        return torch.cat([m(x) for m in self.model_list], dim=1)

class MultiBinaryCLIP(nn.Module):
    def __init__(self,
                num_models,
                ):
        super().__init__()

        self.visual = MultiBinaryVision(num_models)
        self.language = MultiBinaryText(num_models)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        return self.language(text)

    def forward(self, image, text, return_logit_scale=True):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale':self.logit_scale.exp() if return_logit_scale else None}

import parallel_transformer
import parallel_protonet
class ParallelMultiBinaryCLIP(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.visual = parallel_protonet.make_simple_protonet(3, 64*64, 1, 64)
        self.language = parallel_transformer.ParallelTextEncoder(num_models=64,
                                                                output_dim_per_model=1,
                                                                context_length=77,
                                                                vocab_size=49408,
                                                                transformer_width=64,
                                                                transformer_heads=4,
                                                                transformer_layers=6)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        return self.language(text)

    def forward(self, image, text, return_logit_scale=True):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale':self.logit_scale.exp() if return_logit_scale else None}


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

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

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


class SIMCLR(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # ssl
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model

        self.image_mlp = self._build_mlp(in_dim=vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def encode_image(self, image):
        x = self.visual(image)

        return x

    def forward(self, aug1, aug2):
        h1 = self.visual(aug1)
        h2 = self.visual(aug2)

        aug1_embed = self.image_mlp(h1)
        aug2_embed = self.image_mlp(h2)

        return {'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed}


class SLIP(CLIP):
    def __init__(self,
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.image_mlp = self._build_mlp(in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def forward(self, image, text, aug1, aug2):
        aug1_embed = self.image_mlp(self.visual(aug1))
        aug2_embed = self.image_mlp(self.visual(aug2))
        
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp(),
                'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed}


def get_loss(model, ssl_temp, ssl_scale):
    if model.startswith('SLIP'):
        ssl_loss = losses.SIMCLRLoss(temperature=ssl_temp)
        return losses.SLIPLoss(ssl_loss, ssl_scale)
    if model.startswith('CLIP'):
        return losses.CLIPLoss()
    if model.startswith('SIMCLR'):
        return losses.SIMCLRLoss(temperature=ssl_temp)


def get_metric_names(model):
    if model.startswith('SLIP'):
        return ['loss', 'clip_loss', 'ssl_loss', 'clip_acc', 'ssl_acc']
    elif model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    else:
        return ['loss', 'ssl_loss', 'ssl_acc']


@timm.models.registry.register_model
def vit_small_mocov3_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=12, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer('vit_small_patch16_224', **model_kwargs)

    return model


def CLIP_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = CLIP(embed_dim=512, vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=384, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def CLIP_VITB16(embed_dim=512, **kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = CLIP(embed_dim=embed_dim, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=768, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def CLIP_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = CLIP(embed_dim=512, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=1024, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model
