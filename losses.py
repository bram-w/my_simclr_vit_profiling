import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed import gather_tensor_with_backward, get_rank, get_world_size, master_print, reduce_sum_with_backward, xla_all_reduce_sum_with_backward


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class CLIPLoss(nn.Module):
    def __init__(self, use_image_unif_loss=False, use_text_unif_loss=False,
                  unif_scale=0.1, num_normalization_groupings=0,
                  expert_loss=False):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.use_image_unif_loss = use_image_unif_loss
        self.use_text_unif_loss = use_text_unif_loss
        self.unif_scale = unif_scale
        self.num_normalization_groupings = num_normalization_groupings
        if expert_loss:
            assert num_normalization_groupings
        self.expert_loss = expert_loss

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.identity_idx = torch.arange(local_batch_size, device=image_embed.device)
            self.labels = local_batch_size * get_rank() + self.identity_idx
            self.last_local_batch_size = local_batch_size

        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)


        image_embed_all = gather_tensor_with_backward(image_embed)
        text_embed_all = gather_tensor_with_backward(text_embed)
        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()
    
        clip_loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2
 
        return clip_loss


