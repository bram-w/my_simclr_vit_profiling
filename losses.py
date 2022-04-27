import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed import gather_tensor_with_backward, get_rank, get_world_size

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        # image_embed_all, text_embed_all = \
        #     utils.all_gather_batch([image_embed, text_embed])
        image_embed_all = gather_tensor_with_backward(image_embed)
        text_embed_all = gather_tensor_with_backward(text_embed)

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()
        
        nan_in_logits = torch.any(torch.isnan(logits_per_image + logits_per_text))

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2
        master_print(f"Logits {nan_in_logits} loss {torch.isnan(loss)}")

        # compute accuracy
        # with torch.no_grad():
        #     pred = torch.argmax(logits_per_image, dim=-1)
        #     correct = pred.eq(self.labels).sum()
        #     acc = 100 * correct / local_batch_size

        # return {'loss': loss, 'clip_loss': loss, 'clip_acc': acc}
        return loss


class MY_UNIFINSHED_CLIPLoss(nn.Module):
    """

    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py

    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, embeddings_dict):
        # assume SLIP clip type output
        # {'image_embed': image_embed,
        #         'text_embed': text_embed,
        #         'logit_scale': self.logit_scale.exp()}
        # These would be bs x embed dim, unnormalized
        image_embed = embeddings_dict['image_embed']
        text_embed = embeddings_dict['text_embed']
        logit_scale = embeddings_dict['logit_scale']


        # embeddings = F.normalize(embeddings, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # local_batch_size = embeddings.size(0) // 2
        # embedding_dim = embeddings.size(1)
        local_batch_size = image_embed.size(0)
        embedding_dim  = image_embed.size(1)

        # embeddings_reshape = embeddings.view(2, local_batch_size, embedding_dim)
        # q_a = embeddings_reshape[0]
        # q_b = embeddings_reshape[1]
        q_a = image_embed
        q_b = text_embed


        k_a = gather_tensor_with_backward(q_a)
        k_b = gather_tensor_with_backward(q_b)
        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=embeddings.device
            )
            total_batch_size = local_batch_size * get_world_size()
            # self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        # logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        # logits_aa = logits_aa - self.masks
        # logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples
        return loss


class SimCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709

    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py

    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, embeddings):
        embeddings = F.normalize(embeddings, dim=-1, p=2)

        local_batch_size = embeddings.size(0) // 2
        embedding_dim = embeddings.size(1)

        embeddings_reshape = embeddings.view(2, local_batch_size, embedding_dim)
        q_a = embeddings_reshape[0]
        q_b = embeddings_reshape[1]
        k_a = gather_tensor_with_backward(q_a)
        k_b = gather_tensor_with_backward(q_b)
        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=embeddings.device
            )
            total_batch_size = local_batch_size * get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples
        return loss
