import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed import gather_tensor_with_backward, get_rank, get_world_size, master_print

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
        

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2
        # nan_in_logits = torch.any(torch.isnan(logits_per_image + logits_per_text))
        # master_print(f"Logits {nan_in_logits} loss {torch.isnan(loss)}")
        # master_print(f"Image Logits min/max {logits_per_image.min()} {logits_per_image.max()}")
        # master_print(f"Text Logits min/max {logits_per_text.min()} {logits_per_text.max()}")
        # master_print(f"""Logits shape {logits_per_image.shape},
        # {logits_per_text.shape} // Labels min/max {self.labels.min()}
        # {self.labels.max()} // Nan in label?
        # {torch.any(torch.isnan(self.labels.float()))}""")
        # output_from_zero_logits = F.cross_entropy(torch.zeros(*logits_per_image.shape).to(self.labels.device), self.labels)
        # master_print(f"Output from zero logits {output_from_zero_logits}")

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


class ExperimentalClusteringLoss(nn.Module):
    """
    This is adaptation of SimCLR loss from the google slide deck

    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py

    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, n_c, d, temperature=0.1):
        # expect input to be (2*bs, mc*d)
        super().__init__()
        self.tau = temperature
        self.n_c = n_c # num clusterings
        self.d = d # num clusters in clustering
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, embeddings):
        # Not normalizing embeddings, instead softmax the cluster assignments
        # with softmax after reshape
        # embeddings = F.normalize(embeddings, dim=-1, p=2)

        if torch.isnan(embeddings).any().item():
            print("Encountered NaN in embeddings")
            asdf


        local_batch_size = embeddings.size(0) // 2
        embedding_dim = embeddings.size(1)

        embeddings_reshape = embeddings.view(2, local_batch_size, self.n_c, self.d)
        # want softmax over last dim, choose where you belong!
        embeddings_reshape = embeddings_reshape.softmax(dim=-1)
        # print(embeddings_reshape.min(), embeddings_reshape.max(), embeddings_reshape.mean(), embeddings_reshape.std())

        # below are lbs x n_c x d
        # can go ahead and flatten tensor here, want lbs x (n_c * d)
        # sim loss is fine iwth this too
        # needed to do first reshape to get softmax dim out though
        q_a = embeddings_reshape[0]
        q_b = embeddings_reshape[1]

        # first loss is just local self-consistency
        # all positive because it's probabilities!
        # below has fixed sum due to softmax, minimize loss when sum is
        # maximized when all the mass is in the same locations
        self_sim_loss = -1 * (q_a * q_b).sum() / (local_batch_size * self.n_c)
        
        """
        q_a = embeddings_reshape[0].flatten(start_dim=1)
        q_b = embeddings_reshape[1].flatten(start_dim=1)
        # Pretty sure just sharding this is fine
        # b/c doing from perpsectives that CLUSTERS should be diferent instead
        # of BATCHES being different. Complexity works out the same
        cluster_sim_mat_a = (q_a @ q_a.t())
        cluster_sim_mat_b = (q_b @ q_b.t())

        cluster_sim_mat_a = cluster_sim_mat_a - torch.diag(torch.diag(cluster_sim_mat_a))
        cluster_sim_mat_b = cluster_sim_mat_b - torch.diag(torch.diag(cluster_sim_mat_b))

        cluster_sim_a_loss = (cluster_sim_mat_a).sum() # .mean()
        cluster_sim_b_loss = (cluster_sim_mat_b).sum()  # .mean()
        # results are (m_c*d x m_c*d)
        # want to minimize this summ
        cluster_loss = (cluster_sim_a_loss + cluster_sim_b_loss) / 2
        """
        # first is n_c * d * lbs
        # second is n_c * lbs * d
        # product would be d*d with lbs producted away, I think this works with sharding again
        # Squaring to encourage sharp
        q_a = q_a.pow(2)
        q_b = q_b.pow(2)

        mat_1_a = q_a.permute(1, 2, 0).repeat(self.n_c, 1, 1)
        mat_2_a = q_a.permute(1, 0, 2).repeat_interleave(self.n_c, dim=0)
        mat_1_b = q_b.permute(1, 2, 0).repeat(self.n_c, 1, 1)
        mat_2_b = q_b.permute(1, 0, 2).repeat_interleave(self.n_c, dim=0)

        # want to zero digonal or else that term prevents sharp decisions
        # THIS IS WRONG ZEROING WANT TO JUST ELIMAINTE MATRICES OF SAME CLSTERING
        # So where indices of repeat and repeat_interleave match 0-0 1-1 2-2 etc
        # of form self.n_c*i + i = i * (self.n_c + 1)
        mat_products_a = torch.bmm(mat_1_a, mat_2_a)
        mat_products_b = torch.bmm(mat_1_b, mat_2_b)
        idx = torch.arange(self.n_c) * (self.n_c + 1)
        mask = torch.ones(mat_products_a.shape[0], dtype=torch.bool)
        mask[idx] = False
        mat_products_a = mat_products_a[mask]
        mat_products_b = mat_products_b[mask]
        """
        incorrect zeroing of literal diagonal
        idx = torch.arange(self.d)
        mat_products_a[:, idx, idx] *= 0
        mat_products_b[:, idx, idx] *= 0
        """

        """
        cluster_loss_a = mat_products_a.norm(p='fro', dim=(1,2)).mean()
        cluster_loss_b = mat_products_b.norm(p='fro', dim=(1,2)).mean()
        cluster_loss = (cluster_loss_a + cluster_loss_b) / 2
        """
        # matr products is (n_c ^ 2 - n_c) * d * d
        # taking root then summing (could also square, doing this for now b/c easier math)
        raw_cluster_loss_a = mat_products_a.pow(0.5).sum()
        raw_cluster_loss_b = mat_products_b.pow(0.5).sum()
        # my (current) math says would expect values
        # degen LBS^0.5
        # unif LBS^0.5
        # scattered (LBS^0.5) * d
        scale_factor = mat_products_a.shape[0] * (local_batch_size**0.5)
        cluster_loss_a = raw_cluster_loss_a / scale_factor
        cluster_loss_b = raw_cluster_loss_b / scale_factor
        cluster_loss = -1 * (cluster_loss_a + cluster_loss_b) / 2 

        loss = self_sim_loss + cluster_loss
        # master_print(self_sim_loss, cluster_loss)
        master_print(q_a.min(), q_a.max(), mat_products_a.min(), mat_products_a.max())
        if torch.isnan(loss).item():
            print("Encountered NaN in loss")
            asdf
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
