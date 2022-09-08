import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed import gather_tensor_with_backward, get_rank, get_world_size, master_print, reduce_sum_with_backward, xla_all_reduce_sum_with_backward


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, global_bs, lambd=0.005):
        super().__init__()
        self.global_bs = global_bs
        self.lambd = lambd

    def bn(self, z, eps=1e-5):
        # return (z - z.mean(0)) / z.std(0)
        local_mean = z.mean(0)
        local_sqr_mean = (z*z).mean(0)

        # global_mean = xla_all_reduce_sum_with_backward(local_mean) / self.global_bs
        # global_sqr_mean = xla_all_reduce_sum_with_backward(local_sqr_mean) / self.global_bs
        global_mean = reduce_sum_with_backward(local_mean) / self.global_bs
        global_sqr_mean = reduce_sum_with_backward(local_sqr_mean) / self.global_bs

        global_var = global_sqr_mean - global_mean.pow(2)

        return (z - global_mean) / torch.sqrt(global_var + eps)


    def forward(self, outputs):

        z1 = outputs['image_embed']
        z2 = outputs['text_embed']
        # empirical cross-correlation matrix
        c = (self.bn(z1).T @ self.bn(z2)) / self.global_bs

        # sum the cross-correlation matrix between all gpus
        # xla_all_reduce_sum_with_backward(c)
        reduce_sum_with_backward(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class IsolaCLIPLoss(nn.Module):
    def __init__(self, use_image_unif_loss=True, use_text_unif_loss=True,
                 align_scale=3, unif_scale=1):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.use_image_unif_loss = use_image_unif_loss
        self.use_text_unif_loss = use_text_unif_loss
        self.align_scale = align_scale
        self.unif_scale = unif_scale

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)


        align_loss = (image_embed - text_embed).norm(p=2, dim=1).pow(2).mean()

        # gather features from all GPUs
        # image_embed_all, text_embed_all = \
        #     utils.all_gather_batch([image_embed, text_embed])
        image_embed_all = gather_tensor_with_backward(image_embed)
        text_embed_all = gather_tensor_with_backward(text_embed)

        # Just doing uniformity loss on each device. Both are means so should be ok this way, but
        # may need reweighting

        # blocking/indenting this for legibility
        if self.use_text_unif_loss:
            text_unif_loss = two_arr_pdist(text_embed, text_embed_all, p=2).pow(2).mul(-2).exp().mean().log()
        else:
            text_unif_loss = 0

        if self.use_image_unif_loss:
            image_unif_loss = two_arr_pdist(image_embed, image_embed_all, p=2).pow(2).mul(-2).exp().mean().log()
        else:
            image_unif_loss = 0

        unif_loss_divisor = int(self.use_image_unif_loss) + int(self.use_text_unif_loss)
        unif_loss = (text_unif_loss + image_unif_loss) / unif_loss_divisor if unif_loss_divisor else 0
 
        # coefficient was optimal in orig. work       
        loss = self.align_scale * align_loss + self.unif_scale * unif_loss
        return loss

def two_arr_pdist(a, b, p=2):
    # base taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html
    return (a[:, None] - b).norm(dim=2, p=p).flatten()




class CLIPLoss(nn.Module):
    def __init__(self, use_image_unif_loss=False, use_text_unif_loss=False,
                  unif_scale=0.1, num_normalization_groupings=0,
                  expert_loss=False, single_text_target=False,
                  group_t=10):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.use_image_unif_loss = use_image_unif_loss
        self.use_text_unif_loss = use_text_unif_loss
        self.unif_scale = unif_scale
        self.num_normalization_groupings = num_normalization_groupings
        if expert_loss:
            assert num_normalization_groupings
        self.single_text_target = single_text_target
        self.expert_loss = expert_loss
        self.group_t = group_t

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.identity_idx = torch.arange(local_batch_size, device=image_embed.device)
            self.labels = local_batch_size * get_rank() + self.identity_idx
            self.last_local_batch_size = local_batch_size
            self.group_id_mat = torch.eye(self.num_normalization_groupings, device=image_embed.device)
            self.top_k = 5

        # normalized features

        if self.num_normalization_groupings:
            image_embed = image_embed.view(local_batch_size,
                                            self.num_normalization_groupings,
                                            -1)
            if self.single_text_target:
                # Hacky, but just going to take first d dimensions of text embed so don't have to edit CLIP
                # architecture
                expert_d = image_embed.shape[-1]
                text_embed = text_embed[:,:expert_d].unsqueeze(1).repeat(1, self.num_normalization_groupings, 1)
            else:
                text_embed = text_embed.view(local_batch_size,
                                                self.num_normalization_groupings,
                                                -1)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)


        if self.expert_loss:
            image_embed_all = gather_tensor_with_backward(image_embed)
            text_embed_all = gather_tensor_with_backward(text_embed)
            # raise NotImplementedError # this thing still loss collapses
            # Have num_normalization_groups so
            # local embed is LBS x G x D
            # global embed is BS x G x D
            image_group_sims = (image_embed.unsqueeze(1) * text_embed_all).sum(dim=-1)
            text_group_sims = (text_embed.unsqueeze(1) * image_embed_all).sum(dim=-1)
            # print(image_group_sims.min(), image_group_sims.max())
            # now both are LBS x BS x G
            # want to weight *_group_sims[torch.arange(LBS), labels]
            # want to put more weight on largest value

            # image_group_sims[self.identity_idx, self.labels].pow_(2)
            # image_group_sims[self.identity_idx, self.labels].div_(image_group_sims[self.identity_idx, self.labels].sum())
            # text_group_sims[self.identity_idx, self.labels].pow_(2)
            # text_group_sims[self.identity_idx, self.labels].div_(text_group_sims[self.identity_idx, self.labels].sum())

            # current_image_sum = image_group_sims[self.identity_idx, self.labels].sum()
            # A scalar below matters a lot. Doing the previous sum results in tiny tiny losses, 
            # doing nothing results in quite large losses
            # ideally want loss in the 1-6 range or so starting out, where it's beating chance but has a lot to learn
            """
            print(image_group_sims[self.identity_idx, self.labels].sum())
            image_group_sims[self.identity_idx, self.labels] *= image_group_sims[self.identity_idx,
                                                                                    self.labels].add(1).softmax(-1).detach()
            print(image_group_sims[self.identity_idx, self.labels].sum())
            # current_text_sum = text_group_sims[self.identity_idx, self.labels].sum()
            text_group_sims[self.identity_idx, self.labels] *= text_group_sims[self.identity_idx,
                                                                                self.labels].add(1).softmax(-1).detach()

            """
            # train full independent, index select the above sims by some metric (most accurate? best on diag?)
            # Don't want to just do highest
            # full on calc softmax for each and then select base off of that
            # Still might need balancing but that can be next step
            # THINK CAREFULLY ON HOW WNT TO BALANCE SPECIALIZATION WITH DODING COLLAPSE
            """
            THIS IS CARRYING BATCH SIZE WITH IT
            print(image_group_sims[self.identity_idx, self.labels].shape)
            print(image_group_sims[self.identity_idx, self.labels], text_group_sims[self.identity_idx, self.labels])
            print(image_group_sims[self.identity_idx, self.labels].max(), text_group_sims[self.identity_idx, self.labels].max())
            image_group_sims[self.identity_idx, self.labels] = image_group_sims[self.identity_idx, self.labels] - image_group_sims[self.identity_idx, self.labels] + image_group_sims[self.identity_idx, self.labels].max()
            text_group_sims[self.identity_idx, self.labels] = text_group_sims[self.identity_idx, self.labels] - text_group_sims[self.identity_idx, self.labels] + text_group_sims[self.identity_idx, self.labels].max()
            logits_per_image = logit_scale * image_group_sims.sum(-1)
            logits_per_text = logit_scale * text_group_sims.sum(-1)
            """
            image_probs = image_group_sims.softmax(dim=1) # this could be unstable?
            cropped_image_probs = image_probs.index_select(1, self.labels)
            # correct_image_probs = cropped_image_probs.diagonal(0, 0, 1).t()
            correct_image_probs = cropped_image_probs.flatten(0,1).index_select(0, (local_batch_size+1)*self.identity_idx)

            # correct_image_probs = cropped_image_probs[0]
            # This is now LBS x groups
            text_probs = text_group_sims.softmax(dim=1)
            # This should be ok b/c same every time?
            cropped_text_probs = text_probs.index_select(1, self.labels)
            correct_text_probs = cropped_text_probs.flatten(0,1).index_select(0, (local_batch_size+1)*self.identity_idx)


            # if these predictions are uniform than all of them
            # here are 
            BS = image_embed_all.size(0)
            noise_scale = 0.1 * (1 / BS)
            # doing same noise for image and text
            noise = noise_scale * torch.rand_like(correct_image_probs)
            correct_image_probs += noise
            correct_text_probs += noise
            # correct_text_probs = cropped_text_probs[0]
            # correct_text_probs = cropped_text_probs.diagonal(0, 0, 1).t()
            # This is now LBS x groups
            # could just softmax, worried that would make all train equally, want some amount of explicit in here
            # top k would give LBS x k
            # flatten and then selections are LBS*k x g where each is onehot
            # view as LBS x k x g and sum along 1 (keeping dim) yielding LBS x 1 x G

            ####### PREV TOP 1 VERSION ######
            # best_image_models = correct_image_probs.argmax(1)
            # best_text_models = correct_text_probs.argmax(1)
            # image_selections = self.group_id_mat.index_select(0, best_image_models).unsqueeze(1) # selections are LBS x 1 x G now
            # text_selections = self.group_id_mat.index_select(0, best_text_models).unsqueeze(1) # selections are LBS x 1 x G now
            ###############

            """
            ##### TRYING NEW TOPk ######
            topk_image_models = correct_image_probs.topk(self.top_k, 1).indices.flatten() # this will be LBS k vecs catted
            image_selections = self.group_id_mat.index_select(0, topk_image_models) # this is LBS*k x G 
            image_selections = image_selections.view(local_batch_size, self.top_k,
                                                        self.num_normalization_groupings).sum(1, keepdim=True)
            topk_text_models = correct_text_probs.topk(self.top_k, 1).indices.flatten() # this will be LBS k vecs catted
            text_selections = self.group_id_mat.index_select(0, topk_text_models) # this is LBS*k x G 
            text_selections = text_selections.view(local_batch_size, self.top_k,
                                                        self.num_normalization_groupings).sum(1, keepdim=True)
            logits_per_image = logit_scale * (image_group_sims * image_selections).sum(-1)
            logits_per_text = logit_scale * (text_group_sims * text_selections).sum(-1)
            #########
            """

            # text_group_sims is LBS x BS x G
            # correct probs is LBS x G
            # Ultimately going to take Xent of LBS x BS
            # want weighted sum contributions 
            # Run loss separately on each model but don't reduce until take average?
            """
            So make sims LBS x G X BS , correct probsl LBS x G
            View both as LBS*G (x BS for sims)
            Take Xent loss without reduction to get LBS*G
                    Against repeated version of targets
            Normal reduction is mean, I think that's still fine *IF YOU SOFTMAX CORRECT PROBS*
            So BEFORE flattening correct probs take osftmax along last dim and then mult by g
            Then can just proudct and mean ave vecootors
            """
            correct_image_probs = self.num_normalization_groupings * (correct_image_probs*self.group_t).softmax(dim=-1)
            correct_text_probs = self.num_normalization_groupings * (correct_text_probs*self.group_t).softmax(dim=-1)

            # print(correct_text_probs, correct_image_probs)
            # print(image_group_sims.shape, image_group_sims.swapaxes(2, 1).shape, local_batch_size,
            #         self.num_normalization_groupings)
            # image_logits = image_group_sims.swapaxes(2, 1).view(local_batch_size*self.num_normalization_groupings,
            #                                                         -1)
            image_logits = image_group_sims.swapaxes(2, 1).flatten(start_dim=0, end_dim=1) # LBS*G x BS
            image_cross_entropy = F.cross_entropy(logit_scale * image_logits,
                                                    self.labels.repeat_interleave(self.num_normalization_groupings),
                                                    reduction='none')
            image_weighted_cross_entropy = image_cross_entropy *  correct_image_probs.flatten()
            image_loss = image_weighted_cross_entropy.mean()

            # text_logits = text_group_sims.swapaxes(2, 1).view(local_batch_size*self.num_normalization_groupings,
            #                                                         -1)
            text_logits = text_group_sims.swapaxes(2, 1).flatten(start_dim=0, end_dim=1)
            text_cross_entropy = F.cross_entropy(logit_scale * text_logits,
                                                    self.labels.repeat_interleave(self.num_normalization_groupings),
                                                    reduction='none')
            text_weighted_cross_entropy = text_cross_entropy * correct_text_probs.flatten()
            text_loss = text_weighted_cross_entropy.mean()

            clip_loss = (image_loss + text_loss) / 2

            """
            total_model_image_accs = correct_image_probs.mean(0)
            image_entropy = (-1 * total_model_image_accs * total_model_image_accs.log() ).mean()
            total_model_text_accs = correct_text_probs.mean(0)
            text_entropy = (-1 * total_model_text_accs * total_model_text_accs.log() ).mean()
            negative_entropy_loss = -1 *(image_entropy + text_entropy) / 2
            """
            negative_entropy_loss = 0
            # Now want to sample as I thought was [self.identity_idx, self.labels] but it's actually fancy
            # Maybe masked selct (also b/c have fixed selection for each module)
            # once I do that should have LBS x g
            # Then get the argmax across last dimension
            # then index select image _Group sims from that *AND HIT WITH LOGIT SCALE*
        else:
            # the below line doesn't do anything in normal case but flattens
            # in multimodel case
            image_embed = image_embed.view(local_batch_size, -1)
            text_embed = text_embed.view(local_batch_size, -1)
            image_embed_all = gather_tensor_with_backward(image_embed)
            text_embed_all = gather_tensor_with_backward(text_embed)
            # cosine similarity as logits
            logits_per_image = logit_scale * image_embed @ text_embed_all.t()
            logits_per_text = logit_scale * text_embed @ image_embed_all.t()
            negative_entropy_loss = 0
        
            clip_loss = (F.cross_entropy(logits_per_image, self.labels) + \
                F.cross_entropy(logits_per_text, self.labels)) / 2
        # print(clip_loss)
        # blocking/indenting this for legibility
        if self.use_text_unif_loss:
            text_unif_loss = two_arr_pdist(text_embed, text_embed_all, p=2).pow(2).mul(-2).exp().mean().log()
        else:
            text_unif_loss = 0

        if self.use_image_unif_loss:
            image_unif_loss = two_arr_pdist(image_embed, image_embed_all, p=2).pow(2).mul(-2).exp().mean().log()
        else:
            image_unif_loss = 0
        unif_loss_divisor = int(self.use_image_unif_loss) + int(self.use_text_unif_loss)
        unif_loss = (text_unif_loss + image_unif_loss) / unif_loss_divisor if unif_loss_divisor else 0
 
        # coefficient was optimal in orig. work       
        loss = clip_loss + self.unif_scale * unif_loss + negative_entropy_loss

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
