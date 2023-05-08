import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from distributed import (
    get_world_size,
    get_rank,
    is_master,
    is_xla,
    broadcast_xla_master_model_param,
    infer_init_method,
    distributed_init,
    master_print,
    synchronize,
    reduce_tensor,
    save_ckpt,
    load_ckpt,
    load_text_model_ckpt,
    gather_tensor_with_backward
)
import slip_models
####
import transformers
from tqdm import tqdm
# import clip
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from aesthetic_model import AestheticScorer

# from diffusers.loaders import AttnProcsLayers

class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            
            # print(state_dict)
            # print(module.mapping)
            for key, value in state_dict.items():
                # print(key)
                if 'lora_layers' in key: # new
                    num = int(key.split(".")[3 if 'module' in key else 2])  # start is now "lora_layers.layers"
                    # num = int(key.split(".")[1])  # 0 is always "layers"
                    new_key = key.replace(f"lora_layers.layers.{num}", "unet." + module.mapping[num])
                    # new_key = key.replace(f"layers.{num}", module.mapping[num])
                else:
                    new_key = key
                new_state_dict[new_key] = value

            return new_state_dict

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                if 'processor' not in key: continue
                replace_key = key.split(".processor")[0] + ".processor"
                # mine has extra "unet" prepended from what this wants
                replace_key = replace_key.replace("unet.", "")
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class CLIPWeighting(nn.Module):
    """
    Reward from CLIP
    """
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = transformers.CLIPModel.from_pretrained(clip_model_name)
        self.processor = transformers.AutoProcessor.from_pretrained(clip_model_name)
        #### continue, see shell on L ###

    def forward(self, inputs): # img, txt):
        img = inputs['img']
        txt = inputs['txt']
        # img = inputs['img']
        # inputs are normalized to [-1, 1]
        img = 0.5 * (img + 1)
        
        # clip_inputs = self.processor(inputs['txt'], images=img, return_tensors='pt', padding=True)
        clip_inputs = self.processor(txt, images=img, return_tensors='pt', padding=True)
        # These are the cosine sim * 100
        # sims = self.model(**clip_inputs).logits_per_image.diag()
        sims = self.model(input_ids=clip_inputs['input_ids'].to(self.model.device),
                         pixel_values=clip_inputs['pixel_values'].to(self.model.device),
                         attention_mask=clip_inputs['attention_mask'].to(self.model.device)).logits_per_image.diag()

        return sims


class AestheticWeighting(nn.Module):
    """
    Reward from aesthetic model
    """
    def __init__(self, aes_model_name):
        # format of aes_model_name is aes-vit_l_14 / aes-vit_b_32
        super().__init__()
        self.aes_model = AestheticScorer(aes_model_name.split('-')[1])

    def forward(self, inputs):
        # scorer expects things in -1 to 1 like it's getting handed
        return self.aes_model(inputs['img'])

    

def create_sd_model(weighting_model_name='', num_chain_timesteps=1, **kwargs):
    if weighting_model_name:
        if 'clip' in weighting_model_name:
            weighting_module = CLIPWeighting(clip_model_name=weighting_model_name)
        elif 'aes' in weighting_model_name:
            weighting_module = AestheticWeighting(aes_model_name=weighting_model_name)
        else:
            raise NotImplementedError
    else:
        weighting_module = None

    if num_chain_timesteps > 1:
        return FullChainSDModel(weighting_module=weighting_module,
                                num_timesteps=num_chain_timesteps,
                                **kwargs)
    else:
        return SDModel(weighting_module=weighting_module, **kwargs)

class SDModel(nn.Module):
    """
    This is very similar to diffusers StableDiffusionPipeline
    but more minimal and modular
    """
    def __init__(self,
                model_config_name="CompVis/stable-diffusion-v1-4",
                latent_scale = 0.18215,
                pretrained_unet=False,
                 lora=False,
                 cond_dropout=0.1,
                 weighting_module=None, # For reward training
                ):
        super().__init__()
        
        # print("TODO: Freeze VAE + Text encoder")
        # print("TODO: Does Unet need better init?") # seems to converge OK
        self.vae = AutoencoderKL.from_pretrained(model_config_name, 
                                            subfolder="vae")
        self.text_encoder = slip_models.CLIPTextPretrained()
        self.weighting_module = weighting_module
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.weighting_module is not None:
            self.weighting_module.requires_grad_(False)

        if pretrained_unet:
            self.unet = UNet2DConditionModel.from_pretrained(model_config_name,
                                                             subfolder='unet')
        else:
            unet_cfg = UNet2DConditionModel.load_config(model_config_name,
                                                        subfolder="unet") 
            self.unet = UNet2DConditionModel.from_config(unet_cfg)
        if lora:
            self.unet.requires_grad_(False)
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRACrossAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )   

            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        # Switched to DDIM scheduler. The add_noise and num_train_timesteps are the same
        # so only difference is scheduler stepping (which we always DDIM for)
        self.scheduler = DDIMScheduler.from_pretrained(model_config_name,
                                              subfolder='scheduler')
        self.latent_scale = latent_scale

        # Not needed for training but for generation
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
                model_config_name,
            subfolder="tokenizer",
            revision=None
            )
        self.cond_dropout = cond_dropout
        
        # slow startup on cpu but not sure how to handle on DDP
        self.initialize_uncond_hidden_states()
        
    def initialize_uncond_hidden_states(self):
        
        with torch.no_grad():
            uncond_tokenized_inputs = self.tokenizer(
                                          '',
                                          padding="max_length",
                                          max_length=self.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors='pt'
            ).input_ids

            self.encoder_hidden_states_UC = self.text_encoder(uncond_tokenized_inputs).detach()
        
    def train(self, mode=True):
        super(SDModel, self).train(mode=mode)
        self.vae.eval()
        self.text_encoder.eval()
        if self.weighting_module is not None:
            self.weighting_module.eval()

    def weighting(self, input_dict, normalization='0_to_1'):
        # input will have 'img' 'txt' keys
        if self.weighting_module is None:
            return torch.ones(input_dict['img'].size(0),
                            device=input_dict['img'].device)
        weights = self.weighting_module(input_dict).detach()
        all_weights = gather_tensor_with_backward(weights)
        if normalization == 'sum': # keep same total magnitude w/ linear wieghting
            weights = weights * (all_weights.size(0) / all_weights.sum())
        elif normalization == '0_to_1': # keep same total magnitude w/ linear wieghting
            # minimum is 0
            min_weight = all_weights.min()
            global_bs = all_weights.size(0)
            weights = weights - min_weight
            # Make so sum is still same
            weights_that_sum_to_1_globally = weights  / (all_weights.sum() - global_bs*min_weight)
            # can test # print(gather_tensor_with_backward(weights_that_sum_to_1_globally).sum())
            # get global to be the same
            weights = weights_that_sum_to_1_globally * global_bs
        else:
            raise NotImplementedError
        return weights

    def forward(self, img, txt, timesteps=None, print_unweighted_loss=False):
        # Maybe could check if above is initialized but avoiding if/else
        # print(img)
        # print(txt)
        # print(img.device)
        with torch.no_grad():
            # loss_weights = self.weighting_module(img, txt)
            loss_weights = self.weighting({"img":img, "txt":txt})
            tokenized_txt = self.tokenizer(
                                          txt, # ['asdf']*4,
                                          padding="max_length",
                                          max_length=self.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors='pt').input_ids.to(self.unet.device)
            encoder_hidden_states = self.text_encoder(tokenized_txt)
            # Do dropout to null conditioning
            lbs = encoder_hidden_states.size(0)
            mask = (torch.rand((lbs, 1, 1),
                               device=encoder_hidden_states.device) > self.cond_dropout)

            _, l, d = encoder_hidden_states.size()
            mask = mask.repeat(1, l, d)

            uncond_hidden_states = self.encoder_hidden_states_UC.to(encoder_hidden_states.device).repeat(lbs, 1, 1).detach()
            encoder_hidden_states = torch.where(mask, 
                                                encoder_hidden_states,
                                                uncond_hidden_states)
            latents = self.vae.encode(img).latent_dist.sample() * self.latent_scale

            noise = torch.randn_like(latents)
            if timesteps is None:
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps,
                                          (noise.size(0),),
                                          device=latents.device)
                timesteps = timesteps.long()
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents,
                          timesteps, 
                         encoder_hidden_states=encoder_hidden_states).sample

        per_element_mses = (noise - noise_pred) ** 2
        loss = (loss_weights.view(-1, 1, 1, 1) * per_element_mses).mean() # equivalent to below but admits weighting
        
        if print_unweighted_loss and is_master():
            assert not is_xla() # would slow down too much
            with torch.no_grad():
                unweighted_loss = per_element_mses.mean()
                print(f"Unweighted Loss: {unweighted_loss.item():.3f} / Weighted Loss: {loss.item():.3f}")
                # print(f"Unweighted Loss: {reduce_tensor(unweighted_loss, average=True):.3f} / Weighted Loss: {reduce_tensor(loss, average=True):.3f}")
        return loss

    
    def generate(self, prompt, batch_size=1,
                h=512, w=512, T=50, gs=7.5, seed=0,
                silent=False):
        torch.manual_seed(seed)
        device = self.unet.device
        # modeling off of https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L604
        
        
        # Could condense
        cond_tokenized_inputs = self.tokenizer(
                                      prompt,
                                      padding="max_length",
                                      max_length=self.tokenizer.model_max_length,
                                      truncation=True,
                                      return_tensors='pt').input_ids.to(device)
        encoder_hidden_states_C = self.text_encoder(cond_tokenized_inputs)
        
        uncond_tokenized_inputs = self.tokenizer(
                                      '',
                                      padding="max_length",
                                      max_length=self.tokenizer.model_max_length,
                                      truncation=True,
                                      return_tensors='pt').input_ids.to(device)
        encoder_hidden_states_UC = self.text_encoder(uncond_tokenized_inputs)
        
        encoder_hidden_states = torch.cat([encoder_hidden_states_UC,
                                           encoder_hidden_states_C])
        
        # Prepare timesteps
        self.scheduler.set_timesteps(T, device=device)
        timesteps = self.scheduler.timesteps
        
        # Latents
        # Hardcoding scale factor for now
        latents = torch.randn((batch_size, 4, h//8, w//8),
                              device=device, dtype=self.unet.dtype)
        latents = latents * self.scheduler.init_noise_sigma # 1 for DDPM
        
        # Denoising
        for i,t in tqdm(enumerate(timesteps), disable=silent):
            latent_model_input = torch.cat([latents] * 2) # clf-free guidance
            # No change below for DDPM scheduler
            latent_model_input = self.scheduler.scale_model_input(
                                        latent_model_input,
                                         t
            )
            
            noise_pred = self.unet(latent_model_input,
                                   t,
                                   encoder_hidden_states=encoder_hidden_states
                                  ).sample
            
            noise_pred_UC, noise_pred_C = noise_pred.chunk(2)
            noise_pred = noise_pred_UC + gs * (noise_pred_C - noise_pred_UC)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # decode
        latents = latents / self.latent_scale
        image = self.vae.decode(latents).sample
        image = (0.5 * image + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        return image
                                   
            
class FullChainSDModel(SDModel):
    """
    Train multi-diffusion-step objective

    Could maybe subsume into above by renaming forward function and in init setting self.forward?
    """

    def __init__(self, num_timesteps, **kwargs):
        super().__init__(**kwargs)
        self.num_timesteps = num_timesteps
        self.scheduler.config.steps_offset = (self.scheduler.num_train_timesteps // self.num_timesteps) - 1
        self.scheduler.set_timesteps(num_timesteps)

    def forward(self, img, txt):
        with torch.no_grad():
            loss_weights = self.weighting({"img":img, "txt":txt})

            tokenized_txt = self.tokenizer(
                                          txt, 
                                          padding="max_length",
                                          max_length=self.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors='pt').input_ids.to(self.unet.device)
            encoder_hidden_states = self.text_encoder(tokenized_txt)
            # Do dropout to null conditioning
            lbs = encoder_hidden_states.size(0)
            mask = (torch.rand((lbs, 1, 1),
                               device=encoder_hidden_states.device) > self.cond_dropout)

            _, l, d = encoder_hidden_states.size()
            mask = mask.repeat(1, l, d)

            uncond_hidden_states = self.encoder_hidden_states_UC.to(encoder_hidden_states.device).repeat(lbs, 1, 1).detach()
            encoder_hidden_states = torch.where(mask, 
                                                encoder_hidden_states,
                                                uncond_hidden_states)
            latents = self.vae.encode(img).latent_dist.sample() * self.latent_scale

            noise = torch.randn_like(latents)

            # these are 1 2 ... self.num_timesteps
            timestep_idx = torch.randint(1, self.num_timesteps+1,
                                      (noise.size(0),),
                                      device=latents.device)
            # For 4 timesteps, orig would be 1 2 3 4
            # below should be 249 499 749 999
            timesteps = (timestep_idx * self.scheduler.num_train_timesteps / self.num_timesteps) - 1
            timesteps = timesteps.long()
            # Will denoise varying amounts
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)


        # Below was originally single noise pred
        """
        noise_pred = self.unet(noisy_latents,
                          timesteps, 
                         encoder_hidden_states=encoder_hidden_states).sample
        per_element_mses = (noise - noise_pred) ** 2
        """
        # Now want to do iteratively
        # And target will be relative to original latents
        
        for denoising_i in range(self.num_timesteps):
            # predict based off of current timesteps
            noise_pred = self.unet(noisy_latents,
                              timesteps, 
                             encoder_hidden_states=encoder_hidden_states).sample
            # step based off of current timesteps
            # if timestep is <=0 should just ignore
            noisy_latents = batch_ddim_step(self.scheduler, noise_pred, timesteps, noisy_latents)
            timesteps = timesteps -  self.scheduler.num_train_timesteps // self.num_timesteps

        per_element_mses = (noisy_latents - latents) ** 2
        loss = (loss_weights.view(-1, 1, 1, 1) * per_element_mses).mean() # equivalent to below but admits weighting
        
        return loss


def batch_ddim_step(scheduler, model_output, timesteps, sample,
                    eta=0.0,
                    use_clipped_model_output=False):
    # https://github.com/huggingface/diffusers/blob/v0.8.0/src/diffusers/schedulers/scheduling_ddim.py#L197
    if scheduler.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    # 1 get pre step value
    prev_timesteps = timesteps - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2 alpha/betas
    # below abs() is hack to make negative numbers not error since they just get replaced
    alpha_prod_t = scheduler.alphas_cumprod.index_select(0, timesteps.abs())
    # make broadcastable
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    # get previous
    # below abs() is hack to make negative numbers not error since they just get replaced
    # will make -1 1 etc
    alpha_prod_t_prev = scheduler.alphas_cumprod.index_select(0, prev_timesteps.abs())
    alpha_prod_t_prev = torch.where(prev_timesteps>=0, alpha_prod_t_prev, scheduler.final_alpha_cumprod * torch.ones_like(alpha_prod_t_prev) )
    alpha_prod_t_prev = alpha_prod_t_prev.view_as(alpha_prod_t)
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 3 predict orig sample
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 4 clip prediction
    if scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-1, 1)

    # 5 compute variance
    # originally https://github.com/huggingface/diffusers/blob/v0.8.0/src/diffusers/schedulers/scheduling_ddim.py#L171
    # all have same shape
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * (variance ** 0.5)
    if use_clipped_model_output:
        model_output = (sample - (alpha_prod_t**0.5) * pred_original_sample) / (beta_prod_t ** 0.5)

    # 6 compute "direction pointing to x_t"
    pred_sample_direction = ( (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5) * model_output
    # 7 compute x_t 
    prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original_sample + pred_sample_direction

    if eta > 0:
        # https://github.com/huggingface/diffusers/blob/v0.8.0/src/diffusers/schedulers/scheduling_ddim.py#L283
        raise NotImplementedError

    # Want to return prev_sample where timesteps>0 otherwise sample
    update_mask = (timesteps > 0)
    update_mask = update_mask.view(-1, 1, 1, 1).repeat(1, *sample.size()[1:])

    return torch.where(update_mask, prev_sample, sample)


