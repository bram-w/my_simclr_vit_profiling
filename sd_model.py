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
    load_text_model_ckpt
)
import slip_models
####
import transformers
from tqdm import tqdm
# import clip
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

from aesthetic_model import AestheticScorer

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

    

def create_sd_model(weighting_model_name, **kwargs):
    if weighting_model_name:
        if 'clip' in weighting_model_name:
            weighting_module = CLIPWeighting(clip_model_name=weighting_model_name)
        elif 'aes' in weighting_model_name:
            weighting_module = AestheticWeighting(aes_model_name=weighting_model_name)
        else:
            raise NotImplementedError
    else:
        weighting_module = None
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
        
        self.scheduler = DDPMScheduler.from_pretrained(model_config_name,
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

    def weighting(self, input_dict, normalization='default'):
        # input will have 'img' 'txt' keys
        if self.weighting_module is None:
            return torch.ones(input_dict['img'].size(0),
                            device=input_dict['img'].device)
        weights = self.weighting_module(input_dict).detach()
        if normalization == 'default': # keep same total magnitude w/ linear wieghting
            weights = weights * (weights.size(0) / weights.sum())
        else:
            raise NotImplementedError

        return weights

    def forward(self, img, txt, timesteps=None):
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
            # Not sure if scaling is right
            latents = self.vae.encode(img).latent_dist.sample() * self.latent_scale

            noise = torch.randn_like(latents)
            if timesteps is None:
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps,
                                          (noise.size(0),),
                                          device=latents.device)
                timesteps = timesteps.long()
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # print(txt, timesteps)
        

        noise_pred = self.unet(noisy_latents,
                          timesteps, 
                         encoder_hidden_states=encoder_hidden_states).sample

        # print(loss_weights.shape, noise.shape, noise_pred.shape)
        per_element_mses = (noise - noise_pred) ** 2
        loss = (loss_weights.view(-1, 1, 1, 1) * per_element_mses).mean() # equivalent to below but admits weighting
        
        with torch.no_grad():
            unweighted_loss = per_element_mses.mean()
            print(f"Unweighted Loss: {unweighted_loss.item()} / Weighted Loss: {loss.item()}")

        # loss = F.mse_loss(noise, noise_pred, reduction='mean') 
        return loss

    
    def generate(self, prompt, batch_size=1,
                h=512, w=512, T=50, gs=7.5):
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
        for i,t in tqdm(enumerate(timesteps)):
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
                                   
            
        
