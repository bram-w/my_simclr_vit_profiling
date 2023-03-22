import torch
from torch import nn
import torch.nn.functional as F
import torchvision
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
# import clip
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

class SDModel(nn.Module):
    def __init__(self,
                model_config_name="CompVis/stable-diffusion-v1-4"):
        super().__init__()
        # print("TODO: Freeze VAE + Text encoder")
        print("TODO: Does Unet need better init?")
        self.vae = AutoencoderKL.from_pretrained(model_config_name, 
                                            subfolder="vae")
        self.text_encoder = slip_models.CLIPTextPretrained()
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        unet_cfg = UNet2DConditionModel.load_config(model_config_name,
                                                    subfolder="unet") 
        self.unet = UNet2DConditionModel.from_config(unet_cfg)
        
        self.scheduler = DDPMScheduler.from_pretrained(model_config_name,
                                              subfolder='scheduler')
        
    def forward(self, img, txt):
        encoder_hidden_states = self.text_encoder(txt)
        # Not sure if scaling is right
        latents = self.vae.encode(img).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps,
                                  (noise.size(0),),
                                  device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)


        noise_pred = self.unet(noisy_latents,
                          timesteps, 
                         encoder_hidden_states=encoder_hidden_states).sample

        loss = F.mse_loss(noise, noise_pred, reduction='mean')
        return loss

