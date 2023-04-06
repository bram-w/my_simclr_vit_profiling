import sd_model
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler

scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder='scheduler')

scheduler.set_timesteps(4)

print(scheduler.alphas_cumprod.shape)
print(scheduler.timesteps)
scheduler.config.steps_offset = 249
scheduler.set_timesteps(4)
print(scheduler.timesteps)
sd_model.batch_ddim_step(scheduler, torch.randn(4, 3, 64, 64), scheduler.timesteps, torch.randn(4, 3, 64, 64))


sd_model.batch_ddim_step(scheduler, torch.randn(4, 3, 64, 64), -1 * torch.ones_like(scheduler.timesteps), torch.randn(4, 3, 64, 64))
