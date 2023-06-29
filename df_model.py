""" Largely imported from DF codebase (https://github.com/deep-floyd/IF/main/deepfloyd_if/model/gaussian_diffusion.py)"""
from typing import List, Optional
import torch
from torch import nn
from distributed import (
    is_master,
    is_xla,
    gather_tensor_with_backward
)
from PIL import Image
import numpy as np
import math
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from diffusers import UNet2DConditionModel
from losses import normal_kl, discretized_gaussian_log_likelihood, mean_flat

from utils import seed_everything

def get_named_beta_schedule(schedule_name: str = 'cosine', num_diffusion_timesteps: int=1000):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == 'linear':
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f'unknown beta schedule: {schedule_name}')


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class DFModel(nn.Module):
    respacing_modes = {
        'fast27': '10,10,3,2,2',
        'smart27': '7,4,2,1,2,4,7',
        'smart50': '10,6,4,3,2,2,3,4,6,10',
        'smart100': '1,1,1,1,2,2,2,2,2,2,3,3,4,4,5,5,6,7,7,8,9,10,13',
        'smart185': '1,1,2,2,2,3,3,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20',
        'super27': '1,1,1,1,1,1,1,2,5,13',  # for III super-res
        'super40': '2,2,2,2,2,2,3,4,6,15',  # for III super-res
        'super100': '4,4,6,6,8,8,10,10,14,30',  # for III super-res
    }
    
    def __init__(self,
                model_config_name: str = "DeepFloyd/IF-I-M-v1.0",
                beta_schedule_name: str ="cosine",
                num_diffusion_timesteps: str = 1000,
                pretrained_unet: bool = False,
                cond_dropout: float = 0.1,
                rescale_timesteps: bool = False,
                use_t5: bool = True,
                lora=False, #for legacy (currently unsupported)
                weighting_module=None, #for legacy (currently unsupported)
                pixel_space = True, #for legacy (model is always in pixel space)
                ):
        super().__init__()
        assert model_config_name.startswith("DeepFloyd/IF-I"), f"Got config {model_config_name}. Code currently only supports Stage I training."

        ## Setup Diffusion process ##
        betas = get_named_beta_schedule(beta_schedule_name, num_diffusion_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self._set_betas(betas) # Betas will become torch tensor!
        self.original_num_timesteps = self.num_timesteps  # For generations which don't use full chain
        self.timestep_map = np.arange(self.num_timesteps)
        
        self.setup_diffusion_constants(self.betas)
        
        ## Get Models ##
        # Text encoder & Tokenizer
        self.use_clip = not use_t5
        if self.use_clip:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                )
            self.tokenizer_kwargs = {'padding':"max_length",
                                     'max_length':self.tokenizer.model_max_length,
                                     'truncation':True,
                                     'return_tensors':"pt"
                                     }
            self.text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
            )
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_config_name, subfolder="tokenizer")
            self.tokenizer_kwargs = {'max_length': 77,
                                        'padding': "max_length",
                                        'truncation': True,
                                        'return_attention_mask': True,
                                        'add_special_tokens':True,
                                        'return_tensors':"pt"
                                        }
            self.text_encoder = T5EncoderModel.from_pretrained(model_config_name, subfolder="text_encoder")
        # Freeze text encoder
        self.text_encoder.requires_grad_(False)

        # UNet
        if pretrained_unet:
            assert not self.use_clip, "CLIP Text Encoder can NOT be used with pretrained DF UNet(s)."
            self.unet = UNet2DConditionModel.from_pretrained(model_config_name,
                                                             subfolder='unet')
        else:
            unet_cfg = UNet2DConditionModel.load_config(model_config_name,
                                                        subfolder="unet")
            if self.use_clip:
                unet_cfg['cross_attention_dim'] = self.text_encoder.config.hidden_size
                unet_cfg['encoder_hid_dim'] = None
                
            self.unet = UNet2DConditionModel.from_config(unet_cfg)

        self.cond_dropout = cond_dropout
        self.rescale_timesteps = rescale_timesteps
        
        # slow startup on cpu but not sure how to handle on DDP
        self.encoder_hidden_states_UC = self.get_text_embeddings("")
    
    @property
    def device(self):
        return self.unet.device

    def train(self, mode=True):
        super(DFModel, self).train(mode=mode)
        self.text_encoder.eval()

    def _set_betas(self, betas: np.ndarray):
        assert len(betas.shape) == 1, 'betas must be 1-D'
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = torch.tensor(betas, dtype=torch.float64)
        self.num_timesteps = int(self.betas.shape[0])
        
        return

    def setup_diffusion_constants(self, betas):
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0])])
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        return
    
    def dynamic_thresholding(self, x, p=0.995, c=1.7):
        """
        Dynamic thresholding, a diffusion sampling technique from Imagen (https://arxiv.org/abs/2205.11487)
        to leverage high guidance weights and generating more photorealistic and detailed images
        than previously was possible based on x.clamp(-1, 1) vanilla clipping or static thresholding

        p — percentile determine relative value for clipping threshold for dynamic compression,
            helps prevent oversaturation recommend values [0.96 — 0.99]

        c — absolute hard clipping of value for clipping threshold for dynamic compression,
            helps prevent undersaturation and low contrast issues; recommend values [1.5 — 2.]
        """
        x_shapes = x.shape
        s = torch.quantile(x.abs().reshape(x_shapes[0], -1), p, dim=-1)
        s = torch.clamp(s, min=1, max=c)
        x_compressed = torch.clip(x.reshape(x_shapes[0], -1).T, -s, s) / s
        x_compressed = x_compressed.T.reshape(x_shapes)
        return x_compressed
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model_output, x, t, clip_denoised=True, dynamic_thresholding_p=0.99, dynamic_thresholding_c=1.7,
        denoised_fn=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x.shape
        )
        max_log = _extract_into_tensor(torch.log(self.betas), t, x.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                x = self.dynamic_thresholding(x, p=dynamic_thresholding_p, c=dynamic_thresholding_c)
                return x  # x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _scale_timesteps(self, t):
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        new_t = map_tensor[t]
        if self.rescale_timesteps:
            return new_t.float() * (1000.0 / self.original_num_timesteps)
        return new_t
    
    @torch.no_grad()
    def get_text_embeddings(self, text: str):
        # TODO: Move tokenization to dataloading (?)
        tokenized_inputs = self.tokenizer(
                                        text,
                                        **self.tokenizer_kwargs
        ).to(self.device)
        
        if self.use_clip:
            attention_mask = None
        else:
            attention_mask = tokenized_inputs.attention_mask
        
        text_encoder_embs = self.text_encoder(
            input_ids=tokenized_inputs.input_ids,
            attention_mask=attention_mask
        )
        return text_encoder_embs['last_hidden_state'].detach()
        
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=True
    ):
        """
        Ported from DF codebase (https://github.com/deep-floyd/IF/blob/main/deepfloyd_if/model/gaussian_diffusion.py#L686) 
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out['mean'], out['log_variance']
        )
        kl = mean_flat(kl) / math.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out['mean'], log_scales=0.5 * out['log_variance']
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {'output': output, 'pred_xstart': out['pred_xstart']}    

    def forward(self, img, txt, timesteps=None, noise=None, print_unweighted_loss=False):
        
        # Get text embeddings
        encoder_hidden_states = self.get_text_embeddings(txt)
        
        # Do dropout to null conditioning
        lbs = encoder_hidden_states.size(0)
        mask = (torch.rand((lbs, 1, 1),
                            device=self.device) > self.cond_dropout)

        _, l, d = encoder_hidden_states.size()
        mask = mask.repeat(1, l, d)

        uncond_hidden_states = self.encoder_hidden_states_UC.to(self.device).repeat(lbs, 1, 1).detach()
        encoder_hidden_states = torch.where(mask, 
                                            encoder_hidden_states,
                                            uncond_hidden_states)

        if noise is None:
            noise = torch.randn_like(img)

        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps,
                                        (noise.size(0),),
                                        device=img.device)
            timesteps = timesteps.long()
            
        noisy_img_t = self.q_sample(img, timesteps, noise)
        noise_pred = self.unet(noisy_img_t,
                               self._scale_timesteps(timesteps), 
                               encoder_hidden_states=encoder_hidden_states).sample
        
        ## Compute Loss ##
        B, C = img.shape[:2]
        assert noise_pred.shape == (B, C * 2, *img.shape[2:])
        noise_pred, model_var_values = torch.split(noise_pred, C, dim=1)
        # Loss for noise prediction
        loss = mean_flat((noise - noise_pred) ** 2)
        # Learn the variance using the variational bound, but don't let
        # it affect our mean prediction.
        frozen_out = torch.cat([noise_pred.detach(), model_var_values], dim=1)
        vb = self._vb_terms_bpd(
            frozen_out,
            x_start=img,
            x_t=noisy_img_t,
            t=timesteps,
            clip_denoised=False,
        )['output']
        # Divide by 1000 for equivalence with initial implementation.
        # Without a factor of 1/1000, the VB term hurts the MSE term.
        vb *= self.num_timesteps / 1000.0
        loss += vb
        
        if print_unweighted_loss and is_master():
            assert not is_xla()  # would slow down too much
            with torch.no_grad():
                avg_loss = loss.mean()
                print(f"Loss: {avg_loss.item():.3f}")
        return loss.mean()
    
    def p_sample(
        self, model, x, t, clip_denoised=True, dynamic_thresholding_p=0.99, dynamic_thresholding_c=1.7,
        denoised_fn=None, model_kwargs=None, inpainting_mask=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        out = self.p_mean_variance(
            model_output,
            x,
            t,
            clip_denoised=clip_denoised,
            dynamic_thresholding_p=dynamic_thresholding_p,
            dynamic_thresholding_c=dynamic_thresholding_c,
            denoised_fn=denoised_fn,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if inpainting_mask is None:
            inpainting_mask = torch.ones_like(x, device=x.device)

        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        sample = (1 - inpainting_mask)*x + inpainting_mask*sample
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        dynamic_thresholding_p=0.99,
        dynamic_thresholding_c=1.7,
        inpainting_mask=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        sample_fn=None,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for step_idx, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            dynamic_thresholding_p=dynamic_thresholding_p,
            dynamic_thresholding_c=dynamic_thresholding_c,
            denoised_fn=denoised_fn,
            inpainting_mask=inpainting_mask,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        )):
            if sample_fn is not None:
                sample = sample_fn(step_idx, sample)
            final = sample
        return final['sample']

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        inpainting_mask=None,
        noise=None,
        clip_denoised=True,
        dynamic_thresholding_p=0.99,
        dynamic_thresholding_c=1.7,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = self.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    dynamic_thresholding_p=dynamic_thresholding_p,
                    dynamic_thresholding_c=dynamic_thresholding_c,
                    denoised_fn=denoised_fn,
                    inpainting_mask=inpainting_mask,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out['sample']

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        dynamic_thresholding_p=0.99,
        dynamic_thresholding_c=1.7,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        out = self.p_mean_variance(
            model_output,
            x,
            t,
            dynamic_thresholding_p=dynamic_thresholding_p,
            dynamic_thresholding_c=dynamic_thresholding_c,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out['pred_xstart'] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        dynamic_thresholding_p=0.99,
        dynamic_thresholding_c=1.7,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, 'Reverse ODE only for deterministic path'
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            dynamic_thresholding_p=dynamic_thresholding_p,
            dynamic_thresholding_c=dynamic_thresholding_c,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out['pred_xstart']
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out['pred_xstart'] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {'sample': mean_pred, 'pred_xstart': out['pred_xstart']}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        dynamic_thresholding_p=0.99,
        dynamic_thresholding_c=1.7,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        sample_fn=None,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for step_idx, sample in enumerate(self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            dynamic_thresholding_p=dynamic_thresholding_p,
            dynamic_thresholding_c=dynamic_thresholding_c,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        )):
            if sample_fn is not None:
                sample = sample_fn(step_idx, sample)
            final = sample
        return final['sample']

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        dynamic_thresholding_p=0.99,
        dynamic_thresholding_c=1.7,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = self.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    dynamic_thresholding_p=dynamic_thresholding_p,
                    dynamic_thresholding_c=dynamic_thresholding_c,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out['sample']
    
    def _set_diffusion(self, timesteps: Optional[List[int]] = None, betas: Optional[List[float]] = None):
        if timesteps is None:
            return
        
        if betas is None:
            betas = self.betas
        else:
            self._set_betas(np.array(betas, dtype=np.float64))
            self.setup_diffusion_constants(self.betas)
        
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        
        self._set_betas(np.array(new_betas, dtype=np.float64))
        self.timestep_map = timestep_map
        self.setup_diffusion_constants(self.betas)
    
    def generate(self, prompt, batch_size=1,
                h=64, w=64, T=50, gs=7.5, seed=0,
                sample_timestep_respacing='smart185',
                use_ddim: bool = False,
                positive_mixer=0.25,
                batch_repeat=1,
                silent=False):
        # TODO: Check for other stages. Get_img_size for low res, Variables like sample_timestep_respacing, positive_mixer change!
        timesteps = self.respacing_modes.get(sample_timestep_respacing, [1000])
        self._set_diffusion(space_timesteps(1000, timesteps))
        
        bs_scale = 2
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // bs_scale]
            combined = torch.cat([half]*bs_scale, dim=0)
            model_out = self.unet(combined, ts, **kwargs).sample
            eps, rest = model_out[:, :3], model_out[:, 3:]
            if bs_scale == 3:
                cond_eps, pos_cond_eps, uncond_eps = torch.split(eps, len(eps) // bs_scale, dim=0)
                half_eps = uncond_eps + gs * (
                    cond_eps * (1 - positive_mixer) + pos_cond_eps * positive_mixer - uncond_eps)
                pos_half_eps = uncond_eps + gs * (pos_cond_eps - uncond_eps)
                eps = torch.cat([half_eps, pos_half_eps, half_eps], dim=0)
            else:
                cond_eps, uncond_eps = torch.split(eps, len(eps) // bs_scale, dim=0)
                half_eps = uncond_eps + gs * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        seed = seed_everything(seed)
        text_emb = self.get_text_embeddings(prompt).to(self.device, dtype=self.unet.dtype).repeat(batch_size, 1, 1)
        batch_size = text_emb.shape[0] * batch_repeat
        encoder_hidden_states = torch.cat([text_emb,
                                           self.encoder_hidden_states_UC.to(self.device).repeat(batch_size, 1, 1)],
                                          axis=0)
        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
        )
        noise = torch.randn(
                (batch_size * bs_scale, 3, h, w), device=self.device, dtype=self.unet.dtype)
        
        if use_ddim:
            sample = self.ddim_sample_loop(model_fn, 
                                            (batch_size * bs_scale, 3, h, w), 
                                            noise,
                                            clip_denoise=True,
                                            sample_fn=None,
                                            dynamic_thresholding_p=0.95,
                                            dynamic_thresholding_c=1.5,
                                            model_kwargs=model_kwargs,
                                            progress=not silent,
                                            device=self.device)[:batch_size]
        else:
            sample = self.p_sample_loop(model_fn, 
                                            (batch_size * bs_scale, 3, h, w), 
                                            noise, 
                                            clip_denoised=True,
                                            sample_fn=None,
                                            dynamic_thresholding_p=0.95,
                                            dynamic_thresholding_c=0.95,
                                            model_kwargs=model_kwargs,
                                            progress=not silent,
                                            device=self.device)[:batch_size]
        #TODO: validate?
        image = (0.5 * sample + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        return image

def create_df_model(**kwargs):
    return DFModel(**kwargs)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith('ddim'):
            desired_count = int(section_counts[len('ddim'):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f'cannot create exactly {num_timesteps} steps with an integer stride'
            )
        elif section_counts == 'fast27':
            steps = space_timesteps(num_timesteps, '10,10,3,2,2')
            # Help reduce DDIM artifacts from noisiest timesteps.
            steps.remove(num_timesteps - 1)
            steps.add(num_timesteps - 3)
            return steps
        section_counts = [int(x) for x in section_counts.split(',')]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f'cannot divide section of {size} steps into {section_count}'
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def _extract_into_tensor(tensor, timesteps, broadcast_shape):
    """
    Extract values from a 1-D torch tensor for a batch of indices.
    :param tensor: the 1-D torch tensor.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = tensor.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res
    # return res.expand(broadcast_shape)