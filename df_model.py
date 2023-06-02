""" Largely imported from DF codebase (https://github.com/deep-floyd/IF/main/deepfloyd_if/model/gaussian_diffusion.py)"""
import torch
from torch import nn
from distributed import (
    is_master,
    is_xla,
    gather_tensor_with_backward
)
import numpy as np
import math
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from tqdm import tqdm
from diffusers import UNet2DConditionModel
from losses import normal_kl, discretized_gaussian_log_likelihood, mean_flat

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
        self.betas = betas
        assert len(betas.shape) == 1, 'betas must be 1-D'
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        
        self.setup_diffusion_constants()
        
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

    def setup_diffusion_constants(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
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
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
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
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
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
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out['mean'], log_scales=0.5 * out['log_variance']
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

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
            assert not is_xla() # would slow down too much
            with torch.no_grad():
                avg_loss = loss.mean()
                print(f"Loss: {avg_loss.item():.3f}")
        return loss.mean()
    
    def generate(self, prompt, batch_size=1,
                h=512, w=512, T=50, gs=7.5, seed=0,
                silent=False):
        raise NotImplementedError()

def create_df_model(**kwargs):
    return DFModel(**kwargs)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)