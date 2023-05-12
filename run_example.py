try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None

import os
import pprint
import time
import json
from glob import glob
from random import choice

import torch
import torchvision
import torchvision.transforms as T

# import clip_config as config
from caption_cleaner import clean_caption
import config
from losses import CLIPLoss
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
from schedulers import get_warmup_cosine_scheduler, get_warmup_to_constant_scheduler
from transforms import ImgPilColorDistortion, ImgPilGaussianBlur, MultiViewGenerator
from utils import SmoothedValue
import my_webdataset as wds
from my_webdataset import DataPipeline, WebLoader
import slip_models
from tokenizer import SimpleTokenizer
from smart_open import open as smart_open

####
import transformers
# import clip
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from sd_model import create_sd_model

# from transformers import CLIPTextModel, CLIPTokenizer


class SampleGenerator(object):
  """Iterator which returns multiple samples of a given input data.
  Can be used in place of a PyTorch `DataLoader` to generate synthetic data.
  Args:
    data: The data which should be returned at each iterator step.
    sample_count: The maximum number of `data` samples to be returned.
  """

  def __init__(self, data, sample_count):
    self._data = data
    self._sample_count = sample_count
    self._count = 0

  def __iter__(self):
    return SampleGenerator(self._data, self._sample_count)

  def __len__(self):
    return self._sample_count

  def __next__(self):
    return self.next()

  def next(self):
    if self._count >= self._sample_count:
      raise StopIteration
    self._count += 1
    return self._data

# train_dataset_len = 12811 # 67  # Exactly the size of Imagenet dataset.
train_dataset_len = 10055143  # Exactly the size of our vesion of CC12M


def cap_transform(x):
    # print(x)
    return x if isinstance(x, str) else choice(x[1:])
def cap_transform_clean(x):
    # print(x)
    cap = x if isinstance(x, str) else choice(x[1:])
    return clean_caption(cap)

def load_training_data():
    world_size = get_world_size()
    # local_batch_size = cfg.batch_size // world_size
    local_batch_size = cfg.batch_size // (world_size * cfg.accumulate_grad_iter)
    if cfg.fake_data:
        if xu is not None:
            train_loader = xu.SampleGenerator(
                data=(
                    torch.randn(local_batch_size, 3, cfg.image_dim, cfg.image_dim),
                    torch.randint(low=0, high=10000, size=(local_batch_size, 77))
                ),
                sample_count=train_dataset_len // local_batch_size // world_size,
            )
            train_sampler = None
            return [None] * train_dataset_len, train_loader, train_sampler
        else:
            from fake_data import fake_data
            return [None] * train_dataset_len, fake_data(train_dataset_len, local_batch_size, str_cond=True), None
    master_print(f"loading images from : {cfg.data_dir}")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        cfg.model_name, subfolder="tokenizer", revision=None
    )
    
    # tokenizer = SimpleTokenizer()
     
    if cfg.pixel_space:
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
    else:
        img_mean = [0.5, 0.5, 0.5]
        img_std = [0.5, 0.5, 0.5]
    viz_transform =     T.Compose(
                    [
                        T.RandomResizedCrop(cfg.image_dim, scale=(0.5, 1.0)),
                        T.ToTensor(),
                        T.Normalize(mean=img_mean, std=img_std),
                    ]
                        )
    # num_dataset_instances = xm.xrt_world_size() * cfg.num_workers
    num_dataset_instances = world_size * cfg.num_workers
    epoch_size = train_dataset_len // num_dataset_instances
    
    if 'cc12m' in cfg.data_dir:
        train_shards = cfg.data_dir + "/cc12m-{000000..009819}.tar"
    elif 'laion' in cfg.data_dir: # /mnt/sfr-laion400m-data-ssd-pv-us-central1-a/laion115m_capfilt_20220817
        train_shards = glob(os.path.join(cfg.data_dir, "*", "*.tar"))
    else:
        raise NotImplementedError
    # print(len(list(train_shards)))
    
    chosen_cap_transform = cap_transform_clean if cfg.clean_caption else cap_transform

    train_dataset = DataPipeline(
         wds.ResampledShards(train_shards),
        # we now have an iterator over all shards
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(10000, initial=cfg.initial_data_buffer, handler=wds.warn_and_continue),
        wds.decode("pil", handler=wds.warn_and_continue),
        # we now have a list of decompressed train samples from each shard in this worker, in sequence
        wds.to_tuple("ppm;jpg;jpeg;png", "txt;json", handler=wds.warn_and_continue),
        # wds.map_tuple(viz_transform, tokenizer_call, handler=wds.warn_and_continue),
        # wds.map_tuple(viz_transform, lambda x: torch.randint(low=0, high=10000, size=(77,)), handler=wds.warn_and_continue),
        wds.map_tuple(viz_transform, chosen_cap_transform, handler=wds.warn_and_continue),
        wds.batched(local_batch_size),
        )# .with_epoch(epoch_size).with_length(epoch_size) # adds `__len__` method to dataset
    train_loader = WebLoader(train_dataset, num_workers=cfg.num_workers,
            batch_size=None) # , collate_fn=collate_fn) # , collate_fn=collate_fn)
    # train_loader = train_loader.with_length(epoch_size) # adds `__len__` method to dataloader

    train_sampler = None
    synchronize()
    master_print("data loading done!")

    return train_dataset, train_loader, train_sampler



def laion_cap_transform(x):
    return x[0]

def cc12m_cap_transform(x):
    return x

def collate_fn(input_list):
    """
    Orig message from simclr
    For N images with 2 views, it returns (2*N, C, H, W) shape, arranged as
    [img_1_view_1, ..., img_N_view_1, img_1_view_1, ..., img_N_view_1]
    and can be reshaped to (2, N, C, H, W) for loss computation
    """
    master_print(input_list)
    master_print(len(input_list))
    master_print(type(input_list))
    master_print(type(input_list[0]))
    master_print(type(input_list[1]))
    master_print(input_list[0])
    master_print(input_list[1])
    img_list = []
    txt_list = []

    # for img, txt in input_list:
    #     img_list.append(img)
    #     txt_list.append(txt)
    return input_list # torch.stack(img_list), torch.tensor(txt_list, dtype=torch.long)


def train():
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    assert batch_size % get_world_size() == 0
    train_dataset, train_loader, train_sampler = load_training_data()
    local_batch_size = batch_size // (get_world_size() * cfg.accumulate_grad_iter)
    
    model = create_sd_model(
                   weighting_model_name=cfg.weighting_model_name,
                   num_chain_timesteps=cfg.num_chain_timesteps,
                   cond_dropout=cfg.cond_dropout,
                   pretrained_unet=cfg.pretrained_unet,
                    lora=cfg.lora,
                   pixel_space=cfg.pixel_space
                   )
   #  model = SDModel(cond_dropout=cfg.cond_dropout,
   #                 pretrained_unet=cfg.pretrained_unet)
        
    # test  = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    """
    master_print("Initializing text encoder")
    text_encoder = slip_models.CLIPTextPretrained()
    print("VAE start")
    print("VAE needs to be frozen")
    # diffusers is fine though
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    print("VAE end")
    
    master_print("Does UNet need better weight init?")
    unet_cfg = UNet2DConditionModel.load_config("CompVis/stable-diffusion-v1-4", subfolder="unet") 
    unet = UNet2DConditionModel.from_config(unet_cfg)
    
    scheduler = DDPMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4',
                                              subfolder='scheduler')
                                              
    """
    #####
    """
    transformers is weird about loading pretrained models in DDP which is fun?
    master_print("Initializing text encoder")
    # downloading from pretrained might need to come from master initially
    if is_master():
        master_print("Master init")
        # doing this to allow download to happen
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            cfg.model_name, subfolder="text_encoder", revision=None
        )
    synchronize()
    if not is_master():
        print("Non-master init")
        # doing this to allow download to happen
        # could be from state dict bug
        # https://github.com/huggingface/transformers/issues/8649
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            cfg.model_name, subfolder="text_encoder", revision=None
        )
    synchronize()
    """
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    master_print("Pushing to device")
    if is_xla():
        device = xm.xla_device()
        train_loader = pl.MpDeviceLoader(train_loader, device)
        model = model.to(device)
        # text_encoder = text_encoder.to(device)
        # vae = vae.to(device)
        # unet = unet.to(device)
        broadcast_xla_master_model_param(model)
    else:
        device = torch.device(f"cuda:{cfg.device_id}")
        # text_encoder = text_encoder.to(device)
        # vae = vae.to(device)
        # unet = unet.to(device)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.device_id], output_device=cfg.device_id, find_unused_parameters=False
        )
        """
        text_encoder = torch.nn.parallel.DistributedDataParallel(
            text_encoder, device_ids=[cfg.device_id],
            output_device=cfg.device_id, find_unused_parameters=True
        )
        vae = torch.nn.parallel.DistributedDataParallel(
            vae, device_ids=[cfg.device_id],
            output_device=cfg.device_id, find_unused_parameters=True
        )
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[cfg.device_id],
            output_device=cfg.device_id, find_unused_parameters=True
        )
        """
    """
    OLD OPTIM CODE
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": cfg.weight_decay},
                    {"params": p_non_wd, "weight_decay": 0}]
    """
    
    
    # https://github.com/CompVis/latent-diffusion/issues/52#issuecomment-1229188761
    # https://github.com/CompVis/stable-diffusion/issues/452
    # Noting that scale LR seems weird
    # Also some people recommend Adam over AdamW
    
    # Original batch size is 2048
    # They did it by using 256 GPUs w/ BS 4 and 2 accumlate gradients
    # With this LR was 1e-4
    # LAION steps (no aesthetics) was 237k
    # For now just scale LR by num_datapoints / 2048
    # BS for us will be much lower, could adjust steps accordingly
    # but shouldn't matter for initial training signal
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=cfg.weight_decay,
        lr=cfg.lr,
    )

    iters_per_epoch = train_dataset_len / batch_size if (not cfg.iters_per_epoch) else cfg.iters_per_epoch
    lr_scheduler = get_warmup_to_constant_scheduler(
        optimizer,
        warmup_iteration=cfg.warmup_steps, #  / batch_ratio,
    )
    scaler = None
    if cfg.use_pytorch_amp:
        scaler = torch.cuda.amp.GradScaler()
    loss_fn = CLIPLoss()

    resume_ckpt_path = None
    if cfg.resume_training:
        if cfg.resume_ckpt_path == "<auto-resume-latest>":
            # find the lastest checkpoint file
            for e in range(1, num_epochs + 1):
                try_path = os.path.join(
                    cfg.ckpt_dir, f"{cfg.ckpt_prefix}_epoch_{e}.ckpt"
                )
                try:
                    with smart_open(try_path):
                        resume_ckpt_path = try_path
                except:
                    pass
                # if os.path.exists(try_path):
                #     resume_ckpt_path = try_path
        else:
            # assert os.path.exists(cfg.resume_ckpt_file)
            resume_ckpt_path = cfg.resume_ckpt_path
            
    if resume_ckpt_path is not None:
        meta_data = load_ckpt(resume_ckpt_path, model, optimizer,
                              lr_scheduler, scaler,
                            load_model_ckpt_only=cfg.load_model_ckpt_only)
        last_ckpt_epoch = meta_data["epoch"]
        if cfg.override_opt_lr:
            optimizer.param_groups[0]['lr'] = cfg.lr
    else:
        last_ckpt_epoch = 0

    synchronize()
    smoothed_loss = SmoothedValue(window_size=20)
    model.train()

    master_print(
        "training begins (note that the first few XLA iterations "
        "are very slow due to compilation)"
    )
    epoch = last_ckpt_epoch + 1

    logs = []
    log_file = f'log_{cfg.ckpt_prefix}.txt'
    # print(get_rank())
    torch.manual_seed(get_rank())
    # print(torch.randint(0, cfg.num_noise_steps, (local_batch_size,)))
    # print(torch.get_rng_state())
    if is_xla():
        xm.set_rng_state(get_rank(), xm.xla_device())
        print(xm.get_rng_state(xm.xla_device()))
    synchronize()
    while epoch <= num_epochs:
        master_print(f"starting epoch {epoch}")
        time_b = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        for step, (img, txt) in enumerate(train_loader):
            # forward pass
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss = model(img, txt, print_unweighted_loss=cfg.print_unweighted_loss)
                loss = loss / cfg.accumulate_grad_iter
            # backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if ((step+1) % cfg.accumulate_grad_iter == 0):
                if is_xla():
                    # PyTorch XLA requires manually reducing gradients
                    xm.reduce_gradients(optimizer)

                # param update
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            else: # Let's pretend this never happend besides the gradients
                if is_xla():
                    xm.mark_step()
                continue
            step = step // cfg.accumulate_grad_iter

            if (step+1 ) % cfg.log_step_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                # TODO: .detach().cpu(?).numpy()
                if False: # is_xla():
                    reduced_loss = 0
                else:
                    reduced_loss =   cfg.accumulate_grad_iter * reduce_tensor(loss, average=True).item()
                    # reduced_loss =  reduce_tensor(loss, average=True).item()
                master_print(
                        f"epoch {epoch} step {(step + 1)}, lr: {lr:.7f}, "
                        f"loss: {reduced_loss:.3f}, "
                        f"elapsed time: {time.time() - time_b} sec"
                )

            # add termination on steps
            if ((step+1)%int(iters_per_epoch))==0:

                time_elapsed = time.time() - time_b
                master_print(f"epoch {epoch} done ({time_elapsed:.2f} sec)")

                info_dict = {"loss":reduced_loss, "epoch_time":time_elapsed, "epoch":epoch}
                logs.append(info_dict)
                # with open(log_file, 'w') as out_file:
                #     json.dump(logs, out_file)

                if epoch % cfg.ckpt_epoch_interval == 0 or epoch == num_epochs:
                    ckpt_path = os.path.join(
                        cfg.ckpt_dir, f"{cfg.ckpt_prefix}_epoch_{epoch}.ckpt"
                    )
                    meta_data = {"cfg": cfg, "epoch": epoch}
                    master_print("Saving ckpt")
                    save_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data)
                    # save_ckpt(ckpt_path, model, lr_scheduler, lr_scheduler, scaler, meta_data)
                epoch += 1
                time_b = time.time()
                if epoch>num_epochs:
                    break
                master_print(f"starting epoch {epoch}")

    master_print("training completed")


def main(device_id, configuration):
    config.cfg = configuration
    distributed_init(configuration, device_id)
    global cfg
    cfg = configuration

    synchronize()
    master_print("\nconfig:")
    master_print(pprint.pformat(cfg), end="\n\n")
    train()


if __name__ == "__main__":
    config.cfg = config.build_cfg_from_argparse()

    if is_xla():
        if False:
            # changed for v4 but pytorch 2.0 seemed to fix
            tpu_cores_per_node = 1 # CHANGED FOR V4
            main(0, config.cfg)
        else:
            tpu_cores_per_node = None # was 8 for 3
            xmp.spawn(main, args=(config.cfg,), nprocs=tpu_cores_per_node,
                    start_method='fork')
    else:
        infer_init_method(config.cfg)
        if config.cfg.no_spawn:
            assert config.cfg.device_id >= 0
            main(config.cfg.device_id, config.cfg)
        else:
            torch.multiprocessing.spawn(
                main, nprocs=torch.cuda.device_count(), args=(config.cfg,)
            )
