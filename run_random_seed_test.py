import os
import pprint
import time
import json
from glob import glob

import torch
import torchvision
import torchvision.transforms as T

# import clip_config as config
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
from sd_model import SDModel
import torch
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None

def train():
    torch.manual_seed(get_rank())
    print(torch.randint(1000, size=(5,)))


def main(device_id, configuration):
    config.cfg = configuration
    distributed_init(configuration, device_id)
    global cfg
    cfg = configuration

    synchronize()
    train()


if __name__ == "__main__":
    config.cfg = config.build_cfg_from_argparse()

    if is_xla():
        tpu_cores_per_node = 8
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
