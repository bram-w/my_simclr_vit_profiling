"""
Saves to ckpts via master but can't read
"""

import os
import pprint
import time
import json


import torch
import torchvision
import torchvision.transforms as T

# import clip_config as config
import config
from losses import SimCLRLoss, IsolaCLIPLoss, CLIPLoss, BarlowTwinsLoss
from models import SimCLRViTModel
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
)
from schedulers import get_warmup_cosine_scheduler
from transforms import ImgPilColorDistortion, ImgPilGaussianBlur, MultiViewGenerator
from utils import SmoothedValue
import my_webdataset as wds
from my_webdataset import DataPipeline, WebLoader
import slip_models
from tokenizer import SimpleTokenizer


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
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None

def identity(x):
    return x

def load_training_data():
    world_size = get_world_size()
    local_batch_size = cfg.batch_size // world_size
    if cfg.fake_data:
        train_loader = xu.SampleGenerator(
            data=(
                torch.randn(local_batch_size, 3, 224, 224),
                torch.randint(low=0, high=10000, size=(local_batch_size, 77))
            ),
            sample_count=train_dataset_len // local_batch_size // world_size,
        )
        train_sampler = None
        return [None] * train_dataset_len, train_loader, train_sampler

    master_print(f"loading images from : {cfg.data_dir}")
    tokenizer = SimpleTokenizer()
    viz_transform =     T.Compose(
                    [
                        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                        )
    # num_dataset_instances = xm.xrt_world_size() * cfg.num_workers
    num_dataset_instances = world_size * cfg.num_workers
    epoch_size = train_dataset_len // num_dataset_instances

    train_shards = cfg.data_dir + "/cc12m-{000000..009819}.tar"


    train_dataset = DataPipeline(
         wds.ResampledShards(train_shards),
        # we now have an iterator over all shards
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(10000, handler=wds.warn_and_continue),
        wds.decode("pil", handler=wds.warn_and_continue),
        # we now have a list of decompressed train samples from each shard in this worker, in sequence
        wds.to_tuple("ppm;jpg;jpeg;png", "txt", handler=wds.warn_and_continue),
        wds.map_tuple(viz_transform, tokenizer, handler=wds.warn_and_continue),
        wds.batched(local_batch_size),
        )# .with_epoch(epoch_size).with_length(epoch_size) # adds `__len__` method to dataset
    train_loader = WebLoader(train_dataset, num_workers=cfg.num_workers,
            batch_size=None) # , collate_fn=collate_fn)
    # train_loader = train_loader.with_length(epoch_size) # adds `__len__` method to dataloader

    train_sampler = None
    synchronize()
    master_print("data loading done!")

    return train_dataset, train_loader, train_sampler

def load_training_data_cuda():
    world_size = get_world_size()
    local_batch_size = cfg.batch_size // world_size
    if cfg.fake_data:
        train_dataset_len = 1281167  # Exactly the size of Imagenet dataset.
        train_loader = SampleGenerator(
            data=(
                torch.zeros(local_batch_size, 3, 224, 224),
                torch.randint(low=0, high=10000, size=(local_batch_size, 77), dtype=torch.int64)
            ),
            sample_count=train_dataset_len // local_batch_size // world_size,
        )
        train_sampler = None
        return [None] * train_dataset_len, train_loader, train_sampler

    master_print(f"loading images from disk folder: {cfg.data_dir}")
    simclr_transform = MultiViewGenerator(
        T.Compose(
            [
                T.RandomResizedCrop(size=224),
                T.RandomHorizontalFlip(p=0.5),
                ImgPilColorDistortion(strength=0.5),
                ImgPilGaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        n_views=2,
    )
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data_dir, "train"), simclr_transform
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=get_rank(),
        drop_last=cfg.drop_last,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        drop_last=cfg.drop_last,
        collate_fn=collate_fn,
        shuffle=False if train_sampler else True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    synchronize()
    master_print("data loading done!")

    return train_dataset, train_loader, train_sampler

def collate_fn(multi_view_img_list):
    """
    For N images with 2 views, it returns (2*N, C, H, W) shape, arranged as
    [img_1_view_1, ..., img_N_view_1, img_1_view_1, ..., img_N_view_1]
    and can be reshaped to (2, N, C, H, W) for loss computation
    """
    img_list = []
    for n_view in range(2):
       img_list.extend(views[n_view] for views, _ in multi_view_img_list)
    label_list = [label for  _, label in multi_view_img_list]
    return torch.stack(img_list), torch.tensor(label_list, dtype=torch.long)


def train():
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    assert batch_size % get_world_size() == 0
    train_dataset, train_loader, train_sampler = load_training_data()
    # if is_xla():
    #     train_dataset, train_loader, train_sampler = load_training_data()
    # else:
    #     train_dataset, train_loader, train_sampler = load_training_data_cuda()
    if cfg.multi_binary_model:
        model = slip_models.MultiBinaryCLIP(num_models=cfg.embed_dim)
    else:
        model = slip_models.CLIP_VITB16(embed_dim=cfg.embed_dim)
    if is_xla():
        device = xm.xla_device()
        train_loader = pl.MpDeviceLoader(train_loader, device)
        model = model.to(device)
        broadcast_xla_master_model_param(model)
    else:
        device = torch.device(f"cuda:{cfg.device_id}")
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.device_id], output_device=cfg.device_id, find_unused_parameters=True
        )

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


    optimizer = torch.optim.AdamW(
        optim_params,
        weight_decay=cfg.weight_decay,
        lr=cfg.lr,
        betas=(0.9, 0.98)
    )

    iters_per_epoch = train_dataset_len / batch_size
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_iteration=int(iters_per_epoch * cfg.warmup_epochs),
        max_iteration=int(iters_per_epoch * num_epochs),
    )
    scaler = None
    if cfg.use_pytorch_amp:
        scaler = torch.cuda.amp.GradScaler()
    return_logit_scale = False
    if cfg.barlow_twins_loss:
        loss_fn = BarlowTwinsLoss(global_bs=cfg.batch_size)
    elif cfg.isola_align_scale and cfg.isola_unif_scale:
        loss_fn = IsolaCLIPLoss(align_scale=cfg.isola_align_scale,
                                 unif_scale=cfg.isola_unif_scale)
    else:
        return_logit_scale = True
        loss_fn = CLIPLoss(use_image_unif_loss=cfg.isola_unif_scale,
                           use_text_unif_loss=cfg.isola_unif_scale,
                           unif_scale=cfg.isola_unif_scale)
    # if is_master():
    #     os.makedirs(cfg.ckpt_dir, exist_ok=True)
    # master_print("\nmodel:")
    # master_print(model, end="\n\n")

    resume_ckpt_path = None
    if cfg.resume_training:
        if cfg.resume_ckpt_path == "<auto-resume-latest>":
            # find the lastest checkpoint file
            for e in range(1, num_epochs + 1):
                try_path = os.path.join(
                    cfg.ckpt_dir, f"{cfg.ckpt_prefix}_epoch_{e}.ckpt"
                )
                if os.path.exists(try_path):
                    resume_ckpt_path = try_path
        else:
            assert os.path.exists(cfg.resume_ckpt_file)
            resume_ckpt_path = cfg.resume_ckpt_file
    if resume_ckpt_path is not None:
        meta_data = load_ckpt(resume_ckpt_path, model, optimizer, lr_scheduler, scaler,
                            load_model_ckpt_only=cfg.load_model_ckpt_only)
        last_ckpt_epoch = meta_data["epoch"]
    else:
        last_ckpt_epoch = 0

    synchronize()
    smoothed_loss = SmoothedValue(window_size=20)
    model.train()

    master_print(
        "training begins (note that the first few XLA iterations "
        "are very slow due to compilation)"
    )
    # for epoch in range(last_ckpt_epoch + 1, num_epochs + 1):
    epoch = last_ckpt_epoch + 1

    logs = []
    log_file = f'log_{cfg.ckpt_prefix}.txt'
    while epoch <= num_epochs:
        master_print(f"starting epoch {epoch}")
        time_b = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for step, (img, txt) in enumerate(train_loader):
            # forward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(img, txt, return_logit_scale=return_logit_scale)
                loss = loss_fn(output)
            # backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if is_xla():
                # PyTorch XLA requires manually reducing gradients
                xm.reduce_gradients(optimizer)

            # param update
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()

            # with torch.no_grad(): model.logit_scale.data.clamp_(0, 4.6052)

            if (step+1 ) % cfg.log_step_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                reduced_loss = reduce_tensor(loss, average=True).item()
                master_print(
                        f"epoch {epoch} step {(step + 1)}, lr: {lr:.7f}, "
                        f"loss: {reduced_loss:.4f}, "
                )
                """
                smoothed_loss.update(reduced_loss, batch_size=txt.size(0))
                master_print(
                    f"epoch {epoch} step {(step + 1)}, lr: {lr:.4f}, "
                    f"loss: {reduced_loss:.4f}, "
                    f"loss (avg): {smoothed_loss.avg:.4f}, "
                    f"loss (median): {smoothed_loss.median:.4f}"
                )
                """

            # add termination on steps
            if ((step+1)%int(iters_per_epoch))==0:

                time_elapsed = time.time() - time_b
                master_print(f"epoch {epoch} done ({time_elapsed:.2f} sec)")

                info_dict = {"loss":reduced_loss, "epoch_time":time_elapsed, "epoch":epoch}
                logs.append(info_dict)
                with open(log_file, 'w') as out_file:
                    json.dump(logs, out_file)

                if epoch % cfg.ckpt_epoch_interval == 0 or epoch == num_epochs:
                    ckpt_path = os.path.join(
                        cfg.ckpt_dir, f"{cfg.ckpt_prefix}_epoch_{epoch}.ckpt"
                    )
                    meta_data = {"cfg": cfg, "epoch": epoch}
                    save_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data)
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
