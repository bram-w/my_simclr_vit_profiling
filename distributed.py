import os
import subprocess
import socket
from itertools import chain
import time

import torch
from torch import distributed as dist
from smart_open import open as smart_open
from io import BytesIO

from google.cloud import storage
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


class XLAGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all TPU workers with support for backward propagation.
    """

    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        tensor_list = xm.all_gather(x.unsqueeze(dim), dim=dim)
        return tensor_list

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        all_grad_output = xm.all_reduce(xm.REDUCE_SUM, grad_output)
        return all_grad_output.select(dim, xm.get_ordinal()), None


class XLAReduceSumLayer(torch.autograd.Function):
    """
    Reduce tensor on TPUs with support for backward propagation.
    Fixing https://github.com/pytorch/xla/issues/2989
    """

    @staticmethod
    def forward(ctx, x):
        return xm.all_reduce(xm.REDUCE_SUM, x)

    @staticmethod
    def backward(ctx, grad_output):
        return xm.all_reduce(xm.REDUCE_SUM, grad_output)


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_tensor_with_backward(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    if is_xla():
        tensor_list = XLAGatherLayer.apply(tensor, dim)
        tensor_list = tensor_list.flatten(start_dim=dim, end_dim=dim + 1)
    else:
        tensor_list = GatherLayer.apply(tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


def xla_all_reduce_sum_with_backward(tensor):
    return XLAReduceSumLayer.apply(tensor)

def reduce_sum_with_backward(tensor):
    if is_xla():
        return xla_all_reduce_sum_with_backward(tensor)
    else:
        dist.all_reduce(tensor)
        return tensor


def broadcast_xla_master_model_param(model):
    parameters_and_buffers = []
    for p in chain(model.parameters(), model.buffers()):
        # Set all params in non-master devices to zero so that all_reduce is
        # equivalent to broadcasting parameters from master to other devices.
        if not is_master():
            zero = torch.tensor(0, dtype=p.data.dtype, device=p.data.device)
            p.data.mul_(zero)
        parameters_and_buffers.append(p.data)
    xm.wait_device_ops()
    xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
    xm.rendezvous("broadcast_xla_master_model_param")


def is_xla():
    from config import cfg

    return cfg.device == "xla"


def master_print(*args, **kwargs):
    flush = kwargs.pop("flush", True)
    if is_master():
        print(*args, **kwargs, flush=flush)


def reduce_tensor(t, average=False):
    world_size = get_world_size()
    if world_size < 2:
        return t

    with torch.no_grad():
        if is_xla():
            scale = 1.0 / world_size if average else 1.0
            t = xm.all_reduce(xm.REDUCE_SUM, t, scale=scale)
        else:
            dist.reduce(t, dst=0)
            if average:
                t /= world_size
    return t


def get_world_size():
    if is_xla():
        return xm.xrt_world_size()
    return dist.get_world_size()


def get_rank():
    if is_xla():
        return xm.get_ordinal()
    return dist.get_rank()


def is_master():
    return get_rank() == 0


def synchronize(message="sync-workers"):
    if is_xla():
        xm.rendezvous(message)
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    dist.barrier()


# adapted from
# https://github.com/facebookresearch/mmf/blob/master/mmf/utils/distributed.py
def infer_init_method(cfg):
    if cfg.init_method != "":
        return

    # if cfg.rank < 0 (default) after spawning,
    # cfg.rank will be filled as cfg.rank_offset + cfg.device_id
    cfg.rank_offset = 0

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        cfg.init_method = "env://"
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.rank = int(os.environ["RANK"])
        cfg.device_id = int(os.environ["LOCAL_RANK"])
        cfg.no_spawn = True

    # we can determine the init method automatically for Slurm
    else:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            assert cfg.world_size > 0, "world size must be specified for slurm"
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                cfg.init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"), port=cfg.port
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)

                if ntasks_per_node == 1:
                    assert cfg.world_size % nnodes == 0
                    gpus_per_node = cfg.world_size // nnodes
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    cfg.rank_offset = node_id * gpus_per_node
                    # cfg.rank and cfg.device_id will be filled after spawning
                    cfg.no_spawn = False
                else:
                    assert ntasks_per_node == cfg.world_size // nnodes
                    cfg.rank = int(os.environ.get("SLURM_PROCID"))
                    cfg.device_id = int(os.environ.get("SLURM_LOCALID"))
                    cfg.no_spawn = True
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass
        else:
            # launched locally with `python main_simclr_vit.py`
            cfg.world_size = torch.cuda.device_count()
            # cfg.rank and cfg.device_id will be filled after spawning
            cfg.no_spawn = False
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = cfg.port


def distributed_init(cfg, device_id):
    cfg.device_id = device_id
    if is_xla():
        cfg.world_size = xm.xrt_world_size()
        cfg.rank = xm.get_ordinal()
        return
    if dist.is_initialized():
        cfg.world_size = dist.get_world_size()
        cfg.rank = dist.get_rank()
        return

    if cfg.rank < 0:
        cfg.rank = cfg.rank_offset + device_id

    print(f"Distributed Init (Rank {cfg.rank}): {cfg.init_method}\n", end="")
    dist.init_process_group(
        backend="nccl",
        init_method=cfg.init_method,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )
    print(f"Initialized Host {socket.gethostname()} as rank {cfg.rank}\n", end="")

    torch.cuda.set_device(cfg.device_id)
    # perform a dummy all-reduce to initialize the NCCL communicator
    dist.all_reduce(torch.zeros(1).cuda())
    cfg.world_size = dist.get_world_size()
    cfg.rank = dist.get_rank()


def save_ckpt_backup(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "meta_data": meta_data,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if is_xla():
        # hack until I find how to up rendezvous timeout
        # if not is_master():
        #     non_master_sleep = 240
        #     print(f"Non-masters sleeping {non_master_sleep}s to avoid timeout") 
        #     time.sleep(non_master_sleep) # saving took several minutes too long
        if 'gs://' in ckpt_path:
            gcs_path = ckpt_path.replace('gs://', '')
            bucket_name = gcs_path.split('/')[0]
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            # print("Making blob")
            blob = bucket.blob('/'.join(gcs_path.split('/')[1:]))
            if is_master():
                # print("Master saving")
                with blob.open('wb', ignore_flush=True) as f:
                    cpu_ckpt = xm._maybe_convert_to_cpu(ckpt)
                    torch.save(cpu_ckpt, f) # switched to xm save but I think open was instantiating file
                    # xm.save(ckpt, f, global_master=True) # switched to xm save
            # print("waiting for sync")
            """
            # Alternative
            if is_master():
                with smart_open(ckpt_path, 'wb') as f:
                    cpu_ckpt = xm._maybe_convert_to_cpu(ckpt)
                    torch.save(cpu_ckpt, f) # switched to xm save but I think open was instantiating file
                    # torch.save(
            """
            synchronize()
        else:
            xm.save(ckpt, ckpt_path, global_master=True)
    else:
        if is_master():
            torch.save(ckpt, f)

    master_print(f"checkpoint saved to {ckpt_path}")

def sd_to_cpu(sd):
    return {k:v.cpu() for k,v in sd.items()}

def hacked_xla_save(data, file_or_path, master_only=True, global_master=False):
  """Saves the input data into a file.

  The saved data is transfered to PyTorch CPU device before being saved, so a
  following `torch.load()` will load CPU data.

  Args:
    data: The input data to be saved. Any nested combination of Python objects
      (list, tuples, sets, dicts, ...).
    file_or_path: The destination for the data saving operation. Either a file
      path or a Python file object. If `master_only` is ``False`` the path or
      file objects must point to different destinations as otherwise all the
      writes from the same host will override each other.
    master_only (bool, optional): Whether only the master device should save the
      data. If False, the `file_or_path` argument should be a different file or
      path for each of the ordinals taking part to the replication, otherwise
      all the replicas on the same host will be writing to the same location.
      Default: True
    global_master (bool, optional): When ``master_only`` is ``True`` this flag
      controls whether every host's master (if ``global_master`` is ``False``)
      saves the content, or only the global master (ordinal 0).
      Default: False
  """
  print("Determining should_Write")
  should_write_data = not master_only or xm.is_master_ordinal(
      local=not global_master)
  print("Pushing to cpu")
  # cpu_data = xm._maybe_convert_to_cpu(data, convert=should_write_data)
  print("Forcing push to cpu (will fix after vacation but freezing at wokring state)")
  cpu_data = xm._maybe_convert_to_cpu(data, convert=True)
  if True: # should_write_data: # Trying others saving to dummy
    if 'gs://' in file_or_path:
        print("Making blob")
        gcs_path = file_or_path.replace('gs://', '')
        bucket_name = gcs_path.split('/')[0]
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob('/'.join(gcs_path.split('/')[1:]))
        print("Opening blob")
        print("starting 'master' block")
        if should_write_data:
            with blob.open('wb', ignore_flush=True) as f:
                print("Actually saving")
                torch.save(cpu_data, f)
  print("Rendezou")
  # xm.rendezvous('torch_xla.core.xla_model.save')

def orig_xla_save(data, file_or_path, master_only=True, global_master=False):
    print("Determining should_Write", flush=True)
    should_write_data = not master_only or xm.is_master_ordinal(
      local=not global_master)
    print("Pushing to cpu", flush=True)
    cpu_data = xm._maybe_convert_to_cpu(data, convert=should_write_data)
    print("Actually doing save", flush=True)
    if should_write_data:
        torch.save(cpu_data, file_or_path)
    print("Done saving, starting rendezvous", flush=True)
    xm.rendezvous('torch_xla.core.xla_model.save')
    print("Did rendezvous", flush=True)


from torch_xla.utils.serialization import _rewrite_data, _get_tensors_folder
def orig_xser_save(data, path, master_only=True, global_master=False):
    print("Determining should_Write", flush=True)
    should_write_data = not master_only or xm.is_master_ordinal(
        local=not global_master)
    print("Rewriting data", flush=True)
    ref_data = _rewrite_data(_get_tensors_folder(path), data, should_write_data)
    print("Actually doing save", flush=True)
    if should_write_data:
        torch.save(ref_data, path)
    print("Done saving, starting rendezvous", flush=True)
    xm.rendezvous('xser_save')
    print("Did rendezvous", flush=True)



def save_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data):
    master_print("Creatin ckpt")
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "meta_data": meta_data,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if is_xla():
        if 'gs:/' in ckpt_path:
            print("USing hacked xla save")
            hacked_xla_save(ckpt, ckpt_path, global_master=True)
        else:
            master_print("Using gentlemanly disc save")
            # xm.save(ckpt, ckpt_path, global_master=True)
            orig_xla_save(ckpt, ckpt_path, global_master=True)
            # orig_xser_save(ckpt, ckpt_path, global_master=True)
            master_print("Done saving to disc")
    else:
        if is_master():
            if 'gs:/' in ckpt_path:
                gcs_path = ckpt_path.replace('gs://', '')
                bucket_name = gcs_path.split('/')[0]
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob('/'.join(gcs_path.split('/')[1:]))
                print("Opening blob")
                with blob.open('wb', ignore_flush=True) as f:
                    print("Actually saving")
                    torch.save(ckpt, f)
            else:
                torch.save(ckpt, ckpt_path)

    master_print(f"checkpoint saved to {ckpt_path}")



def save_ckpt_hangs_on_cpu_push(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data):
    ####### Can put ckpts together again if CPU fix works ####### 
    ckpt = {
        "model": sd_to_cpu(model.state_dict()),
        "lr_scheduler": sd_to_cpu(lr_scheduler.state_dict()),
        "meta_data": sd_to_cpu(meta_data),
    }
    ckpt_opt = {"optimizer": optimizer.state_dict()}

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if is_xla():
        # hack until I find how to up rendezvous timeout
        # if not is_master():
        #     non_master_sleep = 240
        #     print(f"Non-masters sleeping {non_master_sleep}s to avoid timeout") 
        #     time.sleep(non_master_sleep) # saving took several minutes too long
        if 'gs://' in ckpt_path:
            gcs_path = ckpt_path.replace('gs://', '')
            bucket_name = gcs_path.split('/')[0]
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob('/'.join(gcs_path.split('/')[1:]))
            print(f"(all print) saving to ckpt_path")
            master_print("to cpu")
            # Could be smarter with this and conditionally push but keeping basic for now
            # Breaking out of is_master per https://github.com/pytorch/xla/issues/945#issuecomment-525781994
            cpu_ckpt = ckpt # xm._maybe_convert_to_cpu(ckpt)
            print("Start is_master split")
            if is_master():
                print("Master saving")
                with blob.open('wb', ignore_flush=True) as f:
                    master_print("Saving step")
                    torch.save(cpu_ckpt, f) # switched to xm save but I think open was instantiating file
                    # xm.save(ckpt, f, global_master=True) # switched to xm save

            
            ######### Can comment out below if both work and can put together again #######
            gcs_path = gcs_path.replace('.ckpt', '_opt.ckpt')
            blob = bucket.blob('/'.join(gcs_path.split('/')[1:]))
            master_print("opt to cpu")
            cpu_ckpt_opt = xm._maybe_convert_to_cpu(ckpt_opt)
            if is_master():
                print("Master opt saving")
                with blob.open('wb', ignore_flush=True) as f:
                    master_print("opt saving step")
                    torch.save(cpu_ckpt_opt, f) # switched to xm save but I think open was instantiating file
                    # xm.save(ckpt, f, global_master=True) # switched to xm save
            # print("waiting for sync")
            ############################
            """
            # Alternative
            if is_master():
                with smart_open(ckpt_path, 'wb') as f:
                    cpu_ckpt = xm._maybe_convert_to_cpu(ckpt)
                    torch.save(cpu_ckpt, f) # switched to xm save but I think open was instantiating file
                    # torch.save(
            """
            synchronize()
        else:
            xm.save(ckpt, ckpt_path, global_master=True)
    else:
        if is_master():
            torch.save(ckpt, f)

    master_print(f"checkpoint saved to {ckpt_path}")


def is_part_of_text_sd(k):
    return (k in ['positional_embedding', 'text_projection'] or k.split('.')[0] in ['transformer', 'ln_final', 'token_embedding'])

def load_text_model_ckpt(ckpt_path, model):
    from config import cfg

    if is_xla():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        partial_text_sd = {k:v for k,v in ckpt.items() if is_part_of_text_sd(k)}
    else:
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{cfg.device_id}")
        partial_text_sd = {f"module.{k}":v for k,v in ckpt.items() if is_part_of_text_sd(k)}


    sd_load_return_tup = model.load_state_dict(partial_text_sd, strict=False)
    # print(sd_load_return_tup)
    assert not len(sd_load_return_tup.unexpected_keys)
    master_print(f"Using text model from pretrained checkpoint {ckpt_path}")

def load_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler,
              load_model_ckpt_only=False):
    from config import cfg
    print(ckpt_path)
    """
    with smart_open(ckpt_path, 'rb') as f:
        # buffer_ = BytesIO(f.read())
        if is_xla():
            ckpt = torch.load(BytesIO(f.read()), map_location="cpu")
        else:
            ckpt = torch.load(BytesIO(f.read()), map_location=f"cuda:{cfg.device_id}")
    """

    if is_xla():
        if 'gs://' in ckpt_path:
            gcs_path = ckpt_path.replace('gs://', '')
            bucket_name = gcs_path.split('/')[0]
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob('/'.join(gcs_path.split('/')[1:]))
            with blob.open('rb') as f:
                ckpt = torch.load(f, map_location="cpu")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{cfg.device_id}")
    model_sd = ckpt["model"]
    if is_xla():
        model_sd = {k.replace('module.', ''):v for k,v in model_sd.items()}
    else:
        model_sd = {'module.' + k:v for k,v in model_sd.items() if k[:7]!='module.'}
    model.load_state_dict(model_sd)
    if not load_model_ckpt_only:
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
    meta_data = ckpt["meta_data"]
    master_print(f"resumed from checkpoint {ckpt_path}")
    return meta_data
