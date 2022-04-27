import torch
import torch.nn as nn
import sys
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.nn.functional as F

def main(*a):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    bs = 128
    targets_cpu = torch.arange(bs)
    targets_tpu = torch.arange(bs).to(device)

    logits_cpu = 3*torch.randn(bs, 4096)
    logits_tpu = 3*torch.randn(bs, 4096).to(device)
    print(F.cross_entropy(logits_cpu, targets_cpu))
    xm.mark_step()
    print(F.cross_entropy(logits_tpu, targets_tpu))

    logits_cpu = zeros(bs, 4096)
    logits_tpu = zeros(bs, 4096).to(device)
    print(F.cross_entropy(logits_cpu, targets_cpu))
    xm.mark_step()
    print(F.cross_entropy(logits_tpu, targets_tpu))

    logits_cpu = torch.rand(bs, 4096)
    logits_tpu = torch.rand(bs, 4096).to(device)
    print(F.cross_entropy(logits_cpu, targets_cpu))
    xm.mark_step()
    print(F.cross_entropy(logits_tpu, targets_tpu))

if __name__ == '__main__':
    xmp.spawn(main, args=(1,), nprocs=1)
