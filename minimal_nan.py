import torch
import torch.nn as nn
import sys
import torch_xla.distributed.xla_multiprocessing as xmp


def offset(e, pred, target, lossfn):
    return lossfn(pred-e, target)


def main(*a):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    lossfn = nn.BCELoss()
    targetcpu = torch.Tensor([1.])
    predcpu = torch.Tensor([1.])
    targettpu = torch.Tensor([1.]).to(device)
    predtpu = torch.Tensor([1.]).to(device)
    offsets = [0, 1e-8, 1e-7][::-1]
    msg = '{} - offset {}\n\tinput {}\n\ttarget {}\n\tlossval {}\n'
    msg += '-'*40
    for o in offsets:
        vcpu = offset(o, predcpu, targetcpu, lossfn)
        vtpu = offset(o, predtpu, targettpu, lossfn)
        xm.mark_step()
        print(msg.format('CPU', o, predcpu-o, targetcpu, vcpu))
        print(msg.format('TPU', o, predtpu-o, targettpu, vtpu))


if __name__ == '__main__':
    xmp.spawn(main, args=(1,), nprocs=1)
