import math

import torch


def get_warmup_cosine_scheduler(optimizer, warmup_iteration, max_iteration):
    def _warmup_cosine(step):
        if step < warmup_iteration:
            lr_ratio = step * 1.0 / warmup_iteration
        else:
            where = (step - warmup_iteration) * 1.0 / (max_iteration - warmup_iteration)
            lr_ratio = 0.5 * (1 + math.cos(math.pi * where))

        return lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine)


def get_warmup_to_constant_scheduler(optimizer, warmup_iteration):
    def _warmup_to_constant(step):
        if step < warmup_iteration:
            lr_ratio = step * 1.0 / warmup_iteration
        else:
            lr_ratio = 1

        return lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_to_constant)