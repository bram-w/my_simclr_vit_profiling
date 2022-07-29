from ast import literal_eval
import copy
import numpy as np
import argparse
from utils import AttrDict

cfg = AttrDict()

# --------------------------------------------------------------------------- #
# general options
# --------------------------------------------------------------------------- #
cfg.device = "xla"  # "xla" or "cuda"
cfg.log_step_interval = 50
cfg.ckpt_epoch_interval = 5
cfg.ckpt_dir = "/export/home/<ADD YOUR USERNAME>/"  
cfg.ckpt_prefix = "clip_vit"
cfg.load_model_ckpt_only = False

cfg.resume_training = True
cfg.resume_ckpt_path = "<auto-resume-latest>"
cfg.pretrained_text_ckpt = ''

cfg.use_pytorch_amp = False

# --------------------------------------------------------------------------- #
# data options
# --------------------------------------------------------------------------- #
cfg.fake_data = False
cfg.data_dir = "N/A"
cfg.drop_last = True
cfg.num_workers = 4

# --------------------------------------------------------------------------- #
# model options
# --------------------------------------------------------------------------- #
# see https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# for a list of ViT model classes
# cfg.vit_model_class = "vit_base_patch16_224"
cfg.freeze_patch_embed = True
cfg.embed_dim = 512
cfg.num_models = 0
cfg.simclr_loss_temperature = 0.1

cfg.multi_binary_model = False
cfg.use_mobilenet = False
cfg.use_resnet18 = False
# --------------------------------------------------------------------------- #
# training options
# --------------------------------------------------------------------------- #
cfg.batch_size = 4096
cfg.lr = 5e-4
cfg.weight_decay = 0.5
cfg.num_epochs = 35
cfg.warmup_epochs = 1

# --------------------------------------------------------------------------- #
# linear eval options
# --------------------------------------------------------------------------- #
cfg.linear_eval = AttrDict()
cfg.linear_eval.pretrained_ckpt_path = "please-specify-the-pretrained-checkpoint"
cfg.linear_eval.reset_last_ln = True
cfg.linear_eval.num_classes = 1000
cfg.linear_eval.batch_size = 1024
cfg.linear_eval.lr = 4e-2
cfg.linear_eval.weight_decay = 0
cfg.linear_eval.momentum = 0.9
cfg.linear_eval.num_epochs = 100
cfg.linear_eval.ckpt_epoch_interval = 20
cfg.linear_eval.test_epoch_interval = 10

# --------------------------------------------------------------------------- #
# distributed options
# --------------------------------------------------------------------------- #
cfg.init_method = ""
cfg.port = 20000
cfg.world_size = -1
cfg.rank = -1
cfg.rank_offset = 0
cfg.device_id = -1
cfg.no_spawn = False


# --------------------------------------------------------------------------- #
# Isola options
# --------------------------------------------------------------------------- #
cfg.isola_align_scale = 0.0
cfg.isola_unif_scale = 0.0
cfg.expert_loss = False
cfg.barlow_twins_loss = False
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def build_cfg_from_argparse(args_list=None):
    """Load config with command line options (`--cfg` and a list of options)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args(args_list)
    if args.cfg:
        _merge_cfg_from_file(args.cfg)
    if args.opts:
        opts = args.opts
        if all("=" in v for v in args.opts):
            opts = [s for v in args.opts for s in v.split("=")]
        _merge_cfg_from_list(opts)
    return cfg


def _merge_cfg_from_file(cfg_filename):
    import yaml

    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, "r") as f:
        yaml_cfg = yaml.load(f)
    if yaml_cfg is not None:
        _merge_a_into_b(AttrDict(yaml_cfg), cfg)


def _merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, cfg)


def _merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split(".")
        d = cfg
        for subkey in key_list[:-1]:
            assert subkey in d, "Non-existent key: {}".format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, "Non-existent key: {}".format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), "Argument `a` must be an AttrDict"
    assert isinstance(b, AttrDict), "Argument `b` must be an AttrDict"

    for k, v_ in a.items():
        full_key = ".".join(stack) + "." + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("Non-existent config key: {}".format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list, int->float
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, int) and isinstance(value_b, float):
        value_a = float(value_a)
    else:
        raise ValueError(
            "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
            "key: {}".format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
