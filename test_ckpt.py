from sd_model import SDModel
import torch
torch.set_grad_enabled(False)
import sys
import os
import argparse
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_path", type=str)
parser.add_argument("--num-seeds", type=int, default=5)
parser.add_argument("--gs", type=float, default=7.5)
parser.add_argument("--dim", type=int, default=512)
parser.add_argument("--pixel-space", action='store_true')
parser.add_argument("--conditioning-pos-embedding", action='store_true')
parser.add_argument("--prompts", type=str, default=["A photo of a dog"],
        nargs='+')

# below are a bunch of image cond toys
parser.add_argument("--image-cond", action='store_true')
parser.add_argument("--null-cls", action='store_true') # for image variations
parser.add_argument("--null-spatial", action='store_true')
parser.add_argument("--repeat-cls", action='store_true')
parser.add_argument("--null-bottom-half", action='store_true')
parser.add_argument("--null-top-half", action='store_true')
parser.add_argument("--reflect-vertically", action='store_true')

parser.add_argument("--spatial-dropout", type=float, default=0)


args = parser.parse_args()
print(args.prompts)

if args.null_cls or args.null_spatial:
    print("HAVEN'T PUT IN NULL ARGS INTO SAVING YET, WILL SILENTLY OVERWRITE BASLEINE GENS")
# should do argparsing if working with this script more
# num_seeds = int(sys.argv[2]) if len(sys.argv)>2 else 1
# prompt = sys.argv[3] if len(sys.argv)>3 else 'smoke test'

# ckpt_path = args.ckpt_path #. sys.argv[1]

use_default_pretrained = (args.ckpt_path == "vanilla_pretrained")

use_lora = ('lora' in args.ckpt_path)
a = SDModel(pretrained_unet=use_default_pretrained, lora=use_lora,
           pixel_space=args.pixel_space,
           cond_type=('image' if args.image_cond else 'text'),
            conditioning_pos_embedding=args.conditioning_pos_embedding
           )

if not use_default_pretrained:
    ckpt = torch.load(args.ckpt_path)
    state_dict = ckpt['model']
    state_dict = {k.replace('module.', ''):v for k,v in state_dict.items()}
    missing_keys, unexpected_keys = a.load_state_dict(state_dict, strict=False)
    assert len(missing_keys)==0
    # if use_lora:
    #     a.unet.load_attn_procs(a.lora_layers)
a.cuda()
a.eval()

for prompt in args.prompts:
    if args.image_cond:
        feed_prompt = [Image.open(p) for p in prompt.split('+')]
    else:
        feed_prompt = prompt
    save_dir = f'outputs/{args.dim}x{args.dim}/{prompt.replace(" ","-")[:100]}/{args.ckpt_path.replace(".ckpt", "").replace("sd_ckpts/", "")}/'
    os.makedirs(save_dir, exist_ok=True)
    for s in range(args.num_seeds):
        im = a.generate(feed_prompt, seed=s, w=args.dim, h=args.dim,
                gs=args.gs,
                        null_cls=args.null_cls, null_spatial=args.null_spatial,
                        repeat_cls=args.repeat_cls,
                        null_bottom_half=args.null_bottom_half,
                        null_top_half=args.null_top_half,
                        reflect_vertically=args.reflect_vertically,
                        spatial_dropout=args.spatial_dropout
                       )[0]
        im.save(f'{save_dir}/test_ckpt_run_{s}_gs{args.gs}.jpg')
