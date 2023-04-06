from sd_model import SDModel
import torch
torch.set_grad_enabled(False)
import sys
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_path", type=str)
parser.add_argument("--num-seeds", type=int, default=1)
parser.add_argument("--prompt", type=str, default="smoke test")
args = parser.parse_args()

# should do argparsing if working with this script more
# num_seeds = int(sys.argv[2]) if len(sys.argv)>2 else 1
# prompt = sys.argv[3] if len(sys.argv)>3 else 'smoke test'

# ckpt_path = args.ckpt_path #. sys.argv[1]

use_default_pretrained = (args.ckpt_path == "vanilla_pretrained")
a = SDModel(pretrained_unet=use_default_pretrained)

if not use_default_pretrained:
    ckpt = torch.load(args.ckpt_path)
    state_dict = ckpt['model']
    state_dict = {k.replace('module.', ''):v for k,v in state_dict.items()}
    missing_keys, unexpected_keys = a.load_state_dict(state_dict, strict=False)
    assert len(missing_keys)==0
a.cuda()
a.eval()

save_dir = f'outputs/{args.prompt.replace(" ","-")}/{args.ckpt_path.replace(".ckpt", "").replace("sd_ckpts/", "")}/'
os.makedirs(save_dir, exist_ok=True)

for s in range(args.num_seeds):
    im = a.generate(args.prompt, seed=s)[0]
    im.save(f'{save_dir}/test_ckpt_run_{s}.jpg')
