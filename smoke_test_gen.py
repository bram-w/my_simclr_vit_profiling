from sd_model import SDModel
import torch
torch.set_grad_enabled(False)
import sys

prompt = sys.argv[1] if len(sys.argv) > 1 else 'smoke test'


a = SDModel().cuda()
im = a.generate(prompt)[0]
im.save('outputs/test_untrained.jpg')


a = SDModel(pretrained_unet=True).cuda()
im = a.generate(prompt)[0]
im.save('outputs/test_pretrained.jpg')
