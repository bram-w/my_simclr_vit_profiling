from sd_model import SDModel
import torch
torch.set_grad_enabled(False)

a = SDModel().cuda()
im = a.generate('smoke test')[0]
im.save('outputs/test_untrained.jpg')


a = SDModel(pretrained_unet=True).cuda()
im = a.generate('smoke test')[0]
im.save('outputs/test_pretrained.jpg')