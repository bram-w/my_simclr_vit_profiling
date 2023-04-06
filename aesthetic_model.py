from os.path import expanduser  
import open_clip
import os
from urllib.request import urlretrieve  
import torch
from torch import nn
import torchvision

def get_aesthetic_model_head(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def get_aesthetic_model_stem(clip_model='vit_l_14'):
    clip_stem_name = clip_model.upper().replace('I', 'i').replace('_', '-')
    stem, _, _ = open_clip.create_model_and_transforms(clip_stem_name,
                                                pretrained='openai')
    stem.eval()
    return stem

class AestheticScorer(nn.Module):

    def __init__(self, clip_model='vit_l_14'):
        super().__init__()
        self.head = get_aesthetic_model_head(clip_model)
        self.stem = get_aesthetic_model_stem(clip_model)
        
        resize = torchvision.transforms.Resize(224)
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        self.preprocess = torchvision.transforms.Compose([resize, normalize])

    def forward(self, img):
        # input is in -1, 1
        img = self.preprocess(0.5 * (img + 1)) # put back in 0-1 so can normalize proper
        image_features = self.stem.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return self.head(image_features).squeeze(1)


def aesthetic_score(image, model, amodel):
    raise NotImplementedError # Reference from DOODL
    # im = Image.open(im) if isinstance(im, str) else im
    # image = preprocess(im).unsqueeze(0)
    # with autocast('cuda'):
    if True: # with autocast(device):
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = amodel(image_features).clip(0, 10).mean()
        return prediction
