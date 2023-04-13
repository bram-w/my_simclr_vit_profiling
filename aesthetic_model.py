from os.path import expanduser  
import open_clip
import os
from urllib.request import urlretrieve  
import torch
from torch import nn
import torchvision
from PIL import Image

def get_aesthetic_model_head(clip_model="vit_l_14", version='v1'):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    if version=='v1':
        path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    elif version=='v2':
        path_to_model = cache_folder + "/sac+logos+ava1-l14-linearMSE.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        if version=='v1':
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
            )
        elif version=='v2':
            url_model = 'https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth?raw=true'
        else:
            raise NotImplementedError
        urlretrieve(url_model, path_to_model)

    if version=='v1':
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
    elif version=='v2':
        m = MLP(768)
    else:
        raise NotImplementedError
        
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

    def __init__(self, clip_model='vit_l_14', version='v2'):
        super().__init__()
        if version=='v2':
            assert clip_model=='vit_l_14'
        self.head = get_aesthetic_model_head(clip_model, version=version)
        self.stem = get_aesthetic_model_stem(clip_model)
        
        resize = torchvision.transforms.Resize(224)
        crop = torchvision.transforms.CenterCrop(224)
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        self.preprocess = torchvision.transforms.Compose([resize, crop, normalize])

    def forward(self, img):
        # input is in -1, 1
        img = self.preprocess(0.5 * (img + 1)) # put back in 0-1 so can normalize proper
        image_features = self.stem.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return self.head(image_features).squeeze(1)


def aesthetic_score_v1(image, model, amodel):
    raise NotImplementedError # Reference from DOODL
    # im = Image.open(im) if isinstance(im, str) else im
    # image = preprocess(im).unsqueeze(0)
    # with autocast('cuda'):
    if True: # with autocast(device):
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = amodel(image_features).clip(0, 10).mean()
        return prediction

class MLP(nn.Module):
    #. https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
    
if __name__ == '__main__':
    model = AestheticScorer()
    
    im = Image.open(f'test.jpg')
    im = torchvision.transforms.ToTensor()(im)
    im = 2*im.unsqueeze(0) -1
    print(model(im))
    for i in range(5):
        im = Image.open(f'/export/home/diffusion/my_simclr_vit_profiling/outputs/512x512/A-photo-of-a-dog/vanilla_pretrained/test_ckpt_run_{i}.jpg')
        im = torchvision.transforms.ToTensor()(im)
        im = 2*im.unsqueeze(0) -1
        print(model(im))
    
    